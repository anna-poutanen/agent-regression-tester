"""Public Pydantic AI integration for recording and side-effect-free tool replay.

Replay replaces *every* supported function toolset. It retains original argument
schemas but never delegates execution to a registered business-tool body. This
is not a sandbox for factories, prompts, validators, or arbitrary Python code.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict, deque
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, cast

from pydantic import TypeAdapter
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset, WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.usage import UsageLimits
from pydantic_core import to_jsonable_python

from .checks import json_equal
from .schema import JsonObject, Scenario, ToolCall, ToolFixture

MAX_MODEL_REQUESTS = 20
MAX_TOOL_CALLS = 100
_ARGUMENTS_ADAPTER = TypeAdapter(JsonObject)


def _arguments_from_part(part: ToolCallPart) -> dict[str, Any]:
    """Keep invalid JSON attempts observable, including non-finite numbers."""
    raw = part.args
    try:
        if raw is None:
            return {}
        if isinstance(raw, str):

            def invalid_constant(value: str) -> Any:
                raise ValueError(f"Invalid JSON constant: {value}")

            value = json.loads(raw, parse_constant=invalid_constant)
        else:
            value = raw
        return _ARGUMENTS_ADAPTER.validate_python(value)
    except (ValueError, TypeError):
        return {"INVALID_JSON": raw if isinstance(raw, str) else repr(raw)}


def _json_value(value: Any) -> Any:
    """Use the same JSON representation Pydantic AI uses in model messages."""
    return to_jsonable_python(value)


def _diagnostic_value(value: Any) -> Any:
    """Keep invalid values in validation feedback without corrupting JSON artifacts."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"INVALID_JSON": repr(value)}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _diagnostic_value(item) for key, item in value.items()}
    return {"INVALID_JSON": repr(value)}


def _coerce_messages(messages: Any) -> list[ModelMessage]:
    if isinstance(messages, (str, bytes, bytearray)):
        return ModelMessagesTypeAdapter.validate_json(messages)
    if isinstance(messages, Sequence):
        return ModelMessagesTypeAdapter.validate_python(list(messages))
    raise ValueError("messages must be serialized JSON or a sequence of Pydantic AI messages")


def _display_tool_name(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return f"INVALID_TOOL_NAME:{value!r}"


def calls_from_messages(messages: Sequence[ModelMessage]) -> list[ToolCall]:
    """Extract attempts in model-message order, including validation retries.

    Calls in the same model response share a step: parallel call ordering cannot
    establish that a lookup completed before a dependent action. Malformed JSON
    is retained as ``{'INVALID_JSON': raw_arguments}`` rather than discarded.
    """
    calls: list[ToolCall] = []
    pending: dict[str, deque[ToolCall]] = defaultdict(deque)
    step = 0
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    call = ToolCall(
                        tool_name=_display_tool_name(part.tool_name),
                        arguments=_arguments_from_part(part),
                        tool_call_id=part.tool_call_id,
                        step=step,
                        outcome="pending",
                    )
                    calls.append(call)
                    pending[part.tool_call_id].append(call)
            step += 1
        elif isinstance(message, ModelRequest):
            for request_part in message.parts:
                if not isinstance(request_part, (ToolReturnPart, RetryPromptPart)):
                    continue
                candidates = pending.get(request_part.tool_call_id)
                if not candidates:
                    continue
                # A tool name is important when imported traces contain repeated IDs.
                returned_call = next(
                    (
                        item
                        for item in candidates
                        if item.tool_name == _display_tool_name(request_part.tool_name)
                    ),
                    None,
                )
                if returned_call is None:
                    continue
                candidates.remove(returned_call)
                returned_call.response = (
                    _diagnostic_value(request_part.content)
                    if isinstance(request_part, RetryPromptPart)
                    else _json_value(request_part.content)
                )
                if isinstance(request_part, RetryPromptPart):
                    returned_call.outcome = "retry"
                elif request_part.outcome == "success":
                    returned_call.outcome = "return"
                elif request_part.outcome == "denied":
                    returned_call.outcome = "blocked"
                else:
                    returned_call.outcome = "error"
    return calls


def scenario_from_messages(
    messages: Any, *, name: str, prompt: str, output: Any = None
) -> Scenario:
    """Import a completed or partial text/function-tool trace without executing it.

    Accepts Pydantic AI messages, their JSON string/bytes, or decoded JSON objects.
    Observations are never marked reviewed and never converted to business rules.
    """
    calls = calls_from_messages(_coerce_messages(messages))
    fixtures = [
        ToolFixture(
            tool_name=call.tool_name,
            arguments=copy.deepcopy(call.arguments),
            response=copy.deepcopy(call.response),
            outcome=call.outcome,
        )
        for call in calls
        if call.outcome in {"return", "retry"}
        and "INVALID_JSON" not in call.arguments
        and not call.tool_name.startswith("INVALID_TOOL_NAME:")
    ]
    metadata = {} if output is None else {"recorded_output": _json_value(output)}
    return Scenario(
        name=name,
        prompt=prompt,
        fixtures=fixtures,
        recorded_calls=calls,
        metadata=metadata,
    )


def _validate_capabilities(capability: Any) -> None:
    children = getattr(capability, "capabilities", None)
    if type(capability).__name__ == "CombinedCapability" and children is not None:
        for child in children:
            _validate_capabilities(child)
        return
    allowed = {"ToolSearch", "PendingMessageDrainCapability", "Instrumentation"}
    if (
        type(capability).__name__ not in allowed
        or not type(capability).__module__.startswith("pydantic_ai.capabilities.")
        or getattr(capability, "defer_loading", False)
    ):
        raise ValueError(
            f"Unsupported capability {type(capability).__name__}: replay supports ordinary "
            "function tools without custom/native/deferred capabilities"
        )


def _clone_supported_toolset(toolset: AbstractToolset[Any]) -> AbstractToolset[Any]:
    """Check before entering any lifecycle and isolate builtin wrapper state."""
    module = type(toolset).__module__
    name = type(toolset).__name__
    if type(toolset) is FunctionToolset or (
        name == "_AgentFunctionToolset" and module == "pydantic_ai.agent"
    ):
        for tool in cast(FunctionToolset[Any], toolset).tools.values():
            if tool.requires_approval or tool.defer_loading:
                raise ValueError(f"Unsupported deferred or approval tool: {tool.name}")
        return toolset
    if module.startswith("pydantic_ai.toolsets.") and isinstance(toolset, CombinedToolset):
        return replace(
            toolset, toolsets=[_clone_supported_toolset(child) for child in toolset.toolsets]
        )
    if module.startswith("pydantic_ai.toolsets.") and isinstance(toolset, WrapperToolset):
        if name in {"ApprovalRequiredToolset", "DeferredLoadingToolset"}:
            raise ValueError(f"Unsupported deferred or approval toolset: {name}")
        return replace(toolset, wrapped=_clone_supported_toolset(toolset.wrapped))
    raise ValueError(
        f"Unsupported toolset {name}: only registered function tools and builtin "
        "wrappers around FunctionToolset are supported; remote/dynamic tools cannot replay"
    )


def supported_toolsets(agent: Agent[Any, Any]) -> list[AbstractToolset[Any]]:
    """Validate the supported replay boundary without invoking toolset lifecycle."""
    if not isinstance(agent, Agent):
        raise ValueError("Expected a Pydantic AI Agent")
    if agent.output_type is not str:
        raise ValueError(
            "Only string agent output is supported; structured/deferred output is unsupported"
        )
    _validate_capabilities(agent.root_capability)
    native_tools = agent.root_capability.get_native_tools()
    if any(
        type(tool).__name__ != "ToolSearchTool" or not getattr(tool, "optional", False)
        for tool in native_tools
    ):
        raise ValueError("Provider-native tools cannot be safely replayed")
    return [_clone_supported_toolset(toolset) for toolset in agent.toolsets]


async def record(
    agent: Agent[Any, Any], prompt: str, *, name: str, deps: Any = None, model: Any = None
) -> Scenario:
    """Run and record an agent. Unlike replay, recording executes real tool bodies.

    The caller must choose harmless tools or explicitly authorize their effects.
    A failed recording raises the underlying error; import partial messages with
    ``scenario_from_messages`` if they are useful as a regression scenario.
    """
    supported_toolsets(agent)
    # Validate name/prompt before running functions with possible side effects.
    Scenario(name=name, prompt=prompt)
    kwargs: dict[str, Any] = {
        "deps": deps,
        "usage_limits": UsageLimits(
            request_limit=MAX_MODEL_REQUESTS, tool_calls_limit=MAX_TOOL_CALLS
        ),
    }
    if model is not None:
        kwargs["model"] = model
    # A public override gives an explicit model precedence over an enclosing
    # agent.override context, while omitting it preserves ambient configuration.
    model_context = agent.override(model=model) if model is not None else nullcontext()
    with model_context:
        result = await agent.run(prompt, **kwargs)
    scenario = scenario_from_messages(
        result.all_messages(), name=name, prompt=prompt, output=result.output
    )
    scenario.metadata["pydantic_ai_version"] = __import__("pydantic_ai").__version__
    configured_model = model if model is not None else agent.model
    if configured_model is not None:
        scenario.metadata["model"] = getattr(configured_model, "model_name", str(configured_model))
    return scenario


@dataclass
class ReplayState:
    """Fresh, run-local fixture consumption shared by every intercepted toolset."""

    fixtures: list[ToolFixture]
    consumed: set[int] = field(default_factory=set)
    decisions: dict[str | None, deque[ToolCall]] = field(default_factory=lambda: defaultdict(deque))
    errors: list[str] = field(default_factory=list)

    def resolve(self, name: str, arguments: dict[str, Any], ctx: RunContext[Any]) -> Any:
        call = ToolCall(
            tool_name=name,
            arguments=_json_value(arguments),
            tool_call_id=ctx.tool_call_id,
            step=max(0, ctx.run_step - 1),
            outcome="blocked",
        )
        self.decisions[ctx.tool_call_id].append(call)
        matching = [
            index
            for index, fixture in enumerate(self.fixtures)
            if fixture.tool_name == name and json_equal(fixture.arguments, call.arguments)
        ]
        index = next((index for index in matching if index not in self.consumed), None)
        if index is None:
            rendered = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
            if matching:
                message = (
                    f"Fixture exhausted for {name}({rendered}); repeated call has no saved response"
                )
            elif any(fixture.tool_name == name for fixture in self.fixtures):
                message = f"No fixture matches arguments for {name}({rendered})"
            else:
                message = f"No saved fixture for tool {name}({rendered})"
            call.response = message
            self.errors.append(message)
            raise ModelRetry(message)
        self.consumed.add(index)
        fixture = self.fixtures[index]
        call.response = copy.deepcopy(fixture.response)
        call.outcome = fixture.outcome
        if fixture.outcome == "retry":
            message = (
                fixture.response
                if isinstance(fixture.response, str)
                else json.dumps(fixture.response, ensure_ascii=False)
            )
            raise ModelRetry(message)
        return copy.deepcopy(fixture.response)

    def merge(self, calls: list[ToolCall], *, failed: bool) -> list[ToolCall]:
        """Preserve raw model attempts while applying interception outcomes."""
        decisions = {key: deque(value) for key, value in self.decisions.items()}
        for call in calls:
            candidates = decisions.get(call.tool_call_id, deque())
            decision = next((item for item in candidates if item.tool_name == call.tool_name), None)
            if decision is not None:
                candidates.remove(decision)
                call.outcome = decision.outcome
                call.response = decision.response
            else:
                if call.outcome == "pending" and failed:
                    call.outcome = "error"
                self.errors.append(
                    f"Tool attempt {call.tool_name} was not replayed "
                    f"(outcome={call.outcome}); arguments may be invalid or tool unavailable"
                )
        return calls


def _raw_arguments(name: str, tool_args: dict[str, Any], ctx: RunContext[Any]) -> dict[str, Any]:
    for message in reversed(ctx.messages):
        if isinstance(message, ModelResponse):
            by_id = [
                part
                for part in message.parts
                if isinstance(part, ToolCallPart) and part.tool_call_id == ctx.tool_call_id
            ]
            if len(by_id) > 1:
                raise ModelRetry(
                    f"Ambiguous duplicate tool call ID {ctx.tool_call_id!r}; "
                    "saved responses cannot be safely assigned"
                )
            if by_id and by_id[0].tool_name == name:
                return _arguments_from_part(by_id[0])
    return _ARGUMENTS_ADAPTER.validate_python(_json_value(tool_args))


@dataclass
class ReplayToolset(WrapperToolset[Any]):
    """Return exact saved fixtures while preserving original tool definitions."""

    state: ReplayState

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        tools = await self.wrapped.get_tools(ctx)
        for name, tool in tools.items():
            if tool.tool_def.kind != "function" or tool.tool_def.defer_loading:
                raise ValueError(f"Unsupported deferred/native tool definition: {name}")
        return tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        return self.state.resolve(name, _raw_arguments(name, tool_args, ctx), ctx)
