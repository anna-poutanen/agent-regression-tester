"""Independent adversarial checks of the replay interception boundary."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset, WrapperToolset

from agent_regression.adapter import calls_from_messages
from agent_regression.runner import run_scenario, run_suite
from agent_regression.schema import (
    ForbiddenCall,
    OutputContains,
    RequiredCall,
    Scenario,
    ToolFixture,
)


def _scenario(**kwargs: Any) -> Scenario:
    defaults: dict[str, Any] = {
        "name": "audit",
        "prompt": "request",
        "reviewed": True,
        "checks": [OutputContains(text="done")],
    }
    defaults.update(kwargs)
    return Scenario(**defaults)


def _agent(parts: list[ToolCallPart], tools: list[Any]) -> Agent[Any, Any]:
    def model(messages: Any, info: Any) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=parts)
        return ModelResponse(parts=[TextPart("done")])

    return Agent(FunctionModel(model), tools=tools, retries=5)


async def test_custom_wrapper_lifecycle_is_refused_before_execution() -> None:
    effects: list[str] = []

    class UnsafeWrapper(WrapperToolset[Any]):
        async def __aenter__(self) -> UnsafeWrapper:
            effects.append("custom wrapper entered")
            return self

    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool
    def sentinel() -> str:
        effects.append("business tool executed")
        return "done"

    agent = Agent(FunctionModel(lambda messages, info: None), toolsets=[UnsafeWrapper(toolset)])
    with pytest.raises(ValueError, match="Unsupported toolset"):
        await run_scenario(agent, _scenario())
    assert effects == []


async def test_unknown_wrapper_nested_below_builtin_prefix_is_refused() -> None:
    effects: list[str] = []

    class UnsafeWrapper(WrapperToolset[Any]):
        async def __aenter__(self) -> UnsafeWrapper:
            effects.append("opened external service")
            return self

    wrapped = UnsafeWrapper(FunctionToolset()).prefixed("remote")
    agent = Agent(FunctionModel(lambda messages, info: None), toolsets=[wrapped])
    with pytest.raises(ValueError, match="Unsupported toolset"):
        await run_scenario(agent, _scenario())
    assert effects == []


async def test_nested_renamed_and_prefixed_functions_keep_public_name_and_schema() -> None:
    effects: list[str] = []
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool_plain
    def internal_lookup(customer_id: str) -> str:
        effects.append(customer_id)
        return "REAL CUSTOMER RECORD"

    public = toolset.renamed({"lookup": "internal_lookup"}).prefixed("shop")

    def model(messages: Any, info: Any) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("shop_lookup", {"customer_id": "123"})])
        return ModelResponse(parts=[TextPart("done")])

    result = await run_scenario(
        Agent(FunctionModel(model), toolsets=[public]),
        _scenario(
            fixtures=[
                ToolFixture(
                    tool_name="shop_lookup", arguments={"customer_id": "123"}, response="mock"
                )
            ],
            checks=[RequiredCall(tool_name="shop_lookup")],
        ),
    )
    assert result.passed, result
    assert result.calls[0].response == "mock"
    assert effects == []


@pytest.mark.parametrize("arguments", ['{"value":NaN}', {"value": float("nan")}])
async def test_nonfinite_arguments_fail_with_trace_instead_of_crashing_suite(
    arguments: Any,
) -> None:
    effects: list[str] = []

    def lookup(value: float) -> str:
        effects.append("business tool executed")
        return "done"

    suite = await run_suite(
        _agent([ToolCallPart("lookup", arguments)], [lookup]),
        [_scenario(checks=[ForbiddenCall(tool_name="lookup")])],
    )
    assert suite.failed_runs == 1
    assert len(suite.results[0].calls) == 1
    assert suite.results[0].calls[0].tool_name == "lookup"
    assert effects == []


async def test_invalid_argument_validation_retry_stays_json_safe() -> None:
    def lookup(value: int) -> str:
        raise AssertionError("BUSINESS TOOL EXECUTED")

    suite = await run_suite(
        _agent([ToolCallPart("lookup", {"value": float("nan")})], [lookup]),
        [_scenario(checks=[ForbiddenCall(tool_name="lookup")])],
    )
    assert suite.failed_runs == 1
    assert len(suite.results[0].calls) == 1
    assert suite.results[0].calls[0].outcome == "retry"
    # Report data itself must survive strict JSON encoding after a bad attempt.
    assert '"calls"' in suite.model_dump_json()


def test_unsupported_object_in_validation_diagnostic_is_preserved_safely() -> None:
    calls = calls_from_messages(
        [
            ModelResponse(parts=[ToolCallPart("lookup", {"value": -1}, "invalid")]),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            {
                                "type": "value_error",
                                "loc": ("value",),
                                "msg": "Value must be positive",
                                "input": -1,
                                "ctx": {"error": ValueError("Value must be positive")},
                            }
                        ],
                        tool_name="lookup",
                        tool_call_id="invalid",
                    )
                ]
            ),
        ]
    )
    assert len(calls) == 1
    assert calls[0].outcome == "retry"
    assert "Value must be positive" in calls[0].model_dump_json()


@pytest.mark.parametrize("name", ["", " "])
async def test_blank_model_tool_names_fail_without_crashing_suite(name: str) -> None:
    def lookup() -> str:
        raise AssertionError("BUSINESS TOOL EXECUTED")

    suite = await run_suite(
        _agent([ToolCallPart(name, {})], [lookup]),
        [_scenario()],
    )
    assert suite.failed_runs == 1
    assert suite.results[0].replay_errors or suite.results[0].error


@pytest.mark.parametrize("arguments", ["[]", "null", '"text"', "1", '{"value":'])
async def test_nonobject_json_arguments_are_preserved_and_never_execute(arguments: str) -> None:
    effects: list[str] = []

    def lookup(value: str) -> str:
        effects.append(value)
        return "done"

    result = await run_scenario(
        _agent([ToolCallPart("lookup", arguments)], [lookup]),
        _scenario(checks=[ForbiddenCall(tool_name="lookup")]),
    )
    assert not result.passed
    assert len(result.calls) == 1
    assert result.calls[0].arguments == {"INVALID_JSON": arguments}
    assert result.failures[0].code == "forbidden_call"
    assert effects == []


@pytest.mark.parametrize("container", ["checks", "fixtures", "metadata"])
async def test_nested_artifact_mutations_are_revalidated_before_any_model_call(
    container: str,
) -> None:
    model_calls: list[str] = []

    def model(messages: Any, info: Any) -> ModelResponse:
        model_calls.append("model invoked")
        return ModelResponse(parts=[TextPart("done")])

    scenario = _scenario()
    if container == "checks":
        scenario.checks.clear()
        scenario.checks.append({"kind": "unknown"})  # type: ignore[arg-type]
    elif container == "fixtures":
        scenario.fixtures.append(  # type: ignore[arg-type]
            {"tool_name": "tool", "arguments": {"invalid": object()}}
        )
    else:
        scenario.metadata["unsafe"] = object()  # type: ignore[assignment]
    with pytest.raises(ValueError):
        await run_scenario(Agent(FunctionModel(model)), scenario)
    assert model_calls == []


async def test_duplicate_same_turn_call_ids_never_silently_replay_wrong_arguments() -> None:
    effects: list[str] = []

    def lookup(customer_id: str) -> str:
        effects.append(customer_id)
        return "REAL CUSTOMER"

    scenario = _scenario(
        fixtures=[
            ToolFixture(tool_name="lookup", arguments={"customer_id": "a"}, response="customer a"),
            ToolFixture(
                tool_name="lookup", arguments={"customer_id": "a"}, response="customer a 2"
            ),
            ToolFixture(tool_name="lookup", arguments={"customer_id": "b"}, response="customer b"),
        ],
        checks=[
            RequiredCall(tool_name="lookup", arguments={"customer_id": "a"}),
            RequiredCall(tool_name="lookup", arguments={"customer_id": "b"}),
        ],
    )
    result = await run_scenario(
        _agent(
            [
                ToolCallPart("lookup", {"customer_id": "a"}, "repeated"),
                ToolCallPart("lookup", {"customer_id": "b"}, "repeated"),
            ],
            [lookup],
        ),
        scenario,
    )
    # Accurate raw matching may succeed. Ambiguous IDs may instead fail closed.
    if result.passed:
        assert [call.response for call in result.calls] == ["customer a", "customer b"]
    assert effects == []


async def test_concurrent_shared_agent_keeps_each_scenarios_responses_isolated() -> None:
    effects: list[str] = []

    def lookup(customer_id: str) -> str:
        effects.append(customer_id)
        return "REAL CUSTOMER"

    async def model(messages: Any, info: Any) -> ModelResponse:
        await asyncio.sleep(0.001)
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"customer_id": "same"})])
        latest = messages[-1]
        assert isinstance(latest, ModelRequest)
        returned = next(part for part in latest.parts if isinstance(part, ToolReturnPart))
        return ModelResponse(parts=[TextPart(returned.content)])

    agent = Agent(FunctionModel(model), tools=[lookup])
    scenarios = [
        _scenario(
            name=color,
            fixtures=[
                ToolFixture(tool_name="lookup", arguments={"customer_id": "same"}, response=color)
            ],
            checks=[OutputContains(text=color)],
        )
        for color in ["red", "blue", "green"]
    ]
    suite = await run_suite(agent, scenarios, repeats=4, concurrency=8)
    assert suite.passed_runs == 12
    assert all(run.output == run.scenario_name for run in suite.results)
    assert effects == []


async def test_fixture_contents_are_run_local_when_model_mutates_response_objects() -> None:
    def lookup() -> dict[str, list[str]]:
        raise AssertionError("BUSINESS TOOL EXECUTED")

    def model(messages: Any, info: Any) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {})])
        returned = next(
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        )
        assert returned.content == {"items": ["original"]}
        returned.content["items"].append("mutated by custom model")
        return ModelResponse(parts=[TextPart("done")])

    scenario = _scenario(
        fixtures=[ToolFixture(tool_name="lookup", arguments={}, response={"items": ["original"]})]
    )
    suite = await run_suite(Agent(FunctionModel(model), tools=[lookup]), [scenario], repeats=3)
    assert suite.passed_runs == 3
    assert scenario.fixtures[0].response == {"items": ["original"]}


async def test_prompt_and_output_gate_without_tools_is_valid() -> None:
    def model(messages: Any, info: Any) -> ModelResponse:
        assert isinstance(messages[0].parts[-1], UserPromptPart)
        return ModelResponse(parts=[TextPart("done")])

    result = await run_scenario(Agent(FunctionModel(model)), _scenario())
    assert result.passed
    assert result.calls == []
