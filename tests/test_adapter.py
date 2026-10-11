from __future__ import annotations

import json

import pytest
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from agent_regression.adapter import calls_from_messages, record, scenario_from_messages


def test_import_round_trips_framework_message_formats():
    messages = [
        ModelResponse(parts=[ToolCallPart("lookup", {"id": "123"}, "a")]),
        ModelRequest(parts=[ToolReturnPart("lookup", {"eligible": False}, "a")]),
    ]
    forms = [
        messages,
        ModelMessagesTypeAdapter.dump_json(messages),
        json.loads(ModelMessagesTypeAdapter.dump_json(messages)),
    ]
    for form in forms:
        scenario = scenario_from_messages(form, name="ineligible", prompt="refund", output="no")
        assert not scenario.reviewed
        assert not scenario.checks
        assert scenario.fixtures[0].response == {"eligible": False}
        assert scenario.metadata["recorded_output"] == "no"


def test_extraction_retains_malformed_parallel_retry_and_partial_calls():
    messages = [
        ModelResponse(
            parts=[
                ToolCallPart("lookup", {"id": "123"}, "a"),
                ToolCallPart("refund", '{"id":', "b"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("lookup", False, "a"),
                RetryPromptPart("bad JSON", tool_name="refund", tool_call_id="b"),
            ]
        ),
        ModelResponse(parts=[ToolCallPart("refund", {"id": "123"}, "c")]),
    ]
    calls = calls_from_messages(messages)
    assert [call.step for call in calls] == [0, 0, 1]
    assert [call.outcome for call in calls] == ["return", "retry", "pending"]
    assert calls[1].arguments == {"INVALID_JSON": '{"id":'}
    scenario = scenario_from_messages(messages, name="partial", prompt="refund")
    assert len(scenario.fixtures) == 1


def test_extracts_failed_and_denied_returns():
    messages = [
        ModelResponse(parts=[ToolCallPart("tool", {}, "a"), ToolCallPart("tool", {}, "b")]),
        ModelRequest(
            parts=[
                ToolReturnPart("tool", "failed", "a", outcome="failed"),
                ToolReturnPart("tool", "denied", "b", outcome="denied"),
            ]
        ),
    ]
    assert [call.outcome for call in calls_from_messages(messages)] == ["error", "blocked"]


def test_repeated_call_ids_are_matched_in_order():
    messages = [
        ModelResponse(parts=[ToolCallPart("tool", {"x": 1}, "id")]),
        ModelRequest(parts=[ToolReturnPart("tool", "first", "id")]),
        ModelResponse(parts=[ToolCallPart("tool", {"x": 2}, "id")]),
        ModelRequest(parts=[ToolReturnPart("tool", "second", "id")]),
    ]
    assert [call.response for call in calls_from_messages(messages)] == ["first", "second"]


async def test_record_executes_local_tools_and_preserves_retries():
    count = 0

    def lookup(order_id: str) -> dict[str, bool]:
        nonlocal count
        count += 1
        if count == 1:
            raise ModelRetry("try again")
        return {"eligible": False}

    def model(messages, info):
        calls = sum(isinstance(message, ModelResponse) for message in messages)
        if calls < 2:
            return ModelResponse(parts=[ToolCallPart("lookup", {"order_id": "123"}, str(calls))])
        return ModelResponse(parts=[TextPart("Cannot refund")])

    scenario = await record(Agent(FunctionModel(model), tools=[lookup]), "refund", name="retry")
    assert count == 2
    assert [fixture.outcome for fixture in scenario.fixtures] == ["retry", "return"]
    assert [call.step for call in scenario.recorded_calls] == [0, 1]
    assert not scenario.reviewed
    assert scenario.metadata["recorded_output"] == "Cannot refund"


async def test_record_supports_runcontext_and_async_toolsets():
    toolset = FunctionToolset()

    @toolset.tool
    async def lookup(ctx: RunContext[str], order_id: str) -> str:
        return f"{ctx.deps}:{order_id}"

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"order_id": "123"})])
        return ModelResponse(parts=[TextPart("done")])

    scenario = await record(
        Agent(FunctionModel(model), deps_type=str, toolsets=[toolset]),
        "lookup",
        name="context",
        deps="shop",
    )
    assert scenario.fixtures[0].response == "shop:123"


async def test_record_validates_scenario_before_any_tool_execution():
    calls = []

    def dangerous() -> str:
        calls.append("side effect")
        return "done"

    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("dangerous", {})])

    with pytest.raises(ValueError):
        await record(Agent(FunctionModel(model), tools=[dangerous]), "x", name="")
    assert calls == []


async def test_record_explicit_model_override_wins_over_ambient_model():
    def lookup(order_id: str) -> str:
        return "recorded"

    def forbidden_model(messages, info):
        raise AssertionError("AMBIENT MODEL MUST NOT BE USED")

    def explicit_model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"order_id": "123"})])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(forbidden_model), tools=[lookup])
    with agent.override(model=FunctionModel(forbidden_model)):
        scenario = await record(
            agent, "lookup", name="explicit", model=FunctionModel(explicit_model)
        )
    assert scenario.fixtures[0].response == "recorded"
