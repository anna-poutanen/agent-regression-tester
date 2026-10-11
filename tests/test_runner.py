from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from agent_regression.runner import run_scenario, run_suite, run_suite_sync
from agent_regression.schema import (
    BeforeCheck,
    ForbiddenCall,
    OutputContains,
    RequiredCall,
    Scenario,
    ToolFixture,
)


def fixture(tool_name="lookup", arguments=None, response=None, outcome="return"):
    return ToolFixture(
        tool_name=tool_name,
        arguments={"order_id": "123"} if arguments is None else arguments,
        response={"eligible": False} if response is None else response,
        outcome=outcome,
    )


def scenario(*, fixtures=None, checks=None, reviewed=True):
    return Scenario(
        name="ineligible",
        prompt="Refund order 123 if eligible",
        fixtures=[fixture()] if fixtures is None else fixtures,
        checks=[RequiredCall(tool_name="lookup"), ForbiddenCall(tool_name="refund")]
        if checks is None
        else checks,
        reviewed=reviewed,
    )


def agent_for_calls(calls, *, final="Cannot refund", tools=None):
    def lookup(order_id: str) -> dict[str, bool]:
        raise AssertionError("REAL lookup EXECUTED")

    def refund(order_id: str) -> str:
        raise AssertionError("REAL refund EXECUTED")

    def model(messages, info):
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step < len(calls):
            return ModelResponse(parts=calls[step])
        return ModelResponse(parts=[TextPart(final)])

    return Agent(FunctionModel(model), tools=tools or [lookup, refund], retries=3)


async def test_replay_preserves_configured_model_and_never_executes_functions():
    result = await run_scenario(
        agent_for_calls([[ToolCallPart("lookup", {"order_id": "123"})]]), scenario()
    )
    assert result.passed, result
    assert result.output == "Cannot refund"
    assert result.calls[0].response == {"eligible": False}


async def test_forbidden_call_is_visible_and_missing_fixture_never_falls_back():
    result = await run_scenario(
        agent_for_calls(
            [
                [ToolCallPart("lookup", {"order_id": "123"})],
                [ToolCallPart("refund", {"order_id": "123"})],
            ]
        ),
        scenario(),
    )
    assert not result.passed
    assert result.calls[1].outcome == "blocked"
    assert any(failure.code == "forbidden_call" for failure in result.failures)
    assert "No saved fixture" in result.replay_errors[0]
    assert result.error is None


async def test_wrong_arguments_do_not_replay_another_orders_data():
    result = await run_scenario(
        agent_for_calls([[ToolCallPart("lookup", {"order_id": "456"})]]), scenario()
    )
    assert not result.passed
    assert result.calls[0].arguments == {"order_id": "456"}
    assert "matches arguments" in result.replay_errors[0]


async def test_exact_matching_preserves_boolean_and_number_distinctions():
    def lookup(order_id: int) -> str:
        raise AssertionError("REAL tool EXECUTED")

    result = await run_scenario(
        agent_for_calls([[ToolCallPart("lookup", {"order_id": False})]], tools=[lookup]),
        scenario(fixtures=[fixture(arguments={"order_id": 0})]),
    )
    assert not result.passed
    assert result.calls[0].arguments == {"order_id": False}
    assert "matches arguments" in result.replay_errors[0]


async def test_fixture_matching_uses_raw_arguments_before_schema_coercion():
    def lookup(order_id: int) -> str:
        raise AssertionError("REAL tool EXECUTED")

    result = await run_scenario(
        agent_for_calls([[ToolCallPart("lookup", {"order_id": "123"})]], tools=[lookup]),
        scenario(fixtures=[fixture(arguments={"order_id": "123"})]),
    )
    assert result.passed, result
    assert result.calls[0].arguments == {"order_id": "123"}


async def test_duplicate_fixtures_are_consumed_in_order_then_exhausted():
    calls = [[ToolCallPart("lookup", {"order_id": "123"}, str(i))] for i in range(3)]
    result = await run_scenario(
        agent_for_calls(calls),
        scenario(fixtures=[fixture(response="first"), fixture(response="second")]),
    )
    assert not result.passed
    assert [call.response for call in result.calls[:2]] == ["first", "second"]
    assert result.calls[2].outcome == "blocked"
    assert "exhausted" in result.replay_errors[0]


async def test_saved_retry_is_replayed_then_valid_response_can_pass():
    calls = [[ToolCallPart("lookup", {"order_id": "123"}, str(i))] for i in range(2)]
    result = await run_scenario(
        agent_for_calls(calls),
        scenario(fixtures=[fixture(response="temporary failure", outcome="retry"), fixture()]),
    )
    assert result.passed, result
    assert [call.outcome for call in result.calls] == ["retry", "return"]


async def test_malformed_and_unknown_calls_remain_visible_without_function_execution():
    result = await run_scenario(
        agent_for_calls(
            [
                [ToolCallPart("lookup", '{"order_id":', "invalid")],
                [ToolCallPart("unknown_tool", {}, "unknown")],
                [ToolCallPart("lookup", {"order_id": "123"}, "valid")],
            ]
        ),
        scenario(),
    )
    assert not result.passed
    assert len(result.calls) == 3
    assert result.calls[0].arguments == {"INVALID_JSON": '{"order_id":'}
    assert result.calls[0].outcome == "retry"
    assert len(result.replay_errors) == 2


async def test_schema_invalid_arguments_are_not_hidden_by_a_later_valid_call():
    result = await run_scenario(
        agent_for_calls(
            [[ToolCallPart("lookup", {}, "bad")], [ToolCallPart("lookup", {"order_id": "123"})]]
        ),
        scenario(),
    )
    assert not result.passed
    assert result.calls[0].outcome == "retry"
    assert result.replay_errors


async def test_registered_and_external_function_toolsets_both_intercepted():
    toolset = FunctionToolset(id="extra")

    @toolset.tool
    async def refund(ctx: RunContext[str], order_id: str) -> str:
        raise AssertionError("REAL external function toolset EXECUTED")

    def lookup(order_id: str) -> str:
        raise AssertionError("REAL registered function EXECUTED")

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("lookup", {"order_id": "123"}),
                    ToolCallPart("refund", {"order_id": "123"}),
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    result = await run_scenario(
        Agent(FunctionModel(model), tools=[lookup], toolsets=[toolset], deps_type=str),
        scenario(
            fixtures=[fixture(), fixture(tool_name="refund", response="refunded")],
            checks=[RequiredCall(tool_name="lookup"), RequiredCall(tool_name="refund")],
        ),
        deps="shop",
    )
    assert result.passed, result
    assert [call.step for call in result.calls] == [0, 0]


async def test_prefixed_builtin_wrapper_is_intercepted_under_public_tool_name():
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup(order_id: str) -> str:
        raise AssertionError("REAL prefixed function EXECUTED")

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("shop_lookup", {"order_id": "123"})])
        return ModelResponse(parts=[TextPart("done")])

    result = await run_scenario(
        Agent(FunctionModel(model), toolsets=[toolset.prefixed("shop")]),
        scenario(
            fixtures=[fixture(tool_name="shop_lookup")],
            checks=[RequiredCall(tool_name="shop_lookup")],
        ),
    )
    assert result.passed, result


async def test_parallel_lookup_and_refund_do_not_satisfy_before_check():
    result = await run_scenario(
        agent_for_calls(
            [
                [
                    ToolCallPart("lookup", {"order_id": "123"}),
                    ToolCallPart("refund", {"order_id": "123"}),
                ]
            ]
        ),
        scenario(
            fixtures=[fixture(), fixture(tool_name="refund", response="refunded")],
            checks=[BeforeCheck(first="lookup", then="refund")],
        ),
    )
    assert not result.passed
    assert [call.step for call in result.calls] == [0, 0]


async def test_review_and_empty_check_gate_prevent_model_calls():
    def model(messages, info):
        raise AssertionError("MODEL MUST NOT BE CALLED")

    agent = Agent(FunctionModel(model))
    unreviewed = await run_scenario(agent, scenario(reviewed=False))
    empty = await run_scenario(agent, scenario(checks=[]))
    assert unreviewed.failures[0].code == "unreviewed"
    assert empty.failures[0].code == "no_checks"


async def test_unreviewed_override_explicitly_allows_replay():
    result = await run_scenario(
        agent_for_calls([[ToolCallPart("lookup", {"order_id": "123"})]]),
        scenario(reviewed=False),
        allow_unreviewed=True,
    )
    assert result.passed


async def test_repeats_and_concurrent_runs_get_fresh_fixture_state():
    agent = agent_for_calls([[ToolCallPart("lookup", {"order_id": "123"})]])
    result = await run_suite(agent, [scenario(), scenario()], repeats=3, concurrency=4)
    assert result.total_runs == 6
    assert result.passed_runs == 6
    assert [run.iteration for run in result.results] == [1, 2, 3, 1, 2, 3]


async def test_sync_and_async_factories_are_called_for_each_run():
    calls = []

    async def factory():
        calls.append("factory")
        return agent_for_calls([[ToolCallPart("lookup", {"order_id": "123"})]])

    result = await run_suite(factory, [scenario()], repeats=2, concurrency=2)
    assert result.passed_runs == 2
    assert len(calls) == 2


def test_sync_suite_entry_point():
    result = run_suite_sync(
        lambda: agent_for_calls([[ToolCallPart("lookup", {"order_id": "123"})]]), [scenario()]
    )
    assert result.passed_runs == 1


async def test_sync_entry_point_refuses_nested_event_loop():
    with pytest.raises(ValueError, match="active event loop"):
        run_suite_sync(Agent(FunctionModel(lambda messages, info: None)), [])


async def test_timeout_and_model_error_produce_failed_results():
    cancelled = asyncio.Event()

    async def model(messages, info):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    result = await run_scenario(
        Agent(FunctionModel(model)),
        scenario(fixtures=[], checks=[OutputContains(text="done")]),
        timeout=0.01,
    )
    assert not result.passed
    assert "TimeoutError" in result.error
    assert cancelled.is_set()

    def broken(messages, info):
        raise RuntimeError("provider unavailable")

    result = await run_scenario(
        Agent(FunctionModel(broken)), scenario(checks=[OutputContains(text="done")])
    )
    assert result.error == "RuntimeError: provider unavailable"


async def test_partial_trace_survives_error_after_successful_tool_call():
    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"order_id": "123"})])
        raise RuntimeError("model broke on second turn")

    def lookup(order_id: str) -> str:
        raise AssertionError("REAL TOOL EXECUTED")

    result = await run_scenario(Agent(FunctionModel(model), tools=[lookup]), scenario())
    assert not result.passed
    assert len(result.calls) == 1
    assert result.calls[0].outcome == "return"


async def test_request_budget_stops_fast_tool_loop():
    def lookup(order_id: str) -> str:
        raise AssertionError("REAL TOOL EXECUTED")

    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("lookup", {"order_id": "123"})])

    result = await run_scenario(
        Agent(FunctionModel(model), tools=[lookup]),
        scenario(fixtures=[fixture() for _ in range(30)]),
    )
    assert not result.passed
    assert "request_limit" in result.error
    assert len(result.calls) == 20


async def test_unsupported_toolset_rejected_before_lifecycle_or_tool_execution():
    effects = []

    class RemoteToolset(AbstractToolset):
        @property
        def id(self):
            return "remote"

        async def __aenter__(self):
            effects.append("opened remote connection")
            return self

        async def get_tools(self, ctx):
            effects.append("remote listing")
            return {}

        async def call_tool(self, *args):
            effects.append("remote call")

    agent = Agent(FunctionModel(lambda messages, info: None), toolsets=[RemoteToolset()])
    with pytest.raises(ValueError, match="Unsupported toolset"):
        await run_scenario(agent, scenario())
    assert effects == []


async def test_structured_output_and_deferred_tools_are_refused():
    with pytest.raises(ValueError, match="string agent output"):
        await run_scenario(
            Agent(FunctionModel(lambda messages, info: None), output_type=int), scenario()
        )

    toolset = FunctionToolset(requires_approval=True)

    @toolset.tool_plain
    def lookup(order_id: str) -> str:
        raise AssertionError("REAL TOOL EXECUTED")

    with pytest.raises(ValueError, match="Unsupported deferred"):
        await run_scenario(
            Agent(FunctionModel(lambda messages, info: None), toolsets=[toolset]), scenario()
        )


@pytest.mark.parametrize(
    "option,value", [("repeats", 0), ("concurrency", True), ("timeout", float("nan"))]
)
async def test_invalid_runner_options_fail_early(option, value):
    with pytest.raises(ValueError):
        await run_suite(Agent(FunctionModel(lambda messages, info: None)), [], **{option: value})


async def test_empty_suite_is_not_a_success():
    with pytest.raises(ValueError, match="at least one scenario"):
        await run_suite(Agent(FunctionModel(lambda messages, info: None)), [])


async def test_factory_runtime_errors_are_contained_per_run():
    count = 0

    def factory():
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("factory failed")
        return agent_for_calls([[ToolCallPart("lookup", {"order_id": "123"})]])

    result = await run_suite(factory, [scenario()], repeats=2)
    assert result.failed_runs == 1
    assert result.passed_runs == 1
    assert "factory failed" in result.results[0].error


async def test_async_factory_timeout_is_bounded_and_cancelled():
    cancelled = asyncio.Event()

    async def factory():
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    result = await run_suite(factory, [scenario()], timeout=0.01)
    assert result.failed_runs == 1
    assert "TimeoutError" in result.results[0].error
    assert cancelled.is_set()


async def test_configuration_failure_cancels_sibling_runs():
    count = 0
    cancelled = asyncio.Event()

    async def model(messages, info):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    def factory():
        nonlocal count
        count += 1
        if count == 1:
            return Agent(FunctionModel(model))
        return object()

    with pytest.raises(ValueError, match="must return"):
        await run_suite(factory, [scenario()], repeats=2, concurrency=2)
    await asyncio.sleep(0)
    assert not any(
        not task.done() and task is not asyncio.current_task() for task in asyncio.all_tasks()
    )


@pytest.mark.parametrize(
    "raw", ['{"order_id":NaN}', '{"order_id":Infinity}', {"order_id": float("nan")}]
)
async def test_nonfinite_arguments_are_observable_failures_not_suite_crashes(raw):
    def lookup(order_id: float) -> str:
        raise AssertionError("REAL TOOL EXECUTED")

    result = await run_scenario(
        agent_for_calls([[ToolCallPart("lookup", raw)]], tools=[lookup]),
        scenario(),
    )
    assert not result.passed
    assert "INVALID_JSON" in result.calls[0].arguments


async def test_in_place_artifact_mutations_are_revalidated_before_model_calls():
    case = scenario()
    case.checks.append({"kind": "not_a_check"})

    def model(messages, info):
        raise AssertionError("MODEL MUST NOT BE CALLED")

    with pytest.raises(ValueError):
        await run_scenario(Agent(FunctionModel(model)), case)


async def test_duplicate_call_ids_are_rejected_without_assigning_wrong_responses():
    result = await run_scenario(
        agent_for_calls(
            [
                [
                    ToolCallPart("lookup", {"order_id": "123"}, "same"),
                    ToolCallPart("lookup", {"order_id": "456"}, "same"),
                ]
            ]
        ),
        scenario(
            fixtures=[
                fixture(response="first"),
                fixture(arguments={"order_id": "456"}, response="second"),
            ]
        ),
    )
    assert not result.passed
    assert len(result.calls) == 2
    assert result.calls[0].response != "first"
    assert result.calls[1].response != "second"
    assert result.replay_errors


async def test_nonfinite_validation_error_payload_stays_json_safe():
    def lookup(order_id: int) -> str:
        raise AssertionError("REAL TOOL EXECUTED")

    result = await run_scenario(
        agent_for_calls([[ToolCallPart("lookup", {"order_id": float("nan")})]], tools=[lookup]),
        scenario(),
    )
    assert not result.passed
    assert result.calls[0].response[0]["input"] == {"INVALID_JSON": "nan"}
    assert "NaN" not in result.model_dump_json()


async def test_suite_timeout_preserves_completed_mock_tool_trace():
    def lookup(order_id: str) -> str:
        raise AssertionError("REAL TOOL EXECUTED")

    async def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"order_id": "123"})])
        await asyncio.sleep(10)
        return ModelResponse(parts=[TextPart("done")])

    result = await run_suite(
        Agent(FunctionModel(model), tools=[lookup]), [scenario()], timeout=0.02
    )
    assert result.failed_runs == 1
    assert "TimeoutError" in result.results[0].error
    assert len(result.results[0].calls) == 1
    assert result.results[0].calls[0].outcome == "return"


async def test_explicit_model_override_wins_over_ambient_override():
    agent = agent_for_calls([[ToolCallPart("lookup", {"order_id": "123"})]])

    def forbidden_model(messages, info):
        raise AssertionError("AMBIENT MODEL MUST NOT BE USED")

    def explicit_model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"order_id": "123"})])
        return ModelResponse(parts=[TextPart("Explicit model")])

    with agent.override(model=FunctionModel(forbidden_model)):
        result = await run_scenario(agent, scenario(), model=FunctionModel(explicit_model))
    assert result.passed, result
    assert result.output == "Explicit model"


async def test_no_explicit_model_preserves_ambient_override():
    agent = agent_for_calls([[ToolCallPart("lookup", {"order_id": "123"})]])

    def ambient_model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"order_id": "123"})])
        return ModelResponse(parts=[TextPart("Ambient model")])

    with agent.override(model=FunctionModel(ambient_model)):
        result = await run_scenario(agent, scenario())
    assert result.passed, result
    assert result.output == "Ambient model"
