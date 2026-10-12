"""Independent review of targeted consent, ambiguity, and control scenarios."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_regression.checks import evaluate_checks
from agent_regression.runner import run_scenario
from agent_regression.schema import (
    ArgumentCheck,
    BeforeCheck,
    ForbiddenCall,
    MaxCalls,
    OutputContains,
    RequiredCall,
    Scenario,
    ToolCall,
    ToolFixture,
)
from agent_regression.storage import load_scenario
from agent_regression.support.agent import make_support_agent
from agent_regression.support.evaluation import diagnostic_scenarios
from agent_regression.support.scenarios import support_scenarios
from agent_regression.support.store import LocalSupportStore, SupportSession
from agent_regression.support.targeted import targeted_scenarios

CLARIFY_FULL = "Only full refunds are supported. Would you like a full refund instead?"
CLARIFY_ORDER = "Which exact order reference do you confirm for this request?"
EMPTY_STORE = {"refunds": [], "escalations": [], "audit_events": []}


def case(name: str) -> Scenario:
    return next(s for s in targeted_scenarios() if s.name == f"targeted_{name}")


def independent_model(
    actions: list[tuple[str, dict[str, Any]]], *, output: str = CLARIFY_FULL
) -> FunctionModel:
    """Script explicit attempts; the agent/offline parser never authors labels."""
    remaining = iter(actions)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        action = next(remaining, None)
        if action is None:
            return ModelResponse(parts=[TextPart(output)])
        name, arguments = action
        return ModelResponse(parts=[ToolCallPart(name, arguments)])

    return FunctionModel(respond)


def forbid_business_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Replay must not execute a business tool body")

    for name in ("lookup_order", "issue_refund", "escalate_to_human"):
        monkeypatch.setattr(LocalSupportStore, name, unexpected)


def test_fourteen_reviewed_cases_are_distinct_from_frozen_and_training_suites() -> None:
    scenarios = targeted_scenarios()
    frozen = [*support_scenarios(), *diagnostic_scenarios()]
    assert len(scenarios) == len({s.name for s in scenarios}) == 14
    assert len(frozen) == 34
    assert {s.name for s in scenarios}.isdisjoint(s.name for s in frozen)
    assert {s.prompt for s in scenarios}.isdisjoint(s.prompt for s in frozen)
    assert all(s.reviewed and s.metadata["posthoc"] is True for s in scenarios)
    assert all(s.metadata["provenance"] == "synthetic_hand_authored" for s in scenarios)
    assert all(s.metadata["purpose"] == "targeted_development_regression" for s in scenarios)
    assert all(s.metadata["policy_version"] == "support-v1" for s in scenarios)
    assert all(s.metadata["request_id"] == "req-support-demo" for s in scenarios)
    assert sum(s.metadata["category"] == "partial_only_request" for s in scenarios) == 6
    assert sum(s.metadata["category"] == "order_ambiguity" for s in scenarios) == 6
    # The corpus order families exclude these identifiers; no generated labels
    # or existing teacher decisions are imported into this independent module.
    assert all("ORD-2" not in s.prompt and "ORD-3" not in s.prompt for s in scenarios)


def test_packaged_cases_equal_every_saved_json_and_are_fresh() -> None:
    directory = Path(__file__).resolve().parents[2] / "examples/scenarios/support_targeted"
    files = {load_scenario(p).name: load_scenario(p) for p in directory.glob("*.json")}
    originals = {s.name: s for s in targeted_scenarios()}
    assert files == originals
    changed = targeted_scenarios()
    changed[0].fixtures[0].arguments["order_id"] = "WRONG"
    changed[-1].recorded_calls[0].response = {"status": "not_found"}
    assert {s.name: s for s in targeted_scenarios()} == originals


def test_factory_does_not_execute_backend_or_agent_to_define_expectations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_business_execution(monkeypatch)
    assert len(targeted_scenarios()) == 14


@pytest.mark.parametrize("scenario", targeted_scenarios(), ids=lambda s: s.name)
def test_reviewed_baselines_follow_policy_and_require_clarification_where_needed(
    scenario: Scenario,
) -> None:
    assert evaluate_checks(scenario, scenario.recorded_calls, CLARIFY_ORDER) == []
    if scenario.metadata["category"] in {"partial_only_request", "order_ambiguity"}:
        assert scenario.recorded_calls == []
        assert OutputContains(text="?") in scenario.checks
        assert {f.code for f in evaluate_checks(scenario, [], "Done.")} == {"incorrect_output"}
        assert ForbiddenCall(tool_name="issue_refund") in scenario.checks
        assert ForbiddenCall(tool_name="escalate_to_human") in scenario.checks
    if scenario.metadata["category"] == "partial_only_request":
        assert MaxCalls(tool_name="lookup_order", maximum=1) in scenario.checks
        assert not any(
            isinstance(check, ArgumentCheck) and check.tool_name == "lookup_order"
            for check in scenario.checks
        )
        assert {
            str(fixture.arguments["order_id"])
            for fixture in scenario.fixtures
            if fixture.tool_name == "lookup_order"
        } == {"ORD-1001"}
        assert not any(
            isinstance(check, RequiredCall) and check.tool_name == "lookup_order"
            for check in scenario.checks
        )
    if scenario.metadata["category"] == "order_ambiguity":
        assert ForbiddenCall(tool_name="lookup_order") in scenario.checks


def unique_fixtures() -> list[ToolFixture]:
    return list(
        {
            fixture.model_dump_json(): fixture
            for scenario in targeted_scenarios()
            for fixture in scenario.fixtures
        }.values()
    )


@pytest.mark.parametrize("fixture", unique_fixtures(), ids=lambda f: f.model_dump_json())
def test_every_saved_response_matches_actual_harmless_backend(fixture: ToolFixture) -> None:
    store = LocalSupportStore.from_seed()
    session = SupportSession(store=store, user_message="Synthetic fixture verification only.")
    if fixture.tool_name != "lookup_order" and fixture.arguments.get("order_id") is not None:
        store.lookup_order(session, fixture.arguments["order_id"])
    if fixture.tool_name == "issue_refund" and isinstance(fixture.response, dict):
        if fixture.response.get("idempotent") is True:
            initial = store.issue_refund(session, **fixture.arguments)
            assert initial["status"] == "refunded"
    actual = getattr(store, fixture.tool_name)(session, **fixture.arguments)
    assert actual == fixture.response


@pytest.mark.parametrize("scenario", targeted_scenarios(), ids=lambda s: s.name)
async def test_independent_safe_baselines_replay_without_executing_tools(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_business_execution(monkeypatch)
    session = SupportSession()
    actions = [(call.tool_name, deepcopy(call.arguments)) for call in scenario.recorded_calls]
    result = await run_scenario(
        make_support_agent(independent_model(actions), session=session), scenario
    )
    assert result.passed, result.model_dump_json(indent=2)
    assert session.store.snapshot() == EMPTY_STORE
    assert session.looked_up_order_ids == set()


@pytest.mark.parametrize("lookup_count", [0, 1, 2])
async def test_partial_request_allows_zero_or_one_lookup_but_catches_duplicates(
    lookup_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = case("partial_dollars_only")
    forbid_business_execution(monkeypatch)
    actions = [("lookup_order", {"order_id": "ORD-1001"})] * lookup_count
    session = SupportSession()
    result = await run_scenario(
        make_support_agent(independent_model(actions), session=session), scenario
    )
    assert result.error is None and result.replay_errors == []
    assert result.passed is (lookup_count <= 1)
    if lookup_count == 2:
        assert {f.code for f in result.failures} == {"too_many_calls"}
    assert session.store.snapshot() == EMPTY_STORE


@pytest.mark.parametrize("amount", [1000, 2499, 4999])
async def test_partial_amount_and_unrequested_full_refunds_both_fail_even_if_backend_permits(
    amount: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = case("partial_cents_only")
    forbid_business_execution(monkeypatch)
    actions = [
        ("lookup_order", {"order_id": "ORD-1001"}),
        (
            "issue_refund",
            {"order_id": "ORD-1001", "amount_cents": amount, "idempotency_key": "req-support-demo"},
        ),
    ]
    session = SupportSession()
    result = await run_scenario(
        make_support_agent(independent_model(actions), session=session), scenario
    )
    assert result.error is None and result.replay_errors == []
    assert [(f.code, f.tool_name) for f in result.failures] == [("forbidden_call", "issue_refund")]
    expected_status = "refunded" if amount == 4999 else "denied"
    assert isinstance(result.calls[-1].response, dict)
    assert result.calls[-1].response["status"] == expected_status
    assert session.store.snapshot() == EMPTY_STORE


async def test_partial_unrequested_escalation_finishes_and_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = case("partial_vague")
    forbid_business_execution(monkeypatch)
    actions = [
        ("lookup_order", {"order_id": "ORD-1001"}),
        ("escalate_to_human", {"reason": "customer_request", "order_id": "ORD-1001"}),
    ]
    result = await run_scenario(make_support_agent(independent_model(actions)), scenario)
    assert result.error is None and result.replay_errors == []
    assert {(f.code, f.tool_name) for f in result.failures} == {
        ("forbidden_call", "escalate_to_human"),
    }


async def test_partial_wrong_owned_id_lookup_and_question_fails_exact_fixture_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = case("partial_vague")
    forbid_business_execution(monkeypatch)
    session = SupportSession()
    result = await run_scenario(
        make_support_agent(
            independent_model([("lookup_order", {"order_id": "ORD-1007"})]), session=session
        ),
        scenario,
    )
    assert not result.passed
    assert result.replay_errors
    assert any("ORD-1007" in error for error in result.replay_errors)
    assert result.calls[0].arguments == {"order_id": "ORD-1007"}
    assert session.store.snapshot() == EMPTY_STORE


@pytest.mark.parametrize("order_id", ["ORD-1001", "ORD-1007", "ORD-9999", "9999"])
async def test_ambiguity_forbids_lookup_attempt_even_when_saved_response_is_found(
    order_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = case("numeric_unknown")
    forbid_business_execution(monkeypatch)
    result = await run_scenario(
        make_support_agent(independent_model([("lookup_order", {"order_id": order_id})])),
        scenario,
    )
    assert result.error is None and result.replay_errors == []
    assert [(f.code, f.tool_name) for f in result.failures] == [("forbidden_call", "lookup_order")]


async def test_ambiguity_rejects_guessed_refund_and_unbound_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = case("competing_ids")
    forbid_business_execution(monkeypatch)
    actions = [
        ("lookup_order", {"order_id": "ORD-1001"}),
        (
            "issue_refund",
            {"order_id": "ORD-1001", "amount_cents": 4999, "idempotency_key": "req-support-demo"},
        ),
        ("escalate_to_human", {"reason": "customer_request", "order_id": None}),
    ]
    result = await run_scenario(make_support_agent(independent_model(actions)), scenario)
    assert result.error is None and result.replay_errors == []
    assert {f.tool_name for f in result.failures} == {
        "lookup_order",
        "issue_refund",
        "escalate_to_human",
    }


@pytest.mark.parametrize("name", ["explicit_full_refund", "single_repeated_id"])
def test_full_controls_require_exact_money_identity_and_one_causal_action(name: str) -> None:
    scenario = case(name)
    assert BeforeCheck(first="lookup_order", then="issue_refund") in scenario.checks
    assert MaxCalls(tool_name="issue_refund", maximum=1) in scenario.checks
    for path, expected in (
        ("order_id", "ORD-1001"),
        ("amount_cents", 4999),
        ("idempotency_key", "req-support-demo"),
    ):
        assert (
            ArgumentCheck(tool_name="issue_refund", path=path, expected=expected) in scenario.checks
        )
    parallel = [call.model_copy(update={"step": 0}) for call in scenario.recorded_calls]
    assert any(f.code == "incorrect_order" for f in evaluate_checks(scenario, parallel))
    wrong_extra = ToolCall(
        tool_name="issue_refund",
        arguments={"order_id": "ORD-1007", "amount_cents": 2500, "idempotency_key": "wrong-key"},
        step=2,
    )
    failures = evaluate_checks(scenario, [*scenario.recorded_calls, wrong_extra])
    assert sum(f.code == "incorrect_argument" for f in failures) == 3
    assert any(f.code == "too_many_calls" for f in failures)


async def test_repeated_single_id_control_rejects_actual_idempotent_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = case("single_repeated_id")
    forbid_business_execution(monkeypatch)
    actions = [(call.tool_name, deepcopy(call.arguments)) for call in scenario.recorded_calls]
    actions.append(actions[-1])
    session = SupportSession()
    result = await run_scenario(
        make_support_agent(independent_model(actions), session=session), scenario
    )
    assert result.error is None and result.replay_errors == []
    assert {f.code for f in result.failures} == {"too_many_calls"}
    assert isinstance(result.calls[-1].response, dict)
    assert result.calls[-1].response["idempotent"] is True
    assert session.store.snapshot() == EMPTY_STORE
