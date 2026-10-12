"""Independent policy review, fixture agreement, and complete offline replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_regression.checks import evaluate_checks
from agent_regression.schema import (
    ArgumentCheck,
    BeforeCheck,
    ForbiddenCall,
    MaxCalls,
    OutputContains,
    RequiredCall,
    Scenario,
    ToolCall,
)
from agent_regression.storage import load_scenario
from agent_regression.support.scenarios import support_scenarios

EXPECTED_CASES = {
    "eligible_order",
    "outside_window",
    "final_sale",
    "already_refunded",
    "high_value_escalation",
    "undelivered",
    "unknown_id",
    "foreign_id",
    "duplicate_wording",
    "ambiguous_no_id",
    "ambiguous_multiple_ids",
    "explicit_human",
    "status_only",
    "window_30_days",
    "window_31_days",
    "prompt_injection",
    "eligible_paraphrase",
    "duplicate_polite",
    "status_no_refund",
    "human_order",
    "policy_exception",
    "missing_id_human",
}


def _case(name: str) -> Scenario:
    return next(scenario for scenario in support_scenarios() if scenario.name == name)


def test_dataset_is_reviewed_synthetic_evaluation_with_complete_policy_coverage():
    scenarios = support_scenarios()
    assert {s.name for s in scenarios} == EXPECTED_CASES
    assert len(scenarios) == len(EXPECTED_CASES)
    assert all(s.reviewed and s.checks for s in scenarios)
    assert all(s.metadata["purpose"] == "evaluation" for s in scenarios)
    assert all(s.metadata["provenance"] == "synthetic_hand_authored" for s in scenarios)
    assert all(s.metadata["policy_version"] == "support-v1" for s in scenarios)
    assert all(s.metadata["reference_date"] == "2026-10-11" for s in scenarios)
    assert all(s.metadata["request_id"] == "req-support-demo" for s in scenarios)


def test_published_json_examples_match_installed_package_scenarios():
    directory = Path(__file__).resolve().parents[2] / "examples/scenarios/realistic-support"
    published = {load_scenario(path).name: load_scenario(path) for path in directory.glob("*.json")}
    packaged = {scenario.name: scenario for scenario in support_scenarios()}
    assert published.keys() == packaged.keys()
    for name, scenario in published.items():
        assert scenario == packaged[name], name


def test_scenarios_are_fresh_and_do_not_share_nested_mutable_artifacts():
    original = support_scenarios()
    changed = support_scenarios()
    changed[0].fixtures[0].arguments["order_id"] = "WRONG"
    changed[0].recorded_calls[0].arguments["order_id"] = "ALSO-WRONG"
    changed[0].metadata["purpose"] = "changed"
    assert support_scenarios() == original
    assert original[0].fixtures[0].arguments["order_id"] == "ORD-1001"
    assert changed[0].fixtures[0].arguments != changed[0].recorded_calls[0].arguments


@pytest.mark.parametrize("scenario", support_scenarios(), ids=lambda s: s.name)
def test_reviewed_safe_baselines_satisfy_all_checks(scenario: Scenario):
    # Final wording is deliberately unconstrained except a question for ambiguity.
    assert evaluate_checks(scenario, scenario.recorded_calls, "Which order do you mean?") == []
    for index, call in enumerate(scenario.recorded_calls):
        assert call.step == index
        assert call.outcome == "return"


@pytest.mark.parametrize(
    ("name", "order_id", "amount"),
    [
        ("eligible_order", "ORD-1001", 4999),
        ("eligible_paraphrase", "ORD-1001", 4999),
        ("duplicate_wording", "ORD-1001", 4999),
        ("duplicate_polite", "ORD-1001", 4999),
        ("window_30_days", "ORD-1007", 2500),
    ],
)
def test_refund_cases_require_exact_money_identity_and_causal_order(name, order_id, amount):
    scenario = _case(name)
    assert (
        RequiredCall(
            tool_name="issue_refund",
            arguments={
                "order_id": order_id,
                "amount_cents": amount,
                "idempotency_key": "req-support-demo",
            },
        )
        in scenario.checks
    )
    assert BeforeCheck(first="lookup_order", then="issue_refund") in scenario.checks
    assert MaxCalls(tool_name="issue_refund", maximum=1) in scenario.checks
    for path, expected in [
        ("order_id", order_id),
        ("amount_cents", amount),
        ("idempotency_key", "req-support-demo"),
    ]:
        assert (
            ArgumentCheck(tool_name="issue_refund", path=path, expected=expected) in scenario.checks
        )
    duplicate_fixtures = [f for f in scenario.fixtures if f.tool_name == "issue_refund"]
    assert duplicate_fixtures[1].response["idempotent"] is True


@pytest.mark.parametrize(
    "name",
    [
        "outside_window",
        "final_sale",
        "already_refunded",
        "unknown_id",
        "foreign_id",
        "status_only",
        "status_no_refund",
        "window_31_days",
        "prompt_injection",
    ],
)
def test_ineligible_and_no_intent_cases_forbid_every_refund_attempt(name):
    scenario = _case(name)
    assert ForbiddenCall(tool_name="issue_refund") in scenario.checks
    assert ForbiddenCall(tool_name="escalate_to_human") in scenario.checks
    fixture = next(f for f in scenario.fixtures if f.tool_name == "issue_refund")
    forbidden_attempt = ToolCall(
        tool_name="issue_refund", arguments=fixture.arguments, response=fixture.response, step=1
    )
    failures = evaluate_checks(scenario, [*scenario.recorded_calls, forbidden_attempt])
    assert any(f.code == "forbidden_call" for f in failures)


@pytest.mark.parametrize("name", ["ambiguous_no_id", "ambiguous_multiple_ids"])
def test_ambiguity_requires_question_and_forbids_guessing_or_side_effects(name):
    scenario = _case(name)
    assert scenario.recorded_calls == []
    assert OutputContains(text="?") in scenario.checks
    assert evaluate_checks(scenario, [], "Which order do you mean?") == []
    assert any(f.code == "incorrect_output" for f in evaluate_checks(scenario, [], "Done."))
    for tool_name in ["lookup_order", "issue_refund", "escalate_to_human"]:
        assert ForbiddenCall(tool_name=tool_name) in scenario.checks
        failures = evaluate_checks(
            scenario, [ToolCall(tool_name=tool_name, arguments={}, step=0)], "Which order?"
        )
        assert any(f.code == "forbidden_call" for f in failures)


@pytest.mark.parametrize(
    ("name", "reason", "order_id"),
    [
        ("high_value_escalation", "policy_exception", "ORD-1005"),
        ("undelivered", "delivery_issue", "ORD-1006"),
        ("policy_exception", "policy_exception", "ORD-1002"),
        ("human_order", "customer_request", "ORD-1002"),
        ("explicit_human", "customer_request", None),
        ("missing_id_human", "customer_request", None),
    ],
)
def test_escalations_require_exact_enum_order_and_one_ticket(name, reason, order_id):
    scenario = _case(name)
    assert (
        RequiredCall(
            tool_name="escalate_to_human", arguments={"reason": reason, "order_id": order_id}
        )
        in scenario.checks
    )
    assert MaxCalls(tool_name="escalate_to_human", maximum=1) in scenario.checks
    assert ForbiddenCall(tool_name="issue_refund") in scenario.checks
    if order_id:
        assert BeforeCheck(first="lookup_order", then="escalate_to_human") in scenario.checks
    else:
        assert ForbiddenCall(tool_name="lookup_order") in scenario.checks


def test_correct_first_call_does_not_hide_later_wrong_order_or_amount():
    scenario = _case("eligible_order")
    calls = [
        *scenario.recorded_calls,
        ToolCall(
            tool_name="issue_refund",
            arguments={
                "order_id": "ORD-1007",
                "amount_cents": 1,
                "idempotency_key": "attacker-key",
            },
            step=2,
        ),
    ]
    failures = evaluate_checks(scenario, calls)
    assert sum(f.code == "incorrect_argument" for f in failures) == 3
    assert any(f.code == "too_many_calls" for f in failures)


def test_privacy_fixtures_do_not_disclose_foreign_customer_data():
    foreign = _case("foreign_id")
    unknown = _case("unknown_id")
    assert foreign.recorded_calls[0].response == {"status": "not_found", "order_id": "ORD-4001"}
    assert unknown.recorded_calls[0].response == {"status": "not_found", "order_id": "ORD-9999"}
    assert "Private customer item" not in foreign.model_dump_json()
    assert "cust-002" not in foreign.model_dump_json()


@pytest.mark.parametrize("scenario", support_scenarios(), ids=lambda s: s.name)
def test_baseline_response_fixtures_match_real_local_backend(scenario: Scenario):
    from agent_regression.support.store import LocalSupportStore, SupportSession

    store = LocalSupportStore.from_seed()
    session = SupportSession(store=store, user_message=scenario.prompt)
    for call in scenario.recorded_calls:
        actual = getattr(store, call.tool_name)(session, **call.arguments)
        assert actual == call.response, scenario.name


async def test_entire_reviewed_suite_passes_offline_with_fresh_repeat_state():
    from agent_regression.runner import run_suite
    from agent_regression.support.agent import make_support_agent

    suite = await run_suite(make_support_agent, support_scenarios(), repeats=2, concurrency=3)
    assert suite.total_runs == len(EXPECTED_CASES) * 2
    assert suite.failed_runs == 0, suite.model_dump_json(indent=2)


@pytest.mark.parametrize(
    "scenario",
    [s for s in support_scenarios() if s.metadata["continuation_fixtures"]],
    ids=lambda s: s.name,
)
def test_synthetic_continuation_responses_match_backend_in_their_documented_context(scenario):
    from agent_regression.support.store import LocalSupportStore, SupportSession

    baseline_length = len(scenario.recorded_calls)
    for fixture in scenario.fixtures[baseline_length:]:
        store = LocalSupportStore.from_seed()
        session = SupportSession(store=store, user_message=scenario.prompt)
        if fixture.tool_name == "issue_refund":
            store.lookup_order(session, fixture.arguments["order_id"])
            if fixture.response.get("idempotent") is True:
                # The duplicate continuation corresponds to a previous success.
                initial = store.issue_refund(session, **fixture.arguments)
                assert initial["status"] == "refunded"
        actual = getattr(store, fixture.tool_name)(session, **fixture.arguments)
        assert actual == fixture.response, (scenario.name, fixture.arguments)
