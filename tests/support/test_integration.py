"""Support workflow tests spanning real tools, recording, policy review and replay.

Every model used here is the labeled deterministic test double. Provider requests
remain disabled; direct calls only change disposable synthetic local stores.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from agent_regression.adapter import calls_from_messages, record
from agent_regression.checks import evaluate_checks
from agent_regression.runner import run_scenario, run_suite
from agent_regression.schema import (
    ArgumentCheck,
    BeforeCheck,
    ForbiddenCall,
    MaxCalls,
    RequiredCall,
    Scenario,
    ToolCall,
    ToolFixture,
)
from agent_regression.storage import load_scenario, save_scenario
from agent_regression.support.agent import make_support_agent
from agent_regression.support.scenarios import support_scenarios
from agent_regression.support.store import LocalSupportStore, SupportSession


def _case(name: str) -> Scenario:
    return next(scenario for scenario in support_scenarios() if scenario.name == name)


def _trace_signature(calls: list[ToolCall]) -> list[dict[str, Any]]:
    # Provider/generated call IDs and runtime duration are observational details,
    # not business behavior. Keep the causal turn, response, and outcome.
    return [call.model_dump(exclude={"tool_call_id"}) for call in calls]


@pytest.mark.parametrize(
    ("mutation", "case_name", "required_code"),
    [
        ("refund_ineligible", "outside_window", "forbidden_call"),
        ("refund_ineligible", "final_sale", "forbidden_call"),
        ("refund_ineligible", "already_refunded", "forbidden_call"),
        ("refund_ineligible", "window_31_days", "forbidden_call"),
        ("refund_ineligible", "prompt_injection", "forbidden_call"),
        ("refund_ineligible", "high_value_escalation", "forbidden_call"),
        ("refund_ineligible", "undelivered", "forbidden_call"),
        ("wrong_order_id", "eligible_order", "incorrect_argument"),
        ("wrong_order_id", "outside_window", "incorrect_argument"),
        ("duplicate_refund", "eligible_order", "too_many_calls"),
        ("duplicate_refund", "duplicate_wording", "too_many_calls"),
        ("duplicate_refund", "window_30_days", "too_many_calls"),
        ("skip_lookup", "eligible_order", "incorrect_order"),
        ("parallel_refund", "eligible_order", "incorrect_order"),
        ("guess_ambiguous", "ambiguous_no_id", "forbidden_call"),
        ("guess_ambiguous", "ambiguous_multiple_ids", "forbidden_call"),
        ("ignore_escalation", "high_value_escalation", "missing_call"),
        ("ignore_escalation", "undelivered", "missing_call"),
        ("ignore_escalation", "explicit_human", "missing_call"),
        ("ignore_escalation", "human_order", "missing_call"),
        ("ignore_escalation", "policy_exception", "missing_call"),
        ("over_refund", "eligible_order", "incorrect_argument"),
        ("over_refund", "window_30_days", "incorrect_argument"),
    ],
)
async def test_each_seeded_regression_is_caught_by_an_intended_behavior_rule(
    mutation: str, case_name: str, required_code: str
):
    scenario = _case(case_name)
    safe = await run_scenario(make_support_agent(), scenario)
    assert safe.passed, safe.model_dump_json(indent=2)
    session = SupportSession()
    broken = await run_scenario(
        make_support_agent(session=session, prompt_variant=mutation), scenario
    )
    assert not broken.passed
    assert required_code in {f.code for f in broken.failures}, broken.model_dump_json(indent=2)
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}
    # Continuation fixtures permit the seeded policy errors to complete. The
    # wrong-ID mutation is intentionally unmatched: it must never fall through.
    if mutation != "wrong_order_id":
        assert broken.error is None, broken.model_dump_json(indent=2)
        assert broken.replay_errors == [], broken.model_dump_json(indent=2)


@pytest.mark.parametrize(
    ("case_name", "mutation", "tool_name", "expected", "observed"),
    [
        ("outside_window", "refund_ineligible", "issue_refund", "no issue_refund call", None),
        ("eligible_order", "over_refund", "issue_refund", 4999, 5000),
        ("eligible_order", "duplicate_refund", "issue_refund", 1, 2),
    ],
)
async def test_failure_report_preserves_action_and_expected_observed_evidence(
    case_name, mutation, tool_name, expected, observed
):
    result = await run_scenario(make_support_agent(prompt_variant=mutation), _case(case_name))
    failures = [f for f in result.failures if f.tool_name == tool_name and f.expected == expected]
    assert failures, result.model_dump_json(indent=2)
    if observed is not None:
        assert any(f.observed == observed for f in failures)
    else:
        assert failures[0].observed == {
            "order_id": "ORD-1002",
            "amount_cents": 8999,
            "idempotency_key": "req-support-demo",
        }
        assert failures[0].call_index == 1
    assert any(d.tool_name == tool_name for d in result.differences)


async def test_repeated_parallel_replay_keeps_business_store_and_traces_isolated():
    sessions: list[SupportSession] = []

    def factory():
        session = SupportSession()
        sessions.append(session)
        return make_support_agent(session=session)

    scenarios = support_scenarios()
    suite = await run_suite(factory, scenarios, repeats=3, concurrency=4)
    assert suite.total_runs == len(scenarios) * 3
    assert suite.failed_runs == 0, suite.model_dump_json(indent=2)
    assert len(sessions) == suite.total_runs
    assert len({id(session.store) for session in sessions}) == suite.total_runs
    assert all(
        session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}
        for session in sessions
    )
    assert all(session.looked_up_order_ids == set() for session in sessions)
    grouped = defaultdict(list)
    for result in suite.results:
        grouped[result.scenario_name].append(result)
    for scenario in scenarios:
        results = grouped[scenario.name]
        assert [result.iteration for result in results] == [1, 2, 3]
        expected = _trace_signature(scenario.recorded_calls)
        assert all(_trace_signature(result.calls) == expected for result in results)
        assert all(result.differences == [] for result in results)
        assert len({result.output for result in results}) == 1


async def test_repeated_mutation_reports_real_failure_counts_and_no_state_leaks():
    sessions: list[SupportSession] = []

    def factory():
        session = SupportSession()
        sessions.append(session)
        return make_support_agent(session=session, prompt_variant="refund_ineligible")

    scenarios = [_case("outside_window"), _case("eligible_order")]
    suite = await run_suite(factory, scenarios, repeats=5, concurrency=3)
    assert suite.total_runs == 10
    assert suite.failed_runs == 5
    assert all(not r.passed for r in suite.results if r.scenario_name == "outside_window")
    assert all(r.passed for r in suite.results if r.scenario_name == "eligible_order")
    assert all(s.store.audit_events == [] and s.store.refunds == [] for s in sessions)


async def test_replay_intercepts_both_bound_and_runtime_dependency_stores():
    bound = SupportSession(customer_id="cust-002")
    runtime = SupportSession()
    result = await run_scenario(
        make_support_agent(session=bound), _case("eligible_order"), deps=runtime
    )
    assert result.passed, result.model_dump_json(indent=2)
    for session in [bound, runtime]:
        assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}
        assert session.looked_up_order_ids == set()


@pytest.mark.parametrize(
    ("mutation", "case_name", "denial_reason", "failure_code"),
    [
        ("refund_ineligible", "outside_window", "outside_window", "forbidden_call"),
        ("refund_ineligible", "final_sale", "final_sale", "forbidden_call"),
        ("refund_ineligible", "already_refunded", "invalid_amount", "forbidden_call"),
        ("skip_lookup", "eligible_order", "lookup_required", "incorrect_order"),
        ("over_refund", "eligible_order", "invalid_amount", "incorrect_argument"),
    ],
)
async def test_backend_denial_and_regression_check_independently_detect_prohibited_attempts(
    mutation, case_name, denial_reason, failure_code
):
    scenario = _case(case_name)
    actual_session = SupportSession()
    actual = await make_support_agent(session=actual_session, prompt_variant=mutation).run(
        scenario.prompt
    )
    actual_calls = calls_from_messages(actual.all_messages())
    refund_calls = [call for call in actual_calls if call.tool_name == "issue_refund"]
    assert len(refund_calls) == 1
    assert refund_calls[0].response == {
        "status": "denied",
        "order_id": refund_calls[0].arguments["order_id"],
        "reason_code": denial_reason,
    }
    assert actual_session.store.refunds == []
    assert failure_code in {f.code for f in evaluate_checks(scenario, actual_calls, actual.output)}
    replay_session = SupportSession()
    replay = await run_scenario(
        make_support_agent(session=replay_session, prompt_variant=mutation), scenario
    )
    assert failure_code in {f.code for f in replay.failures}
    assert replay_session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


async def test_duplicate_action_is_a_regression_even_when_backend_idempotency_prevents_damage():
    scenario = _case("duplicate_wording")
    live_session = SupportSession()
    actual = await make_support_agent(session=live_session, prompt_variant="duplicate_refund").run(
        scenario.prompt
    )
    live_calls = calls_from_messages(actual.all_messages())
    refunds = [call for call in live_calls if call.tool_name == "issue_refund"]
    assert len(refunds) == 2
    assert refunds[0].response["refund_id"] == refunds[1].response["refund_id"]
    assert refunds[1].response["idempotent"] is True
    assert len(live_session.store.refunds) == 1
    assert sum(event["action"] == "issue_refund" for event in live_session.store.audit_events) == 1
    assert any(f.code == "too_many_calls" for f in evaluate_checks(scenario, live_calls))
    replay = await run_scenario(make_support_agent(prompt_variant="duplicate_refund"), scenario)
    assert any(f.code == "too_many_calls" for f in replay.failures)
    assert replay.error is None and not replay.replay_errors


async def test_parallel_attempt_fails_causal_check_despite_serial_order_in_trace():
    scenario = _case("eligible_order")
    result = await run_scenario(make_support_agent(prompt_variant="parallel_refund"), scenario)
    assert [call.tool_name for call in result.calls] == ["lookup_order", "issue_refund"]
    assert result.calls[0].outcome == "return"
    assert result.calls[0].step == result.calls[1].step == 0
    assert any(f.code == "incorrect_order" and f.call_index == 1 for f in result.failures)
    assert result.error is None and not result.replay_errors


async def test_record_independent_review_save_and_replay_full_workflow(tmp_path: Path):
    recording_session = SupportSession()
    prompt = "Please refund ORD-1001 if it qualifies."
    recorded = await record(
        make_support_agent(session=recording_session), prompt, name="recorded_eligible"
    )
    assert len(recording_session.store.refunds) == 1
    assert [call.tool_name for call in recorded.recorded_calls] == ["lookup_order", "issue_refund"]
    assert not recorded.reviewed
    assert recorded.checks == []
    # Business expectations are independently specified, rather than accepting
    # whatever the recorder observed as the rule for future runs.
    recorded.checks = [
        RequiredCall(tool_name="lookup_order", arguments={"order_id": "ORD-1001"}),
        RequiredCall(tool_name="issue_refund"),
        ArgumentCheck(tool_name="lookup_order", path="order_id", expected="ORD-1001"),
        ArgumentCheck(tool_name="issue_refund", path="order_id", expected="ORD-1001"),
        ArgumentCheck(tool_name="issue_refund", path="amount_cents", expected=4999),
        ArgumentCheck(
            tool_name="issue_refund", path="idempotency_key", expected="req-support-demo"
        ),
        BeforeCheck(first="lookup_order", then="issue_refund"),
        MaxCalls(tool_name="issue_refund", maximum=1),
        ForbiddenCall(tool_name="escalate_to_human"),
    ]
    recorded.reviewed = True
    saved = save_scenario(recorded, tmp_path / "eligible.json")
    reviewed = load_scenario(saved)
    replay_session = SupportSession()
    safe = await run_scenario(make_support_agent(session=replay_session), reviewed)
    assert safe.passed, safe.model_dump_json(indent=2)
    assert safe.differences == []
    assert replay_session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}
    # Add an explicitly synthetic duplicate return so the policy failure is
    # reportable without ending at an exhausted baseline fixture.
    reviewed.fixtures.append(
        ToolFixture(
            tool_name="issue_refund",
            arguments={
                "order_id": "ORD-1001",
                "amount_cents": 4999,
                "idempotency_key": "req-support-demo",
            },
            response={
                "status": "refunded",
                "order_id": "ORD-1001",
                "refund_id": "refund-0001",
                "amount_cents": 4999,
                "currency": "CAD",
                "idempotent": True,
            },
        )
    )
    broken = await run_scenario(make_support_agent(prompt_variant="duplicate_refund"), reviewed)
    assert not broken.passed
    assert any(f.code == "too_many_calls" for f in broken.failures)
    assert broken.error is None and broken.replay_errors == []


async def test_recording_an_existing_bug_does_not_turn_it_into_expected_behavior():
    live_session = SupportSession()
    recorded = await record(
        make_support_agent(session=live_session, prompt_variant="refund_ineligible"),
        "Please refund ORD-1002.",
        name="observed_bug",
    )
    assert not recorded.reviewed and recorded.checks == []
    assert [call.tool_name for call in recorded.recorded_calls] == ["lookup_order", "issue_refund"]
    assert live_session.store.refunds == []
    blocked_session = SupportSession()
    unreviewed = await run_scenario(make_support_agent(session=blocked_session), recorded)
    assert {f.code for f in unreviewed.failures} == {"unreviewed"}
    assert unreviewed.calls == []
    assert blocked_session.store.audit_events == []
    recorded.checks = [
        RequiredCall(tool_name="lookup_order", arguments={"order_id": "ORD-1002"}),
        ArgumentCheck(tool_name="lookup_order", path="order_id", expected="ORD-1002"),
        ForbiddenCall(tool_name="issue_refund"),
        ForbiddenCall(tool_name="escalate_to_human"),
    ]
    recorded.reviewed = True
    broken = await run_scenario(make_support_agent(prompt_variant="refund_ineligible"), recorded)
    assert not broken.passed
    assert any(f.code == "forbidden_call" for f in broken.failures)
    assert broken.differences == []  # Matching a bad recording is still a failure.
    safe = await run_scenario(make_support_agent(), recorded)
    assert safe.passed, safe.model_dump_json(indent=2)
    assert any(d.kind == "missing_call" and d.tool_name == "issue_refund" for d in safe.differences)


@pytest.mark.parametrize("scenario", support_scenarios(), ids=lambda s: s.name)
async def test_all_safe_requests_execute_real_local_tools_against_reviewed_baselines(scenario):
    store = LocalSupportStore.from_seed()
    session = SupportSession(store=store)
    response = await make_support_agent(session=session).run(scenario.prompt)
    calls = calls_from_messages(response.all_messages())
    assert _trace_signature(calls) == _trace_signature(scenario.recorded_calls), scenario.name
    assert evaluate_checks(scenario, calls, response.output) == []
    expected_refunds = sum(call.tool_name == "issue_refund" for call in scenario.recorded_calls)
    expected_tickets = sum(
        call.tool_name == "escalate_to_human" for call in scenario.recorded_calls
    )
    assert len(store.refunds) == expected_refunds
    assert len(store.escalations) == expected_tickets
    if store.escalations:
        assert store.escalations[0]["user_message"] == scenario.prompt
