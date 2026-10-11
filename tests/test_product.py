"""Product-level mutation tests across realistic deterministic agent loops."""

from pathlib import Path

import pytest

from agent_regression import (
    ForbiddenCall,
    load_scenario,
    record,
    run_scenario,
    run_suite,
    save_scenario,
    suggest_checks,
)
from agent_regression.demo import (
    crm_scenarios,
    make_crm_agent,
    make_search_agent,
    make_support_agent,
    search_scenarios,
    support_scenarios,
)


@pytest.mark.parametrize(
    ("factory", "scenarios"),
    [
        (make_support_agent, support_scenarios),
        (make_search_agent, search_scenarios),
        (make_crm_agent, crm_scenarios),
    ],
)
async def test_all_safe_examples_pass_with_fresh_repeated_state(factory, scenarios):
    suite = await run_suite(factory, scenarios(), repeats=3, concurrency=2)
    assert suite.failed_runs == 0, suite.model_dump_json(indent=2)
    assert suite.total_runs == len(scenarios()) * 3
    assert all(result.calls for result in suite.results)


@pytest.mark.parametrize(
    ("mode", "scenario_index", "expected_code"),
    [
        ("refund_ineligible", 0, "forbidden"),
        ("wrong_id", 0, "incorrect_argument"),
        ("skip_lookup", 1, "incorrect_order"),
        ("duplicate", 1, "too_many_calls"),
        ("parallel", 1, "incorrect_order"),
    ],
)
async def test_seeded_regressions_are_detected(mode, scenario_index, expected_code):
    result = await run_scenario(make_support_agent(mode), support_scenarios()[scenario_index])
    assert not result.passed
    assert any(expected_code in f.code for f in result.failures), result.model_dump_json(indent=2)


async def test_forbidden_refund_report_keeps_full_trace():
    result = await run_scenario(make_support_agent("refund_ineligible"), support_scenarios()[0])
    assert [call.tool_name for call in result.calls] == ["lookup_order", "issue_refund"]
    assert result.calls[0].response == {"order_id": "123", "eligible": False}
    assert result.calls[1].arguments == {"order_id": "123"}
    assert not result.replay_errors
    assert any(diff.tool_name == "issue_refund" for diff in result.differences)


async def test_record_review_roundtrip(tmp_path: Path):
    recorded = await record(
        make_support_agent(), "Refund order 123 if it is eligible.", name="recorded_failure_case"
    )
    assert not recorded.reviewed
    assert len(recorded.fixtures) == 1
    assert recorded.fixtures[0].response == {"order_id": "123", "eligible": False}
    draft_checks = suggest_checks(recorded)
    assert draft_checks
    assert not any(isinstance(check, ForbiddenCall) for check in draft_checks)
    recorded.checks = [*draft_checks, ForbiddenCall(tool_name="issue_refund")]
    recorded.reviewed = True
    path = save_scenario(recorded, tmp_path / "scenario.json")
    scenario = load_scenario(path)
    safe_result = await run_scenario(make_support_agent(), scenario)
    assert safe_result.passed, safe_result.model_dump_json(indent=2)
    broken_result = await run_scenario(make_support_agent("refund_ineligible"), scenario)
    assert not broken_result.passed
    assert any(call.tool_name == "issue_refund" for call in broken_result.calls)


async def test_failed_runs_out_of_n_is_real_count():
    suite = await run_suite(
        lambda: make_support_agent("refund_ineligible"), support_scenarios(), repeats=5
    )
    ineligible = [r for r in suite.results if r.scenario_name == "ineligible_order"]
    eligible = [r for r in suite.results if r.scenario_name == "eligible_order"]
    assert sum(not result.passed for result in ineligible) == 5
    assert sum(not result.passed for result in eligible) == 0
    assert suite.failed_runs == 5
    assert suite.total_runs == 10


async def test_repeat_frequency_reports_three_of_ten_failures():
    """A scheduled mutation gives exact counts without pretending to measure an LLM."""
    created = 0

    def factory():
        nonlocal created
        created += 1
        mode = "refund_ineligible" if created in {3, 6, 9} else "safe"
        return make_support_agent(mode)

    suite = await run_suite(factory, [support_scenarios()[0]], repeats=10, concurrency=1)
    assert suite.total_runs == 10
    assert suite.failed_runs == 3
    assert [result.iteration for result in suite.results if not result.passed] == [3, 6, 9]
