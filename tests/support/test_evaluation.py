from __future__ import annotations

from agent_regression import run_suite
from agent_regression.support import make_support_agent, support_scenarios
from agent_regression.support.evaluation import diagnostic_scenarios


async def test_diagnostic_wording_keeps_independent_policy_and_replay_checks() -> None:
    scenarios = diagnostic_scenarios()
    originals = support_scenarios()
    assert len(scenarios) == 12
    assert {item.prompt for item in scenarios}.isdisjoint(item.prompt for item in originals)
    assert {item.name for item in scenarios}.isdisjoint(item.name for item in originals)
    assert all(item.reviewed and item.checks for item in scenarios)
    suite = await run_suite(make_support_agent, scenarios)
    assert suite.failed_runs == 0, suite.model_dump_json(indent=2)
    assert originals == support_scenarios()


def test_diagnostic_scenarios_are_fresh_and_cannot_mutate_originals() -> None:
    first = diagnostic_scenarios()
    second = diagnostic_scenarios()
    first[0].checks.clear()
    first[0].fixtures.clear()
    assert second == diagnostic_scenarios()
    assert support_scenarios()[0].checks
