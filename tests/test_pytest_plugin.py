from __future__ import annotations

from agent_regression.demo import demo_scenarios, make_broken_support_agent, make_support_agent
from agent_regression.schema import RunResult, SuiteResult


async def test_regression_runner_fixture_returns_assertable_result(regression_runner) -> None:
    result: RunResult = await regression_runner(make_support_agent(), demo_scenarios()[0])
    assert result.passed
    assert result.calls[0].tool_name == "lookup_order"


async def test_regression_runner_fixture_exposes_behavior_failure(regression_runner) -> None:
    result: RunResult = await regression_runner(make_broken_support_agent(), demo_scenarios()[0])
    assert not result.passed
    assert any(failure.tool_name == "issue_refund" for failure in result.failures)


async def test_regression_suite_runner_fixture_repeats_and_factories(
    regression_suite_runner,
) -> None:
    suite: SuiteResult = await regression_suite_runner(
        make_support_agent, demo_scenarios(), repeats=3, concurrency=2
    )
    assert suite.total_runs == suite.passed_runs == 6
    assert suite.failed_runs == 0
