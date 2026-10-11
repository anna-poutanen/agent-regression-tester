from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent_regression.schema import (
    ArgumentCheck,
    BeforeCheck,
    CheckFailure,
    ForbiddenCall,
    MaxCalls,
    OutputContains,
    RequiredCall,
    RunResult,
    Scenario,
    SuiteResult,
    ToolCall,
    ToolFixture,
)


def test_scenario_round_trip_preserves_all_check_variants() -> None:
    checks = [
        RequiredCall(tool_name="lookup_order", arguments={"order_id": "123"}),
        ForbiddenCall(tool_name="issue_refund"),
        ArgumentCheck(tool_name="lookup_order", path="order_id", expected="123"),
        BeforeCheck(first="lookup_order", then="issue_refund"),
        MaxCalls(tool_name="lookup_order", maximum=2),
        OutputContains(text="ineligible"),
    ]
    scenario = Scenario(
        name="ineligible_order",
        prompt="Refund order 123",
        checks=checks,
        fixtures=[
            ToolFixture(
                tool_name="lookup_order",
                arguments={"order_id": "123"},
                response={"eligible": False},
            )
        ],
        recorded_calls=[
            ToolCall(
                tool_name="lookup_order",
                arguments={"order_id": "123"},
                response={"eligible": False},
            )
        ],
        metadata={"source": "test", "versions": [1, None, True]},
    )
    restored = Scenario.model_validate_json(scenario.model_dump_json())
    assert restored == scenario
    assert not restored.reviewed
    assert isinstance(restored.checks[2], ArgumentCheck)


@pytest.mark.parametrize("version", [0, 2, "1", True, 1.0])
def test_reject_unsupported_and_coerced_schema_versions(version: Any) -> None:
    with pytest.raises(ValidationError):
        Scenario.model_validate({"schema_version": version, "name": "test", "prompt": "request"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), (1, 2), {1: "value"}, object()])
def test_reject_non_json_values(value: Any) -> None:
    with pytest.raises(ValidationError):
        ToolCall(tool_name="tool", arguments={"nested": {"value": value}})


def test_unknown_fields_and_invalid_discriminators_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Scenario.model_validate({"name": "test", "prompt": "request", "surprise": 1})
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {"name": "test", "prompt": "request", "checks": [{"kind": "regex", "pattern": ".*"}]}
        )


@pytest.mark.parametrize("name", ["", " ", "\n"])
def test_blank_names_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        ToolCall(tool_name=name, arguments={})


@pytest.mark.parametrize("path", ["", ".id", "id.", "id..value", "id. value"])
def test_invalid_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValidationError):
        ArgumentCheck(tool_name="tool", path=path)


@pytest.mark.parametrize(
    "model,fields",
    [
        (RequiredCall, {"tool_name": "tool", "min_calls": 0}),
        (RequiredCall, {"tool_name": "tool", "min_calls": True}),
        (MaxCalls, {"tool_name": "tool", "maximum": -1}),
        (ToolCall, {"tool_name": "tool", "arguments": {}, "step": -1}),
        (RunResult, {"scenario_name": "test", "iteration": 0}),
        (RunResult, {"scenario_name": "test", "duration_seconds": float("inf")}),
    ],
)
def test_invalid_counts_are_rejected(model: Any, fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model(**fields)


def test_mutable_defaults_are_independent_and_assignment_is_validated() -> None:
    first = Scenario(name="first", prompt="request")
    second = Scenario(name="second", prompt="request")
    first.metadata["value"] = "changed"
    assert second.metadata == {}
    with pytest.raises(ValidationError):
        first.reviewed = "yes"  # type: ignore[assignment]


def test_suite_totals_and_passed_require_every_success_condition() -> None:
    results = [
        RunResult(scenario_name="pass"),
        RunResult(scenario_name="check", failures=[CheckFailure(code="bad", message="wrong")]),
        RunResult(scenario_name="replay", replay_errors=["No fixture"]),
        RunResult(scenario_name="error", error="Timed out"),
        RunResult(scenario_name="empty_error", error=""),
    ]
    suite = SuiteResult(results=results)
    assert (suite.total_runs, suite.passed_runs, suite.failed_runs) == (5, 1, 4)
    assert suite.model_dump()["failed_runs"] == 4


def test_retry_fixtures_allowed_but_error_fixtures_rejected() -> None:
    fixture = ToolFixture(tool_name="lookup", arguments={}, outcome="retry", response="Try again")
    assert fixture.outcome == "retry"
    with pytest.raises(ValidationError):
        ToolFixture.model_validate({"tool_name": "lookup", "arguments": {}, "outcome": "error"})
