from __future__ import annotations

import pytest
from pydantic import JsonValue

from agent_regression.checks import arguments_match, diff_traces, evaluate_checks, json_equal
from agent_regression.schema import (
    ArgumentCheck,
    BeforeCheck,
    Check,
    ForbiddenCall,
    MaxCalls,
    OutputContains,
    RequiredCall,
    Scenario,
    ToolCall,
)


def evaluate(checks: list[Check], calls: list[ToolCall], output: JsonValue = None):  # type: ignore[no-untyped-def]
    return evaluate_checks(Scenario(name="test", prompt="request", checks=checks), calls, output)


@pytest.mark.parametrize(
    "left,right,equal",
    [
        (False, 0, False),
        (True, 1, False),
        ({"flag": False}, {"flag": 0}, False),
        ([True], [1], False),
        (None, None, True),
        (1, 1.0, True),
        ("1", 1, False),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        ([1, 2], [2, 1], False),
    ],
)
def test_json_equality_preserves_json_types(left: JsonValue, right: JsonValue, equal: bool) -> None:
    assert json_equal(left, right) is equal


def test_recursive_subset_arguments_but_exact_lists() -> None:
    actual = {"customer": {"id": "123", "name": "Anna"}, "tags": ["a", "b"]}
    assert arguments_match(actual, {"customer": {"id": "123"}})
    assert not arguments_match(actual, {"customer": {"id": "456"}})
    assert not arguments_match(actual, {"tags": ["a"]})
    assert arguments_match(actual, None)
    assert not arguments_match(actual, {"missing": None})


def test_required_calls_match_subset_count_and_attempts() -> None:
    calls = [
        ToolCall(tool_name="lookup", arguments={"id": "123", "extra": True}, outcome="retry"),
        ToolCall(tool_name="lookup", arguments={"id": "123"}, outcome="blocked"),
        ToolCall(tool_name="lookup", arguments={"id": "456"}),
    ]
    assert (
        evaluate([RequiredCall(tool_name="lookup", arguments={"id": "123"}, min_calls=2)], calls)
        == []
    )
    failure = evaluate(
        [RequiredCall(tool_name="lookup", arguments={"id": "123"}, min_calls=3)], calls
    )[0]
    assert failure.code == "missing_call"
    assert failure.observed == 2


def test_forbidden_reports_every_attempt_including_blocked() -> None:
    calls = [
        ToolCall(tool_name="refund", arguments={"eligible": False}, outcome="blocked"),
        ToolCall(tool_name="refund", arguments={"eligible": True}),
        ToolCall(tool_name="refund", arguments={"eligible": False}, outcome="error"),
    ]
    failures = evaluate([ForbiddenCall(tool_name="refund", arguments={"eligible": False})], calls)
    assert [failure.call_index for failure in failures] == [0, 2]
    assert failures[0].expected == 'no refund call matching {"eligible": false}'
    assert evaluate([ForbiddenCall(tool_name="refund")], calls)[0].expected == "no refund call"


def test_argument_checks_apply_to_every_matching_call_and_distinguish_missing_null() -> None:
    calls = [
        ToolCall(tool_name="update", arguments={"id": "123", "items": [{"value": None}]}),
        ToolCall(tool_name="update", arguments={"id": "123", "items": []}),
        ToolCall(tool_name="update", arguments={"id": "456", "items": [{"value": False}]}),
    ]
    failures = evaluate(
        [
            ArgumentCheck(
                tool_name="update",
                path="items.0.value",
                expected=None,
                match_arguments={"id": "123"},
            )
        ],
        calls,
    )
    assert len(failures) == 1
    assert failures[0].code == "missing_argument"
    assert failures[0].call_index == 1
    assert "<missing>" in failures[0].message


@pytest.mark.parametrize("operator", ["equals", "not_equals", "contains", "exists"])
def test_missing_arguments_always_fail_even_not_equals(operator: str) -> None:
    check = ArgumentCheck.model_validate(
        {"tool_name": "tool", "path": "missing", "operator": operator}
    )
    assert (
        evaluate([check], [ToolCall(tool_name="tool", arguments={})])[0].code == "missing_argument"
    )


def test_missing_matching_argument_calls_fail() -> None:
    failure = evaluate([ArgumentCheck(tool_name="lookup", path="id", expected="123")], [])[0]
    assert failure.code == "missing_call"


@pytest.mark.parametrize(
    "value,operator,expected",
    [
        (None, "exists", None),
        ("customer123", "contains", "123"),
        (["a", "b"], "contains", "b"),
        ({"id": "123"}, "contains", "id"),
        (False, "not_equals", 0),
    ],
)
def test_argument_operators(value: JsonValue, operator: str, expected: JsonValue) -> None:
    check = ArgumentCheck.model_validate(
        {"tool_name": "tool", "path": "value", "operator": operator, "expected": expected}
    )
    assert evaluate([check], [ToolCall(tool_name="tool", arguments={"value": value})]) == []


def test_list_contains_uses_strict_json_equality() -> None:
    failures = evaluate(
        [ArgumentCheck(tool_name="tool", path="flags", expected=0, operator="contains")],
        [ToolCall(tool_name="tool", arguments={"flags": [False]})],
    )
    assert failures[0].code == "incorrect_argument"


def test_before_requires_successful_earlier_model_turn_for_each_action() -> None:
    check = BeforeCheck(first="lookup", then="refund")
    assert evaluate([check], []) == []
    assert (
        evaluate(
            [check],
            [
                ToolCall(tool_name="lookup", arguments={}, step=0),
                ToolCall(tool_name="refund", arguments={}, step=1),
            ],
        )
        == []
    )
    for calls in [
        [ToolCall(tool_name="refund", arguments={}, step=0)],
        [
            ToolCall(tool_name="lookup", arguments={}, step=0),
            ToolCall(tool_name="refund", arguments={}, step=0),
        ],
        [
            ToolCall(tool_name="lookup", arguments={}, step=0, outcome="retry"),
            ToolCall(tool_name="refund", arguments={}, step=1),
        ],
        [
            ToolCall(tool_name="refund", arguments={}, step=0),
            ToolCall(tool_name="lookup", arguments={}, step=1),
        ],
    ]:
        assert evaluate([check], calls)[0].code == "incorrect_order"


def test_limits_and_output_checks() -> None:
    calls = [ToolCall(tool_name="create", arguments={}), ToolCall(tool_name="create", arguments={})]
    assert evaluate([MaxCalls(tool_name="create", maximum=1)], calls)[0].observed == 2
    assert evaluate([MaxCalls(tool_name="create", maximum=2)], calls) == []
    assert evaluate([OutputContains(text="ineligible")], [], "Order is ineligible") == []
    assert (
        evaluate([OutputContains(text="ineligible")], [], {"message": "ineligible"})[0].code
        == "incorrect_output"
    )


def test_before_does_not_accept_later_array_entry_with_forged_earlier_step() -> None:
    calls = [
        ToolCall(tool_name="refund", arguments={}, step=2),
        ToolCall(tool_name="lookup", arguments={}, step=1),
    ]
    assert (
        evaluate([BeforeCheck(first="lookup", then="refund")], calls)[0].code == "incorrect_order"
    )


def test_trace_diff_aligns_missing_calls_and_shows_argument_changes() -> None:
    baseline = [
        ToolCall(tool_name="lookup", arguments={"id": "123"}, step=0),
        ToolCall(tool_name="refund", arguments={"id": "123"}, step=1),
    ]
    current = [ToolCall(tool_name="refund", arguments={"id": "456"}, step=1)]
    differences = diff_traces(baseline, current)
    assert [item.kind for item in differences] == ["missing_call", "arguments_changed"]
    assert differences[1].call_index == 0
    assert "456" in differences[1].message


def test_trace_diff_ignores_ids_but_preserves_boolean_response_and_outcomes() -> None:
    baseline = [ToolCall(tool_name="tool", arguments={}, response=False, tool_call_id="old")]
    current = [
        ToolCall(tool_name="tool", arguments={}, response=0, tool_call_id="new", outcome="blocked")
    ]
    assert [item.kind for item in diff_traces(baseline, current)] == [
        "response_changed",
        "outcome_changed",
    ]
    unchanged = [ToolCall(tool_name="tool", arguments={}, response=False, tool_call_id="new")]
    assert diff_traces(baseline, unchanged) == []


def test_trace_diff_duplicate_additions_and_reorders_are_visible() -> None:
    lookup = ToolCall(tool_name="lookup", arguments={})
    refund = ToolCall(tool_name="refund", arguments={})
    assert [item.kind for item in diff_traces([lookup], [lookup, lookup])] == ["unexpected_call"]
    assert {item.kind for item in diff_traces([lookup, refund], [refund, lookup])} == {
        "unexpected_call",
        "missing_call",
    }


def test_mutated_unknown_checks_cannot_silently_pass_pure_api() -> None:
    scenario = Scenario(name="test", prompt="request")
    scenario.checks.append({"kind": "unknown"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported check"):
        evaluate_checks(scenario, [])
