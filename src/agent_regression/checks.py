"""Pure checks on attempted tool calls and deterministic baseline comparisons."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from pydantic import JsonValue

from .schema import (
    ArgumentCheck,
    BeforeCheck,
    CheckFailure,
    ForbiddenCall,
    MaxCalls,
    OutputContains,
    RequiredCall,
    Scenario,
    ToolCall,
    TraceDifference,
)


def json_equal(left: JsonValue, right: JsonValue) -> bool:
    """Compare JSON values recursively while keeping booleans distinct from numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            json_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def arguments_match(actual: dict[str, JsonValue], expected: dict[str, JsonValue] | None) -> bool:
    """Object expectations are recursive subsets; lists and scalar values are exact."""
    if expected is None:
        return True
    for key, value in expected.items():
        if key not in actual:
            return False
        observed = actual[key]
        if isinstance(value, dict):
            if not isinstance(observed, dict) or not arguments_match(observed, value):
                return False
        elif not json_equal(observed, value):
            return False
    return True


_MISSING = object()


def _read_path(arguments: dict[str, JsonValue], path: str) -> Any:
    value: Any = arguments
    for component in path.split("."):
        if isinstance(value, dict):
            value = value.get(component, _MISSING)
        elif isinstance(value, list) and component.isascii() and component.isdecimal():
            index = int(component)
            value = value[index] if index < len(value) else _MISSING
        else:
            return _MISSING
        if value is _MISSING:
            return _MISSING
    return value


def _contains(observed: JsonValue, expected: JsonValue) -> bool:
    if isinstance(observed, str) and isinstance(expected, str):
        return expected in observed
    if isinstance(observed, list):
        return any(json_equal(item, expected) for item in observed)
    if isinstance(observed, dict) and isinstance(expected, str):
        return expected in observed
    return False


def _display(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _argument_failures(check: ArgumentCheck, calls: list[ToolCall]) -> list[CheckFailure]:
    matching = [
        (index, call)
        for index, call in enumerate(calls)
        if call.tool_name == check.tool_name
        and arguments_match(call.arguments, check.match_arguments)
    ]
    if not matching:
        return [
            CheckFailure(
                code="missing_call",
                message=f"No matching {check.tool_name} call to check argument {check.path!r}",
                tool_name=check.tool_name,
                expected=check.match_arguments,
            )
        ]
    failures: list[CheckFailure] = []
    for index, call in matching:
        observed = _read_path(call.arguments, check.path)
        if observed is _MISSING:
            passed = False
        elif check.operator == "exists":
            passed = True
        elif check.operator == "equals":
            passed = json_equal(observed, check.expected)
        elif check.operator == "not_equals":
            passed = not json_equal(observed, check.expected)
        else:
            passed = _contains(observed, check.expected)
        if not passed:
            actual = "<missing>" if observed is _MISSING else _display(observed)
            expectation = (
                "exist"
                if check.operator == "exists"
                else f"{check.operator} {_display(check.expected)}"
            )
            failures.append(
                CheckFailure(
                    code="missing_argument" if observed is _MISSING else "incorrect_argument",
                    message=(
                        f"{check.tool_name} argument {check.path!r} must {expectation}; "
                        f"observed {actual}"
                    ),
                    tool_name=check.tool_name,
                    expected=check.expected,
                    observed=None if observed is _MISSING else observed,
                    call_index=index,
                )
            )
    return failures


def evaluate_checks(
    scenario: Scenario, calls: list[ToolCall], output: JsonValue = None
) -> list[CheckFailure]:
    """Evaluate reviewed policy checks, retaining every violated call occurrence.

    All checks count attempts, including retries and blocked calls. A ``before``
    check additionally requires the prerequisite to return in an earlier model
    turn, so simultaneous calls cannot establish causal ordering. Argument
    checks require at least one matching call, then inspect every occurrence.
    """
    failures: list[CheckFailure] = []
    for check in scenario.checks:
        if isinstance(check, RequiredCall):
            matching = [
                call
                for call in calls
                if call.tool_name == check.tool_name
                and arguments_match(call.arguments, check.arguments)
            ]
            if len(matching) < check.min_calls:
                failures.append(
                    CheckFailure(
                        code="missing_call",
                        message=(
                            f"Expected at least {check.min_calls} matching {check.tool_name} "
                            f"call(s); observed {len(matching)}"
                        ),
                        tool_name=check.tool_name,
                        expected={"arguments": check.arguments, "min_calls": check.min_calls},
                        observed=len(matching),
                    )
                )
        elif isinstance(check, ForbiddenCall):
            for index, call in enumerate(calls):
                if call.tool_name == check.tool_name and arguments_match(
                    call.arguments, check.arguments
                ):
                    failures.append(
                        CheckFailure(
                            code="forbidden_call",
                            message=(
                                f"Forbidden {check.tool_name} call with "
                                f"arguments {_display(call.arguments)}"
                            ),
                            tool_name=check.tool_name,
                            expected=(
                                f"no {check.tool_name} call"
                                if check.arguments is None
                                else (
                                    f"no {check.tool_name} call matching "
                                    f"{_display(check.arguments)}"
                                )
                            ),
                            observed=call.arguments,
                            call_index=index,
                        )
                    )
        elif isinstance(check, ArgumentCheck):
            failures.extend(_argument_failures(check, calls))
        elif isinstance(check, BeforeCheck):
            first_calls = [
                (index, call)
                for index, call in enumerate(calls)
                if call.tool_name == check.first and call.outcome == "return"
            ]
            for index, call in enumerate(calls):
                if call.tool_name == check.then and not any(
                    prior_index < index and prior.step < call.step
                    for prior_index, prior in first_calls
                ):
                    failures.append(
                        CheckFailure(
                            code="incorrect_order",
                            message=(
                                f"{check.first} must return in an earlier model turn before "
                                f"{check.then}; observed {check.then} at step {call.step}"
                            ),
                            tool_name=check.then,
                            expected=check.first,
                            observed=call.arguments,
                            call_index=index,
                        )
                    )
        elif isinstance(check, MaxCalls):
            count = sum(call.tool_name == check.tool_name for call in calls)
            if count > check.maximum:
                failures.append(
                    CheckFailure(
                        code="too_many_calls",
                        message=(
                            f"Expected at most {check.maximum} {check.tool_name} "
                            f"call(s); observed {count}"
                        ),
                        tool_name=check.tool_name,
                        expected=check.maximum,
                        observed=count,
                    )
                )
        elif isinstance(check, OutputContains):
            if not isinstance(output, str) or check.text not in output:
                failures.append(
                    CheckFailure(
                        code="incorrect_output",
                        message=f"Output must contain {check.text!r}",
                        expected=check.text,
                        observed=output,
                    )
                )
        else:
            # Defend this independently usable API against accidental mutation
            # of Scenario.checks after construction; unknown checks must not pass.
            raise ValueError(f"Unsupported check: {check!r}")
    return failures


def diff_traces(baseline: list[ToolCall], calls: list[ToolCall]) -> list[TraceDifference]:
    """Align tool names to explain additions, omissions, and changed call data."""
    differences: list[TraceDifference] = []
    alignment = SequenceMatcher(
        a=[call.tool_name for call in baseline],
        b=[call.tool_name for call in calls],
        autojunk=False,
    )
    for operation, old_start, old_end, new_start, new_end in alignment.get_opcodes():
        if operation == "equal":
            for old_index, new_index in zip(
                range(old_start, old_end), range(new_start, new_end), strict=True
            ):
                previous, current = baseline[old_index], calls[new_index]
                for field in ("arguments", "response", "outcome", "step"):
                    before, after = getattr(previous, field), getattr(current, field)
                    if not json_equal(before, after):
                        differences.append(
                            TraceDifference(
                                kind=f"{field}_changed",
                                message=(
                                    f"{current.tool_name} {field} changed from "
                                    f"{_display(before)} to {_display(after)}"
                                ),
                                tool_name=current.tool_name,
                                call_index=new_index,
                            )
                        )
        else:
            for previous in baseline[old_start:old_end]:
                differences.append(
                    TraceDifference(
                        kind="missing_call",
                        message=(
                            f"Recorded {previous.tool_name}({_display(previous.arguments)}) "
                            "was not observed"
                        ),
                        tool_name=previous.tool_name,
                    )
                )
            for index in range(new_start, new_end):
                current = calls[index]
                differences.append(
                    TraceDifference(
                        kind="unexpected_call",
                        message=(
                            f"New {current.tool_name}({_display(current.arguments)}) "
                            "was not in the recording"
                        ),
                        tool_name=current.tool_name,
                        call_index=index,
                    )
                )
    return differences
