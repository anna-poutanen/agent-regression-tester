"""Draft observed-call checks without inferring or approving business policy."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

from pydantic import JsonValue

from .schema import Check, RequiredCall, Scenario


def _identity(arguments: dict[str, JsonValue]) -> str:
    # JSON encoding distinguishes booleans from numbers and ignores key order.
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def suggest_checks(scenario: Scenario) -> list[Check]:
    """Suggest required calls seen in the trace, including repeated attempts.

    Suggestions are observations, not policies: a recorded forbidden action can
    appear here. The caller must review them, and this function never mutates or
    marks the scenario reviewed. No ordering, output, or forbidden-call policy
    is inferred from a single recording.
    """
    counts = Counter(
        (call.tool_name, _identity(call.arguments)) for call in scenario.recorded_calls
    )
    suggestions: list[Check] = []
    seen: set[tuple[str, str]] = set()
    for call in scenario.recorded_calls:
        key = call.tool_name, _identity(call.arguments)
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(
            RequiredCall(
                tool_name=call.tool_name,
                arguments=deepcopy(call.arguments),
                min_calls=counts[key],
            )
        )
    return suggestions
