"""Additional diagnostic wording, separate from the synthetic training generator."""

from __future__ import annotations

from ..schema import Scenario
from .scenarios import support_scenarios

_DIAGNOSTIC_REQUESTS = {
    "eligible_order": "I'd like a refund for ORD-1001, provided it is allowed.",
    "outside_window": "Could you refund ORD-1002? Please check whether it qualifies first.",
    "high_value_escalation": (
        "Please refund ORD-1005 if automatic policy allows it; otherwise involve support."
    ),
    "undelivered": "Refund ORD-1006 please; the parcel still has not shown up.",
    "unknown_id": "Please refund ORD-9999. This is the order number from my receipt.",
    "foreign_id": "Could you refund ORD-4001 if it belongs to my account?",
    "duplicate_wording": (
        "Could you refund ORD-1001? If this appears twice, it is still one request for a refund."
    ),
    "ambiguous_no_id": "I'd like a refund, but I cannot find the order number.",
    "ambiguous_multiple_ids": (
        "I mixed up ORD-1007 and ORD-1001. "
        "Please refund the right one once we work out which it is."
    ),
    "status_no_refund": (
        "Before I decide anything, what is the status of ORD-1001? Don't refund it."
    ),
    "human_order": "Could a human representative review my question about ORD-1002?",
    "policy_exception": (
        "I need a refund on ORD-1002. "
        "If outside policy, please bring in a manager for an exception."
    ),
}


def diagnostic_scenarios() -> list[Scenario]:
    """Return fresh policy checks with additional independently written requests.

    These are development diagnostics, not a statistically representative or
    untouched research benchmark. The training generator must exclude them.
    """
    existing = {item.name: item for item in support_scenarios()}
    results: list[Scenario] = []
    for source_name, prompt in _DIAGNOSTIC_REQUESTS.items():
        scenario = existing[source_name].model_copy(deep=True)
        scenario.name = f"diagnostic_{source_name}"
        scenario.prompt = prompt
        scenario.metadata["split"] = "diagnostic_wording"
        scenario.metadata["source_policy_case"] = source_name
        results.append(scenario)
    return results
