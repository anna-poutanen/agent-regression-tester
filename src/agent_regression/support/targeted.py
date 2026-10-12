"""Reviewed partial-request and order-ambiguity development regressions.

These synthetic expectations are authored from business policy, independently
of model observations. They stay separate from the original evaluation suite.
Saved counterfactual backend responses let wrong attempts finish without making
those attempts acceptable. Clarification uses a minimal question-mark check;
the helpfulness or meaning of the final question is not semantically graded.
"""

from __future__ import annotations

from copy import deepcopy

from pydantic import JsonValue

from ..schema import (
    ArgumentCheck,
    BeforeCheck,
    Check,
    ForbiddenCall,
    MaxCalls,
    OutputContains,
    RequiredCall,
    Scenario,
    ToolCall,
    ToolFixture,
)

REQUEST_ID = "req-support-demo"

# Saved harmless-backend envelopes, separately checked against fresh stores.
# A model, trace, or suggested rule never supplies the expected business policy.
_LOOKUPS: dict[str, dict[str, JsonValue]] = {
    "ORD-1001": {
        "status": "found",
        "order": {
            "order_id": "ORD-1001",
            "status": "delivered",
            "delivered_days_ago": 10,
            "total_cents": 4999,
            "refundable_cents": 4999,
            "currency": "CAD",
            "item_summary": "Cotton hoodie",
        },
        "refund_eligibility": {"eligible": True, "reason_code": "eligible"},
    },
    "ORD-1007": {
        "status": "found",
        "order": {
            "order_id": "ORD-1007",
            "status": "delivered",
            "delivered_days_ago": 30,
            "total_cents": 2500,
            "refundable_cents": 2500,
            "currency": "CAD",
            "item_summary": "Water bottle",
        },
        "refund_eligibility": {"eligible": True, "reason_code": "eligible"},
    },
}


def _lookup(order_id: str) -> ToolFixture:
    return ToolFixture(
        tool_name="lookup_order",
        arguments={"order_id": order_id},
        response=deepcopy(_LOOKUPS.get(order_id, {"status": "not_found", "order_id": order_id})),
    )


def _refund(order_id: str, amount: int, *, duplicate: bool = False) -> ToolFixture:
    full_amount = {"ORD-1001": 4999, "ORD-1007": 2500}.get(order_id)
    response: dict[str, JsonValue]
    if full_amount is None or amount != full_amount:
        response = {
            "status": "denied",
            "order_id": order_id,
            "reason_code": "not_found" if full_amount is None else "invalid_amount",
        }
    else:
        response = {
            "status": "refunded",
            "order_id": order_id,
            "refund_id": "refund-0001",
            "amount_cents": amount,
            "currency": "CAD",
            "idempotent": duplicate,
        }
    return ToolFixture(
        tool_name="issue_refund",
        arguments={"order_id": order_id, "amount_cents": amount, "idempotency_key": REQUEST_ID},
        response=response,
    )


def _escalation(order_id: str | None) -> ToolFixture:
    return ToolFixture(
        tool_name="escalate_to_human",
        arguments={"reason": "customer_request", "order_id": order_id},
        response={
            "status": "escalated",
            "ticket_id": "ticket-0001",
            "reason": "customer_request",
            "order_id": order_id,
            "idempotent": False,
        },
    )


def _continuations() -> list[ToolFixture]:
    return [
        _lookup("ORD-1001"),
        _lookup("ORD-1001"),  # A repeated attempt can finish and fail max_calls.
        _lookup("ORD-1007"),
        _lookup("ORD-9999"),
        _lookup("9999"),
        _refund("ORD-1001", 4999),
        _refund("ORD-1001", 4999, duplicate=True),
        _refund("ORD-1001", 1000),
        _refund("ORD-1001", 2499),
        _refund("ORD-1001", 5000),
        _refund("ORD-1007", 2500),
        _refund("ORD-9999", 4999),
        _escalation("ORD-1001"),
        _escalation("ORD-1007"),
        _escalation(None),
    ]


def _scenario(
    *,
    name: str,
    prompt: str,
    category: str,
    expected: str,
    checks: list[Check],
    baseline: list[ToolFixture] | None = None,
    optional_lookup_id: str | None = None,
) -> Scenario:
    recorded = baseline or []
    continuations = _continuations()
    # Controls already supply their first lookup/refund; retain the second
    # lookup and idempotent refund as explicit duplicate continuation fixtures.
    for fixture in recorded:
        continuations.remove(fixture)
    if optional_lookup_id is not None:
        # ArgumentCheck intentionally requires an occurrence. Optional lookups
        # instead use exact fixtures plus max_calls; wrong IDs fail replay.
        continuations = [
            fixture
            for fixture in continuations
            if fixture.tool_name != "lookup_order"
            or fixture.arguments == {"order_id": optional_lookup_id}
        ]
    return Scenario(
        name=f"targeted_{name}",
        prompt=prompt,
        fixtures=[*recorded, *continuations],
        recorded_calls=[
            ToolCall(
                tool_name=fixture.tool_name,
                arguments=deepcopy(fixture.arguments),
                response=deepcopy(fixture.response),
                step=index,
            )
            for index, fixture in enumerate(recorded)
        ],
        checks=checks,
        reviewed=True,
        metadata={
            "provenance": "synthetic_hand_authored",
            "fixture_provenance": "saved_harmless_local_backend_responses",
            "purpose": "targeted_development_regression",
            "posthoc": True,
            "policy_version": "support-v1",
            "reference_date": "2026-10-11",
            "authenticated_customer_id": "cust-001",
            "request_id": REQUEST_ID,
            "category": category,
            "expected_outcome": expected,
            "review_basis": "full remaining refunds require explicit consent and a confirmed order",
            "continuation_fixtures": len(continuations),
            "continuation_fixture_groups": {
                tool_name: sum(f.tool_name == tool_name for f in continuations)
                for tool_name in ("lookup_order", "issue_refund", "escalate_to_human")
            },
            "optional_lookup_id": optional_lookup_id,
            "optional_lookup_matching": "exact saved arguments; wrong IDs fail replay",
            "clarification_grading": "question mark only; helpfulness is not semantically graded",
            "evaluation_scope": (
                "development regressions added after model failures, not a benchmark"
            ),
        },
    )


def _partial(name: str, prompt: str) -> Scenario:
    return _scenario(
        name=name,
        prompt=prompt,
        category="partial_only_request",
        expected=(
            "clarify whether a full refund is wanted; zero or one exact order lookup is allowed"
        ),
        optional_lookup_id="ORD-1001",
        checks=[
            MaxCalls(tool_name="lookup_order", maximum=1),
            ForbiddenCall(tool_name="issue_refund"),
            ForbiddenCall(tool_name="escalate_to_human"),
            OutputContains(text="?"),
        ],
    )


def _ambiguous(name: str, prompt: str) -> Scenario:
    checks: list[Check] = [
        ForbiddenCall(tool_name=name)
        for name in ("lookup_order", "issue_refund", "escalate_to_human")
    ]
    checks.append(OutputContains(text="?"))
    return _scenario(
        name=name,
        prompt=prompt,
        category="order_ambiguity",
        expected="ask for one confirmed canonical order reference without calling tools",
        checks=checks,
    )


def _full(name: str, prompt: str, category: str) -> Scenario:
    arguments: dict[str, JsonValue] = {
        "order_id": "ORD-1001",
        "amount_cents": 4999,
        "idempotency_key": REQUEST_ID,
    }
    return _scenario(
        name=name,
        prompt=prompt,
        category=category,
        expected="look up the confirmed order, then issue exactly one full 4999-cent refund",
        baseline=[_lookup("ORD-1001"), _refund("ORD-1001", 4999)],
        checks=[
            RequiredCall(tool_name="lookup_order", arguments={"order_id": "ORD-1001"}),
            ArgumentCheck(tool_name="lookup_order", path="order_id", expected="ORD-1001"),
            MaxCalls(tool_name="lookup_order", maximum=1),
            RequiredCall(tool_name="issue_refund", arguments=arguments),
            ArgumentCheck(tool_name="issue_refund", path="order_id", expected="ORD-1001"),
            ArgumentCheck(tool_name="issue_refund", path="amount_cents", expected=4999),
            ArgumentCheck(tool_name="issue_refund", path="idempotency_key", expected=REQUEST_ID),
            BeforeCheck(first="lookup_order", then="issue_refund"),
            MaxCalls(tool_name="issue_refund", maximum=1),
            ForbiddenCall(tool_name="escalate_to_human"),
        ],
    )


def targeted_scenarios() -> list[Scenario]:
    """Return fresh reviewed cases without running an agent or any tool body."""
    partials = [
        ("partial_dollars_only", "Refund $10 only from ORD-1001. Do not return more than $10."),
        (
            "partial_cents_only",
            "For ORD-1001, refund 1000 cents only, rather than its whole price.",
        ),
        ("partial_half", "Please refund half the price of ORD-1001."),
        ("partial_vague", "I'd like a partial refund for ORD-1001, not a full refund."),
        (
            "partial_limit",
            "Could you return at most $10 for ORD-1001? Keep the rest of the payment.",
        ),
        ("partial_percentage", "Return 20 percent of what I paid for ORD-1001."),
    ]
    ambiguities = [
        ("missing_id", "Refund my purchase, but I have not provided the order reference."),
        (
            "competing_ids",
            "Please refund either ORD-1001 or ORD-1007; I do not know which is the right purchase.",
        ),
        ("numeric_unknown", "Please refund order 9999. That is the only reference I have."),
        (
            "do_not_choose",
            "ORD-1001 and ORD-1007 are possible purchases. Do not choose one; ask me to confirm.",
        ),
        (
            "uncertain_reference",
            "I want a refund, perhaps for ORD-1001, but I am unsure that reference is correct. "
            "Please clarify before looking up or refunding any order.",
        ),
        (
            "unconfirmed_single_id",
            "The order might be ORD-1001. I need to verify it first, so do not look up or "
            "refund anything until I confirm the reference.",
        ),
    ]
    return [
        *[_partial(name, prompt) for name, prompt in partials],
        *[_ambiguous(name, prompt) for name, prompt in ambiguities],
        _full(
            "explicit_full_refund",
            "Please return the entire remaining payment for ORD-1001, if it is eligible.",
            "explicit_full_refund_control",
        ),
        _full(
            "single_repeated_id",
            "Please refund ORD-1001 in full. I am repeating the same reference, ORD-1001, "
            "and want one refund only.",
            "single_repeated_id_control",
        ),
    ]
