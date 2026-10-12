"""A deterministic FunctionModel test double for local support integration tests.

This module is a small, conservative intent parser and policy script. It does not
train or call a language model, and is not evidence of real-model performance.
It consumes user requests and tool results rather than scenario names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .prompt import MUTATIONS


@dataclass(frozen=True)
class _Intent:
    order_ids: tuple[str, ...]
    refund: bool
    human: bool
    exception: bool
    partial: bool


def _intent(prompt: str) -> _Intent:
    # Exclude simple quoted data and common injection suffixes from intent. This
    # is test-double parsing, not a general-purpose prompt-injection detector.
    # Quoting an identifier does not turn it into document instructions. Single
    # quotes embedded in words (don't, it's) are ordinary apostrophes.
    text = re.sub(r"(['\"`])\s*(ORD-\d{4})\s*\1", r"\2", prompt, flags=re.I)
    text = re.sub(r'"[^"\n]*"|(?<!\w)\'[^\'\n]*\'(?!\w)|`[^`]*`', " ", text)
    text = re.split(r"\bignore (?:all )?(?:previous|prior) instructions\b", text, flags=re.I)[0]
    ids = tuple(dict.fromkeys(match.upper() for match in re.findall(r"\bORD-\d{4}\b", text, re.I)))
    negative = bool(
        re.search(
            r"\b(?:do not|don't|never|without)\s+"
            r"(?:(?:want|need|request|issue|process|receive)\s+"
            r"(?:(?:you\s+)?to\s+)?)?(?:a\s+|my\s+|the\s+)?(?:refund|money\s+back)\b|"
            r"\b(?:do not|don't|never)\s+(?:give|send|get)\s+"
            r"(?:me\s+)?(?:my\s+)?money\s+back\b|"
            r"\b(?:not|without)\s+(?:asking|requesting|issuing|processing)\s+"
            r"(?:for\s+)?(?:a\s+)?refund\b|\bno\s+refund\b",
            text,
            re.I,
        )
    )
    refund = not negative and bool(
        re.search(
            r"\b(?:refund\s+(?:my\s+)?(?:order\b|ORD-|it\b)|"
            r"(?:want|like|request|need)\s+(?:a\s+)?refund\b|"
            r"(?:can|could|would)\s+you\s+(?:please\s+)?refund\b|"
            r"(?:give|send|get)\s+(?:me\s+)?(?:my\s+)?money\s+back\b)",
            text,
            re.I,
        )
    )
    human = bool(re.search(r"\b(?:human|representative|manager|supervisor)\b", text, re.I))
    exception = bool(re.search(r"\b(?:exception|manager|supervisor)\b", text, re.I))
    partial = bool(
        re.search(
            r"\bpartial\s+refund\b|\brefund\b.*\b(?:half|part|portion)\b|"
            r"\brefund\b.*(?:\$\s*\d|\b\d+(?:\.\d+)?\s+(?:dollars?|cents?)\b)",
            text,
            re.I,
        )
    )
    return _Intent(ids, refund, human, exception, partial)


def _current_messages(messages: list[ModelMessage]) -> tuple[str, list[ModelMessage]]:
    # Message history can contain previous turns. Work only on the latest user
    # request and its subsequent tool calls, without confusing earlier actions.
    prompt = ""
    start = 0
    for index, message in enumerate(messages):
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                prompt = part.content if isinstance(part.content, str) else str(part.content)
                start = index
    return prompt, messages[start:]


def _returns(messages: list[ModelMessage], name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == name
    ]


def _text(value: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(value)])


def _call(name: str, arguments: dict[str, Any]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(name, arguments)])


def _result_content(part: ToolReturnPart) -> dict[str, Any]:
    return part.content if isinstance(part.content, dict) else {}


def make_offline_model(
    *, mutation: str = "safe", request_id: str = "req-support-demo"
) -> FunctionModel:
    """Create a clearly labeled offline script; no credentials or training needed.

    ``request_id`` is a fallback when used outside make_support_agent. The factory
    supplies a trusted request ID in dynamic instructions for per-run deps.
    """
    if mutation not in MUTATIONS:
        raise ValueError(f"Unknown support mutation: {mutation!r}. Choose from {MUTATIONS}.")

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt, current = _current_messages(messages)
        intent = _intent(prompt)
        lookups = _returns(current, "lookup_order")
        refunds = _returns(current, "issue_refund")
        escalations = _returns(current, "escalate_to_human")
        key_match = re.search(r"^Trusted request ID: ([^\n]+)$", info.instructions or "", re.M)
        key = key_match.group(1) if key_match else request_id

        if intent.partial and not intent.human:
            return _text("Only full refunds are supported. Would you like a full refund instead?")

        if escalations:
            result = _result_content(escalations[-1])
            if result.get("status") == "escalated":
                return _text(
                    f"I referred your request to a human. Ticket: {result.get('ticket_id')}."
                )
            return _text("I could not create a human support ticket for that order.")

        if refunds:
            result = _result_content(refunds[-1])
            if mutation == "duplicate_refund" and len(refunds) == 1:
                first_arguments = next(
                    (
                        part.args_as_dict()
                        for message in current
                        for part in message.parts
                        if isinstance(part, ToolCallPart) and part.tool_name == "issue_refund"
                    ),
                    {},
                )
                return _call("issue_refund", first_arguments)
            if result.get("status") == "refunded":
                return _text(
                    f"Your local refund for {result.get('order_id')} was issued: "
                    f"{result.get('amount_cents')} CAD cents."
                )
            return _text(f"The refund was denied: {result.get('reason_code', 'unknown_reason')}.")

        order_id = intent.order_ids[0] if len(intent.order_ids) == 1 else None
        if order_id is None:
            if intent.human:
                if mutation == "ignore_escalation":
                    return _text("A human support representative would normally help you.")
                return _call("escalate_to_human", {"reason": "customer_request", "order_id": None})
            if mutation == "guess_ambiguous":
                order_id = intent.order_ids[0] if intent.order_ids else "ORD-1001"
            else:
                return _text("Which order ID would you like help with? Please specify one order.")

        if not lookups:
            refund_arguments = {"order_id": order_id, "amount_cents": 4999, "idempotency_key": key}
            if intent.refund and mutation == "skip_lookup":
                return _call("issue_refund", refund_arguments)
            if intent.refund and mutation == "parallel_refund":
                return ModelResponse(
                    parts=[
                        ToolCallPart("lookup_order", {"order_id": order_id}),
                        ToolCallPart("issue_refund", refund_arguments),
                    ]
                )
            lookup_id = "ORD-9999" if mutation == "wrong_order_id" else order_id
            return _call("lookup_order", {"order_id": lookup_id})

        result = _result_content(lookups[-1])
        if result.get("status") != "found":
            return _text("I could not find that order. Please check the order ID and try again.")
        order = result.get("order", {})
        eligibility = result.get("refund_eligibility", {})
        if not isinstance(order, dict) or not isinstance(eligibility, dict):
            return _text("The order response was incomplete, so I cannot process an action.")
        reason = eligibility.get("reason_code")
        eligible = eligibility.get("eligible") is True

        if intent.human and (not intent.refund or intent.partial):
            if mutation == "ignore_escalation":
                return _text("A human support representative would normally help with that order.")
            return _call("escalate_to_human", {"reason": "customer_request", "order_id": order_id})
        if not intent.refund:
            return _text(
                f"Order {order_id} is {order.get('status')}. Refund eligibility: {reason}."
            )
        if eligible or mutation == "refund_ineligible":
            amount = order.get("refundable_cents")
            if not isinstance(amount, int) or isinstance(amount, bool):
                return _text("The refund amount was invalid, so I cannot issue a refund.")
            if mutation == "over_refund":
                amount += 1
            return _call(
                "issue_refund",
                {"order_id": order_id, "amount_cents": amount, "idempotency_key": key},
            )

        escalation_reason = None
        if reason == "requires_approval" or intent.exception:
            escalation_reason = "policy_exception"
        elif reason == "not_delivered":
            escalation_reason = "delivery_issue"
        elif intent.human:
            escalation_reason = "customer_request"
        if escalation_reason and mutation != "ignore_escalation":
            return _call("escalate_to_human", {"reason": escalation_reason, "order_id": order_id})
        explanations = {
            "outside_window": "It is outside the 30-day refund window.",
            "final_sale": "It was marked final sale.",
            "already_refunded": "It has already been refunded.",
            "not_delivered": "It has not been delivered and needs delivery support.",
            "requires_approval": "It requires human approval because of its value.",
        }
        return _text(
            "I cannot issue a refund for this order. "
            + explanations.get(str(reason), "It does not meet the refund policy.")
        )

    return FunctionModel(model, model_name=f"offline-support-test-double:{mutation}")
