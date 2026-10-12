"""Limited test-double language coverage and real-model routing separation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_regression.runner import run_scenario
from agent_regression.schema import Scenario
from agent_regression.support import offline
from agent_regression.support.agent import make_support_agent
from agent_regression.support.prompt import SAFE_INSTRUCTIONS
from agent_regression.support.store import SupportSession
from agent_regression.support.targeted import targeted_scenarios


def _calls(result: Any) -> list[ToolCallPart]:
    return [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]


@pytest.mark.parametrize(
    "prompt",
    [
        "Please return the entire remaining payment for ORD-1001, if it is eligible.",
        "Could you return my full remaining balance for ORD-1001?",
        "Return all of my remaining payment for ORD-1001.",
    ],
)
def test_explicit_whole_payment_synonyms_issue_one_actual_local_full_refund(prompt: str) -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(prompt)
    assert [part.tool_name for part in _calls(result)] == ["lookup_order", "issue_refund"]
    assert _calls(result)[1].args_as_dict() == {
        "order_id": "ORD-1001",
        "amount_cents": 4999,
        "idempotency_key": "req-support-demo",
    }
    assert len(session.store.refunds) == 1
    assert session.store.escalations == []


@pytest.mark.parametrize(
    "prompt",
    [
        "Could you return at most $10 for ORD-1001? Keep the rest of the payment.",
        "Return 20 percent of what I paid for ORD-1001.",
        "Refund 20% of ORD-1001, please.",
        "Please refund one quarter of ORD-1001.",
        "For ORD-1001, refund 1000 cents only, rather than its whole price.",
    ],
)
def test_percentage_fraction_and_amount_caps_need_full_refund_consent(prompt: str) -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(prompt)
    assert _calls(result) == []
    assert "full refunds" in result.output and "?" in result.output
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


@pytest.mark.parametrize(
    "prompt",
    [
        "I want a refund, perhaps for ORD-1001, but I am unsure that reference is correct. "
        "Please clarify before looking up or refunding any order.",
        "The order might be ORD-1001. I need to verify it first, so do not look up or "
        "refund anything until I confirm the reference.",
        "Please refund maybe ORD-1001; ask me which reference is correct first.",
        "I am not sure the order ID is correct. Please refund ORD-1001 only after confirmation.",
    ],
)
def test_expressly_unconfirmed_single_reference_does_not_trigger_lookup(prompt: str) -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(prompt)
    assert _calls(result) == []
    assert "order ID" in result.output and "?" in result.output
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


@pytest.mark.parametrize(
    "prompt",
    [
        "Don't look up ORD-1001 until I confirm. Please refund it once I confirm.",
        "Please refund ORD-1001. Don't look up anything until I confirm.",
        "Do not look up ORD-1001 until I verify it. Please ask a human about that order later.",
        "Please refund ORD-1001 and involve a human. Do not refund until I confirm.",
    ],
)
def test_explicit_wait_without_a_trailing_reference_noun_blocks_all_bound_actions(
    prompt: str,
) -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(prompt)
    assert _calls(result) == []
    assert "confirmed" in result.output and "?" in result.output
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


@pytest.mark.parametrize(
    "prompt",
    [
        "Do not contact a human; just show ORD-1001 status.",
        "Don't send me to a representative. Check ORD-1001 status only.",
        "I do not need a manager. Please look up ORD-1001 status.",
        "I am not asking for a human; just check ORD-1001 status.",
        "No supervisor please. What is the status of ORD-1001?",
    ],
)
def test_negated_human_help_remains_information_only(prompt: str) -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(prompt)
    assert [part.tool_name for part in _calls(result)] == ["lookup_order"]
    assert session.store.refunds == [] and session.store.escalations == []


def test_negated_manager_mention_does_not_turn_an_expired_refund_into_an_exception() -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(
        "Refund ORD-1002 if it is eligible. Do not ask a manager."
    )
    assert [part.tool_name for part in _calls(result)] == ["lookup_order"]
    assert "cannot issue a refund" in result.output
    assert session.store.refunds == [] and session.store.escalations == []


@pytest.mark.parametrize(
    "prompt",
    [
        "Refund ORD-1002. Do not ask a manager for an exception.",
        "Please refund ORD-1002, but no exception and no human.",
        "Refund ORD-1002 if eligible, without an exception.",
        "Please refund ORD-1002. Don't request an exception.",
        "Please refund ORD-1002. I am not asking for an exception.",
    ],
)
def test_negated_exception_never_creates_a_policy_ticket(prompt: str) -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(prompt)
    assert [part.tool_name for part in _calls(result)] == ["lookup_order"]
    assert "cannot issue a refund" in result.output
    assert session.store.refunds == [] and session.store.escalations == []


def test_positive_standalone_exception_request_still_creates_one_policy_ticket() -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(
        "Please refund ORD-1002. I request an exception to its refund window."
    )
    assert [part.tool_name for part in _calls(result)] == ["lookup_order", "escalate_to_human"]
    assert _calls(result)[1].args_as_dict() == {
        "reason": "policy_exception",
        "order_id": "ORD-1002",
    }
    assert session.store.refunds == []
    assert len(session.store.escalations) == 1


def test_refusing_an_exception_does_not_block_explicit_ordinary_human_help() -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(
        "I am not asking for an exception, but I need a human about ORD-1002."
    )
    assert [part.tool_name for part in _calls(result)] == ["lookup_order", "escalate_to_human"]
    assert _calls(result)[1].args_as_dict() == {
        "reason": "customer_request",
        "order_id": "ORD-1002",
    }
    assert session.store.refunds == []
    assert len(session.store.escalations) == 1


def test_general_human_help_without_an_order_is_still_allowed_while_waiting_for_lookup() -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(
        "Please get me a human. Do not look up an order until I confirm its reference."
    )
    assert [part.tool_name for part in _calls(result)] == ["escalate_to_human"]
    assert _calls(result)[0].args_as_dict() == {"reason": "customer_request", "order_id": None}
    assert session.store.refunds == []
    assert len(session.store.escalations) == 1


@pytest.mark.parametrize(
    "prompt",
    [
        "Please refund ORD-1001 if it is eligible.",
        "I am unsure whether ORD-1001 is eligible. Please refund it if it is.",
        "Please return the entire remaining payment for ORD-1001 if it qualifies.",
    ],
)
def test_policy_uncertainty_does_not_make_a_chosen_order_ambiguous(prompt: str) -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(prompt)
    assert [part.tool_name for part in _calls(result)] == ["lookup_order", "issue_refund"]
    assert len(session.store.refunds) == 1


def test_negated_whole_payment_request_remains_information_only() -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(
        "Do not return the entire remaining payment for ORD-1001. Just check its status."
    )
    assert [part.tool_name for part in _calls(result)] == ["lookup_order"]
    assert session.store.refunds == [] and session.store.escalations == []


@pytest.mark.parametrize(
    "prompt",
    [
        "Do not return $10; just check ORD-1001 status.",
        "Please do not refund 1000 cents for ORD-1001; check its status only.",
        "Don't return 20 percent of ORD-1001. Please look up the status only.",
    ],
)
def test_negated_limited_amount_does_not_ask_for_full_refund_consent(prompt: str) -> None:
    session = SupportSession()
    result = make_support_agent(session=session).run_sync(prompt)
    assert [part.tool_name for part in _calls(result)] == ["lookup_order"]
    assert "full refund" not in result.output
    assert session.store.refunds == [] and session.store.escalations == []


@pytest.mark.parametrize("scenario", targeted_scenarios(), ids=lambda scenario: scenario.name)
async def test_targeted_policy_cases_pass_the_labeled_offline_double_with_mocked_tools(
    scenario: Scenario,
) -> None:
    session = SupportSession()
    result = await run_scenario(make_support_agent(session=session), scenario)
    assert result.passed, result.model_dump_json(indent=2)
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


@pytest.mark.parametrize(
    "prompt",
    [
        "Refund $10 only from ORD-1001.",
        "The order might be ORD-1001. Do not look up anything until I confirm the reference.",
        "Please refund my purchase; I did not provide its reference.",
        "Please refund ORD-1001. Don't look up anything until I confirm.",
        "Do not contact a human; just show ORD-1001 status.",
    ],
)
def test_configured_model_receives_authorization_rules_and_original_request_without_parser(
    prompt: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def parser_must_not_run(value: str) -> None:
        raise AssertionError("Configured models must never use offline intent parsing")

    monkeypatch.setattr(offline, "_intent", parser_must_not_run)
    observed: list[tuple[str, str | None]] = []

    def provided_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        original = next(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        assert isinstance(original, str)
        observed.append((original, info.instructions))
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("Caller-provided model response.")])
        # This deliberately inappropriate lookup remains visible. The agent must
        # not conceal a configured model's behavior using the offline script.
        return ModelResponse(parts=[ToolCallPart("lookup_order", {"order_id": "ORD-1001"})])

    session = SupportSession()
    configured = FunctionModel(provided_model, model_name="independent-routing-contract")
    agent = make_support_agent(configured, session=session)
    result = agent.run_sync(prompt)
    assert agent.model is configured
    assert [part.tool_name for part in _calls(result)] == ["lookup_order"]
    assert all(original == prompt for original, _ in observed)
    assert all(
        instructions is not None and instructions.startswith(SAFE_INSTRUCTIONS)
        for _, instructions in observed
    )
    assert session.user_message == prompt
    assert len(session.store.audit_events) == 1
    assert session.store.refunds == [] and session.store.escalations == []
