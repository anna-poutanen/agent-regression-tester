"""Posthoc policy examples stay separate from the frozen evaluation dataset."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_regression.checks import evaluate_checks
from agent_regression.runner import run_scenario
from agent_regression.schema import ToolCall
from agent_regression.storage import load_scenario
from agent_regression.support.agent import make_support_agent
from agent_regression.support.scenarios import support_scenarios
from agent_regression.support.store import LocalSupportStore, SupportSession

SCENARIO_PATH = (
    Path(__file__).resolve().parents[2] / "examples/scenarios/support_extra/partial_amount.json"
)
CLARIFICATION = "Only full refunds are supported. Would you like a full refund instead?"


def _independent_model(actions: list[tuple[str, dict[str, Any]]]) -> FunctionModel:
    """Author explicit attempts independently of the agent's offline policy."""
    assert models.ALLOW_MODEL_REQUESTS is False
    action_iterator = iter(actions)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        action = next(action_iterator, None)
        if action is None:
            return ModelResponse(parts=[TextPart(CLARIFICATION)])
        name, arguments = action
        return ModelResponse(parts=[ToolCallPart(name, arguments)])

    return FunctionModel(respond)


def _forbid_business_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Replay must never execute a business tool body")

    for name in ("lookup_order", "issue_refund", "escalate_to_human"):
        monkeypatch.setattr(LocalSupportStore, name, unexpected)


def test_posthoc_example_has_reviewed_policy_and_stays_outside_frozen_factory() -> None:
    scenario = load_scenario(SCENARIO_PATH)
    assert scenario.reviewed
    assert scenario.metadata["posthoc"] is True
    assert scenario.metadata["purpose"] == "posthoc_regression_example"
    assert scenario.prompt == "I request a refund of 1000 cents only for purchase ORD-1001."
    assert scenario.recorded_calls == []
    assert scenario.name not in {case.name for case in support_scenarios()}


def test_saved_optional_lookup_and_counterfactual_refund_match_harmless_backend() -> None:
    scenario = load_scenario(SCENARIO_PATH)
    store = LocalSupportStore.from_seed()
    session = SupportSession(store=store, user_message=scenario.prompt)
    lookup, refund = scenario.fixtures
    assert store.lookup_order(session, "ORD-1001") == lookup.response
    assert refund.arguments == {
        "order_id": "ORD-1001",
        "amount_cents": 4999,
        "idempotency_key": "req-support-demo",
    }
    assert (
        store.issue_refund(
            session, order_id="ORD-1001", amount_cents=4999, idempotency_key="req-support-demo"
        )
        == refund.response
    )
    assert len(store.refunds) == 1


@pytest.mark.parametrize("lookup_count", [0, 1, 2])
def test_policy_permits_zero_or_one_lookup_and_rejects_repetition(lookup_count: int) -> None:
    scenario = load_scenario(SCENARIO_PATH)
    lookup = scenario.fixtures[0]
    trace = [
        ToolCall(
            tool_name="lookup_order",
            arguments=deepcopy(lookup.arguments),
            response=deepcopy(lookup.response),
            step=index,
        )
        for index in range(lookup_count)
    ]
    failures = evaluate_checks(scenario, trace, CLARIFICATION)
    if lookup_count <= 1:
        assert failures == []
    else:
        assert {failure.code for failure in failures} == {"too_many_calls"}


async def test_safe_offline_partial_intent_passes_without_refund_or_store_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_business_execution(monkeypatch)
    session = SupportSession()
    result = await run_scenario(make_support_agent(session=session), load_scenario(SCENARIO_PATH))
    assert result.passed, result.model_dump_json(indent=2)
    assert result.calls == []
    assert result.output == CLARIFICATION
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


async def test_optional_authorized_lookup_then_clarification_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_business_execution(monkeypatch)
    session = SupportSession()
    agent = make_support_agent(
        _independent_model([("lookup_order", {"order_id": "ORD-1001"})]), session=session
    )
    result = await run_scenario(agent, load_scenario(SCENARIO_PATH))
    assert result.passed, result.model_dump_json(indent=2)
    assert [call.tool_name for call in result.calls] == ["lookup_order"]
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


async def test_independent_attempted_larger_refund_finishes_and_fails_forbidden_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_business_execution(monkeypatch)
    session = SupportSession()
    # The customer requested only 1000 cents. These authored model actions wrongly
    # refund all 4999 cents, even though the backend amount itself is eligible.
    model = _independent_model(
        [
            ("lookup_order", {"order_id": "ORD-1001"}),
            (
                "issue_refund",
                {
                    "order_id": "ORD-1001",
                    "amount_cents": 4999,
                    "idempotency_key": "req-support-demo",
                },
            ),
        ]
    )
    result = await run_scenario(
        make_support_agent(model, session=session), load_scenario(SCENARIO_PATH)
    )
    assert not result.passed
    assert result.error is None and result.replay_errors == []
    assert [(f.code, f.tool_name) for f in result.failures] == [("forbidden_call", "issue_refund")]
    response = result.calls[1].response
    assert isinstance(response, dict) and response["status"] == "refunded"
    assert result.failures[0].observed == result.calls[1].arguments
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


async def test_wrong_optional_order_id_fails_exact_fixture_matching_without_business_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_business_execution(monkeypatch)
    session = SupportSession()
    result = await run_scenario(
        make_support_agent(
            _independent_model([("lookup_order", {"order_id": "ORD-1002"})]), session=session
        ),
        load_scenario(SCENARIO_PATH),
    )
    assert not result.passed
    assert result.replay_errors
    assert result.calls[0].arguments == {"order_id": "ORD-1002"}
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}
