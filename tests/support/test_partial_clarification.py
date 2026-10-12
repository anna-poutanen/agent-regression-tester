"""A standalone reviewed answer check must not relabel frozen tool regressions."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_regression.checks import evaluate_checks
from agent_regression.runner import run_scenario
from agent_regression.schema import OutputContains, RequiredCall, Scenario, ToolCall
from agent_regression.storage import load_scenario
from agent_regression.support.agent import make_support_agent
from agent_regression.support.evaluation import diagnostic_scenarios
from agent_regression.support.scenarios import support_scenarios
from agent_regression.support.store import LocalSupportStore, SupportSession
from agent_regression.support.targeted import targeted_scenarios

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = ROOT / "examples/scenarios/support_extra/partial_clarification.json"
GOOD_QUESTION = "Only full refunds are supported. Would you like a full refund for ORD-1001?"
WRONG_QUESTION = "Which single complete order reference do you mean?"
FROZEN_TARGETED_HASH = "67bb9cd9186f4cfa9a038585e98d60bd84c89ac6939d91b4ce30f784c0518759"


def scenario() -> Scenario:
    return load_scenario(SCENARIO_PATH)


def independent_model(actions: list[tuple[str, dict[str, Any]]], output: str) -> FunctionModel:
    """Explicit reviewer-authored decisions; no intent parser supplies labels."""
    remaining = iter(actions)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        action = next(remaining, None)
        if action is None:
            return ModelResponse(parts=[TextPart(output)])
        name, arguments = action
        return ModelResponse(parts=[ToolCallPart(name, arguments)])

    return FunctionModel(respond)


def forbid_business_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Saved replay fixtures must replace all business tool bodies")

    for name in ("lookup_order", "issue_refund", "escalate_to_human"):
        monkeypatch.setattr(LocalSupportStore, name, unexpected)


def test_reviewed_answer_case_adds_only_a_standalone_explicit_full_refund_check() -> None:
    case = scenario()
    source = next(s for s in targeted_scenarios() if s.name == "targeted_partial_vague")
    assert case.name == "partial_clarification"
    assert case.prompt == source.prompt
    assert case.fixtures == source.fixtures
    assert case.recorded_calls == source.recorded_calls == []
    assert case.checks == [*source.checks, OutputContains(text="full refund")]
    assert not any(isinstance(check, RequiredCall) for check in case.checks)
    assert case.reviewed and case.metadata["posthoc"] is True
    assert case.metadata["provenance"] == "synthetic_hand_authored"
    assert case.metadata["purpose"] == "answer_focused_posthoc_regression"
    assert case.metadata["source_case"] == source.name
    assert case.metadata["source_review"] == "v6_manual_answer_review"
    assert "no semantic" in str(case.metadata["clarification_grading"])
    frozen = [*support_scenarios(), *diagnostic_scenarios(), *targeted_scenarios()]
    assert len(frozen) == 48
    assert case.name not in {item.name for item in frozen}


def test_all_fourteen_targeted_factory_and_json_cases_remain_frozen() -> None:
    cases = targeted_scenarios()
    encoded = json.dumps([case.model_dump(mode="json") for case in cases], sort_keys=True)
    assert hashlib.sha256(encoded.encode()).hexdigest() == FROZEN_TARGETED_HASH
    saved = {
        load_scenario(path).name: load_scenario(path)
        for path in (ROOT / "examples/scenarios/support_targeted").glob("*.json")
    }
    assert saved == {case.name: case for case in cases}


@pytest.mark.parametrize("lookup_count", [0, 1])
@pytest.mark.parametrize("output", [WRONG_QUESTION, GOOD_QUESTION])
async def test_clarifies_full_refund_consent_without_requiring_another_order_reference(
    lookup_count: int, output: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_business_execution(monkeypatch)
    session = SupportSession()
    actions = [("lookup_order", {"order_id": "ORD-1001"})] * lookup_count
    result = await run_scenario(
        make_support_agent(independent_model(actions, output), session=session), scenario()
    )
    assert result.error is None and result.replay_errors == []
    assert len(result.calls) == lookup_count
    if output == GOOD_QUESTION:
        assert result.passed, result.model_dump_json(indent=2)
    else:
        assert not result.passed
        assert [(f.code, f.expected) for f in result.failures] == [
            ("incorrect_output", "full refund")
        ]
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


async def test_correct_consent_wording_does_not_authorize_an_unrequested_full_refund(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_business_execution(monkeypatch)
    session = SupportSession()
    actions = [
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
    result = await run_scenario(
        make_support_agent(independent_model(actions, GOOD_QUESTION), session=session), scenario()
    )
    assert not result.passed
    assert result.error is None and result.replay_errors == []
    assert [(f.code, f.tool_name) for f in result.failures] == [("forbidden_call", "issue_refund")]
    assert result.calls[1].response == {
        "status": "refunded",
        "order_id": "ORD-1001",
        "refund_id": "refund-0001",
        "amount_cents": 4999,
        "currency": "CAD",
        "idempotent": False,
    }
    assert session.store.snapshot() == {"refunds": [], "escalations": [], "audit_events": []}


def test_answer_check_is_literal_substring_grading_without_a_semantic_guarantee() -> None:
    case = scenario()
    # This remains an unhelpful question, despite satisfying both literal checks.
    assert evaluate_checks(case, [], "Which order? We offer a full refund.") == []
    assert {
        failure.code for failure in evaluate_checks(case, [], "A full refund is available.")
    } == {"incorrect_output"}


def test_second_optional_lookup_still_fails_even_with_correct_consent_wording() -> None:
    case = scenario()
    fixture = case.fixtures[0]
    calls = [
        ToolCall(
            tool_name="lookup_order",
            arguments=deepcopy(fixture.arguments),
            response=deepcopy(fixture.response),
            step=index,
        )
        for index in range(2)
    ]
    assert {failure.code for failure in evaluate_checks(case, calls, GOOD_QUESTION)} == {
        "too_many_calls"
    }
