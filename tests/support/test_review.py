"""Independent review regressions for truthful actions and exposed tool contracts."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_regression.runner import run_scenario
from agent_regression.support.agent import make_support_agent
from agent_regression.support.scenarios import support_scenarios
from agent_regression.support.store import SupportSession


async def test_false_refund_claim_cannot_satisfy_required_refund_tool_call():
    def falsely_claim_refund(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart("Your refund has been issued successfully.")])
        return ModelResponse(parts=[ToolCallPart("lookup_order", {"order_id": "ORD-1001"})])

    session = SupportSession()
    scenario = next(s for s in support_scenarios() if s.name == "eligible_order")
    result = await run_scenario(
        make_support_agent(FunctionModel(falsely_claim_refund), session=session), scenario
    )
    assert result.output == "Your refund has been issued successfully."
    assert not result.passed
    assert any(f.code == "missing_call" and f.tool_name == "issue_refund" for f in result.failures)
    assert [call.tool_name for call in result.calls] == ["lookup_order"]
    assert session.store.refunds == []


async def test_escalation_schema_requires_nullable_order_id_for_exact_replay_contract():
    observed: dict[str, Any] = {}

    def inspecting_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        observed.update(
            {
                definition.name: definition.parameters_json_schema
                for definition in info.function_tools
            }
        )
        return ModelResponse(parts=[TextPart("Hello.")])

    await make_support_agent(FunctionModel(inspecting_model)).run("Hello.")
    schema = observed["escalate_to_human"]
    assert set(schema["required"]) == {"reason", "order_id"}
    assert {option["type"] for option in schema["properties"]["order_id"]["anyOf"]} == {
        "string",
        "null",
    }


async def test_model_can_correct_omitted_escalation_argument_before_execution_and_replay():
    def initially_omit_order(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart("I connected you to a human.")])
        if any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "escalate_to_human", {"reason": "customer_request", "order_id": None}
                    )
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart("escalate_to_human", {"reason": "customer_request"})]
        )

    session = SupportSession()
    direct = await make_support_agent(FunctionModel(initially_omit_order), session=session).run(
        "Please connect me with a human support representative."
    )
    assert "human" in direct.output
    assert len(session.store.escalations) == 1
    assert session.store.escalations[0]["order_id"] is None
    # Tester intentionally preserves all attempted calls, including validation
    # retries. Even after a correction, it reports the malformed initial action.
    scenario = next(s for s in support_scenarios() if s.name == "explicit_human")
    replay = await run_scenario(make_support_agent(FunctionModel(initially_omit_order)), scenario)
    assert not replay.passed
    assert any(f.code == "missing_argument" for f in replay.failures)
    assert replay.calls[-1].arguments == {"reason": "customer_request", "order_id": None}


async def test_string_null_escalation_is_denied_then_corrected_but_both_attempts_are_graded():
    def initially_use_string_null(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if any(
            isinstance(part.content, dict) and part.content.get("status") == "escalated"
            for part in returns
        ):
            return ModelResponse(parts=[TextPart("Human support ticket ticket-0001 was created.")])
        # The actual store denies the valid string "null" as an unknown order;
        # replay reports a fixture miss. Either response lets the model correct it.
        correction = bool(returns) or any(
            isinstance(part, RetryPromptPart) for message in messages for part in message.parts
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "escalate_to_human",
                    {"reason": "customer_request", "order_id": None if correction else "null"},
                )
            ]
        )

    scenario = next(s for s in support_scenarios() if s.name == "explicit_human")
    direct_session = SupportSession()
    direct = await make_support_agent(
        FunctionModel(initially_use_string_null), session=direct_session
    ).run(scenario.prompt)
    returns = [
        part.content
        for message in direct.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert returns[0] == {"status": "denied", "order_id": "null", "reason_code": "not_found"}
    assert isinstance(returns[1], dict) and returns[1]["status"] == "escalated"
    assert len(direct_session.store.escalations) == 1
    assert direct_session.store.escalations[0]["order_id"] is None
    assert direct_session.store.refunds == []
    assert [event["action"] for event in direct_session.store.audit_events] == ["escalate_to_human"]

    replay_session = SupportSession()
    before = replay_session.store.snapshot()
    replay = await run_scenario(
        make_support_agent(FunctionModel(initially_use_string_null), session=replay_session),
        scenario,
    )
    assert not replay.passed
    assert [call.arguments for call in replay.calls] == [
        {"reason": "customer_request", "order_id": "null"},
        {"reason": "customer_request", "order_id": None},
    ]
    assert [call.outcome for call in replay.calls] == ["blocked", "return"]
    assert {"incorrect_argument", "too_many_calls"} <= {f.code for f in replay.failures}
    assert any("No fixture matches arguments" in error for error in replay.replay_errors)
    assert replay_session.store.snapshot() == before
    assert replay_session.looked_up_order_ids == set()


@pytest.mark.parametrize(
    "prompt",
    [
        "I do not want a refund for ORD-1001. Please just check the order status.",
        "I don't want a refund for ORD-1001, only tracking information.",
        "Please do not give me my money back for ORD-1001; check its status only.",
    ],
)
async def test_declined_refund_paraphrases_remain_lookup_only_in_direct_tools_and_replay(prompt):
    session = SupportSession()
    direct = await make_support_agent(session=session).run(prompt)
    assert session.store.refunds == []
    assert session.store.escalations == []
    assert direct.output
    scenario = next(s for s in support_scenarios() if s.name == "status_no_refund")
    scenario.prompt = prompt
    replay = await run_scenario(make_support_agent(), scenario)
    assert replay.passed, replay.model_dump_json(indent=2)
    assert [call.tool_name for call in replay.calls] == ["lookup_order"]
