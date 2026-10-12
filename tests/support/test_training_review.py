"""Independent review of training isolation and policy-expert decision targets.

These checks use the audited full trajectories, not the generator's manifest
claims, and replay responses through a fresh real backend for every case.
Optional MLX/tokenizer checks live in the training script's native preflight.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from agent_regression.support.agent import make_support_agent
from agent_regression.support.evaluation import diagnostic_scenarios
from agent_regression.support.scenarios import support_scenarios
from agent_regression.support.store import LocalSupportStore, SupportSession
from agent_regression.support.training import CATEGORIES, TrainingCorpus, generate_training_data


@pytest.fixture(scope="module")
def corpus() -> TrainingCorpus:
    return generate_training_data()


def _order_ids(value: Any) -> set[str]:
    return {match.upper() for match in re.findall(r"ORD-\d{4}", json.dumps(value), re.I)}


def _deidentified_wording(prompt: str) -> str:
    without_identifiers = re.sub(r"ORD-\d{4}", "ORDER", prompt, flags=re.I)
    without_amounts = re.sub(r"\b\d+\b", "NUMBER", without_identifiers)
    return " ".join(without_amounts.lower().split())


def test_no_canonical_or_diagnostic_evaluation_data_is_in_training(
    corpus: TrainingCorpus,
) -> None:
    evaluations = [*support_scenarios(), *diagnostic_scenarios()]
    reserved_ids = set().union(*(_order_ids(s.model_dump()) for s in evaluations))
    reserved_prompts = {s.prompt for s in evaluations}
    for split, records in corpus.records.items():
        assert not _order_ids(records) & reserved_ids, split
        assert "req-support-demo" not in json.dumps(records), split
    for case in corpus.cases:
        assert not _order_ids(case) & reserved_ids, case["case_id"]
        assert case["prompt"] not in reserved_prompts, case["case_id"]


def test_split_independence_covers_conversations_not_only_manifest_labels(
    corpus: TrainingCorpus,
) -> None:
    split_ids: dict[str, set[str]] = {}
    split_keys: dict[str, set[str]] = {}
    split_templates: dict[str, set[str]] = {}
    split_wordings: dict[str, set[str]] = {}
    for split, records in corpus.records.items():
        cases = [case for case in corpus.cases if case["split"] == split]
        split_ids[split] = _order_ids(records) | _order_ids(cases)
        split_keys[split] = {case["request_id"] for case in cases}
        split_templates[split] = {case["template_id"] for case in cases}
        split_wordings[split] = {_deidentified_wording(case["prompt"]) for case in cases}
        assert len(split_keys[split]) == len(cases)
        assert len(split_templates[split]) == len(cases)
        assert len(split_wordings[split]) == len(cases)
    for first, second in combinations(corpus.records, 2):
        assert not split_ids[first] & split_ids[second]
        assert not split_keys[first] & split_keys[second]
        assert not split_templates[first] & split_templates[second]
        assert not split_wordings[first] & split_wordings[second]


def test_every_assistant_action_has_its_own_unmasked_target_record(
    corpus: TrainingCorpus,
) -> None:
    used_indexes: dict[str, list[int]] = {split: [] for split in corpus.records}
    for case in corpus.cases:
        messages = case["trajectory"]
        assistant_positions = [
            index for index, message in enumerate(messages) if message["role"] == "assistant"
        ]
        assert len(assistant_positions) == len(case["calls"]) + 1
        assert len(case["record_indexes"]) == len(assistant_positions)
        for position, record_index in zip(assistant_positions, case["record_indexes"], strict=True):
            record = corpus.records[case["split"]][record_index]
            assert set(record) == {"messages", "tools"}
            assert record["messages"] == messages[: position + 1]
            assert record["tools"] == corpus.tools
            assert record["messages"][-1]["role"] == "assistant"
            assert (
                record["messages"][-1].get("tool_calls")
                or record["messages"][-1]["content"].strip()
            )
            used_indexes[case["split"]].append(record_index)
    for split, indexes in used_indexes.items():
        assert sorted(indexes) == list(range(len(corpus.records[split])))
        targets = Counter(
            call["function"]["name"]
            for record in corpus.records[split]
            for call in record["messages"][-1].get("tool_calls", [])
        )
        assert targets["lookup_order"] > 0
        assert targets["issue_refund"] > 0
        assert targets["escalate_to_human"] > 0


def test_expert_actions_use_user_identifiers_and_trusted_request_context(
    corpus: TrainingCorpus,
) -> None:
    for case in corpus.cases:
        mentioned = _order_ids(case["prompt"])
        if case["category"] in {"missing_id", "general_human"}:
            assert not mentioned
        elif case["category"] == "multiple_ids":
            assert len(mentioned) == 2
            assert not case["calls"]
        else:
            assert len(mentioned) == 1
            assert mentioned == set(case["requested_order_ids"])
        context = case["trajectory"][0]["content"]
        assert f"Trusted request ID: {case['request_id']}\n" in context
        assert "Trusted authenticated customer ID: cust-001\n" in context
        assert case["trajectory"][1] == {"role": "user", "content": case["prompt"]}
        for call in case["calls"]:
            bound_order = call["arguments"].get("order_id")
            if bound_order is not None:
                assert bound_order in mentioned
                assert len(mentioned) == 1


@pytest.mark.parametrize("category", CATEGORIES)
def test_tool_targets_and_saved_responses_match_actual_authorized_backend(
    corpus: TrainingCorpus, category: str, tmp_path: Path
) -> None:
    for case in (case for case in corpus.cases if case["category"] == category):
        seed_path = tmp_path / "seed.json"
        seed_path.write_text(json.dumps(case["seed"]), encoding="utf-8")
        store = LocalSupportStore.from_seed(seed_path)
        session = SupportSession(
            store=store, request_id=case["request_id"], user_message=case["prompt"]
        )
        trajectory_calls = [
            call for message in case["trajectory"] for call in message.get("tool_calls", [])
        ]
        tool_returns = [message for message in case["trajectory"] if message["role"] == "tool"]
        assert len(trajectory_calls) == len(tool_returns) == len(case["calls"])
        for index, (target, tool_return, observed) in enumerate(
            zip(trajectory_calls, tool_returns, case["calls"], strict=True)
        ):
            function = target["function"]
            arguments = function["arguments"]
            # MLX normalizes earlier OpenAI argument strings into dictionaries
            # before its native template, so saved training prefixes must too.
            assert isinstance(arguments, dict)
            assert target["type"] == "function"
            assert tool_return["tool_call_id"] == target["id"]
            assert tool_return["name"] == function["name"] == observed["tool_name"]
            assert arguments == observed["arguments"]
            assert observed["step"] == index
            result = getattr(store, function["name"])(session, **arguments)
            assert json.loads(tool_return["content"]) == result == observed["response"]
            assert (
                tool_return["content"]
                == ToolReturnPart(
                    tool_name=function["name"], content=result, tool_call_id=target["id"]
                ).model_response_str()
            )
            if function["name"] == "issue_refund":
                assert result["status"] == "refunded"
                assert type(arguments["amount_cents"]) is int
                assert arguments["idempotency_key"] == case["request_id"]
            if function["name"] == "escalate_to_human":
                assert result["status"] == "escalated"
        assert store.snapshot() == case["ledgers"]


@pytest.mark.parametrize("category", CATEGORIES)
def test_independent_policy_expectations_for_every_expert_category(
    corpus: TrainingCorpus, category: str
) -> None:
    refund_categories = {"eligible", "duplicate_wording", "historical_partial", "boundary_30"}
    clarification_categories = {"missing_id", "multiple_ids"}
    escalation_reasons = {
        "high_value": "policy_exception",
        "undelivered": "delivery_issue",
        "bound_human": "customer_request",
        "policy_exception": "policy_exception",
    }
    for case in (case for case in corpus.cases if case["category"] == category):
        calls = case["calls"]
        names = [call["tool_name"] for call in calls]
        final = case["trajectory"][-1]["content"]
        if category in refund_categories:
            assert names == ["lookup_order", "issue_refund"]
            order = calls[0]["response"]["order"]
            assert calls[0]["response"]["refund_eligibility"]["eligible"] is True
            assert calls[1]["arguments"] == {
                "order_id": order["order_id"],
                "amount_cents": order["refundable_cents"],
                "idempotency_key": case["request_id"],
            }
            assert calls[1]["response"]["refund_id"] in final
            assert len(case["ledgers"]["refunds"]) == 1
        elif category in clarification_categories:
            assert names == []
            assert "?" in final
        elif category in escalation_reasons:
            assert names == ["lookup_order", "escalate_to_human"]
            assert calls[1]["arguments"] == {
                "reason": escalation_reasons[category],
                "order_id": case["requested_order_ids"][0],
            }
            assert calls[1]["response"]["ticket_id"] in final
        elif category == "general_human":
            assert names == ["escalate_to_human"]
            assert calls[0]["arguments"] == {"reason": "customer_request", "order_id": None}
            assert calls[0]["response"]["ticket_id"] in final
        else:
            assert names == ["lookup_order"]
            assert not case["ledgers"]["refunds"]
            assert not case["ledgers"]["escalations"]
            if category == "partial_request":
                assert "?" in final
            if category in {"unknown", "foreign"}:
                assert calls[0]["response"] == {
                    "status": "not_found",
                    "order_id": case["requested_order_ids"][0],
                }
        # A model must see completed tool returns before action/receipt targets.
        for index, message in enumerate(case["trajectory"]):
            if message["role"] == "assistant" and message.get("tool_calls"):
                assert case["trajectory"][index + 1]["role"] == "tool"
        if category not in clarification_categories:
            assert case["trajectory"][-2]["role"] == "tool"


@pytest.mark.parametrize("category", ["eligible", "high_value", "general_human"])
@pytest.mark.asyncio
async def test_native_training_prefix_matches_actual_openai_mapping(
    corpus: TrainingCorpus, category: str, tmp_path: Path
) -> None:
    # This optional integration check maps requests without calling any provider.
    # Tokenizer-specific equality is checked separately using local model files.
    pytest.importorskip("openai")
    from agent_regression.support.local_model import close_local_model, make_local_model

    case = next(
        case for case in corpus.cases if case["category"] == category and case["split"] == "train"
    )
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps(case["seed"]), encoding="utf-8")
    session = SupportSession(
        store=LocalSupportStore.from_seed(seed_path),
        request_id=case["request_id"],
        user_message=case["prompt"],
    )
    decisions = [message for message in case["trajectory"] if message["role"] == "assistant"]
    captured: list[tuple[Any, Any]] = []

    def scripted(messages: Any, info: Any) -> ModelResponse:
        target = decisions[len(captured)]
        captured.append((deepcopy(messages), info))
        if target.get("tool_calls"):
            call = target["tool_calls"][0]
            function = call["function"]
            return ModelResponse(
                parts=[
                    ToolCallPart(function["name"], function["arguments"], tool_call_id=call["id"])
                ]
            )
        return ModelResponse(parts=[TextPart(target["content"])])

    await make_support_agent(FunctionModel(scripted), session=session).run(case["prompt"])
    assert len(captured) == len(case["record_indexes"])
    assert session.store.snapshot() == case["ledgers"]
    model = make_local_model()
    try:
        for (messages, info), record_index in zip(captured, case["record_indexes"], strict=True):
            record = corpus.records["train"][record_index]
            settings, parameters = model.prepare_request(
                info.model_settings, info.model_request_parameters
            )
            wire_tools = [
                model._map_tool_definition(tool, settings or {})
                for tool in parameters.function_tools
            ]
            # Native Qwen tojson preserves property order, so dict equality alone
            # would miss a changed token prefix in an exported training record.
            assert json.dumps(wire_tools) == json.dumps(record["tools"])
            wire = await model._map_messages(messages, parameters, model_settings=settings)
            expected = record["messages"][:-1]
            assert len(wire) == len(expected)
            for actual, saved in zip(wire, expected, strict=True):
                assert actual["role"] == saved["role"]
                assert (actual.get("content") or "") == saved["content"]
                actual_calls = actual.get("tool_calls", [])
                saved_calls = saved.get("tool_calls", [])
                assert len(actual_calls) == len(saved_calls)
                for observed, target in zip(actual_calls, saved_calls, strict=True):
                    assert observed["function"]["name"] == target["function"]["name"]
                    # This is the normalization performed by MLX's HTTP server.
                    arguments = json.loads(observed["function"]["arguments"])
                    assert json.dumps(arguments) == json.dumps(target["function"]["arguments"])
    finally:
        await close_local_model(model)
