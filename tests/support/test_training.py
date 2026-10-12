"""Corpus provenance, split isolation, decision supervision, and local export."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from agent_regression.support.evaluation import diagnostic_scenarios
from agent_regression.support.scenarios import support_scenarios
from agent_regression.support.training import (
    CATEGORIES,
    DEFAULT_SEED,
    generate_training_data,
    write_training_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def corpus():
    return generate_training_data()


def test_corpus_is_deterministic_and_only_synthetic(corpus):
    again = generate_training_data(seed=DEFAULT_SEED)
    assert corpus == again
    assert corpus.manifest["customer_data_used"] is False
    assert corpus.manifest["trained_model_used_to_generate_targets"] is False
    assert corpus.manifest["evaluation_fixtures_used"] is False
    assert corpus.manifest["purpose"] == "local_lora_training_experiment"
    assert corpus.manifest["generator_seed"] == DEFAULT_SEED
    different = generate_training_data(seed=DEFAULT_SEED + 1)
    assert different.records != corpus.records
    assert different.cases[0]["seed"] != corpus.cases[0]["seed"]


def test_mutating_a_generated_row_does_not_change_other_records_or_later_generation(corpus):
    changed = generate_training_data()
    first = changed.records["train"][0]
    first["messages"][0]["content"] = "tampered"
    first["tools"][0]["function"]["name"] = "tampered"
    assert changed.records["train"][1] == corpus.records["train"][1]
    assert generate_training_data() == corpus


def test_all_policy_intents_have_whole_request_train_validation_and_test_groups(corpus):
    assert 300 <= len(corpus.records["train"]) <= 500
    assert 40 <= len(corpus.records["valid"]) <= 60
    assert 40 <= len(corpus.records["test"]) <= 60
    assert {case["category"] for case in corpus.cases} == set(CATEGORIES)
    for split in ["train", "valid", "test"]:
        assert {case["category"] for case in corpus.cases if case["split"] == split} == set(
            CATEGORIES
        )
        assert corpus.manifest["records_per_split"][split] == len(corpus.records[split])
        assert corpus.manifest["requests_per_split"][split] == sum(
            case["split"] == split for case in corpus.cases
        )
    assert len({case["case_id"] for case in corpus.cases}) == len(corpus.cases)


def test_no_evaluation_identifiers_keys_or_customer_prompts_enter_training(corpus):
    evaluations = [*support_scenarios(), *diagnostic_scenarios()]
    serialized = json.dumps(
        {
            "records": corpus.records,
            "cases": corpus.cases,
            "tools": corpus.tools,
            "manifest": corpus.manifest,
        }
    )
    evaluation_ids = set()
    for scenario in evaluations:
        evaluation_ids.update(re.findall(r"ORD-\d{4}", scenario.model_dump_json()))
        assert scenario.prompt not in serialized
    for order_id in evaluation_ids:
        assert order_id not in serialized
    assert "req-support-demo" not in serialized
    assert "Private customer item" not in serialized
    assert "@" not in serialized  # No customer email addresses or contact data.


def test_ids_request_keys_and_complete_wording_templates_are_disjoint_across_splits(corpus):
    groups = {}
    for split in ["train", "valid", "test"]:
        cases = [case for case in corpus.cases if case["split"] == split]
        groups[split] = {
            "ids": {
                order
                for case in cases
                for order in [*case["order_ids"], *case["requested_order_ids"]]
            },
            "keys": {case["request_id"] for case in cases},
            "templates": {case["template_id"] for case in cases},
            "normalized_prompts": {
                re.sub(r"\d+", "N", re.sub(r"ORD-\d{4}", "ORDER", case["prompt"], flags=re.I))
                for case in cases
            },
        }
        assert all(re.fullmatch(r"ORD-[23]\d{3}", order) for order in groups[split]["ids"])
        assert len(groups[split]["keys"]) == len(cases)
    for left, right in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        for name in groups[left]:
            assert groups[left][name].isdisjoint(groups[right][name]), (left, right, name)


def test_schema_and_targets_are_tool_call_training_not_only_final_text(corpus):
    assert {tool["function"]["name"] for tool in corpus.tools} == {
        "lookup_order",
        "issue_refund",
        "escalate_to_human",
    }
    definitions = {
        tool["function"]["name"]: tool["function"]["parameters"] for tool in corpus.tools
    }
    assert definitions["escalate_to_human"]["required"] == ["reason", "order_id"]
    for split, rows in corpus.records.items():
        tool_targets = []
        final_targets = []
        for row in rows:
            assert set(row) == {"messages", "tools"}
            assert row["tools"] == corpus.tools
            target = row["messages"][-1]
            assert target["role"] == "assistant"
            assert row["messages"][0]["role"] == "system"
            assert row["messages"][1]["role"] == "user"
            if target.get("tool_calls"):
                assert len(target["tool_calls"]) == 1
                assert target["content"] == ""
                function = target["tool_calls"][0]["function"]
                arguments = function["arguments"]
                assert isinstance(arguments, dict)
                schema = definitions[function["name"]]
                assert set(schema["required"]).issubset(arguments)
                assert set(arguments).issubset(schema["properties"])
                tool_targets.append(function["name"])
            else:
                assert target["content"]
                final_targets.append(target)
        assert set(tool_targets) == {"lookup_order", "issue_refund", "escalate_to_human"}
        assert len(tool_targets) > len(final_targets)
        assert corpus.manifest["tool_call_targets_per_split"][split] == len(tool_targets)


def test_record_index_keeps_all_decisions_of_each_request_in_one_split(corpus):
    consumed = {split: set() for split in corpus.records}
    for case in corpus.cases:
        assistant_positions = [
            i for i, message in enumerate(case["trajectory"]) if message["role"] == "assistant"
        ]
        assert len(case["record_indexes"]) == len(assistant_positions)
        for row_index, message_index in zip(
            case["record_indexes"], assistant_positions, strict=True
        ):
            assert row_index not in consumed[case["split"]]
            consumed[case["split"]].add(row_index)
            row = corpus.records[case["split"]][row_index]
            assert row["messages"] == case["trajectory"][: message_index + 1]
            assert row["messages"][1]["content"] == case["prompt"]
            assert f"Trusted request ID: {case['request_id']}" in row["messages"][0]["content"]
    for split, rows in corpus.records.items():
        assert consumed[split] == set(range(len(rows)))


def test_no_model_request_is_used_to_build_expert_ground_truth(monkeypatch):
    async def prohibited_request(*args, **kwargs):
        raise AssertionError("Training ground truth must not invoke an agent model")

    monkeypatch.setattr(FunctionModel, "request", prohibited_request)
    corpus = generate_training_data()
    assert corpus.records["train"]


@pytest.mark.parametrize("seed", [-1, True, 1.1, "seed"])
def test_invalid_generator_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        generate_training_data(seed=seed)


def test_local_export_roundtrips_all_rows_and_refuses_to_overwrite(tmp_path: Path, corpus):
    destination = tmp_path / "generated"
    manifest = write_training_data(destination)
    assert manifest == corpus.manifest
    assert json.loads((destination / "manifest.json").read_text()) == manifest
    for split, rows in corpus.records.items():
        actual = [
            json.loads(line) for line in (destination / f"{split}.jsonl").read_text().splitlines()
        ]
        assert actual == rows
    assert [
        json.loads(line) for line in (destination / "cases.jsonl").read_text().splitlines()
    ] == corpus.cases
    snapshots = {file.name: file.read_bytes() for file in destination.iterdir()}
    with pytest.raises(FileExistsError):
        write_training_data(destination)
    assert snapshots == {file.name: file.read_bytes() for file in destination.iterdir()}


def test_native_tool_schemas_include_prepared_strict_mode_and_nested_field_order(corpus):
    assert corpus.manifest["corpus_version"] == "synthetic-support-lora-native-v2"
    for tool in corpus.tools:
        assert list(tool) == ["type", "function"]
        assert list(tool["function"]) == ["name", "description", "parameters", "strict"]
        assert tool["function"]["strict"] is True
    escalation = next(
        tool for tool in corpus.tools if tool["function"]["name"] == "escalate_to_human"
    )
    # Pydantic's provider preparation moves union keywords after their metadata.
    # Equality of dictionaries alone cannot catch this template-visible ordering.
    assert list(escalation["function"]["parameters"]["properties"]["order_id"]) == [
        "description",
        "anyOf",
    ]


def test_native_tool_returns_match_actual_pydantic_compact_response_serialization(corpus):
    for case in corpus.cases:
        returns = [message for message in case["trajectory"] if message["role"] == "tool"]
        for message, call in zip(returns, case["calls"], strict=True):
            expected = ToolReturnPart(
                tool_name=call["tool_name"],
                content=call["response"],
                tool_call_id=message["tool_call_id"],
            ).model_response_str()
            assert message["content"] == expected
            assert json.loads(message["content"]) == call["response"]
            if call["tool_name"] == "lookup_order":
                assert message["content"].startswith('{"status":')
                assert message["content"] != json.dumps(call["response"], sort_keys=True)


def test_native_export_preserves_nested_model_facing_object_order(tmp_path, corpus):
    write_training_data(tmp_path / "native")
    for split, rows in corpus.records.items():
        exported = [
            json.loads(line)
            for line in (tmp_path / "native" / f"{split}.jsonl").read_text().splitlines()
        ]
        # JSON semantic equality misses nested order changes from sort_keys=True.
        # Compare complete ordered serialization after reading every exported row.
        assert [json.dumps(row, ensure_ascii=False) for row in exported] == [
            json.dumps(row, ensure_ascii=False) for row in rows
        ]
        for row in exported:
            for message in row["messages"]:
                for call in message.get("tool_calls", []):
                    assert isinstance(call["function"]["arguments"], dict)


async def _capture_openai_wire(case, tmp_path, monkeypatch):
    """Run against an in-memory HTTP responder, never a language model service."""
    import copy

    pytest.importorskip("openai", reason="Optional OpenAI transport contract test")
    import httpx2
    from pydantic_ai import models
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from agent_regression.support.agent import make_support_agent
    from agent_regression.support.store import LocalSupportStore, SupportSession

    # The test is permitted to use only the installed in-memory MockTransport.
    # No socket or inference service is used even when the global guard is lifted.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    captured = []
    calls = case["calls"]

    def respond(request):
        captured.append(json.loads(request.content))
        turn = len(captured) - 1
        if turn < len(calls):
            call = calls[turn]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{turn + 1}",
                        "type": "function",
                        "function": {
                            "name": call["tool_name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": case["trajectory"][-1]["content"]}
            finish = "stop"
        return httpx2.Response(
            200,
            json={
                "id": "mock-native-contract",
                "object": "chat.completion",
                "created": 0,
                "model": "mock-native-contract",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    seed_path = tmp_path / "mocked-wire-seed.json"
    seed_path.write_text(json.dumps(case["seed"]), encoding="utf-8")
    session = SupportSession(
        store=LocalSupportStore.from_seed(seed_path),
        request_id=case["request_id"],
        user_message=case["prompt"],
    )
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond), trust_env=False
    ) as client:
        provider = OpenAIProvider(
            base_url="http://127.0.0.1:9999/v1", api_key="mock-only", http_client=client
        )
        model = OpenAIChatModel("mock-native-contract", provider=provider)
        await make_support_agent(model, session=session).run(case["prompt"])
    # Normalize exactly the plain-text changes MLX makes before template rendering.
    normalized = copy.deepcopy(captured)
    for payload in normalized:
        for message in payload["messages"]:
            if message.get("content") is None:
                message["content"] = ""
            for call in message.get("tool_calls", []):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return captured, normalized


@pytest.mark.parametrize("category", ["eligible", "high_value"])
async def test_native_decision_prefixes_match_actual_mocked_openai_http_frames(
    corpus, category, tmp_path, monkeypatch
):
    case = next(case for case in corpus.cases if case["category"] == category)
    captured, normalized = await _capture_openai_wire(case, tmp_path, monkeypatch)
    assert len(captured) == len(case["record_indexes"]) == 3
    for payload, native, index in zip(captured, normalized, case["record_indexes"], strict=True):
        row = corpus.records[case["split"]][index]
        prefix = row["messages"][:-1]
        assert json.dumps(payload["tools"], ensure_ascii=False) == json.dumps(
            row["tools"], ensure_ascii=False
        )
        assert native["messages"][:2] == prefix[:2]
        assert [message["role"] for message in native["messages"]] == [
            message["role"] for message in prefix
        ]
        for actual, expected in zip(native["messages"], prefix, strict=True):
            assert actual["content"] == expected["content"]
            if actual.get("tool_calls"):
                actual_functions = [call["function"] for call in actual["tool_calls"]]
                expected_functions = [call["function"] for call in expected["tool_calls"]]
                assert [json.dumps(function) for function in actual_functions] == [
                    json.dumps(function) for function in expected_functions
                ]


@pytest.mark.parametrize("category", ["eligible", "high_value"])
async def test_native_tokenizer_prefixes_match_mocked_wire_after_mlx_normalization(
    corpus, category, tmp_path, monkeypatch
):
    utils = pytest.importorskip(
        "mlx_lm.utils", reason="Optional local native-tokenizer contract test"
    )
    server = pytest.importorskip("mlx_lm.server")
    model_path = PROJECT_ROOT / ".models/qwen3-4b-instruct"
    if not (model_path / "tokenizer_config.json").is_file():
        pytest.skip("A local Qwen tokenizer is required; this test never downloads one")
    tokenizer = utils.load_tokenizer(
        model_path, tokenizer_config_extra={"local_files_only": True, "trust_remote_code": False}
    )
    case = next(case for case in corpus.cases if case["category"] == category)
    captured, _ = await _capture_openai_wire(case, tmp_path, monkeypatch)
    for payload, index in zip(captured, case["record_indexes"], strict=True):
        # Exercise the installed MLX server normalizer, not a guessed approximation.
        server.process_message_content(payload["messages"])
        row = corpus.records[case["split"]][index]
        training = tokenizer.apply_chat_template(
            row["messages"][:-1], tools=row["tools"], add_generation_prompt=True, return_dict=False
        )
        inference = tokenizer.apply_chat_template(
            payload["messages"],
            tools=payload["tools"],
            add_generation_prompt=True,
            return_dict=False,
        )
        assert list(training) == list(inference), (category, index)
