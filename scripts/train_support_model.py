#!/usr/bin/env python3
"""Run a bounded, local-only MLX LoRA experiment on synthetic support decisions.

Optional MLX dependencies are imported only after argument and filesystem checks.
This script never downloads a model, reads real customers, or calls a paid API.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _local_path(value: str, *, directory: bool = False, exists: bool = False) -> Path:
    path = Path(value)
    path = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Experiment paths must stay inside this project: {path}")
    if exists and not path.exists():
        raise ValueError(f"Required local path does not exist: {path}")
    if exists and directory and not path.is_dir():
        raise ValueError(f"Expected a local directory: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{line_number}")
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != {"messages", "tools"}:
                raise ValueError(f"Expected messages+tools record at {path}:{line_number}")
            messages = record["messages"]
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"Expected decision-prefix messages at {path}:{line_number}")
            if messages[0].get("role") != "system" or messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"Every record must end in an assistant target: {path}:{line_number}"
                )
            if not any(message.get("role") == "user" for message in messages):
                raise ValueError(f"Missing customer prompt at {path}:{line_number}")
            if not isinstance(record["tools"], list) or len(record["tools"]) != 3:
                raise ValueError(f"Expected all three support tool schemas: {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Training or validation data is empty: {path}")
    return records


def _preflight(
    records: list[dict[str, Any]], tokenizer: Any, max_seq_length: int, chat_dataset: Any
) -> dict[str, Any]:
    dataset = chat_dataset(records, tokenizer, mask_prompt=True)
    token_lengths: list[int] = []
    target_lengths: list[int] = []
    targets: Counter[str] = Counter()
    for index, record in enumerate(records):
        tokens, offset = dataset.process(record)
        if len(tokens) > max_seq_length:
            raise ValueError(
                f"Record {index} has {len(tokens)} tokens, exceeds {max_seq_length}; "
                "increase --max-seq-length. Truncation is never accepted."
            )
        if not 0 < offset < len(tokens) - 1:
            raise ValueError(f"Record {index} has no nonempty supervised assistant target")
        prefix = tokenizer.apply_chat_template(
            record["messages"][:-1],
            tools=record["tools"],
            add_generation_prompt=True,
            return_dict=False,
        )
        if list(tokens[:offset]) != list(prefix):
            raise ValueError(f"Record {index} masking does not match the inference prefix")
        target_text = tokenizer.decode(tokens[offset:])
        target = record["messages"][-1]
        tool_calls = target.get("tool_calls", [])
        if tool_calls:
            rendered = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", target_text, re.S)
            if len(rendered) != len(tool_calls):
                raise ValueError(f"Record {index} lost native Qwen tool-call target syntax")
            for rendered_call, call in zip(rendered, tool_calls, strict=True):
                wire = json.loads(rendered_call)
                function = call["function"]
                arguments = function["arguments"]
                expected = json.loads(arguments) if isinstance(arguments, str) else arguments
                if wire != {"name": function["name"], "arguments": expected}:
                    raise ValueError(f"Record {index} changed tool name or arguments in rendering")
                targets[function["name"]] += 1
        else:
            content = target.get("content")
            if not isinstance(content, str) or not content.strip() or content not in target_text:
                raise ValueError(f"Record {index} lost its assistant text target")
            targets["final_text"] += 1
        token_lengths.append(len(tokens))
        target_lengths.append(len(tokens) - offset)
    return {
        "records": len(records),
        "token_length_min": min(token_lengths),
        "token_length_max": max(token_lengths),
        "target_tokens_min": min(target_lengths),
        "target_tokens_max": max(target_lengths),
        "target_kinds": dict(sorted(targets.items())),
        "masking": "all prefix tokens masked; final assistant decision supervised",
        "truncated_records": 0,
    }


def _check_split_isolation(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    """Reject canonical order IDs and overlap before any training allocation."""
    observed: dict[str, dict[str, set[str]]] = {}
    for split, records in splits.items():
        ids: set[str] = set()
        keys: set[str] = set()
        prompts: set[str] = set()
        for record in records:
            encoded = json.dumps(record)
            found_ids = set(re.findall(r"\bORD-\d{4}\b", encoded, re.I))
            family = "2" if split == "train" else "3"
            if any(identifier.upper()[4] != family for identifier in found_ids):
                raise ValueError(
                    f"{split} contains an order ID outside its separate ORD-{family}xxx family"
                )
            ids.update(identifier.upper() for identifier in found_ids)
            for message in record["messages"]:
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                if message.get("role") == "system":
                    keys.update(re.findall(r"^Trusted request ID: ([^\n]+)$", content, re.M))
                if message.get("role") == "user":
                    prompts.add(content)
        observed[split] = {"order_ids": ids, "request_ids": keys, "customer_prompts": prompts}
    names = list(observed)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            for kind in observed[first]:
                if observed[first][kind] & observed[second][kind]:
                    raise ValueError(f"{kind} overlap between {first} and {second}")
    return {
        split: {kind: len(values) for kind, values in categories.items()}
        for split, categories in observed.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".models/qwen3-4b-instruct")
    parser.add_argument("--data", default=".training-data/support")
    parser.add_argument("--adapter", default=".models/support-lora")
    parser.add_argument("--manifest", default="reports/support-training.json")
    parser.add_argument("--experiment", help="Explicit experiment label recorded in provenance")
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--prepare-only", action="store_true", help="Validate without training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.iters <= 1000 or not 1 <= args.num_layers <= 16 or not 1 <= args.rank <= 32:
        raise ValueError("Use 1..1000 iterations, 1..16 layers, and rank 1..32")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1e-3:
        raise ValueError("Learning rate must be finite and in (0, 0.001]")
    if not 512 <= args.max_seq_length <= 8192:
        raise ValueError("Maximum sequence length must be between 512 and 8192")
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("Seed must be between 0 and 2**32-1 for the NumPy batch sampler")
    if args.experiment is not None and not args.experiment.strip():
        raise ValueError("Experiment label must not be blank")
    model_path = _local_path(args.model, directory=True, exists=True)
    data_path = _local_path(args.data, directory=True, exists=True)
    adapter_path = _local_path(args.adapter)
    manifest_path = _local_path(args.manifest)
    if not adapter_path.is_relative_to(PROJECT_ROOT / ".models"):
        raise ValueError("Adapter output must stay inside the ignored .models directory")
    if not manifest_path.is_relative_to(PROJECT_ROOT / "reports"):
        raise ValueError("Manifest output must stay inside the ignored reports directory")
    if not (model_path / "config.json").is_file() or not list(model_path.glob("*.safetensors")):
        raise ValueError("Model path must contain an already downloaded config and weights")
    if (adapter_path / "adapters.safetensors").exists() and not args.prepare_only:
        raise ValueError("Adapter already exists; choose a new --adapter path to preserve this run")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") != "prepared":
            raise ValueError("Run manifest already exists; choose a new --manifest to preserve it")

    # Force every optional model/tokenizer helper into local-file-only mode.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import numpy as np
        from mlx_lm.lora import CONFIG_DEFAULTS, train_model
        from mlx_lm.tuner.callbacks import TrainingCallback
        from mlx_lm.tuner.datasets import ChatDataset
        from mlx_lm.utils import load, load_tokenizer
    except ImportError as exc:
        raise RuntimeError("Run this script with the optional local MLX environment") from exc

    tokenizer = load_tokenizer(
        model_path, tokenizer_config_extra={"local_files_only": True, "trust_remote_code": False}
    )
    train_records = _load_records(data_path / "train.jsonl")
    valid_records = _load_records(data_path / "valid.jsonl")
    splits = {"train": train_records, "valid": valid_records}
    if (data_path / "test.jsonl").exists():
        splits["test"] = _load_records(data_path / "test.jsonl")
    isolation = _check_split_isolation(splits)
    preflight = {
        "train": _preflight(train_records, tokenizer, args.max_seq_length, ChatDataset),
        "valid": _preflight(valid_records, tokenizer, args.max_seq_length, ChatDataset),
    }
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    corpus_manifest_path = data_path / "manifest.json"
    corpus_metadata = (
        json.loads(corpus_manifest_path.read_text()) if corpus_manifest_path.exists() else {}
    )
    if not isinstance(corpus_metadata, dict):
        raise ValueError("Corpus manifest must be a JSON object")

    config = copy.deepcopy(CONFIG_DEFAULTS)
    config.update(
        model=str(model_path),
        data=str(data_path),
        adapter_path=str(adapter_path),
        train=True,
        fine_tune_type="lora",
        optimizer="adam",
        mask_prompt=True,
        num_layers=args.num_layers,
        batch_size=1,
        iters=args.iters,
        val_batches=10,
        learning_rate=args.learning_rate,
        steps_per_report=10,
        steps_per_eval=50,
        grad_accumulation_steps=1,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=True,
        save_every=50,
        lora_parameters={"rank": args.rank, "dropout": 0.0, "scale": 20.0},
        report_to=None,
        trust_remote_code=False,
        seed=args.seed,
    )
    metadata_path = model_path / ".cache/huggingface/download/config.json.metadata"
    revision = metadata_path.read_text().splitlines()[0] if metadata_path.exists() else None
    model_files = [model_path / "config.json", model_path / "chat_template.jinja"]
    model_files += list(model_path.glob("*.safetensors"))
    manifest: dict[str, Any] = {
        "status": "prepared",
        "started_at": datetime.now(UTC).isoformat(),
        "experiment": args.experiment or "synthetic-support-decision-lora",
        "training_script_sha256": _sha256(Path(__file__).resolve()),
        "base_model": {
            "upstream": "Qwen/Qwen3-4B-Instruct-2507",
            "conversion": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
            "license": "Apache-2.0",
            "local_path": str(model_path),
            "download_revision": revision,
            "files_sha256": {path.name: _sha256(path) for path in model_files},
        },
        "dataset": {
            "path": str(data_path),
            "corpus_version": corpus_metadata.get("corpus_version"),
            "files_sha256": {
                path.name: _sha256(path)
                for path in data_path.glob("*")
                if path.is_file() and path.suffix in {".jsonl", ".json"}
            },
            "preflight": preflight,
            "split_isolation": isolation,
            "canonical_evaluation_data_used_for_training": False,
            "source": "synthetic reviewed policy-expert decision prefixes",
        },
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "mlx_lm": _version("mlx-lm"),
            "mlx": _version("mlx"),
            "transformers": _version("transformers"),
            "numpy": _version("numpy"),
        },
        "hyperparameters": config,
        "training_reports": [],
        "validation_reports": [],
        "evaluation": "Pending independent tool-call regression evaluation; loss is not success.",
        "seed_controls": {
            "numpy_batch_sampler": args.seed,
            "mlx_adapter_initialization": "train_model calls mx.random.seed(args.seed)",
        },
    }
    if corpus_metadata:
        manifest["dataset"]["corpus_manifest"] = corpus_metadata
    _write_json(manifest_path, manifest)
    if args.prepare_only:
        print(f"Preflight passed. Manifest: {manifest_path}", flush=True)
        return 0

    class LocalReports(TrainingCallback):
        def on_train_loss_report(self, info: dict[str, Any]) -> None:
            if not math.isfinite(info["train_loss"]):
                raise ValueError("Training loss became non-finite; stopping this experiment")
            manifest["training_reports"].append(info)
            _write_json(manifest_path, manifest)
            print(json.dumps({"training": info}), flush=True)

        def on_val_loss_report(self, info: dict[str, Any]) -> None:
            if not math.isfinite(info["val_loss"]):
                raise ValueError("Validation loss became non-finite; stopping this experiment")
            manifest["validation_reports"].append(info)
            _write_json(manifest_path, manifest)
            print(json.dumps({"validation": info}), flush=True)

    started = time.perf_counter()
    try:
        manifest["status"] = "training"
        _write_json(manifest_path, manifest)
        model, tokenizer = load(
            str(model_path),
            tokenizer_config={"local_files_only": True, "trust_remote_code": False},
            trust_remote_code=False,
        )
        # train_model seeds MLX internally, but NumPy is seeded only by the
        # library's run() wrapper, which we bypass to retain local callbacks.
        # Batch iteration consumes np.random.permutation, so seed it explicitly.
        np.random.seed(args.seed)
        train_model(
            SimpleNamespace(**config),
            model,
            ChatDataset(train_records, tokenizer, mask_prompt=True),
            ChatDataset(valid_records, tokenizer, mask_prompt=True),
            LocalReports(),
        )
        saved_weights = adapter_path / "adapters.safetensors"
        if not saved_weights.is_file() or saved_weights.stat().st_size == 0:
            raise RuntimeError("Training returned without nonempty saved adapter weights")
        manifest["status"] = "completed"
        manifest["adapter_files_sha256"] = {
            path.name: _sha256(path) for path in adapter_path.glob("*") if path.is_file()
        }
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(manifest_path, manifest)
    print(f"Local adapter saved to {adapter_path}. Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
