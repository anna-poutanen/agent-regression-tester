"""Cheap validation of the optional training script's filesystem/data guards."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "support_training_script", PROJECT_ROOT / "scripts/train_support_model.py"
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def record(order: str, prompt: str, key: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": f"Policy\nTrusted request ID: {key}\n"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [{"order_id": order}]},
        ],
        "tools": [{"name": name} for name in ["lookup", "refund", "escalate"]],
    }


def test_training_script_import_does_not_require_mlx() -> None:
    assert callable(SCRIPT.main)
    assert SCRIPT._parser().parse_args([]).iters == 150


def test_model_paths_cannot_escape_project_even_through_parent_component() -> None:
    with pytest.raises(ValueError, match="inside this project"):
        SCRIPT._local_path("../other-model")


def test_model_paths_cannot_escape_project_through_symlink(tmp_path: Path) -> None:
    link = PROJECT_ROOT / ".training-test-link"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="inside this project"):
            SCRIPT._local_path(str(link))
    finally:
        link.unlink(missing_ok=True)


def test_missing_local_path_fails_before_model_import() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        SCRIPT.main(["--model", ".models/nonexistent-test-model"])


@pytest.mark.parametrize("rate", ["nan", "inf", "0", "0.1"])
def test_nonfinite_or_unbounded_learning_rate_rejected(rate: str) -> None:
    with pytest.raises(ValueError, match="Learning rate"):
        SCRIPT.main(["--learning-rate", rate])


@pytest.mark.parametrize("seed", ["-1", "4294967296"])
def test_numpy_seed_range_rejected_before_model_loading(seed: str) -> None:
    with pytest.raises(ValueError, match="NumPy batch sampler"):
        SCRIPT.main(["--seed", seed])


def test_blank_experiment_label_rejected_before_model_loading() -> None:
    with pytest.raises(ValueError, match="Experiment label"):
        SCRIPT.main(["--experiment", " "])


def test_load_records_rejects_empty_corpus(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        SCRIPT._load_records(path)


def test_load_records_rejects_final_tool_response_target(tmp_path: Path) -> None:
    value = record("ORD-2001", "Refund ORD-2001.", "train-1")
    value["messages"][-1] = {"role": "tool", "content": "refunded"}
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="assistant target"):
        SCRIPT._load_records(path)


def test_split_guard_accepts_independent_order_request_and_prompt_families() -> None:
    summary = SCRIPT._check_split_isolation(
        {
            "train": [record("ORD-2001", "Refund ORD-2001.", "train-1")],
            "valid": [record("ORD-3001", "Please return my money for ORD-3001.", "valid-1")],
            "test": [record("ORD-3101", "Give me money back for ORD-3101.", "test-1")],
        }
    )
    assert summary["train"]["order_ids"] == 1


def test_split_guard_rejects_canonical_evaluation_ids() -> None:
    with pytest.raises(ValueError, match="separate ORD-2xxx"):
        SCRIPT._check_split_isolation(
            {"train": [record("ORD-1001", "Refund ORD-1001.", "train-1")]}
        )


@pytest.mark.parametrize("kind", ["order_ids", "request_ids", "customer_prompts"])
def test_split_guard_rejects_validation_test_leakage(kind: str) -> None:
    first = record("ORD-3001", "First wording.", "first-key")
    second = record("ORD-3101", "Second wording.", "second-key")
    if kind == "order_ids":
        second["messages"][-1]["tool_calls"][0]["order_id"] = "ORD-3001"
    elif kind == "request_ids":
        second["messages"][0]["content"] = first["messages"][0]["content"]
    else:
        second["messages"][1]["content"] = first["messages"][1]["content"]
    with pytest.raises(ValueError, match=f"{kind} overlap"):
        SCRIPT._check_split_isolation({"valid": [first], "test": [second]})


@pytest.mark.parametrize("save_adapter", [True, False])
def test_training_entrypoint_explicitly_seeds_numpy_before_library_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_adapter: bool
) -> None:
    """Direct train_model bypasses MLX-LM run(), where its NumPy seed lives."""
    monkeypatch.setattr(SCRIPT, "PROJECT_ROOT", tmp_path)
    model_path = tmp_path / ".models/base"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}")
    (model_path / "chat_template.jinja").write_text("minimal template")
    (model_path / "model.safetensors").write_bytes(b"dummy weights for guard testing")
    data_path = tmp_path / ".training-data/support"
    data_path.mkdir(parents=True)
    (data_path / "manifest.json").write_text(
        json.dumps({"corpus_version": "synthetic-support-native-test-v2"})
    )
    for split, value in {
        "train": record("ORD-2001", "Training wording.", "train-key"),
        "valid": record("ORD-3001", "Validation wording.", "valid-key"),
    }.items():
        (data_path / f"{split}.jsonl").write_text(json.dumps(value) + "\n")

    events: list[tuple[str, int | None]] = []
    numpy_module = ModuleType("numpy")
    numpy_module.random = SimpleNamespace(seed=lambda value: events.append(("numpy_seed", value)))
    lora_module = ModuleType("mlx_lm.lora")
    lora_module.CONFIG_DEFAULTS = {}

    def train_model(args: Any, model: Any, train: Any, valid: Any, callback: Any) -> None:
        assert events == [("numpy_seed", 321)]
        assert args.seed == 321
        events.append(("train_model", args.seed))
        # MLX-LM's actual train_model independently seeds mx.random(args.seed).
        adapter = Path(args.adapter_path)
        adapter.mkdir(parents=True)
        if save_adapter:
            (adapter / "adapters.safetensors").write_bytes(b"dummy adapter")

    lora_module.train_model = train_model
    callbacks_module = ModuleType("mlx_lm.tuner.callbacks")
    callbacks_module.TrainingCallback = object
    datasets_module = ModuleType("mlx_lm.tuner.datasets")
    datasets_module.ChatDataset = lambda *args, **kwargs: object()
    utils_module = ModuleType("mlx_lm.utils")
    utils_module.load_tokenizer = lambda *args, **kwargs: object()
    utils_module.load = lambda *args, **kwargs: (object(), object())
    for name, module in {
        "numpy": numpy_module,
        "mlx_lm": ModuleType("mlx_lm"),
        "mlx_lm.lora": lora_module,
        "mlx_lm.tuner": ModuleType("mlx_lm.tuner"),
        "mlx_lm.tuner.callbacks": callbacks_module,
        "mlx_lm.tuner.datasets": datasets_module,
        "mlx_lm.utils": utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(SCRIPT, "_preflight", lambda *args: {"records": 1, "truncated_records": 0})
    arguments = [
        "--model",
        ".models/base",
        "--iters",
        "1",
        "--seed",
        "321",
        "--experiment",
        "synthetic-support-native-experiment-v2",
    ]
    if save_adapter:
        assert SCRIPT.main(arguments) == 0
    else:
        with pytest.raises(RuntimeError, match="without nonempty saved adapter weights"):
            SCRIPT.main(arguments)
    assert events == [("numpy_seed", 321), ("train_model", 321)]
    manifest = json.loads((tmp_path / "reports/support-training.json").read_text())
    assert manifest["status"] == ("completed" if save_adapter else "failed")
    assert manifest["seed_controls"]["numpy_batch_sampler"] == 321
    assert manifest["experiment"] == "synthetic-support-native-experiment-v2"
    assert manifest["dataset"]["corpus_version"] == "synthetic-support-native-test-v2"
    assert manifest["dataset"]["corpus_manifest"]["corpus_version"] == (
        "synthetic-support-native-test-v2"
    )
    assert manifest["training_script_sha256"] == SCRIPT._sha256(Path(SCRIPT.__file__))
