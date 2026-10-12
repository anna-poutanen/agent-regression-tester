"""Configuration-only adapter calibration never changes learned weights."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "support_calibration_script", PROJECT_ROOT / "scripts/calibrate_support_adapter.py"
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(SCRIPT, "PROJECT_ROOT", tmp_path)
    adapter = tmp_path / ".models/original"
    adapter.mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"synthetic test adapter weights\x00\xff")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "lora_parameters": {"scale": 20.0, "rank": 8, "dropout": 0.0},
                "num_layers": 8,
                "adapter_path": str(adapter),
                "seed": 20261011,
                "mask_prompt": True,
                "optimizer_config": {"adam": {}},
            },
            indent=2,
        )
    )
    (adapter / "000050_adapters.safetensors").write_bytes(b"unused older checkpoint")
    (adapter / "README.txt").write_text("not exported")
    return adapter


def run_calibration(source: Path, **kwargs: Any) -> dict[str, Any]:
    return SCRIPT.calibrate_adapter(
        source=source,
        output=source.parent / "calibrated",
        gain=kwargs.pop("gain", 0.25),
        manifest=SCRIPT.PROJECT_ROOT / "reports/calibration.json",
        **kwargs,
    )


def test_calibration_preserves_weights_source_and_all_other_configuration(source: Path) -> None:
    source_files = {p.name: p.read_bytes() for p in source.iterdir()}
    original = json.loads(source_files["adapter_config.json"])
    validation = SCRIPT.PROJECT_ROOT / "reports/validation.json"
    validation.write_text('{"split":"validation","measured":48}')
    provenance = run_calibration(source, validation_report=validation)
    output = source.parent / "calibrated"
    assert {p.name for p in output.iterdir()} == {"adapters.safetensors", "adapter_config.json"}
    assert (output / "adapters.safetensors").read_bytes() == source_files["adapters.safetensors"]
    expected = copy.deepcopy(original)
    expected["lora_parameters"]["scale"] = 5.0
    expected["adapter_path"] = str(output)
    assert json.loads((output / "adapter_config.json").read_text()) == expected
    assert {p.name: p.read_bytes() for p in source.iterdir()} == source_files
    assert provenance["source_weights_sha256"] == provenance["output_weights_sha256"]
    assert (
        provenance["source_config_sha256"]
        == hashlib.sha256(source_files["adapter_config.json"]).hexdigest()
    )
    assert (
        provenance["output_config_sha256"]
        == hashlib.sha256((output / "adapter_config.json").read_bytes()).hexdigest()
    )
    assert (
        provenance["validation_report"]["sha256"]
        == hashlib.sha256(validation.read_bytes()).hexdigest()
    )
    assert provenance["training_performed"] is False
    assert provenance["weights_modified"] is False
    assert provenance["gain"] == 0.25
    assert json.loads((SCRIPT.PROJECT_ROOT / "reports/calibration.json").read_text()) == provenance


@pytest.mark.parametrize("gain", [0, -1, 1.01, math.nan, math.inf, True, "0.25"])
def test_invalid_gains_write_nothing(source: Path, gain: Any) -> None:
    with pytest.raises(ValueError, match="Gain"):
        run_calibration(source, gain=gain)
    assert not (source.parent / "calibrated").exists()
    assert not (SCRIPT.PROJECT_ROOT / "reports/calibration.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fine_tune_type", "dora"),
        ("num_layers", 0),
        ("num_layers", True),
        ("num_layers", 8.0),
        ("rank", -8),
        ("rank", "8"),
        ("scale", 0),
        ("scale", math.inf),
        ("scale", True),
    ],
)
def test_invalid_configuration_rejected_before_publication(
    source: Path, field: str, value: Any
) -> None:
    path = source / "adapter_config.json"
    config = json.loads(path.read_text())
    target = config["lora_parameters"] if field in {"rank", "scale"} else config
    target[field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        run_calibration(source)
    assert not (source.parent / "calibrated").exists()


@pytest.mark.parametrize("existing", ["output", "manifest"])
def test_existing_destinations_are_preserved(source: Path, existing: str) -> None:
    path = (
        source.parent / "calibrated"
        if existing == "output"
        else SCRIPT.PROJECT_ROOT / "reports/calibration.json"
    )
    path.write_bytes(b"existing user artifact")
    with pytest.raises(ValueError, match="overwrite"):
        run_calibration(source)
    assert path.read_bytes() == b"existing user artifact"


@pytest.mark.parametrize("part", ["weights", "config", "source", "output", "manifest_parent"])
def test_symlink_inputs_and_destinations_rejected(source: Path, part: str) -> None:
    if part in {"weights", "config"}:
        name = "adapters.safetensors" if part == "weights" else "adapter_config.json"
        original = source / name
        saved = source / f"real-{name}"
        original.rename(saved)
        original.symlink_to(saved)
    elif part == "source":
        linked = source.parent / "source-link"
        linked.symlink_to(source, target_is_directory=True)
        source = linked
    elif part == "output":
        (source.parent / "calibrated").symlink_to(source.parent / "missing")
    else:
        reports = SCRIPT.PROJECT_ROOT / "reports"
        reports.rename(SCRIPT.PROJECT_ROOT / "real-reports")
        reports.symlink_to(SCRIPT.PROJECT_ROOT / "real-reports", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_calibration(source)


def test_empty_weights_rejected(source: Path) -> None:
    (source / "adapters.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        run_calibration(source)


def test_failed_manifest_publication_rolls_back_only_our_new_files(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_link = os.link
    manifest = SCRIPT.PROJECT_ROOT / "reports/calibration.json"

    def raced_link(staged: Path, target: Path) -> None:
        if target == manifest:
            manifest.write_text("concurrent user file")
        original_link(staged, target)

    monkeypatch.setattr(SCRIPT.os, "link", raced_link)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(FileExistsError):
        run_calibration(source)
    assert not (source.parent / "calibrated").exists()
    assert manifest.read_text() == "concurrent user file"
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert not list(source.parent.glob(".calibration-*"))
    assert not list(manifest.parent.glob(".calibration-*"))


def test_cli_uses_only_stdlib_and_does_not_add_absent_adapter_path(source: Path) -> None:
    path = source / "adapter_config.json"
    config = json.loads(path.read_text())
    del config["adapter_path"]
    path.write_text(json.dumps(config))
    assert (
        SCRIPT.main(
            [
                "--source",
                ".models/original",
                "--output",
                ".models/calibrated",
                "--gain",
                "1",
                "--manifest",
                "reports/calibration.json",
            ]
        )
        == 0
    )
    adjusted = json.loads((source.parent / "calibrated/adapter_config.json").read_text())
    assert "adapter_path" not in adjusted
    assert adjusted == config
