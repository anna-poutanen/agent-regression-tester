#!/usr/bin/env python3
"""Copy a local LoRA adapter with a smaller inference scale, without training.

Only the adapter weights and adjusted configuration are exported. All operations
use the Python standard library; this script never loads or executes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _local_path(value: str | Path, area: str) -> Path:
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError("Calibration paths cannot contain parent traversal")
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    base = PROJECT_ROOT / area
    if not path.is_relative_to(base):
        raise ValueError(f"Calibration path must stay inside project {area}: {path}")
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"Calibration paths cannot use symlinks: {candidate}")
        if candidate == PROJECT_ROOT:
            break
    return path


def _regular_file(path: Path) -> None:
    if not path.exists() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Required regular local file does not exist: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _remove_owned(path: Path, identity: tuple[int, int]) -> None:
    """Rollback our own inode without deleting a concurrently replaced file."""
    try:
        current = path.lstat()
        if (current.st_dev, current.st_ino) == identity:
            path.unlink()
    except FileNotFoundError:
        pass


def calibrate_adapter(
    *,
    source: str | Path,
    output: str | Path,
    gain: float,
    manifest: str | Path,
    validation_report: str | Path | None = None,
) -> dict[str, Any]:
    """Publish a fresh scale-adjusted adapter and its provenance manifest."""
    gain = _number(gain, "Gain")
    if gain > 1:
        raise ValueError("Gain must be at most 1")
    source_path = _local_path(source, ".models")
    output_path = _local_path(output, ".models")
    manifest_path = _local_path(manifest, "reports")
    if not source_path.is_dir():
        raise ValueError("Source must be an existing local adapter directory")
    if output_path.is_relative_to(source_path):
        raise ValueError("Output must be separate from the source adapter directory")
    for path in (output_path, manifest_path):
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing calibration output: {path}")
        if not path.parent.is_dir():
            raise ValueError(f"Calibration output parent must already exist: {path.parent}")

    weights = _local_path(source_path / "adapters.safetensors", ".models")
    config_path = _local_path(source_path / "adapter_config.json", ".models")
    for path in (weights, config_path):
        _regular_file(path)
    if weights.stat().st_size == 0:
        raise ValueError("Source adapter weights cannot be empty")
    source_config_bytes = config_path.read_bytes()
    config = json.loads(source_config_bytes)
    if not isinstance(config, dict) or config.get("fine_tune_type") != "lora":
        raise ValueError("Source configuration must have fine_tune_type=lora")
    parameters = config.get("lora_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Source configuration requires lora_parameters")
    original_scale = _number(parameters.get("scale"), "LoRA scale")
    for name, value in (("rank", parameters.get("rank")), ("num_layers", config.get("num_layers"))):
        if type(value) is not int or value <= 0:
            raise ValueError(f"LoRA {name} must be a positive integer")
    effective_scale = _number(original_scale * gain, "Effective LoRA scale")
    parameters["scale"] = effective_scale
    if "adapter_path" in config:
        config["adapter_path"] = str(output_path)
    # Reject nonfinite values elsewhere in an otherwise valid configuration too.
    config_bytes = (json.dumps(config, indent=2, allow_nan=False) + "\n").encode("utf-8")
    source_weights_hash = _sha256(weights)
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "operation": "config_only_inference_scale_calibration",
        "training_performed": False,
        "weights_modified": False,
        "created_at": datetime.now(UTC).isoformat(),
        "source_adapter": str(source_path),
        "output_adapter": str(output_path),
        "source_weights_sha256": source_weights_hash,
        "source_config_sha256": hashlib.sha256(source_config_bytes).hexdigest(),
        "output_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "original_scale": original_scale,
        "effective_scale": effective_scale,
        "gain": gain,
        "rank": parameters["rank"],
        "num_layers": config["num_layers"],
    }
    if validation_report is not None:
        report = _local_path(validation_report, "reports")
        _regular_file(report)
        report_bytes = report.read_bytes()
        json.loads(report_bytes)
        provenance["validation_report"] = {
            "path": str(report),
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
        }

    created_directory = False
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        with (
            TemporaryDirectory(prefix=".calibration-", dir=output_path.parent) as staging,
            TemporaryDirectory(prefix=".calibration-", dir=manifest_path.parent) as report_staging,
        ):
            staging_path = Path(staging)
            staged_weights = staging_path / "adapters.safetensors"
            shutil.copyfile(weights, staged_weights)
            copied_hash = _sha256(staged_weights)
            if copied_hash != source_weights_hash or _sha256(weights) != source_weights_hash:
                raise ValueError("Source adapter weights changed while copying")
            if config_path.read_bytes() != source_config_bytes:
                raise ValueError("Source adapter configuration changed while copying")
            provenance["output_weights_sha256"] = copied_hash
            staged_config = staging_path / "adapter_config.json"
            staged_config.write_bytes(config_bytes)
            staged_manifest = Path(report_staging) / "manifest.json"
            staged_manifest.write_text(
                json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            # Exclusive mkdir reserves the fresh directory. Hard links publish
            # each complete file atomically and refuse raced existing targets.
            _local_path(output_path, ".models")
            _local_path(manifest_path, "reports")
            output_path.mkdir()
            created_directory = True
            for staged, target in (
                (staged_weights, output_path / "adapters.safetensors"),
                (staged_config, output_path / "adapter_config.json"),
                (staged_manifest, manifest_path),
            ):
                os.link(staged, target)
                staged_stat = staged.stat()
                published.append((target, (staged_stat.st_dev, staged_stat.st_ino)))
    except BaseException:
        for path, identity in reversed(published):
            _remove_owned(path, identity)
        if created_directory:
            try:
                output_path.rmdir()
            except (FileNotFoundError, OSError):
                pass
        raise
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gain", required=True, type=float)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-report")
    args = parser.parse_args(argv)
    provenance = calibrate_adapter(
        source=args.source,
        output=args.output,
        gain=args.gain,
        manifest=args.manifest,
        validation_report=args.validation_report,
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
