from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent_regression.support import make_offline_model


def load_script() -> object:
    path = Path(__file__).resolve().parents[2] / "scripts/evaluate_support_model.py"
    spec = importlib.util.spec_from_file_location("support_evaluation_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_evaluation_script_complete_reports_hashes_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script()
    model = make_offline_model()
    close = AsyncMock()
    monkeypatch.setattr(script, "make_local_model", lambda **_kwargs: model)
    monkeypatch.setattr(script, "close_local_model", close)
    output = tmp_path / "results.json"
    args = argparse.Namespace(
        base_url="http://127.0.0.1:8080/v1",
        model_name="default_model",
        label="offline-script-test",
        temperature=0.2,
        repeats=2,
        timeout=20,
        prompt_variant="safe",
        diagnostic=True,
        output=output,
    )
    status = await script.evaluate(args)  # type: ignore[attr-defined]
    assert status == 0
    report = json.loads(output.read_text())
    assert report["total_runs"] == 68
    assert report["failed_runs"] == 0
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["total_runs"] == 68
    assert manifest["completed_at"]
    assert len(manifest["actual_prompt_sha256"]) == 64
    assert len(manifest["tool_schemas_sha256"]) == 64
    assert len(manifest["tool_schema_order_sha256"]) == 64
    assert len(manifest["agent_source_sha256"]) == 64
    assert manifest["instruction_layout"] == "single_system_policy_and_context"
    assert len(manifest["scenarios_sha256"]) == 64
    assert output.with_suffix(".xml").exists()
    close.assert_awaited_once_with(model)


async def test_evaluation_script_preserves_regression_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script()
    model = make_offline_model(mutation="refund_ineligible")
    monkeypatch.setattr(script, "make_local_model", lambda **_kwargs: model)
    monkeypatch.setattr(script, "close_local_model", AsyncMock())
    output = tmp_path / "broken.json"
    args = argparse.Namespace(
        base_url="http://127.0.0.1:8080/v1",
        model_name="default_model",
        label="offline-script-mutation",
        temperature=0.2,
        repeats=1,
        timeout=20,
        prompt_variant="safe",
        diagnostic=False,
        output=output,
    )
    assert await script.evaluate(args) == 1  # type: ignore[attr-defined]
    report = json.loads(output.read_text())
    assert report["failed_runs"] > 0
    assert report["results"]


async def test_invalid_prompt_does_not_allocate_local_http_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script()
    constructed = []
    monkeypatch.setattr(script, "make_local_model", lambda **kwargs: constructed.append(kwargs))
    args = argparse.Namespace(
        base_url="http://127.0.0.1:8080/v1",
        model_name="default_model",
        label="invalid-prompt",
        temperature=0.2,
        repeats=1,
        timeout=20,
        prompt_variant="unknown",
        diagnostic=False,
        output=tmp_path / "report.json",
    )
    with pytest.raises(ValueError, match="Unknown support prompt variant"):
        await script.evaluate(args)  # type: ignore[attr-defined]
    assert constructed == []
