from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


@pytest.mark.parametrize("targeted_only", [False, True])
async def test_intent_case_selection_and_report_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    targeted_only: bool,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from agent_regression.support.targeted import targeted_scenarios

    script = load_script()
    model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("Which order?")]))
    close = AsyncMock()
    monkeypatch.setattr(script, "make_local_model", lambda **_kwargs: model)
    monkeypatch.setattr(script, "close_local_model", close)
    args = argparse.Namespace(
        base_url="http://127.0.0.1:8080/v1",
        model_name="default_model",
        label="intent-case-selection-test",
        temperature=0.2,
        repeats=1,
        timeout=20,
        prompt_variant="safe",
        diagnostic=False,
        targeted=True,
        targeted_only=targeted_only,
        output=tmp_path / "results.json",
    )
    # A clarification-only model must fail the full-refund controls. This test
    # verifies suite selection and provenance, rather than blessing its behavior.
    assert await script.evaluate(args) == 1  # type: ignore[attr-defined]
    report = json.loads(args.output.read_text())
    names = {run["scenario_name"] for run in report["results"]}
    targeted_names = {scenario.name for scenario in targeted_scenarios()}
    assert targeted_names <= names
    if targeted_only:
        assert "eligible_order" not in names
    else:
        assert "eligible_order" in names
    assert report["total_runs"] == len(targeted_names) + (0 if targeted_only else 22)
    manifest = json.loads(args.output.with_suffix(".manifest.json").read_text())
    assert set(manifest["scenario_names"]) == names
    close.assert_awaited_once_with(model)


def test_targeted_only_rejects_a_conflicting_diagnostic_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_support_model.py",
            "--label",
            "conflicting-selection",
            "--output",
            str(tmp_path / "results.json"),
            "--targeted-only",
            "--diagnostic",
        ],
    )
    with pytest.raises(SystemExit) as error:
        script.main()  # type: ignore[attr-defined]
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())
