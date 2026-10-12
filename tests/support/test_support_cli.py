from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.models.function import FunctionModel
from typer.testing import CliRunner

from agent_regression.cli import app
from agent_regression.storage import load_scenario
from agent_regression.support import cli as support_cli
from agent_regression.support import support_scenarios
from agent_regression.support.offline import make_offline_model

runner = CliRunner()


def test_chat_executes_local_refund_and_records_unreviewed_trace(tmp_path: Path) -> None:
    recording = tmp_path / "recording.json"
    result = runner.invoke(
        app,
        ["support", "chat", "Please refund ORD-1001.", "--json", "--record", str(recording)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model_mode"] == "offline_test_double"
    assert [call["tool_name"] for call in payload["calls"]] == ["lookup_order", "issue_refund"]
    assert len(payload["refunds"]) == 1
    assert payload["refunds"][0]["amount_cents"] == 4999
    scenario = load_scenario(recording)
    assert not scenario.reviewed and scenario.checks == []
    assert len(scenario.fixtures) == 2
    original = recording.read_bytes()
    second = runner.invoke(app, ["support", "chat", "Refund ORD-1001.", "--record", str(recording)])
    assert second.exit_code == 2
    assert recording.read_bytes() == original


def test_chat_broken_agent_is_denied_by_real_local_backend() -> None:
    result = runner.invoke(
        app,
        ["support", "chat", "Refund ORD-1002.", "--mutation", "refund_ineligible", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    refund = payload["calls"][-1]
    assert refund["tool_name"] == "issue_refund"
    assert refund["response"]["status"] == "denied"
    assert refund["response"]["reason_code"] == "outside_window"
    assert payload["refunds"] == []


@pytest.mark.parametrize(
    "prompt,tool_names,refunds,tickets",
    [
        ("Refund ORD-1001.", ["lookup_order", "issue_refund"], 1, 0),
        ("Refund ORD-1002.", ["lookup_order"], 0, 0),
        ("Refund ORD-9999.", ["lookup_order"], 0, 0),
        ("Refund ORD-4001.", ["lookup_order"], 0, 0),
        ("Refund ORD-1001 twice please.", ["lookup_order", "issue_refund"], 1, 0),
        ("Refund my order.", [], 0, 0),
        ("Refund ORD-1005.", ["lookup_order", "escalate_to_human"], 0, 1),
    ],
)
def test_chat_policy_branches(
    prompt: str, tool_names: list[str], refunds: int, tickets: int
) -> None:
    result = runner.invoke(app, ["support", "chat", prompt, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [call["tool_name"] for call in payload["calls"]] == tool_names
    assert len(payload["refunds"]) == refunds
    assert len(payload["escalations"]) == tickets


def test_chat_terminal_trace_and_custom_request_key(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "support",
            "chat",
            "Refund ORD-1001.",
            "--request-id",
            "retry-key-42",
            "--record",
            str(tmp_path / "run.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "retry-key-42" in result.output
    assert "Local ledgers: 1 refunds, 0 tickets" in result.output
    assert "Saved unreviewed recording" in result.output


def test_full_support_suite_repeats_and_reports(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    junit = tmp_path / "report.xml"
    result = runner.invoke(
        app,
        [
            "support",
            "demo",
            "--repeats",
            "2",
            "--json-report",
            str(report),
            "--junit-report",
            str(junit),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "44 runs: 44 passed, 0 failed" in result.output
    assert json.loads(report.read_text())["failed_runs"] == 0
    assert junit.exists()


def test_broken_suite_reports_forbidden_refund_without_configuration_error(tmp_path: Path) -> None:
    report = tmp_path / "broken.json"
    result = runner.invoke(
        app,
        [
            "support",
            "demo",
            "--mutation",
            "refund_ineligible",
            "--scenario",
            "outside_window",
            "--repeats",
            "3",
            "--json-report",
            str(report),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "issue_refund" in result.output
    assert "failed 3 of 3 runs" in result.output
    assert json.loads(report.read_text())["failed_runs"] == 3


def test_multiple_scenario_selection() -> None:
    result = runner.invoke(
        app, ["support", "demo", "--scenario", "eligible_order", "--scenario", "foreign_id"]
    )
    assert result.exit_code == 0, result.output
    assert "2 runs: 2 passed" in result.output


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["demo", "--scenario", "no_such_scenario"], "Unknown scenarios"),
        (["demo", "--mutation", "invalid"], "Unknown support mutation"),
        (["chat", "Refund ORD-1001.", "--mutation", "invalid"], "Unknown support mutation"),
        (["demo", "--base-url", "https://example.com/v1"], "loopback"),
        (["chat", "Hi", "--base-url", "https://example.com/v1"], "loopback"),
        (["chat", "Hi", "--request-id", "   "], "nonblank"),
        (
            ["demo", "--base-url", "http://127.0.0.1:8080/v1", "--mutation", "wrong_order_id"],
            "--mutation applies",
        ),
    ],
)
def test_support_configuration_errors(arguments: list[str], expected: str) -> None:
    result = runner.invoke(app, ["support", *arguments])
    assert result.exit_code == 2, result.output
    assert expected in result.output


def test_scenarios_export_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    directory = tmp_path / "scenarios"
    result = runner.invoke(app, ["support", "scenarios", "--export", str(directory)])
    assert result.exit_code == 0, result.output
    assert "22 reviewed synthetic scenarios" in result.output
    expected = {item.name: item for item in support_scenarios()}
    paths = sorted(directory.glob("*.json"))
    assert len(paths) == len(expected)
    for path in paths:
        observed = load_scenario(path)
        assert observed == expected[observed.name]
    original = {path: path.read_bytes() for path in paths}
    failed = runner.invoke(app, ["support", "scenarios", "--export", str(directory)])
    assert failed.exit_code == 2
    assert {path: path.read_bytes() for path in paths} == original


def test_chat_runtime_failure_exit_code_and_local_client_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_messages: object, _info: object) -> None:
        raise RuntimeError("simulated inference failure")

    model = FunctionModel(fail)  # type: ignore[arg-type]
    close = AsyncMock()
    monkeypatch.setattr(support_cli, "make_local_model", lambda **_kwargs: model)
    monkeypatch.setattr(support_cli, "close_local_model", close)
    result = runner.invoke(
        app, ["support", "chat", "Refund ORD-1001.", "--base-url", "http://127.0.0.1:8080/v1"]
    )
    assert result.exit_code == 1, result.output
    assert "simulated inference failure" in result.output
    close.assert_awaited_once_with(model)


def test_local_suite_client_cleanup_occurs_after_all_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_offline_model()
    close = AsyncMock()
    monkeypatch.setattr(support_cli, "make_local_model", lambda **_kwargs: model)
    monkeypatch.setattr(support_cli, "close_local_model", close)
    result = runner.invoke(
        app, ["support", "demo", "--base-url", "http://127.0.0.1:8080/v1", "--repeats", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "44 runs: 44 passed" in result.output
    close.assert_awaited_once_with(model)
