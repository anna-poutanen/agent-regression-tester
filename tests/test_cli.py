from __future__ import annotations

import json
import os
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_regression import cli
from agent_regression.demo import demo_scenarios, make_support_agent
from agent_regression.schema import Scenario, SuiteResult
from agent_regression.storage import load_scenario, save_scenario

runner = CliRunner()


def scenario_path(tmp_path: Path, *, reviewed: bool = True) -> Path:
    scenario = demo_scenarios()[0]
    scenario.reviewed = reviewed
    path = tmp_path / "scenario.json"
    save_scenario(scenario, path)
    return path


def test_demo_safe_and_broken_reports(tmp_path: Path) -> None:
    safe = runner.invoke(cli.app, ["demo", "--repeats", "3"])
    assert safe.exit_code == 0, safe.output
    assert "6 runs: 6 passed, 0 failed" in safe.output
    report_json = tmp_path / "report.json"
    report_xml = tmp_path / "report.xml"
    broken = runner.invoke(
        cli.app,
        [
            "demo",
            "--broken",
            "--repeats",
            "2",
            "--json-report",
            str(report_json),
            "--junit-report",
            str(report_xml),
        ],
    )
    assert broken.exit_code == 1, broken.output
    assert "issue_refund" in broken.output
    assert "failed 2 of 2 runs" in broken.output
    assert json.loads(report_json.read_text())["failed_runs"] == 2
    assert ET.parse(report_xml).getroot().attrib["failures"] == "2"


def test_record_suggest_review_run_workflow(tmp_path: Path) -> None:
    path = tmp_path / "recording.json"
    recorded = runner.invoke(
        cli.app,
        [
            "record",
            "--agent",
            "agent_regression.demo:make_support_agent",
            "--prompt",
            "Refund order 123 if eligible.",
            "--name",
            "recorded_order",
            "--output",
            str(path),
        ],
    )
    assert recorded.exit_code == 0, recorded.output
    assert "executes its configured tools" in recorded.output
    scenario = load_scenario(path)
    assert not scenario.reviewed
    assert scenario.recorded_calls[0].tool_name == "lookup_order"
    assert scenario.checks == []
    assert runner.invoke(cli.app, ["validate", str(path)]).exit_code == 2
    proposal = runner.invoke(cli.app, ["suggest", str(path), "--write"])
    assert proposal.exit_code == 0, proposal.output
    assert load_scenario(path).checks
    assert not load_scenario(path).reviewed
    assert runner.invoke(cli.app, ["validate", str(path)]).exit_code == 0
    unreviewed = runner.invoke(
        cli.app, ["run", str(path), "--agent", "agent_regression.demo:make_support_agent"]
    )
    assert unreviewed.exit_code == 2
    assert "unreviewed" in unreviewed.output
    reviewed = runner.invoke(cli.app, ["review", str(path), "--yes"])
    assert reviewed.exit_code == 0, reviewed.output
    assert load_scenario(path).reviewed
    tested = runner.invoke(
        cli.app,
        ["run", str(path), "--agent", "agent_regression.demo:make_support_agent", "--repeats", "2"],
    )
    assert tested.exit_code == 0, tested.output
    assert "2 runs: 2 passed, 0 failed" in tested.output


def test_suggest_dry_run_preserves_original_and_write_resets_review(tmp_path: Path) -> None:
    path = scenario_path(tmp_path)
    original = path.read_bytes()
    result = runner.invoke(cli.app, ["suggest", str(path)])
    assert result.exit_code == 0
    assert "Proposal only" in result.output
    assert path.read_bytes() == original
    result = runner.invoke(cli.app, ["suggest", str(path), "--write"])
    assert result.exit_code == 0
    assert not load_scenario(path).reviewed


def test_review_declined_is_unchanged_and_empty_checks_are_rejected(tmp_path: Path) -> None:
    path = scenario_path(tmp_path, reviewed=False)
    original = path.read_bytes()
    result = runner.invoke(cli.app, ["review", str(path)], input="n\n")
    assert result.exit_code == 0
    assert path.read_bytes() == original
    empty = tmp_path / "empty.json"
    save_scenario(Scenario(name="empty", prompt="hello"), empty)
    result = runner.invoke(cli.app, ["review", str(empty), "--yes"])
    assert result.exit_code == 2
    assert "without checks" in result.output


def test_run_draft_requires_explicit_flag(tmp_path: Path) -> None:
    path = scenario_path(tmp_path, reviewed=False)
    result = runner.invoke(
        cli.app,
        [
            "run",
            str(path),
            "--agent",
            "agent_regression.demo:make_support_agent",
            "--allow-unreviewed",
        ],
    )
    assert result.exit_code == 0, result.output


def test_run_directory_recurses_and_deduplicates_paths(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    path = scenario_path(nested)
    result = runner.invoke(
        cli.app,
        ["run", str(tmp_path), str(path), "--agent", "agent_regression.demo:make_support_agent"],
    )
    assert result.exit_code == 0, result.output
    assert "1 runs: 1 passed" in result.output


def test_duplicate_names_are_configuration_errors(tmp_path: Path) -> None:
    scenario_path(tmp_path / "one")
    scenario_path(tmp_path / "two")
    result = runner.invoke(
        cli.app, ["run", str(tmp_path), "--agent", "agent_regression.demo:make_support_agent"]
    )
    assert result.exit_code == 2
    assert "Duplicate scenario name" in result.output


@pytest.mark.parametrize("command", ["validate", "suggest", "review"])
def test_malformed_artifacts_are_configuration_errors(tmp_path: Path, command: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json}")
    result = runner.invoke(cli.app, [command, str(path)])
    assert result.exit_code == 2
    assert "Error:" in result.output


def test_missing_empty_and_invalid_agent_configuration(tmp_path: Path) -> None:
    path = scenario_path(tmp_path)
    result = runner.invoke(cli.app, ["run", str(path), "--agent", "not-a-spec"])
    assert result.exit_code == 2
    assert "module:factory" in result.output
    result = runner.invoke(cli.app, ["run", str(path), "--agent", "no_such_module:factory"])
    assert result.exit_code == 2
    assert "Cannot import" in result.output
    empty_directory = tmp_path / "nothing"
    empty_directory.mkdir()
    result = runner.invoke(cli.app, ["run", str(empty_directory), "--agent", "unused:factory"])
    assert result.exit_code == 2
    assert "No scenario JSON files" in result.output
    result = runner.invoke(
        cli.app, ["run", str(path), "--agent", "unused:factory", "--repeats", "0"]
    )
    assert result.exit_code == 2


def test_factory_validation_reuses_first_instance_and_constructs_each_repeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    invocations: list[int] = []
    module = types.ModuleType("local_cli_agents")

    def factory():
        invocations.append(1)
        return make_support_agent()

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = scenario_path(tmp_path)
    result = runner.invoke(
        cli.app,
        ["run", str(path), "--agent", "local_cli_agents:factory", "--repeats", "3"],
    )
    assert result.exit_code == 0, result.output
    assert len(invocations) == 3


def test_loader_accepts_agent_instances_and_rejects_wrong_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("local_cli_agents")
    module.agent = make_support_agent()
    module.not_callable = 7
    module.bad_factory = lambda: "not an agent"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert cli._load_agent_target("local_cli_agents:agent") is module.agent
    with pytest.raises(ValueError, match="neither an Agent nor a factory"):
        cli._load_agent_target("local_cli_agents:not_callable")
    with pytest.raises(ValueError, match="expected a pydantic_ai.Agent"):
        cli._load_agent_target("local_cli_agents:bad_factory")
    with pytest.raises(ValueError, match="Cannot import"):
        cli._load_agent_target("local_cli_agents:no_attribute")


def test_async_cli_factory_gives_useful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("local_cli_agents")

    async def factory():
        return make_support_agent()

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="must be synchronous"):
        cli._load_agent_target("local_cli_agents:factory")


def test_import_messages_and_refuse_accidental_overwrite(tmp_path: Path) -> None:
    result = make_support_agent().run_sync("Refund order 123 if eligible.")
    messages_path = tmp_path / "messages.json"
    messages_path.write_bytes(result.all_messages_json())
    output_path = tmp_path / "imported.json"
    imported = runner.invoke(
        cli.app,
        [
            "record",
            "--messages",
            str(messages_path),
            "--prompt",
            "Refund order 123 if eligible.",
            "--name",
            "imported",
            "--output",
            str(output_path),
        ],
    )
    assert imported.exit_code == 0, imported.output
    assert "executes its configured tools" not in imported.output
    assert load_scenario(output_path).recorded_calls[0].tool_name == "lookup_order"
    original = output_path.read_bytes()
    denied = runner.invoke(
        cli.app,
        [
            "record",
            "--agent",
            "unused:factory",
            "--prompt",
            "x",
            "--name",
            "x",
            "--output",
            str(output_path),
        ],
    )
    assert denied.exit_code == 2
    assert "already exists" in denied.output
    assert output_path.read_bytes() == original


def test_record_requires_exactly_one_source(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["record", "--prompt", "x", "--name", "x", "--output", str(tmp_path / "x.json")],
    )
    assert result.exit_code == 2
    assert "exactly one" in result.output


def test_empty_suite_cannot_return_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = scenario_path(tmp_path)
    monkeypatch.setattr(cli, "run_suite_sync", lambda *args, **kwargs: SuiteResult())
    result = runner.invoke(
        cli.app, ["run", str(path), "--agent", "agent_regression.demo:make_support_agent"]
    )
    assert result.exit_code == 2
    assert "No scenarios ran" in result.output


def test_help_documents_workflow() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ["record", "suggest", "review", "validate", "run", "demo"]:
        assert command in result.output


def test_cli_suppresses_framework_banner_without_overriding_user_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYDANTIC_AI_NO_BANNER", raising=False)
    result = runner.invoke(cli.app, ["demo"])
    assert result.exit_code == 0, result.output
    assert os.environ["PYDANTIC_AI_NO_BANNER"] == "1"
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "0")
    result = runner.invoke(cli.app, ["demo"])
    assert result.exit_code == 0, result.output
    assert os.environ["PYDANTIC_AI_NO_BANNER"] == "0"


def test_import_external_developer_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "developer_agent.py").write_text(
        "from agent_regression.demo import make_support_agent\n"
        "def agent_factory():\n"
        "    return make_support_agent()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    path = scenario_path(tmp_path)
    result = runner.invoke(cli.app, ["run", str(path), "--agent", "developer_agent:agent_factory"])
    assert result.exit_code == 0, result.output
    assert "PASS: ineligible_order" in result.output


def test_invalid_scenarios_fail_before_agent_code_or_model_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory_calls: list[bool] = []
    module = types.ModuleType("config_agent")

    def factory():
        factory_calls.append(True)
        raise AssertionError("Invalid configuration must not construct the agent")

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = tmp_path / "empty_checks.json"
    save_scenario(Scenario(name="empty", prompt="hello", reviewed=True), path)
    result = runner.invoke(cli.app, ["run", str(path), "--agent", "config_agent:factory"])
    assert result.exit_code == 2
    assert "no checks" in result.output
    assert not factory_calls


def test_record_rejects_invalid_name_before_running_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    imported: list[bool] = []

    def load_target(specification):
        imported.append(True)
        raise AssertionError("Invalid recording metadata must not import application code")

    monkeypatch.setattr(cli, "_load_agent_target", load_target)
    result = runner.invoke(
        cli.app,
        [
            "record",
            "--agent",
            "unused:factory",
            "--prompt",
            "hello",
            "--name",
            "",
            "--output",
            str(tmp_path / "bad.json"),
        ],
    )
    assert result.exit_code == 2
    assert not imported
