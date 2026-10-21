from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_regression import cli
from agent_regression.checkpoint import CheckpointStore, build_checkpoint, load_checkpoint
from agent_regression.demo import demo_scenarios
from agent_regression.schema import RunResult, SuiteResult
from agent_regression.storage import save_scenario

runner = CliRunner()
SAFE_AGENT = "agent_regression.demo:make_support_agent"
BROKEN_AGENT = "agent_regression.demo:make_broken_support_agent"


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    path = tmp_path / "scenarios" / "ineligible.json"
    save_scenario(demo_scenarios()[0], path)
    return path


def command(scenario_file: Path, *options: str, agent: str = SAFE_AGENT) -> list[str]:
    return ["run", str(scenario_file), "--agent", agent, *options]


def test_checkpoint_stops_resumes_without_duplicates_and_reports_progress(
    scenario_file: Path, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "runs" / "progress.json"
    json_report = tmp_path / "reports" / "results.json"
    junit_report = tmp_path / "reports" / "results.xml"
    options = [
        "--checkpoint",
        str(checkpoint),
        "--run-id",
        "prompt-v1/model-a/code-a",
        "--repeats",
        "3",
        "--json-report",
        str(json_report),
        "--junit-report",
        str(junit_report),
    ]
    initial = runner.invoke(cli.app, command(scenario_file, *options, "--max-runs", "1"))
    assert initial.exit_code == 3, initial.output
    state = load_checkpoint(checkpoint)
    assert state.status == "incomplete" and state.stop_reason == "max_runs"
    assert [result.iteration for result in state.results] == [1]
    payload = json.loads(json_report.read_text())
    assert payload["status"] == "incomplete"
    assert payload["planned_runs"] == 3
    assert payload["remaining_runs"] == 2
    assert "--resume" in initial.output
    assert ET.parse(junit_report).getroot().attrib["errors"] != "0"

    resumed = runner.invoke(cli.app, command(scenario_file, *options, "--resume"))
    assert resumed.exit_code == 0, resumed.output
    state = load_checkpoint(checkpoint)
    assert state.status == "completed" and state.stop_reason is None
    assert [result.iteration for result in state.results] == [1, 2, 3]
    assert json.loads(json_report.read_text())["remaining_runs"] == 0
    assert ET.parse(junit_report).getroot().attrib["errors"] == "0"


def test_completed_checkpoint_preserves_regressions(scenario_file: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "progress.json"
    result = runner.invoke(
        cli.app,
        command(
            scenario_file,
            "--checkpoint",
            str(checkpoint),
            "--run-id",
            "bad-prompt-v1",
            agent=BROKEN_AGENT,
        ),
    )
    assert result.exit_code == 1, result.output
    assert load_checkpoint(checkpoint).results[0].passed is False


def test_resume_binds_user_identity_and_exact_agent_import_target(
    scenario_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "progress.json"
    options = ["--checkpoint", str(checkpoint), "--run-id", "version-1", "--repeats", "2"]
    initial = runner.invoke(cli.app, command(scenario_file, *options, "--max-runs", "1"))
    assert initial.exit_code == 3, initial.output
    identity = json.loads(load_checkpoint(checkpoint).run_id)
    assert identity == {"user_run_id": "version-1", "agent": SAFE_AGENT}
    unchanged = checkpoint.read_bytes()
    invoked: list[bool] = []

    def forbidden_factory(*args, **kwargs):
        invoked.append(True)
        raise AssertionError("A mismatched resume must not construct the agent")

    monkeypatch.setattr(cli, "_instantiate", forbidden_factory)
    switched = runner.invoke(
        cli.app, command(scenario_file, *options, "--resume", agent=BROKEN_AGENT)
    )
    assert switched.exit_code == 2, switched.output
    assert "mismatch" in switched.output.lower()
    assert checkpoint.read_bytes() == unchanged
    assert not invoked


@pytest.mark.parametrize("malformed", [False, True])
def test_existing_or_malformed_checkpoint_is_not_used_without_a_valid_resume(
    scenario_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: bool,
) -> None:
    checkpoint = tmp_path / "progress.json"
    if malformed:
        checkpoint.write_text('{"schema_version": 999}')
    else:
        checkpoint.write_text("existing checkpoint must not be overwritten")
    original = checkpoint.read_bytes()
    invoked: list[bool] = []

    def forbidden_factory(*args, **kwargs):
        invoked.append(True)
        raise AssertionError("Invalid checkpoint must not construct the agent")

    monkeypatch.setattr(cli, "_instantiate", forbidden_factory)
    options = ["--checkpoint", str(checkpoint), "--run-id", "v1"]
    if malformed:
        options.append("--resume")
    result = runner.invoke(cli.app, command(scenario_file, *options))
    assert result.exit_code == 2, result.output
    assert checkpoint.read_bytes() == original
    assert not invoked


@pytest.mark.parametrize(
    "options, fragment",
    [
        (["--resume"], "requires --checkpoint"),
        (["--run-id", "v1"], "requires --checkpoint"),
        (["--checkpoint", "STATE"], "requires a nonblank --run-id"),
        (["--checkpoint", "STATE", "--run-id", " "], "requires a nonblank --run-id"),
        (["--max-runs", "1", "--concurrency", "2"], "concurrency 1"),
        (["--pause-seconds", "0.1"], "requires --batch-size"),
        (["--max-seconds", "nan"], "finite and positive"),
        (["--max-seconds", "inf"], "finite and positive"),
        (["--max-seconds", "0"], "finite and positive"),
        (["--pause-seconds", "nan", "--batch-size", "1"], "finite and nonnegative"),
        (["--pause-seconds", "inf", "--batch-size", "1"], "finite and nonnegative"),
        (["--pause-seconds", "-1", "--batch-size", "1"], "finite and nonnegative"),
        (["--timeout", "inf"], "finite and positive"),
    ],
)
def test_invalid_controls_fail_before_agent_import(
    scenario_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    fragment: str,
) -> None:
    imported: list[bool] = []

    def load(*args, **kwargs):
        imported.append(True)
        raise AssertionError("Invalid controls must not import application code")

    monkeypatch.setattr(cli, "_load_agent_target", load)
    options = [str(tmp_path / "progress.json") if value == "STATE" else value for value in options]
    result = runner.invoke(cli.app, command(scenario_file, *options))
    assert result.exit_code == 2, result.output
    assert fragment in result.output
    assert not imported


@pytest.mark.parametrize("options", [["--max-runs", "0"], ["--batch-size", "0"]])
def test_nonpositive_counts_are_usage_errors(scenario_file: Path, options: list[str]) -> None:
    result = runner.invoke(cli.app, command(scenario_file, *options))
    assert result.exit_code == 2


def test_budget_without_checkpoint_is_incomplete_and_zero_results_are_not_config_error(
    scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def suite(*args, **kwargs):
        captured.update(kwargs)
        return SuiteResult(
            planned_runs=2, status="incomplete", stop_reason="max_seconds", results=[]
        )

    monkeypatch.setattr(cli, "run_suite_sync", suite)
    result = runner.invoke(
        cli.app, command(scenario_file, "--repeats", "2", "--max-seconds", "0.001")
    )
    assert result.exit_code == 3, result.output
    assert captured["max_seconds"] == 0.001
    assert "PASS:" not in result.output


def test_default_run_does_not_forward_new_control_keywords(
    scenario_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def legacy_suite(target, scenarios, *, repeats, concurrency, timeout, allow_unreviewed):
        return SuiteResult(results=[RunResult(scenario_name=scenarios[0].name)])

    monkeypatch.setattr(cli, "run_suite_sync", legacy_suite)
    result = runner.invoke(cli.app, command(scenario_file))
    assert result.exit_code == 0, result.output


def test_forward_controls_and_bind_identity(
    scenario_file: Path, tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}

    def suite(*args, **kwargs):
        captured.update(kwargs)
        return SuiteResult(results=[RunResult(scenario_name="ineligible_order")])

    monkeypatch.setattr(cli, "run_suite_sync", suite)
    checkpoint = tmp_path / "progress.json"
    result = runner.invoke(
        cli.app,
        command(
            scenario_file,
            "--checkpoint",
            str(checkpoint),
            "--run-id",
            "v1",
            "--resume",
            "--max-runs",
            "2",
            "--max-seconds",
            "10.5",
            "--batch-size",
            "1",
            "--pause-seconds",
            "0.1",
        ),
    )
    assert result.exit_code == 0, result.output
    assert captured["checkpoint"] == checkpoint
    assert captured["resume"] is True
    assert captured["max_runs"] == 2
    assert captured["max_seconds"] == 10.5
    assert captured["batch_size"] == 1
    assert captured["pause_seconds"] == 0.1
    assert json.loads(captured["run_id"]) == {"agent": SAFE_AGENT, "user_run_id": "v1"}


@pytest.mark.parametrize("report_flag", ["--json-report", "--junit-report"])
def test_report_cannot_replace_scenario(
    scenario_file: Path, monkeypatch: pytest.MonkeyPatch, report_flag: str
) -> None:
    original = scenario_file.read_bytes()
    imported: list[bool] = []
    monkeypatch.setattr(cli, "_load_agent_target", lambda *args, **kwargs: imported.append(True))
    result = runner.invoke(cli.app, command(scenario_file, report_flag, str(scenario_file)))
    assert result.exit_code == 2
    assert "alias a scenario" in result.output
    assert scenario_file.read_bytes() == original
    assert not imported


@pytest.mark.parametrize("report_flag", ["--json-report", "--junit-report"])
@pytest.mark.parametrize("alias", ["checkpoint", "lock"])
def test_reports_cannot_replace_checkpoint_or_lock(
    scenario_file: Path, tmp_path: Path, report_flag: str, alias: str
) -> None:
    checkpoint = tmp_path / "progress.json"
    output = checkpoint if alias == "checkpoint" else checkpoint.with_suffix(".json.lock")
    result = runner.invoke(
        cli.app,
        command(
            scenario_file,
            "--checkpoint",
            str(checkpoint),
            "--run-id",
            "v1",
            report_flag,
            str(output),
        ),
    )
    assert result.exit_code == 2
    assert "alias a checkpoint" in result.output
    assert not checkpoint.exists()


def test_report_aliases_include_hardlinks_and_symlinks(scenario_file: Path, tmp_path: Path) -> None:
    original = scenario_file.read_bytes()
    for name, link in (("hardlink", os.link), ("symlink", os.symlink)):
        path = tmp_path / name
        link(scenario_file, path)
        result = runner.invoke(cli.app, command(scenario_file, "--json-report", str(path)))
        assert result.exit_code == 2, result.output
        assert "alias a scenario" in result.output
        assert scenario_file.read_bytes() == original


def test_json_and_junit_reports_must_differ(scenario_file: Path, tmp_path: Path) -> None:
    path = tmp_path / "same_report"
    result = runner.invoke(
        cli.app, command(scenario_file, "--json-report", str(path), "--junit-report", str(path))
    )
    assert result.exit_code == 2
    assert "must differ" in result.output
    assert not path.exists()


def test_json_and_junit_report_aliases_are_rejected(scenario_file: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.write_text("preserve existing report")
    second = tmp_path / "second"
    os.link(first, second)
    result = runner.invoke(
        cli.app,
        command(scenario_file, "--json-report", str(first), "--junit-report", str(second)),
    )
    assert result.exit_code == 2
    assert "must differ" in result.output
    assert first.read_text() == "preserve existing report"


def test_checkpoint_alias_protection_uses_expanded_home_paths(
    scenario_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "progress.json"
    expanduser = Path.expanduser

    def expanded(path):
        return checkpoint if str(path) == "~/progress.json" else expanduser(path)

    monkeypatch.setattr(Path, "expanduser", expanded)
    result = runner.invoke(
        cli.app,
        command(
            scenario_file,
            "--checkpoint",
            "~/progress.json",
            "--run-id",
            "v1",
            "--json-report",
            str(checkpoint),
        ),
    )
    assert result.exit_code == 2
    assert "alias a checkpoint" in result.output
    assert not checkpoint.exists()


def test_checkpoint_cannot_replace_scenario_or_its_lock_alias(
    scenario_file: Path, tmp_path: Path
) -> None:
    original = scenario_file.read_bytes()
    result = runner.invoke(
        cli.app, command(scenario_file, "--checkpoint", str(scenario_file), "--run-id", "v1")
    )
    assert result.exit_code == 2
    assert "alias a scenario" in result.output
    checkpoint = tmp_path / "progress.json"
    os.link(scenario_file, checkpoint.with_suffix(".json.lock"))
    result = runner.invoke(
        cli.app, command(scenario_file, "--checkpoint", str(checkpoint), "--run-id", "v1")
    )
    assert result.exit_code == 2
    assert "alias a scenario" in result.output
    assert scenario_file.read_bytes() == original


@pytest.mark.parametrize("flag", ["--checkpoint", "--json-report"])
def test_json_artifacts_must_stay_outside_recursive_scenario_directory(
    scenario_file: Path, flag: str
) -> None:
    output = scenario_file.parent / "generated.json"
    options = [flag, str(output)]
    if flag == "--checkpoint":
        options += ["--run-id", "v1"]
    result = runner.invoke(cli.app, command(scenario_file.parent, *options))
    assert result.exit_code == 2
    assert "outside scenario directories" in result.output
    assert not output.exists()


def test_ctrl_c_recovers_only_matching_validated_checkpoint_and_exits_130(
    scenario_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "progress.json"
    report = tmp_path / "report.json"

    def interrupted(target, scenarios, **kwargs):
        state = build_checkpoint(
            scenarios, run_id=kwargs["run_id"], repeats=kwargs["repeats"], timeout=kwargs["timeout"]
        )
        state.results = [RunResult(scenario_name=scenarios[0].name)]
        state.stop_reason = "cancelled"
        with CheckpointStore(checkpoint) as store:
            store.create(state)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_suite_sync", interrupted)
    result = runner.invoke(
        cli.app,
        command(
            scenario_file,
            "--checkpoint",
            str(checkpoint),
            "--run-id",
            "v1",
            "--repeats",
            "2",
            "--json-report",
            str(report),
        ),
    )
    assert result.exit_code == 130, result.output
    assert "Interrupted" in result.output
    assert "--resume" in result.output
    assert json.loads(report.read_text())["status"] == "incomplete"
    assert json.loads(report.read_text())["total_runs"] == 1


def test_ctrl_c_does_not_present_unrelated_checkpoint_as_current_progress(
    scenario_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "progress.json"
    report = tmp_path / "report.json"
    state = build_checkpoint(demo_scenarios()[:1], run_id="different-identity", repeats=2)
    with CheckpointStore(checkpoint) as store:
        store.create(state)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_suite_sync", interrupted)
    result = runner.invoke(
        cli.app,
        command(
            scenario_file,
            "--checkpoint",
            str(checkpoint),
            "--run-id",
            "v1",
            "--repeats",
            "2",
            "--json-report",
            str(report),
        ),
    )
    assert result.exit_code == 130
    assert "mismatch" in result.output.lower()
    assert not report.exists()


def test_ctrl_c_without_checkpoint_exits_130(scenario_file: Path, monkeypatch) -> None:
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_suite_sync", interrupted)
    result = runner.invoke(cli.app, command(scenario_file))
    assert result.exit_code == 130
    assert "Interrupted" in result.output
