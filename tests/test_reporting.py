from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from rich.console import Console

from agent_regression.reporting import render_terminal, write_json_report, write_junit_report
from agent_regression.schema import (
    CheckFailure,
    RunResult,
    SuiteResult,
    ToolCall,
    TraceDifference,
)


def failing_run(iteration: int = 1) -> RunResult:
    return RunResult(
        scenario_name="ineligible_order",
        iteration=iteration,
        calls=[ToolCall(tool_name="issue_refund", arguments={"order_id": "123"})],
        failures=[
            CheckFailure(
                code="forbidden_call",
                message="Forbidden call observed",
                tool_name="issue_refund",
                expected="no issue_refund call",
                observed={"order_id": "123"},
            )
        ],
        duration_seconds=0.125,
    )


def rendered(suite: SuiteResult) -> str:
    buffer = io.StringIO()
    render_terminal(suite, Console(file=buffer, width=160, color_system=None))
    return buffer.getvalue()


def test_terminal_explains_failures_and_repeat_frequency() -> None:
    suite = SuiteResult(
        results=[
            failing_run(1),
            RunResult(scenario_name="ineligible_order", iteration=2),
            failing_run(3),
        ]
    )
    report = rendered(suite)
    assert "FAIL: ineligible_order — failed 2 of 3 runs" in report
    assert "Expected:" in report
    assert '"no issue_refund call"' in report
    assert 'Observed: {"order_id": "123"}' in report
    assert "3 runs: 1 passed, 2 failed" in report


def test_terminal_escapes_rich_markup_and_displays_replay_and_run_errors() -> None:
    result = RunResult(
        scenario_name="[bold]scenario[/bold]",
        replay_errors=["No saved response for [lookup]"],
        error="Factory <error>",
        differences=[TraceDifference(kind="added", message="Added [refund]")],
    )
    report = rendered(SuiteResult(results=[result]))
    assert "[bold]scenario[/bold]" in report
    assert "No saved response for [lookup]" in report
    assert "Factory <error>" in report
    assert "Added [refund]" in report


def test_terminal_does_not_claim_an_empty_suite_passed() -> None:
    report = rendered(SuiteResult())
    assert "No scenarios ran" in report
    assert "PASS" not in report


def test_json_report_has_consistent_totals_complete_traces_and_unicode(tmp_path: Path) -> None:
    suite = SuiteResult(results=[failing_run(), RunResult(scenario_name="café", output="👍")])
    path = tmp_path / "nested" / "report.json"
    write_json_report(suite, path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["total_runs"] == 2
    assert payload["passed_runs"] == payload["failed_runs"] == 1
    assert payload["results"][0]["calls"][0]["arguments"] == {"order_id": "123"}
    assert payload["results"][0]["passed"] is False
    assert payload["results"][1]["output"] == "👍"


def test_junit_xml_totals_escaping_and_diagnostics(tmp_path: Path) -> None:
    escaped = failing_run()
    escaped.scenario_name = 'xml <case> & "quotes"\x00'
    escaped.failures[0].message = "Bad <action> & policy\x01"
    errored = RunResult(scenario_name="timed_out", error="Timeout <1s>", duration_seconds=0.25)
    errored.failures = escaped.failures
    suite = SuiteResult(
        results=[escaped, errored, RunResult(scenario_name="pass", duration_seconds=0.5)]
    )
    path = tmp_path / "nested" / "report.xml"
    write_junit_report(suite, path)
    root = ET.parse(path).getroot()
    assert root.tag == "testsuite"
    assert root.attrib["tests"] == "3"
    assert root.attrib["failures"] == root.attrib["errors"] == "1"
    assert float(root.attrib["time"]) == 0.875
    cases = root.findall("testcase")
    assert cases[0].attrib["name"] == 'xml <case> & "quotes"� [run 1]'
    failure = cases[0].find("failure")
    assert failure is not None
    assert failure.text is not None
    assert "Bad <action> & policy�" in failure.text
    assert "Expected:" in failure.text and "Observed:" in failure.text
    assert cases[1].find("error") is not None
    assert cases[1].find("failure") is None
    assert cases[2].find("failure") is None
    trace = cases[0].findtext("system-out")
    assert trace is not None
    assert json.loads(trace)["calls"][0]["tool_name"] == "issue_refund"


def test_junit_replay_errors_fail_and_empty_error_is_an_error(tmp_path: Path) -> None:
    suite = SuiteResult(
        results=[
            RunResult(scenario_name="missing_fixture", replay_errors=["Fixture exhausted"]),
            RunResult(scenario_name="empty_error", error=""),
        ]
    )
    path = tmp_path / "report.xml"
    write_junit_report(suite, path)
    root = ET.parse(path).getroot()
    assert root.attrib["failures"] == root.attrib["errors"] == "1"
    assert "Fixture exhausted" in (root.findtext("testcase/failure") or "")


def test_empty_reports_are_parseable_and_truthful(tmp_path: Path) -> None:
    suite = SuiteResult()
    json_path = tmp_path / "empty.json"
    xml_path = tmp_path / "empty.xml"
    write_json_report(suite, json_path)
    write_junit_report(suite, xml_path)
    assert json.loads(json_path.read_text())["results"] == []
    assert ET.parse(xml_path).getroot().attrib["tests"] == "0"
