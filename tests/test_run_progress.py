"""Incomplete suites must remain visibly incomplete in every report format."""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.console import Console

from agent_regression.reporting import render_terminal, write_json_report, write_junit_report
from agent_regression.schema import CheckFailure, RunResult, SuiteResult


@pytest.mark.parametrize(
    "values",
    [
        {"status": "incomplete"},
        {"results": [RunResult(scenario_name="one")], "planned_runs": 0},
        {"planned_runs": 2},
        {"stop_reason": "cancelled"},
        {"status": "incomplete", "planned_runs": 2, "stop_reason": " "},
        {"planned_runs": True},
        {"status": "unknown"},
    ],
)
def test_progress_rejects_contradictory_or_invalid_counts(values):
    with pytest.raises(ValidationError):
        SuiteResult(**values)


def test_legacy_complete_suite_and_progress_round_trip():
    legacy = SuiteResult(results=[RunResult(scenario_name="one")])
    assert legacy.status == "completed"
    assert legacy.remaining_runs == 0
    partial = SuiteResult(
        results=[RunResult(scenario_name="one")],
        planned_runs=3,
        status="incomplete",
        stop_reason="max_runs",
    )
    restored = SuiteResult.model_validate_json(partial.model_dump_json(round_trip=True))
    assert restored.remaining_runs == 2
    assert restored.passed_runs == 1
    assert restored.status == "incomplete"


@pytest.mark.parametrize("completed", [0, 1])
def test_partial_terminal_never_reports_a_passing_suite(completed):
    suite = SuiteResult(
        results=[RunResult(scenario_name="one")] * completed,
        planned_runs=3,
        status="incomplete",
        stop_reason="max_seconds",
    )
    output = io.StringIO()
    render_terminal(suite, Console(file=output, width=160, color_system=None))
    text = output.getvalue()
    assert "INCOMPLETE" in text
    assert f"{completed} of 3 planned runs completed" in text
    assert "max_seconds" in text
    assert "PASS:" not in text


def test_partial_json_keeps_progress_and_real_completed_failure(tmp_path: Path):
    suite = SuiteResult(
        results=[
            RunResult(
                scenario_name="refund",
                failures=[CheckFailure(code="forbidden_call", message="Unexpected refund")],
            )
        ],
        planned_runs=4,
        status="incomplete",
        stop_reason="cancelled",
    )
    destination = tmp_path / "report.json"
    write_json_report(suite, destination)
    report = json.loads(destination.read_text())
    assert report["total_runs"] == 1
    assert report["failed_runs"] == 1
    assert report["remaining_runs"] == 3
    assert report["planned_runs"] == 4
    assert report["status"] == "incomplete"
    assert report["stop_reason"] == "cancelled"
    assert report["results"][0]["passed"] is False


@pytest.mark.parametrize("completed", [0, 1])
def test_partial_junit_has_explicit_error_even_if_completed_runs_pass(tmp_path: Path, completed):
    suite = SuiteResult(
        results=[RunResult(scenario_name="one")] * completed,
        planned_runs=3,
        status="incomplete",
        stop_reason="max_runs",
    )
    destination = tmp_path / "report.xml"
    write_junit_report(suite, destination)
    root = ET.parse(destination).getroot()
    assert root.attrib["errors"] == "1"
    assert root.attrib["failures"] == "0"
    assert root.attrib["tests"] == str(completed + 1)
    marker = root.find("./testcase[@name='__suite_incomplete__']/error")
    assert marker is not None
    assert marker.attrib["type"] == "IncompleteSuite"
    assert "not passing tests" in marker.text
    properties = {
        item.attrib["name"]: item.attrib["value"] for item in root.findall("./properties/property")
    }
    assert properties["planned_runs"] == "3"
    assert properties["completed_runs"] == str(completed)


def test_partial_junit_preserves_existing_errors_and_xml_safe_stop_reason(tmp_path: Path):
    suite = SuiteResult(
        results=[
            RunResult(scenario_name="one", error="Agent failed"),
            RunResult(
                scenario_name="two",
                failures=[CheckFailure(code="missing_call", message="Missing lookup")],
            ),
        ],
        planned_runs=3,
        status="incomplete",
        stop_reason="cancelled\x00<&",
    )
    destination = tmp_path / "report.xml"
    write_junit_report(suite, destination)
    root = ET.parse(destination).getroot()
    assert root.attrib["tests"] == "3"
    assert root.attrib["errors"] == "2"
    assert root.attrib["failures"] == "1"
    assert len(root.findall("./testcase/error")) == 2
    assert len(root.findall("./testcase/failure")) == 1


def test_report_revalidates_mutated_nested_results_before_writing(tmp_path: Path):
    suite = SuiteResult(planned_runs=0)
    suite.results.append(RunResult(scenario_name="unexpected"))
    for writer, suffix in [(write_json_report, ".json"), (write_junit_report, ".xml")]:
        destination = tmp_path / f"invalid{suffix}"
        with pytest.raises(ValidationError, match="planned_runs"):
            writer(suite, destination)
        assert not destination.exists()
