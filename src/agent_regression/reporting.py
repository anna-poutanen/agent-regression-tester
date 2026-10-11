"""Human-readable and machine-readable regression reports."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console

from .schema import RunResult, SuiteResult


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _failure_details(result: RunResult) -> list[str]:
    """Use the same diagnostic body in terminal and JUnit reports."""
    lines: list[str] = []
    for failure in result.failures:
        tool = f" ({failure.tool_name})" if failure.tool_name else ""
        lines.append(f"{failure.code}{tool}: {failure.message}")
        lines.append(f"  Expected: {_json(failure.expected)}")
        lines.append(f"  Observed: {_json(failure.observed)}")
    for error in result.replay_errors:
        lines.append(f"Replay error: {_json(error) if not isinstance(error, str) else error}")
    if result.error is not None:
        lines.append(f"Run error: {result.error}")
    return lines


def render_terminal(suite: SuiteResult, console: Console | None = None) -> None:
    """Print outcomes grouped by scenario, including repeat failure frequency."""
    console = console or Console()
    if not suite.results:
        console.print("No scenarios ran. An empty suite is not a passing test.", style="yellow")
        return

    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for result in suite.results:
        grouped[result.scenario_name].append(result)

    for name, results in grouped.items():
        failed = sum(not result.passed for result in results)
        status = "FAIL" if failed else "PASS"
        console.print(
            f"{status}: {name} — failed {failed} of {len(results)} runs",
            style="red" if failed else "green",
            markup=False,
            highlight=False,
        )
        for result in results:
            if not result.passed:
                console.print(f"  Run {result.iteration}", markup=False, highlight=False)
                for line in _failure_details(result):
                    console.print(f"    {line}", markup=False, highlight=False)
            if result.differences:
                console.print(
                    f"  Baseline differences (run {result.iteration}):",
                    markup=False,
                    highlight=False,
                )
                for difference in result.differences:
                    console.print(
                        f"    {difference.kind}: {difference.message}",
                        markup=False,
                        highlight=False,
                    )

    console.print(
        f"\n{suite.total_runs} runs: {suite.passed_runs} passed, {suite.failed_runs} failed",
        style="red" if suite.failed_runs else "green",
        markup=False,
        highlight=False,
    )


def write_json_report(suite: SuiteResult, path: str | Path) -> None:
    """Write a stable JSON report containing every run and its complete trace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "total_runs": suite.total_runs,
        "passed_runs": suite.passed_runs,
        "failed_runs": suite.failed_runs,
        "results": [result.model_dump(mode="json") for result in suite.results],
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# XML 1.0 cannot represent control characters such as NUL, even as entities.
_INVALID_XML = re.compile("[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def _xml_text(value: str) -> str:
    return _INVALID_XML.sub("\N{REPLACEMENT CHARACTER}", value)


def write_junit_report(suite: SuiteResult, path: str | Path) -> None:
    """Write JUnit XML; an errored run counts as an error, not two failures."""
    errors = sum(result.error is not None for result in suite.results)
    failures = sum(not result.passed and result.error is None for result in suite.results)
    root = ET.Element(
        "testsuite",
        {
            "name": "agent-regression",
            "tests": str(suite.total_runs),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": "0",
            "time": f"{sum(result.duration_seconds for result in suite.results):.6f}",
        },
    )
    for result in suite.results:
        case = ET.SubElement(
            root,
            "testcase",
            {
                "classname": "agent_regression",
                "name": _xml_text(f"{result.scenario_name} [run {result.iteration}]"),
                "time": f"{result.duration_seconds:.6f}",
            },
        )
        if not result.passed:
            details = "\n".join(_failure_details(result)) or "Regression run failed."
            problem = ET.SubElement(
                case,
                "error" if result.error is not None else "failure",
                {
                    "message": _xml_text(details.splitlines()[0]),
                    "type": "RunError" if result.error is not None else "RegressionFailure",
                },
            )
            problem.text = _xml_text(details)
        trace = ET.SubElement(case, "system-out")
        trace.text = _xml_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
        )
    ET.indent(root)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


__all__ = ["render_terminal", "write_json_report", "write_junit_report"]
