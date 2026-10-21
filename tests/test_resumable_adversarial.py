"""Cross-boundary cancellation/reporting checks using only offline agents."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

import agent_regression.runner as runner
from agent_regression.checkpoint import load_checkpoint
from agent_regression.reporting import write_json_report, write_junit_report
from agent_regression.schema import OutputContains, Scenario
from agent_regression.storage import save_scenario


async def test_cancelling_cooldown_preserves_failed_prefix_and_ci_error_until_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intentional break must not discard a failure or turn it into a green CI result."""
    checkpoint = tmp_path / "checkpoint.json"
    report_json = tmp_path / "report.json"
    report_xml = tmp_path / "report.xml"
    scenario = Scenario(
        name="request",
        prompt="Please answer",
        reviewed=True,
        checks=[OutputContains(text="correct")],
    )
    pause_started = asyncio.Event()
    original_sleep = asyncio.sleep

    async def pause(seconds: float) -> None:
        if seconds == 10:
            pause_started.set()
            await asyncio.Event().wait()
        else:
            await original_sleep(seconds)

    monkeypatch.setattr(asyncio, "sleep", pause)
    outputs = iter(("wrong", "correct", "correct"))
    factory_calls: list[None] = []

    def factory() -> Agent[Any, Any]:
        factory_calls.append(None)
        output = next(outputs)

        def response(messages: Any, info: Any) -> ModelResponse:
            return ModelResponse(parts=[TextPart(output)])

        return Agent(FunctionModel(response))

    task = asyncio.create_task(
        runner.run_suite(
            factory,
            [scenario],
            repeats=3,
            checkpoint=checkpoint,
            run_id="configuration",
            batch_size=1,
            pause_seconds=10,
        )
    )
    await asyncio.wait_for(pause_started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    saved = load_checkpoint(checkpoint)
    assert saved.stop_reason == "cancelled"
    assert len(saved.results) == 1
    failed_result = saved.results[0].model_dump(mode="json")
    assert not saved.results[0].passed
    assert len(factory_calls) == 1

    suite = saved.to_suite()
    write_json_report(suite, report_json)
    write_junit_report(suite, report_xml)
    payload = json.loads(report_json.read_text())
    assert (payload["status"], payload["failed_runs"], payload["remaining_runs"]) == (
        "incomplete",
        1,
        2,
    )
    xml = ET.parse(report_xml).getroot()
    assert (xml.get("tests"), xml.get("failures"), xml.get("errors")) == ("2", "1", "1")
    assert xml.find("testcase/error[@type='IncompleteSuite']") is not None

    final = await runner.run_suite(
        factory,
        [scenario],
        repeats=3,
        checkpoint=checkpoint,
        run_id="configuration",
        resume=True,
    )
    assert len(factory_calls) == 3
    assert final.status == "completed"
    assert final.failed_runs == 1
    assert final.results[0].model_dump(mode="json") == failed_result
    write_junit_report(final, report_xml)
    xml = ET.parse(report_xml).getroot()
    assert (xml.get("tests"), xml.get("failures"), xml.get("errors")) == ("3", "1", "0")
    assert xml.find("testcase/error[@type='IncompleteSuite']") is None


@pytest.mark.skipif(sys.platform == "win32", reason="Tests the POSIX CLI SIGINT contract")
def test_actual_cli_sigint_writes_incomplete_reports_and_resumes_without_duplicate_result(
    tmp_path: Path,
) -> None:
    """Exercise a real interrupt, rather than replacing the runner with a raising stub."""
    marker = tmp_path / "active-model.txt"
    module = tmp_path / "offline_sigint_agent.py"
    module.write_text(
        textwrap.dedent(f"""\
        import asyncio
        from pathlib import Path
        from pydantic_ai import Agent, models
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel

        models.ALLOW_MODEL_REQUESTS = False
        count = 0

        def make_agent():
            global count
            count += 1
            current = count
            async def respond(messages, info):
                if current == 2:
                    Path({str(marker)!r}).write_text('active')
                    await asyncio.Event().wait()
                return ModelResponse(parts=[TextPart('done')])
            return Agent(FunctionModel(respond))
        """),
        encoding="utf-8",
    )
    scenario = tmp_path / "scenario.json"
    save_scenario(
        Scenario(
            name="request",
            prompt="Please answer",
            reviewed=True,
            checks=[OutputContains(text="done")],
        ),
        scenario,
    )
    checkpoint = tmp_path / "checkpoint.json"
    report_json = tmp_path / "report.json"
    report_xml = tmp_path / "report.xml"
    command = [
        sys.executable,
        "-m",
        "agent_regression",
        "run",
        str(scenario),
        "--agent",
        "offline_sigint_agent:make_agent",
        "--repeats",
        "2",
        "--checkpoint",
        str(checkpoint),
        "--run-id",
        "offline-configuration",
        "--json-report",
        str(report_json),
        "--junit-report",
        str(report_xml),
    ]
    environment = {**os.environ, "PYDANTIC_AI_NO_BANNER": "1"}
    process = subprocess.Popen(
        command,
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists():
            if process.poll() is not None:
                output, _ = process.communicate()
                pytest.fail(f"Offline CLI exited before the interrupt: {output}")
            if time.monotonic() >= deadline:
                pytest.fail("Offline CLI did not reach its second iteration")
            time.sleep(0.01)
        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    assert process.returncode == 130, output
    saved = load_checkpoint(checkpoint)
    assert saved.status == "incomplete"
    assert saved.stop_reason == "cancelled"
    assert len(saved.results) == 1
    first = saved.results[0].model_dump(mode="json")
    assert json.loads(report_json.read_text())["remaining_runs"] == 1
    assert (
        ET.parse(report_xml).getroot().find("testcase/error[@type='IncompleteSuite']") is not None
    )

    resumed = subprocess.run(
        [*command, "--resume"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    completed = load_checkpoint(checkpoint)
    assert completed.status == "completed"
    assert len(completed.results) == 2
    assert completed.results[0].model_dump(mode="json") == first
    assert [result.iteration for result in completed.results] == [1, 2]
