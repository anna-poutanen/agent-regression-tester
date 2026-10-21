"""Offline controlled-run integration; no provider request is permitted."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

import agent_regression.runner as runner
from agent_regression.checkpoint import load_checkpoint
from agent_regression.runner import run_suite, run_suite_sync
from agent_regression.schema import OutputContains, RequiredCall, Scenario, ToolFixture


def case(name="case"):
    return Scenario(
        name=name,
        prompt=name,
        reviewed=True,
        fixtures=[ToolFixture(tool_name="lookup", arguments={"id": "123"}, response="saved")],
        checks=[RequiredCall(tool_name="lookup"), OutputContains(text="done")],
    )


def agent(*, output="done", wait=None, started=None, cancelled=None):
    def lookup(id: str) -> str:
        raise AssertionError("A real business tool must never execute")

    async def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"id": "123"})])
        if wait is not None:
            if started is not None:
                started.set()
            try:
                await wait.wait()
            finally:
                if cancelled is not None:
                    cancelled.set()
        return ModelResponse(parts=[TextPart(output)])

    return Agent(FunctionModel(model), tools=[lookup])


async def test_max_runs_resumes_exact_prefix_including_failed_runs(tmp_path):
    path = tmp_path / "run.json"
    factory_calls = []
    observed_prefixes = []

    def factory():
        # The initial snapshot and every prior result must already be durable.
        data = json.loads(path.read_text())
        observed_prefixes.append(len(data["results"]))
        factory_calls.append(len(factory_calls))
        return agent(output="wrong" if len(factory_calls) == 1 else "done")

    scenarios = [case("a"), case("b")]
    first = await run_suite(
        factory, scenarios, repeats=2, checkpoint=path, run_id="same-agent", max_runs=1
    )
    assert (first.status, first.stop_reason, first.planned_runs) == ("incomplete", "max_runs", 4)
    assert first.failed_runs == 1
    failed = first.results[0].model_dump()
    second = await run_suite(
        factory,
        scenarios,
        repeats=2,
        checkpoint=path,
        run_id="same-agent",
        resume=True,
        max_runs=2,
    )
    assert (second.total_runs, second.remaining_runs, second.stop_reason) == (3, 1, "max_runs")
    assert second.results[0].model_dump() == failed
    final = await run_suite(
        factory, scenarios, repeats=2, checkpoint=path, run_id="same-agent", resume=True
    )
    assert final.status == "completed"
    assert final.stop_reason is None
    assert final.failed_runs == 1
    assert observed_prefixes == [0, 1, 2, 3]
    assert len(factory_calls) == 4
    assert [(r.scenario_name, r.iteration) for r in final.results] == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]


async def test_budget_cancels_inflight_model_without_checkpointing_a_fake_failure(tmp_path):
    path = tmp_path / "run.json"
    cancelled = asyncio.Event()
    started = asyncio.Event()
    factory_calls = []

    def factory():
        factory_calls.append(1)
        return agent(
            wait=asyncio.Event() if len(factory_calls) == 2 else None,
            started=started,
            cancelled=cancelled,
        )

    partial = await run_suite(
        factory, [case("a"), case("b")], checkpoint=path, run_id="agent", max_seconds=0.05
    )
    assert started.is_set()
    assert cancelled.is_set()
    assert partial.status == "incomplete"
    assert partial.stop_reason == "max_seconds"
    assert partial.total_runs == partial.passed_runs == 1
    assert partial.failed_runs == 0
    assert load_checkpoint(path).results[0].scenario_name == "a"
    resumed_calls = []

    def resumed_factory():
        resumed_calls.append(1)
        return agent()

    final = await run_suite(
        resumed_factory, [case("a"), case("b")], checkpoint=path, run_id="agent", resume=True
    )
    assert len(resumed_calls) == 1
    assert final.status == "completed"
    assert final.passed_runs == 2
    assert final.results[1].calls[0].response == "saved"


async def test_wall_budget_includes_async_factory_cleanup():
    cancelled = asyncio.Event()

    async def factory():
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    result = await run_suite(factory, [case()], max_seconds=0.02)
    assert cancelled.is_set()
    assert (result.status, result.stop_reason, result.total_runs) == (
        "incomplete",
        "max_seconds",
        0,
    )
    assert result.failed_runs == 0


async def test_async_cancellation_saves_completed_prefix_then_propagates(tmp_path):
    path = tmp_path / "run.json"
    started = asyncio.Event()
    cancelled = asyncio.Event()
    count = 0

    def factory():
        nonlocal count
        count += 1
        return agent(
            wait=asyncio.Event() if count == 2 else None,
            started=started,
            cancelled=cancelled,
        )

    task = asyncio.create_task(
        run_suite(factory, [case("a"), case("b")], checkpoint=path, run_id="agent")
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    saved = load_checkpoint(path)
    assert (saved.status, saved.stop_reason, len(saved.results)) == ("incomplete", "cancelled", 1)
    final = await run_suite(
        lambda: agent(), [case("a"), case("b")], checkpoint=path, run_id="agent", resume=True
    )
    assert final.status == "completed"
    assert final.passed_runs == 2


async def test_completed_checkpoint_does_not_construct_agent_again(tmp_path):
    path = tmp_path / "run.json"
    await run_suite(lambda: agent(), [case()], checkpoint=path, run_id="agent")

    def forbidden_factory():
        raise AssertionError("Completed runs must be skipped")

    result = await run_suite(
        forbidden_factory, [case()], checkpoint=path, run_id="agent", resume=True, max_runs=1
    )
    assert result.status == "completed"
    assert result.passed_runs == 1


async def test_per_run_timeout_is_a_completed_failure_not_invocation_budget(tmp_path):
    path = tmp_path / "run.json"
    cancelled = asyncio.Event()
    count = 0

    def factory():
        nonlocal count
        count += 1
        return agent(wait=asyncio.Event() if count == 1 else None, cancelled=cancelled)

    result = await run_suite(
        factory,
        [case("a"), case("b")],
        checkpoint=path,
        run_id="agent",
        timeout=0.02,
        max_seconds=1,
    )
    assert cancelled.is_set()
    assert (result.status, result.failed_runs, result.total_runs) == ("completed", 1, 2)
    assert "TimeoutError" in result.results[0].error
    assert len(load_checkpoint(path).results) == 2


async def test_batches_pause_only_after_new_work_with_more_runs(tmp_path, monkeypatch):
    original_sleep = asyncio.sleep
    pauses = []

    async def sleep(seconds):
        if seconds > 0:
            pauses.append(seconds)
        await original_sleep(0)

    monkeypatch.setattr(runner.asyncio, "sleep", sleep)
    path = tmp_path / "run.json"
    scenarios = [case(str(i)) for i in range(5)]
    partial = await run_suite(
        lambda: agent(),
        scenarios,
        checkpoint=path,
        run_id="agent",
        max_runs=4,
        batch_size=2,
        pause_seconds=0.01,
    )
    assert partial.total_runs == 4
    assert pauses == [0.01]
    final = await run_suite(
        lambda: agent(),
        scenarios,
        checkpoint=path,
        run_id="agent",
        resume=True,
        batch_size=1,
        pause_seconds=0.01,
    )
    assert final.status == "completed"
    assert pauses == [0.01]


async def test_wall_budget_includes_batch_pause():
    result = await run_suite(
        lambda: agent(),
        [case("a"), case("b")],
        max_seconds=0.03,
        batch_size=1,
        pause_seconds=10,
    )
    assert result.total_runs == result.passed_runs == 1
    assert (result.status, result.stop_reason) == ("incomplete", "max_seconds")


@pytest.mark.parametrize(
    "options",
    [
        {"resume": True},
        {"resume": 1},
        {"run_id": "unbound"},
        {"checkpoint": " "},
        {"checkpoint": "x.json"},
        {"checkpoint": "x.json", "run_id": " "},
        {"max_runs": 0},
        {"max_runs": True},
        {"max_runs": 1.5},
        {"max_seconds": True},
        {"max_seconds": 0},
        {"max_seconds": float("nan")},
        {"max_seconds": float("inf")},
        {"batch_size": 0},
        {"batch_size": False},
        {"pause_seconds": -1},
        {"pause_seconds": True},
        {"pause_seconds": float("nan")},
        {"pause_seconds": 1},
        {"batch_size": 1, "concurrency": 2},
        {"max_runs": 1, "concurrency": 2},
    ],
)
async def test_invalid_controls_fail_before_factory(options):
    effects = []

    def factory():
        effects.append("factory")
        return agent()

    with pytest.raises(ValueError):
        await run_suite(factory, [case()], **options)
    assert effects == []


async def test_controlled_plan_requires_unique_names_but_default_still_accepts_duplicates():
    effects = []

    def factory():
        effects.append("factory")
        return agent()

    with pytest.raises(ValueError, match="unique"):
        await run_suite(factory, [case(), case()], max_runs=1)
    assert effects == []
    result = await run_suite(factory, [case(), case()], concurrency=2)
    assert result.passed_runs == 2
    assert result.planned_runs is None
    assert result.status == "completed"


@pytest.mark.parametrize("change", ["run_id", "prompt", "repeats", "timeout", "review", "cases"])
async def test_mismatched_resume_is_rejected_before_factory(tmp_path, change):
    path = tmp_path / "run.json"
    scenarios = [case("a"), case("b")]
    await run_suite(lambda: agent(), scenarios, checkpoint=path, run_id="agent", max_runs=1)
    options = dict(checkpoint=path, run_id="agent", resume=True)
    if change == "run_id":
        options["run_id"] = "different"
    elif change == "prompt":
        scenarios[1].prompt = "changed prompt"
    elif change == "cases":
        scenarios.append(case("new-case"))
    elif change == "repeats":
        options["repeats"] = 2
    elif change == "timeout":
        options["timeout"] = 15
    else:
        options["allow_unreviewed"] = True
    calls = []

    def forbidden_factory():
        calls.append("factory")
        return agent()

    with pytest.raises(ValueError, match="mismatch"):
        await run_suite(forbidden_factory, scenarios, **options)
    assert calls == []
    assert len(load_checkpoint(path).results) == 1


async def test_fatal_configuration_error_saves_prefix_and_error_status(tmp_path):
    path = tmp_path / "run.json"
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return agent() if calls == 1 else object()

    with pytest.raises(ValueError, match="must return"):
        await run_suite(factory, [case("a"), case("b")], checkpoint=path, run_id="agent")
    saved = load_checkpoint(path)
    assert (saved.status, saved.stop_reason, len(saved.results)) == ("incomplete", "error", 1)


async def test_checkpoint_lock_covers_active_model_and_releases_after_cancel(tmp_path):
    path = tmp_path / "run.json"
    started = asyncio.Event()
    task = asyncio.create_task(
        run_suite(
            lambda: agent(wait=asyncio.Event(), started=started),
            [case()],
            checkpoint=path,
            run_id="agent",
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    effects = []

    def factory():
        effects.append("factory")
        return agent()

    with pytest.raises(ValueError, match="locked"):
        await run_suite(factory, [case()], checkpoint=path, run_id="agent", resume=True)
    assert effects == []
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert load_checkpoint(path).stop_reason == "cancelled"


def test_sync_entry_forwards_controls(tmp_path: Path):
    path = tmp_path / "run.json"
    partial = run_suite_sync(
        lambda: agent(), [case("a"), case("b")], checkpoint=path, run_id="agent", max_runs=1
    )
    assert (partial.status, partial.stop_reason) == ("incomplete", "max_runs")
    final = run_suite_sync(
        lambda: agent(), [case("a"), case("b")], checkpoint=path, run_id="agent", resume=True
    )
    assert final.status == "completed"


@pytest.mark.parametrize("count", [1, 2])
async def test_returned_failure_survives_clock_crossing_and_is_skipped_on_resume(
    tmp_path, monkeypatch, count
):
    """A returned case is completed even if synchronous code crossed the deadline."""
    from agent_regression.schema import CheckFailure, RunResult

    path = tmp_path / "run.json"
    clock = [runner.time.monotonic()]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    factory_calls = []

    def factory():
        factory_calls.append(1)
        # Synchronous application work cannot be preempted by an asyncio timer.
        clock[0] += 2
        return agent()

    async def completed_case(agent, scenario, *, iteration, **kwargs):
        return RunResult(
            scenario_name=scenario.name,
            iteration=iteration,
            failures=[CheckFailure(code="business_failure", message="A real completed failure")]
            if scenario.name == "a"
            else [],
        )

    monkeypatch.setattr(runner, "run_scenario", completed_case)
    scenarios = [case("a"), case("b")][:count]
    partial = await run_suite(factory, scenarios, checkpoint=path, run_id="agent", max_seconds=1)
    assert partial.failed_runs == 1
    assert partial.status == ("completed" if count == 1 else "incomplete")
    assert partial.stop_reason == (None if count == 1 else "max_seconds")
    assert len(factory_calls) == 1
    saved_failure = load_checkpoint(path).results[0].model_dump()
    final = await run_suite(factory, scenarios, checkpoint=path, run_id="agent", resume=True)
    assert len(factory_calls) == count
    assert final.status == "completed"
    assert final.failed_runs == 1
    assert final.results[0].model_dump() == saved_failure


async def test_resumed_batch_counter_counts_only_new_runs(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    scenarios = [case(str(i)) for i in range(5)]
    await run_suite(lambda: agent(), scenarios, checkpoint=path, run_id="agent", max_runs=1)
    original_sleep = asyncio.sleep
    pauses = []

    async def sleep(seconds):
        if seconds > 0:
            pauses.append(seconds)
        await original_sleep(0)

    monkeypatch.setattr(runner.asyncio, "sleep", sleep)
    result = await run_suite(
        lambda: agent(),
        scenarios,
        checkpoint=path,
        run_id="agent",
        resume=True,
        max_runs=3,
        batch_size=2,
        pause_seconds=0.01,
    )
    assert (result.total_runs, result.stop_reason) == (4, "max_runs")
    assert pauses == [0.01]
