"""Bounded asynchronous scenario execution and repeatable suites."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Iterable
from typing import Any

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.usage import UsageLimits

from .adapter import (
    MAX_MODEL_REQUESTS,
    MAX_TOOL_CALLS,
    ReplayState,
    ReplayToolset,
    _json_value,
    calls_from_messages,
    supported_toolsets,
)
from .checks import diff_traces, evaluate_checks
from .schema import CheckFailure, RunResult, Scenario, SuiteResult


def _positive_count(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_timeout(timeout: float) -> None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")
    if not float("-inf") < timeout < float("inf"):
        raise ValueError("timeout must be finite")


async def run_scenario(
    agent: Agent[Any, Any],
    scenario: Scenario,
    *,
    deps: Any = None,
    model: Any = None,
    iteration: int = 1,
    timeout: float = 30,  # noqa: ASYNC109 - public per-run budget uses asyncio.timeout
    allow_unreviewed: bool = False,
) -> RunResult:
    """Replay one scenario with fresh fixtures and the agent's configured model.

    Unsupported agent configurations raise ValueError before running application
    code. Model errors, fixture misses and timeouts become failing results with
    partial traces. A 20-request / 100-tool-call limit also bounds fast loops.
    """
    if not isinstance(scenario, Scenario):
        raise ValueError("Expected a Scenario artifact")
    # Pydantic validates assignment but in-place edits to nested lists/dicts can
    # bypass it. Revalidate a snapshot before any model or application callback.
    scenario = Scenario.model_validate(scenario.model_dump(warnings=False))
    if type(allow_unreviewed) is not bool:
        raise ValueError("allow_unreviewed must be a boolean")
    _positive_count(iteration, "iteration")
    _validate_timeout(timeout)
    started = time.monotonic()
    result = RunResult(scenario_name=scenario.name, iteration=iteration)
    if not scenario.reviewed and not allow_unreviewed:
        result.failures.append(
            CheckFailure(
                code="unreviewed",
                message="Scenario must be reviewed before replay; recorded behavior may be wrong",
            )
        )
        return result
    if not scenario.checks:
        result.failures.append(
            CheckFailure(code="no_checks", message="Scenario has no behavior checks to evaluate")
        )
        return result
    toolsets = supported_toolsets(agent)
    state = ReplayState(fixtures=scenario.fixtures)
    wrapped = [ReplayToolset(toolset, state) for toolset in toolsets]
    kwargs: dict[str, Any] = {
        "deps": deps,
        "usage_limits": UsageLimits(
            request_limit=MAX_MODEL_REQUESTS, tool_calls_limit=MAX_TOOL_CALLS
        ),
    }
    if model is not None:
        kwargs["model"] = model
    with capture_run_messages() as messages:
        try:
            # Empty registered tools is essential: toolsets override alone would
            # leave the agent's directly registered business functions active.
            overrides: dict[str, Any] = {"tools": [], "toolsets": wrapped, "native_tools": []}
            if model is not None:
                overrides["model"] = model
            with agent.override(**overrides):
                async with asyncio.timeout(timeout):
                    response = await agent.run(scenario.prompt, **kwargs)
                    result.output = _json_value(response.output)
        except TimeoutError:
            result.error = f"TimeoutError: scenario exceeded {timeout:g} seconds"
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        try:
            result.calls = state.merge(
                calls_from_messages(messages), failed=result.error is not None
            )
        except Exception as exc:
            result.error = (
                f"{result.error}; " if result.error else ""
            ) + f"TraceExtractionError ({type(exc).__name__}): {exc}"
            result.replay_errors.append("Captured trace could not be fully represented as JSON")
    result.replay_errors.extend(state.errors)
    result.failures = evaluate_checks(scenario, result.calls, output=result.output)
    result.differences = diff_traces(scenario.recorded_calls, result.calls)
    result.duration_seconds = time.monotonic() - started
    return result


async def run_suite(
    agent_or_factory: Agent[Any, Any] | Callable[[], Any],
    scenarios: Iterable[Scenario],
    *,
    repeats: int = 1,
    concurrency: int = 1,
    deps: Any = None,
    model: Any = None,
    timeout: float = 30,  # noqa: ASYNC109 - public per-run budget uses asyncio.timeout
    allow_unreviewed: bool = False,
) -> SuiteResult:
    """Run scenarios in input/iteration order, with run-local fixture state.

    A zero-argument factory (sync or async) provides fresh application state for
    each run. Reusing an Agent is supported; its own dependencies and arbitrary
    callback state remain the application's responsibility.
    """
    _positive_count(repeats, "repeats")
    _positive_count(concurrency, "concurrency")
    _validate_timeout(timeout)
    if not isinstance(agent_or_factory, Agent) and not callable(agent_or_factory):
        raise ValueError("Expected a Pydantic AI Agent or a zero-argument agent factory")
    scenario_list = list(scenarios)
    if any(not isinstance(scenario, Scenario) for scenario in scenario_list):
        raise ValueError("Suite must contain Scenario artifacts")
    scenario_list = [
        Scenario.model_validate(scenario.model_dump(warnings=False)) for scenario in scenario_list
    ]
    if not scenario_list:
        raise ValueError("Suite must contain at least one scenario")
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(scenario: Scenario, iteration: int) -> RunResult:
        async with semaphore:
            started = time.monotonic()
            try:
                # Bound factory creation separately: the scenario timeout must
                # cancel *inside* run_scenario so it can preserve partial traces.
                async with asyncio.timeout(timeout):
                    agent = agent_or_factory
                    if not isinstance(agent, Agent):
                        agent = agent()
                        if inspect.isawaitable(agent):
                            agent = await agent
                if not isinstance(agent, Agent):
                    raise ValueError("Agent factory must return a Pydantic AI Agent")
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError
                result = await run_scenario(
                    agent,
                    scenario,
                    deps=deps,
                    model=model,
                    iteration=iteration,
                    timeout=remaining,
                    allow_unreviewed=allow_unreviewed,
                )
                result.duration_seconds = time.monotonic() - started
                return result
            except ValueError:
                raise
            except TimeoutError:
                return RunResult(
                    scenario_name=scenario.name,
                    iteration=iteration,
                    error=f"TimeoutError: scenario or factory exceeded {timeout:g} seconds",
                    duration_seconds=time.monotonic() - started,
                )
            except Exception as exc:
                return RunResult(
                    scenario_name=scenario.name,
                    iteration=iteration,
                    error=f"{type(exc).__name__}: agent factory failed: {exc}",
                    duration_seconds=time.monotonic() - started,
                )

    tasks = [
        asyncio.create_task(run_one(scenario, iteration))
        for scenario in scenario_list
        for iteration in range(1, repeats + 1)
    ]
    try:
        return SuiteResult(results=list(await asyncio.gather(*tasks)))
    except BaseException:
        # gather alone leaves siblings running after a configuration failure.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def run_suite_sync(
    agent_or_factory: Agent[Any, Any] | Callable[[], Any],
    scenarios: Iterable[Scenario],
    *,
    repeats: int = 1,
    concurrency: int = 1,
    deps: Any = None,
    model: Any = None,
    timeout: float = 30,  # noqa: ASYNC109 - public per-run budget uses asyncio.timeout
    allow_unreviewed: bool = False,
) -> SuiteResult:
    """Sync convenience entry point; use run_suite inside an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_suite(
                agent_or_factory,
                scenarios,
                repeats=repeats,
                concurrency=concurrency,
                deps=deps,
                model=model,
                timeout=timeout,
                allow_unreviewed=allow_unreviewed,
            )
        )
    raise ValueError(
        "run_suite_sync cannot run inside an active event loop; await run_suite instead"
    )
