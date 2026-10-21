"""Local command-line workflow for reviewed agent regression scenarios."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer
from pydantic_ai import Agent

from .adapter import record as record_agent
from .adapter import scenario_from_messages
from .reporting import render_terminal, write_json_report, write_junit_report
from .runner import run_suite_sync
from .schema import Scenario, SuiteResult
from .storage import load_scenario, save_scenario
from .suggest import suggest_checks
from .support.cli import app as support_app

if TYPE_CHECKING:
    from .checkpoint import CheckpointState

app = typer.Typer(
    help="Record, review, and replay Pydantic AI tool-call regression tests.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(support_app, name="support")

AgentTarget = Agent[Any, Any] | Callable[[], Agent[Any, Any]]


@app.callback()
def configure_cli() -> None:
    """Keep framework promotional output out of regression reports."""
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _instantiate(target: Any, specification: str) -> Agent[Any, Any]:
    try:
        value = target()
    except Exception as exc:
        raise ValueError(f"Agent factory {specification!r} failed: {exc}") from exc
    if inspect.isawaitable(value):
        if inspect.iscoroutine(value):
            value.close()
        raise ValueError("CLI agent factories must be synchronous and take no arguments")
    if not isinstance(value, Agent):
        raise ValueError(
            f"Agent factory {specification!r} returned {type(value).__name__}; "
            "expected a pydantic_ai.Agent"
        )
    return value


def _load_agent_target(specification: str, *, eager: bool = True) -> AgentTarget:
    """Import trusted local code, validating factories without an extra invocation."""
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name or ":" in attribute_name:
        raise ValueError("Agent target must have the form module:factory or module:agent")
    if any(not segment for segment in attribute_name.split(".")):
        raise ValueError("Agent attribute must be a nonempty dotted name")
    try:
        target: Any = importlib.import_module(module_name)
        for segment in attribute_name.split("."):
            target = getattr(target, segment)
    except Exception as exc:
        raise ValueError(f"Cannot import agent target {specification!r}: {exc}") from exc
    if isinstance(target, Agent):
        return target
    if not callable(target):
        raise ValueError(f"Agent target {specification!r} is neither an Agent nor a factory")
    first_agent: Agent[Any, Any] | None = _instantiate(target, specification) if eager else None

    def factory() -> Agent[Any, Any]:
        nonlocal first_agent
        if first_agent is not None:
            result = first_agent
            first_agent = None
            return result
        return _instantiate(target, specification)

    return factory


def _as_agent(target: AgentTarget) -> Agent[Any, Any]:
    return target if isinstance(target, Agent) else target()


def _checks_json(scenario: Scenario) -> str:
    return json.dumps(
        [check.model_dump(mode="json") for check in scenario.checks],
        ensure_ascii=False,
        indent=2,
    )


def _scenario_paths(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*.json")))
        else:
            discovered.append(path)
    if not discovered:
        raise ValueError("No scenario JSON files found")
    return discovered


def _read_scenarios(paths: list[Path], *, allow_unreviewed: bool) -> list[Scenario]:
    scenarios: list[Scenario] = []
    seen_paths: set[Path] = set()
    seen_names: set[str] = set()
    for path in _scenario_paths(paths):
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        scenario = load_scenario(path)
        if not scenario.checks:
            raise ValueError(f"{path}: scenario has no checks; define expected behavior first")
        if not scenario.reviewed and not allow_unreviewed:
            raise ValueError(f"{path}: scenario is unreviewed; run `agent-regression review` first")
        if scenario.name in seen_names:
            raise ValueError(f"Duplicate scenario name {scenario.name!r}; use unique names")
        seen_paths.add(resolved)
        seen_names.add(scenario.name)
        scenarios.append(scenario)
    return scenarios


def _paths_alias(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def _validate_run_paths(
    paths: list[Path],
    *,
    checkpoint: Path | None,
    json_report: Path | None,
    junit_report: Path | None,
) -> None:
    """Keep generated artifacts from replacing scenarios, checkpoints, or locks."""
    outputs = [path for path in (checkpoint, json_report, junit_report) if path is not None]
    for directory in (path for path in paths if path.is_dir()):
        for output in outputs:
            if output.suffix == ".json" and output.resolve().is_relative_to(directory.resolve()):
                raise ValueError(
                    "Checkpoint and report JSON files must be outside scenario directories"
                )
    inputs = _scenario_paths(paths)
    protected = [("scenario", path) for path in inputs]
    if checkpoint is not None:
        lock = checkpoint.with_name(checkpoint.name + ".lock")
        if checkpoint.is_symlink() or lock.is_symlink():
            raise ValueError("Checkpoint and checkpoint lock paths must not be symlinks")
        for label, path in (("checkpoint", checkpoint), ("checkpoint lock", lock)):
            if path.exists() and path.is_dir():
                raise ValueError(f"{label.capitalize()} path is a directory: {path}")
            for scenario_path in inputs:
                if _paths_alias(path, scenario_path):
                    raise ValueError(f"{label.capitalize()} path must not alias a scenario file")
            protected.append((label, path))
    if json_report is not None and junit_report is not None:
        if _paths_alias(json_report, junit_report):
            raise ValueError("JSON and JUnit report paths must differ")
    for report in (json_report, junit_report):
        if report is None:
            continue
        if report.exists() and report.is_dir():
            raise ValueError(f"Report path is a directory: {report}")
        for label, path in protected:
            if _paths_alias(report, path):
                raise ValueError(f"Report path must not alias a {label} file")


def _controlled_options(
    *,
    checkpoint: Path | None,
    resume: bool,
    run_id: str | None,
    max_runs: int | None,
    max_seconds: float | None,
    batch_size: int | None,
    pause_seconds: float,
    concurrency: int,
    timeout: float,
    agent: str,
) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--timeout must be finite and positive")
    if resume and checkpoint is None:
        raise ValueError("--resume requires --checkpoint and --run-id")
    if checkpoint is not None and (run_id is None or not run_id.strip()):
        raise ValueError("--checkpoint requires a nonblank --run-id")
    if run_id is not None and checkpoint is None:
        raise ValueError("--run-id requires --checkpoint")
    if max_runs is not None and max_runs < 1:
        raise ValueError("--max-runs must be positive")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("--max-seconds must be finite and positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not math.isfinite(pause_seconds) or pause_seconds < 0:
        raise ValueError("--pause-seconds must be finite and nonnegative")
    if pause_seconds > 0 and batch_size is None:
        raise ValueError("--pause-seconds requires --batch-size")
    controlled = (
        checkpoint is not None
        or resume
        or run_id is not None
        or max_runs is not None
        or max_seconds is not None
        or batch_size is not None
        or pause_seconds != 0
    )
    if not controlled:
        return {}
    if concurrency != 1:
        raise ValueError("Controlled runs require --concurrency 1")
    identity = (
        json.dumps({"user_run_id": run_id, "agent": agent}, sort_keys=True, separators=(",", ":"))
        if run_id is not None
        else None
    )
    return {
        "checkpoint": checkpoint,
        "resume": resume,
        "run_id": identity,
        "max_runs": max_runs,
        "max_seconds": max_seconds,
        "batch_size": batch_size,
        "pause_seconds": pause_seconds,
    }


def _interrupted_report(
    checkpoint: Path | None,
    json_report: Path | None,
    junit_report: Path | None,
    *,
    expected: CheckpointState | None = None,
) -> None:
    typer.echo("Interrupted.", err=True)
    if checkpoint is None:
        return
    try:
        from .checkpoint import load_checkpoint

        if expected is None:
            typer.echo("Checkpoint identity was not validated; no report was recovered.", err=True)
            return
        state = load_checkpoint(checkpoint, expected=expected)
        suite = state.to_suite()
        _report(suite, json_report, junit_report)
        typer.echo(f"Checkpoint: {checkpoint}")
        if suite.status == "incomplete":
            typer.echo("Resume by rerunning this command with --resume.")
    except Exception as exc:
        typer.echo(f"Could not read or report a validated checkpoint: {exc}", err=True)


def _report(
    suite: SuiteResult,
    json_report: Path | None,
    junit_report: Path | None,
) -> None:
    render_terminal(suite)
    if json_report is not None:
        write_json_report(suite, json_report)
        typer.echo(f"JSON report: {json_report}")
    if junit_report is not None:
        write_junit_report(suite, junit_report)
        typer.echo(f"JUnit report: {junit_report}")


@app.command()
def validate(path: Annotated[Path, typer.Argument(help="Scenario JSON file to validate.")]) -> None:
    """Validate the artifact and require at least one explicit behavior check."""
    try:
        scenario = load_scenario(path)
        if not scenario.checks:
            raise ValueError("Scenario has no checks; define expected behavior before running")
    except Exception as exc:
        _fail(str(exc))
    status = "reviewed" if scenario.reviewed else "unreviewed — review required before replay"
    typer.echo(f"Valid: {scenario.name} ({len(scenario.checks)} checks; {status})")


@app.command()
def suggest(
    path: Annotated[Path, typer.Argument(help="Recorded scenario JSON file.")],
    write: Annotated[
        bool, typer.Option("--write", help="Replace saved checks with this unreviewed proposal.")
    ] = False,
) -> None:
    """Propose checks from observations; recorded behavior still needs human review."""
    try:
        scenario = load_scenario(path)
        scenario.checks = suggest_checks(scenario)
        scenario.reviewed = False
        typer.echo(_checks_json(scenario))
        if write:
            save_scenario(scenario, path, overwrite=True)
            typer.echo(f"Saved proposal to {path}; review is required.")
        else:
            typer.echo("Proposal only. Use --write to save, then edit and review the checks.")
        typer.echo("These checks describe recorded calls; they do not establish business policy.")
    except Exception as exc:
        _fail(str(exc))


@app.command()
def review(
    path: Annotated[Path, typer.Argument(help="Scenario whose checks you have reviewed.")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Confirm that these checks express correct behavior."),
    ] = False,
) -> None:
    """Display checks and explicitly mark the scenario reviewed."""
    try:
        scenario = load_scenario(path)
        if not scenario.checks:
            raise ValueError("Cannot review a scenario without checks; edit or suggest them first")
        typer.echo(f"Scenario: {scenario.name}\nPrompt: {scenario.prompt}")
        typer.echo(f"Saved responses: {len(scenario.fixtures)}")
        typer.echo(_checks_json(scenario))
        typer.echo("Confirm these checks express correct behavior, including any business rules.")
        if not yes and not typer.confirm("Mark this scenario reviewed?", default=False):
            typer.echo("Review not confirmed; file unchanged.")
            return
        scenario.reviewed = True
        save_scenario(scenario, path, overwrite=True)
        typer.echo(f"Reviewed: {scenario.name}")
    except (typer.Abort, typer.Exit):
        raise
    except Exception as exc:
        _fail(str(exc))


@app.command("run")
def run_command(
    paths: Annotated[
        list[Path], typer.Argument(help="Scenario files or directories (recursive *.json search).")
    ],
    agent: Annotated[
        str, typer.Option("--agent", help="Trusted local module:factory or module:agent.")
    ],
    repeats: Annotated[int, typer.Option(min=1, help="Runs per scenario.")] = 1,
    concurrency: Annotated[int, typer.Option(min=1, help="Maximum concurrent runs.")] = 1,
    timeout: Annotated[float, typer.Option(min=0.001, help="Timeout per run in seconds.")] = 30.0,
    allow_unreviewed: Annotated[
        bool, typer.Option("--allow-unreviewed", help="Explicitly run draft scenarios with checks.")
    ] = False,
    json_report: Annotated[Path | None, typer.Option(help="Write a complete JSON report.")] = None,
    junit_report: Annotated[
        Path | None, typer.Option(help="Write CI-compatible JUnit XML.")
    ] = None,
    checkpoint: Annotated[
        Path | None, typer.Option(help="Save completed runs atomically for later resume.")
    ] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Resume a matching checkpoint.")] = False,
    run_id: Annotated[
        str | None,
        typer.Option(help="Configuration identity required with --checkpoint; change after edits."),
    ] = None,
    max_runs: Annotated[
        int | None, typer.Option(min=1, help="Maximum new completed runs in this invocation.")
    ] = None,
    max_seconds: Annotated[
        float | None,
        typer.Option(help="Execution deadline in seconds; cancellation is cooperative."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(min=1, help="Completed runs per batch; controlled runs are serial."),
    ] = None,
    pause_seconds: Annotated[
        float, typer.Option(help="Pause between batches; requires --batch-size when positive.")
    ] = 0.0,
) -> None:
    """Replay saved tool responses against the current local agent and model."""
    expected_checkpoint = None
    try:
        if checkpoint is not None:
            checkpoint = checkpoint.expanduser()
        controlled = _controlled_options(
            checkpoint=checkpoint,
            resume=resume,
            run_id=run_id,
            max_runs=max_runs,
            max_seconds=max_seconds,
            batch_size=batch_size,
            pause_seconds=pause_seconds,
            concurrency=concurrency,
            timeout=timeout,
            agent=agent,
        )
        _validate_run_paths(
            paths, checkpoint=checkpoint, json_report=json_report, junit_report=junit_report
        )
        scenarios = _read_scenarios(paths, allow_unreviewed=allow_unreviewed)
        if checkpoint is not None:
            from .checkpoint import build_checkpoint

            expected_checkpoint = build_checkpoint(
                scenarios,
                run_id=controlled["run_id"],
                repeats=repeats,
                timeout=timeout,
                allow_unreviewed=allow_unreviewed,
            )
        target = _load_agent_target(agent, eager=False) if controlled else _load_agent_target(agent)
        suite = run_suite_sync(
            target,
            scenarios,
            repeats=repeats,
            concurrency=concurrency,
            timeout=timeout,
            allow_unreviewed=allow_unreviewed,
            **controlled,
        )
        _report(suite, json_report, junit_report)
    except KeyboardInterrupt:
        _interrupted_report(checkpoint, json_report, junit_report, expected=expected_checkpoint)
        raise typer.Exit(code=130) from None
    except Exception as exc:
        _fail(str(exc))
    if suite.status == "incomplete":
        if checkpoint is not None:
            typer.echo("Resume by rerunning this command with --resume.")
        raise typer.Exit(code=3)
    if not suite.results:
        _fail("No scenarios ran")
    raise typer.Exit(code=1 if suite.failed_runs else 0)


@app.command("record")
def record_command(
    prompt: Annotated[str, typer.Option(help="User request for the scenario.")],
    name: Annotated[str, typer.Option(help="Unique scenario name.")],
    output: Annotated[Path, typer.Option(help="Destination scenario JSON file.")],
    agent: Annotated[
        str | None, typer.Option(help="Trusted local module:factory or module:agent to run.")
    ] = None,
    messages: Annotated[
        Path | None, typer.Option(help="Import saved Pydantic AI messages JSON instead of running.")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing destination file.")] = False,
) -> None:
    """Record a local agent run or import saved messages; checks start unreviewed."""
    if (agent is None) == (messages is None):
        _fail("Specify exactly one of --agent or --messages")
    if output.exists() and not overwrite:
        _fail(f"{output} already exists; use --overwrite to replace it")
    try:
        # Reject malformed artifacts before constructing an agent or executing tools.
        Scenario(name=name, prompt=prompt)
        if messages is not None:
            scenario = scenario_from_messages(messages.read_bytes(), name=name, prompt=prompt)
        else:
            assert agent is not None
            local_agent = _as_agent(_load_agent_target(agent))
            typer.echo("Recording runs the configured agent and executes its configured tools.")
            scenario = asyncio.run(record_agent(local_agent, prompt, name=name))
        save_scenario(scenario, output, overwrite=overwrite)
        typer.echo(f"Recorded: {scenario.name} ({len(scenario.recorded_calls)} tool calls)")
        typer.echo(f"Saved unreviewed scenario: {output}")
        typer.echo("Define checks with `suggest --write` or edit the JSON, then run `review`.")
    except Exception as exc:
        _fail(str(exc))


@app.command()
def demo(
    broken: Annotated[bool, typer.Option(help="Seed an ineligible-refund regression.")] = False,
    repeats: Annotated[int, typer.Option(min=1, help="Runs per scenario.")] = 1,
    json_report: Annotated[Path | None, typer.Option(help="Write a JSON report.")] = None,
    junit_report: Annotated[Path | None, typer.Option(help="Write JUnit XML.")] = None,
) -> None:
    """Run offline support-agent examples without credentials or external requests."""
    from .demo import demo_scenarios, make_broken_support_agent, make_support_agent

    try:
        local_agent = make_broken_support_agent() if broken else make_support_agent()
        suite = run_suite_sync(local_agent, demo_scenarios(), repeats=repeats)
        _report(suite, json_report, junit_report)
    except Exception as exc:
        _fail(str(exc))
    if not suite.results:
        _fail("No scenarios ran")
    raise typer.Exit(code=1 if suite.failed_runs else 0)


if __name__ == "__main__":
    app()
