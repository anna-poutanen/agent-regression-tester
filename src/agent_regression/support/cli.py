"""Local support-agent conversations and policy regression demos."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from ..adapter import scenario_from_messages
from ..reporting import render_terminal, write_json_report, write_junit_report
from ..runner import run_suite
from ..storage import save_scenario
from .agent import make_support_agent
from .local_model import DEFAULT_MODEL_NAME, close_local_model, make_local_model
from .offline import make_offline_model
from .prompt import instructions_for
from .scenarios import support_scenarios
from .store import SupportSession

app = typer.Typer(help="Run the realistic support agent with synthetic local orders.")


def _fail(message: str, *, code: int = 2) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=code)


def _agent(
    session: SupportSession,
    mutation: str,
    base_url: str | None,
    model_name: str,
    temperature: float,
    prompt_variant: str = "safe",
) -> Agent[Any, str]:
    instructions_for(prompt_variant)
    if base_url is not None:
        if mutation != "safe":
            raise ValueError(
                "--mutation applies to the offline test double; omit it for real models"
            )
        model = make_local_model(base_url=base_url, model_name=model_name, temperature=temperature)
    else:
        if prompt_variant != "safe":
            raise ValueError("Use --mutation for offline faults; --prompt-variant needs --base-url")
        model = make_offline_model(mutation=mutation, request_id=session.request_id)
    return make_support_agent(model, session=session, prompt_variant=prompt_variant)


@app.command()
def chat(
    prompt: Annotated[str, typer.Argument(help="Customer request; all actions use local data.")],
    mutation: Annotated[str, typer.Option(help="Offline test-double mutation.")] = "safe",
    prompt_variant: Annotated[
        str, typer.Option(help="Real-model prompt variant: safe or a named seeded fault.")
    ] = "safe",
    base_url: Annotated[
        str | None,
        typer.Option(help="Use a real model at a loopback server, e.g. http://127.0.0.1:8080/v1."),
    ] = None,
    model_name: Annotated[
        str, typer.Option(help="Model identifier at the local server.")
    ] = DEFAULT_MODEL_NAME,
    temperature: Annotated[float, typer.Option(min=0, max=2)] = 0.2,
    request_id: Annotated[
        str, typer.Option(help="Trusted idempotency key for this request.")
    ] = "req-support-demo",
    timeout: Annotated[float, typer.Option(min=0.1, help="Conversation timeout in seconds.")] = 120,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print response, calls and local ledgers as JSON.")
    ] = False,
    record: Annotated[
        Path | None, typer.Option(help="Save an unreviewed scenario from this conversation.")
    ] = None,
) -> None:
    """Talk to the agent and inspect actual local tool results and side effects."""
    if record is not None and record.exists():
        _fail(f"{record} already exists; choose a new recording path")
    try:
        session = SupportSession(request_id=request_id, user_message=prompt)
        agent = _agent(session, mutation, base_url, model_name, temperature, prompt_variant)
    except Exception as exc:
        _fail(str(exc) or type(exc).__name__)

    try:

        async def converse() -> Any:
            try:
                return await asyncio.wait_for(
                    agent.run(
                        prompt, usage_limits=UsageLimits(request_limit=20, tool_calls_limit=100)
                    ),
                    timeout=timeout,
                )
            finally:
                if base_url and isinstance(agent.model, Model):
                    await close_local_model(agent.model)

        result = asyncio.run(converse())
        scenario = scenario_from_messages(
            result.all_messages(), name="support_recording", prompt=prompt, output=result.output
        )
        payload = {
            "model_mode": "local_language_model" if base_url else "offline_test_double",
            "output": result.output,
            "calls": [call.model_dump(mode="json") for call in scenario.recorded_calls],
            "refunds": session.store.refunds,
            "escalations": session.store.escalations,
        }
        if record is not None:
            scenario.metadata = {
                "provenance": "synthetic_local_support",
                "model_mode": payload["model_mode"],
            }
            save_scenario(scenario, record)
            payload["recording"] = str(record)
        if json_output:
            typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            typer.echo(f"Model mode: {payload['model_mode']}")
            for call in scenario.recorded_calls:
                typer.echo(f"{call.tool_name}({json.dumps(call.arguments, sort_keys=True)})")
                typer.echo(f"  → {json.dumps(call.response, sort_keys=True)}")
            typer.echo(f"\nAgent: {result.output}")
            typer.echo(
                f"Local ledgers: {len(session.store.refunds)} refunds, "
                f"{len(session.store.escalations)} tickets"
            )
        if record is not None and not json_output:
            typer.echo(f"Saved unreviewed recording: {record}")
    except Exception as exc:
        _fail(str(exc) or type(exc).__name__, code=1)


@app.command()
def demo(
    mutation: Annotated[str, typer.Option(help="Offline test-double mutation.")] = "safe",
    prompt_variant: Annotated[
        str, typer.Option(help="Real-model prompt variant: safe or a named seeded fault.")
    ] = "safe",
    repeats: Annotated[int, typer.Option(min=1)] = 1,
    scenario: Annotated[
        list[str] | None,
        typer.Option(help="Select scenario names; repeat the option for multiple."),
    ] = None,
    base_url: Annotated[
        str | None, typer.Option(help="Use a real model at a loopback server.")
    ] = None,
    model_name: Annotated[str, typer.Option()] = DEFAULT_MODEL_NAME,
    temperature: Annotated[float, typer.Option(min=0, max=2)] = 0.2,
    timeout: Annotated[float, typer.Option(min=0.1)] = 120,
    json_report: Annotated[Path | None, typer.Option()] = None,
    junit_report: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Replay reviewed policy scenarios; exit 1 when any agent behavior fails."""
    try:
        scenarios = support_scenarios()
        if scenario:
            unknown = set(scenario) - {item.name for item in scenarios}
            if unknown:
                raise ValueError(f"Unknown scenarios: {', '.join(sorted(unknown))}")
            scenarios = [item for item in scenarios if item.name in scenario]
        # Validate configuration before starting runs so a typo exits as configuration error.
        model_agent = _agent(
            SupportSession(), mutation, base_url, model_name, temperature, prompt_variant
        )

        def factory() -> Agent[Any, str]:
            return make_support_agent(
                model_agent.model, session=SupportSession(), prompt_variant=prompt_variant
            )

        typer.echo(f"Model mode: {'local_language_model' if base_url else 'offline_test_double'}")

        async def replay() -> Any:
            try:
                return await run_suite(factory, scenarios, repeats=repeats, timeout=timeout)
            finally:
                if base_url and isinstance(model_agent.model, Model):
                    await close_local_model(model_agent.model)

        suite = asyncio.run(replay())
        render_terminal(suite)
        if json_report:
            write_json_report(suite, json_report)
            typer.echo(f"JSON report: {json_report}")
        if junit_report:
            write_junit_report(suite, junit_report)
            typer.echo(f"JUnit report: {junit_report}")
    except Exception as exc:
        _fail(str(exc) or type(exc).__name__)
    raise typer.Exit(code=1 if suite.failed_runs else 0)


@app.command("scenarios")
def scenarios_command(
    export: Annotated[
        Path | None, typer.Option(help="Export reviewed JSON artifacts to a new directory.")
    ] = None,
) -> None:
    """List the synthetic evaluation scenarios or export them for editing."""
    try:
        scenarios = support_scenarios()
        if export is not None:
            paths = [export / f"{item.name}.json" for item in scenarios]
            if any(path.exists() for path in paths):
                raise ValueError(
                    "Export would replace an existing scenario; choose a new directory"
                )
            for item, path in zip(scenarios, paths, strict=True):
                save_scenario(item, path)
        for item in scenarios:
            typer.echo(f"{item.name}: {item.prompt}")
        typer.echo(f"{len(scenarios)} reviewed synthetic scenarios")
    except Exception as exc:
        _fail(str(exc))
