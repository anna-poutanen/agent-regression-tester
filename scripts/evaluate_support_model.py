"""Bounded local-model evaluation with complete reports and configuration provenance."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from agent_regression import run_suite
from agent_regression.reporting import render_terminal, write_json_report, write_junit_report
from agent_regression.support import SupportSession, close_local_model, make_local_model
from agent_regression.support import agent as support_agent_module
from agent_regression.support.agent import make_support_agent
from agent_regression.support.evaluation import diagnostic_scenarios
from agent_regression.support.local_model import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME
from agent_regression.support.prompt import SAFE_INSTRUCTIONS, instructions_for
from agent_regression.support.scenarios import support_scenarios
from agent_regression.support.training import training_tool_definitions


async def evaluate(arguments: argparse.Namespace) -> int:
    targeted_only = getattr(arguments, "targeted_only", False)
    include_targeted = getattr(arguments, "targeted", False) or targeted_only
    scenarios = [] if targeted_only else support_scenarios()
    if arguments.diagnostic and not targeted_only:
        scenarios += diagnostic_scenarios()
    if include_targeted:
        from agent_regression.support.targeted import targeted_scenarios

        scenarios += targeted_scenarios()
    agent_source = await asyncio.to_thread(Path(support_agent_module.__file__).read_bytes)
    tools = training_tool_definitions()
    manifest = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "pydantic_ai": version("pydantic-ai-slim"),
        "base_url": arguments.base_url,
        "model_name": arguments.model_name,
        "model_label": arguments.label,
        "temperature": arguments.temperature,
        "repeats": arguments.repeats,
        "timeout_seconds": arguments.timeout,
        "prompt_variant": arguments.prompt_variant,
        "instruction_layout": support_agent_module.INSTRUCTION_LAYOUT,
        "agent_source_sha256": hashlib.sha256(agent_source).hexdigest(),
        "safe_prompt_sha256": hashlib.sha256(SAFE_INSTRUCTIONS.encode()).hexdigest(),
        "actual_prompt_sha256": hashlib.sha256(
            instructions_for(arguments.prompt_variant).encode()
        ).hexdigest(),
        "tool_schemas_sha256": hashlib.sha256(
            json.dumps(tools, sort_keys=True).encode()
        ).hexdigest(),
        "tool_schema_order_sha256": hashlib.sha256(
            json.dumps(tools, ensure_ascii=False).encode()
        ).hexdigest(),
        "scenarios_sha256": hashlib.sha256(
            json.dumps(
                [item.model_dump(mode="json") for item in scenarios], sort_keys=True
            ).encode()
        ).hexdigest(),
        "scenario_names": [item.name for item in scenarios],
        "purpose": "synthetic development evaluation; not a representative production benchmark",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    model = make_local_model(
        base_url=arguments.base_url,
        model_name=arguments.model_name,
        temperature=arguments.temperature,
    )
    try:
        suite = await run_suite(
            lambda: make_support_agent(
                model, session=SupportSession(), prompt_variant=arguments.prompt_variant
            ),
            scenarios,
            repeats=arguments.repeats,
            timeout=arguments.timeout,
        )
    finally:
        await close_local_model(model)
    write_json_report(suite, arguments.output)
    write_junit_report(suite, arguments.output.with_suffix(".xml"))
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["total_runs"] = suite.total_runs
    manifest["passed_runs"] = suite.passed_runs
    manifest["failed_runs"] = suite.failed_runs
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    render_terminal(suite)
    print(f"Reports: {arguments.output}, {manifest_path}")
    return 1 if suite.failed_runs else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--label", required=True, help="Model/adapter provenance label for the report"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--prompt-variant", default="safe")
    parser.add_argument("--diagnostic", action="store_true", help="Add 12 new diagnostic wordings")
    parser.add_argument(
        "--targeted", action="store_true", help="Add reviewed partial-refund and ambiguity cases"
    )
    parser.add_argument(
        "--targeted-only", action="store_true", help="Run only the reviewed intent cases"
    )
    arguments = parser.parse_args()
    if arguments.targeted_only and arguments.diagnostic:
        parser.error("--targeted-only cannot be combined with --diagnostic")
    if arguments.repeats < 1 or arguments.timeout <= 0:
        parser.error("Repeats and timeout must be positive")
    if arguments.output.suffix != ".json":
        parser.error("Output must be a .json report path")
    if any(
        path.exists()
        for path in (
            arguments.output,
            arguments.output.with_suffix(".manifest.json"),
            arguments.output.with_suffix(".xml"),
        )
    ):
        parser.error("Output already exists; choose a new report path")
    raise SystemExit(asyncio.run(evaluate(arguments)))


if __name__ == "__main__":
    main()
