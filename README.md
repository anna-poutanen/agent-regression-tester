# Agent Regression Tester

Turn an agent tool-call failure into a reviewed regression test. Record a Pydantic AI trace, write behavior checks, and replay the current agent against saved tool responses. The report shows forbidden or missing calls, wrong arguments, ordering problems, and differences from the recorded trace.

**Local alpha:** supports Pydantic AI 2.55.x, text-output agents, and in-process function tools. Includes deterministic test agents for support, document search, and CRM workflows. No provider credentials are needed to try it.

## Try it locally

Python 3.11+ is required.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
agent-regression demo --repeats 3
agent-regression demo --broken
```

The safe demo passes. The broken demo intentionally refunds an ineligible order and exits 1:

```text
FAIL: ineligible_order — failed 1 of 1 runs
  Run 1
    forbidden_call (issue_refund): Forbidden issue_refund call with arguments {"order_id": "123"}
      Expected: "no issue_refund call"
      Observed: {"order_id": "123"}
```

Business-tool bodies do not execute during replay. Saved fixtures supply the responses, including a synthetic response for the forbidden refund so the trace can finish. Model/provider requests still occur if you use a configured live model. The demo uses FunctionModel, a deterministic script.

## Record → review → replay

Record the harmless local support example:

```sh
agent-regression record \
  --agent agent_regression.demo:make_support_agent \
  --prompt 'Refund order 123 if it is eligible.' \
  --name ineligible_order \
  --output reports/ineligible_order.json
agent-regression suggest reports/ineligible_order.json --write
```

Recording executes the configured tools. For an agent with real side effects, import messages from an existing run instead:

```sh
agent-regression record \
  --messages saved-messages.json \
  --prompt 'Refund order 123 if it is eligible.' \
  --name ineligible_order \
  --output reports/ineligible_order.json
```

Suggestions describe what happened; they cannot decide whether it was correct. Edit the scenario's `checks` to express the intended policy. For this example, require the right order lookup and forbid refunds:

```json
"checks": [
  {"kind": "required", "tool_name": "lookup_order", "arguments": {"order_id": "123"}},
  {"kind": "arguments", "tool_name": "lookup_order", "path": "order_id", "expected": "123"},
  {"kind": "forbidden", "tool_name": "issue_refund"}
]
```

After reviewing those checks and saved responses, mark the scenario reviewed and run it:

```sh
agent-regression review reports/ineligible_order.json --yes
agent-regression validate reports/ineligible_order.json
agent-regression run reports/ineligible_order.json \
  --agent agent_regression.demo:make_support_agent \
  --repeats 10 \
  --json-report reports/results.json \
  --junit-report reports/results.xml
```

For your application, `--agent your_package.support:agent` imports an existing Agent, or `--agent your_package.support:make_agent` calls a zero-argument factory. Factories provide fresh agent state per run. Install your application or make its module importable in the same environment. Imported Python code must be trusted.

The default runner preserves your agent's current instructions, tool schemas, code, and model. It mocks the tools' responses. Install the appropriate optional provider extra, such as `python -m pip install -e '.[openai]'`, only when you want to run against a real model. Credentials and dependency setup remain your application's responsibility. Agents requiring dependencies can use the Python API below.

## Python API

```python
import asyncio
from agent_regression import load_scenario, run_scenario
from agent_regression.demo import make_support_agent


async def main():
    scenario = load_scenario("examples/scenarios/support/ineligible_order.json")
    result = await run_scenario(make_support_agent(), scenario)
    assert result.passed, result.model_dump_json(indent=2)


asyncio.run(main())
```

Record and import from Python:

```python
from agent_regression import record, save_scenario, scenario_from_messages

# Inside an async function; the configured tool functions execute here.
scenario = await record(agent, "Refund order 123 if eligible.", name="ineligible_order", deps=deps)
save_scenario(scenario, "scenarios/ineligible_order.json")

# Importing serialized messages executes no tools.
scenario = scenario_from_messages(
    result.all_messages_json(), name="ineligible_order", prompt="Refund order 123 if eligible."
)
```

Pass `deps=...` or an explicit `model=...` override to `run_scenario` or `run_suite`. For repeated runs, use `await run_suite(make_agent, scenarios, repeats=10, concurrency=2)`; the synchronous equivalent is `run_suite_sync`. Fixture consumption is fresh for each run. Shared dependency objects and user callbacks remain your application's responsibility.

Pytest fixtures are registered automatically when the package is installed:

```python
import pytest
from agent_regression import load_scenario
from agent_regression.demo import make_support_agent


@pytest.mark.asyncio
async def test_support_agent(regression_runner):
    scenario = load_scenario("examples/scenarios/support/ineligible_order.json")
    result = await regression_runner(make_support_agent(), scenario)
    assert result.passed, result.model_dump_json(indent=2)
```

## Checks and fixtures

Supported checks are `required`, `forbidden`, `arguments`, `before`, `max_calls`, and `output_contains`. Argument checks support nested paths and apply to every matching attempt. Parallel lookup/action calls cannot satisfy a `before` check. Required calls count attempts; replay errors still fail separately.

Fixtures match the tool name and exact JSON arguments; repeated identical calls consume matching responses in order. A missing fixture, changed ID, invalid arguments, or exhausted fixture fails the run and never falls back to the original tool. An informational baseline difference does not fail a run by itself; explicit checks define correct behavior.

See the [scenario reference](docs/SCENARIO_FORMAT.md) for fields, matching rules, retries, and limitations.

## Reports and CI

The CLI returns 0 for passing runs, 1 for regressions/runtime failures, and 2 for usage or configuration errors. Reports include the complete trace and each failing iteration. Ten repeated runs produce a count such as “failed 3 of 10 runs”; scripted offline tests do not measure real-model variability.

Run the reviewed example files:

```sh
agent-regression run examples/scenarios/support \
  --agent agent_regression.demo:make_support_agent --repeats 3
agent-regression run examples/scenarios/search \
  --agent agent_regression.demo:make_search_agent
agent-regression run examples/scenarios/crm \
  --agent agent_regression.demo:make_crm_agent
```

The prepared [GitHub Actions workflow](.github/workflows/ci.yml) tests Python 3.11–3.13 and runs the support scenarios on pull requests. It has not been run on GitHub. To use the tester in another project, install this package, add reviewed scenarios, and run `agent-regression run ... --agent your_module:make_agent --junit-report reports/regressions.xml` in that project's CI. Live-model runs require explicitly configured credentials and may incur provider costs.

## Development and current limits

```sh
make check   # lint, format, typing, tests and coverage
make build   # local wheel and source distribution
```

The current build supports one text prompt per scenario, JSON-compatible saved responses, and string outputs. Provider-native tools, remote/dynamic toolsets, custom capabilities, deferred/approval tools, streaming, multimodal inputs, and structured output agents are outside the initial supported scope. Replay intercepts function tools; it is not a sandbox for factories, prompts, hooks, validators, dependencies, or arbitrary application code. Recording and replay have different execution boundaries.

The package has no telemetry, hosted service, automatic publishing, or external user validation. All example data is synthetic. Review saved data before committing it.

- [Full project plan and six-to-eight-week roadmap](docs/PROJECT_PLAN.md)
- [Build status and validation evidence](docs/BUILD_STATUS.md)
- [Architecture and safety boundary](docs/ARCHITECTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Contributor guide](CONTRIBUTING.md)

MIT licensed.
