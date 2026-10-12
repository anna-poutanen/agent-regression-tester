# Agent tool-call regression tester — project plan

## Goal and scope

Build a Python package that turns one agent failure into a reviewed, reusable test. The first release supports Pydantic AI function tools. A developer records or imports a trace, edits explicit behavior checks, reviews the scenario, and reruns it against their current agent. Replay substitutes saved tool results while leaving the agent prompt, code, and configured model in place. No model judges are needed for correctness checks.

Our initial users are ourselves: local, deterministic customer-support, document-search, and business-automation agents. These test fixtures validate the workflow and implementation; they do not prove market demand or real-model reliability. Future users are developers maintaining existing Pydantic AI agents with known tool-call failures.

Everything in this build stays on this computer. Prepare CI configuration, but do not create a remote repository, publish a package, contact users, or call paid models. Use a virtual environment within the project. Keep project Git history independent of the parent personal-projects repository.

## Release acceptance criteria

1. `pip install -e '.[dev]'` installs a runnable package and CLI.
2. Record a completed Pydantic AI run and save a versioned JSON scenario; also import serialized messages from an existing run.
3. Suggest observed-call checks as **unreviewed** proposals. Do not infer business policy or automatically treat a recording as correct.
4. Support required calls, forbidden calls, nested argument expectations, call limits, and causal ordering checks.
5. Replay using the agent's original tool definitions and saved responses. Never fall back to real business tools when a fixture is missing or mismatched.
6. Identify all attempted function calls, including malformed arguments and calls that retry or fail.
7. Match fixtures by tool name and exact JSON arguments, consuming duplicate responses in order. Surface fixture exhaustion and mismatches separately from behavior failures.
8. Repeat scenarios and report failures out of N runs; require review by default; fail CI for any failed run.
9. Produce readable terminal output plus stable JSON and JUnit reports with traces and meaningful baseline differences.
10. Ship working offline examples, CLI integration tests, pytest integration, package checks, and a pull-request CI workflow.
11. Document supported versions, safety boundaries, limitations, and the path to live-model evaluation.

## Architecture and contracts

Use a `src/` layout, package `agent_regression`, distribution `agent-regression-tester`, and command `agent-regression`. Use Pydantic v2 for strict versioned artifacts and Pydantic AI public toolset/override APIs for interception. Use pytest and FunctionModel for deterministic multi-turn agent tests. Pin an explicit supported Pydantic AI dependency range after checking the installed APIs.

Modules:

- `schema.py`: JSON-safe `ToolCall`, `ToolFixture`, discriminated check models, `Scenario`, failures, run and suite results.
- `storage.py`: versioned JSON load/save, validation errors, safe overwrite behavior.
- `checks.py`: pure behavior evaluation and trace differences, independent of Pydantic AI.
- `suggest.py`: deterministic draft checks based only on observed behavior.
- `adapter.py`: message extraction, recorder, and replay toolset; tool functions never run during replay.
- `runner.py`: bounded async suite execution, repeats, timeouts, error reporting, and sync entry point.
- `reporting.py`: terminal, JSON, and JUnit output.
- `cli.py`: `record`, `suggest`, `review`, `validate`, `run`, and `demo` workflow; import local `module:factory` agent targets.
- `pytest_plugin.py`: small pytest fixtures for reusing the runner.
- `examples/`: deterministic support, search, and CRM agents with reviewed scenarios.

Schema v1 uses `Scenario(name, prompt, fixtures, recorded_calls, checks, reviewed=False, metadata={})`. Each fixture stores `tool_name`, exact `arguments`, `response`, and response `outcome` (`return` or `retry`). Calls store `tool_name`, raw `arguments`, optional response, outcome, call ID, and model-turn index. A repeated identical call consumes the next identical fixture. Checks are explicit discriminated unions: `required`, `forbidden`, `arguments`, `before`, `max_calls`, `output_contains`. No regex or executable expressions in scenario files. Dotted paths address nested JSON arguments; equality must preserve boolean/number distinctions.

`record(agent, prompt, *, name, deps=None, model=None)` returns an unreviewed Scenario after a real run. Recording executes the tools configured on that agent: examples must use harmless local functions. `scenario_from_messages(...)` imports traces without executing tools. `run_scenario(agent, scenario, *, deps=None, model=None, iteration=1, timeout=30, allow_unreviewed=False)` returns a RunResult. `run_suite(agent_or_factory, scenarios, *, repeats=1, concurrency=1, ...)` returns a SuiteResult. Existing model remains active when no override is supplied. Explicit model overrides support offline tests. The runner must document that user-defined hooks, factories, dependencies, dynamic prompts, and provider-native tools are outside business-tool interception; refuse unsupported configurations where necessary.

## Work order and parallel assignments

### Phase 1 — foundation and exact contracts (first)

Primary agent: inspect environment; write this plan and shared contracts; configure packaging, local virtual environment, and project Git; establish supported framework APIs. Exit: all contributors can import the same data models and agree on function signatures.

### Phase 2 — parallel implementation

Agent A: strict artifact schema, load/save, pure check evaluation, draft suggestions, and trace differences; own unit tests. Pay special attention to wrong IDs, nested arguments, retries, duplicate calls, false vs zero, and simultaneous actions.

Agent B: Pydantic AI adapter, trace recorder, mocked replay, runner; own framework integration tests. Prove tools cannot execute in replay, including missing fixtures and changed toolsets; handle asynchronous functions, RunContext, retries, partial traces, and timeout.

Agent C: terminal/JSON/JUnit reports, CLI commands, pytest fixtures; own CLI/report tests. Ensure documented exit statuses distinguish failing regressions from invalid configuration and no checks silently pass.

Primary agent: examples, full documentation, CI, package integration, independent adversarial tests, and final end-to-end QA. Keep changes in separate owned files to avoid overwrites; integration fixes follow contributor handoff.

### Phase 3 — integration and hardening

Run the safe agent against reviewed scenarios, then swap in broken agents for forbidden refund, wrong order/customer ID, skipped lookup, duplicate action, simultaneous lookup/action, malformed call, missing fixture, fixture exhaustion, and transient retry. Run repeated suites and verify fresh fixture state. Exercise the installed CLI from another directory and build/install the wheel in a clean environment.

### Phase 4 — release-ready local handoff

Add a short quickstart, architecture explanation, scenario reference, troubleshooting, and contributor guide. Prepare GitHub Actions for Python-version compatibility, lint, typing, tests, and packaging. Record actual validation results and limitations in a build-status document. Leave reproducible commands and a prioritized backlog. Do not claim external validation or publish remotely.

## Testing strategy

### Pure unit tests

- Schema round-trips, unsupported versions, unknown fields, bad types, empty/malformed checks, invalid counts and paths.
- Required/forbidden subset matching, nested values, strict JSON equality, absent arguments, all-occurrence argument checks, ordering, limits, and output checks.
- Duplicate fixture consumption, deterministic suggestions, trace differences, report escaping and totals.

### Framework integration tests

- FunctionModel drives realistic multi-turn Pydantic AI agent loops.
- Sync and async tools plus RunContext/dependencies preserve tool schemas while responses are mocked.
- A sentinel in every original tool raises if replay touches it.
- Missing/wrong fixtures never execute original tools; invalid tool arguments and retry requests remain visible.
- Review gating, fresh state per repetition, concurrent scenario isolation, timeout cleanup, model request limits, and errors produce failing results rather than crashing the whole suite.

### Product tests

- Record → suggest → edit/review → validate → run → inspect report workflow.
- Good support, search and CRM agents pass. Broken agents fail for the intended reason with tool name, expected behavior, observed arguments, and failing run count.
- CLI exit 0 for pass, 1 for regressions, 2 for usage/configuration problems.
- JSON and JUnit reports parse correctly; suite totals agree with terminal output.
- Wheel/sdist build; installed examples work without credentials, internet, or source checkout imports.

### Validation gates

Use Ruff lint/format checks, mypy for package source, pytest with coverage, deterministic example suites, CLI smoke tests, and a clean wheel installation. Run focused tests while developing and the full suite after integration. Broaden tests only to resolve newly found failures. Coverage is diagnostic; meaningful safety and behavior tests take priority over a percentage target.

Live-model testing is a later opt-in gate: keep saved tools mocked, run each scenario 10–20 times, record model/prompt metadata and failure frequency, and assess whether checks detect known prompt mutations. Unit tests cannot establish how a real model follows a prompt.

## Six-to-eight-week roadmap after this local build

| Week | Work | Evidence of completion |
| --- | --- | --- |
| 1 | Recording/import, artifact format, reviewed assertions | One historical trace becomes an editable local test |
| 2 | Strict replay and behavior evaluation | Forbidden and incorrect actions fail without side effects |
| 3 | CLI, repeat runs, reports, pytest | One command explains a seeded regression |
| 4 | Offline examples, adversarial tests, compatibility, packaging | Clean install and CI-ready suite |
| 5 | Opt-in real-model runs; prompt/model mutation experiments | Failure frequencies and limits documented |
| 6 | Developer onboarding and documentation; candidate interviews | Three existing-agent developers identified, no adoption claims |
| 7–8 | Pilot integration, API refinements, first public release | Developers reproduce a real failure and rerun tests after changes |

## Decisions, risks, and deferred scope

- Initial replay uses exact argument matching. If a changed agent needs a different tool response, a developer must add a fixture; no fabricated answers or live fallback.
- Draft suggestions preserve observations, including possible mistakes. Business policies are human-authored and reviewed.
- Parallel tool calls do not prove a lookup happened before an action. Ordering uses model turns as well as call positions.
- A captured fixture is a fixed world snapshot, not a simulator. Stateful workflows need explicitly ordered responses.
- Scenarios may contain customer data. Document sanitization and keep secrets out of committed examples.
- Importing agent targets executes local Python code. Tool interception does not sandbox arbitrary agent/application code.
- Provider-native tools, remote MCP toolsets, streaming, multimodal input, deferred tools, durable execution, and multi-agent orchestration are out of the first supported scope.
- Defer hosted dashboards, automatic production ingestion, model-judged policies, other agent frameworks, and publishing until the workflow is stable.
- No external users yet: assess implementation with test agents and later validate usefulness with real failures.

## Progress tracking

See `docs/BUILD_STATUS.md` for completed components, exact validation results, remaining limitations, and recommended next steps. Update this plan only when implementation reveals a material change to scope or architecture.

### Support application milestone

The realistic local support application and opt-in real-model path are now built.
They provide actual backend actions, independent policy assertions, mutation
coverage, synthetic training-data generation and a bounded local LoRA experiment.
The [support guide](support/README.md) describes the workflow, and [evaluation evidence](support/EVALUATION.md)
separates deterministic testing from observed model reliability. Training remains
optional; the tester's value is exposing behavioral regressions regardless of the
model selected.

Next product work should add more independent failures and unseen request wording,
review final-answer claims, and improve realistic multi-turn coverage before
external pilots. A successful synthetic model run cannot replace testing one
existing developer application when the project is ready for users. No outreach,
publication or hosted deployment has occurred.
