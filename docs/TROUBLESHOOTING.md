# Troubleshooting

## A run fails with a fixture mismatch

Read the observed tool name and arguments in the report. Fixtures use the complete argument object. A changed ID, omitted default argument, new extra field, or different JSON type requires an explicit response fixture. Determine whether the change is appropriate before editing fixtures or checks. Replay never falls back to live tools.

## A duplicate call exhausts its fixture

Supply a separate matching fixture for each allowed repeated call. If the duplicate is unwanted, use `max_calls` and keep the missing-response failure. Fixtures with identical arguments are consumed in the order they appear in the scenario, independently of other tools' fixtures.

## A recorded scenario is rejected as unreviewed

Use `suggest --write` to create observed-call proposals, edit the JSON to express the actual business policy, and run `review` after inspecting it. `--allow-unreviewed` permits exploratory replay with checks; it does not add missing business rules. Scenarios without checks never pass by default.

## A `before` check fails despite the call list order

Calls emitted by the model in one parallel batch share a turn. A lookup must return in an earlier model turn to establish causality. Also check whether the earlier lookup returned successfully: retries, blocked calls, and errors do not establish a prerequisite.

## The agent module cannot be imported

Run in the same virtual environment as your application. Install your application package or add its source to Python's import path. Targets are `module:agent` or `module:make_agent`; CLI factories must be synchronous and take no arguments. The Python API accepts async factories and an explicit dependency object. Importing a module executes trusted local Python code.

## The framework configuration is unsupported

The initial release supports Pydantic AI 2.55.x, string outputs, and in-process function toolsets. Custom capabilities, dynamic/remote toolsets, provider-native tools, deferred/approval tools, and structured output are rejected to protect the interception boundary. Ordinary tools registered with `@agent.tool`, `@agent.tool_plain`, or FunctionToolset are supported. Some built-in wrappers around function toolsets are supported. Keep unsupported orchestration outside the agent under test or wait for a dedicated adapter.

## Recording runs a tool I intended to mock

Recording deliberately runs the configured tools. Replay substitutes saved responses. To avoid running tools while creating a scenario, import `result.all_messages_json()` from an existing run with `record --messages`, or hand-author a JSON scenario. The committed examples only compute synthetic local results.

## Invalid arguments remain failures after the agent corrects them

The report captures every attempted call, including validation retries. The initial replay treats attempts that cannot reach fixture interception as replay errors, even if the agent later corrects them. Business-tool `ModelRetry` responses with valid arguments can be saved and replayed. Decide whether a corrected invalid attempt should be acceptable before weakening a behavior check; tolerating schema-validation retries is a future policy option.

## Pytest fixtures are missing

Install the package into the same environment running pytest (`python -m pip install -e '.[dev]'`). Pytest discovers its plugin through the package entry point. If plugin autoload is disabled, explicitly load `-p agent_regression.pytest_plugin`.

## Provider requests or unrelated Python code execute in replay

Only business function-tool bodies are intercepted. Your configured model remains active, and factories, dependencies, prompt functions, preparation callbacks, validators, and hooks are ordinary application code. Use FunctionModel for credential-free deterministic tests. The runner is not a process/network sandbox. Do not configure provider credentials in offline package tests.

## A run times out

The runner bounds asynchronous runs and factories, and limits model requests to 20 and successful tool calls to 100. A blocking synchronous factory or model callback can block the event loop; use nonblocking code. Increase `--timeout` only when the longer operation is expected. Partial traces are reported where available.
