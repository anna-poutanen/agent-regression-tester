# Local build status — October 21, 2026

## Short, resumable runs

The package now supports durable per-iteration checkpoints, explicit resume,
per-invocation run/time limits, and short sequential batches with optional pauses.
Completed passes and failures are retained exactly; resume starts at the first
iteration without a completed result. Configuration/scenario drift and concurrent
writers are rejected. Atomic checkpoint writes keep the last committed prefix.
The declared configuration ID remains the caller's responsibility for arbitrary
application code, prompts and model servers.

Incomplete terminal/JSON reports show completed, planned and remaining counts.
JUnit adds an `IncompleteSuite` error marker. The CLI exits 3 for an incomplete
bounded run and 130 for Ctrl+C; a complete failing run still exits 1. An actual
subprocess signal test proves cancellation saves progress and resume does not
repeat the completed iteration. A cooldown-cancellation test also verifies that
earlier failures remain visible. The deadline boundary fix preserves a completed
result even when the clock crosses the limit as the callback returns.

Validation used only offline agents and blocked provider requests:

- Python 3.13.2 and 3.11.14: **988 passed, 7 optional checks skipped** on each.
- Ruff lint and formatting pass for **78 Python files**; strict mypy passes for
  **24 source modules**.
- Combined statement/branch coverage: **93%**. Coverage remains diagnostic;
  interruption, identity, storage and failure-preservation behavior are tested.
- Wheel and sdist build successfully. A clean installed wheel, invoked outside
  the checkout, passes short-run/resume checks, refuses changed agent identity,
  preserves earlier failed traces exactly, and returns the expected exit codes.
- The safe installed demo resumes from 2/6 to 6/6 passes. The broken installed
  demo retains its two saved failures and finishes with all three forbidden-refund
  failures, with no duplicate scenario/iteration results.

No model server, inference, training or weight download was started. The earlier
real-model results below remain unchanged. Async cancellation cannot forcibly
interrupt blocking synchronous callbacks or stop an independently managed server.

See [resumable runs](RESUMABLE_RUNS.md) for commands, configuration identity,
interruption semantics, and reporting details. Next product priorities are
saved-report comparison, multi-turn clarification, and receipt-based answer checks.

---

# Local build status — October 12, 2026

## Partial-refund and ambiguity follow-up

The current prompt distinguishes a confirmed order from consent to a full refund.
Fourteen new reviewed development scenarios cover partial amounts, percentages,
uncertain or competing references, and two valid full-refund controls. Original
34-case expectations and the new 14-case suite remain frozen. A separate answer
regression catches a clarification that asks for an already supplied order ID.
Replay keeps business tools mocked; direct validation uses harmless fresh stores.
The evaluator can select targeted cases and records prompt, schema and scenario
hashes. The optional local model receives the request directly, without the
offline intent parser routing or hiding its tool calls.

The deterministic demo also now handles explicit waits and negated human or
exception requests. These bounded parser fixes apply only to the offline test
double. They are tested separately from real inference.

The integrated V14 policy with the pinned local 8B checkpoint in reasoning mode
passed the complete **70/70** targeted replay: **30/30** partial-only requests,
**30/30** ambiguous references and **10/10** authorized full-refund controls.
The fresh calibrated baseline passed **17/70**. All 60 clarification runs used
zero tools; the ten controls performed the exact lookup and refund. The original
suite improved from **72/102** to **90/102**: **59/66** canonical and **31/36**
diagnostic runs pass. Twelve failures remain, including incorrect or premature
handoffs, missing actions and reasoning-token exhaustion. Both separate
partial-refund regressions pass **3/3** with zero tools and full-refund consent
questions. Some passing replies also contain unsupported promises or policy
commentary. The actual-store run was stopped when the user reported heat and
fan activity: **15 of 45** planned conversations completed, all passing; the
remaining 30 have no completed result. This ordered prefix is an incomplete
direct check, not a full-suite score. The owned inference server and evaluation
processes were stopped, and
port 8080 was verified closed. This change combines revised instructions and a
new model configuration; it is not evidence of improved training or a general
robustness guarantee.

Fresh calibrated-model baseline and prompt/model candidate results are documented
in [support evaluation evidence](support/EVALUATION.md). Prompt candidates are
rejected when they hide valid actions or leave the original partial-amount failure
unfixed. These are synthetic development measurements after prompt selection;
they do not establish production reliability or comprehensive answer correctness.

Existing 4B model weights, adapters and training files remain unchanged. The
separate 8B candidate uses newly downloaded public pretrained weights with no
adapter. Fresh corpus exports are versioned native-v3 and hash the revised
instructions; the completed native-v2 training corpus and adapter remain
preserved locally. Training has not been rerun for these prompt changes.

Current source validation on macOS:

- Python 3.13.2 and 3.11.14 core environments: **844 passed, 7 optional checks skipped**
  on each version.
- Python 3.13.2 model environment: **851 passed**, including optional transport and
  local-tokenizer tests. Unit tests block live provider calls.
- **96%** combined statement/branch coverage, including plugin startup.
- Ruff lint and formatting pass for **71 Python files**; strict mypy passes for
  **23 source modules**.
- Offline policy replay: **66/66** original and **42/42** targeted runs pass.
- Clean installed wheel: **50/50** offline scenarios pass outside the checkout;
  bundled data and v3 corpus generation work without OpenAI or MLX. Both wheel
  and sdist include the new source, tests, guides and reviewed scenarios.
- Published-source scan excludes credentials, weights, generated corpora, raw
  reports, virtual environments and build artifacts.

The private source repository is
[anna-poutanen/agent-regression-tester](https://github.com/anna-poutanen/agent-regression-tester).
[GitHub CI run 38279240003](https://github.com/anna-poutanen/agent-regression-tester/actions/runs/38279240003)
passed all four jobs for source commit `e9c368b`: **844 passed and 7 skipped**
on each Linux/Python 3.11–3.13 job, plus lint, types, package builds and reviewed
examples, including detection of the seeded failures. Downloaded JUnit artifacts
agree with those totals. Live model inference and training remain opt-in local
experiments.

The October 11 milestone below is retained as measured history; its counts and
unrun-CI statements describe that snapshot.

---

# Local build status — October 11, 2026

## Realistic support milestone

The support application now includes a Pydantic AI agent, nine synthetic orders,
guarded local refunds and human tickets, 22 reviewed policy scenarios, and twelve
additional diagnostic requests. It supports an offline test double and opt-in
loopback model inference. All eight seeded behavior faults are detected; replay
does not execute the store. The actual backend checks authorization, lookup,
eligibility, money and idempotency independently of the prompt.

The synthetic training generator supplies 384/48/48 train/validation/test decision
records with disjoint identifiers, request keys and wording. Native-v2 export
matches scripted expert inference prompt strings and tokens across all 480 decisions.
Separate review
re-executed all 210 conversations against the backend and checked exclusion of
evaluation cases. Training uses a local LoRA adapter with frozen Qwen base weights;
the [training guide](support/TRAINING.md) documents native token masking, manifests
and optional dependencies. No real customer data or paid model API is involved.

Real-model evaluation already exposed missing actions and guessed orders. The
initial model passed 14/22 runs; a revised prompt passed 45/66. The first paired
comparison passed 60/102 with the base and 54/102 with the initial adapter:
lower training loss did not establish better behavior. The second native-format
comparison passed 57/102 with its matched base and 72/102 with the adapter, but
the adapter attempted zero refunds and failed all 21 refund-required runs.
Reducing inference scale from 20 to 5, with unchanged trained weights, passed
73/102 overall and 13/21 refund-required runs. Ambiguity, missed actions and intent
violations remain. These are synthetic development results, not a production
benchmark. See [evaluation evidence](support/EVALUATION.md)
for experiment configuration, comparison results and limits.

Current source validation passes on macOS:

- **Python 3.13.2 and 3.11.14, core environment:** 705 tests pass on each version;
  seven optional transport/tokenizer checks skip without model dependencies.
- **Python 3.13.2, local-model environment:** all 712 tests pass, including those
  seven checks. These use mocked HTTP and local tokenizer files, not inference.
- **Coverage:** 96% combined statement/branch coverage.
- **Ruff:** all 66 Python files pass lint and formatting checks.
- **Mypy:** strict checks pass for all 22 package source modules.
- **Offline policy replay:** 66/66 runs pass; all eight intentionally broken
  behaviors fail their targeted checks.
- **Actual model failure reproduction:** a separately reviewed partial-only case
  catches the calibrated model's unwanted full refund in 3/3 runs. Safe offline
  replay passes 3/3. The original failing recording stays unreviewed; the case is
  excluded from historical 34-scenario scores.
- **Direct calibrated model:** 12/17 basic ledger/sequence checks pass. Two
  partial-only requests create full refunds; ambiguity and order-bound human help
  also fail. A currency error in a basic passing reply confirms that tool checks
  do not comprehensively grade final answers.
- **Clean wheel installation:** 44/44 replay runs pass from outside the checkout;
  bundled orders, diagnostic scenarios and native-v2 corpus generation work
  without OpenAI or MLX installed. Eligible chat creates one 4,999-cent refund;
  outside-window chat creates none.
- **Installed user workflow:** record, suggest, review, validate and five-repeat
  replay pass for eligible and ineligible requests. Unreviewed replay is rejected.
  Wrong-ID, duplicate-action and ambiguous-request mutations each fail 3/3 with
  matching terminal, JSON and JUnit results. Nineteen additional replays leave
  the actual store and lookup audit unchanged with original methods patched to
  raise if executed.
- **Packaging:** wheel and sdist include source, seed, typing marker and guides.
  The sdist includes experiment scripts and tests. Weights, generated corpora,
  reports and virtual environments are excluded.

At this October 11 snapshot, the prepared CI workflow included the new support
suite and a known failing refund mutation, but had not run remotely; Python 3.12
and Linux were still CI targets. The October 12 section records completed CI.

The original build snapshot below is retained for historical context. Current
source validation and the final model comparison are recorded with this milestone;
the earlier statement that no real model ran applies only to that snapshot.

## Initial alpha snapshot

The first local alpha is implemented. Three agents built the artifact/check engine, framework adapter/runner, and CLI/reporting in parallel; the primary agent integrated examples, documentation, CI, and packaging. A separate adversarial review added regression tests for the replay boundary. No remote repository, package publication, provider-model calls, or user outreach was performed.

## Completed

| Component | Result |
| --- | --- |
| Python package | `agent-regression-tester` 0.1.0, `src/agent_regression`, typed public API, MIT license |
| Artifact format | Strict versioned JSON, JSON Schema reference, duplicate-key rejection, atomic saves and overwrite protection |
| Recorder/importer | Async agent recording and import of Pydantic AI serialized messages; unreviewed draft scenarios |
| Check suggestions | Deterministic required-call/argument proposals based on observations; no inferred business policy |
| Behavior checks | Required/forbidden calls, nested argument operators, causal ordering, maximum calls, output substring |
| Replay | Public Pydantic AI override/toolset integration, original schemas, exact saved-response matching, no function-body fallback |
| Runner | Reviewed scenarios by default, repeats, concurrency, async factories, bounded timeouts, request/tool limits, partial traces |
| Reports | Terminal failure counts and expected/observed details, baseline differences, JSON traces, JUnit XML |
| CLI | `record`, `suggest`, `review`, `validate`, `run`, `demo`; installed entry point and `python -m` entry point |
| Pytest integration | Auto-discovered single-scenario and suite runner fixtures |
| Examples | Customer support, document search, CRM ticket creation; four reviewed JSON scenarios |
| Seeded mutations | Ineligible refund, wrong ID, missing lookup, duplicate action, parallel prerequisite/action |
| CI preparation | Pull-request workflow with Python 3.11/3.12/3.13 matrix, checks, package build, example runs, uploaded reports |
| Documentation | Quickstart, full project plan, scenario reference, architecture, troubleshooting, contributing |

## Validation evidence

Final source validation on macOS:

- **Python 3.13.2:** 186 tests passed.
- **Python 3.11:** 186 tests passed.
- **Coverage:** 95% combined statement/branch coverage, measured with `coverage run -m pytest` so pytest plugin startup is included.
- **Ruff:** lint and formatting checks pass.
- **Mypy:** strict checks pass for all 12 package source modules.
- **CLI smoke:** safe and broken demos return expected exit codes; support scenarios pass 20 runs with concurrency 3; search and CRM scenarios pass.
- **Report validation:** the broken demo reports three ineligible-order failures out of three, three eligible-order passes, six total runs; JSON and JUnit totals agree.
- **Packaging:** wheel and sdist build. Wheel includes type marker and runtime examples; sdist includes tests, docs, reviewed fixtures, and license. Virtual environments and generated reports are excluded.
- **Clean installation:** wheel installed into a separate environment and exercised from outside the checkout, including both CLI entry points and the complete record/review/replay workflow.

The GitHub workflow is prepared but has not run on GitHub. Python 3.12 and Linux remain CI verification targets rather than locally verified environments. No real model, production trace, or external developer has validated the product yet.

Generated local evidence is in ignored `reports/`: Python-version JUnit test results, `support.json`/`support.xml`, `broken-demo.json`/`broken-demo.xml`, smoke validation, and clean-wheel workflow outputs. Coverage XML is `coverage.xml`. Local package builds are in `dist/`.

## Meaningful safety cases verified

Tests prove original sync/async function bodies do not execute on successful replay, wrong arguments, missing fixtures, exhausted fixtures, retries, or malformed/unknown calls. They cover external FunctionToolset registrations and built-in name/prefix wrappers. Unsupported custom wrapper lifecycle is rejected before entering it. Concurrent shared-Agent runs isolate fixture queues and returned-response mutations.

The audit caught and fixed scenario mutation bypasses, nonfinite diagnostic values, blank tool names, ambiguous duplicate call IDs, loss of partial traces on suite timeout, and explicit model overrides losing to an ambient override. Unexpected extraction errors become failing results rather than a silent pass.

## Boundaries and next work

This is a usable local alpha, not a claim of production or market validation. Current support is Pydantic AI 2.55.x, one text prompt, string output, and in-process function toolsets. Model requests use the configured model unless explicitly overridden; imported factories, prompt callbacks, validators, dependencies, and ordinary Python application code remain outside the interception boundary. Recording executes configured tools. Replay is not a Python or network sandbox.

Prioritized next steps:

1. Add opt-in real-model experiments using mocked tools, repeat each policy scenario 10–20 times, and compare known prompt/model mutations. Keep these separate from deterministic package CI.
2. Exercise one existing application with dependencies and an actual historical failure; measure time to record, review, and diagnose the regression.
3. Validate Python 3.12/Linux through the prepared CI workflow when a remote repository is intentionally created.
4. Decide how to tolerate corrected schema-validation retries. Currently every attempt is visible and attempts that never reach fixture interception fail replay; valid business-tool retries can pass.
5. Add business-specific correlated preconditions if needed, such as requiring the lookup and action to refer to the same entity without hard-coded scenario IDs.
6. Extend the compatibility matrix and support structured/deferred outputs or dynamic toolsets only with dedicated interception tests.
7. Find three developers maintaining Pydantic AI agents, reproduce one actual failure each, and observe whether they keep rerunning the tests after changes. This is future user validation; none has happened yet.

See [the project plan](PROJECT_PLAN.md) for sequencing, acceptance criteria, the testing strategy, and the six-to-eight-week roadmap.
