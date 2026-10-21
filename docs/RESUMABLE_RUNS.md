# Short, resumable regression runs

Checkpointed runs save every completed iteration, including failures, before
starting the next one. An interrupted or deliberately shortened run is
**incomplete**, even when every completed result passed. Resume keeps those
results and starts at the first iteration without a completed result.

The examples below use the offline demo. They do not load a language model,
contact a provider or execute business-tool bodies during replay.

## Try a short batch

```sh
agent-regression run examples/scenarios/support \
  --agent agent_regression.demo:make_support_agent \
  --repeats 3 \
  --checkpoint reports/demo.checkpoint.json \
  --run-id support-demo-v1 \
  --max-runs 2 \
  --json-report reports/demo.json \
  --junit-report reports/demo.xml
```

This completes two of six planned iterations and exits **3**, indicating an
incomplete run. The checkpoint contains both completed results. JSON records
the planned and remaining counts; JUnit contains an `IncompleteSuite` error
marker so an interrupted suite cannot look like a passing CI job.

Resume with the same scenario selection, repeats, timeout, review mode, agent
target and configuration ID:

```sh
agent-regression run examples/scenarios/support \
  --agent agent_regression.demo:make_support_agent \
  --repeats 3 \
  --checkpoint reports/demo.checkpoint.json \
  --run-id support-demo-v1 \
  --resume \
  --json-report reports/demo.json \
  --junit-report reports/demo.xml
```

Only the four unfinished iterations run. The final reports contain all six
results. A completed failed result is never retried or discarded by resume.
To evaluate a changed agent, start a new checkpoint with a new configuration ID.

## Limits and breaks

| Option | Behavior |
| --- | --- |
| `--max-runs N` | Complete at most N new iterations in this invocation. Previously saved results do not consume this allowance. |
| `--max-seconds S` | Apply an invocation deadline to asynchronous factory/model execution and pauses. An unfinished active iteration has no completed result and runs fresh on resume. A result that already completed is retained even if the deadline is crossed. |
| `--batch-size N` | Group newly executed iterations into batches. |
| `--pause-seconds S` | Pause between batches when more work remains; requires `--batch-size`. There is no pause after the final iteration or after reaching the run allowance. |
| `--timeout S` | Keep the existing per-iteration timeout. An iteration that reaches this limit is a completed failed result, rather than an unfinished budget-limited iteration. |

For example, add `--max-runs 4 --batch-size 2 --pause-seconds 10` to run at most
four new iterations with a break between the two batches. No break or overall
budget is imposed unless requested. Limits and breaks can change on resume.
Cancellation cleanup and checkpoint writes may extend elapsed time. Blocking
synchronous application code and filesystem operations cannot be forcibly
interrupted by the event loop; use nonblocking factories and model callbacks.

These controls run sequentially: use `--concurrency 1`, which is already the
default. Ordinary runs without these options retain their existing concurrency
support. There is no automatic restart or background scheduler.

Ctrl+C cancels the active run and preserves completed checkpoint results. The
CLI exits **130**. Resume retries the unfinished iteration with fresh fixtures
and a fresh factory result. Checkpointing does not promise exactly-once execution
of arbitrary application factories, callbacks or model requests.

The runner cancels its client operation; it does not manage a separately started
model server or guarantee that the server immediately stops computing an
in-flight request. Stop an independently running server separately when needed.
Breaks reduce continuous work but are not a hardware-temperature controller.

## Configuration identity and files

`--run-id` is your declaration of the agent's configuration. Include the prompt,
model/checkpoint and application-code revision in the ID or use a hash of that
configuration. The CLI also binds it to the exact `--agent` import target.
The package cannot automatically prove that an arbitrary imported factory,
dynamic prompt or local model server still has the same behavior. Change the
ID and checkpoint path whenever those inputs change.

Resume rejects a changed scenario snapshot or order, repeat count, timeout,
review mode, declared identity or recorded runtime versions. Scenario names
must be unique. A normal new run refuses to overwrite an existing checkpoint;
use `--resume` explicitly or choose a new path.

Writes use an atomic replacement and filesystem synchronization. A nonblocking
OS lock prevents two invocations from modifying the same checkpoint at once.
The small `.lock` sidecar can remain after completion; its existence does not
mean a process is running. Do not remove a lock file while a run is active.

Keep checkpoint and report files outside directories used for scenario
discovery. Do not use a scenario file as an output destination. Checkpoints
contain the same trace data as reports; review them before sharing.

## Python API

The same controls are available on `run_suite` and `run_suite_sync`:

```python
from agent_regression import load_checkpoint, run_suite_sync
from agent_regression.demo import demo_scenarios, make_support_agent

suite = run_suite_sync(
    make_support_agent,
    demo_scenarios(),
    repeats=3,
    checkpoint="reports/python-demo.checkpoint.json",
    run_id="support-demo-v1",
    max_runs=2,
)
print(suite.status, suite.total_runs, suite.planned_runs, suite.remaining_runs)

saved = load_checkpoint("reports/python-demo.checkpoint.json")
print(saved.to_suite().status)

suite = run_suite_sync(
    make_support_agent,
    demo_scenarios(),
    repeats=3,
    checkpoint="reports/python-demo.checkpoint.json",
    run_id="support-demo-v1",
    resume=True,
)
assert suite.status == "completed"
assert suite.total_runs == 6
```

The async API propagates cancellation after saving the completed prefix. Python
callers should check `suite.status` as well as `suite.failed_runs`; zero observed
failures does not establish that every planned iteration ran.

## Exit codes and reports

- **0:** all planned iterations completed and passed.
- **1:** a complete suite contains regressions or runtime failures.
- **2:** invalid configuration, incompatible checkpoint or storage error.
- **3:** a run allowance or invocation-time budget left the suite incomplete.
- **130:** interrupted with Ctrl+C.

JSON keeps `total_runs`, `passed_runs` and `failed_runs` as completed-result
counts, adding `planned_runs`, `remaining_runs`, `status` and `stop_reason`.
Unfinished iterations are not fabricated as failed or passing results. JUnit
adds one suite-level error test for an incomplete suite; its test count includes
that marker in addition to completed iterations. Complete-run JUnit counts are
unchanged.
