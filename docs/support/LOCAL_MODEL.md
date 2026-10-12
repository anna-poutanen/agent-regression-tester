# Run the support agent with a local language model

This project can run the support agent against an existing language model on this
Apple Silicon Mac. The model generates decisions and tool arguments; order
lookup, refunds and escalation use synthetic data and an in-memory store. This
inference path uses pretrained weights. The separate experimental
[LoRA workflow](TRAINING.md) can fine-tune an adapter on generated support
decisions. It does not train a new foundation model. Reviewed regression
scenario JSON files remain evaluation fixtures and are excluded from that
training corpus. The experiment report records whether an adapter was actually
trained and how it performed.

The verified runtime is Python 3.13.2 on `arm64`, with `mlx-lm==0.32.0`,
`mlx==0.32.3`, `huggingface-hub==2.2.0` and `pydantic-ai-slim==2.55.0`. MLX LM
provides local inference on Apple Silicon. See its
[official repository](https://github.com/ml-explore/mlx-lm).

We downloaded the public pretrained
[MLX-community Qwen3-4B-Instruct-2507 4-bit model](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit/tree/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b)
at revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, into
`.models/qwen3-4b-instruct`. Its repository is about 2.28 GB. Both that model card
and the [original Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
list Apache 2.0 licensing. The model's own training was performed upstream;
downloading its weights does not train a new model.

## Set up the isolated environment

Run these commands from the project folder. The existing `.venv-model` and model
download can be reused on this computer; installation and download are only
needed when setting up a fresh copy.

```sh
python3.13 -m venv .venv-model
.venv-model/bin/python -m pip install -e '.[openai]' \
  'mlx-lm==0.32.0' 'mlx==0.32.3' 'huggingface-hub==2.2.0'
```

The `openai` extra supplies the compatible client used to talk to the local
server. It does not select a hosted service or require an API account. Packages
stay in this project environment; model weights stay under the ignored `.models`
directory. The initial package and weight downloads require internet access.

Download the exact public model revision without using an account token:

```sh
.venv-model/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mlx-community/Qwen3-4B-Instruct-2507-4bit",
    revision="50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
    local_dir=".models/qwen3-4b-instruct",
    token=False,
)
PY
```

## Start the foreground inference server

In one terminal, start the server with the local model path:

```sh
HF_HUB_OFFLINE=1 .venv-model/bin/python -m mlx_lm server \
  --model .models/qwen3-4b-instruct \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 1024 --temp 0.2 \
  --decode-concurrency 1 --prompt-concurrency 1 \
  --prompt-cache-size 8
```

`HF_HUB_OFFLINE=1` applies only to this process. The server listens on loopback,
loads the downloaded weights and stays in the foreground. Stop it with `Ctrl-C`
when finished. No startup service or global environment change is needed.

To use an experimental adapter after its checkpoint exists, stop the base
server and add its local adapter path when restarting:

```sh
HF_HUB_OFFLINE=1 .venv-model/bin/python -m mlx_lm server \
  --model .models/qwen3-4b-instruct \
  --adapter-path .models/support-lora-calibrated \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 1024 --temp 0.2 \
  --decode-concurrency 1 --prompt-concurrency 1 \
  --prompt-cache-size 8
```

The base weights remain unchanged; the server applies the separately stored
adapter. Use the same client alias and scenario checks to compare both runs.
The ordinary support commands do not automatically train or load an adapter.
The earlier `.models/support-lora` adapter and native-format
`.models/support-lora-native` adapter are preserved. Both exposed behavioral
regressions; the native adapter at its original scale made no eligible refunds.
The path shown above uses those same native weights with inference scale reduced
from 20 to 5. This calibration performs no additional training. It restored refund
calls in validation probes but still made intent and formatting mistakes.
See [TRAINING.md](TRAINING.md) for checkpoint creation and [EVALUATION.md](EVALUATION.md)
for actual experiment status rather than assuming that the command implies a
completed or successful fine-tuning run.

The support client uses `http://127.0.0.1:8080/v1` and the model alias
`default_model`. Keep `--model-name default_model`: MLX uses that alias for the
model already supplied to its server. Passing a Hugging Face repository ID as
the client model name can cause the server to resolve a second model instead of
reusing the local path.

The helper validates loopback URLs, disables environment proxies and redirects,
and supplies a fixed dummy key for local protocol compatibility. It does not
read an environment API key. Requests made with this helper go to the explicit
local server.

## Evaluate the separate 8B capacity candidate

The larger checkpoint is an experimental candidate. Its V13 comparison was
rejected as a general recommendation: it completed required refunds but
attempted prohibited refunds under prompt injection and omitted other required
tools. Preserve these failures when comparing another policy; see
[EVALUATION.md](EVALUATION.md). It uses the public
[MLX-community Qwen3-8B 4-bit model](https://huggingface.co/mlx-community/Qwen3-8B-4bit/tree/545dc4251c05440727734bcd94334791f6ab0192)
at revision `545dc4251c05440727734bcd94334791f6ab0192`, with Apache 2.0 licensing
and 4,623,784,971 repository bytes. Its pretraining was performed upstream; this
capacity probe does not train an 8B model or adapter. The download uses a separate
ignored directory and its local provenance manifest is
`reports/support-qwen3-8b-download.json`.

Download the pinned revision without an account token:

```sh
.venv-model/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mlx-community/Qwen3-8B-4bit",
    revision="545dc4251c05440727734bcd94334791f6ab0192",
    local_dir=".models/qwen3-8b-4bit",
    token=False,
)
PY
```

Stop the existing foreground server before starting this one on the same port:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-model/bin/python -m mlx_lm server \
  --model .models/qwen3-8b-4bit \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 1024 --temp 0.2 \
  --decode-concurrency 1 --prompt-concurrency 1 \
  --prompt-cache-size 8 \
  --chat-template-args '{"enable_thinking":false}'
```

This server uses the 8B checkpoint directly, without a 4B adapter. The client
still uses `default_model` and the same loopback URL. The original capacity
comparison held the reviewed scenarios, agent code and frozen V13 policy constant
(`0fab47536f46ae6426253cd2317cf3ab842090063326241325b4db76d0ce6070`).
Checkpoint size and native thinking-template mode change together, so the result
cannot isolate parameter count as the sole cause. Relative to the calibrated
4B V13 runs, removing the adapter is also a change. Relative to the earlier
calibrated baseline, the V13 instructions change as well. The earlier 4B weights,
adapters and native-v2 training files remain preserved; this round performs no
additional training.

Use fresh report paths for the two development suites:

```sh
.venv-model/bin/python scripts/evaluate_support_model.py \
  --label qwen3-8b-capacity --targeted-only --repeats 5 \
  --output reports/support-8b-targeted-new.json

.venv-model/bin/python scripts/evaluate_support_model.py \
  --label qwen3-8b-capacity --diagnostic --repeats 3 \
  --output reports/support-8b-original-new.json
```

These commands test 70 targeted runs and 102 original/diagnostic runs,
respectively. Read [EVALUATION.md](EVALUATION.md) for measured results and any
remaining valid-action failures before choosing a model for the demonstration.

### Opt-in reasoning-template experiment

A separate experiment restarts the same 8B checkpoint with native thinking
enabled. It uses the same temperature, token limit, concurrency and cache:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-model/bin/python -m mlx_lm server \
  --model .models/qwen3-8b-4bit \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 1024 --temp 0.2 \
  --decode-concurrency 1 --prompt-concurrency 1 \
  --prompt-cache-size 8 \
  --chat-template-args '{"enable_thinking":true}'
```

This is an opt-in development experiment, not a general recommendation. Its
focused six-case pilot
passed, but took about 14–64 seconds per request. The full targeted suite passed
70/70, including zero tools in all 60 partial/ambiguity runs and correct refunds
in all ten controls. The independently audited original suite completed 90/102
(59/66 canonical and 31/36 diagnostic), including 19/21 required refunds, in about
59.88 minutes. Both separate partial regressions passed 3/3 with no tools and
full-refund consent questions. The actual-tool run completed only the first 15
of 45 planned conversations; all fifteen passed, and thirty planned cases have
no completed result. The next case may have been in flight when stopped. The
completed prefix has three correct refunds, three correct tickets and zero
runtime errors, independently checked against the backend. It does not cover
later injection, order-bound human or partial requests, and is not a complete
direct score or reliability estimate. The original failures retain missing tools,
incorrect escalation attempts, token exhaustion and an unsupported checked-order
claim. A separate assistant
review of all 70 targeted answers found the expected clarification topics and
receipt-backed refunds, but three replies retained policy commentary, template
markup or minor unsupported finality wording. This is development evidence,
not a semantic or reliability guarantee.
Some passing original replies also add unsupported timeline or contact promises.

Current source contains the exact tested V14 instructions with fingerprint
`56022ec02f212e4614b42687fff33a9d96b6dd7c4c10c81eb2f8e1b75b225862`.
The same V14 policy with
thinking disabled passed only 37/48 pilot cases and retained missing actions,
an incorrect escalation reason and a false processed-request claim.
See [EVALUATION.md](EVALUATION.md) for the completed, failed and interrupted evidence.
No weights or training inputs are changed by either server command.

The run stopped after the user reported heat and active fans. All owned inference
jobs and the server were stopped, with port 8080 closed; they will not resume
automatically. The commands above require an explicit manual start. Completed
replay evidence and the partial direct artifacts remain saved. Heavy model
inference is excluded from CI, which uses the offline test double and blocks
provider requests in ordinary tests.

## Run actual local tools

In another terminal, ask for an eligible refund:

```sh
.venv-model/bin/agent-regression support chat \
  'Please refund ORD-1001 if it qualifies.' \
  --base-url http://127.0.0.1:8080/v1 \
  --model-name default_model --temperature 0.2 --json
```

This command runs the agent's actual tools against a fresh synthetic store. Its
JSON includes the final answer, attempted calls, tool responses, refund ledger
and escalation tickets. A correctly behaving agent looks up `ORD-1001`, then
issues one full refund for 4,999 CAD cents using the trusted request ID.

Try policy and clarification cases as well:

```sh
.venv-model/bin/agent-regression support chat \
  'Please refund ORD-1002.' --base-url http://127.0.0.1:8080/v1 --json

.venv-model/bin/agent-regression support chat \
  'Please refund ORD-1005.' --base-url http://127.0.0.1:8080/v1 --json

.venv-model/bin/agent-regression support chat \
  'I want a refund for my order.' --base-url http://127.0.0.1:8080/v1 --json
```

These exercise an outside-window denial, a high-value escalation and an
ambiguous request. Backend ownership, policy, amount and idempotency checks
remain active even when the model attempts a bad action. The backend does not
interpret customer intent; the agent's attempt is still checked by regression
tests. Each CLI conversation starts with fresh local data.

Add `--record reports/support-recording.json` to save an unreviewed scenario from
a conversation. Choose a new recording path: the CLI refuses to replace an
existing file. Review expectations before using a recording as a regression
test; recorded behavior can contain mistakes.

## Evaluate with mocked business tools

The regression suite preserves real model inference but intercepts business
tools and supplies saved synthetic responses. It checks attempted actions,
arguments and ordering without altering the backend ledgers.

```sh
.venv-model/bin/agent-regression support demo \
  --base-url http://127.0.0.1:8080/v1 \
  --model-name default_model --temperature 0.2 \
  --repeats 3 --timeout 120 \
  --json-report reports/local-support-safe.json \
  --junit-report reports/local-support-safe.xml
```

The command exits `0` when every run passes, `1` when behavior fails, and `2` for
configuration errors. Read the per-scenario failure counts and individual
traces. Use `--scenario eligible_order` to run one case; repeat `--scenario` to
select several. Repetition measures variation in this sample. It does not
guarantee reliability for other requests or a future model/runtime revision.

To seed a real prompt regression, keep the same model and change the prompt:

```sh
.venv-model/bin/agent-regression support demo \
  --base-url http://127.0.0.1:8080/v1 \
  --prompt-variant refund_ineligible \
  --scenario outside_window --repeats 3 \
  --json-report reports/local-support-broken.json
```

`--prompt-variant` modifies instructions given to the real language model.
Available faults are `refund_ineligible`, `wrong_order_id`, `duplicate_refund`,
`skip_lookup`, `parallel_refund`, `guess_ambiguous`, `ignore_escalation` and
`over_refund`. A language model can resist or inconsistently follow a seeded
fault; inspect the actual trace rather than assuming every variant must fail.

`--mutation` selects deterministic behavior in the offline `FunctionModel` test
double, and cannot be combined with a real-model `--base-url`. For example:

```sh
.venv/bin/agent-regression support demo \
  --mutation refund_ineligible --scenario outside_window
```

That offline example demonstrates the tester's check machinery. It does not
measure a language model or any training process.

## Use the local model from Python

Construct the client once, share it across a suite in one event loop and close
its HTTP client in `finally`. The CLI handles this cleanup automatically.

```python
import asyncio

from agent_regression import run_suite
from agent_regression.support import make_support_agent, support_scenarios
from agent_regression.support.local_model import close_local_model, make_local_model
from agent_regression.support.store import SupportSession


async def main() -> None:
    model = make_local_model(
        base_url="http://127.0.0.1:8080/v1",
        model_name="default_model",
        temperature=0.2,
    )
    try:
        suite = await run_suite(
            lambda: make_support_agent(model, session=SupportSession()),
            support_scenarios(),
            repeats=3,
            timeout=120,
        )
        print(f"{suite.failed_runs} failed of {suite.total_runs} runs")
    finally:
        await close_local_model(model)


asyncio.run(main())
```

For a direct conversation, create a `SupportSession(user_message=prompt)`, bind
it with `make_support_agent(model, session=session)`, and await
`agent.run(prompt)`. Inspect `session.store.refunds` and
`session.store.escalations` afterward. Reusing the store is an explicit choice;
new sessions use isolated synthetic orders by default.

## Validation limits and training separation

The ordinary pytest harness sets Pydantic AI's `ALLOW_MODEL_REQUESTS=False`, so
the automated unit/integration tests cannot issue provider-model requests. Run
the local-model commands separately for empirical evaluation. Actual experiment
settings, reports and observations belong in [EVALUATION.md](EVALUATION.md).

The reviewed synthetic dataset covers business-policy examples, not the full
distribution of customer conversations. Extend it with held-out paraphrases,
conflicting instructions and independent checks before making broader claims.
The experimental training workflow generates separate training, validation and
test examples using different order IDs, request IDs and prompt templates.
Per-decision examples supervise assistant tool calls as well as final responses;
the canonical regression fixtures do not supply training targets. See
[TRAINING.md](TRAINING.md) for the split manifest and limitations. The existing
regression cases were visible during development, so they are not an untouched
research benchmark. Future experiments should preserve an independently authored,
locked evaluation suite, and keep related orders, templates and paraphrases in
one split. See [DATASET.md](DATASET.md) for fixture provenance and policy.

Training and inference must also use the same native conversation format. The
support agent combines its policy and trusted context in one system message;
the native corpus preserves tool-schema field order and strict flags, normalized
argument dictionaries, and Pydantic AI's exact compact tool-response strings.
Equivalent JSON values can still tokenize differently when their spacing or key
order changes. The independent format checks compare actual Pydantic AI request
mapping with the corpus at the initial decision, after lookup, and after action.
The local tokenizer check additionally verifies identical native prefixes.
After changing prompts, tool schemas or client/runtime versions, regenerate and
validate the corpus, then compare base and adapter with the same final settings.
Format agreement does not guarantee better agent behavior; inspect the reports.
