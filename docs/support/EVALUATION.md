# Support-agent evaluation evidence

This is local development evidence from synthetic data. It validates a regression
workflow and a support example; it does not establish production safety or demand
from external users.

## What the checks measure

The 22 canonical scenarios grade required and forbidden attempts, order IDs,
integer amounts, request keys, tool ordering, call counts, escalation categories
and clarification. Twelve additional diagnostic requests keep the same reviewed
policy checks while changing wording. Both groups remain development cases, not
representative customer traffic or an untouched research benchmark. The canonical
cases were visible during prompt development.

The offline agent is a deterministic FunctionModel test double. Passing repeated
offline runs verifies the harness and business flow but cannot measure language
model reliability. Provider requests are blocked by default in pytest. Real-model
runs execute separately through a loopback server.

Replay mocks all store methods. Direct conversations execute the actual local
store. Tests verify that replay leaves lookup audits, refunds and tickets empty,
while direct eligible requests create one refund. Backend validation refuses
unauthorized, premature, ineligible and incorrectly sized refunds. Idempotency
prevents duplicate ledger entries even when the tester reports a second attempt.

Final-answer semantics are only partially checked. Ambiguous cases require a
question. A required action cannot be replaced by a claim that it happened, but
tool-only checks do not comprehensively grade every explanation or factual claim.
For example, an ineligible-order run can pass after the correct lookup while
falsely claiming that a refund succeeded, because the current checks forbid
refund attempts rather than verifying the truth of that final explanation.

## Real failures found

The first direct Qwen run looked up an eligible order and then claimed that a
refund had succeeded without calling `issue_refund`. Its actual local ledger was
empty. The reviewed eligible scenario catches this as `missing_call`.

| Historical run | Configuration | Passed | Failed |
| --- | --- | ---: | ---: |
| Initial real model | Qwen3 4B, initial safe prompt, 22 scenarios × 1 | 14 | 8 |
| Revised prompt | Same base model, clearer action procedure, 22 scenarios × 3 | 45 | 21 |

The revised run missed refunds and escalations, sometimes guessed an ambiguous
identifier, and supplied the string `"null"` before correcting it to JSON `null`.
All attempts are graded, so a correction does not erase an initial wrong call.
The prompt revision improved some cases and worsened others. These numbers do
not justify a claim that prompt wording alone fixed model reliability.

Raw reports are kept locally in `reports/support-local-initial.json` and
`reports/support-local-refined.json`, with terminal and JUnit counterparts. The
first direct failure was recorded in `reports/support-local-first.json` and stays
unreviewed rather than turning its incorrect behavior into an expectation.

Before training, we removed a concrete evaluation identifier from the public
lookup-tool description. Historical runs therefore differ in prompt/schema from
later experiments. A causal base-versus-adapter comparison requires the same
final prompt, schemas, scenarios, sampling and repeat count for both.

## Training experiment and reproducibility

[Training](TRAINING.md) uses a small local LoRA adapter with synthetic policy-expert
decision prefixes. Orders, wording and request keys are separated between its
train/validation/test partitions and excluded from canonical and diagnostic
evaluation cases. Shared public business policy and tool schemas are intentional.
The adapter learns intermediate tool calls as supervised targets, not just final
replies. Corpus provenance is described in [training data](TRAINING_DATA.md).

Training loss is not an agent result. The training manifest records native token
mask checks, no-truncation preflight, data/model hashes, runtime, settings, losses
and saved adapter hashes. Evaluation uses the unchanged reviewed checks and saves
full call traces even when failures remain.

### First paired experiment

The first matched comparison used the same two-system-message live agent for
both models, temperature 0.2, 34 scenarios and three repeats:

| Model | Canonical passed / 66 | Diagnostic passed / 36 | Total passed / 102 |
| --- | ---: | ---: | ---: |
| Base Qwen3 4B | 40 | 20 | 60 |
| Initial 150-update LoRA | 39 | 15 | 54 |

The adapter improved ordinary eligible and repeated-wording refunds but regressed
delivery escalation, order-bound human help, ambiguity and the 30-day boundary.
It also retained false success claims and invalid string `"null"` calls. This is
a training regression that the tester exposed. The original reports remain in
`reports/support-base-final.*` and `reports/support-adapter.*`; the corresponding
adapter and corpus remain unchanged locally.

### Native-format alignment

Review found that the first training export differed from live inference in
system-message layout, prepared schema ordering/metadata and tool-response JSON
serialization. Combining the system instructions alone passed only two of six
diagnostic probe cases; it did not resolve the workflow failures.

The native-v2 generator preserves model-facing key order, prepares schemas with
the public OpenAI profile transformer, uses the public tool-return serializer and
normalizes prior arguments as MLX does. Independent checks matched all 480 saved
prefix strings and token arrays against actual Pydantic AI mapping and the local
Qwen tokenizer. These checks use scripted expert decisions and synthetic tools,
not inference. Policy, labels, templates, identifiers and split membership stay
unchanged. Exact format parity is documented in
`reports/support-native-prefix-parity.json`.

The next experiment uses a fresh base evaluation and a predetermined 384-update
training run through the entire training partition. Format and training duration
both change relative to the first experiment, so their individual contribution
cannot be isolated from the overall result. Evaluation expectations remain fixed.

With the corresponding base or adapter server running:

```sh
.venv-model/bin/python scripts/evaluate_support_model.py \
  --label qwen3-4b-base-native-aligned --diagnostic --repeats 3 \
  --output reports/support-base-native.json

# Restart the server with --adapter-path .models/support-lora-native, then:
.venv-model/bin/python scripts/evaluate_support_model.py \
  --label qwen3-4b-support-lora-native --diagnostic --repeats 3 \
  --output reports/support-adapter-native.json
```

Each run has a JSON trace report, JUnit XML and a manifest with prompt/scenario
hashes, schema order, agent-source hash, instruction layout, model label, sampling,
timing and outcomes. The script preserves existing
report paths. An exit code of 1 is a recorded regression result, not a reason to
discard the experiment. No model weights, generated corpora or raw reports are
committed to Git.
