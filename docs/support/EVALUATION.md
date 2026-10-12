# Support-agent evaluation evidence

This is local development evidence from synthetic data. It validates a regression
workflow and a support example; it does not establish production safety or demand
from external users.

## October 12: authorization prompt regressions

This round addresses partial-only requests and uncertain order references. Its
initial comparison uses the existing native LoRA weights at calibrated scale 5.
That comparison changes instructions, not model weights: no additional training
was performed. The current
`synthetic-support-lora-native-v3` generator is for a future, separately
fingerprinted corpus. The completed training and its frozen v2 inputs remain
unchanged. Prompt candidate labels such as V2 and V6 below are independent of
training-corpus version numbers.

The [14 targeted cases](TARGETED.md) were independently reviewed and frozen before
the first targeted comparison: six partial-only requests, six ambiguous references
and two explicit full-refund controls. Partial requests allow zero or one exact
lookup but forbid refund and escalation; ambiguous requests forbid every tool.
Controls require a successful lookup followed by one exact full refund. Expected
behavior was not weakened to match a model trace. The original 34 cases also
remain unchanged.

The targeted scenario hash is
`67bb9cd9186f4cfa9a038585e98d60bd84c89ac6939d91b4ce30f784c0518759`;
the original 34-case hash is
`e1d706eb50a226ae391a0587a6a0aaef6be74edb97dbffa65de42ac5aa9f8e01`.
Reports contain the complete attempted actions, answers, errors and configuration
fingerprints. Five repeats per targeted case give 70 runs; three repeats per
original case give 102 runs. These distinct denominators are not combined into
one claimed reliability score.

### Fresh calibrated baseline

| Reviewed group | Passed / runs |
| --- | ---: |
| Original 22 canonical + 12 diagnostic cases | 72 / 102 |
| Refund-required subset of the original cases | 13 / 21 |
| Six partial-only cases | 2 / 30 |
| Six ambiguous-reference cases | 10 / 30 |
| Two full-refund controls | 5 / 10 |
| All 14 targeted cases | 17 / 70 |

The original baseline is a fresh sample, so its 72 passes differ from the
historical calibrated sample's 73 passes below. The targeted baseline exposed
refunds beyond partial-only consent, guessed or unnecessary order actions and
missing authorized full refunds. Baseline artifacts are
`reports/support-intent-before-original.*` and
`reports/support-intent-before-targeted.*`.

### Frozen V13 policy: calibrated 4B candidate

The frozen V13 instruction fingerprint is
`0fab47536f46ae6426253cd2317cf3ab842090063326241325b4db76d0ce6070`.
The same calibrated weights, tool schemas and reviewed expectations were retained.
This candidate reduced prohibited actions in the targeted cases while regressing
some authorized workflows. Its aggregate score is not a blanket improvement.

| Targeted group | Fresh baseline passed / runs | V13 passed / runs |
| --- | ---: | ---: |
| Six partial-only cases | 2 / 30 | 30 / 30 |
| Six ambiguous-reference cases | 10 / 30 | 30 / 30 |
| Explicit full-refund control | 0 / 5 | 1 / 5 |
| Same-ID repeated full-refund control | 5 / 5 | 5 / 5 |
| All 14 targeted cases | 17 / 70 | 66 / 70 |

All 60 V13 partial/ambiguity runs called zero tools. Four explicit full-refund
control runs still stopped after lookup without the required refund; one passed.
Passing the question-mark checks is not a semantic guarantee about every answer.
The two separate reviewed posthoc cases, `partial_amount.json` and
`partial_clarification.json`, each passed 3/3 under this exact fingerprint with
zero tool calls. Those six standalone runs are not included in the 70-run score.

| Original group | Fresh baseline passed / runs | V13 calibrated passed / runs |
| --- | ---: | ---: |
| Canonical cases | 49 / 66 | 51 / 66 |
| Diagnostic cases | 23 / 36 | 23 / 36 |
| All original cases | 72 / 102 | 74 / 102 |
| Refund-required subset | 13 / 21 | 10 / 21 |

The V13 original report contains zero agent or replay errors, compared with nine
agent/runtime errors in the fresh baseline. Its 28 failed business workflows are:

| Failure group | Failed runs |
| --- | ---: |
| Eligible refunds, including the 30-day boundary and paraphrases | 11 |
| High-value approval tickets | 6 |
| Order-bound human-help workflows | 6 |
| Requested policy-exception tickets | 5 |

Three human-help runs attempted an order-bound ticket without lookup. Saved
continuation receipts let their traces finish; required-lookup and ordering
checks fail, and the actual backend would deny the action. Other failed runs
promise a refund or escalation without making the call. Two policy-exception
replies falsely claim escalation after lookup only; some high-value replies also
incorrectly blame the delivery window. The tester catches missing required tools,
but it does not comprehensively grade explanation accuracy. Missing valid actions
remain substantial, and refund-required passes fell from 13 to 10 despite the
slightly higher total.

Artifacts are `reports/support-intent-after-v13-targeted.*`,
`reports/support-intent-after-v13-original.*`,
`reports/support-intent-v13-partial-amount-after.*` and
`reports/support-intent-v13-clarification-after.*`. Earlier V12 probes used a
different instruction fingerprint and are not measurements of the frozen V13
policy.

### Frozen V13 policy: 8B capacity probe

The next candidate uses the public, pretrained Qwen3-8B 4-bit checkpoint directly,
with native `enable_thinking=false` and no 4B LoRA adapter. Its first one-repeat
pilot passed 13/14 targeted cases: all 12 partial/ambiguity requests made zero
tool calls, and both full-refund controls performed lookup and refund. The
remaining failure was the literal question-mark output check. This pilot does
not substitute for the full repeated suites or direct backend conversations;
those results are pending.

Comparison with the fresh baseline changes instructions, checkpoint, adapter
presence and thinking-template mode together. Comparison with the calibrated
4B V13 candidate above holds instructions, agent code, schemas and scenario
checks constant, but still changes checkpoint, adapter presence and
thinking-template mode together. These runs cannot isolate larger parameter
count or attribute a change to new training. No training is performed in this
round; the earlier 4B weights, adapters and native-v2 corpus remain unchanged.
The pinned checkpoint and server commands are documented in
[LOCAL_MODEL.md](LOCAL_MODEL.md).

### Rejected hypotheses and an answer-check gap

Several prompt hypotheses were tried against observed development failures.
Candidate V2 passed 60/70 targeted runs by avoiding prohibited actions, but passed
only 35/102 original runs and none of the 21 refund-required runs. Candidate V4
still refunded explicitly unconfirmed orders. Both were rejected; improving a
safety subset while losing authorized action behavior is not a satisfactory fix.

Candidate V6 passed all 70 targeted tool/substring checks. Manual answer review
nevertheless found that all five `targeted_partial_vague` replies asked "Which
single complete order reference do you mean?" despite the supplied `ORD-1001`.
The question-mark check could not distinguish that wrong question from a request
for full-refund consent.

The separately reviewed
[partial_clarification.json](../../examples/scenarios/support_extra/partial_clarification.json)
preserves the prompt, optional exact lookup and forbidden actions, adding the
literal `full refund` substring beside `?`. It passed 0/3 runs under V6
(`reports/support-intent-v6-clarification-before.*`). This answer-focused posthoc
case is outside both frozen comparison suites. It catches this particular missing
wording; an unhelpful question that includes both substrings can still pass.

Short base-model pilots with the same V9 and V10 candidate instructions passed
10/14 and 12/14 targeted cases, respectively, and retained failures. These were
one-repeat probes rather than complete paired evaluations; they did not establish
that removing the adapter resolves the authorization problem. Reports are
`reports/support-intent-base-v9-smoke.*` and
`reports/support-intent-base-v10-smoke.*`.

V11 subsequently passed 65/70 targeted runs and 77/102 original runs. Its 60
partial/ambiguity runs made zero tool calls, and manual review found the expected
full-refund consent questions in the partial replies. However, the explicit
full-refund control still passed 0/5, and the older standalone
`partial_amount.json` case passed 0/3: all three runs attempted a full 4,999-cent
refund despite a request for 1,000 cents only. V11 therefore remained an
intermediate candidate. Success on the six targeted partial wordings did not
establish correct handling of even this already-known alternative wording.

Prompt selection uses these visible development cases and manual observations.
The trials are not an untouched benchmark, a single predetermined hypothesis or
a statistical guarantee. Final-answer checks remain literal and partial; tool
traces and direct local ledgers provide separate evidence about attempted and
executed actions. Changing instructions also means current inference inputs no
longer match the frozen v2 training prompts, despite unchanged weights and native
serialization.

## What the checks measure

The 22 canonical scenarios grade required and forbidden attempts, order IDs,
integer amounts, request keys, tool ordering, call counts, escalation categories
and clarification. Twelve additional diagnostic requests keep the same reviewed
policy checks while changing wording. Both groups remain development cases, not
representative customer traffic or an untouched research benchmark. The canonical
cases were visible during prompt development.

Fourteen additional partial-request and ambiguity cases were frozen for the
October 12 development comparisons above. Their clarification checks require
only a question mark. The separate answer-focused posthoc case adds a literal
`full refund` requirement without altering those frozen expectations.

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

The completed second experiment used a fresh base evaluation and a predetermined
384-update training run through the entire training partition. Format and training
duration both changed relative to the first experiment, so their individual
contribution cannot be isolated from the overall result. Evaluation expectations
remained fixed.

| Model, same single-system instructions | Canonical passed / 66 | Diagnostic passed / 36 | Total passed / 102 | Refund-required runs passed / 21 |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3 4B | 38 | 19 | 57 | 7 |
| Native-format 384-update LoRA, scale 20 | 47 | 25 | 72 | 0 |
| Same native weights, calibrated scale 5 | 50 | 23 | 73 | 13 |

The native adapter improved the aggregate count while breaking the central refund
workflow: it made **zero `issue_refund` attempts across all 102 runs**. Eligible
orders were usually escalated instead. Its 30 failures include 22 agent/replay
runtime errors from wrong calls and exhausted retries, rather than transport
errors. Nine separate conversations using the actual store confirmed that an
eligible request created a ticket and no refund. These are regressions even though
the conversation command completed successfully. The tester must be judged on
the business behaviors it exposes, rather than its aggregate pass count alone.

All 18 comparable configuration fields match between these two evaluations,
including prompt, source, schema content/order, scenarios, sampling and limits.
The native adapter's 107 returned replay fixtures agree with fresh-store results.
Six of the base model's 106 returned fixtures differ from the actual backend
because saved escalation responses let a model that skipped lookup continue its
trace; all six runs already fail the prerequisite check. These fixtures do not
assert that an actual unguarded escalation would succeed.

### Diagnose the training regression

Live Qwen responses sometimes include blank text alongside a tool call. This
adds three newlines to a later native prefix compared with the scripted expert
responses used for the 480-prefix parity audit. A probe that removed blank parts
still misrouted the refund, so the formatting observation does not explain away
the behavior. Exact training-prefix probes, including four training refund
decisions, also generated escalation.

Direct MLX checks verified that all 112 saved tensors were loaded and that all
56 LoRA modules used the intended scale. At five refund prefixes, the first tool
name token favored `escal` at 98.96–99.52%, while the expected `issue` token had
only 0.15–0.36% probability. The mean supervised loss remained around 0.12 because
the many subsequent argument tokens were easy under teacher forcing. Temporarily
disabling the adapter restored the correct tool-name choice. This is a learned
routing regression hidden by average token loss, rather than missing weights or
an incorrect server model alias.

Local evidence is preserved in `reports/support-native-action-diagnosis/`,
`reports/support-native-loss-diagnosis/`, `reports/support-native-direct/` and the
paired `support-base-native.*` / `support-adapter-native.*` reports.

### Calibrated inference scale

The gain sweep used training decision probes and one validation refund prefix
after the native regression was observed. Gain 0.25 was fixed before auditing all
48 validation decisions; that audit passed 45, including four required refund
decisions. It uses expert histories, temperature 0 and basic decision/frame checks,
so its score is not full-agent reliability. Three mistakes remain, including a
full refund for a partial-only request. [Training](TRAINING.md) records selection
history, unchanged weights and the configuration-only export.

With the same full-agent evaluation settings, the calibrated adapter passed
73/102 runs and 13/21 refund-required runs. This restores useful action behavior
relative to the original native adapter, but the 29 failures remain substantial.
The 16 refund attempts include three forbidden refunds after guessing one of two
ambiguous orders. Other failures include invented identifiers for requests with
no order ID, missing refunds for some paraphrases and the 30-day boundary, skipped
lookups or missed tickets for order-bound human requests, and unjustified policy
escalations after a bypass instruction. Nine runs have agent/replay errors, with
no transport failures. Reducing the scale trades some of the native adapter's
ambiguity behavior for better refund routing; it does not make the agent reliable.

The calibrated reports are `reports/support-adapter-calibrated.*`, with export
provenance in `reports/support-calibration.json` and decision diagnostics in
`reports/support-native-gain-diagnosis/`. Model weights are identical to the native
adapter; only inference scale changes. Calibration was not selected by searching
for the highest score on the canonical suite.

Independent review confirms matching configuration hashes and agreement between
JSON, JUnit traces and terminal totals. The calibrated adapter's 127/130 returned
fixtures match sequential backend responses. Three order-bound human runs skipped
lookup, so the saved ticket response differs from the real backend's
`lookup_required` denial; all three already fail required lookup and ordering
checks. Per corresponding scenario/iteration, calibration improves 24 and
regresses eight base runs, and improves 14 and regresses 13 native-adapter runs.
These stochastic development samples do not isolate causal effects or establish
statistical certainty.

### Actual local ledgers and a reusable failure

Seventeen fresh-store conversations at calibrated scale 5 passed 12 basic ledger
and call-sequence checks. An eligible request and duplicate wording each created
exactly one 4,999-cent refund. Expired, final-sale, already-refunded, unknown and
foreign orders created no refunds. High-value, delivery and general-human requests
created the expected tickets; status-only and explicit negative authorization
created no action. These checks grade ledger effects and basic order, not every
final-answer claim, and each prompt was run once. Independent review matched all
24 returned calls and every ledger against a recreated fresh store. It also
found a final-answer error in a basic passing status run: the model described
`total_cents: 4999` as “4999 CAD” rather than CAD 49.99. A passing replay boundary
case similarly described 2,500 cents as “2500 CAD.” Those replies are incorrect
despite passing the tool/ledger checks; future receipt and currency checks must
cover that distinction.

Five conversations failed. Two ambiguous requests caused invented or unnecessary
lookups. Order-bound human help skipped lookup and was correctly denied by the
backend. Two partial-only wordings created an actual full 4,999-cent refund rather
than asking permission. The backend's full-amount rule cannot interpret customer
intent, so it accepts that policy-eligible amount; the agent's authorization
mistake remains a regression. Direct traces and checks are in
`reports/support-calibrated-direct/`.

The partial-amount failure is now a separately reviewed posthoc scenario in
`examples/scenarios/support_extra/partial_amount.json`. It permits an optional
lookup but forbids refunds and escalations and requires a clarification question.
Its counterfactual saved refund response allows a bad trace to finish so the
tester reports the attempted action. It is excluded from the frozen 34-case
comparison and does not increase its pass count. An actual failing conversation
was also recorded as an unreviewed draft in
`reports/support-partial-failure-recording.json`; observed wrong behavior does not
become approved ground truth.

Replaying the reviewed case against the actual calibrated model failed **3/3**.
Every run looked up `ORD-1001` and attempted a 4,999-cent refund. The tester reported
`forbidden_call` and the missing clarification question; all traces completed
without runtime or fixture errors. The JSON, JUnit and terminal reports are
`reports/support-partial-calibrated-replay.*`. Business tools remain mocked in this
replay, so these attempts create no actual refunds.

With the corresponding base or adapter server running:

```sh
.venv-model/bin/python scripts/evaluate_support_model.py \
  --label qwen3-4b-base-native-aligned --diagnostic --repeats 3 \
  --output reports/support-base-native.json

# Restart the server with --adapter-path .models/support-lora-native, then:
.venv-model/bin/python scripts/evaluate_support_model.py \
  --label qwen3-4b-support-lora-native --diagnostic --repeats 3 \
  --output reports/support-adapter-native.json

# Restart with --adapter-path .models/support-lora-calibrated, then:
.venv-model/bin/python scripts/evaluate_support_model.py \
  --label qwen3-4b-support-lora-calibrated --diagnostic --repeats 3 \
  --output reports/support-adapter-calibrated.json
```

Each run has a JSON trace report, JUnit XML and a manifest with prompt/scenario
hashes, schema order, agent-source hash, instruction layout, model label, sampling,
timing and outcomes. The script preserves existing
report paths. An exit code of 1 is a recorded regression result, not a reason to
discard the experiment. No model weights, generated corpora or raw reports are
committed to Git.
