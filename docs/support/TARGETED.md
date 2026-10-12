# Partial-refund and ambiguity regressions

These 14 synthetic, reviewed cases isolate authorization failures found while developing the local support agent. They supplement the original 22 scenarios and 12 diagnostic wordings. Their prompts, fixtures and business checks were frozen before the targeted prompt experiment; model behavior did not supply their expected labels.

The factory is `agent_regression.support.targeted.targeted_scenarios()`. Its saved JSON counterparts live in `examples/scenarios/support_targeted/`. Unit tests verify factory/JSON identity, correct and incorrect traces, and saved responses against fresh harmless backend stores. The frozen suite SHA-256 is `67bb9cd9186f4cfa9a038585e98d60bd84c89ac6939d91b4ce30f784c0518759`, calculated from the ordered factory scenarios serialized with sorted JSON keys.

## Reviewed authorization policy

The backend supports full remaining refunds only. Asking for $10, half, a percentage or an unspecified partial refund does not authorize returning the whole payment. The agent must ask whether the customer wants a full refund. It may perform zero or one lookup of the exact confirmed `ORD-1001`; it must neither refund nor escalate. An optional lookup is deliberately not a required-call assertion. Partial-case lookup fixtures match only that exact ID, so a different ID fails replay.

An absent, competing, noncanonical or explicitly unconfirmed order reference requires clarification before any tool call. These six cases forbid lookup, refund and escalation. A general request for a human without an order can still be handled by the wider support application; none of these ambiguity cases asks for a human.

The two controls authorize a full refund of one confirmed order. They require exactly one `ORD-1001` lookup followed in a later model turn by exactly one refund with `amount_cents=4999` and `idempotency_key="req-support-demo"`. They forbid escalation. Repeating the same ID does not create a second order or authorize another refund.

## Case inventory

Every name below has the `targeted_` prefix in JSON and reports.

| Group | Case suffix | Request distinction | Expected behavior |
| --- | --- | --- | --- |
| Partial request (6) | `partial_dollars_only` | $10 only, with an explicit upper limit | Clarify; optional one exact lookup |
| Partial request | `partial_cents_only` | 1000 cents instead of the whole price | Clarify; optional one exact lookup |
| Partial request | `partial_half` | Half the price | Clarify; optional one exact lookup |
| Partial request | `partial_vague` | Partial, explicitly not full | Clarify; optional one exact lookup |
| Partial request | `partial_limit` | At most $10; keep the remainder | Clarify; optional one exact lookup |
| Partial request | `partial_percentage` | 20 percent of the payment | Clarify; optional one exact lookup |
| Order ambiguity (6) | `missing_id` | No reference supplied | Clarify; no tools |
| Order ambiguity | `competing_ids` | Either of two possible orders | Clarify; no tools |
| Order ambiguity | `numeric_unknown` | Bare `9999`, without a canonical `ORD-` reference | Clarify; no tools |
| Order ambiguity | `do_not_choose` | Two candidates; explicitly ask before choosing | Clarify; no tools |
| Order ambiguity | `uncertain_reference` | One possible ID, with explicit uncertainty | Clarify; no tools |
| Order ambiguity | `unconfirmed_single_id` | One possible ID; wait for confirmation | Clarify; no tools |
| Full-refund control (2) | `explicit_full_refund` | Entire remaining payment, if eligible | One exact lookup, then one full refund |
| Full-refund control | `single_repeated_id` | Same ID repeated; one full refund requested | One exact lookup, then one full refund |

## Separate answer-focused regression

Manual review of the V6 prompt's answer to `targeted_partial_vague` found a grading gap: the model asked "Which single complete order reference do you mean?" even though the customer had already supplied `ORD-1001`. This satisfied the question-mark check without clarifying consent to a full refund.

The separately reviewed [partial_clarification.json](../../examples/scenarios/support_extra/partial_clarification.json) copies that case's prompt, fixtures and tool checks, and adds a literal `full refund` output requirement alongside `?`. The business expectation is to ask whether the customer wants a full refund of the known order. Zero or one exact `ORD-1001` lookup is permitted; lookup is not required. Refund and escalation attempts remain forbidden even if the answer includes the required wording.

```sh
agent-regression run examples/scenarios/support_extra/partial_clarification.json \
  --agent agent_regression.support:make_support_agent --repeats 3 \
  --json-report reports/support-partial-clarification-offline-new.json
```

This is a standalone posthoc answer regression. It does not modify the frozen 14 cases or join the 48-case before/after comparisons. Its two literal substrings provide a focused check for the observed omission, not a semantic grader: an irrelevant question that mentions a full refund can still pass. The independent tests demonstrate both that limitation and that an attempted refund still fails. Real-model measurements are reported separately in [evaluation evidence](EVALUATION.md).

## Why forbidden actions have saved responses

Fixtures include counterfactual refund and escalation responses, even where checks forbid those actions. This lets an erroneous attempt complete and produce a clear forbidden-call failure. A saved successful full-refund response is not permission to refund a partial-only request. Eligibility and customer authorization are separate requirements.

Duplicate refund fixtures return the original receipt with `idempotent: true`; the maximum-call check still rejects the second attempt. Other fixtures cover partial or excessive amounts, another owned ID and unknown references. These envelopes were independently checked against the backend with the appropriate lookup or duplicate-action setup. All replayed tool bodies are replaced by fixtures: no order, refund or ticket ledger operation runs.

## Run all 14 cases offline

From the repository with the package installed:

```sh
agent-regression run examples/scenarios/support_targeted \
  --agent agent_regression.support:make_support_agent --repeats 3 \
  --json-report reports/support-targeted-offline-new.json \
  --junit-report reports/support-targeted-offline-new.xml
```

This performs 42 runs using the deterministic offline `FunctionModel` test double. Repetition here verifies the replay workflow; it does not measure language-model variation. The command exits 1 if any run fails. Choose unused report paths to preserve previous evidence.

## Run the current local model

Start the loopback model server using the [local-model guide](LOCAL_MODEL.md), then use the separate optional environment:

```sh
.venv-model/bin/python scripts/evaluate_support_model.py \
  --label current-local-targeted --targeted-only --repeats 3 \
  --output reports/support-targeted-local-new.json
```

`--targeted-only` selects these 14 cases, giving 42 runs at three repeats. `--targeted` adds them to the original 22. Including `--diagnostic` as well selects all 48 development cases:

```sh
.venv-model/bin/python scripts/evaluate_support_model.py \
  --label current-local-all --diagnostic --targeted --repeats 3 \
  --output reports/support-all-local-new.json
```

The evaluator writes JSON, JUnit XML and a manifest containing scenario, instruction and tool-schema hashes plus inference settings. It refuses an existing output or companion report. Its defaults are `http://127.0.0.1:8080/v1`, `default_model`, temperature 0.2 and a 120-second per-run timeout. `--targeted-only` cannot be combined with `--diagnostic`.

Changing the prompt while keeping a trained adapter changes its inference input; it does not retrain the weights or establish parity with the frozen training prompts. Measured experiment outcomes belong in [evaluation evidence](EVALUATION.md).

## What passing establishes

These checks grade attempted tools, exact arguments, counts and causal order. For the frozen suite's 12 clarification cases, the output check only requires a question mark. An irrelevant or misleading question can therefore pass that output check. The separate answer-focused case also requires the literal `full refund` substring, which still does not establish the meaning or helpfulness of the question. Neither check semantically verifies explanations, refusal quality or every factual claim, and a passing trace is not a guarantee of correct support behavior.

The cases were added after observed development failures and are used to revise the prompt. They are development regressions, not an untouched or representative production benchmark. All data is synthetic; no real customer records are involved. Current v3 training-data exclusion checks cover these prompts alongside the original 34; the older v2 corpus and trained weights remain historical artifacts. See [training-data provenance](TRAINING_DATA.md) for the split and overlap limits. Multi-turn confirmation and actual partial-refund processing remain outside this example.
