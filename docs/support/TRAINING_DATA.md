# Synthetic expert demonstrations for local LoRA

The optional training experiment uses a separate, deterministic corpus of fictional support requests and correct tool decisions. It contains no customer conversations, production traces, contact information, or scraped training material. The generator never asks a language model or the offline intent script to produce labels. It follows the written support policy and executes the harmless local backend to obtain real, internally consistent tool responses.

The dataset generator is `src/agent_regression/support/training.py`. Run it locally into a fresh directory:

```sh
python -m agent_regression.support.training --output .training-data/support-native
```

The current corpus version is `synthetic-support-lora-native-v2`. The default seed is `20261011`; `--seed` changes the synthetic amounts, products, and delivery ages reproducibly. The output directory is ignored by Git. Generation refuses to replace an existing corpus. This avoids accidentally changing the training data for an experiment whose source data has already been fingerprinted.

## Corpus composition

| Split | Whole requests | Assistant decision records | Tool-call targets | Final-answer targets |
| --- | ---: | ---: | ---: | ---: |
| Train | 168 | 384 | 216 | 168 |
| Validation | 21 | 48 | 27 | 21 |
| Test | 21 | 48 | 27 | 21 |

The 21 intent/policy categories cover eligible refunds, outside-window orders, final sale, already-refunded orders, high-value approval, undelivered orders, unknown identifiers, inaccessible orders, missing identifiers, multiple possible identifiers, general human requests, order-bound human requests, status inquiries, explicit refusal of a refund, requests for a partial refund, repeated wording, instructions to bypass policy, policy exceptions, historical partial refunds, and the inclusive 30-day/31-day boundary.

Training uses eight complete wordings per category: four request templates with two different greetings. Validation and test use separately written request templates. Amounts, ages, items, identifiers, and trusted request keys vary independently. This is a compact controlled exercise; eight wordings do not represent the diversity of real support traffic.

## Correct actions and actual backend results

Every case has its own synthetic seed loaded through the same strict `LocalSupportStore.from_seed` API that validates application data. The expert first decides the kind of request from its authored category. Once an order is clear, the actual backend lookup supplies ownership, remaining amount, eligibility, and the policy reason.

Eligible requests produce a lookup, one full refund with the trusted request key, and a receipt-backed confirmation. High-value and delivery problems produce the appropriate human ticket after lookup. Ineligible or inaccessible orders produce no refund. Status inquiries and refusals produce lookup only. A partial-amount request produces a question about whether a full refund is wanted, rather than silently increasing the refund. Ambiguous orders produce a clarification question without tools. A general human request produces an unbound ticket with an explicit null order ID.

Ground truth includes successful receipts and their actual identifiers. It does not include replay continuation fixtures, denied forbidden actions used as successful training examples, or invented tickets. Repeated customer wording produces one action. Historical partial refunds are accounted for by refunding the full positive balance that remains; this differs from a customer requesting a new partial refund.

## Decision supervision format

Each row in `train.jsonl`, `valid.jsonl`, and `test.jsonl` has exactly two top-level fields:

```json
{"messages": ["conversation prefix ending in one assistant decision"], "tools": ["the actual three public function schemas"]}
```

The strings above describe the shape; actual records contain standard role/message objects and complete JSON tool schemas. Assistant tool frames use `tool_calls`, a function name, an argument **mapping**, and a call ID. OpenAI sends arguments as a JSON string, but the MLX server decodes that string before applying its native chat template. The corpus stores the same normalized mapping directly, preserving argument insertion order. Tool replies contain the corresponding call ID, name, and compact JSON-string backend response, serialized through the public `ToolReturnPart.model_response_str()` method used by Pydantic AI. This preserves actual backend response ordering and spacing.

Tool schemas are prepared using Pydantic AI's public `OpenAIJsonSchemaTransformer`, available without the optional OpenAI SDK. They match the local provider's strict preparation, including `function.strict: true`, the required nullable escalation order ID, and the prepared nested property order. For example, the nullable order field renders its description before its `anyOf` union. The function field order is name, description, parameters, then strict. Model-facing JSONL exports preserve insertion order; recursively sorting keys would change the native template even where Python dictionaries compare equal. The policy and trusted context form one combined system message, matching the aligned inference agent.

A single eligible request produces three records:

1. System policy and trusted context, customer request, then the assistant's `lookup_order` call.
2. That prefix plus the lookup result, then the assistant's `issue_refund` call.
3. That prefix plus the refund result, then the assistant's final confirmation.

The MLX trainer's prompt mask must supervise the final assistant decision in each row. Supplying only a complete conversation ending in a final answer would train the wording of the answer while leaving intermediate tool actions unsupervised. The generator therefore creates separate prefixes for every assistant decision.

`cases.jsonl` contains the audit data for each complete request: category, template ID, original customer prompt, exact seed, trusted key, full trajectory, tool calls/results, local ledger snapshot, and the indexes of its decision records. `manifest.json` identifies the generator version, seed, provenance, split strategy, counts, and case index. The audit seeds may contain fictional inaccessible orders; those details are never included in model-facing tool responses for an unauthorized customer.

## Split isolation and evaluation separation

All decisions from a request stay together. Train, validation, and test have disjoint whole request templates, order IDs, and request keys:

- Train identifiers are allocated from ORD-2000 upward.
- Validation identifiers are allocated from ORD-3000 upward.
- Test identifiers are allocated from ORD-3500 upward.
- Every request has a distinct `req-train-{split}-{number}` trusted key.

Neither the original 22 regression prompts nor the additional 12 development diagnostic prompts, their order IDs, or their request key appear in the corpus. The shared business policy and public tool schemas are intentionally the same across training and inference. Independent tests also compare customer wordings after removing identifiers and numbers, so nominally different IDs cannot conceal identical request templates across corpus splits.

The original regression suite was used during agent and prompt development. It is useful for before/after regression comparison but is not an untouched research benchmark. The additional diagnostic suite is also a small development diagnostic, not representative evidence of production reliability. Fine-tuning results should report actual measured tool-call failures and changes without promising broad generalization.

## Validation and limitations

Tests verify deterministic regeneration, all requested intent categories, absence of evaluation text/identifiers, separate normalized wording groups, per-request split grouping, nonempty decision supervision, exact tool schemas, and export preservation. An independent review replays every expert trajectory through a fresh backend and compares the responses and ledgers, checking causal order, amount, authorization, request key, duplicate prevention, and original-message preservation. Native format tests also compare initial, post-lookup, and post-action prefixes with actual in-memory OpenAI HTTP captures, MLX normalization, and the locally installed Qwen tokenizer. Independent validation checks every one of the 480 exported prefixes for both rendered-string and token-array equality. These checks use mock responses and local tokenizer files; they require neither an inference request nor a model download.

This dataset tests one English-language support policy on synthetic single-request conversations. It is small, authored, and visible to the implementation team. It does not establish performance on real customers, new policies, other languages, or arbitrary prompt injection. The adapter remains an experimental local artifact. The backend still enforces ownership, eligibility, amount, lookup, and idempotency independently of the model.


## Format correction after the first experiment

The original `synthetic-support-lora-v1` files remain frozen in `.training-data/support`. Its expert business decisions and split isolation were correct, but its serialized input differed from actual inference: it omitted prepared strict-schema metadata, stored unnormalized argument strings, sorted nested JSON keys, and formatted backend responses differently. The original inference agent also split policy and trusted context into separate system messages. Those changes affected native Qwen prompt tokens even though the JSON had equivalent meaning.

Version 2 corrects the serialization and aligns the single system message. It keeps the same expert requests, categories, identifiers, request keys, split boundaries, and backend outcomes. The new export is separate in `.training-data/support-native`; no original training files were replaced. Fixing format differences does not by itself establish that another adapter will improve behavior. A subsequent training experiment needs its own frozen data fingerprints and measured before/after regression reports.
