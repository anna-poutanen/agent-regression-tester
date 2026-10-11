# Scenario format, version 1

A scenario is a UTF-8 JSON file. Unknown fields, duplicate JSON keys, unsupported schema versions, non-finite numbers, and implicit type conversions are rejected. Names and tool names must be nonblank. A recording may have no checks; a runnable test must have explicit checks and be reviewed by default.

```json
{
  "schema_version": 1,
  "name": "ineligible_order",
  "prompt": "Refund order 123 if eligible.",
  "fixtures": [
    {
      "tool_name": "lookup_order",
      "arguments": {"order_id": "123"},
      "response": {"eligible": false},
      "outcome": "return"
    }
  ],
  "recorded_calls": [],
  "checks": [
    {"kind": "required", "tool_name": "lookup_order", "arguments": {"order_id": "123"}},
    {"kind": "forbidden", "tool_name": "issue_refund"}
  ],
  "reviewed": true,
  "metadata": {"policy": "Ineligible orders cannot be refunded"}
}
```

If this agent calls `issue_refund`, replay blocks execution and reports the forbidden attempt. Add a synthetic `issue_refund` fixture if you also want the agent to continue and see how it explains the attempted action. Synthetic fixtures never authorize an action; checks establish the policy.

## Fixture matching

Fixtures match the tool name and **complete argument object**, independent of object key ordering. Extra arguments, changed IDs, different types, and missing keys do not match. Booleans differ from numbers; JSON numbers `1` and `1.0` compare equal. The next unused matching fixture is consumed. Multiple fixtures with identical arguments therefore supply ordered responses for retries or state changes. Unrelated fixture entries may be interleaved. Matching does not enforce global call order; use `before` checks for ordering.

An unmatched call or exhausted response queue fails the run. There is no live-tool fallback. Unused fixtures are allowed: a safe agent may never need a response prepared for a forbidden action.

To reproduce a tool-requested retry, set `"outcome": "retry"` and supply the saved retry message as `response`. Add a subsequent return fixture if the agent should retry successfully. All attempted calls count toward call constraints, including blocked and invalid calls.

## Behavior checks

| Kind | Fields | Meaning |
| --- | --- | --- |
| `required` | `tool_name`, optional `arguments`, `min_calls` (default 1) | At least this many matching attempted calls |
| `forbidden` | `tool_name`, optional `arguments` | No matching attempted call |
| `arguments` | `tool_name`, `path`, `expected`, `operator`, optional `match_arguments` | Every matching call satisfies the argument expectation; absence also fails |
| `before` | `first`, `then` | Every `then` attempt has a successful `first` result in an earlier model turn |
| `max_calls` | `tool_name`, `maximum` | No more than this many attempted calls |
| `output_contains` | `text` | Final text contains a case-sensitive substring |

Required/forbidden `arguments` and argument-check `match_arguments` are recursive object subsets. Lists and scalar values match exactly. For example, `{"customer": {"id": "cust-42"}}` also matches an object containing `customer.name`.

Argument operators:

- `equals` / `not_equals`: recursive JSON comparison against `expected`.
- `contains`: substring for strings, one equal element for lists, key membership for objects (string expected value).
- `exists`: the path must exist; its value may be null. `expected` is unused.

Dotted paths address object keys and nonnegative list indexes: `customer.id`, `items.0.sku`. Keys containing literal dots cannot be addressed in v1; assert a containing object's value instead. A missing path fails even for `not_equals`, which prevents missing fields from accidentally satisfying a policy.

`before` checks use model-turn indexes, not list position alone. Lookup and action emitted in one parallel batch fail an ordering requirement. This proves a successful earlier lookup occurred, but does not correlate the lookup's ID with the action's ID; add argument checks for the relevant IDs. A `before` check alone allows zero `then` calls; pair it with a required check when the action must happen.

## Recordings, review, and differences

`recorded_calls` is the observed baseline. Each entry has a tool name, raw argument object, response, outcome, optional call ID, and `step` (zero-based model turn). Tool-call IDs and timing are excluded from behavioral comparison. Trace differences highlight added/missing calls and changed arguments, responses, outcomes, and steps; differences are informational unless explicit checks fail.

Suggested checks describe observations. They cannot establish refund eligibility rules or distinguish correct from incorrect recorded behavior. Review and edit both fixtures and checks. Any change to expected behavior should reset `reviewed` to false until reviewed again. The review flag is a workflow convention, not a cryptographic approval record.

A scenario snapshots a fixed world. It is not a stateful API simulator. Hand-author distinct responses when the agent should observe changing state. It has one text prompt per run, no prior conversation history, and a text output agent in the initial release.
