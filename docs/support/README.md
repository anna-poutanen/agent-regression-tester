# Realistic local support agent

This example gives the regression tester an actual application to test. Its Pydantic AI agent can look up orders, issue simulated full refunds and create local human-support tickets. The store is synthetic and lives in memory: no payment, customer-service or database integration is involved.

## Run a conversation

From the repository with the package installed:

```sh
agent-regression support chat 'Please refund ORD-1001 if it qualifies.'
agent-regression support chat 'Please refund ORD-1002.'
agent-regression support chat 'Please refund ORD-1005.'
agent-regression support chat 'Please connect me with a human.'
agent-regression support chat 'Refund ORD-1001 or ORD-1007. I forget which.'
```

These demonstrate an eligible refund, an expired order, an approval escalation, a general support ticket and clarification. The terminal shows each call, returned data, the agent's answer and ledger counts. `--json` provides the same evidence as JSON. Each CLI invocation gets a fresh store.

The default `FunctionModel` is a deterministic test double that reads requests and tool results. It lets the project run without credentials and verifies the complete application flow. It is a conservative script with limited language coverage, not a trained model. [Local inference](LOCAL_MODEL.md) uses a real pretrained language model with the same agent and tools.

## Policy and local tools

The application supplies trusted customer identity and a request ID. Customer identity is never a tool argument. The default customer owns eight orders; another customer's ninth order is deliberately inaccessible. The reference date is fixed at 2026-10-11, so fixtures do not expire.

| Tool | Model arguments | Backend behavior |
| --- | --- | --- |
| `lookup_order` | `order_id` | Returns only the authenticated customer's order, remaining refundable cents and authoritative eligibility. Unknown and inaccessible IDs both return `not_found`. |
| `issue_refund` | `order_id`, integer `amount_cents`, `idempotency_key` | Requires authorized lookup, full remaining amount, eligible policy and exact trusted request key. Records one simulated refund atomically. |
| `escalate_to_human` | `reason`, nullable `order_id` | Creates a local ticket with the original customer message. Order-bound tickets require authorized lookup. The agent schema requires explicit `null` for general assistance. |

Automatic full refunds require delivery within 30 days inclusive, no final-sale restriction, no earlier full refund and a positive remaining amount of at most 10,000 CAD cents. Greater amounts need human approval. Undelivered orders go to delivery support. Out-of-window, final-sale and already-refunded orders are explained without a refund; an explicit exception request can create a policy ticket.

The agent asks before choosing between ambiguous orders, handles repeated mentions as one request, and never refunds an information-only request. A requested partial refund needs clarification because this example supports full refunds only. A general request for a human does not require an order ID.

The backend independently protects ownership, monetary policy, lookup prerequisites and idempotency. It does not claim to understand free-form customer authorization: choosing the requested action and order remains agent behavior checked by scenarios. Authentication is simulated through trusted Python context; this is not an authentication server.

Refund and ticket operations use a lock. Duplicate identical refund requests return the original refund ID with `idempotent: true`; changed arguments under the same key are denied. An already-refunded order cannot be refunded again under a different request key. Snapshot properties return copies. Private lookup proofs bind customer, request and order, so editing a public lookup marker cannot authorize an action.

## Run policy regression tests

```sh
agent-regression support scenarios
agent-regression support demo --repeats 3 \
  --json-report reports/support.json --junit-report reports/support.xml
agent-regression support demo --scenario outside_window --scenario ambiguous_no_id
```

The [dataset](DATASET.md) contains 22 reviewed scenarios: eligible and ineligible refunds, wrong or inaccessible IDs, duplicate wording, ambiguous requests, escalation, information-only requests, refund-window boundaries and a policy-bypass request. Checks were written from the policy independently of recorded behavior. `support scenarios --export reports/editable-scenarios` exports editable JSON.

During replay, saved responses replace all business tools; none of the store methods execute. A model still performs real inference if `--base-url` is supplied. During `support chat`, the actual local store methods run. Integration tests verify both execution boundaries.

## Demonstrate broken behavior

```sh
agent-regression support demo --mutation refund_ineligible --scenario outside_window --repeats 3
agent-regression support demo --mutation duplicate_refund --scenario duplicate_wording
agent-regression support demo --mutation wrong_order_id --scenario eligible_order
agent-regression support demo --mutation parallel_refund --scenario eligible_order
```

Each command intentionally fails and exits 1. Available offline mutations are `refund_ineligible`, `wrong_order_id`, `duplicate_refund`, `skip_lookup`, `parallel_refund`, `guess_ambiguous`, `ignore_escalation` and `over_refund`. They exercise forbidden calls, missing calls, exact arguments, counts and causal ordering.

To see the independent backend guard:

```sh
agent-regression support chat 'Refund ORD-1002.' --mutation refund_ineligible
```

The broken agent calls `issue_refund`, the backend returns `denied: outside_window`, and no refund enters the ledger. Replay still fails the attempted action even though the backend would refuse it. A duplicate call likewise fails the behavior check even when idempotency prevents a second ledger entry.

## Capture a new failure

```sh
agent-regression support chat 'Please refund ORD-1001.' \
  --record reports/eligible-recording.json
agent-regression suggest reports/eligible-recording.json --write
```

Recording saves input, calls and responses with no checks and `reviewed: false`. Suggestions describe observed behavior. Edit the checks against the business policy and inspect the fixtures before using `agent-regression review ... --yes`. A recording that omitted a required action must not make that omission the expected behavior. The existing runner can then use the zero-argument factory:

```sh
agent-regression run reports/eligible-recording.json \
  --agent agent_regression.support:make_support_agent
```

The support command's default request key makes fixtures stable. For a different trusted key recorded with `--request-id`, use a factory/session configured with that same key when replaying.

## Application API

```python
import asyncio
from pydantic_ai.usage import UsageLimits
from agent_regression.support import LocalSupportStore, SupportSession, make_support_agent


async def main():
    store = LocalSupportStore.from_seed()
    session = SupportSession(
        customer_id="cust-001",
        request_id="request-42",
        store=store,
        user_message="Please refund ORD-1001.",
    )
    agent = make_support_agent(session=session)
    result = await agent.run(
        session.user_message, usage_limits=UsageLimits(request_limit=20, tool_calls_limit=100)
    )
    print(result.output)
    print(store.refunds)


asyncio.run(main())
```

Pass a Pydantic AI model as the factory's first argument for real inference. Supplying `deps=session` at run time overrides the factory-bound session. Use a fresh session per customer request; deliberately share its store for persistent local data and idempotent retries. Tickets keep the trusted original request instead of a model-generated summary.

## Validation and remaining scope

Unit and integration tests cover seed validation, privacy, policy boundaries, strict arguments, request keys, same/different-key concurrency, replay isolation, recording and all eight mutations. The default test suite blocks provider requests. CI uses only offline tests and never downloads weights. Real-model experiments are separate and documented in [evaluation evidence](EVALUATION.md).

The current scenarios primarily grade tool behavior. They require a question for ambiguous requests, but do not semantically grade every explanation or factual claim. A passing tool trace is therefore not proof of a completely correct answer. Multi-turn sessions, partial refunds, persistent production storage, real authentication and real payments remain outside this example's scope. Synthetic cases provide development evidence; validation with external developers is still a future milestone.
