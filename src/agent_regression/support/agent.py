"""Pydantic AI support agent with configurable model and local-only tools."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import JsonValue, Strict
from pydantic_ai import Agent, RunContext

from .offline import make_offline_model
from .prompt import instructions_for
from .store import SupportSession

INSTRUCTION_LAYOUT = "single_system_policy_and_context"

EscalationReason = Literal[
    "policy_exception", "delivery_issue", "customer_request", "payment_dispute", "account_access"
]


def make_support_agent(
    model: Any = None,
    *,
    session: SupportSession | None = None,
    prompt_variant: str = "safe",
) -> Agent[Any, str]:
    """Create the example agent; all tool effects stay in a synthetic local store.

    Supply a Pydantic AI Model or model name for real inference. Omitting model
    uses a deterministic FunctionModel test double, never an external API.
    RunContext deps override the factory-bound session when provided. Use a fresh
    session for each customer request; share its store deliberately for retries.
    """
    instructions = instructions_for(prompt_variant)
    bound_session = session if session is not None else SupportSession()
    selected_model = (
        make_offline_model(mutation=prompt_variant, request_id=bound_session.request_id)
        if model is None
        else model
    )
    agent: Agent[Any, str] = Agent(
        selected_model,
        name="local_support_agent",
        # Real tool arguments are strict to avoid silently coercing amounts/IDs.
        retries=1,
    )

    def resolve_session(ctx: RunContext[Any]) -> SupportSession:
        current = ctx.deps if ctx.deps is not None else bound_session
        if not isinstance(current, SupportSession):
            raise TypeError("Support agent deps must be a SupportSession or None.")
        # Capture the actual customer input, never a model-generated ticket body.
        # Callers can set it explicitly; an existing trusted original is retained.
        if not current.user_message and isinstance(ctx.prompt, str):
            current.user_message = ctx.prompt
        return current

    @agent.instructions
    def trusted_context(ctx: RunContext[Any]) -> str:
        current = resolve_session(ctx)
        return (
            instructions + f"\nTrusted authenticated customer ID: {current.customer_id}\n"
            f"Trusted request ID: {current.request_id}\n"
            "Policy reference date: 2026-10-11. Use only these application credentials."
        )

    @agent.tool
    async def lookup_order(
        ctx: RunContext[Any], order_id: Annotated[str, Strict()]
    ) -> dict[str, JsonValue]:
        """Look up an order belonging to the authenticated customer.

        Args:
            order_id: Exact customer-supplied order ID including its ORD- prefix.
        """
        current = resolve_session(ctx)
        return current.store.lookup_order(current, order_id)

    @agent.tool
    async def issue_refund(
        ctx: RunContext[Any],
        order_id: Annotated[str, Strict()],
        amount_cents: Annotated[int, Strict()],
        idempotency_key: Annotated[str, Strict()],
    ) -> dict[str, JsonValue]:
        """Simulate one full refund after a successful eligibility lookup.

        Args:
            order_id: Exact authorized order ID from the completed lookup.
            amount_cents: Integer CAD cents, exactly the lookup refundable_cents.
            idempotency_key: Exact trusted request ID supplied in application instructions.
        """
        current = resolve_session(ctx)
        return current.store.issue_refund(current, order_id, amount_cents, idempotency_key)

    @agent.tool
    async def escalate_to_human(
        ctx: RunContext[Any], reason: EscalationReason, order_id: str | None
    ) -> dict[str, JsonValue]:
        """Create a local human-support ticket containing the original request.

        Args:
            reason: Why human help is needed, using one of the provided categories.
            order_id: Required: authorized looked-up order, or explicit null for general help.
        """
        current = resolve_session(ctx)
        return current.store.escalate_to_human(current, reason, order_id)

    return agent
