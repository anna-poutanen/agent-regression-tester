"""Strict synthetic orders and atomic local support actions.

No operation contacts a payment service or database. Authentication is simulated
by a trusted ``SupportSession`` supplied by the application, never by tool args.
The backend independently checks ownership, policy, lookup and idempotency. The
agent remains responsible for understanding the user's intent: this module does
not claim that a keyword parser could securely authorize a refund.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

REFERENCE_DATE = date(2026, 10, 11)
REFUND_WINDOW_DAYS = 30
AUTOMATIC_REFUND_LIMIT_CENTS = 10_000
DEFAULT_CUSTOMER_ID = "cust-001"
DEFAULT_REQUEST_ID = "req-support-demo"

EscalationReason = Literal[
    "policy_exception", "delivery_issue", "customer_request", "payment_dispute", "account_access"
]
ESCALATION_REASONS = frozenset(
    {"policy_exception", "delivery_issue", "customer_request", "payment_dispute", "account_access"}
)
PolicyReason = Literal[
    "eligible",
    "outside_window",
    "final_sale",
    "already_refunded",
    "not_delivered",
    "requires_approval",
]


def _nonblank(value: str, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _seed_date(value: Any) -> date | None:
    """Require canonical JSON dates, excluding datetimes and coercible numbers."""
    if value is None:
        return None
    if type(value) is not str or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("Seed dates must use YYYY-MM-DD")
    return date.fromisoformat(value)


class _SeedOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    order_id: str = Field(pattern=r"^ORD-\d{4}$")
    customer_id: str
    item_summary: str
    total_cents: int = Field(gt=0)
    refunded_cents: int = Field(default=0, ge=0)
    currency: Literal["CAD"] = "CAD"
    status: Literal["delivered", "shipped"]
    delivered_at: date | None
    final_sale: bool = False

    @field_validator("customer_id", "item_summary")
    @classmethod
    def nonblank(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("delivered_at", mode="before")
    @classmethod
    def valid_date(cls, value: Any) -> date | None:
        return _seed_date(value)

    @model_validator(mode="after")
    def consistent_order(self) -> _SeedOrder:
        if self.refunded_cents > self.total_cents:
            raise ValueError("refunded_cents cannot exceed total_cents")
        if (self.status == "delivered") != (self.delivered_at is not None):
            raise ValueError("Only delivered orders must have a delivered_at date")
        return self


class _SeedData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    reference_date: date
    orders: list[_SeedOrder] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    @field_validator("reference_date", mode="before")
    @classmethod
    def valid_date(cls, value: Any) -> date:
        result = _seed_date(value)
        if result is None:
            raise ValueError("reference_date cannot be null")
        return result

    @model_validator(mode="after")
    def consistent_orders(self) -> _SeedData:
        identifiers = [order.order_id for order in self.orders]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Seed order IDs must be unique")
        if any(
            order.delivered_at is not None and order.delivered_at > self.reference_date
            for order in self.orders
        ):
            raise ValueError("Delivered dates cannot be after the reference date")
        return self


@dataclass(eq=False, slots=True, weakref_slot=True, init=False)
class SupportSession:
    """Trusted, per-request application context with an isolated store by default.

    The fixed default request key makes fixtures stable across fresh test runs.
    Applications should supply a distinct key per real request and reuse it for
    retries. Customer credentials are trusted Python context, not user content.
    """

    customer_id: str
    request_id: str
    store: LocalSupportStore
    user_message: str
    looked_up_order_ids: set[str]

    def __init__(
        self,
        customer_id: str = DEFAULT_CUSTOMER_ID,
        request_id: str = DEFAULT_REQUEST_ID,
        store: LocalSupportStore | None = None,
        user_message: str = "",
    ) -> None:
        self.customer_id = _nonblank(customer_id, "customer_id")
        self.request_id = _nonblank(request_id, "request_id")
        if type(user_message) is not str:
            raise ValueError("user_message must be a string")
        if store is not None and not isinstance(store, LocalSupportStore):
            raise ValueError("store must be a LocalSupportStore")
        self.store = store if store is not None else LocalSupportStore.from_seed()
        self.user_message = user_message
        self.looked_up_order_ids = set()


class LocalSupportStore:
    """In-memory, thread-safe store loaded from strict synthetic JSON.

    Policy priority is not_delivered, final_sale, already_refunded,
    outside_window, requires_approval, eligible. Refunds always cover the full
    positive remaining amount. The 30-day boundary and 10,000-cent cap are
    inclusive. Snapshot properties are administrative testing observations;
    tool responses disclose only the authenticated customer's order data.
    """

    def __init__(self, seed: _SeedData) -> None:
        self._reference_date = seed.reference_date
        self._orders = {order.order_id: order for order in seed.orders}
        self._refunded_cents = {order.order_id: order.refunded_cents for order in seed.orders}
        self._lock = RLock()
        self._successful_lookups: WeakKeyDictionary[SupportSession, set[tuple[str, str, str]]] = (
            WeakKeyDictionary()
        )
        self._refunds: list[dict[str, JsonValue]] = []
        self._escalations: list[dict[str, JsonValue]] = []
        self._audit_events: list[dict[str, JsonValue]] = []
        self._refund_requests: dict[tuple[str, str], tuple[str, int, dict[str, JsonValue]]] = {}
        self._escalation_requests: dict[tuple[str, str, str, str | None], dict[str, JsonValue]] = {}

    @classmethod
    def from_seed(cls, path: str | Path | None = None) -> LocalSupportStore:
        """Load bundled synthetic orders, or validate a custom local seed file."""
        content = (
            Path(path).read_text(encoding="utf-8")
            if path is not None
            else files("agent_regression")
            .joinpath("support/data/orders.json")
            .read_text(encoding="utf-8")
        )
        return cls(_SeedData.model_validate_json(content))

    @property
    def refunds(self) -> list[dict[str, JsonValue]]:
        with self._lock:
            return deepcopy(self._refunds)

    @property
    def escalations(self) -> list[dict[str, JsonValue]]:
        with self._lock:
            return deepcopy(self._escalations)

    @property
    def audit_events(self) -> list[dict[str, JsonValue]]:
        with self._lock:
            return deepcopy(self._audit_events)

    def snapshot(self) -> dict[str, JsonValue]:
        """Return JSON-safe, defensive local ledgers without dumping seed orders."""
        with self._lock:
            return {
                "refunds": [deepcopy(row) for row in self._refunds],
                "escalations": [deepcopy(row) for row in self._escalations],
                "audit_events": [deepcopy(row) for row in self._audit_events],
            }

    def _validate_session(self, session: SupportSession) -> None:
        if not isinstance(session, SupportSession) or session.store is not self:
            raise ValueError("Session must belong to this store")
        _nonblank(session.customer_id, "customer_id")
        _nonblank(session.request_id, "request_id")
        if type(session.user_message) is not str:
            raise ValueError("user_message must be a string")

    def _owned_order(self, session: SupportSession, order_id: str) -> _SeedOrder | None:
        if type(order_id) is not str:
            raise ValueError("order_id must be a string")
        order = self._orders.get(order_id)
        return order if order is not None and order.customer_id == session.customer_id else None

    def _reason(self, order: _SeedOrder) -> PolicyReason:
        if order.status != "delivered" or order.delivered_at is None:
            return "not_delivered"
        if order.final_sale:
            return "final_sale"
        remaining = order.total_cents - self._refunded_cents[order.order_id]
        if remaining <= 0:
            return "already_refunded"
        if (self._reference_date - order.delivered_at).days > REFUND_WINDOW_DAYS:
            return "outside_window"
        if remaining > AUTOMATIC_REFUND_LIMIT_CENTS:
            return "requires_approval"
        return "eligible"

    def _lookup_completed(self, session: SupportSession, order_id: str) -> bool:
        # Bind authorization history to the credentials and request at lookup
        # time, rather than trusting a public marker or subsequently edited ID.
        return (
            session.customer_id,
            session.request_id,
            order_id,
        ) in self._successful_lookups.get(session, set())

    def lookup_order(self, session: SupportSession, order_id: str) -> dict[str, JsonValue]:
        with self._lock:
            self._validate_session(session)
            order = self._owned_order(session, order_id)
            if order is None:
                return {"status": "not_found", "order_id": order_id}
            self._successful_lookups.setdefault(session, set()).add(
                (session.customer_id, session.request_id, order_id)
            )
            session.looked_up_order_ids.add(order_id)
            self._audit_events.append(
                {
                    "action": "lookup_order",
                    "customer_id": session.customer_id,
                    "request_id": session.request_id,
                    "order_id": order_id,
                }
            )
            reason = self._reason(order)
            days: JsonValue = (
                (self._reference_date - order.delivered_at).days
                if order.delivered_at is not None
                else None
            )
            return {
                "status": "found",
                "order": {
                    "order_id": order_id,
                    "status": order.status,
                    "delivered_days_ago": days,
                    "total_cents": order.total_cents,
                    "refundable_cents": order.total_cents - self._refunded_cents[order_id],
                    "currency": order.currency,
                    "item_summary": order.item_summary,
                },
                "refund_eligibility": {"eligible": reason == "eligible", "reason_code": reason},
            }

    def issue_refund(
        self,
        session: SupportSession,
        order_id: str,
        amount_cents: int,
        idempotency_key: str,
    ) -> dict[str, JsonValue]:
        with self._lock:
            self._validate_session(session)

            def denied(reason: str) -> dict[str, JsonValue]:
                return {"status": "denied", "order_id": order_id, "reason_code": reason}

            order = self._owned_order(session, order_id)
            if order is None:
                return denied("not_found")
            if type(idempotency_key) is not str or idempotency_key != session.request_id:
                return denied("invalid_idempotency_key")
            if type(amount_cents) is not int or amount_cents <= 0:
                return denied("invalid_amount")
            request_key = (session.customer_id, session.request_id)
            existing = self._refund_requests.get(request_key)
            if existing is not None:
                if existing[0] != order_id or existing[1] != amount_cents:
                    return denied("idempotency_conflict")
                return {**deepcopy(existing[2]), "idempotent": True}
            if not self._lookup_completed(session, order_id):
                return denied("lookup_required")
            reason = self._reason(order)
            if reason != "eligible":
                return denied(reason)
            if amount_cents != order.total_cents - self._refunded_cents[order_id]:
                return denied("invalid_amount")
            response: dict[str, JsonValue] = {
                "status": "refunded",
                "order_id": order_id,
                "refund_id": f"refund-{len(self._refunds) + 1:04d}",
                "amount_cents": amount_cents,
                "currency": "CAD",
                "idempotent": False,
            }
            self._refunded_cents[order_id] += amount_cents
            record: dict[str, JsonValue] = {
                **response,
                "customer_id": session.customer_id,
                "request_id": session.request_id,
            }
            self._refunds.append(record)
            self._refund_requests[request_key] = (order_id, amount_cents, deepcopy(response))
            self._audit_events.append({"action": "issue_refund", **deepcopy(record)})
            return deepcopy(response)

    def escalate_to_human(
        self,
        session: SupportSession,
        reason: EscalationReason,
        order_id: str | None = None,
    ) -> dict[str, JsonValue]:
        with self._lock:
            self._validate_session(session)
            if type(reason) is not str or reason not in ESCALATION_REASONS:
                raise ValueError(f"Unsupported escalation reason: {reason!r}")
            if order_id is not None:
                if self._owned_order(session, order_id) is None:
                    return {"status": "denied", "order_id": order_id, "reason_code": "not_found"}
                if not self._lookup_completed(session, order_id):
                    return {
                        "status": "denied",
                        "order_id": order_id,
                        "reason_code": "lookup_required",
                    }
            request_key = (session.customer_id, session.request_id, reason, order_id)
            existing = self._escalation_requests.get(request_key)
            if existing is not None:
                return {**deepcopy(existing), "idempotent": True}
            response: dict[str, JsonValue] = {
                "status": "escalated",
                "ticket_id": f"ticket-{len(self._escalations) + 1:04d}",
                "reason": reason,
                "order_id": order_id,
                "idempotent": False,
            }
            record: dict[str, JsonValue] = {
                **response,
                "customer_id": session.customer_id,
                "request_id": session.request_id,
                "user_message": session.user_message,
            }
            self._escalations.append(record)
            self._escalation_requests[request_key] = deepcopy(response)
            self._audit_events.append({"action": "escalate_to_human", **deepcopy(record)})
            return deepcopy(response)
