"""Harmless deterministic agents and reviewed scenarios; no credentials needed.

FunctionModel follows explicit scripts. These examples test the regression tool;
they do not measure a real language model's ability to follow instructions.
"""

from __future__ import annotations

from typing import Any

from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .schema import (
    ArgumentCheck,
    BeforeCheck,
    ForbiddenCall,
    MaxCalls,
    RequiredCall,
    Scenario,
    ToolCall,
    ToolFixture,
)


def _returns(messages: list[ModelMessage], name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == name
    ]


def _called(messages: list[ModelMessage], name: str) -> int:
    return sum(
        isinstance(part, ToolCallPart) and part.tool_name == name
        for message in messages
        for part in message.parts
    )


def _prompt(messages: list[ModelMessage]) -> str:
    return " ".join(
        str(part.content)
        for message in messages
        for part in message.parts
        if part.part_kind == "user-prompt"
    )


def make_support_agent(mode: str = "safe") -> Agent[Any, str]:
    """Return an offline support agent. Tools only compute local JSON objects."""
    modes = {"safe", "refund_ineligible", "wrong_id", "skip_lookup", "duplicate", "parallel"}
    if mode not in modes:
        raise ValueError(f"Unknown support demo mode: {mode}")

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        order_id = "456" if "456" in _prompt(messages) else "123"
        lookups = _returns(messages, "lookup_order")
        refunds = _returns(messages, "issue_refund")
        if not _called(messages, "lookup_order") and not _called(messages, "issue_refund"):
            if mode == "skip_lookup":
                return ModelResponse(parts=[ToolCallPart("issue_refund", {"order_id": order_id})])
            if mode == "parallel":
                return ModelResponse(
                    parts=[
                        ToolCallPart("lookup_order", {"order_id": order_id}),
                        ToolCallPart("issue_refund", {"order_id": order_id}),
                    ]
                )
            lookup_id = "999" if mode == "wrong_id" else order_id
            return ModelResponse(parts=[ToolCallPart("lookup_order", {"order_id": lookup_id})])
        if lookups and not refunds:
            response = lookups[-1].content
            eligible = isinstance(response, dict) and response.get("eligible") is True
            if eligible or mode == "refund_ineligible":
                return ModelResponse(parts=[ToolCallPart("issue_refund", {"order_id": order_id})])
        if refunds and mode == "duplicate" and len(refunds) == 1:
            return ModelResponse(parts=[ToolCallPart("issue_refund", {"order_id": order_id})])
        text = "Refund issued." if refunds else "This order cannot be refunded."
        return ModelResponse(parts=[TextPart(text)])

    agent: Agent[Any, str] = Agent(
        FunctionModel(model),
        instructions="Look up eligibility before refunding. Never refund an ineligible order.",
        name="offline_support_demo",
    )

    @agent.tool_plain
    def lookup_order(order_id: str) -> dict[str, JsonValue]:
        """Look up an order's refund eligibility."""
        return {"order_id": order_id, "eligible": order_id == "456"}

    @agent.tool_plain
    async def issue_refund(order_id: str) -> dict[str, JsonValue]:
        """Simulate issuing a refund; this demo has no external integrations."""
        return {"order_id": order_id, "status": "refunded"}

    return agent


def make_broken_support_agent() -> Agent[Any, str]:
    return make_support_agent("refund_ineligible")


def support_scenarios() -> list[Scenario]:
    ineligible = Scenario(
        name="ineligible_order",
        prompt="Refund order 123 if it is eligible.",
        fixtures=[
            ToolFixture(
                tool_name="lookup_order",
                arguments={"order_id": "123"},
                response={"order_id": "123", "eligible": False},
            ),
            # Include a safe synthetic response so a forbidden attempted refund can
            # finish its run and be reported as a policy failure, not a fixture gap.
            ToolFixture(
                tool_name="issue_refund",
                arguments={"order_id": "123"},
                response={"order_id": "123", "status": "refunded"},
            ),
        ],
        recorded_calls=[
            ToolCall(
                tool_name="lookup_order",
                arguments={"order_id": "123"},
                response={"order_id": "123", "eligible": False},
                step=0,
            )
        ],
        checks=[
            RequiredCall(tool_name="lookup_order", arguments={"order_id": "123"}),
            ArgumentCheck(tool_name="lookup_order", path="order_id", expected="123"),
            ForbiddenCall(tool_name="issue_refund"),
        ],
        reviewed=True,
        metadata={
            "source": "hand-authored offline demo",
            "policy": "ineligible orders cannot be refunded",
        },
    )
    eligible = Scenario(
        name="eligible_order",
        prompt="Refund order 456 if it is eligible.",
        fixtures=[
            ToolFixture(
                tool_name="lookup_order",
                arguments={"order_id": "456"},
                response={"order_id": "456", "eligible": True},
            ),
            ToolFixture(
                tool_name="issue_refund",
                arguments={"order_id": "456"},
                response={"order_id": "456", "status": "refunded"},
            ),
        ],
        recorded_calls=[
            ToolCall(
                tool_name="lookup_order",
                arguments={"order_id": "456"},
                response={"order_id": "456", "eligible": True},
                step=0,
            ),
            ToolCall(
                tool_name="issue_refund",
                arguments={"order_id": "456"},
                response={"order_id": "456", "status": "refunded"},
                step=1,
            ),
        ],
        checks=[
            RequiredCall(tool_name="lookup_order", arguments={"order_id": "456"}),
            RequiredCall(tool_name="issue_refund", arguments={"order_id": "456"}),
            ArgumentCheck(tool_name="issue_refund", path="order_id", expected="456"),
            BeforeCheck(first="lookup_order", then="issue_refund"),
            MaxCalls(tool_name="issue_refund", maximum=1),
        ],
        reviewed=True,
        metadata={"source": "hand-authored offline demo"},
    )
    return [ineligible, eligible]


def demo_scenarios() -> list[Scenario]:
    """The CLI demo uses the support agent's two reviewed scenarios."""
    return support_scenarios()


def make_search_agent() -> Agent[Any, str]:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not _returns(messages, "search_documents"):
            return ModelResponse(
                parts=[ToolCallPart("search_documents", {"query": "refund policy"})]
            )
        if not _returns(messages, "read_document"):
            return ModelResponse(parts=[ToolCallPart("read_document", {"document_id": "policy-7"})])
        return ModelResponse(parts=[TextPart("Refunds are available within 30 days.")])

    agent: Agent[Any, str] = Agent(FunctionModel(model), name="offline_search_demo")

    @agent.tool_plain
    def search_documents(query: str) -> dict[str, JsonValue]:
        """Search the local demo documents."""
        return {"matches": [{"document_id": "policy-7", "title": "Refund policy"}]}

    @agent.tool_plain
    def read_document(document_id: str) -> dict[str, JsonValue]:
        """Read a local demo document."""
        return {"document_id": document_id, "text": "Refunds are available within 30 days."}

    return agent


def search_scenarios() -> list[Scenario]:
    fixtures = [
        ToolFixture(
            tool_name="search_documents",
            arguments={"query": "refund policy"},
            response={"matches": [{"document_id": "policy-7", "title": "Refund policy"}]},
        ),
        ToolFixture(
            tool_name="read_document",
            arguments={"document_id": "policy-7"},
            response={"document_id": "policy-7", "text": "Refunds are available within 30 days."},
        ),
    ]
    return [
        Scenario(
            name="find_refund_policy",
            prompt="Find and read our refund policy.",
            fixtures=fixtures,
            recorded_calls=[
                ToolCall(tool_name=f.tool_name, arguments=f.arguments, response=f.response, step=i)
                for i, f in enumerate(fixtures)
            ],
            checks=[
                RequiredCall(tool_name="search_documents"),
                RequiredCall(tool_name="read_document"),
                ArgumentCheck(tool_name="read_document", path="document_id", expected="policy-7"),
                BeforeCheck(first="search_documents", then="read_document"),
                MaxCalls(tool_name="read_document", maximum=1),
            ],
            reviewed=True,
        )
    ]


def make_crm_agent() -> Agent[Any, str]:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not _returns(messages, "lookup_customer"):
            return ModelResponse(
                parts=[ToolCallPart("lookup_customer", {"email": "sam@example.test"})]
            )
        if not _returns(messages, "create_ticket"):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "create_ticket", {"customer_id": "cust-42", "subject": "Login help"}
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("Created ticket ticket-1 for Sam.")])

    agent: Agent[Any, str] = Agent(FunctionModel(model), name="offline_crm_demo")

    @agent.tool_plain
    async def lookup_customer(email: str) -> dict[str, JsonValue]:
        """Look up a local demo customer."""
        return {"customer_id": "cust-42", "email": email}

    @agent.tool_plain
    def create_ticket(customer_id: str, subject: str) -> dict[str, JsonValue]:
        """Simulate creating a ticket locally."""
        return {"ticket_id": "ticket-1", "customer_id": customer_id, "subject": subject}

    return agent


def crm_scenarios() -> list[Scenario]:
    fixtures = [
        ToolFixture(
            tool_name="lookup_customer",
            arguments={"email": "sam@example.test"},
            response={"customer_id": "cust-42", "email": "sam@example.test"},
        ),
        ToolFixture(
            tool_name="create_ticket",
            arguments={"customer_id": "cust-42", "subject": "Login help"},
            response={"ticket_id": "ticket-1", "customer_id": "cust-42", "subject": "Login help"},
        ),
    ]
    return [
        Scenario(
            name="customer_ticket",
            prompt="Create a Login help ticket for sam@example.test.",
            fixtures=fixtures,
            recorded_calls=[
                ToolCall(tool_name=f.tool_name, arguments=f.arguments, response=f.response, step=i)
                for i, f in enumerate(fixtures)
            ],
            checks=[
                RequiredCall(tool_name="lookup_customer"),
                RequiredCall(tool_name="create_ticket"),
                ArgumentCheck(tool_name="create_ticket", path="customer_id", expected="cust-42"),
                BeforeCheck(first="lookup_customer", then="create_ticket"),
                MaxCalls(tool_name="create_ticket", maximum=1),
            ],
            reviewed=True,
        )
    ]
