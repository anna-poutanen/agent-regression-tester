"""Optional language-model inference through a loopback OpenAI-compatible server."""

from __future__ import annotations

import math
from urllib.parse import urlsplit

from httpx2 import AsyncClient
from pydantic_ai.models import Model

DEFAULT_MODEL_NAME = "default_model"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


def validate_local_url(base_url: str) -> str:
    """Restrict this example to a server on the user's own computer."""
    parsed = urlsplit(base_url)
    if (
        any(character.isspace() for character in base_url)
        or parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Model URL must be an http(s) loopback URL without credentials or query")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Model URL contains an invalid port") from exc
    return base_url.rstrip("/")


def make_local_model(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Model:
    """Build a real model client; construction itself makes no network requests.

    The dummy key is only for compatibility with local inference servers. No
    environment API key or hosted provider is selected by this helper.
    """
    base_url = validate_local_url(base_url)
    if not model_name.strip():
        raise ValueError("Model name must not be blank")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("Temperature must be finite and between 0 and 2")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:
        raise ValueError("Local model support requires pip install '.[openai]'") from exc

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key="local-inference-only",
            http_client=AsyncClient(trust_env=False, follow_redirects=False),
        ),
        settings={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "parallel_tool_calls": False,
        },
    )


async def close_local_model(model: Model) -> None:
    """Close the explicit local HTTP client after all runs in an event loop.

    The provider treats a supplied HTTP client as caller-owned. Call this after
    finishing a suite, rather than after each iteration sharing the model.
    """
    provider = model.provider
    if provider is not None:
        await provider.client.close()
