from __future__ import annotations

import builtins
import sys
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.models.function import FunctionModel

from agent_regression.support.local_model import (
    close_local_model,
    make_local_model,
    validate_local_url,
)


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8080/v1", "http://localhost:11434/v1/", "https://[::1]:8080/v1"],
)
def test_local_url_allowed(url: str) -> None:
    assert validate_local_url(url) == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://example.com/v1",
        "http://127.0.0.1.evil.com/v1",
        "file:///tmp/model",
        "http://localhost@evil.com/v1",
        "http://secret@localhost:8080/v1",
        "http://localhost:8080/v1?key=secret",
        "http://localhost:8080/v1#part",
        "http://localhost:bad/v1",
        "http://localhost:99999/v1",
        "http://localhost:8080/v1\n",
        "localhost:8080/v1",
        "http:///v1",
    ],
)
def test_nonlocal_or_malformed_urls_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        validate_local_url(url)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": " "},
        {"temperature": float("nan")},
        {"temperature": -1},
        {"temperature": 3},
        {"max_tokens": 0},
        {"max_tokens": True},
    ],
)
def test_invalid_settings_rejected_before_optional_imports(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make_local_model(**kwargs)


def test_missing_optional_dependency_has_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def import_without_openai(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pydantic_ai.models.openai":
            raise ImportError("simulated missing optional package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_openai)
    with pytest.raises(ValueError, match=r"\[openai\]"):
        make_local_model()


def test_local_model_explicit_provider_settings_ignore_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def provider(**kwargs: Any) -> object:
        captured["provider"] = kwargs
        return object()

    def model(name: str, **kwargs: Any) -> FunctionModel:
        captured["name"] = name
        captured["settings"] = kwargs["settings"]
        return FunctionModel(lambda _messages, _info: None)  # type: ignore[arg-type]

    provider_module = types.ModuleType("pydantic_ai.providers.openai")
    provider_module.OpenAIProvider = provider  # type: ignore[attr-defined]
    model_module = types.ModuleType("pydantic_ai.models.openai")
    model_module.OpenAIChatModel = model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, provider_module.__name__, provider_module)
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setenv("OPENAI_API_KEY", "must-never-use-this")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com")
    result = make_local_model(model_name="local-model", temperature=0.7, max_tokens=800)
    assert isinstance(result, FunctionModel)
    assert captured["provider"]["base_url"] == "http://127.0.0.1:8080/v1"
    assert captured["provider"]["api_key"] == "local-inference-only"
    client = captured["provider"]["http_client"]
    assert client.trust_env is False
    assert client.follow_redirects is False
    assert captured["name"] == "local-model"
    assert captured["settings"] == {
        "temperature": 0.7,
        "max_tokens": 800,
        "parallel_tool_calls": False,
    }


async def test_explicit_local_client_cleanup() -> None:
    close = AsyncMock()

    class ModelWithClient(FunctionModel):
        @property
        def provider(self) -> Any:
            return types.SimpleNamespace(client=types.SimpleNamespace(close=close))

    model = ModelWithClient(lambda _messages, _info: None)  # type: ignore[arg-type]
    await close_local_model(model)
    close.assert_awaited_once()
    await close_local_model(FunctionModel(lambda _messages, _info: None))  # type: ignore[arg-type]
