"""No test in this repository may call a provider model."""

import pytest
from pydantic_ai import models


@pytest.fixture(autouse=True)
def block_live_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
