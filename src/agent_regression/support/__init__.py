"""Synthetic customer-support agent and reviewed policy regression scenarios."""

from .agent import make_support_agent
from .local_model import close_local_model, make_local_model
from .offline import make_offline_model
from .scenarios import support_scenarios
from .store import LocalSupportStore, SupportSession

__all__ = [
    "LocalSupportStore",
    "SupportSession",
    "close_local_model",
    "make_local_model",
    "make_offline_model",
    "make_support_agent",
    "support_scenarios",
]
