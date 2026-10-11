"""Pytest fixtures for invoking regression runs inside application test suites."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from .runner import run_scenario, run_suite
from .schema import RunResult, SuiteResult


@pytest.fixture
def regression_runner() -> Callable[..., Awaitable[RunResult]]:
    """Return ``run_scenario``; await it and assert on the returned result."""
    return run_scenario


@pytest.fixture
def regression_suite_runner() -> Callable[..., Awaitable[SuiteResult]]:
    """Return ``run_suite`` for repeat runs and multiple scenarios."""
    return run_suite


__all__ = ["regression_runner", "regression_suite_runner"]
