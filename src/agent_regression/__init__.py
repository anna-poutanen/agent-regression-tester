"""Reviewed tool-call regression testing for Pydantic AI."""

from .adapter import record, scenario_from_messages
from .checks import diff_traces, evaluate_checks
from .runner import run_scenario, run_suite, run_suite_sync
from .schema import (
    ArgumentCheck,
    BeforeCheck,
    Check,
    CheckFailure,
    ForbiddenCall,
    MaxCalls,
    OutputContains,
    RequiredCall,
    RunResult,
    Scenario,
    SuiteResult,
    ToolCall,
    ToolFixture,
    TraceDifference,
)
from .storage import load_scenario, save_scenario
from .suggest import suggest_checks

__version__ = "0.1.0"

__all__ = [
    "ArgumentCheck",
    "BeforeCheck",
    "Check",
    "CheckFailure",
    "ForbiddenCall",
    "MaxCalls",
    "OutputContains",
    "RequiredCall",
    "RunResult",
    "Scenario",
    "SuiteResult",
    "ToolCall",
    "ToolFixture",
    "TraceDifference",
    "diff_traces",
    "evaluate_checks",
    "load_scenario",
    "record",
    "run_scenario",
    "run_suite",
    "run_suite_sync",
    "save_scenario",
    "scenario_from_messages",
    "suggest_checks",
]
