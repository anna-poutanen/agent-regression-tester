"""Versioned, strict JSON artifacts shared by recording and regression runs."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
    field_validator,
    model_validator,
)


def _validate_json(value: Any) -> Any:
    """Reject lossy coercions and non-JSON values before Pydantic can convert them."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return value
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            _validate_json(item)
        return value
    raise ValueError(f"Not a JSON value: {type(value).__name__}")


SafeJson = Annotated[JsonValue, BeforeValidator(_validate_json)]
JsonObject = Annotated[dict[str, JsonValue], BeforeValidator(_validate_json)]
NonEmptyString = Annotated[str, Field(min_length=1)]


class ArtifactModel(BaseModel):
    """Artifacts reject unknown fields and avoid implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    @field_validator("*", mode="after")
    @classmethod
    def nonblank_strings(cls, value: Any, info: Any) -> Any:
        if info.field_name in {
            "tool_name",
            "name",
            "scenario_name",
            "prompt",
            "first",
            "then",
            "path",
            "text",
        }:
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{info.field_name} must not be blank")
        return value


class ToolCall(ArtifactModel):
    tool_name: NonEmptyString
    arguments: JsonObject
    response: SafeJson = None
    outcome: Literal["return", "retry", "error", "blocked", "pending"] = "return"
    tool_call_id: str | None = None
    step: int = Field(default=0, ge=0)


class ToolFixture(ArtifactModel):
    tool_name: NonEmptyString
    arguments: JsonObject
    response: SafeJson = None
    outcome: Literal["return", "retry"] = "return"


class RequiredCall(ArtifactModel):
    kind: Literal["required"] = "required"
    tool_name: NonEmptyString
    arguments: JsonObject | None = None
    min_calls: int = Field(default=1, ge=1)


class ForbiddenCall(ArtifactModel):
    kind: Literal["forbidden"] = "forbidden"
    tool_name: NonEmptyString
    arguments: JsonObject | None = None


class ArgumentCheck(ArtifactModel):
    kind: Literal["arguments"] = "arguments"
    tool_name: NonEmptyString
    path: NonEmptyString
    expected: SafeJson = None
    operator: Literal["equals", "not_equals", "contains", "exists"] = "equals"
    match_arguments: JsonObject | None = None

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        if any(not part or part != part.strip() for part in value.split(".")):
            raise ValueError("Argument paths must have nonempty dot-separated components")
        return value


class BeforeCheck(ArtifactModel):
    kind: Literal["before"] = "before"
    first: NonEmptyString
    then: NonEmptyString


class MaxCalls(ArtifactModel):
    kind: Literal["max_calls"] = "max_calls"
    tool_name: NonEmptyString
    maximum: int = Field(ge=0)


class OutputContains(ArtifactModel):
    kind: Literal["output_contains"] = "output_contains"
    text: NonEmptyString


Check = Annotated[
    RequiredCall | ForbiddenCall | ArgumentCheck | BeforeCheck | MaxCalls | OutputContains,
    Field(discriminator="kind"),
]


class Scenario(ArtifactModel):
    schema_version: Literal[1] = 1
    name: NonEmptyString
    prompt: NonEmptyString
    fixtures: list[ToolFixture] = Field(default_factory=list)
    recorded_calls: list[ToolCall] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    reviewed: bool = False
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value


class CheckFailure(ArtifactModel):
    code: NonEmptyString
    message: NonEmptyString
    tool_name: str | None = None
    expected: SafeJson = None
    observed: SafeJson = None
    call_index: int | None = Field(default=None, ge=0)


class TraceDifference(ArtifactModel):
    kind: NonEmptyString
    message: NonEmptyString
    tool_name: str | None = None
    call_index: int | None = Field(default=None, ge=0)


class RunResult(ArtifactModel):
    scenario_name: NonEmptyString
    iteration: int = Field(default=1, ge=1)
    calls: list[ToolCall] = Field(default_factory=list)
    output: SafeJson = None
    failures: list[CheckFailure] = Field(default_factory=list)
    replay_errors: list[str] = Field(default_factory=list)
    differences: list[TraceDifference] = Field(default_factory=list)
    duration_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return not self.failures and not self.replay_errors and self.error is None


class SuiteResult(ArtifactModel):
    results: list[RunResult] = Field(default_factory=list)
    planned_runs: int | None = Field(default=None, ge=0)
    status: Literal["completed", "incomplete"] = "completed"
    stop_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def consistent_progress(self) -> Self:
        if self.planned_runs is not None:
            if self.planned_runs < len(self.results):
                raise ValueError("planned_runs cannot be smaller than the completed result count")
            if self.status == "completed" and self.planned_runs != len(self.results):
                raise ValueError("A completed suite must contain every planned result")
        elif self.status == "incomplete":
            raise ValueError("An incomplete suite requires planned_runs")
        if self.status == "completed" and self.stop_reason is not None:
            raise ValueError("A completed suite cannot have a stop_reason")
        if self.stop_reason is not None and not self.stop_reason.strip():
            raise ValueError("stop_reason must not be blank")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_runs(self) -> int:
        return len(self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed_runs(self) -> int:
        return sum(result.passed for result in self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed_runs(self) -> int:
        return self.total_runs - self.passed_runs

    @computed_field  # type: ignore[prop-decorator]
    @property
    def remaining_runs(self) -> int:
        return (self.planned_runs if self.planned_runs is not None else self.total_runs) - len(
            self.results
        )
