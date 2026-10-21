"""Strict, locked, atomic checkpoints for serial regression suites.

Checkpoints contain completed results, including failures, in the suite's exact
scenario/iteration order. A caller-supplied ``run_id`` attests to application
configuration that cannot safely be discovered by inspecting Python agents.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import stat
import sys
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from .schema import ArtifactModel, RunResult, Scenario, SuiteResult

_RESULT_EXCLUDE: dict[str, Any] = {"results": {"__all__": {"passed"}}}
_IDENTITY_FIELDS = (
    "schema_version",
    "run_id",
    "scenarios_sha256",
    "scenario_names",
    "repeats",
    "timeout",
    "allow_unreviewed",
    "runtime_versions",
)


class CheckpointState(ArtifactModel):
    """A plan identity and its validated, completed result prefix."""

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    scenarios_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_names: list[str] = Field(min_length=1)
    repeats: int = Field(ge=1)
    timeout: float = Field(gt=0, allow_inf_nan=False)
    allow_unreviewed: bool
    runtime_versions: dict[str, str]
    results: list[RunResult] = Field(default_factory=list)
    status: Literal["completed", "incomplete"] = "incomplete"
    stop_reason: str | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("Checkpoint schema_version must be the integer 1")
        return value

    @field_validator("run_id", "stop_reason")
    @classmethod
    def nonblank_identity(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Checkpoint identity/reason must not be blank")
        return value

    @field_validator("scenario_names")
    @classmethod
    def unique_scenario_names(cls, names: list[str]) -> list[str]:
        if any(not name.strip() for name in names):
            raise ValueError("Checkpoint scenario names must not be blank")
        if len(set(names)) != len(names):
            raise ValueError("Checkpoint scenarios must have unique names")
        return names

    @field_validator("runtime_versions")
    @classmethod
    def nonblank_versions(cls, versions: dict[str, str]) -> dict[str, str]:
        if not versions or any(
            not key.strip() or not value.strip() for key, value in versions.items()
        ):
            raise ValueError("Checkpoint runtime_versions must contain nonblank version strings")
        return versions

    @model_validator(mode="after")
    def ordered_result_prefix(self) -> Self:
        planned = len(self.scenario_names) * self.repeats
        if len(self.results) > planned:
            raise ValueError("Checkpoint has more results than planned runs")
        for index, result in enumerate(self.results):
            name = self.scenario_names[index // self.repeats]
            iteration = index % self.repeats + 1
            if result.scenario_name != name or result.iteration != iteration:
                raise ValueError(
                    f"Checkpoint result {index} must be ({name!r}, iteration {iteration}); "
                    f"observed ({result.scenario_name!r}, iteration {result.iteration})"
                )
        if self.status == "completed" and (
            len(self.results) != planned or self.stop_reason is not None
        ):
            raise ValueError(
                "Completed checkpoint must contain all planned runs and no stop_reason"
            )
        return self

    def to_suite(self) -> SuiteResult:
        """Reconstruct a reportable suite using freshly calculated result totals."""
        state = _snapshot(self)
        return SuiteResult(
            results=state.results,
            planned_runs=len(state.scenario_names) * state.repeats,
            status=state.status,
            stop_reason=state.stop_reason,
        )


def runtime_versions() -> dict[str, str]:
    """Identify installed runtimes without importing any provider or model code."""
    versions = {"python": platform.python_version()}
    for key, distribution in (
        ("agent_regression", "agent-regression-tester"),
        ("pydantic_ai", "pydantic-ai-slim"),
        ("pydantic", "pydantic"),
    ):
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = "uninstalled"
    return versions


def build_checkpoint(
    scenarios: Sequence[Scenario],
    *,
    run_id: str,
    repeats: int = 1,
    timeout: float = 30,
    allow_unreviewed: bool = False,
    runtime_versions: dict[str, str] | None = None,
) -> CheckpointState:
    """Hash the full ordered scenario snapshots and create an empty plan state."""
    if any(not isinstance(scenario, Scenario) for scenario in scenarios):
        raise ValueError("Checkpoint plan must contain Scenario artifacts")
    snapshots = [
        Scenario.model_validate(scenario.model_dump(mode="python", warnings=False))
        for scenario in scenarios
    ]
    canonical = json.dumps(
        [scenario.model_dump(mode="json") for scenario in snapshots],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    versions = runtime_versions if runtime_versions is not None else _current_runtime_versions()
    return CheckpointState(
        run_id=run_id,
        scenarios_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        scenario_names=[scenario.name for scenario in snapshots],
        repeats=repeats,
        timeout=timeout,
        allow_unreviewed=allow_unreviewed,
        runtime_versions=versions,
    )


# Keep the public keyword ``runtime_versions`` independent of its helper name.
_current_runtime_versions = runtime_versions


def _snapshot(state: CheckpointState) -> CheckpointState:
    if not isinstance(state, CheckpointState):
        raise ValueError("Expected a CheckpointState artifact")
    return CheckpointState.model_validate(
        state.model_dump(mode="python", exclude=_RESULT_EXCLUDE, warnings=False)
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint JSON object key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Invalid checkpoint JSON constant: {value}")


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Checkpoint paths must not be symlinks: {path}")


def _lock_file(descriptor: int) -> None:
    if sys.platform == "win32":  # pragma: no cover - exercised by Windows installations
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(descriptor: int) -> None:
    if sys.platform == "win32":  # pragma: no cover - exercised by Windows installations
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    if sys.platform == "win32":  # pragma: no cover - Windows cannot open directories this way
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CheckpointStore:
    """Hold a nonblocking OS file lock for an entire controlled suite invocation.

    The ``.lock`` sidecar is deliberately persistent: deleting an active lock's
    inode could let another process obtain an unrelated lock under the same name.
    Every operation requires this context manager's live lock.
    """

    def __init__(self, path: str | Path) -> None:
        requested = Path(path).expanduser().absolute()
        self.path = requested.parent.resolve() / requested.name
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._descriptor: int | None = None
        self._identity: dict[str, Any] | None = None
        self._committed_results: list[str] = []

    def __enter__(self) -> Self:
        if self._descriptor is not None:
            raise ValueError("CheckpointStore is already locked")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink(self.path)
        _reject_symlink(self.lock_path)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Checkpoint lock must be a regular file")
            try:
                _lock_file(descriptor)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise ValueError(f"Checkpoint is already locked: {self.path}") from exc
                raise
            self._descriptor = descriptor
            self._require_lock()
        except BaseException:
            self._descriptor = None
            os.close(descriptor)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        descriptor, self._descriptor = self._descriptor, None
        self._identity = None
        self._committed_results = []
        if descriptor is not None:
            try:
                _unlock_file(descriptor)
            finally:
                os.close(descriptor)

    def _require_lock(self) -> None:
        if self._descriptor is None:
            raise ValueError("CheckpointStore operations require an active context manager")
        _reject_symlink(self.path)
        _reject_symlink(self.lock_path)
        actual = os.fstat(self._descriptor)
        current = self.lock_path.stat(follow_symlinks=False)
        if (actual.st_dev, actual.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("Checkpoint lock file changed while locked")

    @staticmethod
    def _plan_identity(state: CheckpointState) -> dict[str, Any]:
        return {field: deepcopy(getattr(state, field)) for field in _IDENTITY_FIELDS}

    def _check_identity(self, state: CheckpointState, expected: CheckpointState) -> None:
        for field in _IDENTITY_FIELDS:
            if getattr(state, field) != getattr(expected, field):
                raise ValueError(
                    f"Checkpoint plan mismatch: {field} changed; start a new checkpoint"
                )

    @staticmethod
    def _result_payloads(state: CheckpointState) -> list[str]:
        return [
            json.dumps(
                result.model_dump(exclude={"passed"}, mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for result in state.results
        ]

    def create(self, state: CheckpointState) -> CheckpointState:
        """Create a new checkpoint; existing directory entries are never overwritten."""
        self._require_lock()
        state = _snapshot(state)
        self._write(state, overwrite=False)
        self._identity = self._plan_identity(state)
        self._committed_results = self._result_payloads(state)
        return state

    def load(self, expected: CheckpointState | None = None) -> CheckpointState:
        """Load a strict result prefix, optionally verifying its entire plan identity."""
        self._require_lock()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(self.path, flags)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Checkpoint must be a regular file")
            data = json.load(
                stream, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
            )
        state = CheckpointState.model_validate(data)
        if expected is not None:
            self._check_identity(state, _snapshot(expected))
        self._identity = self._plan_identity(state)
        self._committed_results = self._result_payloads(state)
        return state

    def save(self, state: CheckpointState) -> None:
        """Atomically persist the next completed prefix under the existing plan."""
        self._require_lock()
        if self._identity is None:
            raise ValueError("Create or load the checkpoint before saving")
        state = _snapshot(state)
        if self._plan_identity(state) != self._identity:
            raise ValueError("Checkpoint plan identity cannot change during a run")
        results = self._result_payloads(state)
        if results[: len(self._committed_results)] != self._committed_results:
            raise ValueError("Checkpoint completed results cannot be removed or rewritten")
        self._write(state, overwrite=True)
        self._committed_results = results

    def _write(self, state: CheckpointState, *, overwrite: bool) -> None:
        self._require_lock()
        data = state.model_dump(mode="json", exclude=_RESULT_EXCLUDE)
        content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            self._require_lock()
            if overwrite:
                os.replace(temporary_path, self.path)
            else:
                os.link(temporary_path, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path, *, expected: CheckpointState | None = None
) -> CheckpointState:
    """Read a validated checkpoint under the same lock used by suite execution."""
    with CheckpointStore(path) as store:
        return store.load(expected)
