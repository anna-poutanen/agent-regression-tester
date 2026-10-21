from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent_regression import checkpoint
from agent_regression.checkpoint import (
    CheckpointState,
    CheckpointStore,
    build_checkpoint,
    load_checkpoint,
    runtime_versions,
)
from agent_regression.schema import CheckFailure, OutputContains, RunResult, Scenario, ToolFixture


def scenarios() -> list[Scenario]:
    return [
        Scenario(
            name=name,
            prompt=f"Request for {name}",
            reviewed=True,
            checks=[OutputContains(text="done")],
        )
        for name in ("first", "second")
    ]


def plan(**kwargs: Any) -> CheckpointState:
    options: dict[str, Any] = {"run_id": "application-v1", "runtime_versions": {"test": "1"}}
    options.update(kwargs)
    return build_checkpoint(scenarios(), **options)


def write_payload(path: Path, state: CheckpointState, **changes: Any) -> None:
    payload = state.model_dump(mode="json", exclude={"results": {"__all__": {"passed"}}})
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_roundtrip_preserves_failed_and_passed_results_without_trusting_totals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "checkpoint.json"
    state = plan(repeats=2)
    with CheckpointStore(path) as store:
        state = store.create(state)
        state.results.append(
            RunResult(
                scenario_name="first",
                iteration=1,
                failures=[CheckFailure(code="forbidden_call", message="Unexpected refund")],
                replay_errors=["No saved fixture"],
                error="provider failed",
            )
        )
        store.save(state)
        state.results.append(RunResult(scenario_name="first", iteration=2, output="done"))
        state.stop_reason = "max_runs"
        store.save(state)
    loaded = load_checkpoint(path, expected=plan(repeats=2))
    assert [result.passed for result in loaded.results] == [False, True]
    assert "passed" not in json.loads(path.read_text())["results"][0]
    suite = loaded.to_suite()
    assert (suite.total_runs, suite.failed_runs, suite.passed_runs) == (2, 1, 1)
    assert suite.planned_runs == 4
    assert suite.remaining_runs == 2
    assert suite.status == "incomplete"
    assert suite.stop_reason == "max_runs"


def test_full_prefix_can_remain_incomplete_until_final_status_update(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as store:
        state = store.create(plan())
        state.results.extend([RunResult(scenario_name="first"), RunResult(scenario_name="second")])
        store.save(state)
        assert store.load().status == "incomplete"
        state.status = "completed"
        store.save(state)
    assert load_checkpoint(path).to_suite().status == "completed"


def test_new_creation_refuses_overwrite_and_retains_original(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as store:
        store.create(plan())
    before = path.read_bytes()
    with CheckpointStore(path) as store, pytest.raises(FileExistsError):
        store.create(plan(run_id="different"))
    assert path.read_bytes() == before


def test_atomic_replace_failure_preserves_last_committed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as store:
        state = store.create(plan())
        before = path.read_bytes()
        state.results.append(RunResult(scenario_name="first"))

        def fail_replace(source: Any, destination: Any) -> None:
            raise OSError("disk replacement failed")

        monkeypatch.setattr(checkpoint.os, "replace", fail_replace)
        with pytest.raises(OSError, match="disk replacement failed"):
            store.save(state)
        assert path.read_bytes() == before
        assert sorted(item.name for item in tmp_path.iterdir()) == [
            "checkpoint.json",
            "checkpoint.json.lock",
        ]
        monkeypatch.undo()
        store.save(state)
    assert len(load_checkpoint(path).results) == 1


def test_file_fsync_failure_does_not_publish_partial_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as store:
        state = store.create(plan())
        before = path.read_bytes()
        state.results.append(RunResult(scenario_name="first"))

        def fail_sync(descriptor: int) -> None:
            raise OSError("sync failed")

        monkeypatch.setattr(checkpoint.os, "fsync", fail_sync)
        with pytest.raises(OSError, match="sync failed"):
            store.save(state)
        assert path.read_bytes() == before


def test_lock_is_nonblocking_and_persistent_across_sessions(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as first:
        lock_inode = first.lock_path.stat().st_ino
        with pytest.raises(ValueError, match="already locked"), CheckpointStore(path):
            pass
        with pytest.raises(ValueError, match="already locked"):
            first.__enter__()
        first.create(plan())
    with CheckpointStore(path) as second:
        assert second.lock_path.stat().st_ino == lock_inode
        assert second.load().run_id == "application-v1"


def test_os_lock_prevents_an_independent_process_from_entering(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    code = (
        "import sys\n"
        "from agent_regression.checkpoint import CheckpointStore\n"
        "try:\n"
        "    with CheckpointStore(sys.argv[1]):\n"
        "        raise AssertionError('lock was bypassed')\n"
        "except ValueError as exc:\n"
        "    assert 'already locked' in str(exc)\n"
    )
    with CheckpointStore(path):
        child = subprocess.run(
            [sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=10
        )
    assert child.returncode == 0, child.stderr


def test_context_exception_releases_lock(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="cancelled"), CheckpointStore(path):
        raise RuntimeError("cancelled")
    with CheckpointStore(path) as store:
        store.create(plan())


def test_store_operations_require_a_live_lock(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint.json")
    for operation in (
        lambda: store.create(plan()),
        lambda: store.load(),
        lambda: store.save(plan()),
    ):
        with pytest.raises(ValueError, match="active context manager"):
            operation()


@pytest.mark.parametrize("entry", ["checkpoint", "lock"])
def test_symlink_checkpoint_and_lock_are_refused_without_touching_target(
    tmp_path: Path, entry: str
) -> None:
    path = tmp_path / "checkpoint.json"
    target = tmp_path / "original"
    target.write_text("keep original")
    selected = path if entry == "checkpoint" else path.with_name(path.name + ".lock")
    selected.symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"), CheckpointStore(path):
        pass
    assert target.read_text() == "keep original"


def test_parent_aliases_share_the_same_normalized_lock(tmp_path: Path) -> None:
    directory = tmp_path / "actual"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with CheckpointStore(directory / "checkpoint.json") as store:
        assert store.path == directory.resolve() / "checkpoint.json"
        with (
            pytest.raises(ValueError, match="already locked"),
            CheckpointStore(alias / "checkpoint.json"),
        ):
            pass


def test_lock_inode_replacement_is_detected_before_save(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as store:
        state = store.create(plan())
        before = path.read_bytes()
        store.lock_path.unlink()
        store.lock_path.touch()
        with pytest.raises(ValueError, match="lock file changed"):
            store.save(state)
        assert path.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "different"},
        {"repeats": 2},
        {"timeout": 60},
        {"allow_unreviewed": True},
        {"runtime_versions": {"test": "2"}},
    ],
)
def test_resume_rejects_plan_configuration_drift(tmp_path: Path, change: dict[str, Any]) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as store:
        store.create(plan())
    with pytest.raises(ValueError, match="plan mismatch"):
        load_checkpoint(path, expected=plan(**change))


@pytest.mark.parametrize("change", ["prompt", "order", "fixture", "check", "metadata", "review"])
def test_resume_binds_full_ordered_scenario_artifacts(tmp_path: Path, change: str) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as store:
        store.create(plan())
    changed = scenarios()
    if change == "prompt":
        changed[0].prompt = "Different request"
    elif change == "order":
        changed.reverse()
    elif change == "fixture":
        changed[0].fixtures.append(
            ToolFixture(tool_name="lookup", arguments={}, response="different")
        )
    elif change == "check":
        changed[0].checks.append(OutputContains(text="different"))
    elif change == "metadata":
        changed[0].metadata["version"] = "changed"
    else:
        changed[0].reviewed = False
    expected = build_checkpoint(changed, run_id="application-v1", runtime_versions={"test": "1"})
    with pytest.raises(ValueError, match="plan mismatch"):
        load_checkpoint(path, expected=expected)


def test_hash_ignores_object_key_order_but_keeps_boolean_numeric_difference() -> None:
    first = scenarios()
    second = scenarios()
    first[0].metadata = {"a": False, "b": 1}
    second[0].metadata = {"b": 1, "a": False}
    assert (
        build_checkpoint(first, run_id="x").scenarios_sha256
        == build_checkpoint(second, run_id="x").scenarios_sha256
    )
    second[0].metadata["a"] = 0
    assert (
        build_checkpoint(first, run_id="x").scenarios_sha256
        != build_checkpoint(second, run_id="x").scenarios_sha256
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"scenario_names": ["first", "first"]},
        {"scenario_names": []},
        {"repeats": 0},
        {"repeats": True},
        {"timeout": 0},
        {"allow_unreviewed": "false"},
        {"run_id": " "},
        {"runtime_versions": {"test": True}},
        {"runtime_versions": {}},
        {"unexpected": "field"},
        {"status": "completed"},
        {"results": [{"scenario_name": "second"}]},
        {"results": [{"scenario_name": "first", "iteration": 2}]},
        {"results": [{"scenario_name": "first", "passed": True}]},
        {"results": [{"scenario_name": "first", "duration_seconds": -1}]},
    ],
)
def test_strict_load_rejects_corrupt_identity_or_result_grid(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    path = tmp_path / "checkpoint.json"
    write_payload(path, plan(), **payload)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_duplicate_keys_and_nonfinite_json_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    write_payload(path, plan())
    original = path.read_text()
    path.write_text(
        original.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    )
    with pytest.raises(ValueError, match="Duplicate checkpoint JSON"):
        load_checkpoint(path)
    path.write_text(original.replace('"timeout": 30.0', '"timeout": NaN'))
    with pytest.raises(ValueError, match="Invalid checkpoint JSON constant"):
        load_checkpoint(path)


def test_empty_and_duplicate_name_plans_are_rejected() -> None:
    with pytest.raises(ValidationError):
        build_checkpoint([], run_id="x")
    with pytest.raises(ValidationError, match="unique names"):
        build_checkpoint([scenarios()[0], scenarios()[0]], run_id="x")


def test_runtime_version_identity_contains_no_model_inspection() -> None:
    versions = runtime_versions()
    assert set(versions) == {"python", "agent_regression", "pydantic_ai", "pydantic"}
    assert all(isinstance(value, str) and value for value in versions.values())


def test_save_revalidates_nested_results_and_prevents_identity_mutation(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as store:
        state = store.create(plan())
        state.runtime_versions["test"] = "changed"
        with pytest.raises(ValueError, match="identity cannot change"):
            store.save(state)
        state = store.load()
        state.results.append({"scenario_name": "first", "iteration": 3})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="result 0"):
            store.save(state)


def test_committed_failed_results_cannot_be_rewritten_or_removed(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    with CheckpointStore(path) as store:
        state = store.create(plan())
        state.results.append(RunResult(scenario_name="first", error="failed"))
        store.save(state)
        state.results[0].error = None
        with pytest.raises(ValueError, match="cannot be removed or rewritten"):
            store.save(state)
        state = store.load()
        state.results.clear()
        with pytest.raises(ValueError, match="cannot be removed or rewritten"):
            store.save(state)
