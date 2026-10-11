from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_regression.schema import Scenario
from agent_regression.storage import load_scenario, save_scenario


def test_roundtrip_creates_parent_directory_and_preserves_unicode(tmp_path: Path) -> None:
    scenario = Scenario(name="å-order", prompt="Find customer's 📦", metadata={"limit": None})
    path = tmp_path / "scenarios" / "order.json"
    assert save_scenario(scenario, path) == path
    assert load_scenario(path) == scenario
    assert path.read_text().endswith("\n")


def test_overwrite_requires_explicit_permission_and_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "order.json"
    original = Scenario(name="first", prompt="request")
    replacement = Scenario(name="second", prompt="request")
    save_scenario(original, path)
    with pytest.raises(FileExistsError):
        save_scenario(replacement, path)
    assert load_scenario(path) == original
    save_scenario(replacement, path, overwrite=True)
    assert load_scenario(path) == replacement
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "text",
    [
        '{"name":"first", "name":"second", "prompt":"request"}',
        '{"schema_version":2, "name":"test", "prompt":"request"}',
        '{"name":"test", "prompt":"request", "metadata":{"bad":NaN}}',
        "not JSON",
        "[]",
    ],
)
def test_invalid_or_ambiguous_files_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_scenario(path)


def test_nested_mutation_cannot_persist_invalid_values(tmp_path: Path) -> None:
    scenario = Scenario(name="test", prompt="request")
    scenario.metadata["not_json"] = object()  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        save_scenario(scenario, tmp_path / "invalid.json")
    assert not list(tmp_path.iterdir())


def test_failure_to_save_preserves_previous_artifact(tmp_path: Path) -> None:
    path = tmp_path / "saved.json"
    scenario = Scenario(name="test", prompt="request")
    save_scenario(scenario, path)
    before = path.read_bytes()
    scenario.metadata["bad"] = float("nan")
    with pytest.raises(ValidationError):
        save_scenario(scenario, path, overwrite=True)
    assert path.read_bytes() == before


def test_symlink_never_overwrites_its_target(tmp_path: Path) -> None:
    target = tmp_path / "original.json"
    target.write_text("Keep this file")
    symlink = tmp_path / "scenario.json"
    symlink.symlink_to(target)
    scenario = Scenario(name="test", prompt="request")
    with pytest.raises(FileExistsError):
        save_scenario(scenario, symlink)
    assert target.read_text() == "Keep this file"
    save_scenario(scenario, symlink, overwrite=True)
    assert not symlink.is_symlink()
    assert load_scenario(symlink) == scenario
    assert target.read_text() == "Keep this file"
