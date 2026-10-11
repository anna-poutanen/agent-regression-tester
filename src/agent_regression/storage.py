"""Validated, atomic scenario persistence with explicit overwrite protection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .schema import Scenario


def save_scenario(scenario: Scenario, path: str | Path, *, overwrite: bool = False) -> Path:
    """Save an artifact atomically; existing files require an explicit overwrite."""
    # Do not resolve a file symlink: overwrite applies to this directory entry,
    # rather than silently replacing a different file through the symlink.
    destination = Path(path).absolute()
    # Revalidate even after a caller has mutated an artifact's nested containers.
    validated = Scenario.model_validate(scenario.model_dump(mode="python", warnings=False))
    content = validated.model_dump_json(indent=2) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            # A hard link creates the final filename atomically and refuses races.
            os.link(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_scenario(path: str | Path) -> Scenario:
    """Read JSON, reject ambiguous duplicate keys, and validate schema version 1."""
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    return Scenario.model_validate(data)
