"""Manifest persistence.

Manifests are committed to git (locked, Section 2.4) and live at:

    manifests/<season>_<season_type>.json

where `<season>` is underscore form (e.g. 2024_25) and `<season_type>` is
lowercase snake_case (e.g. regular_season). One manifest per
(season, season_type) combo; aggregates across all endpoints pulled for
that combo.

Schema enforcement is deliberately deferred. `_schema.json` is a
worked-example reference, not formal JSON Schema (locked, Session 1).
Reads and writes pass through a plain dict.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from nba_four_factors.config import MANIFESTS_DIR, SeasonType

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _season_type_slug(season_type: SeasonType) -> str:
    value = _CAMEL_BOUNDARY.sub("_", season_type.value)
    return value.replace(" ", "_").replace("-", "_").lower()


def manifest_path(season: str, season_type: SeasonType) -> Path:
    """Path for the manifest file for a given (season, season_type)."""
    return MANIFESTS_DIR / f"{season}_{_season_type_slug(season_type)}.json"


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    """Write manifest as pretty-printed JSON atomically. Creates parents if needed.

    Manifests are committed to git, so we indent for diff-friendliness.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict[str, Any]:
    """Read manifest JSON. Raises FileNotFoundError / JSONDecodeError on failure."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} must be a JSON object, got {type(data).__name__}")
    return data


def exists_manifest(path: Path) -> bool:
    return path.is_file()
