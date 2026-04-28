"""Raw layer persistence.

Path scheme (locked, Section 2.3):
    Season-level:  data/raw/<endpoint>/<season>/<season_type>.json
    Per-game:      data/raw/<endpoint>/<game_id>.json

Season-type slugs are lowercase snake_case ("Regular Season" -> "regular_season",
"PlayIn" -> "play_in"). Game IDs use nba_api's native 10-digit string form.

Writes are atomic: serialize to a sibling `.tmp` file, then os.replace into
place. That prevents a half-written file from ever being observable on disk
if the process is killed mid-write.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from nba_four_factors.config import RAW_DIR, Endpoint, SeasonType

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")


def season_type_slug(season_type: SeasonType) -> str:
    value = _CAMEL_BOUNDARY.sub("_", season_type.value)
    return value.replace(" ", "_").replace("-", "_").lower()


def raw_season_path(endpoint: Endpoint, season: str, season_type: SeasonType) -> Path:
    """Path for season-level endpoint pulls (currently only leaguegamelog)."""
    return RAW_DIR / endpoint.value / season / f"{season_type_slug(season_type)}.json"


def raw_game_path(endpoint: Endpoint, game_id: str) -> Path:
    """Path for per-game endpoint pulls (boxscore* endpoints)."""
    if not game_id or not game_id.isdigit():
        raise ValueError(f"game_id must be a digit string, got {game_id!r}")
    return RAW_DIR / endpoint.value / f"{game_id}.json"


def save_raw(path: Path, data: Any) -> None:
    """Write `data` as JSON to `path` atomically. Creates parents if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)


def load_raw(path: Path) -> Any:
    """Read JSON from `path`. Raises FileNotFoundError if absent, JSONDecodeError if malformed."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def exists_raw(path: Path) -> bool:
    """Convenience check; equivalent to path.is_file()."""
    return path.is_file()
