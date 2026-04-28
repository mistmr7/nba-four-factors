"""Checkpoint storage for ingestion progress tracking.

Per-(season, season_type) JSON files under data/checkpoints/. Tracks
schedule fetch status (status enum) and per-game completion lists for
the three V3 box-score endpoints (completed/failed lists, no status —
completeness is derived).

I/O model: pure-functional mutators on in-memory checkpoint dicts;
callers control flush cadence via load_checkpoint / save_checkpoint.
Atomic write via sibling .tmp file + os.replace, mirroring
storage/raw.py and storage/manifest.py.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nba_four_factors.config import DATA_DIR, Endpoint, SeasonType
from nba_four_factors.storage.raw import season_type_slug

CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

BOX_SCORE_ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint.BOXSCORE_TRADITIONAL,
    Endpoint.BOXSCORE_ADVANCED,
    Endpoint.BOXSCORE_SUMMARY,
)

# Schedule status state machine.
# not_started -> in_progress -> {complete, failed}
# failed -> in_progress  (retry on next run)
# complete is terminal.
_VALID_SCHEDULE_TRANSITIONS: dict[str, frozenset[str]] = {
    "not_started": frozenset({"in_progress"}),
    "in_progress": frozenset({"complete", "failed"}),
    "complete": frozenset(),
    "failed": frozenset({"in_progress"}),
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    """Format a UTC datetime as ISO 8601 with trailing Z."""
    return dt.isoformat().replace("+00:00", "Z")


def checkpoint_path(season: str, season_type: SeasonType) -> Path:
    """Return the canonical checkpoint file path for (season, season_type)."""
    return CHECKPOINTS_DIR / f"{season}_{season_type_slug(season_type)}.json"


def initialize_checkpoint(
    season: str,
    season_type: SeasonType,
    *,
    now: Callable[[], datetime] = _utcnow,
) -> dict[str, Any]:
    """Return a fresh checkpoint dict for (season, season_type).

    Does not write to disk. Caller decides when to persist via save_checkpoint.
    The ``now`` injection is a no-op at init time (no timestamps are set on
    fresh checkpoints) but accepted for signature symmetry with mutators.
    """
    del now  # signature symmetry only; init sets no timestamps
    return {
        "season": season,
        "season_type": season_type_slug(season_type),
        "schedule": {
            "endpoint": Endpoint.SCHEDULE.value,
            "status": "not_started",
            "started_at": None,
            "completed_at": None,
        },
        "box_scores": {
            ep.value: {
                "completed": [],
                "failed": [],
                "started_at": None,
                "last_updated": None,
            }
            for ep in BOX_SCORE_ENDPOINTS
        },
    }


def load_checkpoint(season: str, season_type: SeasonType) -> dict[str, Any] | None:
    """Load checkpoint for (season, season_type), or None if file does not exist."""
    path = checkpoint_path(season, season_type)
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def save_checkpoint(season: str, season_type: SeasonType, data: dict[str, Any]) -> None:
    """Atomically write checkpoint to disk via sibling .tmp + os.replace."""
    path = checkpoint_path(season, season_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)


# Schedule mutators ---------------------------------------------------------


def _transition_schedule(
    checkpoint: dict[str, Any],
    target: str,
    *,
    now: Callable[[], datetime],
) -> None:
    current: str = checkpoint["schedule"]["status"]
    if target not in _VALID_SCHEDULE_TRANSITIONS[current]:
        raise ValueError(f"Invalid schedule transition {current!r} -> {target!r}")
    timestamp = _iso(now())
    checkpoint["schedule"]["status"] = target
    if target == "in_progress":
        checkpoint["schedule"]["started_at"] = timestamp
        # Reset completed_at when retrying after a prior failure.
        checkpoint["schedule"]["completed_at"] = None
    elif target in ("complete", "failed"):
        checkpoint["schedule"]["completed_at"] = timestamp


def mark_schedule_started(
    checkpoint: dict[str, Any], *, now: Callable[[], datetime] = _utcnow
) -> None:
    """Transition schedule to in_progress. Valid from not_started or failed."""
    _transition_schedule(checkpoint, "in_progress", now=now)


def mark_schedule_complete(
    checkpoint: dict[str, Any], *, now: Callable[[], datetime] = _utcnow
) -> None:
    """Transition schedule to complete. Valid from in_progress."""
    _transition_schedule(checkpoint, "complete", now=now)


def mark_schedule_failed(
    checkpoint: dict[str, Any], *, now: Callable[[], datetime] = _utcnow
) -> None:
    """Transition schedule to failed. Valid from in_progress."""
    _transition_schedule(checkpoint, "failed", now=now)


# Box-score mutators --------------------------------------------------------


def _ensure_box_score_endpoint(endpoint: Endpoint) -> None:
    if endpoint not in BOX_SCORE_ENDPOINTS:
        raise ValueError(f"Endpoint {endpoint!r} is not a box-score endpoint")


def _box_score_block(checkpoint: dict[str, Any], endpoint: Endpoint) -> dict[str, Any]:
    return checkpoint["box_scores"][endpoint.value]


def mark_game_complete(
    checkpoint: dict[str, Any],
    endpoint: Endpoint,
    game_id: str,
    *,
    now: Callable[[], datetime] = _utcnow,
) -> None:
    """Record a game as completed for the given box-score endpoint.

    Idempotent: re-marking an already-completed game is a no-op (does not
    update last_updated). If the game is currently in failed[], it is
    removed from failed[] and added to completed[] (retry success path).
    """
    _ensure_box_score_endpoint(endpoint)
    block = _box_score_block(checkpoint, endpoint)
    if game_id in block["completed"]:
        return
    timestamp = _iso(now())
    if game_id in block["failed"]:
        block["failed"].remove(game_id)
    block["completed"].append(game_id)
    if block["started_at"] is None:
        block["started_at"] = timestamp
    block["last_updated"] = timestamp


def mark_game_failed(
    checkpoint: dict[str, Any],
    endpoint: Endpoint,
    game_id: str,
    *,
    now: Callable[[], datetime] = _utcnow,
) -> None:
    """Record a game as failed for the given box-score endpoint.

    Idempotent: re-marking an already-failed game is a no-op. Does not
    downgrade a completed game to failed — if game_id is in completed[],
    this is a no-op as well.
    """
    _ensure_box_score_endpoint(endpoint)
    block = _box_score_block(checkpoint, endpoint)
    if game_id in block["failed"] or game_id in block["completed"]:
        return
    timestamp = _iso(now())
    block["failed"].append(game_id)
    if block["started_at"] is None:
        block["started_at"] = timestamp
    block["last_updated"] = timestamp


# Queries -------------------------------------------------------------------


def is_schedule_complete(checkpoint: dict[str, Any]) -> bool:
    """True if the schedule fetch has reached terminal complete state."""
    return checkpoint["schedule"]["status"] == "complete"


def pending_games(
    checkpoint: dict[str, Any],
    endpoint: Endpoint,
    all_game_ids: Iterable[str],
) -> list[str]:
    """Return games not yet completed for the given endpoint.

    Failed games ARE pending (retry semantics — see mark_game_complete).
    Order preserved from all_game_ids.
    """
    _ensure_box_score_endpoint(endpoint)
    block = _box_score_block(checkpoint, endpoint)
    completed: set[str] = set(block["completed"])
    return [gid for gid in all_game_ids if gid not in completed]


def is_endpoint_complete(
    checkpoint: dict[str, Any],
    endpoint: Endpoint,
    expected_count: int,
) -> bool:
    """True when completed + failed == expected_count.

    Lenient definition: "we tried everything." A non-empty failed[] does
    not block completeness; orchestrator surfaces failures via a separate
    has_failures check.
    """
    _ensure_box_score_endpoint(endpoint)
    block = _box_score_block(checkpoint, endpoint)
    return len(block["completed"]) + len(block["failed"]) == expected_count
