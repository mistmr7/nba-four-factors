"""Shared fixtures for orchestration unit tests.

The orchestration layer imports its dependencies at module level
(``from ..storage.checkpoint import save_checkpoint``, etc.).  The
:func:`fake_storage` fixture monkeypatches each of those imported names
inside the orchestration modules with methods on a single
:class:`FakeStorage` instance, so a test can configure scheduling /
failure behavior in one place and assert against one set of recorded
calls.

This is functionally equivalent to constructor-style DI (§9 of the spec)
without requiring a parameter-passing rewrite of the drivers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nba_four_factors.api.client import RetriesExhaustedError
from nba_four_factors.config import Endpoint, SeasonType

# Box-score endpoints in the locked stripe order (§3.1).  Duplicated here
# rather than imported from _common so this file has no orchestration-layer
# dependency at import time (avoids circular-import surprises in collection).
_BOX_SCORE_ENDPOINTS = (
    Endpoint.BOXSCORE_TRADITIONAL,
    Endpoint.BOXSCORE_ADVANCED,
    Endpoint.BOXSCORE_SUMMARY,
)


# --------------------------------------------------------------------------- #
# Helper: build checkpoint dicts in the real shape
# --------------------------------------------------------------------------- #


def make_ckpt(
    *,
    schedule_status: str = "pending",
    completed: dict[Endpoint, set[str] | list[str]] | None = None,
    failed: dict[Endpoint, set[str] | list[str]] | None = None,
) -> dict:
    """Build a checkpoint dict in the canonical on-disk shape.

    Tests should use this helper rather than constructing checkpoints
    inline — keeps the shape consistent with the real ``storage.checkpoint``
    module.  Pass per-endpoint sets/lists; missing endpoints default to empty.
    """
    completed = completed or {}
    failed = failed or {}
    return {
        "schedule": {"status": schedule_status},
        "box_scores": {
            ep.value: {
                "completed": list(completed.get(ep, [])),
                "failed": list(failed.get(ep, [])),
            }
            for ep in _BOX_SCORE_ENDPOINTS
        },
    }


# --------------------------------------------------------------------------- #
# FakeStorage
# --------------------------------------------------------------------------- #


class FakeStorage:
    """In-memory replacement for storage.checkpoint, storage.raw, and
    storage.manifest.  Tracks every interaction; supports failure injection.

    All public methods match the signatures the orchestration layer expects.
    """

    def __init__(self) -> None:
        # Recorded interactions
        self.saves: list[tuple[str, SeasonType, dict]] = []
        self.schedule_fetches: list[tuple[str, SeasonType]] = []
        self.game_fetches: list[tuple[Endpoint, str]] = []
        self.manifests_written: list[tuple[str, SeasonType]] = []

        # Failure injection
        self.fail_schedule_for: set[tuple[str, SeasonType]] = set()
        self.fail_game_ids: set[str] = set()

        # Schedule contents per (season, season_type)
        self.game_ids_per_pair: dict[tuple[str, SeasonType], list[str]] = {}

        # Pre-existing checkpoints (for testing resume / skip-already-done)
        self.preexisting_checkpoints: dict[tuple[str, SeasonType], dict] = {}

    # ----- storage.checkpoint -----
    #
    # Real checkpoint shape (locked Session 7):
    #   ckpt = {
    #     "season": str,
    #     "season_type": SeasonType,
    #     "schedule":   {"endpoint": str, "status": "pending"|"started"|"complete"|"failed", ...},
    #     "box_scores": {
    #       "<endpoint url string>": {"completed": list[str], "failed": list[str], ...}
    #     }
    #   }
    # The fake stores "completed"/"failed" as lists (not sets) to match the
    # on-disk shape exactly.  Helpers below convert to sets for set-style
    # operations and write back as lists.

    def load_checkpoint(self, season: str, season_type: SeasonType):
        return self.preexisting_checkpoints.get((season, season_type))

    def initialize_checkpoint(self, season: str, season_type: SeasonType) -> dict:
        return {
            "season": season,
            "season_type": season_type,
            "schedule": {"status": "pending"},
            "box_scores": {e.value: {"completed": [], "failed": []} for e in _BOX_SCORE_ENDPOINTS},
        }

    def save_checkpoint(self, season, season_type, ckpt) -> None:
        # Snapshot mutable state so later mutations don't retroactively
        # change what we recorded.
        self.saves.append(
            (
                season,
                season_type,
                {
                    "schedule": dict(ckpt["schedule"]),
                    "box_scores": {
                        ep_value: {
                            "completed": list(state["completed"]),
                            "failed": list(state["failed"]),
                        }
                        for ep_value, state in ckpt["box_scores"].items()
                    },
                },
            )
        )

    def mark_schedule_started(self, ckpt) -> None:
        ckpt["schedule"]["status"] = "started"

    def mark_schedule_complete(self, ckpt) -> None:
        ckpt["schedule"]["status"] = "complete"

    def mark_schedule_failed(self, ckpt) -> None:
        ckpt["schedule"]["status"] = "failed"

    def is_schedule_complete(self, ckpt) -> bool:
        return ckpt["schedule"]["status"] == "complete"

    def mark_game_complete(self, ckpt, endpoint, game_id) -> None:
        state = ckpt["box_scores"][endpoint.value]
        if game_id not in state["completed"]:
            state["completed"].append(game_id)
        if game_id in state["failed"]:
            state["failed"].remove(game_id)

    def mark_game_failed(self, ckpt, endpoint, game_id) -> None:
        state = ckpt["box_scores"][endpoint.value]
        if game_id not in state["failed"]:
            state["failed"].append(game_id)
        # Don't discard from completed here — real checkpoint preserves
        # successful state if a game later fails (it doesn't downgrade).
        # See test_storage_checkpoint::test_does_not_downgrade_completed.

    def is_endpoint_complete(self, ckpt, endpoint, expected_count) -> bool:
        # Lenient (§7.1): completed U failed covers every expected game.
        state = ckpt["box_scores"][endpoint.value]
        return len(state["completed"]) + len(state["failed"]) == expected_count

    def pending_games(self, ckpt, endpoint, all_game_ids):
        # "Failed are pending" per Session 7 lock — they re-fetch on next run.
        state = ckpt["box_scores"][endpoint.value]
        completed = set(state["completed"])
        return [g for g in all_game_ids if g not in completed]

    # ----- _io (storage-binding shim) -----

    def fetch_and_save_schedule(self, client, season, season_type) -> None:
        # ``client`` is ignored by the fake — its behavior is fully
        # determined by fail_schedule_for / schedule_fetches.
        if (season, season_type) in self.fail_schedule_for:
            raise RetriesExhaustedError(
                f"injected: schedule {season} {season_type}",
                last_reason="injected fake failure",
            )
        self.schedule_fetches.append((season, season_type))

    def fetch_and_save_game(self, client, endpoint, game_id) -> None:
        if game_id in self.fail_game_ids:
            raise RetriesExhaustedError(
                f"injected: game {endpoint} {game_id}",
                last_reason="injected fake failure",
            )
        self.game_fetches.append((endpoint, game_id))

    def load_schedule_game_ids(self, season, season_type):
        return list(self.game_ids_per_pair.get((season, season_type), []))

    def write_pair_manifest(self, season, season_type, ckpt) -> None:
        self.manifests_written.append((season, season_type))


# --------------------------------------------------------------------------- #
# Patch-target table
# --------------------------------------------------------------------------- #
#
# Each storage / _io symbol is imported into one or more orchestration
# modules.  monkeypatch.setattr replaces the imported reference inside each
# importing module — we have to patch every import site, not the source
# module, because Python `from X import Y` creates a local binding in the
# importing module.
#
# `_common` is special: it imports `load_schedule_game_ids` lazily inside
# `is_pair_done` (deferred import to break a circular dependency), so we
# patch it at the source — the `_io` module itself.

_PATCH_TARGETS = {
    "historical": "nba_four_factors.orchestration.historical",
    "incremental": "nba_four_factors.orchestration.incremental",
    "common": "nba_four_factors.orchestration._common",
    "io": "nba_four_factors.orchestration._io",
}

_ATTR_LOCATIONS: dict[str, list[str]] = {
    # storage.checkpoint — same as before
    "load_checkpoint": ["historical", "incremental"],
    "save_checkpoint": ["historical", "incremental"],
    "initialize_checkpoint": ["historical", "incremental"],
    "mark_schedule_started": ["historical"],
    "mark_schedule_complete": ["historical", "incremental"],
    "mark_schedule_failed": ["historical"],
    "mark_game_complete": ["historical", "incremental"],
    "mark_game_failed": ["historical", "incremental"],
    "is_schedule_complete": ["historical", "common"],
    "is_endpoint_complete": ["historical", "common"],
    "pending_games": ["historical", "incremental"],
    # _io shim
    # load_schedule_game_ids: _common imports it lazily from `_io`, so the
    # source-module patch ("io") covers _common automatically.
    "fetch_and_save_schedule": ["historical", "incremental"],
    "fetch_and_save_game": ["historical", "incremental"],
    "load_schedule_game_ids": ["historical", "incremental", "io"],
    "write_pair_manifest": ["historical"],
}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_storage(monkeypatch):
    """Patch every storage symbol the orchestration layer imports."""
    fs = FakeStorage()
    for attr, modules in _ATTR_LOCATIONS.items():
        for module_key in modules:
            monkeypatch.setattr(
                f"{_PATCH_TARGETS[module_key]}.{attr}",
                getattr(fs, attr),
            )
    return fs


@pytest.fixture
def make_backfill_args():
    """Factory for :class:`SimpleNamespace` mimicking argparse output."""

    def _make(**overrides):
        defaults = dict(
            season_range=None,
            reverse=False,
            season_type=list(SeasonType),
            endpoint=list(Endpoint),
            dry_run=False,
            max_consecutive_failures=10,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    return _make


@pytest.fixture
def make_incremental_args():
    """Factory for :class:`SimpleNamespace` mimicking incremental argparse output."""

    def _make(**overrides):
        defaults = dict(season=None, dry_run=False)
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    return _make
