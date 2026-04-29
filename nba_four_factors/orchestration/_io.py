"""Storage-binding shim for the orchestration layer.

The orchestration policy (historical.py / incremental.py) speaks in terms of
"fetch and save the schedule for this pair" or "load the game IDs we already
have."  But the underlying layers expose narrower verbs:

* ``api/endpoints/*.py``      — ``fetch_*(client, ...)`` returns a parsed dict.
* ``storage/raw.py``          — ``raw_season_path``, ``raw_game_path``,
                                 ``save_raw``, ``load_raw``.
* ``storage/manifest.py``     — ``manifest_path``, ``save_manifest``.

This module composes those verbs into the four operations the policy layer
actually needs.  Keeping the composition here (rather than inlining it in
historical.py) means:

1. ``_common.is_pair_done`` and the drivers all call the same canonical
   "load schedule game IDs" routine — no drift.
2. Tests fake one well-defined surface (this module) instead of patching
   four storage / endpoint symbols across two driver modules.
3. If the storage layout changes, the blast radius is one file.

This module is package-internal (underscore-prefixed) per §9.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..api.client import Client
from ..api.endpoints.box_score_advanced import fetch_box_score_advanced
from ..api.endpoints.box_score_summary import fetch_box_score_summary
from ..api.endpoints.box_score_traditional import fetch_box_score_traditional
from ..api.endpoints.league_game_log import fetch_league_game_log
from ..config import Endpoint, SeasonType, season_to_hyphen
from ..storage.manifest import manifest_path, save_manifest
from ..storage.raw import (
    load_raw,
    raw_game_path,
    raw_season_path,
    save_raw,
)

# --------------------------------------------------------------------------- #
# Box-score endpoint dispatch
# --------------------------------------------------------------------------- #
#
# Three box-score endpoints, three near-identical signatures.  A small
# dispatch table keeps fetch_and_save_game endpoint-agnostic and matches the
# Endpoint enum's role as the canonical identifier across the codebase.
# Schedule is intentionally absent — fetch_and_save_schedule has a different
# call shape (takes season/season_type, not game_id) and is exposed as its
# own function.

_BOX_SCORE_FETCHERS = {
    Endpoint.BOXSCORE_TRADITIONAL: fetch_box_score_traditional,
    Endpoint.BOXSCORE_ADVANCED: fetch_box_score_advanced,
    Endpoint.BOXSCORE_SUMMARY: fetch_box_score_summary,
}


# --------------------------------------------------------------------------- #
# Fetch-and-save operations
# --------------------------------------------------------------------------- #


def fetch_and_save_schedule(
    client: Client,
    season: str,
    season_type: SeasonType,
) -> None:
    """Fetch the leaguegamelog for one (season, season_type) and persist it.

    Raises ``RetriesExhaustedError`` (from the api layer) on failure; the
    orchestration drivers catch this and record the failure in the
    checkpoint.  Does not catch other exceptions — schema-level surprises
    should bubble up loudly.
    """
    payload = fetch_league_game_log(client, season, season_type)
    path = raw_season_path(Endpoint.SCHEDULE, season, season_type)
    save_raw(path, payload)


def fetch_and_save_game(
    client: Client,
    endpoint: Endpoint,
    game_id: str,
) -> None:
    """Fetch one box-score endpoint for one game and persist it.

    Dispatches via :data:`_BOX_SCORE_FETCHERS`.  Passing a non-box-score
    Endpoint (e.g. ``Endpoint.SCHEDULE``) raises ``KeyError`` — that's a
    programming error in the orchestration layer, not a runtime condition,
    so a loud failure is correct.
    """
    fetcher = _BOX_SCORE_FETCHERS[endpoint]
    payload = fetcher(client, game_id)
    path = raw_game_path(endpoint, game_id)
    save_raw(path, payload)


# --------------------------------------------------------------------------- #
# Schedule introspection
# --------------------------------------------------------------------------- #


def load_schedule_game_ids(season: str, season_type: SeasonType) -> list[str]:
    """Return the deduplicated, sorted list of game IDs for a saved schedule.

    leaguegamelog returns one row per (game, team), so each game appears
    twice in ``rowSet``.  We dedupe to return distinct game IDs.

    The response shape is the standard nba_api stats-API format::

        {
          "resultSets": [
            {
              "name": "LeagueGameLog",
              "headers": ["SEASON_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", ...],
              "rowSet": [[...], [...], ...]
            }
          ],
          ...
        }

    If the response shape changes (e.g. NBA pivots leaguegamelog to V3
    with a hierarchical envelope), this is the one place that needs updating.
    """
    path = raw_season_path(Endpoint.SCHEDULE, season, season_type)
    payload = load_raw(path)

    result_sets = payload.get("resultSets") or payload.get("resultSet")
    if not result_sets:
        raise ValueError(
            f"schedule payload for {season} {season_type.name.lower()} "
            f"has no resultSets; got top-level keys: {list(payload.keys())}"
        )
    table = result_sets[0]
    headers = table["headers"]
    rows = table["rowSet"]
    try:
        game_id_idx = headers.index("GAME_ID")
    except ValueError as e:
        raise ValueError(f"GAME_ID column not in schedule headers: {headers}") from e

    # set + sort gives stable ordering across runs, which makes checkpoint
    # diffs and pending_games iteration deterministic.
    return sorted({row[game_id_idx] for row in rows})


# --------------------------------------------------------------------------- #
# Manifest writing
# --------------------------------------------------------------------------- #


def write_pair_manifest(
    season: str,
    season_type: SeasonType,
    ckpt: dict[str, Any],
) -> None:
    """Write the end-of-pair manifest using the minimal schema (option a).

    Computed from checkpoint state and the schedule on disk; no row-counting
    or version-discovery (deferred — see manifests/_schema.json for the full
    target schema).  ``pulled_at`` is the time this function is called, which
    matches the spec's "manifest written at end of pair" semantic.

    Field choices:

    * ``season`` / ``season_type`` — hyphen / display form, matching the
      schema example (``"2024-25"``, ``"Regular Season"``).
    * ``game_ids`` — every game ID from the saved schedule.
    * ``expected_game_count`` — same length, redundant with the list but
      cheap to include and matches the schema.
    * ``actual_game_count`` — successful completions across all three
      box-score endpoints (intersection: a game counts as "actual" only if
      every requested endpoint succeeded for it).
    * ``missing_or_failed`` — game IDs in the schedule but not in the
      successful intersection; these are the ones that need attention.
    """
    game_ids = load_schedule_game_ids(season, season_type)

    # Real checkpoint shape (locked Session 7):
    #   ckpt["box_scores"][<endpoint url string>]["completed"] -> list[str]
    #   ckpt["box_scores"][<endpoint url string>]["failed"]    -> list[str]
    # Endpoint enum values ARE those URL strings (StrEnum), so endpoint.value
    # is the right key.  The fakes in tests/unit/conftest.py used a different
    # shape — that mismatch is what hid this bug.
    box_scores = ckpt.get("box_scores", {})
    box_score_endpoints = [
        Endpoint.BOXSCORE_TRADITIONAL,
        Endpoint.BOXSCORE_ADVANCED,
        Endpoint.BOXSCORE_SUMMARY,
    ]

    def _completed(ep: Endpoint) -> set[str]:
        return set(box_scores.get(ep.value, {}).get("completed", []))

    # A game is "actual" only if all three box-score endpoints succeeded.
    # Intersection is correct here; a game with traditional+advanced but
    # missing summary isn't fully ingested.
    successful = set.intersection(*(_completed(ep) for ep in box_score_endpoints))

    missing_or_failed = sorted(set(game_ids) - successful)

    payload: dict[str, Any] = {
        "season": season_to_hyphen(season),
        "season_type": season_type.value,
        "pulled_at": _utc_now_iso(),
        "expected_game_count": len(game_ids),
        "actual_game_count": len(successful),
        "game_ids": game_ids,
        "missing_or_failed": missing_or_failed,
    }

    save_manifest(manifest_path(season, season_type), payload)


def _utc_now_iso() -> str:
    """ISO-8601 UTC with trailing ``Z``, matching the manifest schema example."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
