"""Single source of truth for the NBA four-factors pipeline.

All constants that govern season coverage, storage layout, API behavior,
and endpoint/season-type enumeration live here. Changes to this module
are changes to the contract the rest of the pipeline relies on.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
# REPO_ROOT is derived from this file's location: nba_four_factors/config.py
# sits one level below the repo root. All project paths derive from it so
# there is one anchor, not many.

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURES_DIR: Path = DATA_DIR / "features"

MANIFESTS_DIR: Path = REPO_ROOT / "manifests"
LOGS_DIR: Path = REPO_ROOT / "logs"


# ---------------------------------------------------------------------------
# Season coverage
# ---------------------------------------------------------------------------
# Canonical season identifiers use underscores for filesystem and code safety:
# "1997_98" internally, "1997-98" for display and external API calls
# (via season_to_hyphen). Covers 1997-98 through 2025-26 inclusive
# (29 seasons).

SEASONS: tuple[str, ...] = tuple(f"{start}_{str(start + 1)[-2:]}" for start in range(1997, 2026))


def season_to_hyphen(season: str) -> str:
    """Convert canonical '1997_98' to hyphenated '1997-98'.

    Used for both display and nba_api calls, which both expect the
    hyphenated form. Kept as a single function under DRY; if display
    and API formats ever diverge, split then.
    """
    return season.replace("_", "-")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# Fixed 30 requests/minute with jitter. Jitter is additive uniform noise on
# the inter-request delay to avoid synchronized bursts across retries.

RATE_LIMIT_PER_MINUTE: int = 30
RATE_LIMIT_WINDOW_SECONDS: float = 60.0
RATE_LIMIT_JITTER_SECONDS: float = 0.25


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
# 5 attempts total, exponential backoff on 429 / 5xx / timeouts only.
# Non-retryable errors (4xx other than 429, malformed responses) fail fast.
# Timeouts are handled at the exception layer, not via status codes.

RETRY_MAX_ATTEMPTS: int = 5
RETRY_BACKOFF_BASE_SECONDS: float = 1.0
RETRY_BACKOFF_FACTOR: float = 2.0
RETRY_BACKOFF_MAX_SECONDS: float = 60.0
RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
# Values are the nba_api endpoint module names, which double as the raw-layer
# subdirectory name under RAW_DIR. One string, two purposes, no mapping dict.
#
# Box-score endpoints are V3-only: empirical findings (Session 5) confirmed
# boxscoreadvancedv2 returns '{}' for ALL tested historical seasons (the
# endpoint is dead at the API level, not just deprecated for new data), and
# V3 reaches cleanly back to 1997-98 across traditional, advanced, and
# summary. The Python member names drop the V2/V3 distinction since only V3
# exists; the value strings retain "v3" because the URL path requires them.
#
# Scope is deliberately narrow: every member corresponds to an actual
# Phase 2 call site. New endpoints are added alongside the code that calls
# them, not in anticipation.


class Endpoint(StrEnum):
    SCHEDULE = "leaguegamelog"
    BOXSCORE_TRADITIONAL = "boxscoretraditionalv3"
    BOXSCORE_ADVANCED = "boxscoreadvancedv3"
    BOXSCORE_SUMMARY = "boxscoresummaryv3"
    PLAYER_AWARDS = "playerawards"


# ---------------------------------------------------------------------------
# Season types
# ---------------------------------------------------------------------------
# Values are the strings nba_api's SeasonType parameter expects.
# Play-In is kept as its own bucket at the ingest layer; the league's
# own treatment of play-in game statistics is inconsistent (sometimes
# regular season, sometimes playoffs), so we preserve the distinction
# in raw data and let the analysis layer decide how to bucket.
# Play-In is only valid 2020-21 onward.


class SeasonType(StrEnum):
    REGULAR = "Regular Season"
    PLAYOFFS = "Playoffs"
    PLAY_IN = "PlayIn"


PLAY_IN_FIRST_SEASON: str = "2020_21"


def _candidate_season(today: date | None = None) -> str:
    """The season that should be active by NBA calendar convention."""
    today = today or date.today()
    year = today.year
    start = year if today.month >= 10 else year - 1
    end_yy = (start + 1) % 100
    return f"{start}_{end_yy:02d}"


def current_season() -> str:
    """Season currently in progress, else previous one (§12.3).

    A season is "in progress" once at least one regular-season game has
    been scheduled and saved to disk.  Pure local-file lookup.
    """
    from nba_four_factors.storage.raw import (
        exists_raw,
        load_raw,
        raw_season_path,
    )

    candidate = _candidate_season()

    if candidate not in SEASONS:
        return SEASONS[-1]

    path = raw_season_path(Endpoint.SCHEDULE, candidate, SeasonType.REGULAR)
    if not exists_raw(path):
        idx = SEASONS.index(candidate)
        return SEASONS[idx - 1] if idx > 0 else candidate

    try:
        payload = load_raw(path)
        result_sets = payload.get("resultSets") or payload.get("resultSet") or []
        rows = result_sets[0]["rowSet"] if result_sets else []
        if rows:
            return candidate
    except Exception:
        pass

    idx = SEASONS.index(candidate)
    return SEASONS[idx - 1] if idx > 0 else candidate
