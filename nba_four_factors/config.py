"""Single source of truth for the NBA four-factors pipeline.

All constants that govern season coverage, storage layout, API behavior,
and endpoint/season-type enumeration live here. Changes to this module
are changes to the contract the rest of the pipeline relies on.
"""

from __future__ import annotations

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
# Scope is deliberately narrow: every member corresponds to an actual
# Phase 2 call site. New endpoints are added alongside the code that calls
# them, not in anticipation.


class Endpoint(StrEnum):
    SCHEDULE = "leaguegamelog"
    BOXSCORE_TRADITIONAL = "boxscoretraditionalv2"
    BOXSCORE_ADVANCED = "boxscoreadvancedv2"
    BOXSCORE_SUMMARY = "boxscoresummaryv2"
    # V3 endpoints. V2 box scores stop being published as of the 2025-26
    # season (Traditional and Summary are deprecated; Advanced V2 is not but
    # is held in V2 for consistency in pre-2025-26 backfill). V3 is used for
    # 2025-26 onward via endpoint_for_season(). Both versions carry the
    # four-factor inputs needed for downstream calculations; the processed
    # layer normalizes to a single canonical schema.
    BOXSCORE_TRADITIONAL_V3 = "boxscoretraditionalv3"
    BOXSCORE_ADVANCED_V3 = "boxscoreadvancedv3"
    BOXSCORE_SUMMARY_V3 = "boxscoresummaryv3"


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


# ---------------------------------------------------------------------------
# V2/V3 endpoint cutover
# ---------------------------------------------------------------------------
# stats.nba.com stopped publishing data via boxscoretraditionalv2 and
# boxscoresummaryv2 as of the 2025-26 season. V3 equivalents exist with
# different column naming (camelCase, expanded names) but carry the same
# four-factor inputs. We keep both versions wired and dispatch by season:
#
#     1997-98 .. 2024-25 -> V2 (proven historical coverage)
#     2025-26 ..         -> V3
#
# Advanced V2 is not formally deprecated, but we cut over with the others
# at 2025-26 to keep one boundary instead of three. Both raw schemas land
# on disk under their own endpoint subdirectory; the processed layer
# normalizes to a single canonical schema.

V3_CUTOVER_SEASON: str = "2025_26"

_V2_TO_V3: dict[Endpoint, Endpoint] = {
    Endpoint.BOXSCORE_TRADITIONAL: Endpoint.BOXSCORE_TRADITIONAL_V3,
    Endpoint.BOXSCORE_ADVANCED: Endpoint.BOXSCORE_ADVANCED_V3,
    Endpoint.BOXSCORE_SUMMARY: Endpoint.BOXSCORE_SUMMARY_V3,
}


def endpoint_for_season(endpoint: Endpoint, season: str) -> Endpoint:
    """Resolve a logical endpoint to its concrete V2 or V3 form for `season`.

    Pass the V2 enum member as the logical name; this function returns
    either the same V2 member (for seasons before V3_CUTOVER_SEASON) or
    the corresponding V3 member (for V3_CUTOVER_SEASON onward).

    SCHEDULE has no V2/V3 split and is returned unchanged. V3 enum members
    passed in are returned unchanged so callers that already know they
    want V3 can be explicit.
    """
    if endpoint not in _V2_TO_V3:
        return endpoint
    if season >= V3_CUTOVER_SEASON:
        return _V2_TO_V3[endpoint]
    return endpoint
