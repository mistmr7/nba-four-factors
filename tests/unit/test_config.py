"""Unit tests for config.py.

These tests lock in the contract: season coverage count, canonical
forms, path derivation, rate/retry constant sanity, and enum membership.
Changes here should be deliberate and reviewed.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from nba_four_factors import config
from nba_four_factors.config import (
    PLAY_IN_FIRST_SEASON,
    RATE_LIMIT_JITTER_SECONDS,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_WINDOW_SECONDS,
    RETRY_BACKOFF_BASE_SECONDS,
    RETRY_BACKOFF_FACTOR,
    RETRY_BACKOFF_MAX_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_STATUS_CODES,
    SEASONS,
    Endpoint,
    SeasonType,
    season_to_hyphen,
)

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------


class TestSeasons:
    def test_count(self):
        # 1997-98 through 2025-26 inclusive
        assert len(SEASONS) == 29

    def test_first_season(self):
        assert SEASONS[0] == "1997_98"

    def test_last_season(self):
        assert SEASONS[-1] == "2025_26"

    def test_all_canonical_form(self):
        # Underscore form, four-digit start, two-digit end
        for s in SEASONS:
            assert len(s) == 7
            assert s[4] == "_"
            assert s[:4].isdigit()
            assert s[5:].isdigit()

    def test_century_rollover_handled(self):
        # 1999_00 and 2000_01 both exist and are well-formed
        assert "1999_00" in SEASONS
        assert "2000_01" in SEASONS

    def test_sequential(self):
        for prev, curr in pairwise(SEASONS):
            prev_start = int(prev[:4])
            curr_start = int(curr[:4])
            assert curr_start == prev_start + 1

    def test_no_duplicates(self):
        assert len(set(SEASONS)) == len(SEASONS)

    def test_is_tuple(self):
        # Immutability matters: SEASONS is a contract
        assert isinstance(SEASONS, tuple)


class TestSeasonToHyphen:
    def test_basic_conversion(self):
        assert season_to_hyphen("1997_98") == "1997-98"
        assert season_to_hyphen("2025_26") == "2025-26"

    def test_century_rollover(self):
        assert season_to_hyphen("1999_00") == "1999-00"
        assert season_to_hyphen("2000_01") == "2000-01"

    def test_all_seasons_convert_cleanly(self):
        for s in SEASONS:
            converted = season_to_hyphen(s)
            assert "_" not in converted
            assert converted.count("-") == 1
            assert len(converted) == 7


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Contract here is path DERIVATION, not filesystem presence. Tests that
# asserted dir existence were dropped — data/ and logs/ are gitignored, so
# fresh clones and CI would fail those assertions without adding state the
# config module itself does not create.


class TestPaths:
    def test_repo_root_contains_package(self):
        # Sanity: REPO_ROOT should be the directory containing the
        # nba_four_factors package
        assert (config.REPO_ROOT / "nba_four_factors").is_dir()

    def test_repo_root_contains_pyproject(self):
        # Second sanity check: pyproject.toml lives at repo root
        assert (config.REPO_ROOT / "pyproject.toml").is_file()

    def test_data_dir_under_repo_root(self):
        assert config.DATA_DIR == config.REPO_ROOT / "data"

    def test_data_subdirs(self):
        assert config.RAW_DIR == config.DATA_DIR / "raw"
        assert config.PROCESSED_DIR == config.DATA_DIR / "processed"
        assert config.FEATURES_DIR == config.DATA_DIR / "features"

    def test_data_subdir_names(self):
        assert config.RAW_DIR.name == "raw"
        assert config.PROCESSED_DIR.name == "processed"
        assert config.FEATURES_DIR.name == "features"

    def test_manifests_dir(self):
        assert config.MANIFESTS_DIR == config.REPO_ROOT / "manifests"

    def test_logs_dir(self):
        assert config.LOGS_DIR == config.REPO_ROOT / "logs"

    def test_paths_are_path_objects(self):
        for path in (
            config.REPO_ROOT,
            config.DATA_DIR,
            config.RAW_DIR,
            config.PROCESSED_DIR,
            config.FEATURES_DIR,
            config.MANIFESTS_DIR,
            config.LOGS_DIR,
        ):
            assert isinstance(path, Path)


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_per_minute_positive(self):
        assert RATE_LIMIT_PER_MINUTE > 0

    def test_locked_value(self):
        # Locked Phase 2 decision: 30/min fixed
        assert RATE_LIMIT_PER_MINUTE == 30

    def test_window_is_60s(self):
        assert RATE_LIMIT_WINDOW_SECONDS == 60.0

    def test_jitter_non_negative(self):
        assert RATE_LIMIT_JITTER_SECONDS >= 0.0

    def test_jitter_reasonable(self):
        # Jitter should be a small fraction of the inter-request interval,
        # not so large it undermines the rate limit.
        interval = RATE_LIMIT_WINDOW_SECONDS / RATE_LIMIT_PER_MINUTE
        assert interval > RATE_LIMIT_JITTER_SECONDS


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class TestRetry:
    def test_locked_attempts(self):
        # Locked Phase 2 decision: 5 attempts
        assert RETRY_MAX_ATTEMPTS == 5

    def test_backoff_params_positive(self):
        assert RETRY_BACKOFF_BASE_SECONDS > 0
        assert RETRY_BACKOFF_FACTOR > 1.0
        assert RETRY_BACKOFF_MAX_SECONDS > RETRY_BACKOFF_BASE_SECONDS

    def test_status_codes_retryable_set(self):
        # Locked decision: retry on 429 / 5xx / timeouts only.
        # Timeouts are handled at the exception layer, not status codes.
        assert 429 in RETRY_STATUS_CODES
        assert 500 in RETRY_STATUS_CODES
        assert 502 in RETRY_STATUS_CODES
        assert 503 in RETRY_STATUS_CODES
        assert 504 in RETRY_STATUS_CODES

    def test_status_codes_non_retryable(self):
        # 4xx other than 429 must fail fast
        for code in (400, 401, 403, 404, 422):
            assert code not in RETRY_STATUS_CODES

    def test_status_codes_is_frozenset(self):
        # Immutability: retry policy is a contract
        assert isinstance(RETRY_STATUS_CODES, frozenset)


# ---------------------------------------------------------------------------
# Endpoint enum
# ---------------------------------------------------------------------------


class TestEndpoint:
    def test_is_str_enum(self):
        assert issubclass(Endpoint, str)

    def test_expected_members(self):
        names = {e.name for e in Endpoint}
        assert names == {
            "SCHEDULE",
            "BOXSCORE_TRADITIONAL",
            "BOXSCORE_ADVANCED",
            "BOXSCORE_SUMMARY",
        }

    def test_schedule_endpoint_value(self):
        # Locked decision: leaguegamelog, not leaguegamefinder
        assert Endpoint.SCHEDULE.value == "leaguegamelog"

    def test_boxscore_endpoint_values_pinned(self):
        # V3-only design (Session 5 pivot). Member names dropped the _V3
        # suffix, but value strings retain "v3" because the URL path
        # requires them. This test is the one place that lock is asserted
        # — without it, the v3-in-the-value invariant becomes invisible.
        assert Endpoint.BOXSCORE_TRADITIONAL.value == "boxscoretraditionalv3"
        assert Endpoint.BOXSCORE_ADVANCED.value == "boxscoreadvancedv3"
        assert Endpoint.BOXSCORE_SUMMARY.value == "boxscoresummaryv3"

    def test_values_are_filesystem_safe(self):
        # Enum value doubles as raw-layer subdir name; must be safe
        for e in Endpoint:
            assert "/" not in e.value
            assert " " not in e.value
            assert e.value == e.value.lower()


# ---------------------------------------------------------------------------
# SeasonType enum
# ---------------------------------------------------------------------------


class TestSeasonType:
    def test_is_str_enum(self):
        assert issubclass(SeasonType, str)

    def test_expected_members(self):
        names = {s.name for s in SeasonType}
        assert names == {"REGULAR", "PLAYOFFS", "PLAY_IN"}

    def test_values_match_nba_api(self):
        assert SeasonType.REGULAR.value == "Regular Season"
        assert SeasonType.PLAYOFFS.value == "Playoffs"
        assert SeasonType.PLAY_IN.value == "PlayIn"

    def test_play_in_first_season_in_coverage(self):
        assert PLAY_IN_FIRST_SEASON in SEASONS

    def test_play_in_first_season_is_2020_21(self):
        assert PLAY_IN_FIRST_SEASON == "2020_21"
