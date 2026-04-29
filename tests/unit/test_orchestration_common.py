"""Tests for ``nba_four_factors.orchestration._common``.

Pure-logic tests for target-set construction (§3.2, §3.3), the
schedule-aware done predicate (§7.1, §7.3), and small helpers.
"""

from __future__ import annotations

import pytest

from nba_four_factors.config import (
    PLAY_IN_FIRST_SEASON,
    SEASONS,
    Endpoint,
    SeasonType,
)
from nba_four_factors.orchestration._common import (
    BOX_SCORE_ENDPOINTS,
    SEASON_TYPE_ORDER,
    any_box_score_in,
    compute_targeted_set,
    is_pair_done,
    parse_season_range,
)

# --------------------------------------------------------------------------- #
# parse_season_range
# --------------------------------------------------------------------------- #


def test_parse_season_range_valid():
    assert parse_season_range("2010_11:2024_25") == ("2010_11", "2024_25")


def test_parse_season_range_strips_whitespace_around_colon():
    assert parse_season_range("  2010_11  :  2024_25  ") == ("2010_11", "2024_25")


def test_parse_season_range_missing_colon_raises():
    with pytest.raises(ValueError, match="YYYY_YY"):
        parse_season_range("2010_11")


# --------------------------------------------------------------------------- #
# compute_targeted_set — ordering (§3.2, §3.3)
# --------------------------------------------------------------------------- #


def test_default_is_newest_first():
    result = compute_targeted_set(
        season_range=None,
        season_types=[SeasonType.REGULAR],
        reverse=False,
    )
    seasons_in_result = [p[0] for p in result]
    # SEASONS is canonically oldest→newest; default ordering reverses that.
    assert seasons_in_result == list(reversed(SEASONS))


def test_reverse_flag_yields_chronological():
    result = compute_targeted_set(
        season_range=None,
        season_types=[SeasonType.REGULAR],
        reverse=True,
    )
    seasons_in_result = [p[0] for p in result]
    assert seasons_in_result == list(SEASONS)


def test_within_season_order_is_canonical():
    """Even if user passes types in a weird order, output respects
    regular → play_in → playoffs."""
    # Pick a season that supports play-in
    eligible = [s for s in SEASONS if s >= PLAY_IN_FIRST_SEASON]
    if not eligible:
        pytest.skip("no post-play-in seasons in SEASONS")
    season = eligible[0]
    result = compute_targeted_set(
        season_range=f"{season}:{season}",
        # Deliberately scrambled
        season_types=[SeasonType.PLAYOFFS, SeasonType.REGULAR, SeasonType.PLAY_IN],
        reverse=False,
    )
    types = [p[1] for p in result]
    assert types == list(SEASON_TYPE_ORDER)


# --------------------------------------------------------------------------- #
# compute_targeted_set — filtering and play-in skip
# --------------------------------------------------------------------------- #


def test_play_in_skipped_for_pre_play_in_seasons():
    pre = [s for s in SEASONS if s < PLAY_IN_FIRST_SEASON]
    if not pre:
        pytest.skip("no pre-play-in seasons in SEASONS")
    result = compute_targeted_set(
        season_range=f"{pre[0]}:{pre[0]}",
        season_types=[SeasonType.PLAY_IN],
        reverse=False,
    )
    assert result == []


def test_play_in_kept_for_post_play_in_seasons():
    post = [s for s in SEASONS if s >= PLAY_IN_FIRST_SEASON]
    if not post:
        pytest.skip("no post-play-in seasons in SEASONS")
    result = compute_targeted_set(
        season_range=f"{post[0]}:{post[0]}",
        season_types=[SeasonType.PLAY_IN],
        reverse=False,
    )
    assert result == [(post[0], SeasonType.PLAY_IN)]


def test_season_range_inclusive_on_both_ends():
    if len(SEASONS) < 3:
        pytest.skip("need ≥3 seasons for meaningful range test")
    start, end = SEASONS[1], SEASONS[-2]
    result = compute_targeted_set(
        season_range=f"{start}:{end}",
        season_types=[SeasonType.REGULAR],
        reverse=True,
    )
    seasons_in_result = [p[0] for p in result]
    assert seasons_in_result[0] == start
    assert seasons_in_result[-1] == end


def test_season_range_invalid_endpoint_raises():
    with pytest.raises(ValueError, match="must be in SEASONS"):
        compute_targeted_set(
            season_range="1900_01:2024_25",
            season_types=[SeasonType.REGULAR],
            reverse=False,
        )


def test_season_range_reversed_endpoints_still_works():
    """Passing end:start (chronologically reversed) should clamp correctly."""
    if len(SEASONS) < 2:
        pytest.skip("need ≥2 seasons")
    a, b = SEASONS[0], SEASONS[-1]
    result_forward = compute_targeted_set(
        season_range=f"{a}:{b}",
        season_types=[SeasonType.REGULAR],
        reverse=True,
    )
    result_swapped = compute_targeted_set(
        season_range=f"{b}:{a}",
        season_types=[SeasonType.REGULAR],
        reverse=True,
    )
    assert result_forward == result_swapped


# --------------------------------------------------------------------------- #
# any_box_score_in
# --------------------------------------------------------------------------- #


def test_any_box_score_in_with_box_score_endpoint():
    assert any_box_score_in([Endpoint.SCHEDULE, Endpoint.BOXSCORE_TRADITIONAL]) is True


def test_any_box_score_in_schedule_only():
    assert any_box_score_in([Endpoint.SCHEDULE]) is False


def test_any_box_score_in_empty():
    assert any_box_score_in([]) is False


# --------------------------------------------------------------------------- #
# is_pair_done — needs storage patches because of is_*_complete + load_*
# --------------------------------------------------------------------------- #


def _fresh_ckpt():
    return {
        "schedule": {"status": "pending"},
        "box_scores": {ep.value: {"completed": [], "failed": []} for ep in BOX_SCORE_ENDPOINTS},
    }


def test_is_pair_done_schedule_filter_complete(fake_storage):
    ckpt = _fresh_ckpt()
    ckpt["schedule"]["status"] = "complete"
    assert (
        is_pair_done(
            ckpt,
            "2024_25",
            SeasonType.REGULAR,
            endpoints=[Endpoint.SCHEDULE],
        )
        is True
    )


def test_is_pair_done_schedule_filter_pending(fake_storage):
    ckpt = _fresh_ckpt()
    assert (
        is_pair_done(
            ckpt,
            "2024_25",
            SeasonType.REGULAR,
            endpoints=[Endpoint.SCHEDULE],
        )
        is False
    )


def test_is_pair_done_box_score_request_with_incomplete_schedule(fake_storage):
    """Box-score stripes can't be complete without a schedule (§7.1)."""
    ckpt = _fresh_ckpt()
    assert (
        is_pair_done(
            ckpt,
            "2024_25",
            SeasonType.REGULAR,
            endpoints=[Endpoint.BOXSCORE_TRADITIONAL],
        )
        is False
    )


def test_is_pair_done_all_endpoints_complete(fake_storage):
    season, st = "2024_25", SeasonType.REGULAR
    fake_storage.game_ids_per_pair[(season, st)] = ["G1", "G2"]
    ckpt = _fresh_ckpt()
    ckpt["schedule"]["status"] = "complete"
    for e in BOX_SCORE_ENDPOINTS:
        ckpt["box_scores"][e.value]["completed"] = ["G1", "G2"]
    assert is_pair_done(ckpt, season, st, endpoints=list(Endpoint)) is True


def test_is_pair_done_lenient_treats_failed_games_as_complete(fake_storage):
    """§7.1 lenient: completed + failed counts toward done."""
    season, st = "2024_25", SeasonType.REGULAR
    fake_storage.game_ids_per_pair[(season, st)] = ["G1", "G2"]
    ckpt = _fresh_ckpt()
    ckpt["schedule"]["status"] = "complete"
    for e in BOX_SCORE_ENDPOINTS:
        ckpt["box_scores"][e.value]["completed"] = ["G1"]
        ckpt["box_scores"][e.value]["failed"] = ["G2"]
    assert is_pair_done(ckpt, season, st, endpoints=list(Endpoint)) is True


def test_is_pair_done_endpoint_filter_only_checks_targeted_endpoints(fake_storage):
    """§7.3: filter modulates which endpoints matter for done-status."""
    season, st = "2024_25", SeasonType.REGULAR
    fake_storage.game_ids_per_pair[(season, st)] = ["G1"]
    ckpt = _fresh_ckpt()
    ckpt["schedule"]["status"] = "complete"
    # Only traditional is complete; advanced and summary are untouched
    ckpt["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]["completed"] = ["G1"]

    assert (
        is_pair_done(
            ckpt,
            season,
            st,
            endpoints=[Endpoint.SCHEDULE, Endpoint.BOXSCORE_TRADITIONAL],
        )
        is True
    )
    assert (
        is_pair_done(
            ckpt,
            season,
            st,
            endpoints=[Endpoint.SCHEDULE, Endpoint.BOXSCORE_ADVANCED],
        )
        is False
    )
