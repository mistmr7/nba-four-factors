"""Tests for ``nba_four_factors.orchestration.incremental``.

The incremental driver is intentionally narrow (§8.3): single season, all
endpoints, no circuit breaker, fast no-op-when-nothing-new (§1).  Tests
focus on those distinguishing behaviors plus the play-in skip rule.
"""

from __future__ import annotations

import pytest

from nba_four_factors.config import (
    PLAY_IN_FIRST_SEASON,
    SEASONS,
    Endpoint,
    SeasonType,
)
from nba_four_factors.orchestration.incremental import run_incremental

from .conftest import make_ckpt

# --------------------------------------------------------------------------- #
# Dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_makes_no_api_or_storage_calls(fake_storage, make_incremental_args, capsys):
    args = make_incremental_args(season="2024_25", dry_run=True)
    rc = run_incremental(args)

    assert rc == 0
    assert fake_storage.schedule_fetches == []
    assert fake_storage.game_fetches == []
    assert fake_storage.saves == []

    out = capsys.readouterr().out
    assert "DRY RUN" in out


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_pulls_pending_games_for_all_box_score_endpoints(fake_storage, make_incremental_args):
    season = next((s for s in SEASONS if s >= PLAY_IN_FIRST_SEASON), SEASONS[-1])
    fake_storage.game_ids_per_pair[(season, SeasonType.REGULAR)] = ["G1", "G2"]
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAY_IN)] = []
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAYOFFS)] = []
    args = make_incremental_args(season=season)

    rc = run_incremental(args)

    assert rc == 0
    # Schedule re-fetched for every applicable season_type
    assert len(fake_storage.schedule_fetches) == 3
    # 2 games x 3 box-score endpoints
    assert len(fake_storage.game_fetches) == 6


def test_resume_only_fetches_pending_games(fake_storage, make_incremental_args):
    """Games already in completed[] should not be re-fetched."""
    season = next((s for s in SEASONS if s >= PLAY_IN_FIRST_SEASON), SEASONS[-1])
    fake_storage.game_ids_per_pair[(season, SeasonType.REGULAR)] = ["G1", "G2"]
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAY_IN)] = []
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAYOFFS)] = []
    fake_storage.preexisting_checkpoints[(season, SeasonType.REGULAR)] = make_ckpt(
        schedule_status="complete",
        completed={
            Endpoint.BOXSCORE_TRADITIONAL: {"G1"},
            Endpoint.BOXSCORE_ADVANCED: {"G1"},
        },
    )
    args = make_incremental_args(season=season)
    run_incremental(args)

    # G1 should be fetched only for summary; G2 for all three.
    fetched = [(e, g) for e, g in fake_storage.game_fetches]
    assert (Endpoint.BOXSCORE_TRADITIONAL, "G1") not in fetched
    assert (Endpoint.BOXSCORE_ADVANCED, "G1") not in fetched
    assert (Endpoint.BOXSCORE_SUMMARY, "G1") in fetched
    assert (Endpoint.BOXSCORE_TRADITIONAL, "G2") in fetched


# --------------------------------------------------------------------------- #
# Play-in skip
# --------------------------------------------------------------------------- #


def test_play_in_skipped_for_pre_play_in_season(fake_storage, make_incremental_args):
    pre = [s for s in SEASONS if s < PLAY_IN_FIRST_SEASON]
    if not pre:
        pytest.skip("no pre-play-in seasons in SEASONS")
    season = pre[0]
    fake_storage.game_ids_per_pair[(season, SeasonType.REGULAR)] = []
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAYOFFS)] = []
    args = make_incremental_args(season=season)

    run_incremental(args)

    types_visited = {st for _, st in fake_storage.schedule_fetches}
    assert SeasonType.PLAY_IN not in types_visited
    assert SeasonType.REGULAR in types_visited
    assert SeasonType.PLAYOFFS in types_visited


# --------------------------------------------------------------------------- #
# Failure handling — no abort, no circuit breaker (§8.3)
# --------------------------------------------------------------------------- #


def test_schedule_failure_skips_to_next_season_type(fake_storage, make_incremental_args):
    season = next((s for s in SEASONS if s >= PLAY_IN_FIRST_SEASON), SEASONS[-1])
    fake_storage.fail_schedule_for.add((season, SeasonType.REGULAR))
    fake_storage.game_ids_per_pair[(season, SeasonType.REGULAR)] = ["G1"]
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAY_IN)] = ["G2"]
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAYOFFS)] = []
    args = make_incremental_args(season=season)
    rc = run_incremental(args)

    assert rc == 0  # incremental tolerates per-pair failures
    # No box-score for regular (schedule failed); play_in proceeds normally.
    fetches = list(fake_storage.game_fetches)
    assert (Endpoint.BOXSCORE_TRADITIONAL, "G1") not in fetches
    assert (Endpoint.BOXSCORE_TRADITIONAL, "G2") in fetches


def test_empty_schedule_is_fast_no_op(fake_storage, make_incremental_args):
    """Off-season / playoffs-not-yet-started: schedule has zero games."""
    season = next((s for s in SEASONS if s >= PLAY_IN_FIRST_SEASON), SEASONS[-1])
    for st in (SeasonType.REGULAR, SeasonType.PLAY_IN, SeasonType.PLAYOFFS):
        fake_storage.game_ids_per_pair[(season, st)] = []
    args = make_incremental_args(season=season)
    rc = run_incremental(args)

    assert rc == 0
    # Schedule re-fetch still happens (current-season schedule appears daily)
    assert len(fake_storage.schedule_fetches) == 3
    # But no games to pull
    assert fake_storage.game_fetches == []


def test_game_failure_marks_failed_and_continues(fake_storage, make_incremental_args):
    season = next((s for s in SEASONS if s >= PLAY_IN_FIRST_SEASON), SEASONS[-1])
    fake_storage.game_ids_per_pair[(season, SeasonType.REGULAR)] = [
        "G_BAD",
        "G_GOOD",
    ]
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAY_IN)] = []
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAYOFFS)] = []
    fake_storage.fail_game_ids.add("G_BAD")
    args = make_incremental_args(season=season)
    rc = run_incremental(args)

    assert rc == 0
    fetched_ids = [g for _, g in fake_storage.game_fetches]
    assert fetched_ids.count("G_GOOD") == 3  # all three box-score endpoints
    assert "G_BAD" not in fetched_ids


def test_no_circuit_breaker_in_incremental(fake_storage, make_incremental_args):
    """§8.3: no --max-consecutive-failures; many failures don't abort."""
    season = next((s for s in SEASONS if s >= PLAY_IN_FIRST_SEASON), SEASONS[-1])
    games = [f"G{i}" for i in range(50)]
    fake_storage.game_ids_per_pair[(season, SeasonType.REGULAR)] = games
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAY_IN)] = []
    fake_storage.game_ids_per_pair[(season, SeasonType.PLAYOFFS)] = []
    for g in games:
        fake_storage.fail_game_ids.add(g)
    args = make_incremental_args(season=season)
    rc = run_incremental(args)

    # Despite 50 games x 3 endpoints = 150 failures, the run completes.
    assert rc == 0


# --------------------------------------------------------------------------- #
# Default season fallback (§12.3 helper)
# --------------------------------------------------------------------------- #


def test_default_season_uses_current_season_helper(
    fake_storage, make_incremental_args, monkeypatch
):
    """When --season is None, run_incremental calls current_season()."""
    monkeypatch.setattr(
        "nba_four_factors.orchestration.incremental.current_season",
        lambda: "2025_26",
    )
    fake_storage.game_ids_per_pair[("2025_26", SeasonType.REGULAR)] = []
    fake_storage.game_ids_per_pair[("2025_26", SeasonType.PLAY_IN)] = []
    fake_storage.game_ids_per_pair[("2025_26", SeasonType.PLAYOFFS)] = []
    args = make_incremental_args(season=None)

    run_incremental(args)

    seasons_visited = {s for s, _ in fake_storage.schedule_fetches}
    assert seasons_visited == {"2025_26"}
