"""Tests for ``nba_four_factors.orchestration.historical``.

Exercises the backfill driver end-to-end with the :func:`fake_storage`
fixture standing in for storage / raw / manifest.  Every successful
mutation is expected to flush (§4); failures bubble through the
state machine without aborting the run unless the circuit breaker
trips (§5.4).
"""

from __future__ import annotations

from nba_four_factors.config import Endpoint, SeasonType
from nba_four_factors.orchestration._common import BOX_SCORE_ENDPOINTS
from nba_four_factors.orchestration.historical import run_backfill

from .conftest import make_ckpt

# A canonical small target for most tests: one season, one season_type,
# scoped via --season-range so we don't iterate the full SEASONS list.
SEASON = "2024_25"
ST = SeasonType.REGULAR


# --------------------------------------------------------------------------- #
# Dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_makes_no_api_or_storage_calls(fake_storage, make_backfill_args, capsys):
    args = make_backfill_args(dry_run=True)
    rc = run_backfill(args)

    assert rc == 0
    assert fake_storage.schedule_fetches == []
    assert fake_storage.game_fetches == []
    assert fake_storage.saves == []
    assert fake_storage.manifests_written == []

    out = capsys.readouterr().out
    assert "DRY RUN" in out


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_happy_path_one_pair_all_four_endpoints(fake_storage, make_backfill_args):
    fake_storage.game_ids_per_pair[(SEASON, ST)] = ["G1", "G2"]
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
    )
    rc = run_backfill(args)

    assert rc == 0
    # Schedule fetched once
    assert fake_storage.schedule_fetches == [(SEASON, ST)]
    # 2 games X 3 box-score endpoints
    assert len(fake_storage.game_fetches) == 6
    # Stripe order: traditional → advanced → summary (§3.1)
    endpoints_seen = [e for e, _ in fake_storage.game_fetches]
    assert endpoints_seen == (
        [Endpoint.BOXSCORE_TRADITIONAL] * 2
        + [Endpoint.BOXSCORE_ADVANCED] * 2
        + [Endpoint.BOXSCORE_SUMMARY] * 2
    )
    # Manifest at end-of-pair
    assert fake_storage.manifests_written == [(SEASON, ST)]


def test_per_mutation_flush_count(fake_storage, make_backfill_args):
    """§4: every successful checkpoint mutation triggers a save."""
    fake_storage.game_ids_per_pair[(SEASON, ST)] = ["G1", "G2"]
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
    )
    run_backfill(args)

    # Saves: schedule_started + schedule_complete + 6 game completions = 8
    assert len(fake_storage.saves) == 8
    # First save reflects schedule "started", second reflects "complete"
    assert fake_storage.saves[0][2]["schedule"]["status"] == "started"
    assert fake_storage.saves[1][2]["schedule"]["status"] == "complete"


# --------------------------------------------------------------------------- #
# Skip-already-done (§7.3)
# --------------------------------------------------------------------------- #


def test_skip_already_done_pair_makes_no_api_calls(fake_storage, make_backfill_args):
    fake_storage.game_ids_per_pair[(SEASON, ST)] = ["G1"]
    fake_storage.preexisting_checkpoints[(SEASON, ST)] = make_ckpt(
        schedule_status="complete",
        completed={e: {"G1"} for e in BOX_SCORE_ENDPOINTS},
    )
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
    )
    rc = run_backfill(args)

    assert rc == 0
    assert fake_storage.schedule_fetches == []
    assert fake_storage.game_fetches == []
    assert fake_storage.manifests_written == []


def test_resume_partial_pair_skips_complete_stripes(fake_storage, make_backfill_args):
    """If traditional was already complete in a prior run, don't re-fetch it."""
    fake_storage.game_ids_per_pair[(SEASON, ST)] = ["G1"]
    fake_storage.preexisting_checkpoints[(SEASON, ST)] = make_ckpt(
        schedule_status="complete",
        completed={Endpoint.BOXSCORE_TRADITIONAL: {"G1"}},  # only traditional done
    )
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
    )
    run_backfill(args)

    fetched_endpoints = {e for e, _ in fake_storage.game_fetches}
    assert Endpoint.BOXSCORE_TRADITIONAL not in fetched_endpoints
    assert Endpoint.BOXSCORE_ADVANCED in fetched_endpoints
    assert Endpoint.BOXSCORE_SUMMARY in fetched_endpoints


# --------------------------------------------------------------------------- #
# Failure handling (§5)
# --------------------------------------------------------------------------- #


def test_schedule_failure_skips_box_score_stripes(fake_storage, make_backfill_args):
    """§5.3: schedule failure aborts the current pair; box-score can't run."""
    fake_storage.fail_schedule_for.add((SEASON, ST))
    fake_storage.game_ids_per_pair[(SEASON, ST)] = ["G1", "G2"]
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
    )
    rc = run_backfill(args)

    assert rc == 0  # one schedule failure does not trip the default breaker
    assert fake_storage.game_fetches == []
    # Schedule attempt was rejected before recording → no entry in
    # schedule_fetches, but checkpoint records started + failed (2 saves)
    assert fake_storage.schedule_fetches == []
    assert len(fake_storage.saves) == 2
    assert fake_storage.saves[0][2]["schedule"]["status"] == "started"
    assert fake_storage.saves[1][2]["schedule"]["status"] == "failed"
    # No manifest for a pair whose schedule never landed
    assert fake_storage.manifests_written == []


def test_game_failure_marks_failed_and_continues(fake_storage, make_backfill_args):
    """§5.2: per-game failure doesn't abort the stripe."""
    fake_storage.game_ids_per_pair[(SEASON, ST)] = ["G_BAD", "G_GOOD"]
    fake_storage.fail_game_ids.add("G_BAD")
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
        endpoint=[Endpoint.SCHEDULE, Endpoint.BOXSCORE_TRADITIONAL],
    )
    rc = run_backfill(args)

    assert rc == 0
    # G_GOOD fetched, G_BAD did not
    fetched_ids = [g for _, g in fake_storage.game_fetches]
    assert "G_GOOD" in fetched_ids
    assert "G_BAD" not in fetched_ids
    # Final checkpoint has G_BAD in failed[traditional], G_GOOD in completed
    final = fake_storage.saves[-1][2]
    assert "G_BAD" in final["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]["failed"]
    assert "G_GOOD" in final["box_scores"][Endpoint.BOXSCORE_TRADITIONAL.value]["completed"]
    # Manifest still written despite the failure (lenient completeness)
    assert (SEASON, ST) in fake_storage.manifests_written


# --------------------------------------------------------------------------- #
# Circuit breaker (§5.4)
# --------------------------------------------------------------------------- #


def test_circuit_breaker_trips_at_threshold(fake_storage, make_backfill_args):
    fake_storage.game_ids_per_pair[(SEASON, ST)] = [f"G{i}" for i in range(20)]
    for i in range(20):
        fake_storage.fail_game_ids.add(f"G{i}")
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
        max_consecutive_failures=3,
    )
    rc = run_backfill(args)

    assert rc == 1
    assert fake_storage.game_fetches == []  # all 3 attempts failed
    # 2 schedule saves + 3 failed-game saves + 1 paranoid post-trip flush
    # (§5.4 step 2) = 6
    assert len(fake_storage.saves) == 6
    # No manifest for an aborted pair
    assert fake_storage.manifests_written == []


def test_circuit_breaker_resets_on_success(fake_storage, make_backfill_args):
    """3 fails → 1 success → 3 fails should NOT trip a threshold of 5."""
    games = ["B1", "B2", "B3", "OK", "B4", "B5", "B6"]
    fake_storage.game_ids_per_pair[(SEASON, ST)] = games
    for g in ("B1", "B2", "B3", "B4", "B5", "B6"):
        fake_storage.fail_game_ids.add(g)
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
        endpoint=[Endpoint.SCHEDULE, Endpoint.BOXSCORE_TRADITIONAL],
        max_consecutive_failures=5,
    )
    rc = run_backfill(args)

    # Counter goes 1,2,3, reset(0), 1,2,3.  Threshold 5 never reached.
    assert rc == 0


def test_circuit_breaker_default_threshold_is_ten(fake_storage, make_backfill_args):
    """Default threshold from CLI is 10; verify that's actually consumed."""
    games = [f"G{i}" for i in range(15)]
    fake_storage.game_ids_per_pair[(SEASON, ST)] = games
    for g in games:
        fake_storage.fail_game_ids.add(g)
    # Use the default threshold (10)
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
    )
    rc = run_backfill(args)

    assert rc == 1
    # 2 schedule saves + 10 failed-game saves + 1 paranoid post-trip flush = 13
    assert len(fake_storage.saves) == 13


# --------------------------------------------------------------------------- #
# §8.4 strict-reject
# --------------------------------------------------------------------------- #


def test_strict_reject_when_schedule_incomplete_and_box_score_only(
    fake_storage, make_backfill_args, capsys
):
    """§8.4: --endpoint=traditional with schedule incomplete → reject, exit 2."""
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
        endpoint=[Endpoint.BOXSCORE_TRADITIONAL],  # box-score-only
    )
    rc = run_backfill(args)

    assert rc == 2
    err = capsys.readouterr().err
    assert "schedule must be complete" in err
    assert SEASON in err
    # No fetches happen
    assert fake_storage.schedule_fetches == []
    assert fake_storage.game_fetches == []


def test_box_score_only_is_allowed_when_schedule_already_complete(fake_storage, make_backfill_args):
    """§8.4 only rejects when schedule is incomplete."""
    fake_storage.game_ids_per_pair[(SEASON, ST)] = ["G1"]
    fake_storage.preexisting_checkpoints[(SEASON, ST)] = make_ckpt(
        schedule_status="complete",
    )
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
        endpoint=[Endpoint.BOXSCORE_TRADITIONAL],
    )
    rc = run_backfill(args)

    assert rc == 0
    # Schedule not re-fetched (not in --endpoint), but traditional is
    assert fake_storage.schedule_fetches == []
    assert any(e == Endpoint.BOXSCORE_TRADITIONAL for e, _ in fake_storage.game_fetches)


# --------------------------------------------------------------------------- #
# Schedule-only run
# --------------------------------------------------------------------------- #


def test_schedule_only_run_writes_manifest_and_skips_box_score(fake_storage, make_backfill_args):
    fake_storage.game_ids_per_pair[(SEASON, ST)] = ["G1", "G2"]
    args = make_backfill_args(
        season_range=f"{SEASON}:{SEASON}",
        season_type=[ST],
        endpoint=[Endpoint.SCHEDULE],
    )
    rc = run_backfill(args)

    assert rc == 0
    assert fake_storage.schedule_fetches == [(SEASON, ST)]
    assert fake_storage.game_fetches == []
    assert fake_storage.manifests_written == [(SEASON, ST)]
