"""Tests for ``nba_four_factors.processed.pipeline``.

End-to-end coverage: monkeypatch ``read_schedule`` (so we don't need a
real raw JSON on disk) and ``_PROCESSED_ROOT`` (so output goes to
``tmp_path`` instead of polluting the working directory), then call
``process_pair`` and verify the Parquet round-trips correctly.

The factor numbers tested here are the same hand-computed values pinned
in :mod:`test_factors` — so if the pipeline composition differs from
``add_factors`` applied directly, this test catches the seam.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_four_factors.config import SeasonType
from nba_four_factors.processed import pipeline as pipeline_mod
from nba_four_factors.processed.pipeline import (
    _attach_opponent,
    process_pair,
    processed_path,
)
from nba_four_factors.processed.schedule import _tidy_schedule

# --------------------------------------------------------------------------- #
# Path construction
# --------------------------------------------------------------------------- #


def test_processed_path_uses_snake_season_type():
    p = processed_path("2024_25", SeasonType.REGULAR)
    assert p == Path("data/processed/2024_25/regular_season.parquet")


def test_processed_path_play_in():
    p = processed_path("2023_24", SeasonType.PLAY_IN)
    assert p == Path("data/processed/2023_24/play_in.parquet")


def test_processed_path_playoffs():
    p = processed_path("2024_25", SeasonType.PLAYOFFS)
    assert p == Path("data/processed/2024_25/playoffs.parquet")


# --------------------------------------------------------------------------- #
# _attach_opponent (self-join)
# --------------------------------------------------------------------------- #


def test_attach_opponent_preserves_row_count(schedule_payload):
    team_df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    joined = _attach_opponent(team_df)
    assert len(joined) == len(team_df)


def test_attach_opponent_pairs_correct_teams(schedule_payload):
    """Each row must be paired with the other team's row from the same game."""
    team_df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    joined = _attach_opponent(team_df)

    # For game 0022400001 (LAL vs BOS), LAL's row should have BOS as opponent.
    lal_row = joined[(joined["game_id"] == "0022400001") & (joined["team_abbr"] == "LAL")].iloc[0]
    assert lal_row["opp_abbr"] == "BOS"
    assert lal_row["opp_fgm"] == 38  # BOS's FGM from the fixture
    assert lal_row["opp_pts"] == 100


def test_attach_opponent_no_self_pairs(schedule_payload):
    """team_id must never equal opp_team_id."""
    team_df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    joined = _attach_opponent(team_df)
    assert (joined["team_id"] != joined["opp_team_id"]).all()


def test_attach_opponent_raises_on_corrupted_join():
    """If a game appears 3+ times, the self-join produces extra pairings.
    Synthetic case (we can't normally get here because _tidy_schedule
    catches it, but _attach_opponent's assertion is a defence in depth)."""
    # Build a 3-row "game" by hand, bypassing _tidy_schedule's validation.
    df = pd.DataFrame(
        [
            {
                "game_id": "G",
                "team_id": 1,
                "team_abbr": "A",
                "fgm": 1,
                "fga": 1,
                "fg3m": 0,
                "fg3a": 0,
                "ftm": 0,
                "fta": 0,
                "oreb": 0,
                "dreb": 0,
                "tov": 0,
                "pts": 1,
                "game_date": pd.Timestamp("2024-01-01"),
                "season": "2023_24",
                "season_type": "regular_season",
                "is_home": True,
                "is_neutral": False,
            },
            {
                "game_id": "G",
                "team_id": 2,
                "team_abbr": "B",
                "fgm": 1,
                "fga": 1,
                "fg3m": 0,
                "fg3a": 0,
                "ftm": 0,
                "fta": 0,
                "oreb": 0,
                "dreb": 0,
                "tov": 0,
                "pts": 1,
                "game_date": pd.Timestamp("2024-01-01"),
                "season": "2023_24",
                "season_type": "regular_season",
                "is_home": False,
                "is_neutral": False,
            },
            {
                "game_id": "G",
                "team_id": 3,
                "team_abbr": "C",
                "fgm": 1,
                "fga": 1,
                "fg3m": 0,
                "fg3a": 0,
                "ftm": 0,
                "fta": 0,
                "oreb": 0,
                "dreb": 0,
                "tov": 0,
                "pts": 1,
                "game_date": pd.Timestamp("2024-01-01"),
                "season": "2023_24",
                "season_type": "regular_season",
                "is_home": False,
                "is_neutral": False,
            },
        ]
    )
    with pytest.raises(AssertionError, match="Self-join produced"):
        _attach_opponent(df)


# --------------------------------------------------------------------------- #
# process_pair: end-to-end with monkeypatched I/O
# --------------------------------------------------------------------------- #


@pytest.fixture
def patched_pipeline(monkeypatch, tmp_path):
    """Patch the processed root to tmp_path and yield the patched module.

    The pipeline normally writes to ``data/processed/`` (cwd-relative).
    We can't have tests writing into the repo, so redirect to tmp_path.
    """
    monkeypatch.setattr(pipeline_mod, "_PROCESSED_ROOT", tmp_path / "processed")
    return tmp_path


def _patch_read_schedule(monkeypatch, schedule_payload):
    """Patch the name as imported INTO pipeline.py (not the source module).

    This is the standard trap: ``from .schedule import read_schedule``
    binds a local name in pipeline.py.  Monkeypatching
    ``schedule.read_schedule`` would not affect pipeline's reference.
    """

    def fake_read_schedule(season: str, season_type: SeasonType) -> pd.DataFrame:
        return _tidy_schedule(
            schedule_payload,
            season=season,
            season_type=season_type,
        )

    monkeypatch.setattr(pipeline_mod, "read_schedule", fake_read_schedule)


def test_process_pair_writes_parquet_at_expected_path(
    monkeypatch,
    patched_pipeline,
    schedule_payload,
):
    _patch_read_schedule(monkeypatch, schedule_payload)
    out_path = process_pair("2024_25", SeasonType.REGULAR)
    expected = patched_pipeline / "processed" / "2024_25" / "regular_season.parquet"
    assert out_path == expected
    assert out_path.exists()


def test_process_pair_round_trip_preserves_schema(
    monkeypatch,
    patched_pipeline,
    schedule_payload,
):
    """Schema, row count, and dtypes survive the Parquet write/read cycle."""
    _patch_read_schedule(monkeypatch, schedule_payload)
    out_path = process_pair("2024_25", SeasonType.REGULAR)

    df = pd.read_parquet(out_path)

    assert len(df) == 4  # 2 games x 2 teams
    expected_cols = {
        "game_id",
        "game_date",
        "season",
        "season_type",
        "team_id",
        "team_abbr",
        "opp_team_id",
        "opp_abbr",
        "is_home",
        "is_neutral",
        "fgm",
        "fga",
        "fg3m",
        "fg3a",
        "ftm",
        "fta",
        "oreb",
        "dreb",
        "tov",
        "pts",
        "opp_fgm",
        "opp_fga",
        "opp_fg3m",
        "opp_fg3a",
        "opp_ftm",
        "opp_fta",
        "opp_oreb",
        "opp_dreb",
        "opp_tov",
        "opp_pts",
        "off_efg_pct",
        "off_tov_pct",
        "off_orb_pct",
        "off_ft_rate",
        "def_efg_pct",
        "def_tov_pct",
        "def_orb_pct",
        "def_ft_rate",
        "margin",
    }
    assert set(df.columns) == expected_cols
    assert pd.api.types.is_datetime64_any_dtype(df["game_date"])
    assert pd.api.types.is_bool_dtype(df["is_home"])
    assert pd.api.types.is_bool_dtype(df["is_neutral"])


def test_process_pair_factor_values_match_hand_computed(
    monkeypatch,
    patched_pipeline,
    schedule_payload_factor_check,
):
    """Run the full pipeline on the §7.1 fixture; verify factor outputs
    match the pinned hand-computed values."""
    _patch_read_schedule(monkeypatch, schedule_payload_factor_check)
    out_path = process_pair("2024_25", SeasonType.REGULAR)
    df = pd.read_parquet(out_path)

    lal = df[df["team_abbr"] == "LAL"].iloc[0]
    assert lal["off_efg_pct"] == pytest.approx(0.5294117647058824)
    assert lal["off_tov_pct"] == pytest.approx(0.12987012987012989)
    assert lal["off_orb_pct"] == pytest.approx(0.26666666666666666)
    assert lal["off_ft_rate"] == pytest.approx(0.23529411764705882)
    assert lal["def_efg_pct"] == pytest.approx(0.4772727272727273)
    assert lal["def_tov_pct"] == pytest.approx(0.14295925661186562)
    assert lal["def_orb_pct"] == pytest.approx(0.2222222222222222)
    assert lal["def_ft_rate"] == pytest.approx(0.20454545454545456)
    assert lal["margin"] == 10


def test_process_pair_is_idempotent(
    monkeypatch,
    patched_pipeline,
    schedule_payload,
):
    """Running process_pair twice produces byte-identical Parquet content
    (verified via DataFrame equality after re-reading)."""
    _patch_read_schedule(monkeypatch, schedule_payload)

    out_path = process_pair("2024_25", SeasonType.REGULAR)
    df1 = pd.read_parquet(out_path)

    out_path = process_pair("2024_25", SeasonType.REGULAR)
    df2 = pd.read_parquet(out_path)

    pd.testing.assert_frame_equal(df1, df2)


def test_process_pair_creates_parent_directory(
    monkeypatch,
    patched_pipeline,
    schedule_payload,
):
    """The {season} subdirectory is created on demand (spec §5)."""
    _patch_read_schedule(monkeypatch, schedule_payload)
    out_path = process_pair("2024_25", SeasonType.REGULAR)
    # Parent didn't exist before the call; pipeline must mkdir.
    assert out_path.parent.is_dir()


def test_process_pair_overwrites_existing_file(
    monkeypatch,
    patched_pipeline,
    schedule_payload,
):
    """Idempotency requires overwriting (spec §5)."""
    _patch_read_schedule(monkeypatch, schedule_payload)
    out_path = process_pair("2024_25", SeasonType.REGULAR)

    # Sabotage the file
    out_path.write_bytes(b"corrupted")
    assert out_path.read_bytes() == b"corrupted"

    out_path = process_pair("2024_25", SeasonType.REGULAR)
    # Now it's a real Parquet again
    df = pd.read_parquet(out_path)
    assert len(df) == 4
