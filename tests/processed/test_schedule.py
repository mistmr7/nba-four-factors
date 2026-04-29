"""Tests for ``nba_four_factors.processed.schedule``.

All tests target ``_tidy_schedule`` directly with hand-crafted JSON dicts
from ``conftest.py`` — pure transformation, no disk.  ``read_schedule``
itself is a three-line I/O wrapper around storage primitives that already
have their own coverage; the value-add of separately testing it is low.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from nba_four_factors.config import SeasonType
from nba_four_factors.processed.schedule import _tidy_schedule

# --------------------------------------------------------------------------- #
# Shape and schema
# --------------------------------------------------------------------------- #


def test_tidy_schedule_row_count_matches_input(schedule_payload):
    df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    assert len(df) == 4  # 2 games x 2 teams


def test_tidy_schedule_has_expected_columns(schedule_payload):
    df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    expected = {
        "game_id",
        "game_date",
        "season",
        "season_type",
        "team_id",
        "team_abbr",
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
    }
    assert set(df.columns) == expected
    # _matchup and _separator scratch columns must be dropped
    assert "_matchup" not in df.columns
    assert "_separator" not in df.columns
    assert "MATCHUP" not in df.columns


def test_tidy_schedule_season_columns_populate_from_args(schedule_payload):
    df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.PLAY_IN,
    )
    assert (df["season"] == "2024_25").all()
    # PLAY_IN.value is "PlayIn" or "Play In" depending on enum source; the
    # snake conversion should land it consistently.
    assert df["season_type"].nunique() == 1
    val = df["season_type"].iloc[0]
    # Must be the snake form (lowercased, underscores).
    assert val == val.lower()
    assert " " not in val


def test_tidy_schedule_date_parsed_to_datetime(schedule_payload):
    df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    assert pd.api.types.is_datetime64_any_dtype(df["game_date"])
    # Original strings were "2024-10-22" and "2024-12-17"
    dates_str = df["game_date"].dt.strftime("%Y-%m-%d").tolist()
    assert "2024-10-22" in dates_str
    assert "2024-12-17" in dates_str


def test_tidy_schedule_box_columns_are_integers(schedule_payload):
    df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    for col in ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov", "pts"]:
        assert pd.api.types.is_integer_dtype(
            df[col]
        ), f"{col} should be integer dtype, got {df[col].dtype}"


def test_tidy_schedule_game_id_preserves_zero_padding(schedule_payload):
    """GAME_ID is a 10-char zero-padded string in the source; must not be
    coerced to int.  pandas can silently do this if the column happens to
    be all-numeric.

    pandas 2.x may use ``StringDtype`` or ``object``; either is fine —
    what matters is the values remain strings with leading zeros.
    """
    df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    assert not pd.api.types.is_integer_dtype(df["game_id"])
    assert isinstance(df["game_id"].iloc[0], str)
    assert df["game_id"].iloc[0].startswith("00224")
    assert len(df["game_id"].iloc[0]) == 10


# --------------------------------------------------------------------------- #
# is_home / is_neutral derivation
# --------------------------------------------------------------------------- #


def test_tidy_schedule_standard_game_is_home_set_correctly(schedule_payload):
    """Game 1: LAL hosts BOS (standard 'vs.' / '@' pair)."""
    df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    g1 = df[df["game_id"] == "0022400001"]
    lal = g1[g1["team_abbr"] == "LAL"].iloc[0]
    bos = g1[g1["team_abbr"] == "BOS"].iloc[0]

    assert lal["is_home"] is True or lal["is_home"] == True  # noqa: E712
    assert bos["is_home"] is False or bos["is_home"] == False  # noqa: E712
    assert not lal["is_neutral"]
    assert not bos["is_neutral"]


def test_tidy_schedule_neutral_game_marks_both_teams(schedule_payload):
    """Game 2: NBA Cup neutral site — both rows have '@'.  Both teams must
    be marked is_neutral=True, is_home=False."""
    df = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    g2 = df[df["game_id"] == "0022400002"]
    assert g2["is_neutral"].all()
    assert (~g2["is_home"]).all()


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_tidy_schedule_is_deterministic(schedule_payload):
    """Two runs produce byte-identical DataFrames (idempotency requirement
    from spec §4)."""
    df1 = _tidy_schedule(
        schedule_payload,
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    # Deep copy to ensure the second call doesn't see any mutations.
    df2 = _tidy_schedule(
        copy.deepcopy(schedule_payload),
        season="2024_25",
        season_type=SeasonType.REGULAR,
    )
    pd.testing.assert_frame_equal(df1, df2)


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


def test_tidy_schedule_missing_resultsets_raises():
    payload = {"resource": "leaguegamelog", "parameters": {}}
    with pytest.raises(ValueError, match="no resultSets"):
        _tidy_schedule(payload, season="2024_25", season_type=SeasonType.REGULAR)


def test_tidy_schedule_missing_required_column_raises(schedule_payload):
    """If a required column is missing from headers, fail loudly rather than
    producing all-NaN output."""
    payload = copy.deepcopy(schedule_payload)
    headers = payload["resultSets"][0]["headers"]
    rows = payload["resultSets"][0]["rowSet"]

    fgm_idx = headers.index("FGM")
    payload["resultSets"][0]["headers"] = [h for h in headers if h != "FGM"]
    payload["resultSets"][0]["rowSet"] = [
        [v for i, v in enumerate(r) if i != fgm_idx] for r in rows
    ]

    with pytest.raises(ValueError, match="missing required columns"):
        _tidy_schedule(payload, season="2024_25", season_type=SeasonType.REGULAR)


def test_tidy_schedule_game_appearing_once_raises(schedule_payload):
    """If a game has only one row (data corruption), self-join would
    silently drop it later — catch it at the tidy stage instead."""
    payload = copy.deepcopy(schedule_payload)
    # Drop one row of game 0022400001
    rows = payload["resultSets"][0]["rowSet"]
    headers = payload["resultSets"][0]["headers"]
    gid_idx = headers.index("GAME_ID")
    abbr_idx = headers.index("TEAM_ABBREVIATION")
    payload["resultSets"][0]["rowSet"] = [
        r for r in rows if not (r[gid_idx] == "0022400001" and r[abbr_idx] == "BOS")
    ]

    with pytest.raises(ValueError, match="do not appear exactly twice"):
        _tidy_schedule(payload, season="2024_25", season_type=SeasonType.REGULAR)


def test_tidy_schedule_unrecognised_matchup_pattern_raises(schedule_payload):
    """If both rows of a game have 'vs.' (no '@'), MATCHUP semantics are
    broken.  We've never seen this in real data, but we shouldn't silently
    accept it."""
    payload = copy.deepcopy(schedule_payload)
    rows = payload["resultSets"][0]["rowSet"]
    headers = payload["resultSets"][0]["headers"]
    matchup_idx = headers.index("MATCHUP")
    gid_idx = headers.index("GAME_ID")

    # Flip both rows of game 0022400001 to 'vs.' on both sides.
    for r in rows:
        if r[gid_idx] == "0022400001":
            parts = r[matchup_idx].split()
            r[matchup_idx] = f"{parts[0]} vs. {parts[2]}"

    with pytest.raises(ValueError, match="unrecognised MATCHUP patterns"):
        _tidy_schedule(payload, season="2024_25", season_type=SeasonType.REGULAR)
