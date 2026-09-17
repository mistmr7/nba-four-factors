"""Tests for the player_game_score fact table.

Unit tests run anywhere. Table-dependent tests skip when the parquet has not
been built. Golden targets are checked against published Game Scores; the
Kobe Bryant 81-point game value of 63.5 was verified against the official
box line before being asserted here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nba_four_factors.player_state.gamescore import (
    TABLE,
    classify,
    game_score,
    parse_minutes,
)


def test_parse_minutes_mmss():
    assert parse_minutes("36:04") == pytest.approx(36 + 4 / 60)
    assert parse_minutes("0:45") == pytest.approx(0.75)


def test_parse_minutes_empty_and_bad():
    assert parse_minutes(None) == 0.0
    assert parse_minutes("") == 0.0
    assert parse_minutes("DNP") == 0.0


def test_classify():
    assert classify("", 31.5) == "played"
    assert classify("DNP - Coach's Decision", 0.0) == "dnp_coach"
    assert classify("DND - Injury/Illness", 0.0) == "dnp_other"
    assert classify("NWT - Not With Team", 0.0) == "dnp_other"


def test_game_score_arithmetic():
    row = {
        "pts": 81.0,
        "fgm": 28.0,
        "fga": 46.0,
        "ftm": 18.0,
        "fta": 20.0,
        "oreb": 2.0,
        "dreb": 4.0,
        "stl": 3.0,
        "ast": 2.0,
        "blk": 1.0,
        "pf": 1.0,
        "tov": 3.0,
    }
    assert game_score(row) == pytest.approx(63.5, abs=0.05)


needs_table = pytest.mark.skipif(not TABLE.exists(), reason="fact table not built")


@needs_table
def test_null_discipline():
    df = pd.read_parquet(TABLE, columns=["status", "game_score"])
    assert ((df.status != "played") & df.game_score.notna()).sum() == 0
    assert ((df.status == "played") & df.game_score.isna()).sum() == 0


@needs_table
def test_golden_kobe_81():
    df = pd.read_parquet(TABLE)
    r = df[(df.game_id == "0020500591") & (df.person_id == 977)].iloc[0]
    assert r.pts == 81
    assert r.game_score == pytest.approx(63.5, abs=0.05)
    assert r.status == "played"


@needs_table
def test_no_played_and_inactive_conflict():
    df = pd.read_parquet(TABLE, columns=["game_id", "person_id", "status"])
    played = df[df.status == "played"][["game_id", "person_id"]]
    inactive = df[df.status == "inactive"][["game_id", "person_id"]]
    both = played.merge(inactive, on=["game_id", "person_id"])
    assert len(both) == 0
