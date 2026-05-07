"""Tests for nba_four_factors.analysis.pivots.

Coverage:

* ``pivot_to_game_level``
    - Single 2-team game produces a single wide row with correct values
    - Two independent games produce two rows with no contamination
    - Hand-computed ``home_margin`` and ``home_win`` match expectations
    - Neutral games excluded by default; included with ``include_neutral=True``
    - Empty input returns empty DataFrame with the canonical schema

* ``aggregate_to_team_season``
    - 2 teams x 1 season fixture produces 2 rows
    - ``games`` / ``wins`` / ``win_pct`` / ``mean_margin`` arithmetic
    - Factor averages match hand-computed values
    - Franchise relocation produces separate rows per ``team_abbr``
    - Mixing season types raises ValueError
    - Empty input returns empty DataFrame with the canonical schema

The pivot functions are pure (DataFrame in, DataFrame out), so tests
build hand-crafted DataFrames inline rather than going through the
``make_processed_tree`` fixture from conftest.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nba_four_factors.analysis.pivots import (
    _GAME_LEVEL_COLUMNS,
    _TEAM_SEASON_COLUMNS,
    aggregate_to_team_season,
    pivot_to_game_level,
)


def _row(
    game_id: str,
    team_abbr: str,
    opp_abbr: str,
    is_home: bool,
    pts: int,
    opp_pts: int,
    is_neutral: bool = False,
    season: str = "2023_24",
    season_type: str = "regular_season",
    game_date: str = "2023-11-01",
    off_efg_pct: float = 0.50,
    off_tov_pct: float = 0.13,
    off_orb_pct: float = 0.25,
    off_ft_rate: float = 0.23,
    def_efg_pct: float = 0.50,
    def_tov_pct: float = 0.13,
    def_orb_pct: float = 0.25,
    def_ft_rate: float = 0.23,
) -> dict:
    """Build a single processed-schema row as a dict.

    Box-stat numerics are placeholders; tests assert on the
    factor and points values that vary per case.
    """
    return {
        "game_id": game_id,
        "game_date": pd.Timestamp(game_date),
        "season": season,
        "season_type": season_type,
        "team_id": 0,
        "team_abbr": team_abbr,
        "opp_team_id": 0,
        "opp_abbr": opp_abbr,
        "is_home": is_home,
        "is_neutral": is_neutral,
        "fgm": 40,
        "fga": 85,
        "fg3m": 10,
        "fg3a": 30,
        "ftm": 15,
        "fta": 20,
        "oreb": 10,
        "dreb": 33,
        "tov": 14,
        "pts": pts,
        "opp_fgm": 38,
        "opp_fga": 88,
        "opp_fg3m": 8,
        "opp_fg3a": 28,
        "opp_ftm": 14,
        "opp_fta": 18,
        "opp_oreb": 9,
        "opp_dreb": 32,
        "opp_tov": 16,
        "opp_pts": opp_pts,
        "off_efg_pct": off_efg_pct,
        "off_tov_pct": off_tov_pct,
        "off_orb_pct": off_orb_pct,
        "off_ft_rate": off_ft_rate,
        "def_efg_pct": def_efg_pct,
        "def_tov_pct": def_tov_pct,
        "def_orb_pct": def_orb_pct,
        "def_ft_rate": def_ft_rate,
        "margin": pts - opp_pts,
    }


def test_pivot_single_game_basic():
    """Single 2-team game produces one wide row with correct values."""
    df = pd.DataFrame(
        [
            _row("0001", "BOS", "LAL", is_home=True, pts=110, opp_pts=100, off_efg_pct=0.55),
            _row("0001", "LAL", "BOS", is_home=False, pts=100, opp_pts=110, off_efg_pct=0.48),
        ]
    )

    result = pivot_to_game_level(df)

    assert len(result) == 1
    assert tuple(result.columns) == _GAME_LEVEL_COLUMNS

    row = result.iloc[0]
    assert row["game_id"] == "0001"
    assert row["home_team_abbr"] == "BOS"
    assert row["away_team_abbr"] == "LAL"
    assert row["home_efg"] == 0.55
    assert row["away_efg"] == 0.48
    assert row["home_margin"] == 10
    assert row["home_win"] == 1


def test_pivot_two_games_no_contamination():
    """Independent games produce independent rows with correct factor routing."""
    df = pd.DataFrame(
        [
            _row("0001", "BOS", "LAL", is_home=True, pts=110, opp_pts=100, off_efg_pct=0.55),
            _row("0001", "LAL", "BOS", is_home=False, pts=100, opp_pts=110, off_efg_pct=0.48),
            _row("0002", "GSW", "DEN", is_home=True, pts=120, opp_pts=125, off_efg_pct=0.60),
            _row("0002", "DEN", "GSW", is_home=False, pts=125, opp_pts=120, off_efg_pct=0.52),
        ]
    )

    result = pivot_to_game_level(df).sort_values("game_id").reset_index(drop=True)

    assert len(result) == 2

    g1 = result.iloc[0]
    assert g1["home_team_abbr"] == "BOS"
    assert g1["away_team_abbr"] == "LAL"
    assert g1["home_efg"] == 0.55
    assert g1["away_efg"] == 0.48
    assert g1["home_margin"] == 10
    assert g1["home_win"] == 1

    g2 = result.iloc[1]
    assert g2["home_team_abbr"] == "GSW"
    assert g2["away_team_abbr"] == "DEN"
    assert g2["home_efg"] == 0.60
    assert g2["away_efg"] == 0.52
    assert g2["home_margin"] == -5
    assert g2["home_win"] == 0


def test_pivot_home_loss_yields_zero_win():
    """home_win is exactly the boolean cast of home_margin > 0."""
    df = pd.DataFrame(
        [
            _row("0001", "BOS", "LAL", is_home=True, pts=95, opp_pts=100),
            _row("0001", "LAL", "BOS", is_home=False, pts=100, opp_pts=95),
        ]
    )

    result = pivot_to_game_level(df)

    assert result.iloc[0]["home_margin"] == -5
    assert result.iloc[0]["home_win"] == 0


def test_pivot_excludes_neutral_by_default():
    """Neutral-site games dropped when include_neutral is False (default)."""
    df = pd.DataFrame(
        [
            _row("0001", "BOS", "LAL", is_home=True, pts=110, opp_pts=100),
            _row("0001", "LAL", "BOS", is_home=False, pts=100, opp_pts=110),
            _row("0002", "GSW", "DEN", is_home=False, pts=120, opp_pts=115, is_neutral=True),
            _row("0002", "DEN", "GSW", is_home=False, pts=115, opp_pts=120, is_neutral=True),
        ]
    )

    result = pivot_to_game_level(df)

    assert len(result) == 1
    assert result.iloc[0]["game_id"] == "0001"


def test_pivot_includes_neutral_when_requested():
    """include_neutral=True keeps neutrals; home/away assigned alphabetically."""
    df = pd.DataFrame(
        [
            _row("0001", "BOS", "LAL", is_home=True, pts=110, opp_pts=100),
            _row("0001", "LAL", "BOS", is_home=False, pts=100, opp_pts=110),
            _row("0002", "GSW", "DEN", is_home=False, pts=115, opp_pts=120, is_neutral=True),
            _row("0002", "DEN", "GSW", is_home=False, pts=120, opp_pts=115, is_neutral=True),
        ]
    )

    result = (
        pivot_to_game_level(df, include_neutral=True).sort_values("game_id").reset_index(drop=True)
    )

    assert len(result) == 2

    standard = result[result["game_id"] == "0001"].iloc[0]
    assert standard["home_team_abbr"] == "BOS"
    assert not standard["is_neutral"]

    neutral = result[result["game_id"] == "0002"].iloc[0]
    assert neutral["is_neutral"]
    assert neutral["home_team_abbr"] == "DEN"
    assert neutral["away_team_abbr"] == "GSW"
    assert neutral["home_margin"] == 120 - 115


def test_pivot_empty_input_returns_empty_with_schema():
    df = pd.DataFrame(columns=list(_row("0001", "X", "Y", True, 0, 0).keys()))

    result = pivot_to_game_level(df)

    assert result.empty
    assert tuple(result.columns) == _GAME_LEVEL_COLUMNS


def test_aggregate_basic_two_team_one_season():
    """Two teams playing one game: aggregate gives 2 rows with correct stats."""
    df = pd.DataFrame(
        [
            _row("0001", "BOS", "LAL", is_home=True, pts=110, opp_pts=100, off_efg_pct=0.55),
            _row("0001", "LAL", "BOS", is_home=False, pts=100, opp_pts=110, off_efg_pct=0.48),
        ]
    )

    result = aggregate_to_team_season(df).sort_values("team_abbr").reset_index(drop=True)

    assert len(result) == 2
    assert tuple(result.columns) == _TEAM_SEASON_COLUMNS

    bos = result[result["team_abbr"] == "BOS"].iloc[0]
    assert bos["games"] == 1
    assert bos["wins"] == 1
    assert bos["win_pct"] == 1.0
    assert bos["mean_margin"] == 10
    assert bos["off_efg"] == 0.55

    lal = result[result["team_abbr"] == "LAL"].iloc[0]
    assert lal["games"] == 1
    assert lal["wins"] == 0
    assert lal["win_pct"] == 0.0
    assert lal["mean_margin"] == -10


def test_aggregate_factor_averages_match_handcomputed():
    """Two-game team-season averages: hand-computed mean of off_efg_pct."""
    df = pd.DataFrame(
        [
            _row("0001", "BOS", "LAL", is_home=True, pts=110, opp_pts=100, off_efg_pct=0.50),
            _row("0001", "LAL", "BOS", is_home=False, pts=100, opp_pts=110, off_efg_pct=0.45),
            _row(
                "0002",
                "BOS",
                "NYK",
                is_home=True,
                pts=105,
                opp_pts=95,
                off_efg_pct=0.60,
                game_date="2023-11-03",
            ),
            _row(
                "0002",
                "NYK",
                "BOS",
                is_home=False,
                pts=95,
                opp_pts=105,
                off_efg_pct=0.42,
                game_date="2023-11-03",
            ),
        ]
    )

    result = aggregate_to_team_season(df)
    bos = result[result["team_abbr"] == "BOS"].iloc[0]

    assert bos["games"] == 2
    assert bos["wins"] == 2
    assert bos["win_pct"] == 1.0
    assert bos["mean_margin"] == 10.0
    assert bos["off_efg"] == pytest.approx((0.50 + 0.60) / 2)


def test_aggregate_franchise_relocation_separate_rows():
    """Same franchise under different team_abbr in different seasons gets separate rows."""
    df = pd.DataFrame(
        [
            _row("0001", "SEA", "LAL", is_home=True, pts=110, opp_pts=100, season="2007_08"),
            _row("0001", "LAL", "SEA", is_home=False, pts=100, opp_pts=110, season="2007_08"),
            _row(
                "0002",
                "OKC",
                "LAL",
                is_home=True,
                pts=105,
                opp_pts=95,
                season="2008_09",
                game_date="2008-11-01",
            ),
            _row(
                "0002",
                "LAL",
                "OKC",
                is_home=False,
                pts=95,
                opp_pts=105,
                season="2008_09",
                game_date="2008-11-01",
            ),
        ]
    )

    result = aggregate_to_team_season(df)

    abbrs = set(result["team_abbr"])
    assert "SEA" in abbrs
    assert "OKC" in abbrs
    assert len(result[result["team_abbr"] == "SEA"]) == 1
    assert len(result[result["team_abbr"] == "OKC"]) == 1


def test_aggregate_mixed_season_types_raises():
    df = pd.DataFrame(
        [
            _row(
                "0001",
                "BOS",
                "LAL",
                is_home=True,
                pts=110,
                opp_pts=100,
                season_type="regular_season",
            ),
            _row(
                "0001",
                "LAL",
                "BOS",
                is_home=False,
                pts=100,
                opp_pts=110,
                season_type="regular_season",
            ),
            _row("0002", "BOS", "LAL", is_home=True, pts=110, opp_pts=100, season_type="playoffs"),
            _row("0002", "LAL", "BOS", is_home=False, pts=100, opp_pts=110, season_type="playoffs"),
        ]
    )

    with pytest.raises(ValueError, match="single season_type"):
        aggregate_to_team_season(df)


def test_aggregate_empty_input_returns_empty_with_schema():
    df = pd.DataFrame(columns=list(_row("0001", "X", "Y", True, 0, 0).keys()))

    result = aggregate_to_team_season(df)

    assert result.empty
    assert tuple(result.columns) == _TEAM_SEASON_COLUMNS
