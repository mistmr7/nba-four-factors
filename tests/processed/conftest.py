"""Shared fixtures for processed-layer tests.

Two payload fixtures plus their expected derived values:

* ``schedule_payload`` — a 4-row, 2-game LeagueGameLog envelope mimicking
  the real shape, with one standard home/away game and one neutral-site
  (NBA Cup-style) game.
* ``schedule_payload_factor_check`` — a 2-row, 1-game payload using the
  exact box-score numbers whose four-factor outputs were hand-computed
  for the §7.1 fixture.  Lets pipeline tests assert against those numbers.

Tests of pure transformation logic feed these dicts directly to
``_tidy_schedule`` or ``add_factors``; no disk involved.
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------------------- #
# Shape of the real LeagueGameLog envelope
# --------------------------------------------------------------------------- #

LEAGUE_GAME_LOG_HEADERS: list[str] = [
    "SEASON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "PLUS_MINUS",
    "VIDEO_AVAILABLE",
]


def _row(
    *,
    team_id: int,
    team_abbr: str,
    team_name: str,
    game_id: str,
    game_date: str,
    matchup: str,
    wl: str,
    fgm: int,
    fga: int,
    fg3m: int,
    fg3a: int,
    ftm: int,
    fta: int,
    oreb: int,
    dreb: int,
    tov: int,
    pts: int,
    # Filler for columns we don't read but must populate to match shape.
    season_id: str = "22024",
    minutes: int = 240,
    reb: int | None = None,
    ast: int = 25,
    stl: int = 8,
    blk: int = 5,
    pf: int = 18,
    plus_minus: int = 0,
    video: int = 1,
) -> list:
    """Build one rowSet entry in column order, with sensible defaults."""
    fg_pct = round(fgm / fga, 3) if fga else 0.0
    fg3_pct = round(fg3m / fg3a, 3) if fg3a else 0.0
    ft_pct = round(ftm / fta, 3) if fta else 0.0
    if reb is None:
        reb = oreb + dreb
    return [
        season_id,
        team_id,
        team_abbr,
        team_name,
        game_id,
        game_date,
        matchup,
        wl,
        minutes,
        fgm,
        fga,
        fg_pct,
        fg3m,
        fg3a,
        fg3_pct,
        ftm,
        fta,
        ft_pct,
        oreb,
        dreb,
        reb,
        ast,
        stl,
        blk,
        tov,
        pf,
        pts,
        plus_minus,
        video,
    ]


# --------------------------------------------------------------------------- #
# Fixture 1: 2 games, 4 rows, one standard + one neutral
# --------------------------------------------------------------------------- #


@pytest.fixture
def schedule_payload() -> dict:
    """Four rows: one standard home/away game and one neutral-site game.

    Game 1 (0022400001): LAL vs. BOS at LAL — standard (one ``vs.``, one ``@``).
    Game 2 (0022400002): ATL vs. MIL at neutral site — both rows ``@``.

    The neutral game mimics the 5-per-season NBA Cup pattern observed in
    real 2024-25 data.  The point of this fixture is to verify that
    ``_tidy_schedule`` correctly produces ``is_home`` and ``is_neutral``
    columns for both patterns.
    """
    return {
        "resource": "leaguegamelog",
        "parameters": {},
        "resultSets": [
            {
                "name": "LeagueGameLog",
                "headers": LEAGUE_GAME_LOG_HEADERS,
                "rowSet": [
                    _row(
                        team_id=1610612747,
                        team_abbr="LAL",
                        team_name="Los Angeles Lakers",
                        game_id="0022400001",
                        game_date="2024-10-22",
                        matchup="LAL vs. BOS",
                        wl="W",
                        fgm=40,
                        fga=85,
                        fg3m=10,
                        fg3a=30,
                        ftm=15,
                        fta=20,
                        oreb=12,
                        dreb=35,
                        tov=14,
                        pts=110,
                    ),
                    _row(
                        team_id=1610612738,
                        team_abbr="BOS",
                        team_name="Boston Celtics",
                        game_id="0022400001",
                        game_date="2024-10-22",
                        matchup="BOS @ LAL",
                        wl="L",
                        fgm=38,
                        fga=88,
                        fg3m=8,
                        fg3a=28,
                        ftm=14,
                        fta=18,
                        oreb=10,
                        dreb=33,
                        tov=16,
                        pts=100,
                    ),
                    _row(
                        team_id=1610612737,
                        team_abbr="ATL",
                        team_name="Atlanta Hawks",
                        game_id="0022400002",
                        game_date="2024-12-17",
                        matchup="ATL @ MIL",
                        wl="W",
                        fgm=47,
                        fga=88,
                        fg3m=17,
                        fg3a=40,
                        ftm=6,
                        fta=10,
                        oreb=9,
                        dreb=35,
                        tov=16,
                        pts=117,
                    ),
                    _row(
                        team_id=1610612749,
                        team_abbr="MIL",
                        team_name="Milwaukee Bucks",
                        game_id="0022400002",
                        game_date="2024-12-17",
                        matchup="MIL @ ATL",
                        wl="L",
                        fgm=42,
                        fga=90,
                        fg3m=12,
                        fg3a=35,
                        ftm=9,
                        fta=12,
                        oreb=11,
                        dreb=30,
                        tov=14,
                        pts=105,
                    ),
                ],
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Fixture 2: 1 game, 2 rows, exact-numbers fixture for factor verification
# --------------------------------------------------------------------------- #


@pytest.fixture
def schedule_payload_factor_check() -> dict:
    """Single game using the §7.1 hand-computed factor inputs.

    Team (LAL):  FGM=40, FGA=85, FG3M=10, FG3A=30, FTM=15, FTA=20,
                 OREB=12, DREB=35, TOV=14, PTS=110
    Opp (BOS):   FGM=38, FGA=88, FG3M=8,  FG3A=28, FTM=14, FTA=18,
                 OREB=10, DREB=33, TOV=16, PTS=100

    Expected factor values are pinned in
    ``test_pipeline.EXPECTED_FACTORS_LAL`` and ``EXPECTED_FACTORS_BOS``.
    """
    return {
        "resource": "leaguegamelog",
        "parameters": {},
        "resultSets": [
            {
                "name": "LeagueGameLog",
                "headers": LEAGUE_GAME_LOG_HEADERS,
                "rowSet": [
                    _row(
                        team_id=1610612747,
                        team_abbr="LAL",
                        team_name="Los Angeles Lakers",
                        game_id="0022400001",
                        game_date="2024-10-22",
                        matchup="LAL vs. BOS",
                        wl="W",
                        fgm=40,
                        fga=85,
                        fg3m=10,
                        fg3a=30,
                        ftm=15,
                        fta=20,
                        oreb=12,
                        dreb=35,
                        tov=14,
                        pts=110,
                    ),
                    _row(
                        team_id=1610612738,
                        team_abbr="BOS",
                        team_name="Boston Celtics",
                        game_id="0022400001",
                        game_date="2024-10-22",
                        matchup="BOS @ LAL",
                        wl="L",
                        fgm=38,
                        fga=88,
                        fg3m=8,
                        fg3a=28,
                        ftm=14,
                        fta=18,
                        oreb=10,
                        dreb=33,
                        tov=16,
                        pts=100,
                    ),
                ],
            }
        ],
    }
