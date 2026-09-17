"""Tests for the stint builder.

Synthetic tests run anywhere. Table-dependent tests skip when the stint
table has not been built.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nba_four_factors.player_state.stints import (
    STINTS,
    GameRoster,
    build_game_stints,
    elapsed,
    fold,
    infer_period_starters,
    normalise_events,
    parse_clock,
)

HOME_TEAM, AWAY_TEAM = 1610612700, 1610612701
PBP_COLUMNS = [
    "gameId",
    "actionNumber",
    "clock",
    "period",
    "teamId",
    "personId",
    "location",
    "description",
    "actionType",
    "subType",
]


def roster():
    rows = []
    for i in range(1, 11):
        rows.append(
            {
                "game_id": "0020000001",
                "team_id": HOME_TEAM,
                "is_home": True,
                "person_id": 100 + i,
                "first_name": f"Home{i}",
                "family_name": f"Hplayer{i}",
                "name_i": f"H. Hplayer{i}",
                "starter": i <= 5,
                "seconds": 0.0,
            }
        )
        rows.append(
            {
                "game_id": "0020000001",
                "team_id": AWAY_TEAM,
                "is_home": False,
                "person_id": 200 + i,
                "first_name": f"Away{i}",
                "family_name": f"Aplayer{i}",
                "name_i": f"A. Aplayer{i}",
                "starter": i <= 5,
                "seconds": 0.0,
            }
        )
    return pd.DataFrame(rows)


def row(n, clock, period, team, pid, action, desc, sub_type=None):
    loc = "h" if team == HOME_TEAM else "v"
    return ["0020000001", n, clock, period, team, pid, loc, desc, action, sub_type]


def synthetic_game():
    """One 12 minute period. Home 101..105 start, 101 leaves for 106 at 6:00.

    Home scores a two and a three before the sub; away scores a two after it.
    Home player 105 never touches the ball.
    """
    rows = [
        row(1, "PT12M00.00S", 1, 0, 0, "period", "Start of 1st Period"),
        row(2, "PT11M30.00S", 1, HOME_TEAM, 102, "Made Shot", "Hplayer2 Jump Shot (2 PTS)"),
        row(3, "PT10M00.00S", 1, AWAY_TEAM, 201, "Missed Shot", "MISS Aplayer1 3PT Jump Shot"),
        row(4, "PT09M58.00S", 1, HOME_TEAM, 104, "Rebound", "Hplayer4 REBOUND (Off:0 Def:1)"),
        row(5, "PT09M00.00S", 1, HOME_TEAM, 101, "Made Shot", "Hplayer1 3PT Jump Shot (3 PTS)"),
        row(6, "PT08M00.00S", 1, AWAY_TEAM, 202, "Foul", "Aplayer2 P.FOUL", "Personal"),
        row(7, "PT06M00.00S", 1, HOME_TEAM, 101, "Substitution", "SUB: Hplayer6 FOR Hplayer1"),
        row(8, "PT05M00.00S", 1, AWAY_TEAM, 203, "Made Shot", "Aplayer3 Layup (2 PTS)"),
        row(
            9,
            "PT02M00.00S",
            1,
            AWAY_TEAM,
            204,
            "Turnover",
            "Aplayer4 Bad Pass Turnover",
            "Bad Pass",
        ),
        row(10, "PT01M00.00S", 1, HOME_TEAM, 106, "Missed Shot", "MISS Hplayer6 Layup"),
        row(11, "PT00M58.00S", 1, AWAY_TEAM, 205, "Rebound", "Aplayer5 REBOUND (Off:0 Def:1)"),
        row(12, "PT00M00.00S", 1, 0, 0, "period", "End of 1st Period"),
    ]
    return pd.DataFrame(rows, columns=PBP_COLUMNS)


def test_clock_parsing():
    assert parse_clock("PT12M00.00S") == 720
    assert parse_clock("PT00M04.50S") == pytest.approx(4.5)
    assert elapsed(1, "PT06M00.00S") == 360
    assert elapsed(5, "PT02M30.00S") == 150


def test_fold_strips_accents_and_case():
    assert fold("Šarić") == "saric"
    assert fold(" Vučević ") == "vucevic"


def test_name_resolution_prefers_name_i_then_family_then_prefix():
    r = roster()
    r.loc[r.person_id == 107, ["first_name", "family_name", "name_i"]] = [
        "Jalen",
        "Green",
        "Ja. Green",
    ]
    r.loc[r.person_id == 108, ["first_name", "family_name", "name_i"]] = [
        "Jeff",
        "Green",
        "Je. Green",
    ]
    gr = GameRoster(r)
    assert gr.resolve("home", "Hplayer3") == 103
    assert gr.resolve("home", "H. Hplayer3") == 103
    assert gr.resolve("home", "Green") is None
    assert gr.resolve("home", "Ja. Green") == 107
    assert gr.resolve("home", "Je. Green") == 108
    assert gr.resolve("away", "Hplayer3") is None


def test_period_starters_need_every_player_visible():
    gr = GameRoster(roster())
    events = [e for _, e in normalise_events(synthetic_game(), gr)]
    st = infer_period_starters(events)
    assert st["home"] == {101, 102, 104}
    assert st["away"] == {201, 202, 203, 204, 205}


def test_box_starters_and_stint_totals():
    rows, diag = build_game_stints(synthetic_game(), roster())
    assert diag[0]["usable"]
    st = pd.DataFrame(rows)
    assert len(st) == 2
    assert st["seconds"].tolist() == [360.0, 360.0]
    first, second = st.iloc[0], st.iloc[1]
    assert sorted([first.h1, first.h2, first.h3, first.h4, first.h5]) == [101, 102, 103, 104, 105]
    assert sorted([second.h1, second.h2, second.h3, second.h4, second.h5]) == [
        102,
        103,
        104,
        105,
        106,
    ]
    assert first.home_pts == 5 and first.away_pts == 0
    assert second.home_pts == 0 and second.away_pts == 2
    assert first.home_fga == 2 and first.away_fga == 1 and first.home_oreb == 0
    assert second.away_tov == 1 and second.home_fga == 1


def test_sub_in_without_sub_out_is_not_a_starter():
    """A player whose first action is logged just before their own sub-in did not start."""
    g = synthetic_game()
    g.loc[g.actionNumber == 4, "personId"] = 107
    extra = pd.DataFrame(
        [row(4.5, "PT09M58.00S", 1, HOME_TEAM, 104, "Substitution", "SUB: Hplayer7 FOR Hplayer4")],
        columns=PBP_COLUMNS,
    )
    g = pd.concat([g.iloc[:4], extra, g.iloc[4:]], ignore_index=True)
    gr = GameRoster(roster())
    st = infer_period_starters([e for _, e in normalise_events(g, gr)])
    assert 107 not in st["home"]
    assert 104 in st["home"]


def test_minutes_repair_fills_a_silent_full_period_player():
    """Without box starters, 105 is invisible in the feed and only box minutes can place them."""
    r = roster()
    r["starter"] = False
    r.loc[r.person_id.isin([101, 102, 103, 104, 105]), "seconds"] = 720.0
    r.loc[r.person_id == 101, "seconds"] = 360.0
    r.loc[r.person_id == 106, "seconds"] = 360.0
    r.loc[r.person_id.isin([201, 202, 203, 204, 205]), "seconds"] = 720.0
    rows, diag = build_game_stints(synthetic_game(), r)
    assert diag[0]["usable"]
    st = pd.DataFrame(rows)
    assert 105 in {st.iloc[0].h1, st.iloc[0].h2, st.iloc[0].h3, st.iloc[0].h4, st.iloc[0].h5}


needs_table = pytest.mark.skipif(
    not (STINTS / "2023_24_regular_season.parquet").exists(), reason="stints not built"
)


@needs_table
def test_period_seconds_sum_to_period_length():
    st = pd.read_parquet(STINTS / "2023_24_regular_season.parquet")
    per = st.groupby(["game_id", "period"])["seconds"].sum()
    expected = per.index.get_level_values("period").map(lambda p: 720 if p <= 4 else 300)
    assert (abs(per.to_numpy() - expected.to_numpy()) < 0.5).all()


@needs_table
def test_every_stint_has_ten_distinct_players():
    st = pd.read_parquet(STINTS / "2023_24_regular_season.parquet")
    ids = st[["h1", "h2", "h3", "h4", "h5", "a1", "a2", "a3", "a4", "a5"]].to_numpy()
    assert all(len(set(r)) == 10 for r in ids[:5000])
