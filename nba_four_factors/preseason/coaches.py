"""Head coach history per team-season and new-coach flags.

Source: Basketball-Reference season coaches pages (data/raw/preseason/
coaches/bbref_coaches_raw.txt, one block per season-ending year, rows in
page order, which is chronological within a team). Validated by
check_against_games: each team's coached games must equal games played,
and wins plus losses must equal games.

Flags for season S, all knowable before S tips off:

    new_coach         opening coach of S is not the coach who finished S-1
    offseason_hire    new_coach, and he did not coach the team at all in S-1
    promoted_interim  opening coach of S took over during S-1
    first_year        opening coach's first season with the franchise
                      (Basketball-Reference's count, which is cumulative,
                      so a returning coach is not first_year)
"""

from __future__ import annotations

import pandas as pd

from nba_four_factors.preseason import inputs as io

RAW = io.RAW_PRE / "coaches" / "bbref_coaches_raw.txt"

TEAM_ID = {
    "ATL": 1610612737,
    "BOS": 1610612738,
    "BRK": 1610612751,
    "NJN": 1610612751,
    "CHA": 1610612766,
    "CHO": 1610612766,
    "CHH": 1610612766,
    "NOH": 1610612740,
    "NOK": 1610612740,
    "NOP": 1610612740,
    "CHI": 1610612741,
    "CLE": 1610612739,
    "DAL": 1610612742,
    "DEN": 1610612743,
    "DET": 1610612765,
    "GSW": 1610612744,
    "HOU": 1610612745,
    "IND": 1610612754,
    "LAC": 1610612746,
    "LAL": 1610612747,
    "MEM": 1610612763,
    "VAN": 1610612763,
    "MIA": 1610612748,
    "MIL": 1610612749,
    "MIN": 1610612750,
    "NYK": 1610612752,
    "OKC": 1610612760,
    "SEA": 1610612760,
    "ORL": 1610612753,
    "PHI": 1610612755,
    "PHO": 1610612756,
    "POR": 1610612757,
    "SAC": 1610612758,
    "SAS": 1610612759,
    "TOR": 1610612761,
    "UTA": 1610612762,
    "WAS": 1610612764,
}


def load_coach_stints() -> pd.DataFrame:
    rows = []
    year = None
    order: dict[str, int] = {}
    for line in RAW.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            year = int(line[1:])
            order = {}
            continue
        coach, team, yrs, g, w, lo = line.rsplit(",", 5)
        order[team] = order.get(team, 0) + 1
        rows.append(
            {
                "season": io.season_id(int(year) - 1),
                "bbref_team": team,
                "team_id": TEAM_ID[team],
                "coach": coach,
                "stint": order[team],
                "yrs_franchise": int(yrs),
                "g": int(g),
                "w": int(w),
                "l": int(lo),
            }
        )
    return pd.DataFrame(rows)


def check_against_games(stints: pd.DataFrame) -> pd.DataFrame:
    """Coached games and wins per team-season vs the processed game table."""
    out = []
    for s in sorted(stints.season.unique()):
        if not (io.DATA / "processed" / s).exists():
            continue
        g = io.load_games(s)
        actual = g.groupby("team_id").agg(
            games=("margin", "size"), wins=("margin", lambda m: int((m > 0).sum()))
        )
        c = stints[stints.season == s].groupby("team_id")[["g", "w", "l"]].sum()
        j = actual.join(c, how="outer")
        j["season"] = s
        out.append(j.reset_index())
    chk = pd.concat(out, ignore_index=True)
    chk["games_ok"] = chk.games == chk.g
    chk["wins_ok"] = chk.wins == chk.w
    chk["wl_ok"] = chk.w + chk.l == chk.g
    return chk


def coach_flags(stints: pd.DataFrame) -> pd.DataFrame:
    first = stints.sort_values("stint").groupby(["season", "team_id"]).first().reset_index()
    last = stints.sort_values("stint").groupby(["season", "team_id"]).last().reset_index()
    n = stints.groupby(["season", "team_id"]).size().rename("n_coaches").reset_index()
    f = first[["season", "team_id", "coach", "yrs_franchise"]].rename(
        columns={"coach": "opening_coach"}
    )
    f = f.merge(n, on=["season", "team_id"])
    f["prev"] = f.season.map(io.prev_season)
    prev_last = last[["season", "team_id", "coach"]].rename(
        columns={"season": "prev", "coach": "prev_final_coach"}
    )
    prev_first = first[["season", "team_id", "coach"]].rename(
        columns={"season": "prev", "coach": "prev_opening_coach"}
    )
    prev_all = (
        stints.groupby(["season", "team_id"]).coach.apply(set).rename("prev_coaches").reset_index()
    )
    prev_all = prev_all.rename(columns={"season": "prev"})
    prev_n = n.rename(columns={"season": "prev", "n_coaches": "prev_n_coaches"})
    for p in [prev_last, prev_first, prev_all, prev_n]:
        f = f.merge(p, on=["prev", "team_id"], how="left")
    has_prev = f.prev_final_coach.notna()
    f["new_coach"] = has_prev & (f.opening_coach != f.prev_final_coach)
    in_prev = [
        isinstance(s, set) and c in s for c, s in zip(f.opening_coach, f.prev_coaches, strict=True)
    ]
    f["offseason_hire"] = f.new_coach & ~pd.Series(in_prev, index=f.index)
    f["promoted_interim"] = (
        has_prev & pd.Series(in_prev, index=f.index) & (f.opening_coach != f.prev_opening_coach)
    )
    f["first_year"] = f.yrs_franchise == 1
    f["prev_midseason_change"] = f.prev_n_coaches.fillna(1) > 1
    return f.drop(columns=["prev_coaches"])
