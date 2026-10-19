"""Assemble one row per (season, team, rostered player) with everything known preseason.

Feature columns use only seasons before S. Target columns (suffix _S) come
from season S itself and are used only for fitting on past seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_four_factors.preseason import history
from nba_four_factors.preseason import inputs as io

AGE_CENTER = 27.0


def _age_lookup(bios: pd.DataFrame) -> pd.DataFrame:
    """Age during each season: own bio row, else prior bio row plus one."""
    if bios.empty:
        return pd.DataFrame(columns=["season", "person_id", "age"])
    b = bios[["season", "person_id", "age"]].dropna()
    nxt = b.copy()
    nxt["season"] = nxt.season.map(lambda s: io.season_id(io.start_year(s) + 1))
    nxt["age"] = nxt.age + 1
    both = pd.concat([b.assign(src=0), nxt.assign(src=1)])
    return both.sort_values("src").drop_duplicates(["season", "person_id"])[
        ["season", "person_id", "age"]
    ]


def build_frame(rosters: pd.DataFrame, prior: str = "theta") -> pd.DataFrame:
    ps = history.player_seasons()
    rapm = io.load_rapm_end_of_season(prior)
    rs = io.load_rapm_single_season()
    draft = io.load_draft()
    ages = _age_lookup(io.load_bios())

    f = rosters.copy()
    f["prev"] = f.season.map(io.prev_season)
    f["prev2"] = f.prev.map(io.prev_season)

    p1 = ps[["season", "person_id", "mpg", "gp_frac", "minutes", "main_team"]].rename(
        columns={
            "season": "prev",
            "mpg": "mpg_p",
            "gp_frac": "gp_frac_p",
            "minutes": "min_p",
            "main_team": "team_p",
        }
    )
    f = f.merge(p1, on=["prev", "person_id"], how="left")
    p2 = ps[["season", "person_id", "mpg"]].rename(columns={"season": "prev2", "mpg": "mpg_p2"})
    f = f.merge(p2, on=["prev2", "person_id"], how="left")

    played = ps[ps.minutes > 0][["season", "person_id"]]
    first = played.groupby("person_id").season.min().rename("first_season")
    f = f.merge(first, on="person_id", how="left")
    f["has_history"] = f.first_season.notna() & (f.first_season < f.season)

    r1 = rapm.rename(
        columns={"season": "prev", "v": "v_p", "o": "o_p", "d": "d_p", "poss": "poss_p"}
    )
    f = f.merge(r1, on=["prev", "person_id"], how="left")

    f = f.merge(ages, on=["season", "person_id"], how="left")
    f["age"] = pd.to_numeric(f.age, errors="coerce").astype(float)
    if "age_live" in f.columns:
        f["age"] = f.age.fillna(f.age_live)
    f = f.merge(draft, on="person_id", how="left")
    f["overall_pick"] = pd.to_numeric(f.overall_pick, errors="coerce").astype(float)
    f["draft_year"] = pd.to_numeric(f.draft_year, errors="coerce").astype(float)
    yr = f.season.map(io.start_year)
    f["drafted"] = f.overall_pick.notna()
    f["pick"] = f.overall_pick.fillna(io.UNDRAFTED_PICK).clip(upper=io.UNDRAFTED_PICK)
    f["log_pick"] = np.log(f.pick)
    f["rookie"] = ~f.has_history & ((f.draft_year == yr) | f.draft_year.isna())
    f["years_since_draft"] = (yr - f.draft_year).where(f.drafted)
    f["new_team"] = f.team_p.notna() & (f.team_p != f.team_id)
    for c in ["mpg_p", "gp_frac_p", "min_p", "mpg_p2", "poss_p"]:
        f[c] = f[c].fillna(0.0)
    f["age_c"] = f.age - AGE_CENTER

    tgt = ps[["season", "person_id", "mpg", "minutes"]].rename(
        columns={"mpg": "mpg_S", "minutes": "min_S"}
    )
    f = f.merge(tgt, on=["season", "person_id"], how="left")
    f["mpg_S"] = f.mpg_S.fillna(0.0)
    rs_t = rs.rename(columns={"v_season": "v_S", "poss_season": "poss_S"})
    f = f.merge(rs_t, on=["season", "person_id"], how="left")
    return f
