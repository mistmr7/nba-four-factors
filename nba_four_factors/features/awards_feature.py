"""Accolade-weighted availability: value an absence by the player's star label.

Each player-season gets a star score from their honors, and we attribute it to
LATER seasons only (a player's label entering season S comes from S-1 and earlier),
so it is causal. All-NBA is voted after the season, so using the prior season's
result as the current label introduces no lookahead.

Star score for a season:
    All-NBA 1st/2nd/3rd  -> 5.0 / 3.5 / 2.5
    All-Star (any)       -> +1.5
    All-Defensive 1st/2nd-> +1.5 / +1.0
    MVP                  -> +4.0
    DPOY                 -> +2.0

Two lags are built: acc1 uses the prior season only; acc3 uses the best of the
prior three seasons, so a star who missed last year to injury is still flagged. A
binary "All-Star caliber last few years" count is also produced.

Output (home perspective, per game): home/away sums and the differential for each,
written to availability_accolades.parquet. Run:
    python -m nba_four_factors.features.awards_feature
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
FEAT = REPO / "data" / "features"
NBA_PTS = {1: 5.0, 2: 3.5, 3: 2.5}
DEF_PTS = {1: 1.5, 2: 1.0}


def honor_scores() -> dict:
    a = pd.read_parquet(FEAT / "accolades.parquet")
    a["yr"] = a["season"].str[:4].astype(int)
    score = {}
    for (pid, yr), g in a.groupby(["person_id", "yr"]):
        s = 0.0
        aw = set(g["award"])
        if "all_nba" in aw:
            t = int(g.loc[g["award"] == "all_nba", "tier"].min())
            s += NBA_PTS.get(t, 2.5)
        if "all_star" in aw:
            s += 1.5
        if "all_defensive" in aw:
            t = int(g.loc[g["award"] == "all_defensive", "tier"].min())
            s += DEF_PTS.get(t, 1.0)
        if "mvp" in aw:
            s += 4.0
        if "dpoy" in aw:
            s += 2.0
        score[(int(pid), int(yr))] = s
    return score


def game_dates() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_parquet(f) for f in glob.glob(str(PROC / "*/regular_season.parquet"))],
        ignore_index=True,
    )
    return df[["game_id"]].drop_duplicates()


def build():
    score = honor_scores()
    ina = pd.read_parquet(FEAT / "inactives.parquet")
    ina = ina[ina["season_type"] == "regular_season"].copy()
    ina["yr"] = ina["season"].str[:4].astype(int)

    def acc1(pid, yr):
        return score.get((pid, yr - 1), 0.0)

    def acc3(pid, yr):
        return max(
            score.get((pid, yr - 1), 0.0),
            score.get((pid, yr - 2), 0.0),
            score.get((pid, yr - 3), 0.0),
        )

    ina["acc1"] = [acc1(int(p), int(y)) for p, y in zip(ina["person_id"], ina["yr"], strict=False)]
    ina["acc3"] = [acc3(int(p), int(y)) for p, y in zip(ina["person_id"], ina["yr"], strict=False)]
    ina["star"] = (ina["acc3"] >= 1.5).astype(float)  # All-Star caliber in last 3 yrs
    ina["allnba"] = (ina["acc3"] >= 2.5).astype(float)  # All-NBA caliber in last 3 yrs

    metrics = ["acc1", "acc3", "star", "allnba"]
    agg = ina.groupby(["game_id", "is_home"])[metrics].sum().reset_index()
    home = agg[agg["is_home"]].set_index("game_id")[metrics].add_prefix("home_")
    away = agg[~agg["is_home"]].set_index("game_id")[metrics].add_prefix("away_")
    games = game_dates().set_index("game_id")
    f = games.join(home).join(away).fillna(0.0).reset_index()
    for m in metrics:
        f[m + "_diff"] = f["home_" + m] - f["away_" + m]
    f.to_parquet(FEAT / "availability_accolades.parquet", index=False)

    print(f"availability_accolades.parquet  {len(f):,} games")
    print(
        f"  games with an All-Star-caliber availability gap (|star_diff|>=1): "
        f"{100*(f['star_diff'].abs() >= 1).mean():.1f}%"
    )
    print(
        f"  games with an All-NBA-caliber gap (|allnba_diff|>=1): "
        f"{100*(f['allnba_diff'].abs() >= 1).mean():.1f}%"
    )
    print("  mean |diff|:  " + "  ".join(f"{m} {f[m+'_diff'].abs().mean():.3f}" for m in metrics))


if __name__ == "__main__":
    build()
