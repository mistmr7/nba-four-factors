"""Where the model misses relative to Vegas, three cuts, 2007+ lined universe.

Same fold protocol as scripts/avail_512_evidence.py (A0 = Kalman +
intercept, A1 = Kalman + availability correction, Vegas closing line),
then MAE by:

    1. |Vegas closing line| bucket: 0-2, 2-4, 4-6, 6-8, 8-10, 10+
    2. season game number (mean of the two teams'): 1-10 ... 71+
    3. final-season-wins bracket of the involved teams, wins scaled to an
       82-game pace for shortened seasons (2011-12, 2019-20, 2020-21);
       each game counts once per involved team

Run from repo root:
    PYTHONPATH=. python3 scripts/avail_512_where_we_miss.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from nba_four_factors.player_state.followup import _apply, _fit_beta, team_features

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "data" / "features"


def fold_preds() -> pd.DataFrame:
    t = team_features()
    rows = []
    for ty in range(2007, int(t.test_yr.max()) + 1):
        tr = t[(t.test_yr >= 2005) & (t.test_yr < ty)]
        te = t[t.test_yr == ty].dropna(subset=["vegas_home_margin"]).copy()
        if len(te) < 200:
            continue
        te["a0"] = _apply(te, _fit_beta(tr, []), [])
        te["a1"] = _apply(te, _fit_beta(tr, ["gmscs_diff"]), ["gmscs_diff"])
        rows.append(te)
    d = pd.concat(rows, ignore_index=True)
    d["ae0"] = (d.a0 - d.margin).abs()
    d["ae1"] = (d.a1 - d.margin).abs()
    d["aev"] = (d.vegas_home_margin - d.margin).abs()
    return d


def team_store() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_parquet(f, columns=["game_id", "team_id", "season", "game_date",
                                     "is_neutral", "margin"])
         for f in sorted(glob.glob(str(REPO / "data" / "processed" / "*" / "regular_season.parquet")))],
        ignore_index=True)
    df = df[~df.is_neutral].copy()
    df["game_date"] = pd.to_datetime(df.game_date)
    df = df.sort_values(["team_id", "season", "game_date"])
    df["gn"] = df.groupby(["team_id", "season"]).cumcount() + 1
    df["win"] = (df.margin > 0).astype(int)
    return df


def report(d: pd.DataFrame, key: str, order) -> None:
    g = d.groupby(key, observed=True).agg(
        n=("ae0", "size"), kalman=("ae0", "mean"),
        avail=("ae1", "mean"), vegas=("aev", "mean"))
    g = g.reindex(order)
    g["gap"] = g.avail - g.vegas
    print(g.round(3).to_string())


def main() -> None:
    d = fold_preds()
    ts = team_store()

    mt = pd.read_parquet(FEAT / "modeling_table.parquet",
                         columns=["game_id", "team_id", "away_team_id", "season"])
    mt = mt.rename(columns={"season": "season_mt"})
    d = d.merge(mt, on="game_id", how="left")

    # Cut 1: |Vegas line|
    b1 = [0, 2, 4, 6, 8, 10, np.inf]
    l1 = ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"]
    d["line_b"] = pd.cut(d.vegas_home_margin.abs(), b1, labels=l1,
                         include_lowest=True, right=True)
    print("=== 1. By |Vegas closing line| ===")
    report(d, "line_b", l1)

    # Cut 2: season game number (mean of the two teams')
    gp = {(g, int(t)): int(n)
          for g, t, n in ts[["game_id", "team_id", "gn"]].itertuples(index=False)}
    d["ngames"] = [
        (gp.get((g, int(h)), np.nan) + gp.get((g, int(a)), np.nan)) / 2
        for g, h, a in zip(d.game_id, d.team_id, d.away_team_id, strict=False)]
    b2 = [0, 10, 20, 30, 40, 50, 60, 70, np.inf]
    l2 = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71+"]
    d["gn_b"] = pd.cut(d.ngames, b2, labels=l2, right=True)
    print("\n=== 2. By season game number (mean of both teams) ===")
    report(d, "gn_b", l2)

    # Cut 3: final-season wins of involved teams (82-game pace)
    wins = ts.groupby(["team_id", "season"]).agg(w=("win", "sum"), g=("win", "size"))
    wins["w82"] = wins.w / wins.g * 82.0
    wmap = wins["w82"].to_dict()
    long = pd.concat([
        d.assign(t=d.team_id), d.assign(t=d.away_team_id)], ignore_index=True)
    long["w82"] = [wmap.get((int(t), s), np.nan)
                   for t, s in zip(long.t, long.season_mt, strict=False)]
    b3 = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, np.inf]
    l3 = ["<=20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50",
          "51-55", "56-60", "61-65", "66+"]
    long["w_b"] = pd.cut(long.w82, b3, labels=l3, right=True)
    print("\n=== 3. By involved team's final wins (82-game pace; games count once per team) ===")
    report(long, "w_b", l3)

    d.to_parquet(FEAT / "avail_512_where_we_miss.parquet", index=False)


if __name__ == "__main__":
    main()
