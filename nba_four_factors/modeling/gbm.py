"""Gradient-boosted trees (HistGradientBoosting) for game prediction.

The strongest default model class for tabular data, and a test of whether
nonlinearities and interactions the linear regression cannot represent (rest x
travel, strength x consistency, prior x games-played) are worth capturing. The
tree model is given the rich feature set, including the Vegas prior and games-
played, so it can learn the seed-blend and the reliability shrinkage on its own
instead of us hand-engineering them. Missing early-season features are handled
natively by the trees.

Evaluated expanding-window by season (train on prior seasons, predict the next),
the same protocol as the RNN and regression, and compared per season to the
Vegas closing line. Run: python -m nba_four_factors.modeling.gbm
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, mean_absolute_error

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
LINES = REPO / "data" / "features" / "vegas_game_lines.parquet"
START_TEST_YEAR = 2010


def games_played() -> dict:
    df = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "*/regular_season.parquet")))],
        ignore_index=True,
    )
    df = df[~df["is_neutral"]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_id", "season", "game_date"])
    df["gn"] = df.groupby(["team_id", "season"]).cumcount() + 1
    return {
        (g, int(t)): int(n) for g, t, n in df[["game_id", "team_id", "gn"]].itertuples(index=False)
    }


def build():
    m = pd.read_parquet(TABLE).copy()
    gp = games_played()
    m["ngames"] = [
        0.5 * (gp.get((g, int(t)), 1) + gp.get((g, int(o)), 1))
        for g, t, o in zip(m["game_id"], m["team_id"], m["away_team_id"], strict=False)
    ]
    m["home_dev"] = m["form_mean"] - m["base_comp"]
    m["away_dev"] = m["away_form_mean"] - m["away_base_comp"]
    feats = [
        "wintotal_diff",
        "d_efg_d",
        "d_oreb_d",
        "d_tov_d",
        "d_ftmfga_d",
        "rest_diff",
        "b2b_diff",
        "miles7d_diff",
        "pace_diff",
        "home_dev",
        "away_dev",
        "form_var",
        "away_form_var",
        "ngames",
    ]
    return m, feats


def main():
    m, feats = build()
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    m = m.merge(lines, on="game_id", how="left")
    m["yr"] = m["season"].str[:4].astype(int)
    seasons = sorted(m["season"].unique(), key=lambda s: int(s[:4]))

    reg_kw = dict(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
    )
    rows = []
    for ts in seasons:
        ty = int(ts[:4])
        if ty < START_TEST_YEAR:
            continue
        tr = m[m["yr"] < ty]
        te = m[m["yr"] == ty]
        te = te.dropna(subset=["home_margin", "home_win"])
        if len(tr) < 2000 or len(te) < 100:
            continue
        rg = HistGradientBoostingRegressor(**reg_kw).fit(tr[feats], tr["home_margin"])
        cl = HistGradientBoostingClassifier(**reg_kw).fit(tr[feats], tr["home_win"])
        pm = rg.predict(te[feats])
        pw = np.clip(cl.predict_proba(te[feats])[:, 1], 1e-6, 1 - 1e-6)
        mae = mean_absolute_error(te["home_margin"], pm)
        ll = log_loss(te["home_win"], pw)
        v = te.dropna(subset=["vegas_home_margin"])
        vmae = mean_absolute_error(v["home_margin"], v["vegas_home_margin"]) if len(v) else np.nan
        rows.append((ty, len(te), mae, ll, vmae))
        print(
            f"  {ty}  n={len(te):4d}  GBM MAE {mae:6.3f} ll {ll:.4f}  |  Vegas MAE {vmae:6.3f}",
            flush=True,
        )

    a = np.array([[r[2], r[3], r[4]] for r in rows])
    print(f"\nExpanding-window averages over {len(rows)} seasons:")
    print(f"  GBM        : MAE {a[:,0].mean():.3f}  logloss {a[:,1].mean():.4f}")
    print(f"  Vegas line : MAE {a[:,2].mean():.3f}")
    print(
        "  reference (same protocol): RNN+seed MAE 10.133 ll 0.6104 | M3+seed MAE 10.189 ll 0.6139"
    )
    print(
        f"  GBM minus RNN+seed (MAE): {a[:,0].mean()-10.133:+.3f}   GBM minus Vegas: {a[:,0].mean()-a[:,2].mean():+.3f}"
    )
    print(f"\n(trained {len(rows)} folds; features used: {len(feats)})")


if __name__ == "__main__":
    main()
