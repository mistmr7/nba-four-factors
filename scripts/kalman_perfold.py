"""Per-fold expanding-window tuning for the scalar Kalman filter.

Replaces the one-shot pre-2010 tune in modeling/statespace_scalar.py with
the ladder's walk-forward convention. Seasons are sorted, the first three
are burn-in, and every later season is a test season. For each test season,
the seed mapping (win-total slope, mean win total, HCA prior) and the noise
levels (Q, R, P0) are fit or tuned on all completed prior seasons only,
then frozen through the test season. Tuning criterion is one-step-ahead
predictive negative log likelihood of observed margins, which identifies
the absolute noise scale that point-error metrics cannot.

Grid (same as the one-shot version):
    Q in {0.002, 0.01, 0.03, 0.08}
    R in {110, 130, 150, 170}
    P0 in {4, 9, 16}
    P0_hca fixed at 1.0

Reports overall margin MAE and win log loss across all test seasons, and
the Vegas closing-line head-to-head on the 2007+ rows where lines exist.

Outputs:
    data/features/kalman_perfold_preds.parquet
    data/features/kalman_perfold_params.json

Run from repo root:
    python3 scripts/kalman_perfold.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nba_four_factors.modeling import statespace_scalar as ss

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "data" / "features"
PRED_PATH = FEAT / "kalman_perfold_preds.parquet"
PARAM_PATH = FEAT / "kalman_perfold_params.json"

GRID_Q = [0.002, 0.01, 0.03, 0.08]
GRID_R = [110.0, 130.0, 150.0, 170.0]
GRID_P0 = [4.0, 9.0, 16.0]
P0_HCA = 1.0
BURN = 3


def phi(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def main():
    m = ss.load()
    m = m.sort_values(["yr", "game_date"]).reset_index(drop=True)
    years = sorted(m["yr"].unique())
    test_years = years[BURN:]

    params_log = {}
    pred_frames = []
    t_start = time.time()
    for ty in test_years:
        pre = m[m["yr"] < ty]
        slope, mean_wt, hca = ss.seed_fit(pre)
        pre_seasons = [g.sort_values("game_date") for _, g in pre.groupby("season")]

        best, best_nll = None, np.inf
        for q in GRID_Q:
            for r_ in GRID_R:
                for p0 in GRID_P0:
                    p = (q, r_, p0, P0_HCA)
                    nll = float(
                        np.mean([ss.season_nll(g, p, slope, mean_wt, hca) for g in pre_seasons])
                    )
                    if nll < best_nll:
                        best_nll, best = nll, p

        rows = []
        for _, g in m[m["yr"] == ty].groupby("season"):
            g = g.sort_values("game_date")
            pm, pv, z, w = ss.run_season(g, best, slope, mean_wt, hca)
            rows.append(
                pd.DataFrame(
                    {
                        "game_id": g["game_id"].values,
                        "season": g["season"].values,
                        "test_yr": ty,
                        "pred_m": pm,
                        "pred_v": pv,
                        "margin": z,
                        "win": w,
                    }
                )
            )
        pred_frames.append(pd.concat(rows, ignore_index=True))
        params_log[str(ty)] = {
            "Q": best[0],
            "R": best[1],
            "P0": best[2],
            "nll": round(best_nll, 5),
            "slope": round(slope, 4),
            "hca": round(hca, 3),
            "n_tune_seasons": len(pre_seasons),
        }
        el = time.time() - t_start
        print(
            f"fold {ty}: Q={best[0]} R={best[1]} P0={best[2]} nll={best_nll:.4f} "
            f"slope={slope:.3f} hca={hca:.2f} [{el:.0f}s]",
            flush=True,
        )

    FEAT.mkdir(parents=True, exist_ok=True)
    r = pd.concat(pred_frames, ignore_index=True)
    r.to_parquet(PRED_PATH, index=False)
    PARAM_PATH.write_text(json.dumps(params_log, indent=1))

    r["ae"] = (r.margin - r.pred_m).abs()
    r["p_win"] = np.clip(phi(r.pred_m / np.sqrt(r.pred_v)), 1e-6, 1 - 1e-6)
    r["ll"] = -(r.win * np.log(r.p_win) + (1 - r.win) * np.log(1 - r.p_win))

    print(f"\ntest seasons: {r.test_yr.nunique()}  rows: {len(r):,}")
    print(f"overall margin MAE : {r.ae.mean():.4f}")
    print(f"overall win logloss: {r.ll.mean():.4f}")
    print(f"MAE on 2010+ rows  : {r[r.test_yr >= 2010].ae.mean():.4f}")

    lines = pd.read_parquet(ss.LINES)[["game_id", "vegas_home_margin"]].dropna()
    v = r.merge(lines, on="game_id", how="inner")
    if len(v):
        v["vae"] = (v.margin - v.vegas_home_margin).abs()
        print(f"\nVegas head-to-head rows: {len(v):,} ({v.test_yr.min()}-{v.test_yr.max()})")
        print(f"  Kalman MAE {v.ae.mean():.4f}   Vegas MAE {v.vae.mean():.4f}")
        by = v.groupby("test_yr").agg(k=("ae", "mean"), veg=("vae", "mean"))
        wins = int((by.k < by.veg).sum())
        print(f"  seasons Kalman beats closing line: {wins}/{len(by)}")


if __name__ == "__main__":
    main()
