"""Flat-start (uniform prior) arm for the scalar Kalman filter.

Head-to-head against the Vegas-seeded filter: every team begins each season
at the league mean (seed slope forced to zero) rather than at its preseason
win-total seed. Noise levels are re-tuned per fold under the flat start,
with the P0 grid widened since a flat start warrants more initial
uncertainty. Same folds and rows as scripts/kalman_perfold.py.

Outputs:
    data/features/kalman_flat_perfold_preds.parquet
    data/features/kalman_flat_perfold_params.json

Run from repo root:
    python3 scripts/kalman_flat_start.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nba_four_factors.modeling import statespace_scalar as ss

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "data" / "features"
PRED_PATH = FEAT / "kalman_flat_perfold_preds.parquet"
PARAM_PATH = FEAT / "kalman_flat_perfold_params.json"

GRID_Q = [0.002, 0.01, 0.03, 0.08]
GRID_R = [110.0, 130.0, 150.0, 170.0]
GRID_P0 = [4.0, 9.0, 16.0, 25.0, 36.0]
BURN = 3
TIME_BUDGET = 140.0


def main() -> None:
    t0 = time.time()
    m = ss.load()
    years = sorted(m["yr"].unique())
    test_years = years[BURN:]

    params_log = {}
    if PARAM_PATH.exists():
        params_log = json.loads(PARAM_PATH.read_text())
    frames, done = [], set()
    if PRED_PATH.exists():
        prev = pd.read_parquet(PRED_PATH)
        frames = [prev]
        done = set(prev["test_yr"].unique())

    for ty in test_years:
        if ty in done:
            continue
        if time.time() - t0 > TIME_BUDGET:
            print("time budget reached; re-invoke to continue", flush=True)
            return
        pre = m[m["yr"] < ty]
        _slope, mean_wt, hca = ss.seed_fit(pre)
        slope = 0.0
        pre_seasons = [g.sort_values("game_date") for _, g in pre.groupby("season")]

        best, best_nll = None, np.inf
        for q in GRID_Q:
            for r_ in GRID_R:
                for p0 in GRID_P0:
                    p = (q, r_, p0, 1.0)
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
                        "test_yr": ty,
                        "pred_m": pm,
                        "pred_v": pv,
                        "margin": z,
                        "win": w,
                    }
                )
            )
        fold = pd.concat(rows, ignore_index=True)
        frames.append(fold)
        pd.concat(frames, ignore_index=True).to_parquet(PRED_PATH, index=False)
        params_log[str(ty)] = {"Q": best[0], "R": best[1], "P0": best[2], "nll": round(best_nll, 5)}
        PARAM_PATH.write_text(json.dumps(params_log, indent=1))
        mae = float(np.abs(fold.pred_m - fold.margin).mean())
        print(
            f"fold {ty}: Q={best[0]} R={best[1]} P0={best[2]} MAE {mae:.3f} "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
    print("all folds complete")


if __name__ == "__main__":
    main()
