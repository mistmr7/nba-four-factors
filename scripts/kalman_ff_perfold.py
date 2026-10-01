"""Per-fold expanding-window tuning for the four-factor state-space model.

Mirrors scripts/kalman_perfold.py for the factor-tracking variant in
modeling/statespace_factors.py. Seasons sorted, first three burn-in, every
later season a test season. For each test season the readout calibration and
the noise levels (Q, R, P0) are fit or tuned on all completed prior seasons
only (one-step-ahead predictive NLL, same grid as the one-shot version),
then frozen through the test season.

Outputs:
    data/features/kalman_ff_perfold_preds.parquet
    data/features/kalman_ff_perfold_params.json

Run from repo root:
    python3 scripts/kalman_ff_perfold.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nba_four_factors.modeling import statespace_factors as sf

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "data" / "features"
PRED_PATH = FEAT / "kalman_ff_perfold_preds.parquet"
PARAM_PATH = FEAT / "kalman_ff_perfold_params.json"

GRID_Q = [0.002, 0.01, 0.03]
GRID_R = [0.5, 1.0, 2.0, 4.0]
GRID_P0 = [0.5, 1.0]
BURN = 3
TIME_BUDGET = 145.0


def main() -> None:
    t0 = time.time()
    m = sf.load()
    years = sorted(m["yr"].unique())
    test_years = years[BURN:]

    params_log = {}
    if PARAM_PATH.exists():
        params_log = json.loads(PARAM_PATH.read_text())
    done = set()
    frames = []
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
        cal = sf.calibrate(pre)
        seasons = [g for _, g in pre.groupby("season")]
        best, best_nll = None, np.inf
        for q in GRID_Q:
            for r_ in GRID_R:
                for p0 in GRID_P0:
                    nll = float(np.mean([sf.season_nll(g, (q, r_, p0), cal) for g in seasons]))
                    if nll < best_nll:
                        best_nll, best = nll, (q, r_, p0)

        rows = []
        for _, g in m[m["yr"] == ty].groupby("season"):
            pm, pv, z, w = sf.run_season(g, best, cal)
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
        params_log[str(ty)] = {
            "Q": best[0],
            "R": best[1],
            "P0": best[2],
            "nll": round(best_nll, 5),
        }
        PARAM_PATH.write_text(json.dumps(params_log, indent=1))
        mae = float(np.abs(fold.pred_m - fold.margin).mean())
        print(
            f"fold {ty}: Q={best[0]} R={best[1]} P0={best[2]} nll={best_nll:.4f} "
            f"MAE {mae:.3f} [{time.time() - t0:.0f}s]",
            flush=True,
        )
    print("all folds complete")


if __name__ == "__main__":
    main()
