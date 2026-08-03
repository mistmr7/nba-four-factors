"""Sweep the availability weightings: which one helps against our worst games?

For each weighting metric we add its home-minus-away differential to the additive
four-factor model and re-evaluate expanding-window (train on prior seasons, test on
the next, 2010-2025). We report three things per metric:

  - overall out-of-sample MAE change versus the no-availability baseline
  - the fitted coefficient (points per unit), for sign and size
  - the change ON the games where that metric flags the biggest imbalance (its top
    decile of |diff|), alongside the Vegas error on those same games, since closing
    the gap there is the whole point

A metric that helps should do almost nothing on balanced games and a lot on the
imbalance decile, pulling our error on those games toward the line.

Run: python -m nba_four_factors.modeling.availability_sweep
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
AVAIL = REPO / "data" / "features" / "availability_features.parquet"
LINES = REPO / "data" / "features" / "vegas_game_lines.parquet"
BASE = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
METRICS = ["n", "rot", "starter", "mpg", "pts", "gmsc", "pm"]


def load():
    m = pd.read_parquet(TABLE).dropna(subset=["home_margin", "home_win"]).copy()
    m["yr"] = m["season"].str[:4].astype(int)
    av = pd.read_parquet(AVAIL)
    m = m.merge(av, on="game_id", how="inner")
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    m = m.merge(lines, on="game_id", how="left")
    feat_cols = BASE + [met + "_diff" for met in METRICS]
    m[feat_cols] = m[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return m[m["yr"] >= 2010].reset_index(drop=True)


def expanding_pred(m, cols):
    y = m["home_margin"].to_numpy(dtype=float)
    yr = m["yr"].to_numpy()
    X = m[cols].to_numpy(dtype=float)
    pred = np.full(len(m), np.nan)
    sig = np.full(len(m), np.nan)
    for ty in range(2010, yr.max() + 1):
        tr, te = yr < ty, yr == ty
        if tr.sum() < 2000 or te.sum() < 50:
            continue
        X1 = np.column_stack([np.ones(tr.sum()), X[tr]])
        beta = np.linalg.lstsq(X1, y[tr], rcond=None)[0]
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ beta
        sig[te] = np.std(y[tr] - X1 @ beta) + 1e-6
    return pred, sig


def logloss(pred, sig, w):
    p = np.clip(0.5 * (1 + np.vectorize(math.erf)(pred / (sig * math.sqrt(2)))), 1e-6, 1 - 1e-6)
    return -np.mean(w * np.log(p) + (1 - w) * np.log(1 - p))


def main():
    m = load()
    y = m["home_margin"].to_numpy(dtype=float)
    w = m["home_win"].to_numpy(dtype=float)
    veg_ae = np.abs(m["vegas_home_margin"] - m["home_margin"]).to_numpy()
    ok = m["vegas_home_margin"].notna().to_numpy()

    base_pred, base_sig = expanding_pred(m, BASE)
    valid = ~np.isnan(base_pred)
    base_ae = np.abs(base_pred - y)
    print(f"Test games 2010-2025: {valid.sum():,}")
    print(
        f"Baseline additive four factors: OOS MAE {base_ae[valid].mean():.4f}  "
        f"logloss {logloss(base_pred[valid], base_sig[valid], w[valid]):.4f}"
    )
    print(f"Vegas line (same games)       : MAE {veg_ae[valid & ok].mean():.4f}\n")

    print(
        f"{'metric':>8s}{'OOSdMAE':>9s}{'dlogloss':>10s}{'coef(pts)':>11s}"
        f"{'  | imbalance decile: base':>0s}"
    )
    print(f"{'':>38s}{'n':>6s}{'base':>7s}{'+feat':>7s}{'Vegas':>7s}{'closed%':>9s}")
    print("-" * 84)
    full = []
    for met in METRICS:
        col = met + "_diff"
        pred, sig = expanding_pred(m, [*BASE, col])
        v = ~np.isnan(pred)
        ae = np.abs(pred - y)
        dmae = ae[v].mean() - base_ae[v].mean()
        dll = logloss(pred[v], sig[v], w[v]) - logloss(base_pred[v], base_sig[v], w[v])
        coef = np.polyfit(m[col].to_numpy(dtype=float), y - base_pred, 1)[0] if v.any() else np.nan

        d = m[col].abs().to_numpy()
        thr = np.nanquantile(d[v], 0.9)
        hi = v & (d >= thr) & ok
        b = base_ae[hi].mean()
        fft = ae[hi].mean()
        vg = veg_ae[hi].mean()
        closed = 100 * (b - fft) / (b - vg) if (b - vg) > 0 else np.nan
        full.append((met, dmae, dll))
        print(
            f"{met:>8s}{dmae:>+9.4f}{dll:>+10.4f}{coef:>11.3f}"
            f"{hi.sum():>6d}{b:>7.2f}{fft:>7.2f}{vg:>7.2f}{closed:>8.1f}%"
        )

    print("\nBest by overall OOS MAE:", min(full, key=lambda r: r[1])[0])
    best = min(full, key=lambda r: r[1])[0]
    # combine best continuous value metric with starter count
    combo = list(dict.fromkeys([best, "starter", "mpg"]))
    cols = BASE + [c + "_diff" for c in combo]
    pred, sig = expanding_pred(m, cols)
    v = ~np.isnan(pred)
    print(
        f"Combined {combo}: OOS MAE {np.abs(pred-y)[v].mean():.4f} "
        f"(baseline {base_ae[v].mean():.4f}, Vegas {veg_ae[v & ok].mean():.4f})"
    )


if __name__ == "__main__":
    main()
