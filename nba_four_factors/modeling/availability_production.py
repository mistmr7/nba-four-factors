"""Layer the winning availability feature onto the best model (the scalar Kalman).

The Kalman filter tracks strength from margins and the seed but is blind to who is
playing tonight. We add the reliability-shrunk Game-Score availability differential
as a correction to its margin prediction:

    pred = kalman_margin + beta * (home_value_out - away_value_out)

with beta fit once on the pre-2010 seasons (the same block the filter's noise was
tuned on) and frozen. We then evaluate 2010-2025 and ask how much of the real gap
to the closing line this closes, overall and on the games where a real
availability imbalance exists.

Run: python -m nba_four_factors.modeling.availability_production
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.availability_shrunk import player_gmsc, shrunk_diff
from .statespace_scalar import load, run_season, seed_fit, tune

REPO = Path(__file__).resolve().parents[2]
LINES = REPO / "data" / "features" / "vegas_game_lines.parquet"
K_SHRINK = 12.0


def kalman_per_game():
    m = load()
    pre = m[m["yr"] < 2010]
    slope, mean_wt, hca = seed_fit(pre)
    params, _ = tune(m, slope, mean_wt, hca)
    out = []
    for ty in range(2005, m["yr"].max() + 1):
        g = m[m["yr"] == ty]
        if len(g) < 100:
            continue
        pm, pv, z, w = run_season(g, params, slope, mean_wt, hca)
        d = g[["game_id", "yr", "home_margin", "home_win"]].copy()
        d["kal"] = pm
        out.append(d)
    return pd.concat(out, ignore_index=True)


def main():
    f = kalman_per_game()
    sd = shrunk_diff(player_gmsc(), K_SHRINK)
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    f = f.merge(sd, on="game_id", how="left").merge(lines, on="game_id", how="left")
    f["gmscs_diff"] = f["gmscs_diff"].fillna(0.0)

    y = f["home_margin"].to_numpy(dtype=float)
    av = f["gmscs_diff"].to_numpy(dtype=float)
    kal = f["kal"].to_numpy(dtype=float)
    veg = np.abs(f["vegas_home_margin"] - f["home_margin"]).to_numpy()
    ok = f["vegas_home_margin"].notna().to_numpy()

    # Fit the availability coefficient on pre-2010 residuals, then freeze.
    pre = f["yr"] < 2010
    r = y[pre] - kal[pre]
    A = np.column_stack([np.ones(pre.sum()), av[pre]])
    a, beta = np.linalg.lstsq(A, r, rcond=None)[0]
    print(
        f"availability coefficient beta = {beta:.4f} points per Game-Score unit (intercept {a:+.3f})"
    )
    print("  (negative as expected: more value out for the home team lowers its margin)\n")

    test = (f["yr"] >= 2010).to_numpy()
    pred_adj = kal + a + beta * av
    kal_ae = np.abs(kal - y)
    adj_ae = np.abs(pred_adj - y)

    t = test & ok
    print(f"Test games 2010-2025: {t.sum():,}")
    print(f"  Kalman alone        : MAE {kal_ae[t].mean():.4f}")
    print(
        f"  Kalman + availability: MAE {adj_ae[t].mean():.4f}   ({adj_ae[t].mean()-kal_ae[t].mean():+.4f})"
    )
    print(f"  Vegas line          : MAE {veg[t].mean():.4f}")
    closed_all = 100 * (kal_ae[t].mean() - adj_ae[t].mean()) / (kal_ae[t].mean() - veg[t].mean())
    print(f"  -> closes {closed_all:.1f}% of the overall gap to Vegas\n")

    d = np.abs(av)
    hi = t & (d >= np.nanquantile(d[t], 0.9))
    b, ff, vg = kal_ae[hi].mean(), adj_ae[hi].mean(), veg[hi].mean()
    print(f"On the availability-imbalance decile ({hi.sum():,} games):")
    print(f"  Kalman {b:.3f}  ->  +avail {ff:.3f}   (Vegas {vg:.3f})")
    print(f"  -> closes {100*(b-ff)/(b-vg):.1f}% of the gap to Vegas on those games")

    mid = t & (d >= np.nanquantile(d[t], 0.5)) & (d < np.nanquantile(d[t], 0.9))
    low = t & (d < np.nanquantile(d[t], 0.5))
    print("\n  effect by imbalance level (MAE change, Kalman -> +avail):")
    for nm, mask in [
        ("balanced (bottom 50%)", low),
        ("moderate (50-90%)", mid),
        ("imbalanced (top 10%)", hi),
    ]:
        print(
            f"    {nm:22s} {kal_ae[mask].mean():.3f} -> {adj_ae[mask].mean():.3f}  ({adj_ae[mask].mean()-kal_ae[mask].mean():+.3f})"
        )


if __name__ == "__main__":
    main()
