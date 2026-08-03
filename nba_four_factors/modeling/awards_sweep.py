"""Do accolade labels add anything over the Game-Score availability weighting?

Compares, expanding-window 2010-2025 on the additive four-factor model:
  baseline, + raw Game Score, + shrunk Game Score (current best), + accolade
  scores (1-year and 3-year-max lag, All-Star count, All-NBA count), and the
  combination of shrunk Game Score with the 3-year accolade score.

All are scored overall and on the same high-absence stratum (the top decile of
|shrunk Game Score differential|, the established "a real player is out" games),
so the question is clean: on the games that matter, does a star label beat or add
to valuing the absence by box-score production?

Run: python -m nba_four_factors.modeling.awards_sweep
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.availability_shrunk import player_gmsc, shrunk_diff
from .availability_sweep import BASE, LINES, REPO, TABLE, expanding_pred, logloss

FEAT = REPO / "data" / "features"


def main():
    m = pd.read_parquet(TABLE).dropna(subset=["home_margin", "home_win"]).copy()
    m["yr"] = m["season"].str[:4].astype(int)
    gm = pd.read_parquet(FEAT / "availability_features.parquet")[["game_id", "gmsc_diff"]]
    ac = pd.read_parquet(FEAT / "availability_accolades.parquet")[
        ["game_id", "acc1_diff", "acc3_diff", "star_diff", "allnba_diff"]
    ]
    sh = shrunk_diff(player_gmsc(), 12.0)
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    m = (
        m.merge(gm, on="game_id", how="left")
        .merge(ac, on="game_id", how="left")
        .merge(sh, on="game_id", how="left")
        .merge(lines, on="game_id", how="left")
    )
    feat = ["gmsc_diff", "gmscs_diff", "acc1_diff", "acc3_diff", "star_diff", "allnba_diff"]
    m[BASE + feat] = m[BASE + feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    m = m[m["yr"] >= 2010].reset_index(drop=True)

    y = m["home_margin"].to_numpy(float)
    w = m["home_win"].to_numpy(float)
    veg = np.abs(m["vegas_home_margin"] - m["home_margin"]).to_numpy()
    ok = m["vegas_home_margin"].notna().to_numpy()

    bp, bs = expanding_pred(m, BASE)
    v = ~np.isnan(bp)
    base_ae = np.abs(bp - y)
    # common high-absence stratum: top decile of shrunk Game Score imbalance
    d = m["gmscs_diff"].abs().to_numpy()
    hi = v & (d >= np.nanquantile(d[v], 0.9)) & ok

    print(
        f"Baseline four factors: OOS MAE {base_ae[v].mean():.4f}  ll {logloss(bp[v],bs[v],w[v]):.4f}"
    )
    print(
        f"Vegas (overall) {veg[v&ok].mean():.4f} | Vegas (high-absence decile) {veg[hi].mean():.4f}\n"
    )
    print(
        f"{'+ feature(s)':28s}{'OOS MAE':>9s}{'dMAE':>8s}{'dll':>9s}{'hiDec MAE':>10s}{'closed%':>9s}"
    )
    print("-" * 75)

    specs = [
        ("gmsc (raw)", ["gmsc_diff"]),
        ("gmsc shrunk (best so far)", ["gmscs_diff"]),
        ("accolade lag1", ["acc1_diff"]),
        ("accolade lag3-max", ["acc3_diff"]),
        ("All-Star count out", ["star_diff"]),
        ("All-NBA count out", ["allnba_diff"]),
        ("gmsc shrunk + accolade3", ["gmscs_diff", "acc3_diff"]),
        ("gmsc shrunk + allnba", ["gmscs_diff", "allnba_diff"]),
    ]
    for name, cols in specs:
        pred, sig = expanding_pred(m, BASE + cols)
        ae = np.abs(pred - y)
        dmae = ae[v].mean() - base_ae[v].mean()
        dll = logloss(pred[v], sig[v], w[v]) - logloss(bp[v], bs[v], w[v])
        b, ff, vg = base_ae[hi].mean(), ae[hi].mean(), veg[hi].mean()
        closed = 100 * (b - ff) / (b - vg) if (b - vg) > 0 else np.nan
        print(f"{name:28s}{ae[v].mean():>9.4f}{dmae:>+8.4f}{dll:>+9.4f}{ff:>10.3f}{closed:>8.1f}%")


if __name__ == "__main__":
    main()
