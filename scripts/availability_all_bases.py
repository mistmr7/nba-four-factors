"""Availability correction applied to every base model (base-agnostic test).

Same correction harness as the production availability feature (intercept and
beta fit per fold on prior out-of-sample residuals, frozen through the test
season, reliability-shrunk Game Score value-out differential), applied to each
base model's predictions from the master table. Scored on the identical
Vegas-lined rows where every base has a prediction, folds 2007-2025.

Run from repo root:
    PYTHONPATH=. python3 scripts/availability_all_bases.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_four_factors.features.availability_shrunk import player_gmsc, shrunk_diff
from nba_four_factors.player_state.followup import _apply, _fit_beta

FEAT = Path("data/features")
BASES = [("m0", "M0 (seed)"), ("m2", "M2"), ("m3", "M3"),
         ("rnn_diff", "RNN-diff"), ("rnn_ff", "RNN-ff"),
         ("rnn_raw", "RNN-raw"), ("kal_m", "Kalman")]


def main() -> None:
    inc = shrunk_diff(player_gmsc(), 12.0)
    inc["game_id"] = inc["game_id"].astype(str).str.zfill(10)
    m = pd.read_parquet(FEAT / "results_master_table.parquet")
    m["game_id"] = m["game_id"].astype(str).str.zfill(10)
    t = m.dropna(subset=[b for b, _ in BASES] + ["vegas_home_margin"])
    t = t.merge(inc, on="game_id", how="left")
    t["gmscs_diff"] = t["gmscs_diff"].fillna(0.0)
    print(f"identical lined rows: {len(t)}")
    print(f"{'base':11s} {'base MAE':>9s} {'+avail':>8s} {'closed':>7s} "
          f"{'mean beta':>10s} {'helped':>8s}")
    for base, lbl in BASES:
        tb = t.rename(columns={base: "pred_m"})
        rows, helped, nf, betas = [], 0, 0, []
        for ty in range(2007, int(tb.test_yr.max()) + 1):
            tr = tb[(tb.test_yr >= 2005) & (tb.test_yr < ty)]
            te = tb[tb.test_yr == ty].copy()
            if len(te) < 200:
                continue
            te["a0"] = _apply(te, _fit_beta(tr, []), [])
            coef = _fit_beta(tr, ["gmscs_diff"])
            betas.append(coef[-1])
            te["a1"] = _apply(te, coef, ["gmscs_diff"])
            nf += 1
            if (te.a1 - te.margin).abs().mean() < (te.a0 - te.margin).abs().mean():
                helped += 1
            rows.append(te)
        d = pd.concat(rows)
        k = (d.a0 - d.margin).abs().mean()
        a = (d.a1 - d.margin).abs().mean()
        v = (d.vegas_home_margin - d.margin).abs().mean()
        print(f"{lbl:11s} {k:9.4f} {a:8.4f} {100 * (k - a) / (k - v):6.1f}% "
              f"{np.mean(betas):10.3f} {helped:>5d}/{nf}")


if __name__ == "__main__":
    main()
