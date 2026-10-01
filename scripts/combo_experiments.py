"""Kalman-M3 combination experiments (Table 7 and Appendix C1), reproducible.

Rebuilds the prediction-level combination arms and the seed-substitution
arms from the master table and the Kalman fold outputs. Run with
--m3 v1 to validate against the previously published numbers (defective
M3 construction, results_master_table_v1_defective.parquet) and with
--m3 v2 (default) for the corrected values.

Table 7 arms (24 test seasons, 2002-2025; fitted combos need one completed
test season): Kalman alone, M3 alone, 50/50 average, precision-weighted
(per-game filter predictive variance vs per-fold prior OOS MSE for M3),
stacked OLS (fit on prior test seasons' out-of-sample predictions).

C1 arms (25 seasons): seed alone, gated production (seed through average
five games, then (1-alpha) seed + alpha M3), drop-in (filter in the seed
slot), Kalman alone; early window = average games <= 10.

Run from repo root:
    PYTHONPATH=. python3 scripts/combo_experiments.py [--m3 v1|v2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEAT = Path("data/features")
BLEND_K = 7.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", choices=["v1", "v2"], default="v2")
    a = ap.parse_args()
    src = (
        "results_master_table_v1_defective.parquet"
        if a.m3 == "v1"
        else "results_master_table.parquet"
    )
    t = pd.read_parquet(FEAT / src)
    t = t.dropna(subset=["m0", "m3", "kal_m", "rnn_diff", "rnn_ff", "rnn_raw", "kal_ff"])

    d = np.load(FEAT / "rnn_sequences.npz", allow_pickle=True)
    ngmap = dict(zip(d["game_id"], d["ngames"], strict=False))
    t["ng"] = t.game_id.map(ngmap)

    kv = pd.read_parquet(FEAT / "kalman_perfold_preds.parquet")[["game_id", "pred_v"]]
    kv["game_id"] = kv.game_id.astype(str).str.zfill(10)
    t["game_id"] = t.game_id.astype(str).str.zfill(10)
    t = t.merge(kv, on="game_id", how="left")

    years = sorted(t.test_yr.unique())

    # ---- Table 7 arms, 24 seasons ----
    arms = {k: [] for k in ["kal", "m3", "avg", "prec", "stack", "fas"]}
    wlog = []
    for ty in years[1:]:
        tr = t[t.test_yr < ty]
        te = t[t.test_yr == ty]
        arms["kal"].append(np.abs(te.kal_m - te.margin).mean())
        arms["m3"].append(np.abs(te.m3 - te.margin).mean())
        arms["avg"].append(np.abs(0.5 * te.kal_m + 0.5 * te.m3 - te.margin).mean())
        mse_m3 = float(((tr.m3 - tr.margin) ** 2).mean())
        w = (1.0 / te.pred_v) / (1.0 / te.pred_v + 1.0 / mse_m3)
        wlog.append(float(w.mean()))
        arms["prec"].append(np.abs(w * te.kal_m + (1 - w) * te.m3 - te.margin).mean())
        X = np.column_stack([np.ones(len(tr)), tr.kal_m, tr.m3])
        c, *_ = np.linalg.lstsq(X, tr.margin.to_numpy(), rcond=None)
        Xt = np.column_stack([np.ones(len(te)), te.kal_m, te.m3])
        arms["stack"].append(np.abs(Xt @ c - te.margin).mean())
        al = te.ng / (te.ng + BLEND_K)
        gate = te.ng > 5
        fas = np.where(gate, (1 - al) * te.kal_m + al * te.m3, te.kal_m)
        arms["fas"].append(np.abs(fas - te.margin).mean())

    print(f"Table 7 arms ({a.m3}), 24 seasons, per-fold means:")
    kalf = np.array(arms["kal"])
    for name, lbl in [
        ("kal", "Kalman alone"),
        ("m3", "M3 alone"),
        ("avg", "50/50 average"),
        ("prec", "Precision-weighted"),
        ("stack", "Stacked OLS"),
        ("fas", "Filter as seed"),
    ]:
        v = np.array(arms[name])
        beats = "-" if name == "kal" else f"{int((v < kalf).sum())}/24"
        print(f"  {lbl:20s} {v.mean():.4f}   seasons > filter: {beats}")
    print(f"  mean precision weight on filter: {np.mean(wlog):.3f}")
    ekal = np.concatenate(
        [(t[t.test_yr == ty].kal_m - t[t.test_yr == ty].margin) for ty in years[1:]]
    )
    em3 = np.concatenate([(t[t.test_yr == ty].m3 - t[t.test_yr == ty].margin) for ty in years[1:]])
    print(f"  error correlation kal vs m3: {np.corrcoef(ekal, em3)[0, 1]:.3f}")

    # ---- C1 arms, 25 seasons ----
    al = t.ng / (t.ng + BLEND_K)
    gate = t.ng > 5
    prod = np.where(gate, (1 - al) * t.m0 + al * t.m3, t.m0)
    drop = np.where(gate, (1 - al) * t.kal_m + al * t.m3, t.kal_m)
    early = t.ng <= 10
    print(f"\nC1 arms ({a.m3}), 25 seasons:")
    prod_f = pd.Series(np.abs(prod - t.margin)).groupby(t.test_yr.values).mean()
    drop_f = pd.Series(np.abs(drop - t.margin)).groupby(t.test_yr.values).mean()
    kal_f = (t.kal_m - t.margin).abs().groupby(t.test_yr).mean()
    seed_f = (t.m0 - t.margin).abs().groupby(t.test_yr).mean()
    for lbl, f, e in [
        ("Seed alone", seed_f, np.abs(t.m0 - t.margin)[early].mean()),
        ("Production", prod_f, np.abs(prod - t.margin)[early.values].mean()),
        ("Drop-in", drop_f, np.abs(drop - t.margin)[early.values].mean()),
        ("Kalman alone", kal_f, np.abs(t.kal_m - t.margin)[early].mean()),
    ]:
        beats = f"{int((f.values < kal_f.values).sum())}/25" if lbl != "Kalman alone" else "-"
        print(f"  {lbl:14s} MAE {f.mean():.4f}   early {e:.4f}   seasons > filter: {beats}")
    dp = pd.Series(np.abs(drop - t.margin)).groupby(t.test_yr.values).mean()
    pr = pd.Series(np.abs(prod - t.margin)).groupby(t.test_yr.values).mean()
    print(
        f"  drop-in beats production: {int((dp.values < pr.values).sum())}/25, "
        f"delta {pr.mean() - dp.mean():+.4f}"
    )


if __name__ == "__main__":
    main()
