"""Evidence numbers for Results 5.12 on the 2007+ identical-rows universe.

Replicates the Part B fold protocol (train pool test_yr >= 2005, test folds
2007+, Vegas-lined rows, >= 200-row fold gate) for three arms:

    A0  Kalman + intercept-only correction (no availability feature)
    A1  Kalman + a + beta * shrunk Game Score value-out differential
    Vegas closing line

Outputs:
    data/features/avail_512_decile.csv    MAE by |value-out diff| decile
    data/features/avail_512_season.csv    per-season MAE and gap closed

Run from repo root:
    python3 scripts/avail_512_evidence.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_four_factors.player_state.followup import _apply, _fit_beta, team_features

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "data" / "features"

ERAS = [(2007, 2012), (2013, 2019), (2020, 2025)]


def main() -> None:
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
    d["imb"] = d.gmscs_diff.abs()

    # Deciles of absence imbalance, pooled 2007+.
    d["dec"] = pd.qcut(d.imb.rank(method="first"), 10, labels=range(1, 11))
    dec = (
        d.groupby("dec", observed=True)
        .agg(
            n=("ae0", "size"),
            imb_lo=("imb", "min"),
            imb_hi=("imb", "max"),
            kalman=("ae0", "mean"),
            avail=("ae1", "mean"),
            vegas=("aev", "mean"),
        )
        .reset_index()
    )
    dec["closed_pct"] = 100 * (dec.kalman - dec.avail) / (dec.kalman - dec.vegas)
    dec.to_csv(FEAT / "avail_512_decile.csv", index=False)
    print(dec.round(3).to_string(index=False))

    # Per-season and per-era gap closed.
    sea = (
        d.groupby("test_yr")
        .agg(
            n=("ae0", "size"),
            kalman=("ae0", "mean"),
            avail=("ae1", "mean"),
            vegas=("aev", "mean"),
        )
        .reset_index()
    )
    sea["closed_pct"] = 100 * (sea.kalman - sea.avail) / (sea.kalman - sea.vegas)
    sea.to_csv(FEAT / "avail_512_season.csv", index=False)
    print("\n" + sea.round(3).to_string(index=False))

    print("\nera summary:")
    for lo, hi in ERAS:
        e = d[(d.test_yr >= lo) & (d.test_yr <= hi)]
        k, a, v = e.ae0.mean(), e.ae1.mean(), e.aev.mean()
        print(
            f"  {lo}-{hi}: kalman {k:.4f}  +avail {a:.4f}  vegas {v:.4f}  "
            f"closed {100 * (k - a) / (k - v):.1f}%"
        )
    k, a, v = d.ae0.mean(), d.ae1.mean(), d.aev.mean()
    print(
        f"  overall: kalman {k:.4f}  +avail {a:.4f}  vegas {v:.4f}  "
        f"closed {100 * (k - a) / (k - v):.1f}%  (n={len(d):,})"
    )


if __name__ == "__main__":
    main()
