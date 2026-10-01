"""Master Results comparison: every model family on identical rows.

Seed-blended production form throughout, walk-forward, folds 2001-2025 with
train = seasons before the fold's validation year (matching the RNN harness
that produced data/features/rnn_ladder/). Row universe: games present in all
three RNN arm builds (the production universe), with the Vegas column on the
2007+ subset where closing lines exist.

Columns: M0 (seed), M1-M3 (+seed), RNN diff/ff/raw (+seed), Kalman-Margin
(scalar, per-fold retuned), Kalman-FF (four-factor DLM, per-fold retuned),
Vegas closing line.

Run from repo root:
    python3 scripts/results_master_table.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "data" / "features"
RNN_DIR = FEAT / "rnn_ladder"
W = 15
BLEND_K = 7.0


def _lin_fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def _lin_pred(coef, X):
    return np.column_stack([np.ones(len(X)), X]) @ coef


def ladder_preds() -> pd.DataFrame:
    d = np.load(FEAT / "rnn_sequences.npz", allow_pickle=True)
    yr = np.array([int(s[:4]) for s in d["season"]])
    ctx, ym = d["ctx"], d["y_margin"]
    wt, ng, gid = d["wintotal_diff"], d["ngames"], d["game_id"]
    f = d["form"]
    hfm, hb, hfv, afm, ab, afv = (f[:, i] for i in range(6))

    # Rows where either team has fewer than five prior games carry zero-FILLED
    # form pieces (form_mean = form_var = 0 against a real base_comp), which is
    # not a neutral value: it fabricates a large negative deviation at full
    # shrinkage weight. Regression columns are therefore fit on rows with real
    # form only, and the form deviation is imputed to a true neutral zero at
    # prediction time on the short-history rows. See Session notes 2026-09-08.
    contam = (hfv == 0) | (afv == 0)

    rows = []
    for ty in range(2001, int(yr.max()) + 1):
        tr, te = yr < ty - 1, yr == ty
        if tr.sum() < 2000 or te.sum() < 100:
            continue
        trc = tr & ~contam
        sc = _lin_fit(wt[tr].reshape(-1, 1), ym[tr])
        seed = _lin_pred(sc, wt[te].reshape(-1, 1))

        fm = np.concatenate([hfm[trc], afm[trc]])
        fv = np.concatenate([hfv[trc], afv[trc]])
        n_eff = float(np.clip(np.nanmean(fv) / np.nanvar(fm), 1, W))
        sv = max(np.nanvar(fm) - np.nanmean(fv) / n_eff, 1e-6)
        sdd = (sv / (sv + hfv / n_eff)) * (hfm - hb) - (sv / (sv + afv / n_eff)) * (afm - ab)
        sdd = np.where(contam, 0.0, sdd)

        X1, X2 = ctx[:, 3:7], ctx
        X3 = np.column_stack([ctx, sdd])
        al = ng[te] / (ng[te] + BLEND_K)
        out = {"game_id": gid[te], "test_yr": ty, "margin": ym[te], "m0": seed}
        for name, X in (("m1", X1), ("m2", X2), ("m3", X3)):
            c = _lin_fit(X[trc], ym[trc])
            out[name] = (1 - al) * seed + al * _lin_pred(c, X[te])
        rows.append(pd.DataFrame(out))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    by_arm: dict[str, list[pd.DataFrame]] = {}
    for p in sorted(RNN_DIR.glob("*.parquet")):
        arm = p.stem.rsplit("_", 1)[0]
        df = pd.read_parquet(p, columns=["game_id", "rnn_blend_m"])
        by_arm.setdefault(arm, []).append(df)
    rnn = None
    for arm, parts in by_arm.items():
        a = pd.concat(parts, ignore_index=True).rename(columns={"rnn_blend_m": f"rnn_{arm}"})
        rnn = a if rnn is None else rnn.merge(a, on="game_id", how="inner")

    lad = ladder_preds()
    t = lad.merge(rnn, on="game_id", how="inner")
    t["gid10"] = t["game_id"].astype(str).str.zfill(10)

    for name, path in (
        ("kal_m", FEAT / "kalman_perfold_preds.parquet"),
        ("kal_ff", FEAT / "kalman_ff_perfold_preds.parquet"),
    ):
        k = pd.read_parquet(path, columns=["game_id", "pred_m"])
        k["gid10"] = k["game_id"].astype(str).str.zfill(10)
        t = t.merge(k[["gid10", "pred_m"]].rename(columns={"pred_m": name}), on="gid10")

    lines = pd.read_parquet(FEAT / "vegas_game_lines.parquet")[
        ["game_id", "vegas_home_margin"]
    ].dropna()
    lines["gid10"] = lines["game_id"].astype(str).str.zfill(10)
    t = t.merge(lines[["gid10", "vegas_home_margin"]], on="gid10", how="left")

    cols = ["m0", "m1", "m2", "m3", "rnn_diff", "rnn_ff", "rnn_raw", "kal_m", "kal_ff"]
    labels = {
        "m0": "M0 seed",
        "m1": "M1+seed",
        "m2": "M2+seed",
        "m3": "M3+seed",
        "rnn_diff": "RNN-diff",
        "rnn_ff": "RNN-ff",
        "rnn_raw": "RNN-raw",
        "kal_m": "Kal-Margin",
        "kal_ff": "Kal-FF",
    }
    print(
        f"universe: {len(t):,} games, {t.test_yr.nunique()} seasons "
        f"({t.test_yr.min()}-{t.test_yr.max()})"
    )

    print(f"\n{'season':>6s}" + "".join(f"{labels[c]:>11s}" for c in cols) + f"{'Vegas':>11s}")
    for ty, g in t.groupby("test_yr"):
        line = f"{ty:>6d}"
        for c in cols:
            line += f"{np.abs(g[c] - g.margin).mean():>11.3f}"
        v = g.dropna(subset=["vegas_home_margin"])
        veg = np.abs(v.vegas_home_margin - v.margin).mean() if len(v) > 200 else np.nan
        line += f"{veg:>11.3f}" if not np.isnan(veg) else f"{'n/a':>11s}"
        print(line)

    print(f"\n{'window':>18s}" + "".join(f"{labels[c]:>11s}" for c in cols) + f"{'Vegas':>11s}")
    by = t.groupby("test_yr")
    means_full = {c: by.apply(lambda g, c=c: np.abs(g[c] - g.margin).mean()).mean() for c in cols}
    line = f"{'2001+ (all rows)':>18s}" + "".join(f"{means_full[c]:>11.4f}" for c in cols)
    print(line + f"{'n/a':>11s}")

    v = t.dropna(subset=["vegas_home_margin"])
    v = v[v.test_yr >= 2007]
    byv = v.groupby("test_yr")
    means_v = {c: byv.apply(lambda g, c=c: np.abs(g[c] - g.margin).mean()).mean() for c in cols}
    veg = byv.apply(lambda g: np.abs(g.vegas_home_margin - g.margin).mean()).mean()
    line = f"{'2007+ (lined rows)':>18s}" + "".join(f"{means_v[c]:>11.4f}" for c in cols)
    print(line + f"{veg:>11.4f}")

    out = FEAT / "results_master_table.parquet"
    t.drop(columns=["gid10"]).to_parquet(out, index=False)
    print(f"\nrow-level table written to {out}")


if __name__ == "__main__":
    main()
