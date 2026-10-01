"""Temporal-ramp transplant, matched master harness (final thesis version).

Fits every arm inside the exact machinery of results_master_table.py (npz
training pool, yr < ty-1 folds, sdd shrinkage baseline, seed blend
al = n/(n+7)), restricted to rows where the 20-game ramp features exist,
scored per-fold-mean on the identical 27,984 rows as the published RNN and
Kalman references. Supersedes scripts/ramp_transplant.py's cross-harness
comparison, whose fuller training pool flattered the linear arms.

Run from repo root:
    PYTHONPATH=. python3 scripts/ramp_transplant_matched.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEAT = Path("data/features")
W, BLEND_K = 15, 7.0
HGRID = [3, 5, 8, 12, 20, 40]


def _fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def _pred(c, X):
    return np.column_stack([np.ones(len(X)), X]) @ c


def main() -> None:
    d = np.load(FEAT / "rnn_sequences.npz", allow_pickle=True)
    yr = np.array([int(s[:4]) for s in d["season"]])
    ctx, ym = d["ctx"], d["y_margin"]
    wt, ng, gid = d["wintotal_diff"], d["ngames"], d["game_id"]
    f = d["form"]
    hfm, hb, hfv, afm, ab, afv = (f[:, i] for i in range(6))

    wf = pd.read_parquet(FEAT / "ramp_transplant_forms.parquet")
    cols20 = ["flat20", "attn20"] + [f"geo{h}" for h in HGRID]
    mt = pd.read_parquet(FEAT / "modeling_table.parquet")[
        ["game_id", "team_id", "away_team_id", "base_comp", "away_base_comp"]
    ]
    m = mt.merge(
        wf.rename(columns={c: f"h_{c}" for c in cols20}), on=["game_id", "team_id"], how="left"
    )
    m = m.merge(
        wf.rename(columns={"team_id": "away_team_id", **{c: f"a_{c}" for c in cols20}}),
        on=["game_id", "away_team_id"],
        how="left",
    )
    devmap = {
        c: dict(
            zip(
                m.game_id,
                (m[f"h_{c}"] - m.base_comp) - (m[f"a_{c}"] - m.away_base_comp),
                strict=False,
            )
        )
        for c in cols20
    }
    DEV = {c: np.array([devmap[c].get(g, np.nan) for g in gid]) for c in cols20}

    master = pd.read_parquet(FEAT / "results_master_table.parquet")
    mref = master.dropna(subset=["m0", "m3", "rnn_diff", "kal_m"]).set_index("game_id")
    inref = np.array([g in mref.index for g in gid])
    present = ~np.isnan(DEV["flat20"])
    evalrow = present & inref

    arms = {"m3 sdd": [], "flat20": [], "geo": [], "attn": []}
    refs = {"rnn": [], "kal": [], "m3pub": []}
    for ty in range(2001, int(yr.max()) + 1):
        tr = (yr < ty - 1) & present
        te = (yr == ty) & evalrow
        if tr.sum() < 2000 or te.sum() < 100:
            continue
        sc = _fit(wt[tr].reshape(-1, 1), ym[tr])
        seed = _pred(sc, wt[te].reshape(-1, 1))
        fm = np.concatenate([hfm[tr], afm[tr]])
        fv = np.concatenate([hfv[tr], afv[tr]])
        n_eff = float(np.clip(np.nanmean(fv) / np.nanvar(fm), 1, W))
        sv = max(np.nanvar(fm) - np.nanmean(fv) / n_eff, 1e-6)
        sdd = (sv / (sv + hfv / n_eff)) * (hfm - hb) - (sv / (sv + afv / n_eff)) * (afm - ab)
        al = ng[te] / (ng[te] + BLEND_K)
        vy = yr[tr].max()
        tri = tr & (yr < vy)
        va = (yr == vy) & present
        scv = _fit(wt[tri].reshape(-1, 1), ym[tri])
        seedv = _pred(scv, wt[va].reshape(-1, 1))
        alv = ng[va] / (ng[va] + BLEND_K)
        best_h, best = None, np.inf
        for h in HGRID:
            X = np.column_stack([ctx, DEV[f"geo{h}"]])
            c = _fit(X[tri], ym[tri])
            mae = float(np.abs((1 - alv) * seedv + alv * _pred(c, X[va]) - ym[va]).mean())
            if mae < best:
                best, best_h = mae, h
        for name, feat in [
            ("m3 sdd", sdd),
            ("flat20", DEV["flat20"]),
            ("geo", DEV[f"geo{best_h}"]),
            ("attn", DEV["attn20"]),
        ]:
            X = np.column_stack([ctx, feat])
            c = _fit(X[tr], ym[tr])
            arms[name].append(np.abs((1 - al) * seed + al * _pred(c, X[te]) - ym[te]).mean())
        sub = mref.loc[gid[te]]
        refs["rnn"].append(np.abs(sub.rnn_diff - sub.margin).mean())
        refs["kal"].append(np.abs(sub.kal_m - sub.margin).mean())
        refs["m3pub"].append(np.abs(sub.m3 - sub.margin).mean())

    out = {
        "published M3": refs["m3pub"],
        "in-harness sdd baseline": arms["m3 sdd"],
        "flat-20": arms["flat20"],
        "geo ramp": arms["geo"],
        "attention ramp": arms["attn"],
        "published RNN-diff": refs["rnn"],
        "published Kalman": refs["kal"],
    }
    for n, v in out.items():
        print(f"  {n:24s} {np.mean(v):.4f}")
    for a in ["flat20", "geo", "attn"]:
        fd = np.array(arms["m3 sdd"]) - np.array(arms[a])
        t, p = stats.ttest_1samp(fd, 0)
        print(
            f"{a:8s} vs sdd: {fd.mean():+.4f}, t={t:.2f}, p={p:.4f}, "
            f"better {int((fd > 0).sum())}/{len(fd)}"
        )
    fd = np.array(refs["rnn"]) - np.array(arms["attn"])
    t, p = stats.ttest_1samp(fd, 0)
    print(f"attn vs RNN: {fd.mean():+.4f}, t={t:.2f}, p={p:.4f}")


if __name__ == "__main__":
    main()
