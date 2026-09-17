"""Expanding-window evaluation: RNN+seed vs the full production regression (M3+seed).

This is the paper-grade comparison. For each test season it trains the RNN on all
earlier seasons (validating on the season just before, for early stopping) and
scores the season, then averages, the same protocol behind the regression's 9.87
figure, so the two are finally on equal footing.

It is a long run: one RNN is trained per test season (~minutes each on CPU), so
the default 2010-2025 span is ~16 fits. Lower START_TEST_YEAR for more folds or
raise it for a quicker pass. Run locally:
    python -m nba_four_factors.rnn.expanding_window_keras
"""

from __future__ import annotations

import numpy as np
from tensorflow.keras import callbacks

from .train_keras import (
    CONFIG,
    W,
    _lin_fit,
    _lin_pred,
    _phi,
    feat_count,
    load,
    make_model,
    metrics,
    standardize,
)

START_TEST_YEAR = 2010  # first season to test on; folds run this..latest


def run():
    d = load()
    cfg = dict(CONFIG)
    L = min(cfg["seq_len"], int(d["L"]))
    F = feat_count(d, cfg)
    yr = np.array([int(s[:4]) for s in d["season"]])
    home, away = d["home_seq"][:, -L:, :], d["away_seq"][:, -L:, :]
    hm, am = d["home_mask"][:, -L:], d["away_mask"][:, -L:]
    ctx0, ym, yw, wt, ng = d["ctx"], d["y_margin"], d["y_win"], d["wintotal_diff"], d["ngames"]
    f = d["form"]
    hfm, hb, hfv, afm, ab, afv = (f[:, i] for i in range(6))

    rows = []
    for ty in range(START_TEST_YEAR, yr.max() + 1):
        tr, va, te = yr < ty - 1, yr == ty - 1, yr == ty
        if tr.sum() < 2000 or te.sum() < 100 or va.sum() < 100:
            continue
        H, A = standardize(home, away, hm, am, tr, F)
        ctx = (ctx0 - ctx0[tr].mean(0)) / (ctx0[tr].std(0) + 1e-6)

        model = make_model(cfg, L, F, ctx.shape[1])
        es = callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["patience"], restore_best_weights=True
        )
        model.fit(
            [H[tr], A[tr], ctx[tr]],
            {"margin": ym[tr], "win": yw[tr]},
            validation_data=([H[va], A[va], ctx[va]], {"margin": ym[va], "win": yw[va]}),
            epochs=cfg["epochs"],
            batch_size=cfg["batch"],
            callbacks=[es],
            verbose=0,
        )
        rnn_m, rnn_w = (v.ravel() for v in model.predict([H[te], A[te], ctx[te]], verbose=0))

        # Seed + full production M3 (context + shrunk recent form), train-only shrinkage.
        sc = _lin_fit(wt[tr].reshape(-1, 1), ym[tr])
        ssig = float(np.std(ym[tr] - _lin_pred(sc, wt[tr].reshape(-1, 1)))) + 1e-6
        seed_m = _lin_pred(sc, wt[te].reshape(-1, 1))
        seed_w = _phi(seed_m, ssig)
        # Rows where either team has fewer than five prior games carry zero-FILLED
        # form pieces; fit shrinkage stats and the regression on real-form rows only
        # and impute the deviation to a true neutral zero elsewhere (2026-09-08 fix,
        # mirrors scripts/results_master_table.py).
        _contam = (hfv == 0) | (afv == 0)
        _trc = tr & ~_contam
        fm = np.concatenate([hfm[_trc], afm[_trc]])
        fv = np.concatenate([hfv[_trc], afv[_trc]])
        n_eff = float(np.clip(np.nanmean(fv) / np.nanvar(fm), 1, W))
        sv = max(np.nanvar(fm) - np.nanmean(fv) / n_eff, 1e-6)
        sdd = (sv / (sv + hfv / n_eff)) * (hfm - hb) - (sv / (sv + afv / n_eff)) * (afm - ab)
        sdd = np.where(_contam, 0.0, sdd)
        X3 = np.column_stack([ctx, sdd])
        r3 = _lin_fit(X3[_trc], ym[_trc])
        rsig = float(np.std(ym[_trc] - _lin_pred(r3, X3[_trc]))) + 1e-6
        reg3_m = _lin_pred(r3, X3[te])
        reg3_w = _phi(reg3_m, rsig)

        al = ng[te] / (ng[te] + cfg["blend_k"])
        rnn_bm, rnn_bw = (1 - al) * seed_m + al * rnn_m, (1 - al) * seed_w + al * rnn_w
        reg_bm, reg_bw = (1 - al) * seed_m + al * reg3_m, (1 - al) * seed_w + al * reg3_w
        rmae, rll, _ = metrics(rnn_bm, rnn_bw, ym[te], yw[te])
        gmae, gll, _ = metrics(reg_bm, reg_bw, ym[te], yw[te])
        rows.append((ty, te.sum(), rmae, rll, gmae, gll))
        print(
            f"  {ty}  n={int(te.sum()):4d}  RNN+seed MAE {rmae:6.3f} ll {rll:.4f}  |  M3+seed MAE {gmae:6.3f} ll {gll:.4f}",
            flush=True,
        )

    r = np.array([[x[2], x[3], x[4], x[5]] for x in rows])
    print("\nExpanding-window averages over", len(rows), "test seasons:")
    print(f"  RNN + seed : MAE {r[:,0].mean():.3f}  logloss {r[:,1].mean():.4f}")
    print(f"  M3  + seed : MAE {r[:,2].mean():.3f}  logloss {r[:,3].mean():.4f}")
    print(f"  RNN minus M3 (MAE): {r[:,0].mean()-r[:,2].mean():+.3f}  (negative = RNN better)")
    print(f"  seasons RNN beats M3 on MAE: {int((r[:,0]<r[:,2]).sum())}/{len(rows)}")


if __name__ == "__main__":
    run()
