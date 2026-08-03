"""Architecture study: LSTM vs GRU at EQUAL parameters, across depths, with seeds.

Two methodological points the earlier random search missed:
  1. Equal parameters, not equal cells. A GRU cell has ~3/4 the parameters of an
     LSTM cell, so comparing them at the same width is unfair. For each GRU width
     we solve the LSTM width that matches total RNN parameters.
  2. Multiple seeds. Neural-net training is stochastic; a single run's
     differences are mostly noise. Each config runs SEEDS times and we report
     mean +/- sd, so a real architecture effect has to clear the seed band.

Reports test margin MAE (RNN alone, no seed blend, since we're comparing
networks) on the train<=2017 / test>=2021 split. Long run: GRID x SEEDS fits.
Run: python -m nba_four_factors.rnn.arch_search
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from .train_keras import CONFIG, feat_count, load, make_model, metrics, split_idx, standardize

GRU_WIDTHS = [64, 96, 128]
DEPTHS = [1, 2, 3]
SEEDS = [0, 1, 2]


def lstm_units_matching(gru_units: int, F: int) -> int:
    """LSTM width with ~the same per-layer parameter count as GRU(gru_units).

    GRU ~ 3*u*(F+u+1) params; LSTM ~ 4*u*(F+u+1). Solve 4 u^2 + 4(F+1)u = target.
    """
    target = 3 * gru_units * (F + gru_units + 1)
    a, b, c = 4.0, 4.0 * (F + 1), -target
    return max(1, round((-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)))


def run_one(d, rnn_type, units, depth, seed):
    tf.keras.utils.set_random_seed(seed)
    cfg = {**CONFIG, "rnn_type": rnn_type, "units": units, "layers_n": depth}
    L = min(cfg["seq_len"], int(d["L"]))
    F = feat_count(d, cfg)
    home, away = d["home_seq"][:, -L:, :], d["away_seq"][:, -L:, :]
    hm, am = d["home_mask"][:, -L:], d["away_mask"][:, -L:]
    tr, va, te = split_idx(d["season"], cfg["train_max_year"], cfg["val_years"])
    H, A = standardize(home, away, hm, am, tr, F)
    ctx = d["ctx"]
    ctx = (ctx - ctx[tr].mean(0)) / (ctx[tr].std(0) + 1e-6)
    ym, yw = d["y_margin"], d["y_win"]
    model = make_model(cfg, L, F, ctx.shape[1])
    params = model.count_params()
    from tensorflow.keras import callbacks

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
    pm, pw = (v.ravel() for v in model.predict([H[te], A[te], ctx[te]], verbose=0))
    mae, ll, _ = metrics(pm, pw, ym[te], yw[te])
    return mae, ll, params


def main():
    d = load()
    F = feat_count(d, CONFIG)
    print(
        f"{'arch':6s}{'depth':>6s}{'units':>6s}{'params':>9s}{'MAE mean±sd':>16s}{'logloss mean±sd':>18s}"
    )
    print("-" * 62)
    for gw in GRU_WIDTHS:
        lw = lstm_units_matching(gw, F)
        for rnn_type, u in [("GRU", gw), ("LSTM", lw)]:
            for depth in DEPTHS:
                res = [run_one(d, rnn_type, u, depth, s) for s in SEEDS]
                mae = np.array([r[0] for r in res])
                ll = np.array([r[1] for r in res])
                params = res[0][2]
                print(
                    f"{rnn_type:6s}{depth:>6d}{u:>6d}{params:>9d}"
                    f"{mae.mean():>9.3f}±{mae.std():.3f}{ll.mean():>11.4f}±{ll.std():.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
