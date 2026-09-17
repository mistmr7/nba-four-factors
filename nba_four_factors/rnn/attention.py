"""Attention-augmented Siamese RNN, the thesis's interpretable contribution.

The earlier RNN pooled each team's 20-game encoder sequence by a fixed rule
(average or last). Here we replace that with a temporal attention layer: the model
learns a weight for every one of the team's recent games and forms a weighted
summary. Two payoffs:

  1. Adaptive weighting. Instead of the regression's fixed 15-game window with
     reliability shrinkage, the network decides for itself how far back to look and
     how much to trust each game.
  2. Interpretability. The attention weights are extractable, so we can read off the
     learned recency profile (which positions in the last 20 games the model leans
     on) and how concentrated it is. That is the academic novelty, independent of
     whether it moves accuracy (the convergence result says it likely will not much).

It trains on the same split as train_keras, reports margin MAE and win log loss
against the seed blend and the M3+seed regression, then extracts the attention
weights on the test set, prints the average recency profile, and writes it to
data/features/attention_profile.csv for plotting.

Needs TensorFlow; run locally:
    python -m nba_four_factors.rnn.attention
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

from .train_keras import (
    CONFIG,
    W,
    _lin_fit,
    _lin_pred,
    _phi,
    feat_count,
    load,
    metrics,
    split_idx,
    standardize,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "features" / "attention_profile.csv"
ATTN_UNITS = 32


class TemporalAttention(layers.Layer):
    """Additive (Bahdanau-style) attention pooling over the time axis.

    Returns the context vector (weighted sum of the sequence) and the attention
    weights (batch, L, 1), with padded positions masked out before the softmax.
    """

    def __init__(self, units=ATTN_UNITS, **kw):
        super().__init__(**kw)
        self.W = layers.Dense(units, activation="tanh")
        self.V = layers.Dense(1)

    def call(self, seq, mask):
        e = self.V(self.W(seq))  # (b, L, 1)
        e = e + tf.expand_dims((1.0 - mask) * -1e9, axis=-1)  # mask padding
        a = tf.nn.softmax(e, axis=1)  # (b, L, 1)
        ctx = tf.reduce_sum(a * seq, axis=1)  # (b, H)
        return ctx, a

    def get_config(self):
        return {**super().get_config(), "units": self.W.units}


def build(cfg, L, F, ctx_dim):
    RNN = layers.LSTM if cfg["rnn_type"] == "LSTM" else layers.GRU
    masking = layers.Masking(0.0)
    rnns = []
    for _ in range(cfg["layers_n"]):
        r = RNN(cfg["units"], return_sequences=True, dropout=cfg["dropout"])
        rnns.append(layers.Bidirectional(r) if cfg["bidirectional"] else r)
    attn = TemporalAttention(cfg.get("attn_units", ATTN_UNITS))
    pad_mask = layers.Lambda(lambda t: tf.cast(tf.reduce_max(tf.abs(t), axis=-1) > 0, tf.float32))

    def encode(inp):
        x = masking(inp)
        for r in rnns:
            x = r(x)
        ctx, a = attn(x, pad_mask(inp))
        return ctx, a

    h_in, a_in, c_in = layers.Input((L, F)), layers.Input((L, F)), layers.Input((ctx_dim,))
    h, h_attn = encode(h_in)
    aw, a_attn = encode(a_in)
    z = layers.Concatenate()([h, aw, layers.Subtract()([h, aw]), c_in])
    z = layers.Dropout(cfg["dropout"])(layers.Dense(cfg["dense"], activation="relu")(z))
    model = models.Model(
        [h_in, a_in, c_in],
        [layers.Dense(1, name="margin")(z), layers.Dense(1, activation="sigmoid", name="win")(z)],
    )
    model.compile(
        optimizer=optimizers.Adam(cfg["lr"]),
        loss={"margin": "mse", "win": "binary_crossentropy"},
        loss_weights={"margin": 1.0, "win": cfg["loss_weight_win"]},
    )
    attn_model = models.Model([h_in, a_in, c_in], [h_attn, a_attn])  # weights for interpretation
    return model, attn_model


def run():
    d = load()
    cfg = {**CONFIG, "pooling": "attention", "attn_units": ATTN_UNITS}
    L = min(cfg["seq_len"], int(d["L"]))
    F = feat_count(d, cfg)
    home, away = d["home_seq"][:, -L:, :], d["away_seq"][:, -L:, :]
    hm, am = d["home_mask"][:, -L:], d["away_mask"][:, -L:]
    tr, va, te = split_idx(d["season"], cfg["train_max_year"], cfg["val_years"])
    H, A = standardize(home, away, hm, am, tr, F)
    ctx = d["ctx"]
    ctx = (ctx - ctx[tr].mean(0)) / (ctx[tr].std(0) + 1e-6)
    ym, yw, wt, ng = d["y_margin"], d["y_win"], d["wintotal_diff"], d["ngames"]

    model, attn_model = build(cfg, L, F, ctx.shape[1])
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
        verbose=int(os.environ.get("RNN_VERBOSE", "1")),
    )
    pm, pw = (x.ravel() for x in model.predict([H[te], A[te], ctx[te]], verbose=0))

    # seed blend + M3+seed comparator (numpy, same as train_keras)
    sc = _lin_fit(wt[tr].reshape(-1, 1), ym[tr])
    sig_s = float(np.std(ym[tr] - _lin_pred(sc, wt[tr].reshape(-1, 1)))) + 1e-6
    seed_m = _lin_pred(sc, wt[te].reshape(-1, 1))
    seed_w = _phi(seed_m, sig_s)
    f = d["form"]
    hfm, hb, hfv, afm, ab, afv = (f[:, i] for i in range(6))
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
    sig3 = float(np.std(ym[_trc] - _lin_pred(r3, X3[_trc]))) + 1e-6
    reg3_m = _lin_pred(r3, X3[te])
    reg3_w = _phi(reg3_m, sig3)
    al = ng[te] / (ng[te] + cfg["blend_k"])
    attn_bm, attn_bw = (1 - al) * seed_m + al * pm, (1 - al) * seed_w + al * pw
    reg_bm, reg_bw = (1 - al) * seed_m + al * reg3_m, (1 - al) * seed_w + al * reg3_w

    print(f"\n{'method':22s}{'margin MAE':>12s}{'win logloss':>13s}")
    print("-" * 47)
    for nm, (mae, ll, _) in [
        ("Attention RNN", metrics(pm, pw, ym[te], yw[te])),
        ("Attention RNN + seed", metrics(attn_bm, attn_bw, ym[te], yw[te])),
        ("M3 + seed (regression)", metrics(reg_bm, reg_bw, ym[te], yw[te])),
    ]:
        print(f"{nm:22s}{mae:12.3f}{ll:13.4f}")
    print("reference avg-pooling RNN+seed (same split, prior run): ~10.95 MAE")

    # ---- interpret the attention: average recency profile on full-history games ----
    ha, _ = attn_model.predict([H[te], A[te], ctx[te]], verbose=0)
    ha = ha[..., 0]  # (n, L) home attention
    full = hm[te].sum(1) == L  # games with a full 20-game history
    prof = ha[full].mean(0)  # mean weight per position (0=oldest, L-1=newest)
    prof_n = prof / prof.sum()
    ent = float(-(prof_n * np.log(prof_n + 1e-12)).sum())
    print(
        f"\nlearned recency profile over {int(full.sum()):,} full-history games "
        f"(position 1 = oldest of last {L}, {L} = most recent):"
    )
    for i in range(L):
        bar = "#" * int(round(prof_n[i] * 300))
        print(f"  g{i+1:>2d}  {prof_n[i]:.3f}  {bar}")
    print(
        f"\n  attention entropy {ent:.2f} (max {math.log(L):.2f} = uniform; lower = more concentrated)"
    )
    print(
        f"  weight on the 5 most recent games: {prof_n[-5:].sum():.2f}  "
        f"vs the 5 oldest: {prof_n[:5].sum():.2f}"
    )
    np.savetxt(
        OUT,
        np.column_stack([np.arange(1, L + 1), prof_n]),
        delimiter=",",
        header="position,weight",
        comments="",
    )
    print(f"  saved profile to {OUT}")


if __name__ == "__main__":
    run()
