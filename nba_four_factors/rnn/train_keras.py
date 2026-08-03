"""Siamese LSTM/GRU with the season-to-date summary as context, the Vegas-seed
blend, and a regression ensemble. Keras / TensorFlow.

Run locally (needs tensorflow); build tensors first with rnn.dataset:
    python -m nba_four_factors.rnn.train_keras

Two structural changes from the first version, both aimed at the gaps the 138-
trial search exposed:

1. Summary context. The model's context input now carries the season-to-date
   four-factor differentials (the regression's inputs) alongside the schedule
   terms, so the RNN inherits the regression's inductive bias and only has to add
   temporal signal on top.
2. Seed blend. At prediction time the RNN output is blended with a Vegas-seed
   model by alpha(n) = n / (n + k), n = games played, so early-season games lean
   on the market prior (the same cold-start fix the regression uses).

It then reports four predictors on the test seasons: the RNN alone, the
seed-blended RNN, a plain regression on the same context, and an ensemble
(average of the seed-blended RNN and the regression), for margin MAE and win
log loss / accuracy. Defaults are the best config from the random search.
"""

from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

# NOTE: do not import scikit-learn here. On macOS, sklearn + TensorFlow load two
# OpenMP runtimes and deadlock on the first fit. The seed/regression heads are
# plain linear fits, done below with numpy plus a normal-CDF for win prob.

# Run on the GPU with a gradual warmup by default: this is the exact path the
# smoke test proved works end to end (stages T1-T8). Set FORCE_CPU=1 to hide the
# GPU instead (note: on Apple Metal this may not actually disable it).
if os.environ.get("FORCE_CPU"):
    with contextlib.suppress(Exception):
        tf.config.set_visible_devices([], "GPU")

REPO = Path(__file__).resolve().parents[2]
# Sequence tensors. Defaults to the four-factor build; set RNN_NPZ to point at an
# alternate (e.g. rnn_sequences_raw.npz) for the raw-box-score experiment.
NPZ = Path(os.environ.get("RNN_NPZ", str(REPO / "data" / "features" / "rnn_sequences.npz")))

REG_BASELINE_MAE = 9.87
VEGAS_FLOOR_MAE = 9.68
W = 15  # recent-form window used to build the saved form pieces (for shrinkage)

CONFIG = dict(
    rnn_type="GRU",
    units=96,
    layers_n=1,
    dropout=0.2,
    dense=64,
    pooling="last",
    bidirectional=True,
    use_pace=False,
    loss_weight_win=5.0,
    lr=2e-4,
    batch=256,
    epochs=80,
    patience=10,
    seq_len=20,
    blend_k=7.0,  # seed-to-RNN handoff: alpha = n/(n+k)
    train_max_year=2017,
    val_years=(2018, 2019, 2020),
)


def load():
    return np.load(NPZ, allow_pickle=True)


def feat_count(d, cfg):
    """Number of sequence channels to use. The four-factor build is 9 wide (the
    last channel is pace, dropped unless use_pace). Any other width (e.g. the raw
    box-score build) is used in full.
    """
    w = int(d["home_seq"].shape[-1])
    return w if w != 9 else (9 if cfg["use_pace"] else 8)


def split_idx(season, train_max, val_years):
    yr = np.array([int(s[:4]) for s in season])
    return yr <= train_max, np.isin(yr, list(val_years)), yr > max(val_years)


def standardize(home, away, hmask, amask, tr, F):
    real = np.concatenate([home[tr][hmask[tr] == 1], away[tr][amask[tr] == 1]], axis=0)
    mu, sd = np.nanmean(real[:, :F], 0), np.nanstd(real[:, :F], 0) + 1e-6

    def norm(x, m):
        return np.nan_to_num(
            ((x[..., :F] - mu) / sd) * m[..., None], nan=0.0, posinf=0.0, neginf=0.0
        )

    return norm(home, hmask), norm(away, amask)


def make_encoder(cfg, L, F):
    RNN = layers.LSTM if cfg["rnn_type"] == "LSTM" else layers.GRU
    inp = layers.Input((L, F))
    x = layers.Masking(0.0)(inp)
    for i in range(cfg["layers_n"]):
        rs = True if cfg["pooling"] == "avg" else (i < cfg["layers_n"] - 1)
        rnn = RNN(cfg["units"], return_sequences=rs, dropout=cfg["dropout"])
        x = layers.Bidirectional(rnn)(x) if cfg["bidirectional"] else rnn(x)
    if cfg["pooling"] == "avg":
        x = layers.GlobalAveragePooling1D()(x)
    return models.Model(inp, x)


def make_model(cfg, L, F, ctx_dim):
    enc = make_encoder(cfg, L, F)
    h_in, a_in, c_in = layers.Input((L, F)), layers.Input((L, F)), layers.Input((ctx_dim,))
    h, a = enc(h_in), enc(a_in)
    z = layers.Concatenate()([h, a, layers.Subtract()([h, a]), c_in])
    z = layers.Dropout(cfg["dropout"])(layers.Dense(cfg["dense"], activation="relu")(z))
    m = models.Model(
        [h_in, a_in, c_in],
        [layers.Dense(1, name="margin")(z), layers.Dense(1, activation="sigmoid", name="win")(z)],
    )
    m.compile(
        optimizer=optimizers.Adam(cfg["lr"]),
        loss={"margin": "mse", "win": "binary_crossentropy"},
        loss_weights={"margin": 1.0, "win": cfg["loss_weight_win"]},
    )
    return m


def _lin_fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef


def _lin_pred(coef, X):
    return np.column_stack([np.ones(len(X)), X]) @ coef


def _phi(x, sigma):
    """Normal CDF, no scipy: map an expected margin to a win probability."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / (sigma * math.sqrt(2.0))))


def metrics(margin_pred, win_pred, ym, yw):
    p = np.clip(win_pred, 1e-6, 1 - 1e-6)
    mae = float(np.mean(np.abs(margin_pred - ym)))
    ll = float(-np.mean(yw * np.log(p) + (1 - yw) * np.log(1 - p)))
    acc = float(np.mean((p > 0.5) == (yw > 0.5)))
    return mae, ll, acc


def evaluate_all(d, cfg=CONFIG, verbose=0):
    L = min(cfg["seq_len"], int(d["L"]))
    F = feat_count(d, cfg)
    home, away = d["home_seq"][:, -L:, :], d["away_seq"][:, -L:, :]
    hm, am = d["home_mask"][:, -L:], d["away_mask"][:, -L:]
    tr, va, te = split_idx(d["season"], cfg["train_max_year"], cfg["val_years"])
    H, A = standardize(home, away, hm, am, tr, F)
    ctx = d["ctx"]
    cmu, csd = ctx[tr].mean(0), ctx[tr].std(0) + 1e-6
    ctx = (ctx - cmu) / csd
    ym, yw, wt, ng = d["y_margin"], d["y_win"], d["wintotal_diff"], d["ngames"]

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
        verbose=verbose,
    )
    rnn_m, rnn_w = (x.ravel() for x in model.predict([H[te], A[te], ctx[te]], verbose=0))

    # Seed (Vegas line), and the full production regression M3 (= context plus the
    # shrunk recent-form term) for a fair comparison, numpy only.
    sc = _lin_fit(wt[tr].reshape(-1, 1), ym[tr])
    sig_s = float(np.std(ym[tr] - _lin_pred(sc, wt[tr].reshape(-1, 1)))) + 1e-6
    seed_m = _lin_pred(sc, wt[te].reshape(-1, 1))
    seed_w = _phi(seed_m, sig_s)

    # Shrunk recent-form term from the saved form pieces, train-only shrinkage.
    f = d["form"]  # [form_mean, base_comp, form_var, away_form_mean, away_base_comp, away_form_var]
    hfm, hb, hfv, afm, ab, afv = (f[:, i] for i in range(6))
    fm = np.concatenate([hfm[tr], afm[tr]])
    fv = np.concatenate([hfv[tr], afv[tr]])
    n_eff = float(np.clip(np.nanmean(fv) / np.nanvar(fm), 1, W))
    sv = max(np.nanvar(fm) - np.nanmean(fv) / n_eff, 1e-6)
    sdd = (sv / (sv + hfv / n_eff)) * (hfm - hb) - (sv / (sv + afv / n_eff)) * (afm - ab)
    X3 = np.column_stack([ctx, sdd])
    r2 = _lin_fit(ctx[tr], ym[tr])
    reg2_m = _lin_pred(r2, ctx[te])
    reg2_w = _phi(reg2_m, float(np.std(ym[tr] - _lin_pred(r2, ctx[tr]))) + 1e-6)
    r3 = _lin_fit(X3[tr], ym[tr])
    sig3 = float(np.std(ym[tr] - _lin_pred(r3, X3[tr]))) + 1e-6
    reg3_m = _lin_pred(r3, X3[te])
    reg3_w = _phi(reg3_m, sig3)

    alpha = ng[te] / (ng[te] + cfg["blend_k"])
    blend_m = (1 - alpha) * seed_m + alpha * rnn_m
    blend_w = (1 - alpha) * seed_w + alpha * rnn_w
    rm3_m = (1 - alpha) * seed_m + alpha * reg3_m  # M3 + seed (the real comparator)
    rm3_w = (1 - alpha) * seed_w + alpha * reg3_w
    ens_m = 0.5 * (blend_m + rm3_m)
    ens_w = 0.5 * (blend_w + rm3_w)

    out = {
        "RNN alone": metrics(rnn_m, rnn_w, ym[te], yw[te]),
        "RNN + seed": metrics(blend_m, blend_w, ym[te], yw[te]),
        "Reg M2": metrics(reg2_m, reg2_w, ym[te], yw[te]),
        "Reg M3 + seed": metrics(rm3_m, rm3_w, ym[te], yw[te]),
        "Ensemble": metrics(ens_m, ens_w, ym[te], yw[te]),
    }
    return out, int(tr.sum()), int(te.sum())


def _warmup(d):
    """Cold-start workaround. On Apple Metal a fresh process hangs compiling the
    bidirectional GRU-96 kernel cold. The smoke test only reached that kernel
    after compiling the smaller GRU/bidirectional kernels first, so we replicate
    that gradual ramp here on a tiny slice before the real fit.
    """
    print("warming up TF kernels (gradual ramp)...", flush=True)
    x = d["home_seq"][:300, :, :8].astype("float32")
    y = d["y_margin"][:300].astype("float32")
    a = d["away_seq"][:300, :, :8].astype("float32")
    c = d["ctx"][:300].astype("float32")
    U = CONFIG["units"]

    def fit(model, X):
        model.compile("adam", "mse")
        model.fit(X, y, epochs=1, batch_size=64, verbose=0)

    i = layers.Input((20, 8))
    fit(models.Model(i, layers.Dense(1)(layers.GRU(32)(i))), x)
    i = layers.Input((20, 8))
    fit(models.Model(i, layers.Dense(1)(layers.GRU(32)(layers.Masking()(i)))), x)
    i = layers.Input((20, 8))
    fit(
        models.Model(i, layers.Dense(1)(layers.Bidirectional(layers.GRU(32))(layers.Masking()(i)))),
        x,
    )
    ei = layers.Input((20, 8))
    enc = models.Model(ei, layers.Bidirectional(layers.GRU(32))(layers.Masking()(ei)))
    hi, ai, ci = layers.Input((20, 8)), layers.Input((20, 8)), layers.Input((c.shape[1],))
    z = layers.Concatenate()([enc(hi), enc(ai), ci])
    fit(
        models.Model([hi, ai, ci], layers.Dense(1)(layers.Dense(32, activation="relu")(z))),
        [x, a, c],
    )
    i = layers.Input((20, 8))
    fit(
        models.Model(i, layers.Dense(1)(layers.Bidirectional(layers.GRU(U))(layers.Masking()(i)))),
        x,
    )
    print("warmup done.", flush=True)


def main():
    d = load()
    _warmup(d)  # gradual ramp: the proven-working cold-start path (smoke T1-T8)
    # verbose=1 shows the in-epoch step bar.
    out, ntr, nte = evaluate_all(d, CONFIG, verbose=int(os.environ.get("RNN_VERBOSE", "1")))
    print(f"\n{'method':14s}{'margin MAE':>12s}{'win logloss':>13s}{'win acc':>9s}")
    print("-" * 48)
    for nm, (mae, ll, acc) in out.items():
        print(f"{nm:14s}{mae:12.3f}{ll:13.4f}{acc:9.3f}")
    print("-" * 48)
    print("Compare against the Regression row above (same train/test split).")
    print(
        f"For reference only (different all-seasons expanding eval): regression {REG_BASELINE_MAE}, Vegas floor {VEGAS_FLOOR_MAE}."
    )
    print(f"(train n={ntr:,}, test n={nte:,})")


if __name__ == "__main__":
    main()
