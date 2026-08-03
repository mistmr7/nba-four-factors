"""Self-contained RNN trainer for Google Colab (clean CUDA GPU, no Metal).

Setup in Colab:
  1. Runtime > Change runtime type > GPU.
  2. Upload data/features/rnn_sequences.npz to the session (left panel, or
     files.upload()), so it sits at /content/rnn_sequences.npz.
  3. Paste this whole file into a cell and run it (or %run colab_rnn.py).

No package imports, no local dependencies; everything is inline. It trains the
Siamese GRU with the season-to-date summary context, blends with the Vegas seed,
and prints RNN / RNN+seed / Regression / Ensemble for margin MAE and win
log loss / accuracy. Tune by editing CONFIG and re-running; flip RNN_TYPE,
POOLING, BIDIRECTIONAL, USE_PACE to compare.
"""

import math

import numpy as np
from tensorflow.keras import callbacks, layers, models, optimizers

# No scikit-learn: on macOS sklearn + TensorFlow deadlock on a doubled OpenMP
# runtime. The seed/regression heads are linear fits done with numpy below.
NPZ_PATH = "/content/rnn_sequences.npz"


def _lin_fit(X, y):
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
    return coef


def _lin_pred(coef, X):
    return np.column_stack([np.ones(len(X)), X]) @ coef


def _phi(x, sigma):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / (sigma * math.sqrt(2.0))))


CONFIG = dict(
    rnn_type="GRU",
    units=96,
    layers_n=1,
    dropout=0.2,
    dense=64,
    pooling="avg",
    bidirectional=True,
    use_pace=False,
    loss_weight_win=5.0,
    lr=2e-4,
    batch=256,
    epochs=80,
    patience=10,
    seq_len=20,
    blend_k=7.0,
    train_max_year=2017,
    val_years=(2018, 2019, 2020),  # test = start_year > 2020
)


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


def make_model(cfg, L, F, ctx_dim):
    RNN = layers.LSTM if cfg["rnn_type"] == "LSTM" else layers.GRU
    ei = layers.Input((L, F))
    x = layers.Masking(0.0)(ei)
    for i in range(cfg["layers_n"]):
        rs = True if cfg["pooling"] == "avg" else (i < cfg["layers_n"] - 1)
        r = RNN(cfg["units"], return_sequences=rs, dropout=cfg["dropout"])
        x = layers.Bidirectional(r)(x) if cfg["bidirectional"] else r(x)
    if cfg["pooling"] == "avg":
        x = layers.GlobalAveragePooling1D()(x)
    enc = models.Model(ei, x)
    h_in, a_in, c_in = layers.Input((L, F)), layers.Input((L, F)), layers.Input((ctx_dim,))
    h, a = enc(h_in), enc(a_in)
    z = layers.Concatenate()([h, a, layers.Subtract()([h, a]), c_in])
    z = layers.Dropout(cfg["dropout"])(layers.Dense(cfg["dense"], activation="relu")(z))
    m = models.Model(
        [h_in, a_in, c_in],
        [layers.Dense(1, name="margin")(z), layers.Dense(1, activation="sigmoid", name="win")(z)],
    )
    m.compile(
        optimizers.Adam(cfg["lr"]),
        loss={"margin": "mse", "win": "binary_crossentropy"},
        loss_weights={"margin": 1.0, "win": cfg["loss_weight_win"]},
    )
    return m


def metrics(mp, wp, ym, yw):
    p = np.clip(wp, 1e-6, 1 - 1e-6)
    return (
        float(np.mean(np.abs(mp - ym))),
        float(-np.mean(yw * np.log(p) + (1 - yw) * np.log(1 - p))),
        float(np.mean((p > 0.5) == (yw > 0.5))),
    )


def run(cfg=CONFIG):
    d = np.load(NPZ_PATH, allow_pickle=True)
    L = min(cfg["seq_len"], int(d["L"]))
    F = 9 if cfg["use_pace"] else 8
    home, away = d["home_seq"][:, -L:, :], d["away_seq"][:, -L:, :]
    hm, am = d["home_mask"][:, -L:], d["away_mask"][:, -L:]
    tr, va, te = split_idx(d["season"], cfg["train_max_year"], cfg["val_years"])
    H, A = standardize(home, away, hm, am, tr, F)
    ctx = d["ctx"]
    ctx = (ctx - ctx[tr].mean(0)) / (ctx[tr].std(0) + 1e-6)
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
        verbose=1,
    )
    rnn_m, rnn_w = (v.ravel() for v in model.predict([H[te], A[te], ctx[te]], verbose=0))

    sc = _lin_fit(wt[tr].reshape(-1, 1), ym[tr])
    sig_s = float(np.std(ym[tr] - _lin_pred(sc, wt[tr].reshape(-1, 1)))) + 1e-6
    seed_m = _lin_pred(sc, wt[te].reshape(-1, 1))
    seed_w = _phi(seed_m, sig_s)
    rc = _lin_fit(ctx[tr], ym[tr])
    sig_r = float(np.std(ym[tr] - _lin_pred(rc, ctx[tr]))) + 1e-6
    reg_m = _lin_pred(rc, ctx[te])
    reg_w = _phi(reg_m, sig_r)
    al = ng[te] / (ng[te] + cfg["blend_k"])
    bm, bw = (1 - al) * seed_m + al * rnn_m, (1 - al) * seed_w + al * rnn_w
    em, ew = 0.5 * (bm + reg_m), 0.5 * (bw + reg_w)

    rows = {
        "RNN alone": (rnn_m, rnn_w),
        "RNN + seed": (bm, bw),
        "Regression": (reg_m, reg_w),
        "Ensemble": (em, ew),
    }
    print(f"\n{'method':14s}{'margin MAE':>12s}{'win logloss':>13s}{'win acc':>9s}")
    print("-" * 48)
    for nm, (mp, wp) in rows.items():
        mae, ll, acc = metrics(mp, wp, ym[te], yw[te])
        print(f"{nm:14s}{mae:12.3f}{ll:13.4f}{acc:9.3f}")
    print(f"(train n={int(tr.sum()):,}, test n={int(te.sum()):,}); compare to the Regression row.")


if __name__ == "__main__":
    run()
