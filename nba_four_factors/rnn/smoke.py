"""Staged smoke test to localize the training hang. Run:
    TF_CPP_MIN_LOG_LEVEL=2 uv run python -m nba_four_factors.rnn.smoke

Each stage builds a slightly more complex model and fits 1 epoch on 500 games,
printing as it passes. Whatever is the LAST line printed is the layer/combination
that hangs. Also prints the TF and Keras versions (Keras 3 mask handling is a
common suspect).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

NPZ = Path(__file__).resolve().parents[2] / "data" / "features" / "rnn_sequences.npz"
N = 500


def fit(model, X, y):
    model.compile("adam", "mse")
    model.fit(X, y, epochs=1, batch_size=64, verbose=0)


def main():
    import keras

    print(f"tensorflow {tf.__version__} | keras {keras.__version__}", flush=True)
    d = np.load(NPZ, allow_pickle=True)
    hs = d["home_seq"][:N, :, :8].astype("float32")
    as_ = d["away_seq"][:N, :, :8].astype("float32")
    ctx = d["ctx"][:N].astype("float32")
    y = d["y_margin"][:N].astype("float32")

    i = layers.Input((20, 8))
    o = layers.Dense(1)(layers.GRU(32)(i))
    fit(models.Model(i, o), hs, y)
    print("T1 plain GRU ok", flush=True)

    i = layers.Input((20, 8))
    x = layers.Masking()(i)
    o = layers.Dense(1)(layers.GRU(32)(x))
    fit(models.Model(i, o), hs, y)
    print("T2 masked GRU ok", flush=True)

    i = layers.Input((20, 8))
    x = layers.Bidirectional(layers.GRU(32))(layers.Masking()(i))
    o = layers.Dense(1)(x)
    fit(models.Model(i, o), hs, y)
    print("T3 masked BiGRU ok", flush=True)

    ei = layers.Input((20, 8))
    enc = models.Model(ei, layers.Bidirectional(layers.GRU(32))(layers.Masking()(ei)))
    hi, ai, ci = layers.Input((20, 8)), layers.Input((20, 8)), layers.Input((ctx.shape[1],))
    z = layers.Concatenate()([enc(hi), enc(ai), ci])
    o = layers.Dense(1)(layers.Dense(32, activation="relu")(z))
    fit(models.Model([hi, ai, ci], o), [hs, as_, ctx], y)
    print("T4 siamese ok", flush=True)

    yw = d["y_win"][:N].astype("float32")

    def two_head():
        ei2 = layers.Input((20, 8))
        enc2 = models.Model(ei2, layers.Bidirectional(layers.GRU(32))(layers.Masking()(ei2)))
        h2, a2, c2 = layers.Input((20, 8)), layers.Input((20, 8)), layers.Input((ctx.shape[1],))
        zz = layers.Dense(32, activation="relu")(layers.Concatenate()([enc2(h2), enc2(a2), c2]))
        m = models.Model(
            [h2, a2, c2],
            [
                layers.Dense(1, name="margin")(zz),
                layers.Dense(1, activation="sigmoid", name="win")(zz),
            ],
        )
        m.compile(
            "adam",
            {"margin": "mse", "win": "binary_crossentropy"},
            loss_weights={"margin": 1.0, "win": 5.0},
        )
        return m

    # T5: two-output head (margin + win), the part the trainer adds.
    two_head().fit([hs, as_, ctx], {"margin": y, "win": yw}, epochs=1, batch_size=64, verbose=0)
    print("T5 two-output ok", flush=True)

    # T6: + validation_data + EarlyStopping(restore_best_weights) on a shared encoder.
    es = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    two_head().fit(
        [hs[:400], as_[:400], ctx[:400]],
        {"margin": y[:400], "win": yw[:400]},
        validation_data=([hs[400:], as_[400:], ctx[400:]], {"margin": y[400:], "win": yw[400:]}),
        epochs=4,
        batch_size=64,
        callbacks=[es],
        verbose=0,
    )
    print("T6 two-output + val + EarlyStopping(restore) ok", flush=True)

    # T7: full-size single-output on all train seasons, to test pure data scale.
    tr = np.array([int(s[:4]) for s in d["season"]]) <= 2017
    Hf = d["home_seq"][tr][:, :, :8].astype("float32")
    yf = d["y_margin"][tr].astype("float32")
    print(
        f"T7 fitting full train size n={int(tr.sum()):,} (verbose=1, watch the step bar)...",
        flush=True,
    )
    i = layers.Input((20, 8))
    o = layers.Dense(1)(layers.Bidirectional(layers.GRU(96))(layers.Masking()(i)))
    mm = models.Model(i, o)
    mm.compile("adam", "mse")
    mm.fit(Hf, yf, epochs=1, batch_size=256, verbose=1)
    print("T7 full-size ok", flush=True)

    # T8: the exact real trainer (full-scale Siamese + val + callbacks + 2 heads),
    # capped at 2 epochs. This is the one combination nothing above covered.
    from .train_keras import CONFIG, evaluate_all

    print("T8 running the real evaluate_all (2 epochs, verbose=1)...", flush=True)
    out, ntr, nte = evaluate_all(d, {**CONFIG, "epochs": 2, "patience": 2}, verbose=1)
    print("T8 real trainer ok:", {k: round(v[0], 2) for k, v in out.items()}, flush=True)
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
