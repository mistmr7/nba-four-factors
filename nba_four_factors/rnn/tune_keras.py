"""Time-budgeted random-search tuner for the Siamese LSTM/GRU. Keras/TensorFlow.

Run locally for as long as you like (default ~75 min):
    python -m nba_four_factors.rnn.tune_keras

It samples architectures and hyperparameters, trains each with early stopping on
a validation split, and logs every trial to data/features/rnn_tuning_results.csv
(written incrementally, so partial results survive an interruption). Model
selection is on the validation objective only; test metrics are logged but never
used to choose. The best config is saved to data/features/rnn_best_config.json
and the best is refit at the end for a clean test readout.

Search space: rnn type (LSTM/GRU), units, layers, dropout, dense width, learning
rate, batch size, sequence length, pace on/off, pooling (last state vs masked
average), and bidirectional. Selection objective and time budget are at the top.
"""

from __future__ import annotations

import csv
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

from .train_keras import load, split_idx, standardize

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data" / "features" / "rnn_tuning_results.csv"
BEST = REPO / "data" / "features" / "rnn_best_config.json"

TIME_BUDGET_MIN = 75  # wall-clock budget for the search
OBJECTIVE = "val_margin_mae"  # or "val_win_logloss"
TRAIN_MAX_YEAR = 2017
VAL_YEARS = (2018, 2019, 2020)  # test is start_year > 2020

SPACE = dict(
    rnn_type=["LSTM", "GRU"],
    units=[32, 48, 64, 96, 128],
    layers_n=[1, 2],
    dropout=[0.2, 0.3, 0.4, 0.5],
    dense=[32, 64, 128],
    lr=[2e-4, 5e-4, 1e-3, 2e-3],
    batch=[128, 256, 512],
    seq_len=[10, 15, 20, 30],
    use_pace=[False, True],
    pooling=["last", "avg"],
    bidirectional=[False, True],
    loss_weight_win=[1.0, 5.0, 10.0],
    epochs=[60],
    patience=[7],
)


def sample(rng):
    return {k: rng.choice(v) for k, v in SPACE.items()}


def make_model(p, L, F, ctx_dim):
    RNN = layers.LSTM if p["rnn_type"] == "LSTM" else layers.GRU
    inp = layers.Input((L, F))
    x = layers.Masking(mask_value=0.0)(inp)
    for i in range(p["layers_n"]):
        last = i == p["layers_n"] - 1
        rs = True if p["pooling"] == "avg" else (not last)
        rnn = RNN(p["units"], return_sequences=rs, dropout=p["dropout"])
        x = layers.Bidirectional(rnn)(x) if p["bidirectional"] else rnn(x)
    if p["pooling"] == "avg":
        x = layers.GlobalAveragePooling1D()(x)
    enc = models.Model(inp, x)

    h_in, a_in, c_in = layers.Input((L, F)), layers.Input((L, F)), layers.Input((ctx_dim,))
    h, a = enc(h_in), enc(a_in)
    z = layers.Concatenate()([h, a, layers.Subtract()([h, a]), c_in])
    z = layers.Dropout(p["dropout"])(layers.Dense(p["dense"], activation="relu")(z))
    margin = layers.Dense(1, name="margin")(z)
    win = layers.Dense(1, activation="sigmoid", name="win")(z)
    m = models.Model([h_in, a_in, c_in], [margin, win])
    m.compile(
        optimizer=optimizers.Adam(p["lr"]),
        loss={"margin": "mse", "win": "binary_crossentropy"},
        loss_weights={"margin": 1.0, "win": p["loss_weight_win"]},
    )
    return m


def prep(d, p):
    L = min(p["seq_len"], int(d["L"]))
    F = 9 if p["use_pace"] else 8
    home, away = d["home_seq"][:, -L:, :], d["away_seq"][:, -L:, :]
    hm, am = d["home_mask"][:, -L:], d["away_mask"][:, -L:]
    tr, va, te = split_idx(d["season"], TRAIN_MAX_YEAR, VAL_YEARS)
    H, A = standardize(home, away, hm, am, tr, F)
    ctx = d["ctx"]
    cmu, csd = ctx[tr].mean(0), ctx[tr].std(0) + 1e-6
    ctx = (ctx - cmu) / csd
    ym, yw = d["y_margin"], d["y_win"]

    def pack(m):
        return [H[m], A[m], ctx[m]], {"margin": ym[m], "win": yw[m]}

    return pack(tr), pack(va), pack(te), L, F, ctx.shape[1]


def metrics(model, X, Y):
    pm, pw = model.predict(X, verbose=0)
    pm, pw = pm.ravel(), np.clip(pw.ravel(), 1e-6, 1 - 1e-6)
    mae = float(np.mean(np.abs(pm - Y["margin"])))
    ll = float(-np.mean(Y["win"] * np.log(pw) + (1 - Y["win"]) * np.log(1 - pw)))
    acc = float(np.mean((pw > 0.5) == (Y["win"] > 0.5)))
    return mae, ll, acc


def trial(d, p):
    # Clear the graph each trial so a long multi-hundred-trial GPU run does not
    # leak memory or slow down as TensorFlow accumulates models.
    tf.keras.backend.clear_session()
    gc.collect()
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), L, F, cdim = prep(d, p)
    m = make_model(p, L, F, cdim)
    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=p["patience"], restore_best_weights=True
    )
    m.fit(
        Xtr,
        Ytr,
        validation_data=(Xva, Yva),
        epochs=p["epochs"],
        batch_size=p["batch"],
        callbacks=[es],
        verbose=0,
    )
    vmae, vll, vacc = metrics(m, Xva, Yva)
    tmae, tll, tacc = metrics(m, Xte, Yte)
    return dict(
        val_margin_mae=vmae,
        val_win_logloss=vll,
        val_win_acc=vacc,
        test_margin_mae=tmae,
        test_win_logloss=tll,
        test_win_acc=tacc,
    )


def main():
    d = load()
    rng = random.Random(0)
    fields = [
        *SPACE,
        "val_margin_mae",
        "val_win_logloss",
        "val_win_acc",
        "test_margin_mae",
        "test_win_logloss",
        "test_win_acc",
        "secs",
    ]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    rows, t0, n = [], time.time(), 0
    while time.time() - t0 < TIME_BUDGET_MIN * 60:
        p = sample(rng)
        n += 1
        ts = time.time()
        try:
            r = trial(d, p)
        except Exception as e:
            print(f"trial {n} failed: {e}")
            continue
        row = {**p, **r, "secs": round(time.time() - ts, 1)}
        rows.append(row)
        with RESULTS.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(
            f"[{n:>3}] {p['rnn_type']} u{p['units']} l{p['layers_n']} d{p['dropout']} "
            f"{p['pooling']}{'+bi' if p['bidirectional'] else ''} pace={int(p['use_pace'])} "
            f"sl{p['seq_len']} lr{p['lr']:.0e} -> val MAE {r['val_margin_mae']:.3f} "
            f"test MAE {r['test_margin_mae']:.3f} acc {r['test_win_acc']:.3f} "
            f"({row['secs']}s, {int(time.time()-t0)}s elapsed)"
        )
    best = min(rows, key=lambda x: x[OBJECTIVE])
    BEST.write_text(json.dumps(best, indent=2, default=str))
    print(f"\nRan {len(rows)} trials. Best by {OBJECTIVE}:")
    for k in [
        "rnn_type",
        "units",
        "layers_n",
        "dropout",
        "dense",
        "lr",
        "batch",
        "seq_len",
        "use_pace",
        "pooling",
        "bidirectional",
        "loss_weight_win",
    ]:
        print(f"  {k}: {best[k]}")
    print(
        f"  val MAE {best['val_margin_mae']:.3f} | test MAE {best['test_margin_mae']:.3f} "
        f"| test win acc {best['test_win_acc']:.3f} | test win logloss {best['test_win_logloss']:.3f}"
    )
    print("  (regression baseline 9.87, Vegas floor 9.68)")
    print(f"\nFull log: {RESULTS}\nBest config: {BEST}")


if __name__ == "__main__":
    main()
