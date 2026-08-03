"""Optimize the recent-form rolling window W.

The season-to-date four-factor baseline does not depend on W; only the recent-
form term does. For each candidate window we rebuild the trailing rolling
composite mean and variance, form the reliability-weighted recent-form deviation
(shrinkage estimated on the training fold), add it to the four-factor-plus-
context model, and score out of sample. The no-form model (M2) is scored on the
identical rows as a baseline, so the gap is the value the recent-form term adds
at that window.

Pick the W that minimizes out-of-sample margin MAE (and check win log loss).
That W is also the natural point for the seed-to-model handoff: the recent-form
signal is only trustworthy once the window is full.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from .features import _add_composite, _team_game_frame

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
FIG = REPO / "figures" / "modeling"

STATIC = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d", "rest_diff", "b2b_diff", "miles7d_diff"]
W_GRID = [5, 8, 10, 12, 15, 20, 25, 30, 40]
MIN_PERIODS = 5


def composite_frame() -> pd.DataFrame:
    tf = _add_composite(_team_game_frame())
    tf = tf.sort_values(["team_id", "season", "game_date"])
    g = tf.groupby(["team_id", "season"], sort=False)["composite"]
    tf["base_comp"] = g.transform(lambda s: s.shift(1).expanding().mean())
    return tf[["season", "game_id", "team_id", "opp_team_id", "is_home", "composite", "base_comp"]]


def form_at(tf: pd.DataFrame, w: int) -> pd.DataFrame:
    """Per-game home/away rolling form mean, variance, and season baseline at W."""
    out = tf.copy()
    g = out.groupby(["team_id", "season"], sort=False)["composite"]
    out["fm"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=MIN_PERIODS).mean())
    out["fv"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=MIN_PERIODS).var(ddof=1))
    home = out[out["is_home"]][["game_id", "fm", "fv", "base_comp"]].rename(
        columns={"fm": "h_fm", "fv": "h_fv", "base_comp": "h_base"}
    )
    away = out[~out["is_home"]][["game_id", "fm", "fv", "base_comp"]].rename(
        columns={"fm": "a_fm", "fv": "a_fv", "base_comp": "a_base"}
    )
    return home.merge(away, on="game_id", how="inner")


def shrunk(df: pd.DataFrame, signal_var: float, n_eff: float) -> np.ndarray:
    w_h = signal_var / (signal_var + df["h_fv"] / n_eff)
    w_a = signal_var / (signal_var + df["a_fv"] / n_eff)
    return (w_h * (df["h_fm"] - df["h_base"]) - w_a * (df["a_fm"] - df["a_base"])).to_numpy()


def evaluate(w: int, static: pd.DataFrame, tf: pd.DataFrame) -> dict:
    feats = form_at(tf, w)
    df = static.merge(feats, on="game_id", how="inner")
    seasons = sorted(df["season"].unique(), key=lambda s: int(s[:4]))
    full_mae, m2_mae, full_ll, m2_ll = [], [], [], []
    for i in range(3, len(seasons)):
        tr = df[df["season"].isin(set(seasons[:i]))].dropna(
            subset=[*STATIC, "h_fm", "a_fm", "h_fv", "a_fv", "home_margin", "home_win"]
        )
        te = df[df["season"] == seasons[i]].dropna(
            subset=[*STATIC, "h_fm", "a_fm", "h_fv", "a_fv", "home_margin", "home_win"]
        )
        if len(tr) < 500 or len(te) < 100 or te["home_win"].nunique() < 2:
            continue
        # In-fold shrinkage components from the training rolling form (both sides).
        fm = np.concatenate([tr["h_fm"], tr["a_fm"]])
        fv = np.concatenate([tr["h_fv"], tr["a_fv"]])
        n_eff = float(np.clip(np.nanmean(fv) / np.nanvar(fm), 1.0, w))
        signal_var = max(float(np.nanvar(fm) - np.nanmean(fv) / n_eff), 1e-6)
        tr = tr.assign(sdd=shrunk(tr, signal_var, n_eff))
        te = te.assign(sdd=shrunk(te, signal_var, n_eff))

        for feats_set, mae_list, ll_list in [
            (STATIC, m2_mae, m2_ll),
            ([*STATIC, "sdd"], full_mae, full_ll),
        ]:
            lin = LinearRegression().fit(tr[feats_set], tr["home_margin"])
            mae_list.append(mean_absolute_error(te["home_margin"], lin.predict(te[feats_set])))
            sc = StandardScaler().fit(tr[feats_set])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(tr[feats_set]), tr["home_win"])
            ll_list.append(
                log_loss(te["home_win"], clf.predict_proba(sc.transform(te[feats_set]))[:, 1])
            )
    return dict(
        W=w,
        full_MAE=np.mean(full_mae),
        noform_MAE=np.mean(m2_mae),
        full_logloss=np.mean(full_ll),
        noform_logloss=np.mean(m2_ll),
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    static = pd.read_parquet(TABLE)[["game_id", "season", *STATIC, "home_margin", "home_win"]]
    tf = composite_frame()
    res = pd.DataFrame([evaluate(w, static, tf) for w in W_GRID]).set_index("W")
    res["MAE_gain"] = res["noform_MAE"] - res["full_MAE"]
    res["logloss_gain"] = res["noform_logloss"] - res["full_logloss"]
    print("Recent-form window sweep (OOS; gain = no-form minus with-form, higher is better):\n")
    print(res.round(4).to_string())
    best = res["full_MAE"].idxmin()
    best_ll = res["full_logloss"].idxmin()
    print(f"\nBest window by margin MAE: {best}.  Best by win log loss: {best_ll}.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(res.index, res["full_MAE"], marker="o", color="#534AB7", label="four factors + form")
    ax[0].axhline(res["noform_MAE"].mean(), color="#888780", ls="--", label="no-form baseline")
    ax[0].set_title("Margin MAE by recent-form window")
    ax[0].set_xlabel("Window W (games)")
    ax[0].set_ylabel("OOS margin MAE")
    ax[0].legend()
    ax[1].plot(
        res.index, res["full_logloss"], marker="o", color="#1D9E75", label="four factors + form"
    )
    ax[1].axhline(res["noform_logloss"].mean(), color="#888780", ls="--", label="no-form baseline")
    ax[1].set_title("Win log loss by recent-form window")
    ax[1].set_xlabel("Window W (games)")
    ax[1].set_ylabel("OOS log loss")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "window_sweep.png", dpi=130)
    plt.close(fig)
    print(f"\nFigure written to {FIG / 'window_sweep.png'}")


if __name__ == "__main__":
    main()
