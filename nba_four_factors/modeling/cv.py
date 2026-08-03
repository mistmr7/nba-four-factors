"""Expanding-window cross-validation of the regression ladder (game level).

Trains on all prior seasons and tests on the next, walking forward through the
29 seasons. Four nested rungs, each refit jointly (not frozen stepwise):

* M0  Vegas only: the preseason win-total differential.
* M1  Four factors: home-minus-away season-to-date eFG, OREB, TOV, FTR diffs.
* M2  + context: rest, back-to-back, and seven-day-travel differentials.
* M3  + shrunk recent form: the reliability-weighted recent-form deviation,
       w * (recent - season baseline), composite level, with the signal variance
       and effective sample size estimated on the training fold only.

All rungs are scored on the identical row set (games where every M3 feature is
available) so the ladder is a clean apples-to-apples comparison. Two heads are
fit per rung: linear regression for point margin (MAE) and logistic regression
for the home win (log loss, Brier, accuracy). Target is raw point margin per the
Session 14 decision. Vegas's preseason win-total MAE (about 6.6 wins) is printed
as the season-level bar; the season-level model-vs-Vegas comparison needs the
seed-and-blend and is the next step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
SEED_MAP = REPO / "data" / "features" / "vegas_seed_mapping.json"
WINDOW = 15

# Fourth factor uses makes-based FT/FGA (Oliver's original), which beat the
# attempt-based FTA/FGA in the head-to-head and is less inflated by intentional
# fouls. See cv_vegas.py and docs/Session14_vegas_headtohead.md.
FACTORS = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
CONTEXT = ["rest_diff", "b2b_diff", "miles7d_diff"]
LADDER = {
    "M0_vegas": ["wintotal_diff"],
    "M1_four_factors": FACTORS,
    "M2_plus_context": FACTORS + CONTEXT,
    "M3_plus_recent_form": FACTORS + CONTEXT + ["shrunk_dev_diff"],
}
ALL_FEATURES = FACTORS + CONTEXT + ["wintotal_diff", "shrunk_dev_diff"]


def add_shrunk_form(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the composite-level shrunk recent-form deviation, train-only estimates."""
    # Effective sample size from the variance ratio (EDA 06 method), train only.
    sigma_pergame_sq = float(np.nanmean(train["form_var"]))
    var_form_mean = float(np.nanvar(train["form_mean"]))
    n_eff = sigma_pergame_sq / var_form_mean if var_form_mean > 0 else float(WINDOW)
    n_eff = float(np.clip(n_eff, 1.0, WINDOW))
    signal_var = max(var_form_mean - sigma_pergame_sq / n_eff, 1e-6)

    out = []
    for frame in (train, test):
        f = frame.copy()
        dev_h = f["form_mean"] - f["base_comp"]
        dev_a = f["away_form_mean"] - f["away_base_comp"]
        w_h = signal_var / (signal_var + f["form_var"] / n_eff)
        w_a = signal_var / (signal_var + f["away_form_var"] / n_eff)
        f["shrunk_dev_diff"] = w_h * dev_h - w_a * dev_a
        out.append(f)
    return out[0], out[1]


def run() -> pd.DataFrame:
    df = pd.read_parquet(TABLE)
    df["yr"] = df["season"].str[:4].astype(int)
    seasons = sorted(df["season"].unique(), key=lambda s: int(s[:4]))

    rows = []
    for i in range(3, len(seasons)):
        train_s = set(seasons[:i])
        test_s = seasons[i]
        tr = df[df["season"].isin(train_s)].copy()
        te = df[df["season"] == test_s].copy()
        tr, te = add_shrunk_form(tr, te)

        # Common evaluable set: every M3 feature present, both folds.
        tr_e = tr.dropna(subset=[*ALL_FEATURES, "home_margin", "home_win"])
        te_e = te.dropna(subset=[*ALL_FEATURES, "home_margin", "home_win"])
        if len(tr_e) < 500 or len(te_e) < 100 or te_e["home_win"].nunique() < 2:
            continue

        for name, feats in LADDER.items():
            Xtr, Xte = tr_e[feats].to_numpy(), te_e[feats].to_numpy()
            # Margin head.
            lin = LinearRegression().fit(Xtr, tr_e["home_margin"])
            mae = mean_absolute_error(te_e["home_margin"], lin.predict(Xte))
            # Win head (standardized for stable logistic).
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), tr_e["home_win"])
            p = clf.predict_proba(sc.transform(Xte))[:, 1]
            rows.append(
                dict(
                    test_season=test_s,
                    model=name,
                    n_test=len(te_e),
                    margin_mae=mae,
                    win_logloss=log_loss(te_e["home_win"], p),
                    win_brier=brier_score_loss(te_e["home_win"], p),
                    win_acc=accuracy_score(te_e["home_win"], (p > 0.5).astype(int)),
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    res = run()
    summary = res.groupby("model").agg(
        margin_mae=("margin_mae", "mean"),
        win_logloss=("win_logloss", "mean"),
        win_brier=("win_brier", "mean"),
        win_acc=("win_acc", "mean"),
        seasons=("test_season", "nunique"),
    )
    order = ["M0_vegas", "M1_four_factors", "M2_plus_context", "M3_plus_recent_form"]
    summary = summary.loc[order]
    vegas_mae = json.loads(SEED_MAP.read_text())["vegas_win_total_mae"]
    print(
        "Expanding-window OOS results, averaged over test seasons (lower is better except acc):\n"
    )
    print(summary.round(4).to_string())
    print(f"\nReference: Vegas preseason win-total MAE = {vegas_mae:.2f} wins (season-level bar).")
    print("Game-level margin MAE above is points-per-game, not directly comparable;")
    print(
        "the season-level expected-wins-vs-Vegas test is the next step (needs the M0 seed + blend)."
    )
    out = REPO / "data" / "features" / "cv_ladder_results.csv"
    res.to_csv(out, index=False)
    print(f"\nPer-season results written to {out}")


if __name__ == "__main__":
    main()
