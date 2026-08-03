"""Head-to-head against the Vegas closing line, on the identical matched games.

Restricts evaluation to games that have a validated closing spread
(2007-08 onward), so the model's margin MAE sits next to the line's MAE on the
exact same rows. Also tests the fourth-factor definition: attempt-based FTR
(FTA/FGA) versus makes-based FT/FGA, swapped into the four-factor block.

Vegas baseline: predicted margin is the closing line itself; the Vegas win
probability is a logistic of the home win on the closing margin, fit on the
training fold, so log loss is comparable to the models.

Reports overall MAE and log loss per model and for Vegas, plus an MAE breakdown
by closing-spread magnitude (pick'em games through large favorites) to show
where the model is closest to the market.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from .cv import add_shrunk_form

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
LINES = REPO / "data" / "features" / "vegas_game_lines.parquet"

CORE3 = ["d_efg_d", "d_oreb_d", "d_tov_d"]
CONTEXT = ["rest_diff", "b2b_diff", "miles7d_diff"]
MODELS = {
    "FF_FTA": [*CORE3, "d_ftr_d"],
    "FF_FTM": [*CORE3, "d_ftmfga_d"],
    "FF_BOTH": [*CORE3, "d_ftr_d", "d_ftmfga_d"],
    "FULL_FTA": [*CORE3, "d_ftr_d", *CONTEXT, "shrunk_dev_diff"],
    "FULL_FTM": [*CORE3, "d_ftmfga_d", *CONTEXT, "shrunk_dev_diff"],
}
NEEDED = sorted({c for cols in MODELS.values() for c in cols} - {"shrunk_dev_diff"})


def run() -> pd.DataFrame:
    df = pd.read_parquet(TABLE)
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    df = df.merge(lines, on="game_id", how="left")
    seasons = sorted(df["season"].unique(), key=lambda s: int(s[:4]))

    out = []
    for i in range(3, len(seasons)):
        tr = df[df["season"].isin(set(seasons[:i]))].copy()
        te = df[df["season"] == seasons[i]].copy()
        tr, te = add_shrunk_form(tr, te)

        need = [*NEEDED, "shrunk_dev_diff", "home_margin", "home_win"]
        tr_e = tr.dropna(subset=need)
        te_e = te.dropna(subset=[*need, "vegas_home_margin"])  # matched-to-Vegas test rows
        if len(tr_e) < 500 or len(te_e) < 50 or te_e["home_win"].nunique() < 2:
            continue

        rec = te_e[["season", "home_margin", "home_win", "vegas_home_margin"]].copy()
        for name, feats in MODELS.items():
            lin = LinearRegression().fit(tr_e[feats], tr_e["home_margin"])
            rec[f"pred_{name}"] = lin.predict(te_e[feats])
            sc = StandardScaler().fit(tr_e[feats])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(tr_e[feats]), tr_e["home_win"])
            rec[f"p_{name}"] = clf.predict_proba(sc.transform(te_e[feats]))[:, 1]
        # Vegas win prob: logistic of win on closing margin, train-fit.
        vclf = (
            LogisticRegression(max_iter=1000).fit(
                tr_e[["vegas_home_margin"]].fillna(0), tr_e["home_win"]
            )
            if tr_e["vegas_home_margin"].notna().sum() > 100
            else None
        )
        rec["p_VEGAS"] = vclf.predict_proba(te_e[["vegas_home_margin"]])[:, 1] if vclf else np.nan
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    r = run()
    n = len(r)
    print(f"Head-to-head on {n:,} Vegas-matched test games (expanding-window OOS).\n")
    rows = []
    for name in MODELS:
        rows.append(
            (
                name,
                mean_absolute_error(r["home_margin"], r[f"pred_{name}"]),
                log_loss(r["home_win"], r[f"p_{name}"]),
            )
        )
    rows.append(
        (
            "VEGAS_line",
            mean_absolute_error(r["home_margin"], r["vegas_home_margin"]),
            log_loss(r["home_win"], r["p_VEGAS"]) if r["p_VEGAS"].notna().all() else float("nan"),
        )
    )
    tab = pd.DataFrame(rows, columns=["model", "margin_MAE", "win_logloss"]).set_index("model")
    tab["MAE_gap_vs_vegas"] = tab["margin_MAE"] - tab.loc["VEGAS_line", "margin_MAE"]
    print(tab.round(4).to_string())

    # Where is the model closest to the line? Bucket by closing-spread magnitude.
    best = (
        "FULL_FTM"
        if tab.loc["FULL_FTM", "margin_MAE"] <= tab.loc["FULL_FTA", "margin_MAE"]
        else "FULL_FTA"
    )
    r = r.assign(absline=r["vegas_home_margin"].abs())
    r["bucket"] = pd.cut(
        r["absline"],
        [-0.1, 2.5, 5.5, 9.5, 100],
        labels=["pick'em (0-2.5)", "small (3-5.5)", "medium (6-9.5)", "large (10+)"],
    )
    g = r.groupby("bucket", observed=True).apply(
        lambda d: pd.Series(
            {
                "n": len(d),
                f"{best}_MAE": mean_absolute_error(d["home_margin"], d[f"pred_{best}"]),
                "vegas_MAE": mean_absolute_error(d["home_margin"], d["vegas_home_margin"]),
            }
        ),
        include_groups=False,
    )
    g["gap"] = g[f"{best}_MAE"] - g["vegas_MAE"]
    print(f"\nMAE by closing-spread magnitude (best model = {best}):")
    print(g.round(3).to_string())
    print(
        "\nLower margin_MAE is better; gap is model minus Vegas (negative = model beats the line)."
    )


if __name__ == "__main__":
    main()
