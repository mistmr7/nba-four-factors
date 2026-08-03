"""Does stylistic interplay between teams carry predictive signal?

Two tests, both causal (every team tendency is a trailing season-to-date mean
computed from prior games only).

Part A, direct measurement. Beyond the additive expectation (your offense plus
their defense), does the opponent's STYLE bend your single-game performance? The
headline case is the one we hypothesized: does facing a strong offensive-rebounding
team suppress your eFG, and does a fast opponent change your pace or shooting? We
regress a team's single-game factor on its own tendency, the opponent's matching
defensive tendency, and the opponent's cross-factor style, and read the cross term.

Part B, out-of-sample value. We add a small, pre-declared set of cross-team
interaction terms to the additive four-factor model and test, expanding-window,
whether they lower margin error. If targeted interactions cannot beat the additive
baseline out of sample, stylistic interplay carries no exploitable predictive
weight at the team-four-factor level.

Run: python -m nba_four_factors.modeling.matchup_probe
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
TUNE_BEFORE = 2010

FACTORS = [
    "off_efg",
    "def_efg",
    "off_oreb",
    "def_oreb",
    "off_tov",
    "def_tov",
    "off_ftr",
    "def_ftr",
    "pace",
]


def team_game_table() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "*/regular_season.parquet")))],
        ignore_index=True,
    )
    df = df[(~df["is_neutral"]) & (df["season"] != "1998_99")].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["off_efg"] = df["off_efg_pct"]
    df["def_efg"] = df["def_efg_pct"]
    df["off_oreb"] = df["off_orb_pct"]
    df["def_oreb"] = df["def_orb_pct"]
    df["off_tov"] = df["off_tov_pct"]
    df["def_tov"] = df["def_tov_pct"]
    df["off_ftr"] = df["ftm"] / df["fga"]
    df["def_ftr"] = df["opp_ftm"] / df["opp_fga"]
    df["pace"] = 0.5 * (
        (df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"])
        + (df["opp_fga"] + 0.44 * df["opp_fta"] - df["opp_oreb"] + df["opp_tov"])
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FACTORS)
    df = df.sort_values(["team_id", "season", "game_date"])
    # Trailing season-to-date tendency for each factor (prior games only).
    for c in FACTORS:
        df[c + "_t"] = df.groupby(["team_id", "season"])[c].transform(
            lambda s: s.shift(1).expanding().mean()
        )
    return df


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    xtx = X1.T @ X1
    beta = np.linalg.solve(xtx, X1.T @ y)
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(xtx)))
    t = beta / se
    return beta, t, resid


def part_a(tg: pd.DataFrame):
    print("=" * 72)
    print("PART A. Does opponent style bend your single-game performance?")
    print("  (pooled team perspective; all predictors are trailing tendencies)")
    print("=" * 72)
    cols = FACTORS + [c + "_t" for c in FACTORS]
    g = tg.dropna(subset=[c + "_t" for c in FACTORS])
    # Build opponent trailing tendencies by joining on (game_id, the other team).
    base = g[["game_id", "team_id", *cols]].copy()
    opp = g[["game_id", "team_id"] + [c + "_t" for c in FACTORS]].copy()
    opp = opp.rename(
        columns={**{"team_id": "opp_id"}, **{c + "_t": "opp_" + c + "_t" for c in FACTORS}}
    )
    merged = base.merge(opp, on="game_id")
    merged = merged[merged["team_id"] != merged["opp_id"]]

    def reg(y_col, own_t, oppdef_t, cross_terms, label):
        d = merged.dropna()
        y = d[y_col].to_numpy(dtype=float)
        names = [own_t, oppdef_t, *cross_terms]
        X = d[names].to_numpy(dtype=float)
        beta, t, _ = ols(X, y)
        print(f"\n  {label}")
        print(f"    outcome: single-game {y_col}")
        for nm, b, tt in zip(names, beta[1:], t[1:], strict=False):
            star = "  <-- cross-style" if nm.startswith("opp_") and nm not in (oppdef_t,) else ""
            print(f"      {nm:18s} beta {b:+.4f}  t {tt:+6.1f}{star}")

    reg(
        "off_efg",
        "off_efg_t",
        "opp_def_efg_t",
        ["opp_off_oreb_t", "opp_pace_t"],
        "Your shooting vs opponent rebounding/pace (the OREB-suppresses-eFG test)",
    )
    reg(
        "pace",
        "pace_t",
        "opp_pace_t",
        [],
        "Game pace: do the two paces simply average, or interact?",
    )
    reg(
        "off_oreb",
        "off_oreb_t",
        "opp_def_oreb_t",
        ["opp_pace_t"],
        "Your offensive rebounding vs opponent pace",
    )


def standardize_levels(tg, train_mask):
    lvl = ["off_efg_t", "off_oreb_t", "off_tov_t", "off_ftr_t", "pace_t", "def_efg_t", "def_oreb_t"]
    mu = tg.loc[train_mask, lvl].mean()
    sd = tg.loc[train_mask, lvl].std()
    return lvl, mu, sd


def build_game_features():
    tg = team_game_table()
    m = pd.read_parquet(TABLE).dropna(subset=["home_margin", "home_win"]).copy()
    m["yr"] = m["season"].str[:4].astype(int)
    tcols = [c + "_t" for c in FACTORS]
    key = tg[["game_id", "team_id", *tcols]].dropna()
    h = key.rename(columns={**{"team_id": "team_id"}, **{c: "h_" + c for c in tcols}})
    a = key.rename(columns={**{"team_id": "away_team_id"}, **{c: "a_" + c for c in tcols}})
    m = m.merge(h, on=["game_id", "team_id"], how="inner")
    m = m.merge(a, on=["game_id", "away_team_id"], how="inner")
    return m


def zser(s, mu, sd):
    return (s - mu) / sd


def part_b(m):
    print("\n" + "=" * 72)
    print("PART B. Do targeted interaction terms beat the additive model OOS?")
    print("=" * 72)
    train = m["yr"] < TUNE_BEFORE
    lvl = ["off_efg_t", "off_oreb_t", "off_tov_t", "off_ftr_t", "pace_t", "def_efg_t", "def_oreb_t"]
    mu = {c: m.loc[train, "h_" + c].mean() for c in lvl}
    sd = {c: m.loc[train, "h_" + c].std() for c in lvl}

    def z(side, c):
        return ((m[f"{side}_{c}"] - mu[c]) / sd[c]).to_numpy()

    add = np.column_stack([m["d_efg_d"], m["d_oreb_d"], m["d_tov_d"], m["d_ftmfga_d"]])
    # Pre-declared cross-team style interactions (standardized deviations).
    inter = np.column_stack(
        [
            z("h", "off_oreb_t") * z("a", "off_efg_t"),
            z("a", "off_oreb_t") * z("h", "off_efg_t"),
            z("h", "off_oreb_t") * z("a", "pace_t"),
            z("a", "off_oreb_t") * z("h", "pace_t"),
            z("h", "pace_t") * z("a", "pace_t"),
            (z("h", "pace_t") - z("a", "pace_t")) ** 2,
        ]
    )
    inter_names = [
        "hOREBxaEFG",
        "aOREBxhEFG",
        "hOREBxaPACE",
        "aOREBxhPACE",
        "PACExPACE",
        "PACEmismatch^2",
    ]
    y = m["home_margin"].to_numpy(dtype=float)
    w = m["home_win"].to_numpy(dtype=float)
    yr = m["yr"].to_numpy()

    def fit_eval(X):
        maes, lls, ins = [], [], []
        for ty in range(TUNE_BEFORE, yr.max() + 1):
            tr, te = yr < ty, yr == ty
            if tr.sum() < 2000 or te.sum() < 100:
                continue
            X1 = np.column_stack([np.ones(tr.sum()), X[tr]])
            beta = np.linalg.lstsq(X1, y[tr], rcond=None)[0]
            Xte = np.column_stack([np.ones(te.sum()), X[te]])
            pm = Xte @ beta
            sig = np.std(y[tr] - X1 @ beta) + 1e-6
            p = np.clip(
                0.5 * (1 + np.vectorize(math.erf)((pm) / (sig * math.sqrt(2)))), 1e-6, 1 - 1e-6
            )
            maes.append(np.mean(np.abs(pm - y[te])))
            lls.append(-np.mean(w[te] * np.log(p) + (1 - w[te]) * np.log(1 - p)))
            ins.append(np.mean(np.abs((X1 @ beta) - y[tr])))
        return np.mean(maes), np.mean(lls), np.mean(ins)

    mae0, ll0, in0 = fit_eval(add)
    mae1, ll1, in1 = fit_eval(np.column_stack([add, inter]))
    print(
        f"\n  additive four factors        : OOS MAE {mae0:.4f}  logloss {ll0:.4f}  (in-sample MAE {in0:.4f})"
    )
    print(
        f"  additive + 6 interaction terms: OOS MAE {mae1:.4f}  logloss {ll1:.4f}  (in-sample MAE {in1:.4f})"
    )
    print(f"  interaction effect on OOS MAE : {mae1 - mae0:+.4f}  (negative = interactions help)")
    print(
        f"  interaction effect in-sample  : {in1 - in0:+.4f}  (how much they fit the training noise)"
    )

    # Coefficients and t-stats of the interaction terms on the full set, for sign/size.
    Xall = np.column_stack([add, inter])
    beta, t, _ = ols(Xall, y)
    print("\n  interaction term coefficients (points, full-sample):")
    for nm, b, tt in zip(inter_names, beta[5:], t[5:], strict=False):
        sig = " significant" if abs(tt) > 2 else ""
        print(f"    {nm:16s} beta {b:+.3f}  t {tt:+6.1f}{sig}")


def main():
    tg = team_game_table()
    part_a(tg)
    m = build_game_features()
    part_b(m)
    print("\n" + "=" * 72)
    print("Read: a cross-style coefficient can be real (Part A) yet add nothing to")
    print("prediction (Part B) if it is small against single-game noise.")


if __name__ == "__main__":
    main()
