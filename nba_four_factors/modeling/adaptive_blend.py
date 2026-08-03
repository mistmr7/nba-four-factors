"""Adaptive seed forgetting: discount the prior in proportion to disagreement.

The constant blend keeps seed weight k/(k+n) for every team. The adaptive blend
keeps that for teams whose play agrees with their preseason line, and forgets the
seed faster for teams whose play contradicts it. The disagreement is the
standardized gap between a team's four-factor rating and its seed:

    D = (four_factor_rating - seed)^2 / (sigma_game^2 / n + sigma_prior^2)

D is about 1 when the seed is consistent with the data (the gap is what sampling
plus prior noise would produce) and grows when the team has genuinely departed
from its line. The seed weight becomes

    inflation     = 1 + max(0, D - 1)
    seed_weight   = k / (k + n * inflation)

so a consistent team (D<=1, inflation=1) gets exactly the constant blend, and a
divergent team behaves as if it had played n*inflation games of contradicting
evidence, shedding the seed. This is a closed-form robust-Bayes / adaptive-Kalman
update: the prior is down-weighted by the innovation it failed to predict, with
no extra tuned parameter (D~1 is the no-divergence null).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .features import _team_game_frame
from .frozen_blend import project
from .frozen_vs_market import game_level, market_ratings, team_timeline

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
VEGAS = REPO / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"

RATING_FEATS = ["sd_efg_d", "sd_oreb_d", "sd_tov_d", "sd_ftmfga_d"]
DIFF_FEATS = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
SINGLE_FEATS = ["efg_d", "oreb_d", "tov_d", "ftmfga_d"]
X_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
INFL_CAP = 10.0
EXAMPLE_SEASON = "2025_26"


def abbr_map() -> dict:
    df = pd.read_parquet(PROC / "2024_25" / "regular_season.parquet")
    return {
        int(t): a for t, a in df[["team_id", "team_abbr"]].drop_duplicates().itertuples(index=False)
    }


def fold_params(train_seasons, single, tbl, netmap, wt):
    train = tbl[tbl["season"].isin(train_seasons)].dropna(
        subset=[*DIFF_FEATS, "home_margin", "home_win"]
    )
    lin = LinearRegression().fit(train[DIFF_FEATS], train["home_margin"])
    beta, hca = lin.coef_, lin.intercept_
    win_sigma = float(np.std(train["home_margin"] - lin.predict(train[DIFF_FEATS])))
    s = single[single["season"].isin(train_seasons)].copy()
    s["rate"] = s[SINGLE_FEATS].to_numpy() @ beta
    sigma_game2 = float(s.groupby(["season", "team_id"])["rate"].var(ddof=1).mean())
    pairs = [
        (wt.get((ss, t)), netmap.get((ss, t)))
        for ss in train_seasons
        for t in s.loc[s["season"] == ss, "team_id"].unique()
    ]
    arr = np.array(
        [
            (a, b)
            for a, b in pairs
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
        ]
    )
    a_s, b_s = np.polyfit(arr[:, 0], arr[:, 1], 1)
    sigma_prior2 = float(np.var(arr[:, 1] - (a_s * arr[:, 0] + b_s)))
    k = float(np.clip(sigma_game2 / sigma_prior2, 1.0, 60.0))
    return beta, hca, win_sigma, sigma_game2, sigma_prior2, k, a_s, b_s


def run():
    tl = team_timeline()
    gl = game_level(tl)
    single = _team_game_frame()[["season", "team_id", *SINGLE_FEATS]]
    veg = pd.read_parquet(VEGAS)[["season", "team_id", "win_total"]]
    wt = {(s, int(t)): float(w) for s, t, w in veg.itertuples(index=False)}
    netmap = {
        (s, int(t)): float(v)
        for (s, t), v in _team_game_frame().groupby(["season", "team_id"])["margin"].mean().items()
    }
    actual = tl.groupby(["season", "team_id"])["won"].sum().rename("actual_wins")
    tbl = pd.read_parquet(TABLE)
    seasons = sorted(tl["season"].unique(), key=lambda s: int(s[:4]))

    rows, ex = [], []
    for i in range(3, len(seasons)):
        test_s = seasons[i]
        gl_te = gl[gl["season"] == test_s]
        if gl_te["vegas_home_margin"].notna().sum() < 200:
            continue
        beta, hca, wsig, sg2, sp2, k, a_s, b_s = fold_params(
            set(seasons[:i]), single, tbl, netmap, wt
        )
        te = tl[tl["season"] == test_s]
        teams = list(te["team_id"].unique())
        seed = {t: a_s * wt.get((test_s, t), np.nan) + b_s for t in teams}

        for X in X_GRID:
            ff = {}
            for t in teams:
                g = te[te["team_id"] == t].sort_values("gameno")
                at = g[g["gameno"] == X]
                at = at if not at.empty else g.iloc[[-1]]
                ff[t] = float(np.dot(beta, np.nan_to_num(at[RATING_FEATS].to_numpy()[0])))
            sw_c = k / (k + X)
            const_b, adapt_b = {}, {}
            for t in teams:
                D = (ff[t] - seed[t]) ** 2 / (sg2 / X + sp2)
                infl = min(1.0 + max(0.0, D - 1.0), INFL_CAP)
                sw_a = k / (k + X * infl)
                const_b[t] = sw_c * seed[t] + (1 - sw_c) * ff[t]
                adapt_b[t] = sw_a * seed[t] + (1 - sw_a) * ff[t]
                if test_s == EXAMPLE_SEASON and X == 30:
                    ex.append(
                        dict(
                            team_id=t,
                            seed=seed[t],
                            ff=ff[t],
                            D=D,
                            infl=infl,
                            sw_const=sw_c,
                            sw_adapt=sw_a,
                        )
                    )
            obs = gl_te[(gl_te["home_gameno"] <= X) & (gl_te["away_gameno"] <= X)].dropna(
                subset=["vegas_home_margin"]
            )
            krate, hca_k = (
                market_ratings(obs, teams)
                if len(obs) >= len(teams)
                else ({t: 0.0 for t in teams}, hca)
            )
            for label, rating, h in [
                ("const_blend", const_b, hca),
                ("adaptive_blend", adapt_b, hca),
                ("market", krate, hca_k),
            ]:
                fc = project(te, X, rating, h, wsig)
                for t, val in fc.items():
                    rows.append(
                        dict(season=test_s, team_id=int(t), X=X, method=label, forecast=val)
                    )
        # divergence per team-season (fixed reference at full season): |realized net - seed|
    fc = pd.DataFrame(rows).merge(actual.reset_index(), on=["season", "team_id"])
    fc = fc.merge(veg, on=["season", "team_id"], how="left").dropna(subset=["win_total"])
    return fc, pd.DataFrame(ex)


def main():
    fc, ex = run()
    # divergence: |realized net - seed| using a global seed fit for stratification only
    single = _team_game_frame()
    net = single.groupby(["season", "team_id"])["margin"].mean()
    veg = pd.read_parquet(VEGAS)[["season", "team_id", "win_total"]].dropna()
    j = veg.merge(net.reset_index().rename(columns={"margin": "net"}), on=["season", "team_id"])
    a_s, b_s = np.polyfit(j["win_total"], j["net"], 1)
    seednet = {(s, int(t)): a_s * w + b_s for s, t, w in veg.itertuples(index=False)}
    dmap = {
        (s, int(t)): abs(float(v) - seednet.get((s, int(t)), np.nan))
        for (s, t), v in net.items()
        if (s, int(t)) in seednet
    }
    fc["divergence"] = [
        dmap.get((s, t), np.nan) for s, t in zip(fc["season"], fc["team_id"], strict=False)
    ]
    fc["ae"] = (fc["forecast"] - fc["actual_wins"]).abs()

    tab = fc.pivot_table(index="X", columns="method", values="ae", aggfunc="mean")[
        ["const_blend", "adaptive_blend", "market"]
    ]
    print("Final-win MAE by freeze point X:\n")
    print(tab.round(3).to_string())

    d = (
        fc[fc["method"].isin(["const_blend", "adaptive_blend"]) & fc["X"].isin([30, 40])]
        .dropna(subset=["divergence"])
        .copy()
    )
    terc = (
        d.drop_duplicates(["season", "team_id"])["divergence"].quantile([1 / 3, 2 / 3]).to_numpy()
    )
    d["bucket"] = np.where(
        d["divergence"] <= terc[0],
        "1 low diverge",
        np.where(d["divergence"] <= terc[1], "2 mid", "3 high diverge"),
    )
    strat = d.pivot_table(index=["bucket", "X"], columns="method", values="ae", aggfunc="mean")
    strat["adapt_minus_const"] = strat["adaptive_blend"] - strat["const_blend"]
    print("\nMAE by preseason-line divergence (negative = adaptive helps):")
    print(strat.round(3).to_string())

    am = abbr_map()
    ex["abbr"] = ex["team_id"].map(am)
    ex = ex.sort_values("D", ascending=False)
    show = pd.concat([ex.head(3), ex.tail(3)])
    print(f"\nWorked example, {EXAMPLE_SEASON} at game 30 (k applied; sw=seed weight):")
    print(
        show[["abbr", "seed", "ff", "D", "infl", "sw_const", "sw_adapt"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
