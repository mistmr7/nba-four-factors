"""Seed-to-model blend, constant-prior vs forgetting-prior, raced vs the market.

The constant-precision blend alpha(n)=n/(n+k) assumes team strength is a fixed
quantity that the seed and the model both estimate. It is not: strength drifts
within a season (deadline trades, injuries, young teams improving). Under that
assumption the preseason seed keeps a 1/n tail of weight forever (about 15% at
game 40), which is harmless on average but actively wrong for teams that diverge
from their preseason line.

The fix is a forgetting prior (the Kalman / Glicko move): each game inflates the
prior's uncertainty, so old information, the seed most of all, decays
geometrically toward zero. The seed weight becomes

    seed_w(n) = k * exp(-n / tau) / (k * exp(-n / tau) + n)

with tau the forgetting timescale. tau = 10 is a stated modeling choice (not
tuned): it leaves the seed ~50% at game 5, ~10% at 15, ~1% at 30, ~0 by 40, so
the prior is gone once the four-factor estimate is mature. We compare it to the
constant blend overall and specifically on teams whose realized strength
diverged most from the preseason line, where a stale prior should hurt.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from .features import _team_game_frame
from .frozen_vs_market import game_level, market_ratings, team_timeline

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
VEGAS = REPO / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"
FIG = REPO / "figures" / "modeling"

RATING_FEATS = ["sd_efg_d", "sd_oreb_d", "sd_tov_d", "sd_ftmfga_d"]
DIFF_FEATS = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
SINGLE_FEATS = ["efg_d", "oreb_d", "tov_d", "ftmfga_d"]
X_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
TAU = 10.0  # forgetting timescale (games); stated choice, not tuned


def _win_prob(m, sigma):
    return norm.cdf(m / sigma)


def project(te, X, rating, hca, sigma):
    banked = te[te["gameno"] <= X].groupby("team_id")["won"].sum()
    rem = te[te["gameno"] > X]
    if rem.empty:
        return banked.astype(float)
    rt = rem["team_id"].map(rating).fillna(0.0).to_numpy()
    ro = rem["opp_team_id"].map(rating).fillna(0.0).to_numpy()
    sgn = np.where(rem["is_home"], 1.0, -1.0)
    p = _win_prob(rt - ro + sgn * hca, sigma)
    exp = pd.Series(p, index=rem["team_id"].to_numpy()).groupby(level=0).sum()
    return banked.astype(float).add(exp, fill_value=0.0)


def estimate_k(train_seasons, single, beta, netmap, wt):
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
    return float(np.clip(sigma_game2 / sigma_prior2, 1.0, 60.0)), a_s, b_s


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

    rows, ks, div = [], [], {}
    for i in range(3, len(seasons)):
        test_s = seasons[i]
        gl_te = gl[gl["season"] == test_s]
        if gl_te["vegas_home_margin"].notna().sum() < 200:
            continue
        train_seasons = set(seasons[:i])
        train = tbl[tbl["season"].isin(train_seasons)].dropna(
            subset=[*DIFF_FEATS, "home_margin", "home_win"]
        )
        lin = LinearRegression().fit(train[DIFF_FEATS], train["home_margin"])
        beta, hca = lin.coef_, lin.intercept_
        sigma = float(np.std(train["home_margin"] - lin.predict(train[DIFF_FEATS])))
        k, a_s, b_s = estimate_k(train_seasons, single, beta, netmap, wt)
        ks.append(k)

        te = tl[tl["season"] == test_s]
        teams = list(te["team_id"].unique())
        seed = {t: a_s * wt.get((test_s, t), np.nan) + b_s for t in teams}
        for t in teams:
            if (test_s, t) in netmap and np.isfinite(seed[t]):
                div[(test_s, t)] = abs(netmap[(test_s, t)] - seed[t])

        for X in X_GRID:
            ff = {}
            for t in teams:
                g = te[te["team_id"] == t].sort_values("gameno")
                at = g[g["gameno"] == X]
                at = at if not at.empty else g.iloc[[-1]]
                ff[t] = float(np.dot(beta, np.nan_to_num(at[RATING_FEATS].to_numpy()[0])))
            sw_const = k / (k + X)
            ek = k * np.exp(-X / TAU)
            sw_decay = ek / (ek + X)
            const_blend = {t: sw_const * seed[t] + (1 - sw_const) * ff[t] for t in teams}
            decay_blend = {t: sw_decay * seed[t] + (1 - sw_decay) * ff[t] for t in teams}
            obs = gl_te[(gl_te["home_gameno"] <= X) & (gl_te["away_gameno"] <= X)].dropna(
                subset=["vegas_home_margin"]
            )
            krate, hca_k = (
                market_ratings(obs, teams)
                if len(obs) >= len(teams)
                else ({t: 0.0 for t in teams}, hca)
            )

            for label, rating, h in [
                ("seed_only", seed, hca),
                ("no_seed", ff, hca),
                ("const_blend", const_blend, hca),
                ("decay_blend", decay_blend, hca),
                ("market", krate, hca_k),
            ]:
                fc = project(te, X, rating, h, sigma)
                for t, val in fc.items():
                    rows.append(
                        dict(season=test_s, team_id=int(t), X=X, method=label, forecast=val)
                    )
    fc = pd.DataFrame(rows).merge(actual.reset_index(), on=["season", "team_id"])
    fc = fc.merge(veg, on=["season", "team_id"], how="left").dropna(subset=["win_total"])
    fc["divergence"] = [
        div.get((s, t), np.nan) for s, t in zip(fc["season"], fc["team_id"], strict=False)
    ]
    return fc, float(np.mean(ks))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fc, k = run()
    fc["ae"] = (fc["forecast"] - fc["actual_wins"]).abs()
    pre = float(
        (
            fc.drop_duplicates(["season", "team_id"])["win_total"]
            - fc.drop_duplicates(["season", "team_id"])["actual_wins"]
        )
        .abs()
        .mean()
    )
    tab = fc.pivot_table(index="X", columns="method", values="ae", aggfunc="mean")[
        ["seed_only", "no_seed", "const_blend", "decay_blend", "market"]
    ]
    print(f"k = {k:.1f}, tau = {TAU:.0f}\n\nFinal-win MAE by freeze point X:\n")
    print(tab.round(3).to_string())
    print(f"\npreseason line MAE = {pre:.2f}")

    # Divergence-stratified: where a stale seed should hurt.
    d = (
        fc[fc["method"].isin(["const_blend", "decay_blend"]) & fc["X"].isin([30, 40])]
        .dropna(subset=["divergence"])
        .copy()
    )
    terc = (
        d.drop_duplicates(["season", "team_id"])["divergence"].quantile([1 / 3, 2 / 3]).to_numpy()
    )
    d["bucket"] = np.where(
        d["divergence"] <= terc[0],
        "low diverge",
        np.where(d["divergence"] <= terc[1], "mid", "high diverge"),
    )
    strat = d.pivot_table(index=["bucket", "X"], columns="method", values="ae", aggfunc="mean")
    strat["decay_minus_const"] = strat["decay_blend"] - strat["const_blend"]
    print("\nMAE by preseason-line divergence (negative decay_minus_const = forgetting helps):")
    print(strat.round(3).to_string())

    # Race figure with both blends.
    fig, ax = plt.subplots(figsize=(10, 5))
    sty = {
        "seed_only": ("#1D9E75", "Vegas seed only"),
        "no_seed": ("#888780", "four-factor only"),
        "const_blend": ("#9E86FF", "constant-prior blend"),
        "decay_blend": ("#534AB7", "forgetting-prior blend"),
        "market": ("#D62728", "Vegas in-season"),
    }
    for m, (c, lab) in sty.items():
        ax.plot(tab.index, tab[m], marker="o", color=c, label=lab)
    ax.axhline(pre, color="#333", ls="--", lw=1, label=f"preseason line ({pre:.2f})")
    ax.set_xlabel("Forecast made after game X")
    ax.set_ylabel("Final win-total MAE (wins)")
    ax.set_title("Constant-prior vs forgetting-prior seed blend")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "frozen_blend.png", dpi=130)
    plt.close(fig)

    # Seed-weight decay figure: the whole point.
    n = np.arange(1, 83)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n, k / (k + n), color="#9E86FF", lw=2, label=f"constant prior  k/(n+k), k={k:.1f}")
    ek = k * np.exp(-n / TAU)
    ax.plot(n, ek / (ek + n), color="#534AB7", lw=2, label=f"forgetting prior  (tau={TAU:.0f})")
    for gx in (20, 40):
        ax.axvline(gx, color="#ccc", lw=0.8)
    ax.set_xlabel("Games played (n)")
    ax.set_ylabel("Weight still on the Vegas seed")
    ax.set_title("Forgetting the preseason seed: weight decays to zero by ~game 30")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "seed_weight_decay.png", dpi=130)
    plt.close(fig)
    print(f"\nFigures: {FIG / 'frozen_blend.png'} and {FIG / 'seed_weight_decay.png'}")


if __name__ == "__main__":
    main()
