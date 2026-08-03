"""Frozen-at-X final-win forecast: our model vs Vegas's IN-SEASON number.

The preseason line is the wrong opponent because Vegas updates the win total
after every game. We reconstruct Vegas's in-season view from the per-game
closing spreads we already have: at freeze point X, solve a market power rating
from the spreads of all games played through X (ridge least squares of
closing_home_margin = rating_home - rating_away + home_court), then project each
team's remaining schedule with that market rating. That is Vegas's own
frozen-at-X final-win forecast, built from the same information horizon as ours,
so the comparison is apples-to-apples.

Our model uses a four-factor power rating frozen at X (same construction, ratings
from season-to-date four-factor differentials). Both forecast final wins as
banked wins plus the sum of projected remaining win probabilities. Reported
against actual final wins, with the preseason line as a static reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from .features import _add_composite, _add_trailing, _team_game_frame

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
LINES = REPO / "data" / "features" / "vegas_game_lines.parquet"
VEGAS = REPO / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"
FIG = REPO / "figures" / "modeling"

RATING_FEATS = ["sd_efg_d", "sd_oreb_d", "sd_tov_d", "sd_ftmfga_d"]
DIFF_FEATS = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
X_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
RIDGE = 1.0


def team_timeline() -> pd.DataFrame:
    tf = _add_trailing(_add_composite(_team_game_frame()))
    tf = tf.sort_values(["team_id", "season", "game_date"])
    tf["gameno"] = tf.groupby(["team_id", "season"]).cumcount() + 1
    tf["won"] = (tf["margin"] > 0).astype(int)
    return tf[
        ["season", "team_id", "opp_team_id", "is_home", "game_date", "gameno", "won", *RATING_FEATS]
    ]


def game_level(tl: pd.DataFrame) -> pd.DataFrame:
    """Game-level home-perspective frame with both teams' game numbers and the line."""
    home = tl[tl["is_home"]].rename(
        columns={"team_id": "home_id", "opp_team_id": "away_id", "gameno": "home_gameno"}
    )
    away = tl[~tl["is_home"]][["season", "team_id", "opp_team_id", "gameno"]].rename(
        columns={"team_id": "away_id", "opp_team_id": "home_id", "gameno": "away_gameno"}
    )
    tbl = pd.read_parquet(TABLE)[
        ["game_id", "season", "team_id", "away_team_id", "game_date", "home_win", "home_margin"]
    ]
    tbl = tbl.rename(columns={"team_id": "home_id", "away_team_id": "away_id"})
    tbl["game_date"] = pd.to_datetime(tbl["game_date"])
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    g = tbl.merge(lines, on="game_id", how="left")
    g = g.merge(
        home[["season", "home_id", "away_id", "game_date", "home_gameno"]],
        on=["season", "home_id", "away_id", "game_date"],
        how="left",
    )
    g = g.merge(away, on=["season", "home_id", "away_id"], how="left")
    return g


def market_ratings(obs: pd.DataFrame, teams: list[int]) -> tuple[dict, float]:
    """Ridge least-squares power rating from closing spreads of observed games."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    A = np.zeros((len(obs), n + 1))
    for r, (_, gm) in enumerate(obs.iterrows()):
        A[r, idx[gm["home_id"]]] = 1.0
        A[r, idx[gm["away_id"]]] = -1.0
        A[r, n] = 1.0  # home court
    y = obs["vegas_home_margin"].to_numpy()
    reg = RIDGE * np.eye(n + 1)
    reg[n, n] = 0.0  # do not shrink the home-court term
    sol = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return {t: float(sol[idx[t]]) for t in teams}, float(sol[n])


def run() -> pd.DataFrame:
    tl = team_timeline()
    gl = game_level(tl)
    veg = pd.read_parquet(VEGAS)[["season", "team_id", "win_total"]]
    actual = tl.groupby(["season", "team_id"])["won"].sum().rename("actual_wins")
    seasons = sorted(tl["season"].unique(), key=lambda s: int(s[:4]))
    tbl = pd.read_parquet(TABLE)

    rows = []
    for i in range(3, len(seasons)):
        test_s = seasons[i]
        gl_te = gl[gl["season"] == test_s]
        if gl_te["vegas_home_margin"].notna().sum() < 200:
            continue  # spreads only from 2007-08 on
        train = tbl[tbl["season"].isin(set(seasons[:i]))].dropna(
            subset=[*DIFF_FEATS, "home_margin", "home_win"]
        )
        lin = LinearRegression().fit(train[DIFF_FEATS], train["home_margin"])
        beta, hca_m = lin.coef_, lin.intercept_
        winmap_model = LogisticRegression(max_iter=1000).fit(
            lin.predict(train[DIFF_FEATS]).reshape(-1, 1), train["home_win"]
        )
        gl_tr = gl[gl["season"].isin(set(seasons[:i]))].dropna(
            subset=["vegas_home_margin", "home_win"]
        )
        winmap_mkt = (
            LogisticRegression(max_iter=1000).fit(gl_tr[["vegas_home_margin"]], gl_tr["home_win"])
            if len(gl_tr) > 200
            else winmap_model
        )

        te = tl[tl["season"] == test_s]
        teams = list(te["team_id"].unique())

        for X in X_GRID:
            # Model ratings + banked wins at X.
            mrate, wins_far = {}, {}
            for t in teams:
                g = te[te["team_id"] == t].sort_values("gameno")
                at = g[g["gameno"] == X]
                at = at if not at.empty else g.iloc[[-1]]
                mrate[t] = float(np.dot(beta, np.nan_to_num(at[RATING_FEATS].to_numpy()[0])))
                wins_far[t] = int(g[g["gameno"] <= X]["won"].sum())
            # Market ratings from spreads of games both teams have played through X.
            obs = gl_te[(gl_te["home_gameno"] <= X) & (gl_te["away_gameno"] <= X)].dropna(
                subset=["vegas_home_margin"]
            )
            krate, hca_k = (
                market_ratings(obs, teams)
                if len(obs) > n_min(teams)
                else ({t: 0.0 for t in teams}, hca_m)
            )

            for t in teams:
                g = te[te["team_id"] == t].sort_values("gameno")
                rem = g[g["gameno"] > X]
                em_model = em_mkt = 0.0
                for _, gm in rem.iterrows():
                    o = gm["opp_team_id"]
                    sgn = 1.0 if gm["is_home"] else -1.0
                    pm_model = mrate[t] - mrate.get(o, 0.0) + sgn * hca_m
                    pm_mkt = krate[t] - krate.get(o, 0.0) + sgn * hca_k
                    em_model += float(winmap_model.predict_proba([[pm_model]])[0, 1])
                    em_mkt += float(winmap_mkt.predict_proba([[pm_mkt]])[0, 1])
                rows.append(
                    dict(
                        season=test_s,
                        team_id=t,
                        X=X,
                        model=wins_far[t] + em_model,
                        market=wins_far[t] + em_mkt,
                        pace=(wins_far[t] / X) * len(g),
                    )
                )
    fc = (
        pd.DataFrame(rows)
        .merge(actual.reset_index(), on=["season", "team_id"])
        .merge(veg, on=["season", "team_id"], how="left")
    )
    return fc


def n_min(teams) -> int:
    return len(teams)  # need at least ~one game per team for a stable solve


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fc = run().dropna(subset=["win_total"])
    pre = float(
        (
            fc.drop_duplicates(["season", "team_id"])["win_total"]
            - fc.drop_duplicates(["season", "team_id"])["actual_wins"]
        )
        .abs()
        .mean()
    )
    g = fc.groupby("X").apply(
        lambda d: pd.Series(
            {
                "model_MAE": (d["model"] - d["actual_wins"]).abs().mean(),
                "market_MAE": (d["market"] - d["actual_wins"]).abs().mean(),
                "pace_MAE": (d["pace"] - d["actual_wins"]).abs().mean(),
            }
        ),
        include_groups=False,
    )
    g["preseason_line"] = pre
    print("Final-win forecast MAE by freeze point X (2008-2026 team-seasons):\n")
    print(g.round(3).to_string())
    print(
        f"\nMarket = Vegas IN-SEASON number rebuilt from closing spreads. Preseason line MAE = {pre:.2f}."
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g.index, g["model_MAE"], marker="o", color="#534AB7", label="our model (in-season)")
    ax.plot(
        g.index,
        g["market_MAE"],
        marker="o",
        color="#D62728",
        label="Vegas in-season (from spreads)",
    )
    ax.plot(g.index, g["pace_MAE"], marker="o", color="#888780", label="current pace")
    ax.axhline(pre, color="#333", ls="--", lw=1, label=f"Vegas preseason line ({pre:.2f})")
    ax.set_xlabel("Forecast made after game X")
    ax.set_ylabel("Final win-total MAE (wins)")
    ax.set_title("Our in-season forecast vs Vegas's in-season number (rebuilt from spreads)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "frozen_vs_market.png", dpi=130)
    plt.close(fig)
    print(f"\nFigure written to {FIG / 'frozen_vs_market.png'}")


if __name__ == "__main__":
    main()
