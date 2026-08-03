"""Frozen-at-game-X forecast: the honest "beat Vegas by game X" measuring stick.

At forecast point X (after a team has played X games), predict its FINAL
regular-season win total using only information through game X:

    forecast_wins = wins_so_far(X) + sum over remaining games of P(win)

P(win) comes from a four-factor power rating frozen at game X. Each team's
rating is beta . (its season-to-date four-factor differentials at game X), where
beta and the home-court intercept are fit on prior seasons; a game's expected
margin is home_rating - away_rating +/- home court, mapped to a win probability
by a trained logistic. The remaining schedule and home/away are known, so this
is a genuine forecast of final wins from a fixed information point.

Both the model (at each X) and the Vegas preseason line (X = 0) are then
forecasts of the same target, final wins, so comparing their MAE is fair. The
output is MAE versus X, and the X at which the model's forecast overtakes the
preseason line. A current-pace extrapolation is included as a naive floor.
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
VEGAS = REPO / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"
FIG = REPO / "figures" / "modeling"

RATING_FEATS = ["sd_efg_d", "sd_oreb_d", "sd_tov_d", "sd_ftmfga_d"]
DIFF_FEATS = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
X_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]


def team_timeline() -> pd.DataFrame:
    """Per team-game: frozen season-to-date four-factor features, schedule, result."""
    tf = _add_trailing(_add_composite(_team_game_frame()))
    tf = tf.sort_values(["team_id", "season", "game_date"])
    tf["gameno"] = tf.groupby(["team_id", "season"]).cumcount() + 1
    tf["won"] = (tf["margin"] > 0).astype(int)
    return tf[["season", "team_id", "opp_team_id", "is_home", "gameno", "won", *RATING_FEATS]]


def forecast() -> pd.DataFrame:
    tl = team_timeline()
    table = pd.read_parquet(TABLE)
    veg = pd.read_parquet(VEGAS)[["season", "team_id", "win_total"]]
    actual = tl.groupby(["season", "team_id"])["won"].sum().rename("actual_wins")

    seasons = sorted(tl["season"].unique(), key=lambda s: int(s[:4]))
    rows = []
    for i in range(3, len(seasons)):
        train_s = set(seasons[:i])
        test_s = seasons[i]

        # Fit four-factor margin model and the margin->win mapping on prior seasons.
        tr = table[table["season"].isin(train_s)].dropna(
            subset=[*DIFF_FEATS, "home_margin", "home_win"]
        )
        if len(tr) < 500:
            continue
        lin = LinearRegression().fit(tr[DIFF_FEATS], tr["home_margin"])
        beta, hca = lin.coef_, lin.intercept_
        pred_tr = lin.predict(tr[DIFF_FEATS]).reshape(-1, 1)
        winmap = LogisticRegression(max_iter=1000).fit(pred_tr, tr["home_win"])

        te = tl[tl["season"] == test_s]
        teams = te["team_id"].unique()

        for X in X_GRID:
            # Frozen rating at game X for every team.
            rating = {}
            wins_so_far = {}
            for t in teams:
                g = te[te["team_id"] == t].sort_values("gameno")
                at = g[g["gameno"] == X]
                if at.empty:
                    at = g.iloc[[-1]]  # short season fallback
                feats = at[RATING_FEATS].to_numpy()[0]
                rating[t] = float(np.dot(beta, np.nan_to_num(feats)))
                wins_so_far[t] = int(g[g["gameno"] <= X]["won"].sum())

            for t in teams:
                g = te[te["team_id"] == t].sort_values("gameno")
                rem = g[g["gameno"] > X]
                exp_rem = 0.0
                for _, gm in rem.iterrows():
                    opp_r = rating.get(gm["opp_team_id"], 0.0)
                    pm = rating[t] - opp_r + (hca if gm["is_home"] else -hca)
                    exp_rem += float(winmap.predict_proba([[pm]])[0, 1])
                rows.append(
                    dict(
                        season=test_s,
                        team_id=t,
                        X=X,
                        forecast=wins_so_far[t] + exp_rem,
                        pace=(wins_so_far[t] / X) * len(g),
                    )
                )
    fc = pd.DataFrame(rows)
    fc = fc.merge(actual.reset_index(), on=["season", "team_id"], how="left")
    fc = fc.merge(veg, on=["season", "team_id"], how="left")
    return fc


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fc = forecast()
    have_line = fc.dropna(subset=["win_total"]).copy()  # same set for a fair 3-way

    vegas_mae = float(
        (
            have_line.drop_duplicates(["season", "team_id"])["win_total"]
            - have_line.drop_duplicates(["season", "team_id"])["actual_wins"]
        )
        .abs()
        .mean()
    )

    g = have_line.groupby("X").apply(
        lambda d: pd.Series(
            {
                "model_MAE": (d["forecast"] - d["actual_wins"]).abs().mean(),
                "pace_MAE": (d["pace"] - d["actual_wins"]).abs().mean(),
                "n": d["team_id"].size,
            }
        ),
        include_groups=False,
    )
    g["vegas_MAE"] = vegas_mae

    print("Final-win forecast MAE by freeze point X (team-seasons with a Vegas line):\n")
    print(g[["model_MAE", "pace_MAE", "vegas_MAE"]].round(3).to_string())
    crossed = g.index[g["model_MAE"] < vegas_mae]
    msg = (
        f"model beats the preseason line from game X = {int(crossed.min())}"
        if len(crossed)
        else "model does not beat the preseason line at any tested X"
    )
    print(f"\nVegas preseason line MAE = {vegas_mae:.2f} wins. {msg}.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g.index, g["model_MAE"], marker="o", color="#534AB7", label="frozen-at-X model")
    ax.plot(g.index, g["pace_MAE"], marker="o", color="#888780", label="current-pace extrapolation")
    ax.axhline(vegas_mae, color="#D62728", ls="--", label=f"Vegas preseason line ({vegas_mae:.2f})")
    ax.set_xlabel("Forecast made after game X")
    ax.set_ylabel("Final win-total MAE (wins)")
    ax.set_title("Forecasting final wins: when does the model overtake the preseason line?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "frozen_forecast_vs_vegas.png", dpi=130)
    plt.close(fig)
    print(f"\nFigure written to {FIG / 'frozen_forecast_vs_vegas.png'}")


if __name__ == "__main__":
    main()
