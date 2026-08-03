"""Season-level measuring stick: expected wins vs Vegas, and the MAE curve.

Predicts every game by blending the Vegas-seeded M0 (preseason win-total
differential) with the full model, the blend weight rising as each team
accumulates games: alpha = min(avg_games_played / K, 1). Early season leans on
the seed (no trailing features yet), late season on the model. This is the
seed-to-model handoff from the Session 14 plan.

Two outputs, both expanding-window out-of-sample:
* Season expected-wins MAE: sum each team's per-game win probability across its
  schedule, compare to actual wins. Three columns: Vegas preseason line,
  seed-only (M0), and blended model, on the same test team-seasons.
* MAE-by-game-number curve: game-level margin MAE for the blended model and the
  Vegas closing line as a function of how far into the season the game is.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from .cv import FACTORS, add_shrunk_form

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
LINES = REPO / "data" / "features" / "vegas_game_lines.parquet"
FIG = REPO / "figures" / "modeling"

CONTEXT = ["rest_diff", "b2b_diff", "miles7d_diff"]
FULL = FACTORS + CONTEXT + ["shrunk_dev_diff"]
K = 20  # games after which the model fully takes over from the seed


def games_played_map() -> dict:
    """(game_id, team_id) -> game number within the team's season (1-based)."""
    files = sorted(glob.glob(str(PROC / "*/regular_season.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[~df["is_neutral"]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_id", "season", "game_date"])
    df["gameno"] = df.groupby(["team_id", "season"]).cumcount() + 1
    return {
        (g, int(t)): int(n)
        for g, t, n in df[["game_id", "team_id", "gameno"]].itertuples(index=False)
    }


def actual_wins() -> pd.DataFrame:
    files = sorted(glob.glob(str(PROC / "*/regular_season.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[~df["is_neutral"]].assign(w=lambda d: (d["margin"] > 0).astype(int))
    return df.groupby(["season", "team_id"])["w"].sum().rename("actual_wins").reset_index()


def predict_all() -> pd.DataFrame:
    df = pd.read_parquet(TABLE)
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    df = df.merge(lines, on="game_id", how="left")
    gp = games_played_map()
    df["home_gameno"] = [
        gp.get((g, int(t)), np.nan) for g, t in zip(df["game_id"], df["team_id"], strict=False)
    ]
    df["away_gameno"] = [
        gp.get((g, int(t)), np.nan) for g, t in zip(df["game_id"], df["away_team_id"], strict=False)
    ]
    df["avg_gameno"] = df[["home_gameno", "away_gameno"]].mean(axis=1)

    seasons = sorted(df["season"].unique(), key=lambda s: int(s[:4]))
    out = []
    for i in range(3, len(seasons)):
        tr = df[df["season"].isin(set(seasons[:i]))].copy()
        te = df[df["season"] == seasons[i]].copy()
        tr, te = add_shrunk_form(tr, te)

        seed_tr = tr.dropna(subset=["wintotal_diff", "home_margin", "home_win"])
        full_tr = tr.dropna(subset=[*FULL, "home_margin", "home_win"])
        if len(seed_tr) < 300 or len(full_tr) < 300 or seed_tr["home_win"].nunique() < 2:
            continue

        # Seed (M0) fit on win-total differential.
        seed_lin = LinearRegression().fit(seed_tr[["wintotal_diff"]], seed_tr["home_margin"])
        seed_clf = LogisticRegression(max_iter=1000).fit(
            seed_tr[["wintotal_diff"]], seed_tr["home_win"]
        )
        # Full model fit.
        sc = StandardScaler().fit(full_tr[FULL])
        full_lin = LinearRegression().fit(full_tr[FULL], full_tr["home_margin"])
        full_clf = LogisticRegression(max_iter=1000).fit(
            sc.transform(full_tr[FULL]), full_tr["home_win"]
        )

        t = te.dropna(subset=["wintotal_diff", "home_margin", "home_win", "avg_gameno"]).copy()
        if t.empty:
            continue
        t["seed_margin"] = seed_lin.predict(t[["wintotal_diff"]])
        t["seed_p"] = seed_clf.predict_proba(t[["wintotal_diff"]])[:, 1]
        has_full = t[FULL].notna().all(axis=1)
        t["full_margin"] = np.nan
        t["full_p"] = np.nan
        if has_full.any():
            t.loc[has_full, "full_margin"] = full_lin.predict(t.loc[has_full, FULL])
            t.loc[has_full, "full_p"] = full_clf.predict_proba(sc.transform(t.loc[has_full, FULL]))[
                :, 1
            ]
        alpha = np.minimum(t["avg_gameno"] / K, 1.0).where(has_full, 0.0)
        t["blend_margin"] = (1 - alpha) * t["seed_margin"] + alpha * t["full_margin"].fillna(
            t["seed_margin"]
        )
        t["blend_p"] = (1 - alpha) * t["seed_p"] + alpha * t["full_p"].fillna(t["seed_p"])
        out.append(
            t[
                [
                    "season",
                    "team_id",
                    "away_team_id",
                    "home_win",
                    "home_margin",
                    "vegas_home_margin",
                    "avg_gameno",
                    "home_win_total",
                    "away_win_total",
                    "seed_p",
                    "blend_p",
                    "blend_margin",
                ]
            ]
        )
    return pd.concat(out, ignore_index=True)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pred = predict_all()
    aw = actual_wins().set_index(["season", "team_id"])["actual_wins"]

    # Expected wins per team-season: attribute each game's home-win prob to both sides.
    def expected_wins(pcol: str) -> pd.Series:
        home = pred[["season", "team_id", pcol]].rename(columns={"team_id": "tid", pcol: "p"})
        away = pred[["season", "away_team_id", pcol]].rename(columns={"away_team_id": "tid"})
        away["p"] = 1 - pred[pcol].to_numpy()
        allp = pd.concat([home, away], ignore_index=True)
        return allp.groupby(["season", "tid"])["p"].sum()

    ew_seed = expected_wins("seed_p")
    ew_blend = expected_wins("blend_p")
    # Vegas preseason expected wins = the team's own win total.
    veg = (
        pred[["season", "team_id", "home_win_total"]]
        .drop_duplicates()
        .rename(columns={"team_id": "tid", "home_win_total": "wt"})
    )
    veg = veg.set_index(["season", "tid"])["wt"]

    idx = ew_blend.index.intersection(aw.index).intersection(veg.dropna().index)
    a = aw.reindex(idx)
    res = pd.DataFrame(
        {
            "Vegas preseason line": (veg.reindex(idx) - a).abs(),
            "seed only (M0)": (ew_seed.reindex(idx) - a).abs(),
            "blended model": (ew_blend.reindex(idx) - a).abs(),
        }
    )
    print(f"Season expected-wins MAE vs actual, {len(idx):,} OOS team-seasons:\n")
    print(res.mean().round(3).to_string())
    print("\nFAIR comparison is preseason-vs-preseason: seed-only (M0) vs the Vegas line.")
    print("The 'blended model' number uses in-season information (it sums per-game")
    print("probabilities that already reflect how the season is going), so it is NOT")
    print("comparable to a preseason line. A fair in-season test needs the frozen-at-")
    print("game-X forecast (predict FINAL wins using only info through game X).")

    # MAE-by-game-number curve (matched-to-Vegas games for a fair line comparison).
    mt = pred.dropna(subset=["vegas_home_margin"]).copy()
    mt["bucket"] = pd.cut(
        mt["avg_gameno"],
        [0, 10, 20, 30, 45, 60, 82],
        labels=["1-10", "11-20", "21-30", "31-45", "46-60", "61-82"],
    )
    curve = mt.groupby("bucket", observed=True).apply(
        lambda d: pd.Series(
            {
                "n": len(d),
                "model_MAE": (d["blend_margin"] - d["home_margin"]).abs().mean(),
                "vegas_MAE": (d["vegas_home_margin"] - d["home_margin"]).abs().mean(),
            }
        ),
        include_groups=False,
    )
    print("\nMAE by game-number bucket (blended model vs Vegas line):")
    print(curve.round(3).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(curve))
    ax.plot(x, curve["model_MAE"], marker="o", color="#534AB7", label="blended model")
    ax.plot(x, curve["vegas_MAE"], marker="o", color="#D62728", label="Vegas closing line")
    ax.set_xticks(list(x))
    ax.set_xticklabels(curve.index)
    ax.set_xlabel("Games into the season (avg of the two teams)")
    ax.set_ylabel("Margin MAE (pts)")
    ax.set_title("Per-game margin MAE through the season: model vs Vegas closing line")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "mae_by_gamenumber.png", dpi=130)
    plt.close(fig)
    print(f"\nCurve written to {FIG / 'mae_by_gamenumber.png'}")


if __name__ == "__main__":
    main()
