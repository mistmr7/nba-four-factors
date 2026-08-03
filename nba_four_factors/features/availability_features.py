"""Per-game availability features: how much value each team is missing.

For every regular-season game from 2005-06 on, value each inactive player by what
they had been contributing this season UP TO that game (a causal, as-of read of
their trailing per-game minutes, points, Game Score, and plus-minus), then sum per
team. The result is a menu of weightings, computed side by side so the sweep can
decide which matters:

  n        raw count of inactives (baseline; mostly two-way / deep-bench noise)
  mpg      summed trailing minutes per game of the inactives (share of rotation)
  pts      summed trailing points per game
  gmsc     summed trailing Game Score (a one-number box-score value)
  pm       summed trailing plus-minus per game
  starter  count of inactives whose trailing minutes were >= 24 (starter-ish)
  rot      count of inactives whose trailing minutes were >= 15 (rotation)

Output (home perspective, one row per game): home_<m>_out, away_<m>_out, and the
differential <m>_diff = home minus away, for each metric m. Everything is known at
tip-off and uses only prior games, so it is leak-free and comparable to the line.

Run: python -m nba_four_factors.features.availability_features
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
FEAT = REPO / "data" / "features"
METRICS = ["mpg", "pts", "gmsc", "pm", "n", "starter", "rot"]


def game_dates() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_parquet(f) for f in glob.glob(str(PROC / "*/regular_season.parquet"))],
        ignore_index=True,
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df[["game_id", "game_date"]].drop_duplicates()


def player_trailing() -> pd.DataFrame:
    """Cumulative season-to-date per-game value at each played game."""
    logs = pd.read_parquet(FEAT / "player_game_logs.parquet")
    logs = logs[(logs["season_type"] == "regular_season") & (logs["min"] > 0)].copy()
    logs["gmsc"] = (
        logs["pts"]
        + 0.4 * logs["fgm"]
        - 0.7 * logs["fga"]
        - 0.4 * (logs["fta"] - logs["ftm"])
        + 0.7 * logs["oreb"]
        + 0.3 * logs["dreb"]
        + logs["stl"]
        + 0.7 * logs["ast"]
        + 0.7 * logs["blk"]
        - 0.4 * logs["pf"]
        - logs["tov"]
    )
    logs = logs.merge(game_dates(), on="game_id", how="left").dropna(subset=["game_date"])
    logs = logs.sort_values(["person_id", "season", "game_date"])
    g = logs.groupby(["person_id", "season"])
    cnt = g.cumcount() + 1
    out = logs[["person_id", "season", "game_date"]].copy()
    for src in ["min", "pts", "gmsc", "plus_minus"]:
        out["cum_" + src] = g[src].cumsum().to_numpy() / cnt.to_numpy()
    return out.sort_values("game_date")


def build():
    dates = game_dates()
    trail = player_trailing()
    ina = pd.read_parquet(FEAT / "inactives.parquet")
    ina = ina[ina["season_type"] == "regular_season"].merge(dates, on="game_id", how="left")
    ina = ina.dropna(subset=["game_date"]).sort_values("game_date")

    # As-of: each inactive gets the player's cumulative value as of their last
    # played game before this game's date (backward, no leakage).
    j = pd.merge_asof(
        ina,
        trail,
        by=["person_id", "season"],
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
    )
    j["mpg"] = j["cum_min"].fillna(0.0)
    j["pts"] = j["cum_pts"].fillna(0.0)
    j["gmsc"] = j["cum_gmsc"].fillna(0.0)
    j["pm"] = j["cum_plus_minus"].fillna(0.0)
    j["n"] = 1.0
    j["starter"] = (j["cum_min"] >= 24).astype(float)
    j["rot"] = (j["cum_min"] >= 15).astype(float)

    agg = j.groupby(["game_id", "is_home"])[METRICS].sum().reset_index()
    home = agg[agg["is_home"]].set_index("game_id")[METRICS].add_prefix("home_").add_suffix("_out")
    away = agg[~agg["is_home"]].set_index("game_id")[METRICS].add_prefix("away_").add_suffix("_out")

    games = dates[["game_id"]].drop_duplicates().set_index("game_id")
    f = games.join(home, how="left").join(away, how="left").fillna(0.0).reset_index()
    for m in METRICS:
        f[m + "_diff"] = f["home_" + m + "_out"] - f["away_" + m + "_out"]
    f.to_parquet(FEAT / "availability_features.parquet", index=False)

    n_with = (f[[m + "_diff" for m in METRICS]].abs().sum(axis=1) > 0).sum()
    print(f"availability_features.parquet  {len(f):,} games, {n_with:,} with any imbalance")
    print("\nmean |diff| and share of games with a starter-imbalance:")
    for m in METRICS:
        print(
            f"  {m:8s} mean|diff| {f[m+'_diff'].abs().mean():8.3f}   "
            f"max|diff| {f[m+'_diff'].abs().max():8.1f}"
        )
    sd = f["starter_diff"]
    print(
        f"\ngames with a net starter-availability edge (|starter_diff|>=1): "
        f"{100 * (sd.abs() >= 1).mean():.1f}%"
    )


if __name__ == "__main__":
    build()
