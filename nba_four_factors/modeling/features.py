"""Assemble the one-row-per-game modeling table (home-team perspective).

Each regular-season game becomes a single row from the home team's point of
view, so the targets are the home margin and a home win flag and every feature
is a home-minus-away differential. This avoids the mirrored-duplication problem
of the two-row layout.

Feature blocks
* Four-factor static block: home-minus-away season-to-date (expanding, trailing)
  differential in each of eFG, OREB, TOV, FTR. These are the team-strength
  predictors and stay causal because the expanding mean is shifted to exclude
  the current game.
* Recent-form block: each team's trailing rolling-window composite mean and
  variance, plus its season-to-date composite baseline. The model later forms
  the shrunk recent-form deviation w * (recent - baseline); the reliability
  weight w is estimated inside each CV fold, so only the raw rolling pieces are
  stored here.
* Context: rest differential and travel load (seven-day miles, back-to-back)
  from the travel feature.
* Vegas: each team's preseason win total, for the M0 seed and the baseline.

Leakage note (v1): the composite is built with fixed eda_07 weights on
pooled-standardized factor differentials. Pooled standardization mixes a little
cross-season spread into the scale; it is a spread rescale, not a level leak,
and is flagged for tightening to in-fold standardization later. All trailing
means and variances use only a team's own prior games within the season, so the
time dimension is leakage-free.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
FEATURES = REPO / "data" / "features"
VEGAS = REPO / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"

WINDOW = 15
MIN_PERIODS = 5
COMPOSITE_WEIGHTS = {"efg_d": 0.40, "tov_d": 0.25, "oreb_d": 0.20, "ftr_d": 0.15}
FACTORS = ["efg_d", "oreb_d", "tov_d", "ftr_d"]

# Seasons excluded from the modeling dataset. 1998-99 (the lockout) has no
# preseason Vegas win total, so it cannot be seeded or fairly benchmarked.
DROP_SEASONS = {"1998_99"}


def _team_game_frame() -> pd.DataFrame:
    """All regular-season team-game rows with single-game factor differentials."""
    files = sorted(glob.glob(str(PROC / "*/regular_season.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(~df["is_neutral"]) & (~df["season"].isin(DROP_SEASONS))].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["efg_d"] = df["off_efg_pct"] - df["def_efg_pct"]
    df["oreb_d"] = df["off_orb_pct"] - df["def_orb_pct"]
    df["tov_d"] = df["def_tov_pct"] - df["off_tov_pct"]  # positive: force more TOs
    df["ftr_d"] = df["off_ft_rate"] - df["def_ft_rate"]  # FTA/FGA based (attempt rate)
    # Alternative fourth factor: makes per field-goal attempt (Oliver's FT/FGA).
    # Less inflated by intentional-foul situations than the FTA-based rate.
    df["ftmfga_d"] = (df["ftm"] / df["fga"]) - (df["opp_ftm"] / df["opp_fga"])
    # Pace: possessions (standard estimate), averaged with the opponent since
    # both teams share a game's pace.
    poss = df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"]
    opp_poss = df["opp_fga"] + 0.44 * df["opp_fta"] - df["opp_oreb"] + df["opp_tov"]
    df["pace"] = 0.5 * (poss + opp_poss)
    return df


def _add_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled-standardized weighted composite of the four factor differentials."""
    out = df.copy()
    comp = np.zeros(len(out))
    for col, w in COMPOSITE_WEIGHTS.items():
        z = (out[col] - out[col].mean()) / out[col].std(ddof=0)
        comp = comp + w * z.to_numpy()
    out["composite"] = comp
    return out


def _add_trailing(df: pd.DataFrame) -> pd.DataFrame:
    """Per team-season, trailing (causal) season-to-date and rolling features."""
    out = df.sort_values(["team_id", "season", "game_date"]).copy()
    g = out.groupby(["team_id", "season"], sort=False)
    # Season-to-date expanding means of each factor, shifted to exclude this game.
    for col in [*FACTORS, "ftmfga_d", "pace"]:
        out[f"sd_{col}"] = g[col].transform(lambda s: s.shift(1).expanding().mean())
    # Season-to-date composite baseline.
    out["base_comp"] = g["composite"].transform(lambda s: s.shift(1).expanding().mean())
    # Trailing rolling-window composite mean and variance (recent form).
    out["form_mean"] = g["composite"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    )
    out["form_var"] = g["composite"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_PERIODS).var(ddof=1)
    )
    return out


def _vegas_lookup() -> dict[tuple[str, int], float]:
    v = pd.read_parquet(VEGAS)
    return {
        (s, int(t)): w for s, t, w in v[["season", "team_id", "win_total"]].itertuples(index=False)
    }


def build_modeling_table() -> pd.DataFrame:
    """Build and return the one-row-per-game home-perspective modeling table."""
    tf = _add_trailing(_add_composite(_team_game_frame()))
    travel = pd.read_parquet(FEATURES / "travel.parquet")[
        ["game_id", "team_id", "days_rest", "is_b2b", "miles_7d"]
    ]
    tf = tf.merge(travel, on=["game_id", "team_id"], how="left")

    feat_cols = [f"sd_{c}" for c in [*FACTORS, "ftmfga_d", "pace"]] + [
        "base_comp",
        "form_mean",
        "form_var",
        "days_rest",
        "is_b2b",
        "miles_7d",
    ]
    keep = [
        "game_id",
        "season",
        "game_date",
        "team_id",
        "opp_team_id",
        "is_home",
        "margin",
        *feat_cols,
    ]
    tf = tf[keep]

    home = tf[tf["is_home"]].copy()
    away = tf[~tf["is_home"]].copy()
    away = away.rename(columns={c: f"away_{c}" for c in [*feat_cols, "team_id"]})
    away = away[["game_id", "away_team_id", *[f"away_{c}" for c in feat_cols]]]

    m = home.merge(away, on="game_id", how="inner")
    veg = _vegas_lookup()
    m["home_win_total"] = [
        veg.get((s, int(t)), np.nan) for s, t in zip(m["season"], m["team_id"], strict=False)
    ]
    m["away_win_total"] = [
        veg.get((s, int(t)), np.nan) for s, t in zip(m["season"], m["away_team_id"], strict=False)
    ]

    # Targets.
    m["home_margin"] = m["margin"]
    m["home_win"] = (m["margin"] > 0).astype(int)

    # Home-minus-away differentials for the static four-factor block.
    for c in [*FACTORS, "ftmfga_d"]:
        m[f"d_{c}"] = m[f"sd_{c}"] - m[f"away_sd_{c}"]
    # Pace: home-minus-away tendency, and the expected game pace (sum).
    m["pace_diff"] = m["sd_pace"] - m["away_sd_pace"]
    m["pace_sum"] = m["sd_pace"] + m["away_sd_pace"]
    # Context differentials.
    m["rest_diff"] = m["days_rest"] - m["away_days_rest"]
    m["b2b_diff"] = m["is_b2b"].astype(float) - m["away_is_b2b"].astype(float)
    m["miles7d_diff"] = m["miles_7d"] - m["away_miles_7d"]
    m["wintotal_diff"] = m["home_win_total"] - m["away_win_total"]

    out_cols = [
        "game_id",
        "season",
        "game_date",
        "team_id",
        "away_team_id",
        "home_margin",
        "home_win",
        "d_efg_d",
        "d_oreb_d",
        "d_tov_d",
        "d_ftr_d",
        "d_ftmfga_d",
        "form_mean",
        "base_comp",
        "form_var",
        "away_form_mean",
        "away_base_comp",
        "away_form_var",
        "rest_diff",
        "b2b_diff",
        "miles7d_diff",
        "pace_diff",
        "pace_sum",
        "home_win_total",
        "away_win_total",
        "wintotal_diff",
    ]
    return m[out_cols].sort_values(["season", "game_date"]).reset_index(drop=True)


def main() -> None:
    table = build_modeling_table()
    FEATURES.mkdir(parents=True, exist_ok=True)
    path = FEATURES / "modeling_table.parquet"
    table.to_parquet(path, index=False)
    n_static = table[["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftr_d"]].dropna().shape[0]
    print(f"Wrote {len(table):,} games to {path}")
    print(f"  rows with complete static four-factor block: {n_static:,}")
    print(
        f"  rows with recent-form block (both teams): "
        f"{table[['form_mean','away_form_mean']].dropna().shape[0]:,}"
    )
    print(
        f"  rows with Vegas totals (both teams): {table[['home_win_total','away_win_total']].dropna().shape[0]:,}"
    )


if __name__ == "__main__":
    main()
