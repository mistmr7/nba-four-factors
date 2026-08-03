"""Join the Kaggle per-game closing lines to our games and validate them.

The Kaggle file (data/kaggle/nba_2008-2026.csv, sourced from
sportsbookreviewsonline) has closing spread, total, moneylines, and quarter
scores for every game from 2007-08 through 2025-26. Its team codes use
present-day franchise abbreviations, so a relocated franchise (New Jersey to
Brooklyn, Seattle to Oklahoma City) is coded by its current city even in older
seasons. We therefore join on franchise team_id, which is era-stable in our
processed layer, not on abbreviation.

Validation is against ground truth we already hold: every joined game's home and
away score must match our box scores. The payoff is a per-game Vegas closing
spread, which gives the closing-line MAE benchmark and a true per-game Vegas
column to sit beside the model, plus quarter scores for within-game analysis.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
KAGGLE = REPO / "data" / "kaggle" / "nba_2008-2026.csv"
OUT = REPO / "data" / "features" / "vegas_game_lines.parquet"

# Kaggle code -> current NBA tricode for the codes that differ from ours.
CODE_FIX = {"gs": "GSW", "no": "NOP", "ny": "NYK", "sa": "SAS", "utah": "UTA", "wsh": "WAS"}


def tricode_to_team_id() -> dict[str, int]:
    """Current tricode to franchise team_id, from a recent processed season."""
    df = pd.read_parquet(PROC / "2024_25" / "regular_season.parquet")
    return {
        a: int(t) for a, t in df[["team_abbr", "team_id"]].drop_duplicates().itertuples(index=False)
    }


def kaggle_code_to_team_id() -> dict[str, int]:
    t2id = tricode_to_team_id()
    out = {}
    for code in [
        "atl",
        "bkn",
        "bos",
        "cha",
        "chi",
        "cle",
        "dal",
        "den",
        "det",
        "gs",
        "hou",
        "ind",
        "lac",
        "lal",
        "mem",
        "mia",
        "mil",
        "min",
        "no",
        "ny",
        "okc",
        "orl",
        "phi",
        "phx",
        "por",
        "sa",
        "sac",
        "tor",
        "utah",
        "wsh",
    ]:
        tri = CODE_FIX.get(code, code.upper())
        out[code] = t2id[tri]
    return out


def our_games() -> pd.DataFrame:
    """One row per game (home perspective) across regular season and playoffs."""
    files = sorted(glob.glob(str(PROC / "*/regular_season.parquet"))) + sorted(
        glob.glob(str(PROC / "*/playoffs.parquet"))
    )
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df["is_home"]) & (~df["is_neutral"])].copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df.rename(columns={"team_id": "home_team_id", "opp_team_id": "away_team_id"})[
        ["game_id", "game_date", "home_team_id", "away_team_id", "margin", "pts", "opp_pts"]
    ]


def build() -> pd.DataFrame:
    k = pd.read_csv(KAGGLE)
    k["date"] = pd.to_datetime(k["date"]).dt.normalize()
    code = kaggle_code_to_team_id()
    k["home_team_id"] = k["home"].map(code)
    k["away_team_id"] = k["away"].map(code)
    # Vegas closing home margin: positive spread favors the team in whos_favored.
    k["vegas_home_margin"] = np.where(k["whos_favored"] == "home", k["spread"], -k["spread"])

    games = our_games()
    m = games.merge(
        k[
            [
                "date",
                "home_team_id",
                "away_team_id",
                "score_home",
                "score_away",
                "vegas_home_margin",
                "spread",
                "total",
                "whos_favored",
                "moneyline_home",
                "moneyline_away",
                "q1_home",
                "q2_home",
                "q3_home",
                "q4_home",
                "ot_home",
                "q1_away",
                "q2_away",
                "q3_away",
                "q4_away",
                "ot_away",
            ]
        ],
        left_on=["game_date", "home_team_id", "away_team_id"],
        right_on=["date", "home_team_id", "away_team_id"],
        how="left",
    )
    return m


def main() -> None:
    m = build()
    matched = m["vegas_home_margin"].notna()
    sub = m[matched].copy()
    # Score agreement check.
    score_ok = (sub["pts"] == sub["score_home"]) & (sub["opp_pts"] == sub["score_away"])
    print(f"Our games (2008-26 window joinable): {len(m):,}")
    print(f"Matched to a Kaggle line: {matched.sum():,} ({matched.mean():.1%})")
    print(f"Of matched, home+away score agrees with our box score: {score_ok.mean():.4f}")

    good = sub[score_ok].copy()
    mae = float(np.mean(np.abs(good["margin"] - good["vegas_home_margin"])))
    b = np.polyfit(good["vegas_home_margin"], good["margin"], 1)
    fav_cover = float(np.mean(np.sign(good["margin"]) == np.sign(good["vegas_home_margin"])))
    print("\nVegas closing-spread benchmark (validated games):")
    print(f"  per-game closing-spread MAE vs actual margin: {mae:.2f} pts")
    print(
        f"  actual = {b[0]:.3f} * vegas_margin {b[1]:+.2f}  (slope ~1, intercept ~0 if efficient)"
    )
    print(f"  favorite straight-up win rate: {fav_cover:.3f}")

    # Save validated lines keyed by our game_id.
    keep = good[
        [
            "game_id",
            "vegas_home_margin",
            "spread",
            "total",
            "whos_favored",
            "moneyline_home",
            "moneyline_away",
            "q1_home",
            "q2_home",
            "q3_home",
            "q4_home",
            "ot_home",
            "q1_away",
            "q2_away",
            "q3_away",
            "q4_away",
            "ot_away",
        ]
    ]
    keep.to_parquet(OUT, index=False)
    print(f"\nWrote {len(keep):,} validated game lines to {OUT}")
    print("Note: coverage starts 2007-08; the 1997-2007 seasons have no per-game line.")


if __name__ == "__main__":
    main()
