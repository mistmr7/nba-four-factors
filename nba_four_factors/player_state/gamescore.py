"""Build and validate the player_game_score fact table.

One row per (game_id, person_id) for every player who appears in a game's
traditional box score, plus one row per pregame inactive (2005-06 onward,
where inactive lists exist). Parsed from the raw boxscoretraditionalv3 JSONs
so the table covers all seasons, 1997-98 through the present.

Rules that matter:
    game_score is NULL unless status is "played". A player who logged twelve
    scoreless minutes has a Game Score near zero; a player who did not play
    has no Game Score at all. Collapsing those to the same value silently
    corrupts every downstream rolling statistic.

    The eleven Game Score components are stored, not discarded, so later
    phases never recompute them from source.

    Raw game_score and game_score_per36 are separate columns. Raw conflates
    quality with playing time; the rate is what the filter consumes.

Status enum:
    played       minutes > 0
    dnp_coach    comment begins DNP - Coach
    dnp_other    any other DNP / DND / NWT comment (injury, rest, personal)
    inactive     on the pregame inactive list (2005-06 onward)

Build is resumable: one parquet part per season and season type under
data/player_state/parts/, concatenated at the end.

Run from repo root:
    python -m nba_four_factors.player_state.gamescore build
    python -m nba_four_factors.player_state.gamescore validate
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW_BOX = REPO / "data" / "raw" / "boxscoretraditionalv3"
PROC = REPO / "data" / "processed"
INACTIVES = REPO / "data" / "features" / "inactives.parquet"
OUT_DIR = REPO / "data" / "player_state"
PARTS = OUT_DIR / "parts"
TABLE = OUT_DIR / "player_game_score.parquet"

STAT_MAP = {
    "points": "pts",
    "fieldGoalsMade": "fgm",
    "fieldGoalsAttempted": "fga",
    "freeThrowsMade": "ftm",
    "freeThrowsAttempted": "fta",
    "reboundsOffensive": "oreb",
    "reboundsDefensive": "dreb",
    "steals": "stl",
    "assists": "ast",
    "blocks": "blk",
    "foulsPersonal": "pf",
    "turnovers": "tov",
}
COMPONENTS = list(STAT_MAP.values())


def parse_minutes(raw: str | None) -> float:
    if not raw:
        return 0.0
    s = str(raw)
    if ":" in s:
        mm, ss = s.split(":", 1)
        try:
            return float(mm) + float(ss) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def classify(comment: str, minutes: float) -> str:
    if minutes > 0:
        return "played"
    c = (comment or "").strip().upper()
    if c.startswith("DNP - COACH"):
        return "dnp_coach"
    return "dnp_other"


def game_score(row: dict) -> float:
    return (
        row["pts"]
        + 0.4 * row["fgm"]
        - 0.7 * row["fga"]
        - 0.4 * (row["fta"] - row["ftm"])
        + 0.7 * row["oreb"]
        + 0.3 * row["dreb"]
        + row["stl"]
        + 0.7 * row["ast"]
        + 0.7 * row["blk"]
        - 0.4 * row["pf"]
        - row["tov"]
    )


def season_games() -> pd.DataFrame:
    """Map every game to its season, season type, and date from the processed store."""
    frames = []
    for st in ("regular_season", "playoffs"):
        for f in sorted(glob.glob(str(PROC / f"*/{st}.parquet"))):
            d = pd.read_parquet(f, columns=["game_id", "season", "game_date"])
            d = d.drop_duplicates("game_id")
            d["season_type"] = st
            frames.append(d)
    g = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    g["game_date"] = pd.to_datetime(g["game_date"])
    return g


def parse_game(path: Path) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    bs = d.get("boxScoreTraditional", d)
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = bs.get(side) or {}
        tid = team.get("teamId")
        for p in team.get("players") or []:
            st = p.get("statistics") or {}
            minutes = parse_minutes(st.get("minutes"))
            row = {
                "game_id": str(bs["gameId"]).zfill(10),
                "person_id": int(p["personId"]),
                "team_id": int(tid) if tid is not None else None,
                "minutes": minutes,
                "comment": (p.get("comment") or "").strip(),
            }
            for src, dst in STAT_MAP.items():
                v = st.get(src)
                row[dst] = float(v) if v is not None else 0.0
            row["status"] = classify(row["comment"], minutes)
            if row["status"] == "played":
                gs = game_score(row)
                row["game_score"] = gs
                row["game_score_per36"] = 36.0 * gs / minutes
            else:
                row["game_score"] = np.nan
                row["game_score_per36"] = np.nan
                for c in COMPONENTS:
                    row[c] = np.nan
            rows.append(row)
    return rows


def build(time_budget: float | None = None) -> None:
    t0 = time.time()
    PARTS.mkdir(parents=True, exist_ok=True)
    games = season_games()
    by_part = games.groupby(["season", "season_type"])
    for (season, stype), g in sorted(by_part, key=lambda kv: kv[0]):
        part = PARTS / f"{season}_{stype}.parquet"
        if part.exists():
            continue
        if time_budget is not None and time.time() - t0 > time_budget:
            print("time budget reached; re-invoke to continue", flush=True)
            return
        rows = []
        missing = 0
        for gid in g["game_id"]:
            path = RAW_BOX / f"{str(gid).zfill(10)}.json"
            if not path.exists():
                missing += 1
                continue
            rows.extend(parse_game(path))
        df = pd.DataFrame(rows)
        df = df.merge(g[["game_id", "season", "season_type", "game_date"]], on="game_id")
        df.to_parquet(part, index=False)
        print(
            f"{season} {stype}: {len(df):,} player rows, {missing} missing box files "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
    assemble()


def assemble() -> None:
    parts = sorted(PARTS.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    ina = pd.read_parquet(INACTIVES)
    dates = season_games()
    ina = ina.merge(dates, on="game_id", how="inner", suffixes=("", "_g"))
    ina_rows = pd.DataFrame(
        {
            "game_id": ina["game_id"].astype(str).str.zfill(10),
            "person_id": ina["person_id"].astype(int),
            "team_id": ina["team_id"].astype(int),
            "minutes": 0.0,
            "comment": "",
            "status": "inactive",
            "season": ina["season"],
            "season_type": ina["season_type"],
            "game_date": ina["game_date"],
        }
    )
    for c in [*COMPONENTS, "game_score", "game_score_per36"]:
        ina_rows[c] = np.nan

    already = set(zip(df["game_id"], df["person_id"], strict=False))
    keep = [
        (g, p) not in already
        for g, p in zip(ina_rows["game_id"], ina_rows["person_id"], strict=False)
    ]
    dup_inactives = len(ina_rows) - sum(keep)
    df = pd.concat([df, ina_rows[keep]], ignore_index=True)
    df = df.sort_values(["game_date", "game_id", "team_id", "person_id"]).reset_index(drop=True)
    df.to_parquet(TABLE, index=False)
    print(f"\nwrote {TABLE}")
    print(f"  {len(df):,} rows, {df.game_id.nunique():,} games, seasons {df.season.min()}-{df.season.max()}")
    print(f"  inactive rows also present in box score (skipped as duplicates): {dup_inactives}")
    print(df.status.value_counts().to_string())


def validate() -> None:
    df = pd.read_parquet(TABLE)
    failures = []

    played = df[df.status == "played"]
    bad_zero = df[(df.status != "played") & df.game_score.notna()]
    bad_null = df[(df.status == "played") & df.game_score.isna()]
    print(f"null discipline: {len(bad_zero)} non-played rows with a game_score, "
          f"{len(bad_null)} played rows without one")
    if len(bad_zero) or len(bad_null):
        failures.append("null discipline")

    reg = played[played.season_type == "regular_season"]
    big = reg[reg.minutes >= 25]
    m = big.groupby("season").game_score.mean()
    print("\nper-season mean Game Score, players with 25+ minutes:")
    print(m.round(2).to_string())
    # A column-mapping break in one season's endpoint version shows up as a
    # discontinuous jump. Smooth drift is the real scoring environment (the
    # 25+ minute mean rises from ~11.5 in 1997-98 to ~14 by 2023-24) and is
    # expected, so the gate is on season-over-season change, not level.
    jumps = m.diff().abs()
    if (jumps > 1.5).any():
        print(f"  jump(s) > 1.5: {jumps[jumps > 1.5].round(2).to_dict()}")
        failures.append("per-season discontinuity")

    ts = played.groupby(["game_id", "team_id"]).game_score.sum().reset_index()
    margins = []
    for f in sorted(glob.glob(str(PROC / "*/regular_season.parquet"))):
        d = pd.read_parquet(f, columns=["game_id", "team_id", "margin"])
        margins.append(d)
    mg = pd.concat(margins, ignore_index=True)
    j = ts.merge(mg, on=["game_id", "team_id"])
    corr = np.corrcoef(j.game_score, j.margin)[0, 1]
    print(f"\nteam-sum Game Score vs own margin correlation: {corr:.3f} (n={len(j):,})")
    if corr < 0.4:
        failures.append("team-level correlation")

    ina = df[df.status == "inactive"]
    both = df[df.status == "played"].merge(
        ina[["game_id", "person_id"]], on=["game_id", "person_id"], how="inner"
    )
    print(f"\nroster reconciliation: {len(both)} players both played and inactive in the "
          f"same game (should be ~0, logged not raised)")

    if failures:
        print(f"\nVALIDATION FAILURES: {failures}")
    else:
        print("\nall checks passed (golden-record check runs separately once "
              "published values are verified)")


def golden(game_id: str, person_id: int) -> None:
    df = pd.read_parquet(TABLE)
    r = df[(df.game_id == str(game_id).zfill(10)) & (df.person_id == person_id)]
    if len(r) == 0:
        print("no such row")
        return
    r = r.iloc[0]
    print(r[["game_id", "season", "game_date", "minutes", *COMPONENTS,
             "game_score", "game_score_per36", "status"]].to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "assemble", "validate", "golden"])
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--game-id", default=None)
    ap.add_argument("--person-id", type=int, default=None)
    a = ap.parse_args()
    if a.command == "build":
        build(a.time_budget)
    elif a.command == "assemble":
        assemble()
    elif a.command == "validate":
        validate()
    elif a.command == "golden":
        golden(a.game_id, a.person_id)


if __name__ == "__main__":
    main()
