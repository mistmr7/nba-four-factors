"""Collect player availability data from the box-score raw files.

Two tables, both pure extraction (no modeling choices baked in yet):

  inactives.parquet      one row per (game, team, player) who was inactive,
                         meaning not on the active roster (injury, rest, two-way
                         or G-League assignment). Reliable from 2005-06 onward.
  player_game_logs.parquet  one row per (game, team, player) who dressed, with
                         minutes parsed to a float and the full traditional stat
                         line, so any per-player value metric (minutes share,
                         Game Score, points, plus-minus, a PER-like index) can be
                         derived downstream without re-reading the raw JSON.

The point of keeping these raw is that the weighting of who-is-out is the open
research question. This module just makes the facts available; the feature builder
decides how to value an absence.

Run: python -m nba_four_factors.features.availability
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "data" / "raw" / "boxscoresummaryv3"
TRAD = REPO / "data" / "raw" / "boxscoretraditionalv3"
OUT = REPO / "data" / "features"

TYPE = {"1": "preseason", "2": "regular_season", "3": "allstar", "4": "playoffs", "5": "playin"}


def parse_gid(gid: str):
    season_type = TYPE.get(gid[2], "other")
    y = int(gid[3:5])
    start = 1900 + y if y >= 90 else 2000 + y
    season = f"{start}_{(start + 1) % 100:02d}"
    return season, start, season_type


def parse_minutes(m) -> float:
    if not m or not isinstance(m, str):
        return 0.0
    m = m.strip()
    if not m:
        return 0.0
    try:
        if ":" in m:
            a, b = m.split(":")
            return float(a) + float(b) / 60.0
        return float(m)
    except ValueError:
        return 0.0


def build_inactives() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(SUMMARY / "*.json"))):
        gid = Path(f).stem
        season, start, stype = parse_gid(gid)
        if start < 2005:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)["boxScoreSummary"]
        except Exception:
            continue
        for side in ("homeTeam", "awayTeam"):
            t = d.get(side) or {}
            tid = t.get("teamId")
            is_home = side == "homeTeam"
            for p in t.get("inactives", []) or []:
                rows.append(
                    (
                        gid,
                        season,
                        start,
                        stype,
                        tid,
                        is_home,
                        p.get("personId"),
                        p.get("firstName", ""),
                        p.get("familyName", ""),
                    )
                )
    return pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "yr",
            "season_type",
            "team_id",
            "is_home",
            "person_id",
            "first_name",
            "family_name",
        ],
    )


def build_player_logs() -> pd.DataFrame:
    rows = []
    cols_stat = [
        ("points", "pts"),
        ("fieldGoalsMade", "fgm"),
        ("fieldGoalsAttempted", "fga"),
        ("threePointersMade", "fg3m"),
        ("threePointersAttempted", "fg3a"),
        ("freeThrowsMade", "ftm"),
        ("freeThrowsAttempted", "fta"),
        ("reboundsOffensive", "oreb"),
        ("reboundsDefensive", "dreb"),
        ("reboundsTotal", "reb"),
        ("assists", "ast"),
        ("steals", "stl"),
        ("blocks", "blk"),
        ("turnovers", "tov"),
        ("foulsPersonal", "pf"),
        ("plusMinusPoints", "plus_minus"),
    ]
    for f in sorted(glob.glob(str(TRAD / "*.json"))):
        gid = Path(f).stem
        season, start, stype = parse_gid(gid)
        if start < 2005:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)["boxScoreTraditional"]
        except Exception:
            continue
        for side in ("homeTeam", "awayTeam"):
            t = d.get(side) or {}
            tid = t.get("teamId")
            is_home = side == "homeTeam"
            for p in t.get("players", []) or []:
                st = p.get("statistics", {}) or {}
                name = f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
                r = [
                    gid,
                    season,
                    start,
                    stype,
                    tid,
                    is_home,
                    p.get("personId"),
                    name,
                    p.get("comment", ""),
                    parse_minutes(st.get("minutes")),
                ]
                r += [st.get(src, 0) or 0 for src, _ in cols_stat]
                rows.append(r)
    cols = [
        "game_id",
        "season",
        "yr",
        "season_type",
        "team_id",
        "is_home",
        "person_id",
        "name",
        "comment",
        "min",
    ] + [dst for _, dst in cols_stat]
    return pd.DataFrame(rows, columns=cols)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ina = build_inactives()
    ina.to_parquet(OUT / "inactives.parquet", index=False)
    logs = build_player_logs()
    logs.to_parquet(OUT / "player_game_logs.parquet", index=False)

    print(
        f"inactives.parquet      {len(ina):>8,} rows  "
        f"({ina['game_id'].nunique():,} games, {ina['yr'].min()}-{ina['yr'].max()})"
    )
    print(
        f"player_game_logs.parquet {len(logs):>8,} rows  "
        f"({logs['game_id'].nunique():,} games, {logs['person_id'].nunique():,} players)"
    )
    reg = logs[logs["season_type"] == "regular_season"]
    played = reg[reg["min"] > 0]
    print(f"  regular-season player-games that dressed and played: {len(played):,}")
    print(
        f"  inactives per team-game (regular, 2010+): "
        f"{ina[(ina.season_type=='regular_season') & (ina.yr>=2010)].groupby(['game_id','team_id']).size().mean():.2f}"
    )


if __name__ == "__main__":
    main()
