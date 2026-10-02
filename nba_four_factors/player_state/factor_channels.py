"""Individual four-factor channels from the possession table.

Decomposes the team four factors to players using only attributable
actions, with denominators counted from on-floor lineups (opportunities
faced), never outcome credit. See Credit_Engine_Design.md, "Individual
four factors".

Per player-game output columns:
    fga, fgm_w (FGM + 0.5*3PM), fg3m, tov, fta, ftm, oreb, dreb,
    stl, blk, shooting_fouls,
    oreb_chances, dreb_chances   (rebound events while on floor, own side)
    off_poss, def_poss           (possessions on floor, each side)
    opp_fga_faced                (opponent FGAs while on floor)

Rates are computed downstream; this module persists counts so baselines
and shrinkage stay walk-forward decisions.

Run from repo root:
    python -m nba_four_factors.player_state.factor_channels build 2025_26
    python -m nba_four_factors.player_state.factor_channels validate 2025_26
    python -m nba_four_factors.player_state.factor_channels leaders 2025_26
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
POSS = REPO / "data" / "player_state" / "possessions"
OUT_DIR = REPO / "data" / "player_state" / "factor_channels"

COUNT_COLS = [
    "fga",
    "fgm_w",
    "fg3m",
    "tov",
    "fta",
    "ftm",
    "oreb",
    "dreb",
    "stl",
    "blk",
    "shooting_fouls",
    "oreb_chances",
    "dreb_chances",
    "off_poss",
    "def_poss",
    "opp_fga_faced",
]


def _home_ids(season: str) -> dict[str, int]:
    df = pd.read_parquet(
        PROC / season / "regular_season.parquet",
        columns=["game_id", "team_id", "is_home"],
    )
    h = df[df.is_home].drop_duplicates("game_id")
    return dict(zip(h.game_id.astype(str), h.team_id.astype(int), strict=False))


def build(season: str) -> pd.DataFrame:
    pos = pd.read_parquet(POSS / f"{season}_regular_season.parquet")
    home = _home_ids(season)
    acc: dict[tuple, defaultdict] = {}

    def bump(gid: str, pid: int, col: str, amt: float = 1.0) -> None:
        if pid == 0:
            return
        key = (gid, pid)
        if key not in acc:
            acc[key] = defaultdict(float)
        acc[key][col] += amt

    for r in pos.itertuples(index=False):
        gid = r.game_id
        h = home.get(str(gid))
        if h is None or r.home_lineup is None or r.away_lineup is None:
            continue
        off5 = list(r.home_lineup) if r.off_team_id == h else list(r.away_lineup)
        def5 = list(r.away_lineup) if r.off_team_id == h else list(r.home_lineup)
        for p in off5:
            bump(gid, int(p), "off_poss")
        for p in def5:
            bump(gid, int(p), "def_poss")
        events = json.loads(r.events)
        for e in events:
            t = e["t"]
            pid = int(e.get("pid") or 0)
            if t == "shot":
                bump(gid, pid, "fga")
                for p in def5:
                    bump(gid, int(p), "opp_fga_faced")
                if e.get("made"):
                    w = 1.5 if e.get("pts") == 3 else 1.0
                    bump(gid, pid, "fgm_w", w)
                    if e.get("pts") == 3:
                        bump(gid, pid, "fg3m")
                blk = e.get("blk")
                if blk:
                    bump(gid, int(blk), "blk")
            elif t == "tov":
                bump(gid, pid, "tov")
                stl = e.get("stl")
                if stl:
                    bump(gid, int(stl), "stl")
            elif t == "ft":
                bump(gid, pid, "fta")
                if e.get("made"):
                    bump(gid, pid, "ftm")
            elif t == "reb":
                if e.get("off"):
                    bump(gid, pid, "oreb")
                    for p in off5:
                        bump(gid, int(p), "oreb_chances")
                    for p in def5:
                        bump(gid, int(p), "dreb_chances")
                else:
                    bump(gid, pid, "dreb")
                    for p in off5:
                        bump(gid, int(p), "oreb_chances")
                    for p in def5:
                        bump(gid, int(p), "dreb_chances")
            elif t == "foul":
                if "Shooting" in (e.get("sub") or ""):
                    bump(gid, pid, "shooting_fouls")

    rows = []
    for (gid, pid), c in acc.items():
        row = {"game_id": gid, "person_id": pid}
        for col in COUNT_COLS:
            row[col] = c.get(col, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["season"] = season
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / f"{season}.parquet", index=False)
    print(f"{season}: {len(df):,} player-games, {df.person_id.nunique()} players")
    return df


def validate(season: str) -> None:
    df = pd.read_parquet(OUT_DIR / f"{season}.parquet")
    box = pd.read_parquet(PROC / season / "regular_season.parquet")
    box = box[~box.is_neutral] if "is_neutral" in box else box
    team = box.groupby("game_id").agg(
        fga=("fga", "sum"),
        tov=("tov", "sum"),
        fta=("fta", "sum"),
        oreb=("oreb", "sum"),
        fgm=("fgm", "sum"),
    )
    mine = df.groupby("game_id")[["fga", "tov", "fta", "oreb"]].sum()
    j = mine.join(team, how="inner", rsuffix="_box")
    for a, b in [("fga", "fga_box"), ("tov", "tov_box"), ("oreb", "oreb_box")]:
        diff = (j[a] - j[b]).abs()
        print(f"  {a:5s}: exact {(diff == 0).sum():,}/{len(j):,}  " f"mean|diff| {diff.mean():.3f}")
    fta_diff = (j["fta"] - j["fta_box"]).abs()
    print(
        f"  fta  : exact {(fta_diff == 0).sum():,}/{len(j):,}  "
        f"mean|diff| {fta_diff.mean():.3f}  (technicals excluded here)"
    )


def leaders(season: str) -> None:
    df = pd.read_parquet(OUT_DIR / f"{season}.parquet")
    s = df.groupby("person_id")[COUNT_COLS].sum()
    names = _names(season)
    s.index = [names.get(p, str(p)) for p in s.index]

    def show(title: str, series: pd.Series, asc: bool = False, n: int = 5) -> None:
        print(f"\n{title}")
        print(series.sort_values(ascending=asc).head(n).round(3).to_string())

    vol = s[s.fga >= 300]
    show("eFG% (min 300 FGA)", vol.fgm_w / vol.fga)
    used = s.fga + 0.44 * s.fta + s.tov
    hi_use = s[used >= 400]
    show(
        "TOV% (min 400 poss used)",
        hi_use.tov / (hi_use.fga + 0.44 * hi_use.fta + hi_use.tov),
        asc=True,
    )
    rb = s[s.dreb_chances >= 800]
    show("OREB% of on-floor boards (min 800 chances)", rb.oreb / rb.oreb_chances)
    show("DREB% of on-floor boards (min 800 chances)", rb.dreb / rb.dreb_chances)
    d = s[s.def_poss >= 1500]
    show("Steals per 100 opponent possessions (min 1500)", 100 * d.stl / d.def_poss)
    f = s[s.opp_fga_faced >= 1500]
    show(
        "Shooting fouls per 100 opponent FGA (min 1500), lowest",
        100 * f.shooting_fouls / f.opp_fga_faced,
        asc=True,
    )


def _names(season: str) -> dict[int, str]:
    rosters = REPO / "data" / "player_state" / "rosters" / f"{season}_regular_season.parquet"
    if not rosters.exists():
        return {}
    r = pd.read_parquet(rosters)
    if "family_name" not in r.columns:
        return {}
    full = r.first_name.astype(str).str[0] + ". " + r.family_name.astype(str)
    return dict(zip(r.person_id.astype(int), full, strict=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "validate", "leaders"])
    ap.add_argument("season")
    args = ap.parse_args()
    seasons = (
        sorted(p.stem.replace("_regular_season", "") for p in POSS.glob("*_regular_season.parquet"))
        if args.season == "all"
        else [args.season]
    )
    for s in seasons:
        if args.cmd == "build":
            build(s)
        elif args.cmd == "validate":
            validate(s)
        else:
            leaders(s)


if __name__ == "__main__":
    main()
