"""Loaders for every input the preseason lane reads.

Thesis tables are read, never written. The three small pulls from
scripts/pull_preseason_inputs.py (draft history, season bios, live rosters)
are optional: every loader returns an empty frame with the right columns
when the raw files are missing, so the backtest runs with or without them
and the docs say which arms need them.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
PS = DATA / "player_state"
RAW_PRE = DATA / "raw" / "preseason"
OUT = DATA / "preseason"
VEGAS_HIST = DATA / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"
VEGAS_LIVE = DATA / "vegas" / "live_win_totals.csv"

UNDRAFTED_PICK = 61


def season_id(start_year: int) -> str:
    return f"{start_year}_{str(start_year + 1)[2:]}"


def start_year(season: str) -> int:
    return int(season[:4])


def prev_season(season: str) -> str:
    return season_id(start_year(season) - 1)


def schedule_length(season: str) -> int:
    """Games each team was scheduled to play when the preseason line was set."""
    if season == "2011_12":
        return 66
    if season == "2020_21":
        return 72
    if season == "1998_99":
        return 50
    return 82


def processed_seasons() -> list[str]:
    return sorted(p.name for p in (DATA / "processed").iterdir() if p.is_dir())


def _result_frame(payload: dict, name: str | None = None) -> pd.DataFrame:
    sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(sets, dict):
        sets = [sets]
    for rs in sets:
        if name is None or rs.get("name") == name:
            return pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    return pd.DataFrame()


def load_games(season: str) -> pd.DataFrame:
    """Team-game rows. Drops the canceled 0-0 placeholder (BOS-IND, 16 Apr 2013,
    known_anomalies.md): NBA games cannot end level, so margin 0 is never real."""
    g = pd.read_parquet(
        DATA / "processed" / season / "regular_season.parquet",
        columns=["game_id", "game_date", "team_id", "margin"],
    )
    return g[g.margin != 0]


def load_vegas() -> pd.DataFrame:
    v = pd.read_parquet(VEGAS_HIST)
    v = v[["season", "team_id", "tricode", "win_total", "actual_wins"]].copy()
    if VEGAS_LIVE.exists():
        live = pd.read_csv(VEGAS_LIVE)
        live = live.groupby(["season", "team_id", "tricode"], as_index=False).win_total.mean()
        live["actual_wins"] = np.nan
        v = pd.concat([v[~v.season.isin(live.season.unique())], live], ignore_index=True)
    return v


def load_draft() -> pd.DataFrame:
    cols = ["person_id", "draft_year", "overall_pick"]
    f = RAW_PRE / "drafthistory.json"
    if not f.exists():
        return pd.DataFrame(columns=cols)
    d = _result_frame(json.loads(f.read_text()))
    d = d.rename(
        columns={"PERSON_ID": "person_id", "SEASON": "draft_year", "OVERALL_PICK": "overall_pick"}
    )
    d["draft_year"] = pd.to_numeric(d.draft_year, errors="coerce")
    d["overall_pick"] = pd.to_numeric(d.overall_pick, errors="coerce")
    d = d.dropna(subset=["draft_year", "overall_pick"])
    d = d.sort_values("draft_year").drop_duplicates("person_id", keep="last")
    return d[cols].astype({"person_id": int, "draft_year": int, "overall_pick": int})


def load_bios() -> pd.DataFrame:
    cols = ["season", "person_id", "age", "height_in"]
    d = RAW_PRE / "bios"
    if not d.exists():
        return pd.DataFrame(columns=cols)
    frames = []
    for f in sorted(d.glob("*.json")):
        b = _result_frame(json.loads(f.read_text()))
        if b.empty:
            continue
        b = b.rename(
            columns={"PLAYER_ID": "person_id", "AGE": "age", "PLAYER_HEIGHT_INCHES": "height_in"}
        )
        b["season"] = f.stem
        frames.append(b[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["age"] = pd.to_numeric(out.age, errors="coerce")
    out["height_in"] = pd.to_numeric(out.height_in, errors="coerce")
    return out.drop_duplicates(["season", "person_id"])


def load_live_rosters(season: str) -> pd.DataFrame:
    cols = ["season", "team_id", "person_id", "age", "exp"]
    d = RAW_PRE / "rosters" / season
    if not d.exists():
        return pd.DataFrame(columns=cols)
    frames = []
    for f in sorted(d.glob("*.json")):
        r = _result_frame(json.loads(f.read_text()), "CommonTeamRoster")
        if r.empty:
            continue
        r = r.rename(
            columns={"TeamID": "team_id", "PLAYER_ID": "person_id", "AGE": "age", "EXP": "exp"}
        )
        r["season"] = season
        frames.append(r[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["age"] = pd.to_numeric(out.age, errors="coerce")
    out["exp"] = pd.to_numeric(out.exp.replace("R", 0), errors="coerce")
    return out


def load_rapm_end_of_season(prior: str) -> pd.DataFrame:
    """Last weekly as-of rating inside each season, the value known at season end."""
    r = pd.read_parquet(
        PS / f"rapm_asof_{prior}.parquet",
        columns=["person_id", "orapm", "drapm", "rapm", "off_poss", "asof_date", "season"],
    )
    last = r.groupby("season").asof_date.transform("max")
    r = r[r.asof_date == last].drop(columns="asof_date")
    return r.rename(columns={"orapm": "o", "drapm": "d", "rapm": "v", "off_poss": "poss"})


def load_rapm_single_season() -> pd.DataFrame:
    r = pd.read_parquet(
        PS / "rapm_season.parquet", columns=["person_id", "season", "rapm", "off_poss"]
    )
    return r.rename(columns={"rapm": "v_season", "off_poss": "poss_season"})


def name_key(name: str) -> str:
    """Accent-folded, lowercased, punctuation-free name for matching hand-kept lists."""
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    keep = "".join(c for c in folded.lower() if c.isalnum() or c == " ")
    parts = [p for p in keep.split() if p not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(parts)


def load_live_roster_names(season: str) -> pd.DataFrame:
    cols = ["person_id", "player", "team_id", "tricode"]
    d = RAW_PRE / "rosters" / season
    if not d.exists():
        return pd.DataFrame(columns=cols)
    frames = []
    for f in sorted(d.glob("*.json")):
        r = _result_frame(json.loads(f.read_text()), "CommonTeamRoster")
        if not r.empty:
            frames.append(
                r.rename(
                    columns={"PLAYER_ID": "person_id", "PLAYER": "player", "TeamID": "team_id"}
                )
            )
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    tri = pd.read_parquet(VEGAS_HIST, columns=["team_id", "tricode"]).drop_duplicates("team_id")
    return out.merge(tri, on="team_id", how="left")[cols]


def load_live_injuries(season: str) -> pd.DataFrame:
    """Hand-kept expected games missed: data/preseason/live_injuries_<season>.csv."""
    f = OUT / f"live_injuries_{season}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["player", "games_out"])
    return pd.read_csv(f)
