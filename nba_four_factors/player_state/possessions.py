"""Possession parser: playbyplayv3 events -> possession table.

Phase 1 of the play-by-play lane (docs/Possession_Parser_Spec.md).
A possession belongs to the team with the ball and ends only when the
ball changes hands: made FG (unless an and-1 free throw is pending),
made final free throw of a non-technical trip, defensive rebound,
turnover, or period end. Offensive rebounds continue the possession and
increment the chance index, so points per possession are unbounded.

Lineups come from the stint table (built by stints.py); this module
never re-derives substitutions. Technical free throws are scored as
standalone events outside any possession and tracked separately so the
game-level points reconciliation still closes.

Feed facts this parser relies on (verified against the raw store):
- Row order is chronological; actionNumber is not. Stable-sort by
  period only, exactly as stints.py does.
- Free throws carry no shotResult; a make is the absence of the
  description's leading "MISS".
- Foul-row parentheticals name the referee, not the fouled player.
- 1997-98 lacks shotValue ("3PT" in the description marks threes) and
  the Heave actionType (flagged heuristically: last 3 seconds of a
  period from 40+ feet).

Run from repo root:
    python -m nba_four_factors.player_state.possessions build 2025_26
    python -m nba_four_factors.player_state.possessions validate 2025_26
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "playbyplay"
PROC = REPO / "data" / "processed"
STINTS = REPO / "data" / "player_state" / "stints"
OUT_DIR = REPO / "data" / "player_state" / "possessions"

SHOT_TYPES = frozenset({"Made Shot", "Missed Shot", "Heave"})
ADMIN_TYPES = frozenset(
    {"Substitution", "Timeout", "Instant Replay", "Ejection", "period", "Jump Ball"}
)

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")
_AST_RE = re.compile(r"\((\S[^()]*?) (\d+) AST\)")
_STL_RE = re.compile(r"\((\S[^()]*?) (\d+) STL\)")
_BLK_RE = re.compile(r"\((\S[^()]*?) (\d+) BLK\)")
_FT_NOFM = re.compile(r"(\d+) of (\d+)")


def _s(v) -> str:
    return "" if pd.isna(v) else str(v)


def _elapsed(clock: str, period: int) -> float:
    m = _CLOCK_RE.match(clock or "")
    if not m:
        return np.nan
    remaining = int(m.group(1)) * 60 + float(m.group(2))
    plen = 720.0 if period <= 4 else 300.0
    return plen - remaining


def _shot_points(row) -> int:
    sv = getattr(row, "shotValue", None)
    if sv is not None and pd.notna(sv) and int(sv) in (2, 3):
        return int(sv)
    return 3 if "3PT" in _s(row.description) else 2


def _is_heave(row, elapsed: float) -> bool:
    if row.actionType == "Heave":
        return True
    plen = 720.0 if row.period <= 4 else 300.0
    dist = row.shotDistance
    return bool(pd.notna(dist) and dist >= 40 and (plen - elapsed) <= 3.0)


class _Possession:
    __slots__ = (
        "off_team",
        "period",
        "start_e",
        "end_e",
        "events",
        "points",
        "chance",
        "start_type",
        "end_type",
        "flags",
    )

    def __init__(self, off_team: int, period: int, start_e: float, start_type: str):
        self.off_team = off_team
        self.period = period
        self.start_e = start_e
        self.end_e = start_e
        self.events: list[dict] = []
        self.points = 0
        self.chance = 1
        self.start_type = start_type
        self.end_type = ""
        self.flags: set[str] = set()

    def add(self, ev: dict, elapsed: float) -> None:
        ev["chance"] = self.chance
        self.events.append(ev)
        self.end_e = elapsed


def _parse_game(g: pd.DataFrame, home_id: int, away_id: int) -> tuple[list[dict], list[dict], int]:
    """Parse one game's feed rows into possessions.

    Returns (possession dicts, technical scoring events, orphan count).
    """
    rows = list(g.itertuples(index=False))
    poss_out: list[dict] = []
    tech_events: list[dict] = []
    orphans = 0
    late_orebs = 0
    dropped_rebs = 0
    cur: _Possession | None = None
    pending_and1: int | None = None
    last_miss_team: int | None = None
    gid = rows[0].gameId

    def close(end_type: str) -> None:
        nonlocal cur
        if cur is None:
            return
        cur.end_type = end_type
        poss_out.append(
            {
                "game_id": gid,
                "period": cur.period,
                "off_team_id": cur.off_team,
                "def_team_id": away_id if cur.off_team == home_id else home_id,
                "start_elapsed": cur.start_e,
                "end_elapsed": cur.end_e,
                "points": cur.points,
                "n_chances": cur.chance,
                "start_type": cur.start_type,
                "end_type": end_type,
                "flags": ",".join(sorted(cur.flags)),
                "events": cur.events,
            }
        )
        cur = None

    def ensure(off_team: int, elapsed: float, period: int, start_type: str) -> None:
        nonlocal cur
        if cur is not None and cur.off_team != off_team:
            cur.flags.add("forced_close")
            close("anomaly")
        if cur is None:
            cur = _Possession(off_team, period, elapsed, start_type)

    i = 0
    n = len(rows)
    while i < n:
        r = rows[i]
        at = _s(r.actionType)
        period = int(r.period)
        e = _elapsed(r.clock, period)

        if at == "period":
            if _s(r.subType) == "end":
                if cur is not None:
                    cur.flags.add("period_end")
                    close("period_end")
                pending_and1 = None
                last_miss_team = None
            i += 1
            continue

        if at in ADMIN_TYPES or at == "Violation":
            i += 1
            continue

        team = int(r.teamId) if pd.notna(r.teamId) else 0
        pid = int(r.personId) if pd.notna(r.personId) else 0
        if team == 0 and pid >= 1610000000:
            team = pid
            pid = 0

        if at in SHOT_TYPES:
            made = at == "Made Shot" or _s(r.shotResult) == "Made"
            desc = _s(r.description)
            ast = _AST_RE.search(desc) if made else None
            blk = _BLK_RE.search(desc) if not made else None
            ensure(team, e, period, "unknown" if cur is None else "")
            ev = {
                "t": "shot",
                "pid": pid,
                "made": int(made),
                "x": None if pd.isna(r.xLegacy) else float(r.xLegacy),
                "y": None if pd.isna(r.yLegacy) else float(r.yLegacy),
                "dist": None if pd.isna(r.shotDistance) else float(r.shotDistance),
                "sub": _s(r.subType),
                "pts": _shot_points(r) if made else 0,
                "ast": _name_pid(g, ast.group(1)) if ast else None,
                "blk": _name_pid(g, blk.group(1)) if blk else None,
            }
            if _is_heave(r, e):
                cur.flags.add("heave")
            cur.add(ev, e)
            if made:
                cur.points += ev["pts"]
                nxt = _next_material(rows, i + 1)
                if _is_and1(rows, i, nxt, pid):
                    pending_and1 = pid
                else:
                    close("make")
                    pending_and1 = None
                last_miss_team = None
            else:
                last_miss_team = team
            i += 1
            continue

        if at == "Free Throw":
            desc = _s(r.description)
            made = not desc.startswith("MISS")
            sub = _s(r.subType)
            if "Technical" in sub:
                tech_events.append(
                    {
                        "game_id": gid,
                        "period": period,
                        "elapsed": e,
                        "pid": pid,
                        "team": team,
                        "made": int(made),
                    }
                )
                i += 1
                continue
            nofm = _FT_NOFM.search(sub)
            n_of, m_of = (int(nofm.group(1)), int(nofm.group(2))) if nofm else (1, 1)
            ensure(team, e, period, "unknown" if cur is None else "")
            if pending_and1 is not None and pid == pending_and1:
                cur.flags.add("and1")
            cur.add(
                {
                    "t": "ft",
                    "pid": pid,
                    "made": int(made),
                    "n": n_of,
                    "m": m_of,
                    "flagrant": int("Flagrant" in sub),
                },
                e,
            )
            if made:
                cur.points += 1
            if n_of == m_of:
                pending_and1 = None
                if "Flagrant" in sub:
                    cur.chance += 1
                    cur.flags.add("flagrant_retain")
                elif made:
                    close("ft")
                    last_miss_team = None
                else:
                    last_miss_team = team
            i += 1
            continue

        if at == "Rebound":
            is_team_reb = pid == 0 or _s(r.playerName) == ""
            offensive = last_miss_team is not None and team == last_miss_team
            if cur is None:
                last = poss_out[-1] if poss_out else None
                if (
                    last is not None
                    and last["end_type"] == "make"
                    and last["period"] == period
                    and team == last["off_team_id"]
                    and abs(e - last["end_elapsed"]) <= 3.0
                ):
                    last["events"].append(
                        {
                            "t": "reb",
                            "pid": 0 if is_team_reb else pid,
                            "off": 1,
                            "team_reb": int(is_team_reb),
                            "late_logged": 1,
                            "chance": last["n_chances"],
                        }
                    )
                    last["n_chances"] += 1
                    late_orebs += 1
                elif last_miss_team is None:
                    dropped_rebs += 1
                else:
                    orphans += 1
                i += 1
                continue
            cur.add(
                {
                    "t": "reb",
                    "pid": 0 if is_team_reb else pid,
                    "off": int(offensive),
                    "team_reb": int(is_team_reb),
                },
                e,
            )
            if offensive:
                cur.chance += 1
            else:
                close("dreb")
            last_miss_team = None
            i += 1
            continue

        if at == "Turnover":
            desc = _s(r.description)
            stl = _STL_RE.search(desc)
            ensure(team, e, period, "unknown" if cur is None else "")
            cur.add(
                {
                    "t": "tov",
                    "pid": pid,
                    "sub": _s(r.subType),
                    "stl": _name_pid(g, stl.group(1)) if stl else None,
                },
                e,
            )
            close("tov")
            last_miss_team = None
            i += 1
            continue

        if at == "Foul":
            if cur is not None:
                cur.add({"t": "foul", "pid": pid, "sub": _s(r.subType), "team": team}, e)
            i += 1
            continue

        i += 1

    if cur is not None:
        cur.flags.add("game_end_open")
        close("period_end")
    return poss_out, tech_events, (orphans, late_orebs, dropped_rebs)


def _next_material(rows: list, j: int):
    while j < len(rows):
        at = _s(rows[j].actionType)
        if at not in ADMIN_TYPES and at not in ("Foul", "Violation"):
            return rows[j]
        if at == "period":
            return rows[j]
        j += 1
    return None


def _is_and1(rows: list, i: int, nxt, shooter_pid: int) -> bool:
    if nxt is None or _s(nxt.actionType) != "Free Throw":
        return False
    if pd.isna(nxt.personId) or int(nxt.personId) != shooter_pid:
        return False
    sub = _s(nxt.subType)
    return "1 of 1" in sub or "Flagrant" in sub


_NAME_CACHE: dict[tuple, int | None] = {}


def _name_pid(g: pd.DataFrame, name: str) -> int | None:
    """Resolve a description name fragment to a personId within this game."""
    key = (g.iloc[0].gameId, name)
    if key in _NAME_CACHE:
        return _NAME_CACHE[key]
    fam = name.split()[-1].lower()
    cands = g.loc[g.playerName.astype(str).str.lower() == fam, "personId"].dropna().unique()
    pid = int(cands[0]) if len(cands) == 1 else None
    _NAME_CACHE[key] = pid
    return pid


def _home_away(season: str) -> pd.DataFrame:
    df = pd.read_parquet(
        PROC / season / "regular_season.parquet",
        columns=["game_id", "team_id", "opp_team_id", "is_home"],
    )
    h = df[df.is_home][["game_id", "team_id", "opp_team_id"]].drop_duplicates("game_id")
    return h.rename(columns={"team_id": "home_id", "opp_team_id": "away_id"})


def _attach_lineups(pos: pd.DataFrame, season: str) -> pd.DataFrame:
    st = pd.read_parquet(STINTS / f"{season}_regular_season.parquet")
    st = st.sort_values(["game_id", "period", "start_elapsed"])
    keys = {}
    for gp, grp in st.groupby(["game_id", "period"], sort=False):
        keys[gp] = (
            grp["start_elapsed"].to_numpy(),
            grp[["h1", "h2", "h3", "h4", "h5"]].to_numpy(),
            grp[["a1", "a2", "a3", "a4", "a5"]].to_numpy(),
        )
    hl, al = [], []
    for r in pos.itertuples(index=False):
        k = (r.game_id, r.period)
        if k not in keys:
            hl.append(None)
            al.append(None)
            continue
        starts, harr, aarr = keys[k]
        lookup = max(float(r.start_elapsed), 0.0) + 1e-6
        idx = int(np.searchsorted(starts, lookup, side="right")) - 1
        idx = max(idx, 0)
        hl.append(harr[idx].tolist())
        al.append(aarr[idx].tolist())
    pos["home_lineup"] = hl
    pos["away_lineup"] = al
    return pos


def build(season: str) -> pd.DataFrame:
    pbp = pd.read_parquet(RAW / f"{season}_regular_season.parquet")
    pbp = pbp.sort_values("period", kind="stable")
    ha = _home_away(season).set_index("game_id")
    all_pos: list[dict] = []
    all_tech: list[dict] = []
    total_orphans = 0
    total_late = 0
    total_dropped = 0
    skipped = 0
    for gid, g in pbp.groupby("gameId", sort=False):
        if gid not in ha.index:
            skipped += 1
            continue
        home_id, away_id = int(ha.loc[gid, "home_id"]), int(ha.loc[gid, "away_id"])
        p, t, (o, lo, dr) = _parse_game(g.reset_index(drop=True), home_id, away_id)
        all_pos.extend(p)
        all_tech.extend(t)
        total_orphans += o
        total_late += lo
        total_dropped += dr
    pos = pd.DataFrame(all_pos)
    pos["possession_idx"] = pos.groupby("game_id").cumcount()
    pos = _attach_lineups(pos, season)
    pos["events"] = pos["events"].apply(json.dumps)
    pos["season"] = season
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pos.to_parquet(OUT_DIR / f"{season}_regular_season.parquet", index=False)
    tech = pd.DataFrame(all_tech)
    tech.to_parquet(OUT_DIR / f"{season}_technicals.parquet", index=False)
    print(
        f"{season}: {len(pos):,} possessions, {len(tech):,} technical FTs, "
        f"{total_orphans} orphans, {total_late} late putback rebounds, "
        f"{total_dropped} jumpball rebounds dropped, {skipped} games skipped"
    )
    return pos


def validate(season: str, quiet: bool = False) -> dict:
    pos = pd.read_parquet(OUT_DIR / f"{season}_regular_season.parquet")
    tech = pd.read_parquet(OUT_DIR / f"{season}_technicals.parquet")
    ev = pos.assign(events=pos.events.apply(json.loads))

    box = pd.read_parquet(PROC / season / "regular_season.parquet")
    box = box[~box.is_neutral] if "is_neutral" in box else box

    rec = []
    for r in ev.itertuples(index=False):
        for e in r.events:
            rec.append(
                {
                    "game_id": r.game_id,
                    "team": r.off_team_id,
                    "t": e["t"],
                    "made": e.get("made", 0),
                    "pts": e.get("pts", 0),
                    "off": e.get("off", 0),
                    "team_reb": e.get("team_reb", 0),
                }
            )
    evf = pd.DataFrame(rec)

    shots = (
        evf[evf.t == "shot"]
        .groupby(["game_id", "team"])
        .agg(fga=("t", "size"), fgm=("made", "sum"), shot_pts=("pts", "sum"))
    )
    fts = (
        evf[evf.t == "ft"].groupby(["game_id", "team"]).agg(fta=("t", "size"), ftm=("made", "sum"))
    )
    if len(tech):
        tf = tech.groupby(["game_id", "team"]).agg(fta_t=("made", "size"), ftm_t=("made", "sum"))
        fts = fts.join(tf, how="outer").fillna(0)
        fts["fta"] = fts["fta"] + fts["fta_t"]
        fts["ftm"] = fts["ftm"] + fts["ftm_t"]
        fts = fts[["fta", "ftm"]]
    tovs = evf[evf.t == "tov"].groupby(["game_id", "team"]).size().rename("tov")
    orbs = (
        evf[(evf.t == "reb") & (evf.off == 1) & (evf.team_reb == 0)]
        .groupby(["game_id", "team"])
        .size()
        .rename("oreb")
    )
    pts = pos.groupby(["game_id", "off_team_id"])["points"].sum().rename("pts")
    pts.index.names = ["game_id", "team"]
    tech_pts = (
        tech.groupby(["game_id", "team"])["made"].sum().rename("tech_pts")
        if len(tech)
        else pd.Series(dtype=float, name="tech_pts")
    )

    def _norm(piece):
        piece = piece.reset_index()
        piece["game_id"] = piece["game_id"].astype(str)
        piece["team"] = piece["team"].astype("int64")
        return piece.set_index(["game_id", "team"])

    pieces = [
        _norm(x.to_frame() if isinstance(x, pd.Series) else x)
        for x in (shots, fts, tovs, orbs, pts, tech_pts)
        if len(x)
    ]
    parsed = pd.concat(pieces, axis=1).fillna(0)
    bx = box.rename(columns={"team_id": "team"}).copy()
    bx["game_id"] = bx["game_id"].astype(str)
    bx["team"] = bx["team"].astype("int64")
    bx = bx.set_index(["game_id", "team"])[["pts", "fga", "fgm", "fta", "ftm", "tov", "oreb"]]
    j = parsed.join(bx, how="inner", rsuffix="_box")
    if "tech_pts" not in j:
        j["tech_pts"] = 0.0
    j["pts_all"] = j["pts"] + j["tech_pts"]

    summary: dict = {"season": season, "team_games": len(j), "possessions": len(pos)}
    if not quiet:
        print(f"team-games compared: {len(j):,}")
    for a, b in [
        ("pts_all", "pts_box"),
        ("fga", "fga_box"),
        ("fgm", "fgm_box"),
        ("fta", "fta_box"),
        ("ftm", "ftm_box"),
        ("tov", "tov_box"),
        ("oreb", "oreb_box"),
    ]:
        diff = (j[a] - j[b]).abs()
        bad = int((diff > 0).sum())
        summary[f"exact_{a}"] = round(1 - bad / len(j), 4)
        summary[f"maxdiff_{a}"] = int(diff.max())
        if not quiet:
            print(
                f"  {a:8s} vs {b:9s}: exact {len(j) - bad:,}/{len(j):,}  "
                f"mean|diff| {diff.mean():.3f}  max {diff.max():.0f}"
            )

    per_game = pos.groupby("game_id").size()
    est = box.groupby("game_id").apply(
        lambda t: 0.5 * (t.fga.sum() + 0.44 * t.fta.sum() - t.oreb.sum() + t.tov.sum()),
        include_groups=False,
    )
    both = pd.concat([(per_game / 2).rename("parsed"), est.rename("oliver")], axis=1).dropna()
    summary["poss_per_tg"] = round(both.parsed.mean(), 2)
    summary["oliver_per_tg"] = round(both.oliver.mean(), 2)
    summary["poss_corr"] = round(both.parsed.corr(both.oliver), 4)
    if not quiet:
        print(
            f"possessions/team-game: parsed {both.parsed.mean():.1f}  "
            f"oliver {both.oliver.mean():.1f}  corr {both.parsed.corr(both.oliver):.4f}"
        )

    ppp = pos["points"].value_counts().sort_index()
    summary["mean_ppp"] = round(float((pos.points).mean()), 4)
    summary["mean_chances"] = round(float(pos.n_chances.mean()), 4)
    summary["anomalies"] = int((pos.end_type == "anomaly").sum())
    if not quiet:
        print("points-per-possession distribution:")
        print((ppp / len(pos)).round(4).to_string())
        print(
            f"chances/possession mean {pos.n_chances.mean():.3f}  "
            f"anomaly closes: {(pos.end_type == 'anomaly').sum():,}"
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "validate"])
    ap.add_argument("season")
    args = ap.parse_args()
    seasons = (
        sorted(p.name for p in PROC.iterdir() if p.is_dir())
        if args.season == "all"
        else [args.season]
    )
    for s in seasons:
        if not (RAW / f"{s}_regular_season.parquet").exists():
            continue
        if args.cmd == "build":
            build(s)
        else:
            validate(s)


if __name__ == "__main__":
    main()
