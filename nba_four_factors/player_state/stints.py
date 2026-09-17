"""Lineup stints from the raw play-by-play store (playbyplayv3 layout).

A stint is a stretch of one period during which the same ten players are on
the floor. Each row carries the five home and five away person IDs, seconds,
points for both sides, and the four ingredients of Oliver's possession
estimate (FGA, FTA, OREB, TOV) per side, so possessions are counted exactly
the way the rest of the thesis counts them.

Inputs, all already on disk:
    data/raw/playbyplay/<season>_<season_type>.parquet   playbyplayv3 rows
    data/raw/boxscoretraditionalv3/<game_id>.json         names, minutes, starters

Outputs, under data/player_state/:
    rosters/<season>_<season_type>.parquet   one row per player per game
    stints/<season>_<season_type>.parquet    the stint table
    stints/<season>_<season_type>_diag.parquet   one row per team-period

Three facts about the v3 feed drive the design.

1. Row order is chronological; actionNumber is not. A substitution entered
   late by the scorer gets an actionNumber appended at the end of the game
   while its row sits at the right place in the feed. Never sort on
   actionNumber; the stable sort on period below preserves feed order.
2. A substitution row names only the outgoing player by ID. The incoming
   player is in the description ("SUB: Crowder FOR Beasley") and is resolved
   against that game's box score roster by nameI ("J. Crowder"), family
   name, or first-name prefix plus family ("Ja. Green", "Marc Morris"), with
   accents folded. The box carries current names while descriptions carry
   the name in use that night, so the leading name of every description in
   the game is also registered as an alias for its person ID.
3. The feed does not say who starts a period after the first. Starters are
   inferred: a player substituted out before being substituted in started the
   period, and so did a player with a recorded action (shot, free throw,
   rebound, turnover, foul, violation, steal, block) who is never substituted
   in. A player who plays a whole period without any recorded action is
   invisible to that rule, so a second pass fills any short lineup with the
   roster player whose box score minutes are least accounted for by the
   stints already built. Period 1 starters come from the box score when it
   marks exactly five per side (the mid-2000s on); older files are inferred.

Run from repo root:
    python -m nba_four_factors.player_state.stints rosters
    python -m nba_four_factors.player_state.stints build
    python -m nba_four_factors.player_state.stints validate
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PBP_DIR = REPO / "data" / "raw" / "playbyplay"
RAW_BOX = REPO / "data" / "raw" / "boxscoretraditionalv3"
PROC = REPO / "data" / "processed"
OUT_DIR = REPO / "data" / "player_state"
ROSTERS = OUT_DIR / "rosters"
STINTS = OUT_DIR / "stints"

REGULATION_PERIOD_SECONDS = 720
OVERTIME_PERIOD_SECONDS = 300
FT_WEIGHT = 0.44

HOME_COLS = ["h1", "h2", "h3", "h4", "h5"]
AWAY_COLS = ["a1", "a2", "a3", "a4", "a5"]

# Foul subtypes that are not evidence the fouler was on the floor.
TECHNICAL_SUBTYPES = {
    "Technical",
    "Double Technical",
    "Hanging Technical",
    "Delay Technical",
    "Excess Timeout Technical",
    "Too Many Players Technical",
    "Non-Unsportsmanlike Technical",
    "Defense 3 Second",
}

SUB_RE = re.compile(r"^SUB:\s*(?P<in>.+?)\s+FOR\s+(?P<out>.+?)\s*$")
REB_COUNT_RE = re.compile(r"\(Off:(\d+) Def:(\d+)\)")


def period_seconds(period: int) -> int:
    return REGULATION_PERIOD_SECONDS if period <= 4 else OVERTIME_PERIOD_SECONDS


def parse_clock(clock: str) -> float:
    """'PT06M43.00S' to seconds remaining in the period."""
    m = re.match(r"PT(\d+)M([\d.]+)S", str(clock))
    if not m:
        raise ValueError(f"bad clock {clock!r}")
    return int(m.group(1)) * 60 + float(m.group(2))


def elapsed(period: int, clock: str) -> float:
    return period_seconds(period) - parse_clock(clock)


def parse_minutes_seconds(raw: str | None) -> float:
    if not raw:
        return 0.0
    s = str(raw)
    if ":" not in s:
        return 0.0
    mm, ss = s.split(":", 1)
    try:
        return float(mm) * 60.0 + float(ss)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Rosters from the raw box scores
# ---------------------------------------------------------------------------


def roster_from_box(path: Path) -> list[dict]:
    with open(path) as fh:
        d = json.load(fh)["boxScoreTraditional"]
    rows = []
    for side in ("homeTeam", "awayTeam"):
        t = d[side]
        for p in t["players"]:
            rows.append(
                {
                    "game_id": d["gameId"],
                    "team_id": int(t["teamId"]),
                    "is_home": side == "homeTeam",
                    "person_id": int(p["personId"]),
                    "first_name": p.get("firstName", "") or "",
                    "family_name": p.get("familyName", "") or "",
                    "name_i": p.get("nameI", "") or "",
                    "starter": bool(p.get("position", "")),
                    "seconds": parse_minutes_seconds(p.get("statistics", {}).get("minutes")),
                }
            )
    return rows


def build_rosters(season: str, season_type: str) -> pd.DataFrame:
    pbp_path = PBP_DIR / f"{season}_{season_type}.parquet"
    game_ids = pd.read_parquet(pbp_path, columns=["gameId"])["gameId"].unique()
    rows: list[dict] = []
    for gid in game_ids:
        p = RAW_BOX / f"{gid}.json"
        if p.exists():
            rows.extend(roster_from_box(p))
    roster = pd.DataFrame(rows)
    ROSTERS.mkdir(parents=True, exist_ok=True)
    roster.to_parquet(ROSTERS / f"{season}_{season_type}.parquet", index=False)
    return roster


def load_rosters(season: str, season_type: str) -> pd.DataFrame:
    return pd.read_parquet(ROSTERS / f"{season}_{season_type}.parquet")


# ---------------------------------------------------------------------------
# Event normalisation
# ---------------------------------------------------------------------------

STRONG_ACTIONS = {
    "Made Shot",
    "Missed Shot",
    "Free Throw",
    "Rebound",
    "Turnover",
    "Foul",
    "Violation",
}


def _norm(s) -> str:
    return str(s).strip() if isinstance(s, str) else ""


def fold(s) -> str:
    """Lowercase, accent-stripped, whitespace-trimmed: 'Šarić' and 'Saric' agree."""
    return (
        unicodedata.normalize("NFKD", _norm(s)).encode("ascii", "ignore").decode().lower().strip()
    )


PREFIX_RE = re.compile(r"^(?P<prefix>[a-z]{1,4})\.\s+(?P<family>.+)$")
SUFFIXES = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv"}


def leading_name(desc: str) -> str:
    """The actor's name at the front of a description: 'MISS B. Lopez 28\' 3PT' -> 'b. lopez'."""
    words = desc.split()
    if words and words[0].upper() == "MISS":
        words = words[1:]
    if not words or words[0].upper() in ("SUB:", "JUMP"):
        return ""
    name = words[0]
    if name.endswith(".") and len(words) > 1:
        name = f"{name} {words[1]}"
    return fold(name)


class GameRoster:
    """Name resolution and side lookup for one game."""

    def __init__(self, roster_g: pd.DataFrame):
        self.home_team = int(roster_g.loc[roster_g.is_home, "team_id"].iloc[0])
        self.away_team = int(roster_g.loc[~roster_g.is_home, "team_id"].iloc[0])
        self.side_of_team = {self.home_team: "home", self.away_team: "away"}
        self.team_of_side = {"home": self.home_team, "away": self.away_team}
        self.by_name_i: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_family: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.names: dict[int, tuple[str, str]] = {}
        self.seconds: dict[int, float] = {}
        self.side_of_player: dict[int, str] = {}
        self.starters: dict[str, set[int]] = {"home": set(), "away": set()}
        self.players: dict[str, set[int]] = {"home": set(), "away": set()}
        for r in roster_g.itertuples(index=False):
            side = "home" if r.is_home else "away"
            pid = int(r.person_id)
            self.by_name_i[(side, fold(r.name_i))].append(pid)
            self.by_family[(side, fold(r.family_name))].append(pid)
            self.names[pid] = (fold(r.first_name), fold(r.family_name))
            self.seconds[pid] = float(r.seconds)
            self.side_of_player[pid] = side
            self.players[side].add(pid)
            if r.starter:
                self.starters[side].add(pid)

    def add_feed_names(self, g: pd.DataFrame) -> None:
        """Register the names the feed's own descriptions use for each person ID.

        The box score and the feed's playerName column both carry a player's
        current name, while description text keeps the name in use that night
        ("SUB: Kanter FOR Ibaka" for the player the box calls Freedom). The
        leading name of every primary-actor description is therefore an alias
        worth knowing before substitutions are resolved.
        """
        if "description" not in g.columns:
            return
        acted = g[(g["personId"] > 0) & (g["personId"] < 1_000_000_000) & g["description"].notna()]
        seen: set[tuple[int, str]] = set()
        for r in acted[["personId", "description"]].itertuples(index=False):
            pid = int(r.personId)
            side = self.side_of_player.get(pid)
            if side is None:
                continue
            alias = leading_name(str(r.description))
            if not alias or (pid, alias) in seen:
                continue
            seen.add((pid, alias))
            key = (side, alias)
            if pid not in self.by_family[key]:
                self.by_family[key].append(pid)

    def resolve(self, side: str, name: str) -> int | None:
        """Person ID for a name as the feed writes it: 'Crowder', 'B. Lopez', 'Ja. Green', 'Marc Morris'."""
        cands = self.candidates(side, name)
        return cands[0] if len(cands) == 1 else None

    def candidates(self, side: str, name: str) -> list[int]:
        """Every roster player a feed name could mean; one entry when the name is unambiguous."""
        key = fold(name)
        hits = self.by_name_i.get((side, key)) or self.by_family.get((side, key)) or []
        if len(hits) == 1:
            return list(hits)
        m = PREFIX_RE.match(key)
        if m:
            prefix, fam = m.group("prefix"), m.group("family")
        elif " " in key and key.split(" ", 1)[1] not in SUFFIXES:
            prefix, fam = key.split(" ", 1)
        else:
            prefix, fam = "", key
        cands = [
            pid
            for pid in self.players[side]
            if self._family_match(self.names[pid][1], fam) and self.names[pid][0].startswith(prefix)
        ]
        if len(cands) == 1:
            return cands
        # One-name players the feed writes by the other name (the box has
        # first "Yao", family "Ming"; the feed writes "Ming").
        other = [pid for pid in self.players[side] if key in self.names[pid]]
        if len(other) == 1:
            return other
        return sorted(set(cands) | set(hits))

    @staticmethod
    def _family_match(roster_family: str, feed_family: str) -> bool:
        # "Hardaway Jr." vs "Hardaway", "Porter Jr." and similar suffix drift
        if roster_family == feed_family:
            return True
        return roster_family.startswith(feed_family + " ") or feed_family.startswith(
            roster_family + " "
        )


@dataclass
class Event:
    kind: str
    side: str | None
    pid: int
    at: float
    is_three: bool = False
    is_miss: bool = False
    is_oreb: bool | None = None
    technical: bool = False
    sub_in: int | None = None
    sub_out: int | None = None
    resolved: bool = True
    sub_in_candidates: tuple[int, ...] = ()


def normalise_events(g: pd.DataFrame, roster: GameRoster) -> list[tuple[int, Event]]:
    """Turn one game's v3 rows into (period, Event) in feed order."""
    out: list[tuple[int, Event]] = []
    off_counts: dict[int, int] = {}
    for r in g.itertuples(index=False):
        period = int(r.period)
        action = _norm(r.actionType)
        desc = _norm(r.description)
        pid = int(r.personId) if not pd.isna(r.personId) else 0
        team = int(r.teamId) if not pd.isna(r.teamId) else 0
        side = roster.side_of_team.get(team)
        if side is None and pid in roster.side_of_team:
            side = roster.side_of_team[pid]
            pid = 0
        if side is None:
            loc = _norm(r.location)
            side = "home" if loc == "h" else "away" if loc == "v" else None
        if pid > 1_000_000_000:
            pid = 0
        at = elapsed(period, r.clock)
        upper = desc.upper()

        if action == "Substitution":
            m = SUB_RE.match(desc)
            cands = roster.candidates(side, m.group("in")) if (m and side) else []
            sub_in = cands[0] if len(cands) == 1 else None
            out.append(
                (
                    period,
                    Event(
                        "sub",
                        side,
                        pid,
                        at,
                        sub_in=sub_in,
                        sub_out=pid,
                        resolved=sub_in is not None,
                        sub_in_candidates=tuple(cands),
                    ),
                )
            )
        elif action == "Made Shot":
            out.append((period, Event("fg", side, pid, at, is_three="3PT" in upper)))
        elif action == "Missed Shot":
            out.append((period, Event("fg", side, pid, at, is_miss=True)))
        elif action == "Free Throw":
            out.append((period, Event("ft", side, pid, at, is_miss="MISS" in upper)))
        elif action == "Rebound":
            m = REB_COUNT_RE.search(desc)
            is_oreb: bool | None = None
            if m and pid:
                off_count = int(m.group(1))
                is_oreb = off_count > off_counts.get(pid, 0)
                off_counts[pid] = off_count
            out.append((period, Event("reb", side, pid, at, is_oreb=is_oreb)))
        elif action == "Turnover":
            if "NO TURNOVER" not in upper:
                out.append((period, Event("tov", side, pid, at)))
        elif action == "Foul":
            out.append(
                (
                    period,
                    Event("foul", side, pid, at, technical=_norm(r.subType) in TECHNICAL_SUBTYPES),
                )
            )
        elif action == "Violation":
            out.append((period, Event("viol", side, pid, at)))
        elif action == "" and ("STEAL" in upper or "BLOCK" in upper):
            out.append((period, Event("stlblk", side, pid, at)))
    return out


def is_evidence(e: Event) -> bool:
    if e.pid == 0 or e.side is None:
        return False
    if e.kind == "foul":
        return not e.technical
    return e.kind in ("fg", "ft", "reb", "tov", "viol", "stlblk")


# ---------------------------------------------------------------------------
# Starters
# ---------------------------------------------------------------------------


def infer_period_starters(events: list[Event]) -> dict[str, set[int]]:
    """Players on the floor at the start of a period, from the period's events."""
    first_in: dict[tuple[str, int], int] = {}
    first_out: dict[tuple[str, int], int] = {}
    first_evt: dict[tuple[str, int], int] = {}
    for pos, e in enumerate(events):
        if e.kind == "sub":
            if e.side is None:
                continue
            first_out.setdefault((e.side, e.sub_out or 0), pos)
            # An ambiguous incoming name marks every candidate as substituted
            # in; a candidate who really started is short-listed instead and
            # restored by the minutes pass, which is the safe direction.
            for pid in (e.sub_in,) if e.sub_in is not None else e.sub_in_candidates:
                first_in.setdefault((e.side, pid), pos)
        elif is_evidence(e):
            first_evt.setdefault((e.side or "", e.pid), pos)
    starters: dict[str, set[int]] = {"home": set(), "away": set()}
    for key in set(first_out) | set(first_evt):
        side, pid = key
        if pid == 0 or side not in starters:
            continue
        fin = first_in.get(key, np.inf)
        fout = first_out.get(key, np.inf)
        # A player substituted in without first being substituted out did not
        # start the period, whatever the feed shows before the sub: those are
        # ordering glitches (a sub logged after the free throw it preceded).
        if fout < fin or (fin == np.inf and key in first_evt):
            starters[side].add(pid)
    return starters


# ---------------------------------------------------------------------------
# Stint accumulation
# ---------------------------------------------------------------------------


@dataclass
class Stint:
    game_id: str
    period: int
    stint_idx: int
    start: float
    home: tuple[int, ...]
    away: tuple[int, ...]
    end: float = 0.0
    c: dict = field(default_factory=lambda: defaultdict(int))

    def row(self) -> dict:
        c = self.c
        home_poss = c["home_fga"] - c["home_oreb"] + c["home_tov"] + FT_WEIGHT * c["home_fta"]
        away_poss = c["away_fga"] - c["away_oreb"] + c["away_tov"] + FT_WEIGHT * c["away_fta"]
        out = {
            "game_id": self.game_id,
            "period": self.period,
            "stint_idx": self.stint_idx,
            "start_elapsed": self.start,
            "end_elapsed": self.end,
            "seconds": max(self.end - self.start, 0.0),
            "home_pts": c["home_pts"],
            "away_pts": c["away_pts"],
            "home_fga": c["home_fga"],
            "away_fga": c["away_fga"],
            "home_fta": c["home_fta"],
            "away_fta": c["away_fta"],
            "home_oreb": c["home_oreb"],
            "away_oreb": c["away_oreb"],
            "home_tov": c["home_tov"],
            "away_tov": c["away_tov"],
            "home_poss": home_poss,
            "away_poss": away_poss,
            "poss": 0.5 * (home_poss + away_poss),
        }
        for i, pid in enumerate(sorted(self.home)):
            out[HOME_COLS[i]] = pid
        for i, pid in enumerate(sorted(self.away)):
            out[AWAY_COLS[i]] = pid
        return out


def walk_period(
    game_id: str, period: int, events: list[Event], starters: dict[str, set[int]]
) -> tuple[list[dict], str]:
    """Cut one period into stints. Returns (rows, problem) with problem '' when clean."""
    lineup = {"home": set(starters["home"]), "away": set(starters["away"])}
    plen = period_seconds(period)
    rows: list[dict] = []
    idx = 0
    cur = Stint(game_id, period, idx, 0.0, tuple(lineup["home"]), tuple(lineup["away"]))
    pending: list[Event] = []
    pending_at: float | None = None
    last_miss: str | None = None

    def flush(at: float) -> str:
        nonlocal cur, idx
        cur.end = at
        rows.append(cur.row())
        for s in pending:
            if s.sub_in is None:
                # An ambiguous feed name ("SUB: Williams FOR Foye" with two
                # Williamses in uniform) is settled by whoever is not already
                # on the floor, when that leaves exactly one candidate.
                off_floor = [pid for pid in s.sub_in_candidates if pid not in lineup[s.side]]
                if len(off_floor) != 1:
                    return "unresolved_sub"
                s.sub_in = off_floor[0]
                s.resolved = True
            lineup[s.side].discard(s.sub_out)
            lineup[s.side].add(s.sub_in)
        pending.clear()
        if len(lineup["home"]) != 5 or len(lineup["away"]) != 5:
            return "lineup_size"
        idx += 1
        cur = Stint(game_id, period, idx, at, tuple(lineup["home"]), tuple(lineup["away"]))
        return ""

    for e in events:
        if e.kind == "sub":
            if e.side is None:
                continue
            if pending and pending_at is not None and e.at != pending_at:
                problem = flush(pending_at)
                if problem:
                    return rows, problem
            pending.append(e)
            pending_at = e.at
            continue
        if pending:
            problem = flush(pending_at if pending_at is not None else e.at)
            if problem:
                return rows, problem
        s = e.side
        if s is None:
            continue
        c = cur.c
        if e.kind == "fg":
            c[f"{s}_fga"] += 1
            if e.is_miss:
                last_miss = s
            else:
                c[f"{s}_pts"] += 3 if e.is_three else 2
                last_miss = None
        elif e.kind == "ft":
            c[f"{s}_fta"] += 1
            if e.is_miss:
                last_miss = s
            else:
                c[f"{s}_pts"] += 1
                last_miss = None
        elif e.kind == "reb":
            # Team rebounds in the feed are mostly dead-ball bookkeeping (a
            # missed first free throw, a miss followed by a foul) and the box
            # score does not count them, so neither does the thesis. Only
            # player rebounds count toward OREB. Two signals, either suffices:
            # the rebound follows the rebounding side's own miss, or the
            # feed's running (Off:n) counter for the player went up. Each
            # signal only ever undercounts (some seasons log a putback miss
            # after the rebound; the counter sometimes lags a row), so their
            # union reconciles with the box score.
            if e.pid != 0 and (e.is_oreb or last_miss == s):
                c[f"{s}_oreb"] += 1
            last_miss = None
        elif e.kind == "tov":
            c[f"{s}_tov"] += 1
            last_miss = None

    if pending:
        problem = flush(pending_at if pending_at is not None else plen)
        if problem:
            return rows, problem
    cur.end = plen
    rows.append(cur.row())
    return rows, ""


def player_seconds(rows: list[dict]) -> dict[int, float]:
    acc: dict[int, float] = defaultdict(float)
    for r in rows:
        for c in HOME_COLS + AWAY_COLS:
            acc[r[c]] += r["seconds"]
    return acc


def build_game_stints(g: pd.DataFrame, roster_g: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """All stints for one game plus one diagnostic row per team-period."""
    game_id = str(g["gameId"].iloc[0])
    roster = GameRoster(roster_g)
    roster.add_feed_names(g)
    g = g.sort_values("period", kind="stable")
    by_period: dict[int, list[Event]] = defaultdict(list)
    for period, e in normalise_events(g, roster):
        by_period[period].append(e)

    starters: dict[int, dict[str, set[int]]] = {}
    for period, events in by_period.items():
        st = infer_period_starters(events)
        # The box score names the game starters, but only from the mid-2000s
        # on does its position field mark exactly five per side; older files
        # tag most of the roster, so the box lineup is used only when clean.
        box = roster.starters
        if period == 1 and len(box["home"]) == 5 and len(box["away"]) == 5:
            st = {"home": set(box["home"]), "away": set(box["away"])}
        starters[period] = st

    # Pass 1: periods whose starters resolve to five a side.
    rows_by_period: dict[int, list[dict]] = {}
    problems: dict[int, str] = {}
    for period, events in by_period.items():
        st = starters[period]
        if len(st["home"]) == 5 and len(st["away"]) == 5:
            rows, problem = walk_period(game_id, period, events, st)
            if problem:
                problems[period] = problem
            else:
                rows_by_period[period] = rows
        elif len(st["home"]) > 5 or len(st["away"]) > 5:
            problems[period] = "too_many_starters"

    # Pass 2: fill short lineups from box score minutes not yet accounted for.
    accounted = player_seconds([r for rows in rows_by_period.values() for r in rows])
    for period, events in by_period.items():
        if period in rows_by_period or period in problems:
            continue
        st = {s: set(v) for s, v in starters[period].items()}
        subbed_in = {(e.side, e.sub_in) for e in events if e.kind == "sub" and e.sub_in is not None}
        for side in ("home", "away"):
            need = 5 - len(st[side])
            if need <= 0:
                continue
            cands = [
                (roster.seconds[pid] - accounted.get(pid, 0.0), pid)
                for pid in roster.players[side]
                if pid not in st[side] and (side, pid) not in subbed_in
            ]
            cands.sort(reverse=True)
            for gap, pid in cands[:need]:
                if gap > 0:
                    st[side].add(pid)
        if len(st["home"]) == 5 and len(st["away"]) == 5:
            rows, problem = walk_period(game_id, period, events, st)
            if problem:
                problems[period] = problem
            else:
                rows_by_period[period] = rows
                starters[period] = st
        else:
            problems[period] = "short_starters"

    all_rows = [r for p in sorted(rows_by_period) for r in rows_by_period[p]]
    diag = [
        {
            "game_id": game_id,
            "period": p,
            "home_starters": len(starters[p]["home"]),
            "away_starters": len(starters[p]["away"]),
            "usable": p in rows_by_period,
            "problem": problems.get(p, ""),
            "unresolved_subs": sum(1 for e in by_period[p] if e.kind == "sub" and not e.resolved),
        }
        for p in sorted(by_period)
    ]
    return all_rows, diag


# ---------------------------------------------------------------------------
# Season drivers
# ---------------------------------------------------------------------------


def game_dates(season: str, season_type: str) -> pd.DataFrame:
    p = pd.read_parquet(PROC / season / f"{season_type}.parquet", columns=["game_id", "game_date"])
    p["game_date"] = pd.to_datetime(p["game_date"])
    return p.drop_duplicates("game_id")


def build_season(
    season: str, season_type: str = "regular_season"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp = pd.read_parquet(PBP_DIR / f"{season}_{season_type}.parquet")
    roster_path = ROSTERS / f"{season}_{season_type}.parquet"
    roster = (
        pd.read_parquet(roster_path) if roster_path.exists() else build_rosters(season, season_type)
    )
    rows: list[dict] = []
    diag: list[dict] = []
    roster_groups = {gid: rg for gid, rg in roster.groupby("game_id")}
    for gid, g in pbp.groupby("gameId", sort=True):
        rg = roster_groups.get(str(gid))
        if rg is None:
            continue
        r, d = build_game_stints(g, rg)
        rows.extend(r)
        diag.extend(d)
    stints = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag)
    dates = game_dates(season, season_type)
    stints = stints.merge(dates, on="game_id", how="left")
    stints["season"] = season
    stints["season_type"] = season_type
    STINTS.mkdir(parents=True, exist_ok=True)
    stints.to_parquet(STINTS / f"{season}_{season_type}.parquet", index=False)
    diag_df.to_parquet(STINTS / f"{season}_{season_type}_diag.parquet", index=False)
    return stints, diag_df


def load_stints(seasons: list[str], season_type: str = "regular_season") -> pd.DataFrame:
    parts = [pd.read_parquet(STINTS / f"{s}_{season_type}.parquet") for s in seasons]
    return pd.concat(parts, ignore_index=True)


def validate_season(season: str, season_type: str = "regular_season") -> dict:
    """Stint totals against the team box score and player minutes against the roster."""
    st = pd.read_parquet(STINTS / f"{season}_{season_type}.parquet")
    diag = pd.read_parquet(STINTS / f"{season}_{season_type}_diag.parquet")
    roster = load_rosters(season, season_type)
    team = pd.read_parquet(PROC / season / f"{season_type}.parquet")
    full = diag.groupby("game_id")["usable"].all()
    full_games = full[full].index

    # Per-team-game totals from stints, home and away stacked.
    home = st.groupby("game_id")[
        ["home_pts", "home_fga", "home_fta", "home_oreb", "home_tov"]
    ].sum()
    home.columns = ["pts", "fga", "fta", "oreb", "tov"]
    home["is_home"] = True
    away = st.groupby("game_id")[
        ["away_pts", "away_fga", "away_fta", "away_oreb", "away_tov"]
    ].sum()
    away.columns = ["pts", "fga", "fta", "oreb", "tov"]
    away["is_home"] = False
    mine = pd.concat([home, away]).reset_index()
    mine = mine[mine.game_id.isin(full_games)]
    box = team[["game_id", "is_home", "pts", "fga", "fta", "oreb", "tov"]]
    j = mine.merge(box, on=["game_id", "is_home"], suffixes=("_stint", "_box"))
    report = {
        "season": season,
        "games": int(st.game_id.nunique()),
        "fully_usable_games": len(full_games),
    }
    for c in ["pts", "fga", "fta", "oreb", "tov"]:
        diff = (j[f"{c}_stint"] - j[f"{c}_box"]).abs()
        report[f"{c}_exact_share"] = float((diff == 0).mean())
        report[f"{c}_mean_abs_diff"] = float(diff.mean())

    # Player seconds from stints against the box score.
    parts = [
        st[["game_id", c, "seconds"]].rename(columns={c: "person_id"})
        for c in HOME_COLS + AWAY_COLS
    ]
    ps = pd.concat(parts).groupby(["game_id", "person_id"], as_index=False)["seconds"].sum()
    ps = ps[ps.game_id.isin(full_games)]
    jr = roster.merge(ps, on=["game_id", "person_id"], how="left", suffixes=("_box", "_stint"))
    jr = jr[jr.game_id.isin(full_games)]
    jr["seconds_stint"] = jr["seconds_stint"].fillna(0.0)
    d = (jr["seconds_stint"] - jr["seconds_box"]).abs()
    report["player_seconds_mean_abs_diff"] = float(d.mean())
    report["player_seconds_within_60s_share"] = float((d <= 60).mean())
    report["team_periods_usable_share"] = float(diag.usable.mean())
    report["unresolved_subs"] = int(diag.unresolved_subs.sum())
    return report


def available_seasons(season_type: str) -> list[str]:
    """Seasons with a play-by-play file on disk, oldest first."""
    suffix = f"_{season_type}.parquet"
    return sorted(p.name[: -len(suffix)] for p in PBP_DIR.glob(f"*{suffix}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rosters", "build", "validate"):
        s = sub.add_parser(name)
        s.add_argument(
            "--seasons", nargs="*", default=None, help="canonical ids like 2023_24; default all"
        )
        s.add_argument("--season-type", default="regular_season")
    a = ap.parse_args()
    seasons = a.seasons or available_seasons(a.season_type)
    for season in seasons:
        t0 = time.time()
        if a.cmd == "rosters":
            r = build_rosters(season, a.season_type)
            print(
                f"{season} rosters {len(r):,} rows {r.game_id.nunique():,} games {time.time() - t0:.0f}s"
            )
        elif a.cmd == "build":
            st, diag = build_season(season, a.season_type)
            print(
                f"{season} stints {len(st):,} rows {st.game_id.nunique():,} games, "
                f"team-periods usable {diag.usable.mean():.4f}, {time.time() - t0:.0f}s"
            )
        else:
            rep = validate_season(season, a.season_type)
            print(json.dumps(rep))


if __name__ == "__main__":
    main()
