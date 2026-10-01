"""Probe stats.nba.com endpoint coverage across all seasons on disk.

The play-by-play lane's design docs (Credit_Engine_Design.md) mark every
endpoint availability window as "verify at ingest". This script does the
verifying: for each candidate endpoint and each season, it makes ONE
request (a mid-season game for game-level endpoints, the season itself
for dashboards) and records whether real rows came back.

Roughly 230 requests total; at the default rate limiter this is under
ten minutes. Run from repo root on a machine with network access:

    PYTHONPATH=. python3 scripts/endpoint_coverage_probe.py

Output: data/features/endpoint_coverage.csv plus a printed matrix.
Statuses: ok(n) with row count, empty (200 but no rows), error(code),
timeout. A season is "covered" only on ok with plausible row counts.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from nba_four_factors.api.client import Client, FetchError

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "data" / "features" / "endpoint_coverage.csv"

# Game-level endpoints: probed with one mid-season GameID per season.
GAME_ENDPOINTS: dict[str, dict[str, str]] = {
    "gamerotation": {"LeagueID": "00"},
    "boxscorehustlev2": {},
    "boxscoreplayertrackv3": {
        "EndPeriod": "0",
        "EndRange": "0",
        "RangeType": "0",
        "StartPeriod": "0",
        "StartRange": "0",
    },
    "boxscorematchupsv3": {
        "EndPeriod": "0",
        "EndRange": "0",
        "RangeType": "0",
        "StartPeriod": "0",
        "StartRange": "0",
    },
}


# Season-level dashboards: probed once per season.
# Each entry: (params_builder, notes)
def _season_str(season_dir: str) -> str:
    y0 = season_dir[:4]
    return f"{y0}-{season_dir[5:]}"


SEASON_ENDPOINTS: dict[str, dict[str, str]] = {
    "leaguehustlestatsplayer": {
        "PerMode": "Totals",
        "SeasonType": "Regular Season",
    },
    "leaguedashptdefend": {
        "DefenseCategory": "Overall",
        "LeagueID": "00",
        "PerMode": "Totals",
        "SeasonType": "Regular Season",
    },
    "leaguedashplayerptshot": {
        "LeagueID": "00",
        "PerMode": "Totals",
        "SeasonType": "Regular Season",
    },
    "leaguedashptstats": {
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "GameScope": "",
        "Height": "",
        "LastNGames": "0",
        "LeagueID": "00",
        "Location": "",
        "Month": "0",
        "OpponentTeamID": "0",
        "Outcome": "",
        "PORound": "0",
        "PerMode": "Totals",
        "PlayerExperience": "",
        "PlayerOrTeam": "Player",
        "PlayerPosition": "",
        "PtMeasureType": "Passing",
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "StarterBench": "",
        "TeamID": "0",
        "Weight": "",
    },
    "synergyplaytypes": {
        "LeagueID": "00",
        "PerMode": "PerGame",
        "PlayerOrTeam": "P",
        "PlayType": "Isolation",
        "SeasonType": "Regular Season",
        "TypeGrouping": "offensive",
    },
}


def _mid_season_game(season_dir: str) -> str | None:
    """A game id from the middle of the season (stable, never opening night)."""
    f = PROC / season_dir / "regular_season.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f, columns=["game_id", "game_date"])
    df = df.sort_values("game_date")
    return str(df["game_id"].iloc[len(df) // 2])


def _count_rows(payload: dict) -> int:
    """Total data rows in either resultSets (v2) or nested (v3) payloads."""
    n = 0
    for rs in payload.get("resultSets", []):
        if isinstance(rs, dict):
            n += len(rs.get("rowSet", []) or [])
    if not n:

        def walk(node: object) -> int:
            if isinstance(node, list):
                return (
                    len(node) if node and isinstance(node[0], dict) else sum(walk(x) for x in node)
                )
            if isinstance(node, dict):
                return sum(walk(v) for v in node.values())
            return 0

        for key in ("boxScoreHustle", "boxScoreMatchups", "boxScorePlayerTrack", "game"):
            if key in payload:
                n = walk(payload[key])
                break
        else:
            n = walk(payload)
    return n


def _signal(node: object) -> int:
    """Count nonzero numeric leaf values: shell payloads score near zero."""
    if isinstance(node, bool):
        return 0
    if isinstance(node, int | float):
        return 1 if node not in (0, 0.0) else 0
    if isinstance(node, list):
        return sum(_signal(x) for x in node)
    if isinstance(node, dict):
        return sum(_signal(v) for v in node.values())
    return 0


def probe(client: Client, endpoint: str, params: dict[str, str]) -> str:
    shim = SimpleNamespace(value=endpoint)
    try:
        payload = client.fetch(shim, params)  # type: ignore[arg-type]
    except FetchError as exc:
        return f"error({exc})"[:60]
    except (json.JSONDecodeError, ValueError):
        return "badjson"
    rows = _count_rows(payload)
    if not rows:
        return "empty"
    sig = _signal(payload)
    return f"ok({rows}/s{sig})"


def main() -> None:
    client = Client()
    seasons = sorted(
        p.name for p in PROC.iterdir() if p.is_dir() and glob.glob(str(p / "*.parquet"))
    )
    records: list[dict[str, str]] = []
    for season in seasons:
        gid = _mid_season_game(season)
        for ep, extra in GAME_ENDPOINTS.items():
            if gid is None:
                records.append({"season": season, "endpoint": ep, "status": "no_game"})
                continue
            status = probe(client, ep, {"GameID": gid, **extra})
            records.append({"season": season, "endpoint": ep, "status": status})
            print(f"{season}  {ep:26s} {status}", flush=True)
        for ep, base in SEASON_ENDPOINTS.items():
            params = dict(base)
            key = "SeasonYear" if ep == "synergyplaytypes" else "Season"
            params[key] = _season_str(season)
            status = probe(client, ep, params)
            records.append({"season": season, "endpoint": ep, "status": status})
            print(f"{season}  {ep:26s} {status}", flush=True)

    retry = [(k, rec) for k, rec in enumerate(records) if rec["status"].startswith("error")]
    if retry:
        print(f"\nretrying {len(retry)} errored probes once...")
        for k, rec in retry:
            season, ep = rec["season"], rec["endpoint"]
            if ep in GAME_ENDPOINTS:
                gid = _mid_season_game(season)
                if gid is None:
                    continue
                status = probe(client, ep, {"GameID": gid, **GAME_ENDPOINTS[ep]})
            else:
                params = dict(SEASON_ENDPOINTS[ep])
                key = "SeasonYear" if ep == "synergyplaytypes" else "Season"
                params[key] = _season_str(season)
                status = probe(client, ep, params)
            records[k]["status"] = status
            print(f"{season}  {ep:26s} {status}", flush=True)

    df = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")

    def _covered(s: str) -> bool:
        if not s.startswith("ok("):
            return False
        body = s[3:-1]
        rows_part, _, sig_part = body.partition("/s")
        try:
            return int(rows_part) > 2 and (not sig_part or int(sig_part) > 100)
        except ValueError:
            return False

    matrix = df.assign(ok=df["status"].map(_covered)).pivot_table(
        index="season", columns="endpoint", values="ok", aggfunc="first"
    )
    print(matrix.replace({True: "Y", False: "-"}).to_string())


if __name__ == "__main__":
    main()
