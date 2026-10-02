"""Overnight puller for the credit-engine auxiliary endpoints.

Coverage-informed by data/features/endpoint_coverage.csv (probe run
2026-09-30): per-game endpoints pull only their measured windows, the
gamerotation endpoint is attempted era-wide because it holds data back
to 1997-98 but 500s intermittently, and season dashboards pull from
2013-14 (Synergy from 2012-13).

Standalone by design: reuses the api Client and rate limiter but keeps
its own lightweight checkpoints (data/checkpoints/aux/<endpoint>_<season>.json
holding completed and failed game id lists) rather than wiring into the
orchestration state machine. Raw payloads land under
data/raw/<endpoint>/ as one json per game (or per season for
dashboards). Fully resumable: rerun the same command and it continues.

Typical overnight commands, run from repo root:

    PYTHONPATH=. python3 scripts/pull_aux_endpoints.py --endpoint boxscoreplayertrackv3
    PYTHONPATH=. python3 scripts/pull_aux_endpoints.py --endpoint boxscorehustlev2
    PYTHONPATH=. python3 scripts/pull_aux_endpoints.py --endpoint boxscorematchupsv3
    PYTHONPATH=. python3 scripts/pull_aux_endpoints.py --endpoint gamerotation
    PYTHONPATH=. python3 scripts/pull_aux_endpoints.py --endpoint dashboards

Use --max-requests to cap a session (e.g. 12000 for about seven hours
at the default limiter). Failed ids are retried automatically on the
next run (two attempts inside a run; permanent 500s accumulate in the
failed list and are reported, not fatal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from nba_four_factors.api.client import Client, FetchError

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"
CKPT = REPO / "data" / "checkpoints" / "aux"

V3_RANGE = {
    "EndPeriod": "0",
    "EndRange": "0",
    "RangeType": "0",
    "StartPeriod": "0",
    "StartRange": "0",
}

GAME_ENDPOINTS: dict[str, tuple[str, dict[str, str]]] = {
    "boxscoreplayertrackv3": ("2013_14", V3_RANGE),
    "boxscorehustlev2": ("2016_17", {}),
    "boxscorematchupsv3": ("2017_18", V3_RANGE),
    "gamerotation": ("1997_98", {"LeagueID": "00"}),
}

DASH_PARAMS: dict[str, dict[str, str]] = {
    "leaguehustlestatsplayer": {"PerMode": "Totals", "SeasonType": "Regular Season"},
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
}

PTSTATS_BOILER = {
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
    "SeasonSegment": "",
    "SeasonType": "Regular Season",
    "StarterBench": "",
    "TeamID": "0",
    "Weight": "",
}
PTSTATS_MEASURES = ["Passing", "Rebounding", "Defense", "SpeedDistance", "Possessions"]

SYNERGY_PLAYTYPES = [
    "Isolation",
    "Transition",
    "PRBallHandler",
    "PRRollman",
    "Postup",
    "Spotup",
    "Handoff",
    "Cut",
    "OffScreen",
    "OffRebound",
    "Misc",
]


def _seasons(start: str) -> list[str]:
    return sorted(p.name for p in PROC.iterdir() if p.is_dir() and p.name >= start)


def _season_str(season_dir: str) -> str:
    return f"{season_dir[:4]}-{season_dir[5:]}"


def _game_ids(season: str) -> list[str]:
    df = pd.read_parquet(PROC / season / "regular_season.parquet", columns=["game_id", "game_date"])
    return df.sort_values("game_date")["game_id"].astype(str).unique().tolist()


def _load_ckpt(endpoint: str, season: str) -> dict:
    f = CKPT / f"{endpoint}_{season}.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"completed": [], "failed": []}


def _save_ckpt(endpoint: str, season: str, ck: dict) -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    (CKPT / f"{endpoint}_{season}.json").write_text(json.dumps(ck))


def _fetch(client: Client, endpoint: str, params: dict) -> dict | None:
    shim = SimpleNamespace(value=endpoint)
    for _attempt in range(2):
        try:
            return client.fetch(shim, params)  # type: ignore[arg-type]
        except FetchError:
            continue
    return None


def pull_game_endpoint(client: Client, endpoint: str, budget: list[int], sample: int = 0) -> None:
    start, extra = GAME_ENDPOINTS[endpoint]
    out_dir = RAW / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    for season in _seasons(start):
        ck = _load_ckpt(endpoint, season)
        done = set(ck["completed"])
        prior_failed = set(ck["failed"])
        ck["failed"] = []
        todo = [g for g in _game_ids(season) if g not in done]
        if sample:
            import random

            rng = random.Random(f"{endpoint}_{season}")
            todo = rng.sample(todo, min(sample, len(todo)))
        if not todo:
            continue
        n_ok = n_fail = 0
        for gid in todo:
            if budget[0] <= 0:
                _save_ckpt(endpoint, season, ck)
                print(
                    f"{endpoint} {season}: budget exhausted "
                    f"({n_ok} ok, {n_fail} failed this run)"
                )
                return
            payload = _fetch(client, endpoint, {"GameID": gid, **extra})
            budget[0] -= 1
            if payload is None:
                ck["failed"].append(gid)
                n_fail += 1
            else:
                (out_dir / f"{gid}.json").write_text(json.dumps(payload))
                ck["completed"].append(gid)
                n_ok += 1
            if (n_ok + n_fail) % 200 == 0:
                _save_ckpt(endpoint, season, ck)
            if (n_ok + n_fail) % 100 == 0:
                print(
                    f"  {endpoint} {season}: {n_ok + n_fail}/{len(todo)} " f"({n_fail} failed)",
                    flush=True,
                )
        _save_ckpt(endpoint, season, ck)
        retried = len(prior_failed & set(ck["completed"]))
        print(
            f"{endpoint} {season}: {n_ok} ok, {n_fail} failed, "
            f"{retried} previously-failed recovered",
            flush=True,
        )


def pull_dashboards(client: Client, budget: list[int]) -> None:
    out_dir = RAW / "season_dashboards"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str, season: str, params: dict) -> None:
        f = out_dir / f"{name}_{season}.json"
        if f.exists() or budget[0] <= 0:
            return
        payload = _fetch(client, name.split("__")[0], params)
        budget[0] -= 1
        if payload is not None:
            f.write_text(json.dumps(payload))
            print(f"dash {name} {season} ok", flush=True)
        else:
            print(f"dash {name} {season} FAILED", flush=True)

    for season in _seasons("2013_14"):
        s = _season_str(season)
        for ep, base in DASH_PARAMS.items():
            save(ep, season, {**base, "Season": s})
        for meas in PTSTATS_MEASURES:
            save(
                f"leaguedashptstats__{meas}",
                season,
                {**PTSTATS_BOILER, "PtMeasureType": meas, "Season": s},
            )
    for season in _seasons("2012_13"):
        s = _season_str(season)
        for pt in SYNERGY_PLAYTYPES:
            for grouping in ("offensive", "defensive"):
                save(
                    f"synergyplaytypes__{pt}_{grouping}",
                    season,
                    {
                        "LeagueID": "00",
                        "PerMode": "PerGame",
                        "PlayerOrTeam": "P",
                        "PlayType": pt,
                        "SeasonType": "Regular Season",
                        "SeasonYear": s,
                        "TypeGrouping": grouping,
                    },
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, choices=[*GAME_ENDPOINTS, "dashboards"])
    ap.add_argument("--max-requests", type=int, default=100_000)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    impatient = args.endpoint == "gamerotation"
    client = Client(max_attempts=2) if impatient else Client()
    budget = [args.max_requests]
    if args.endpoint == "dashboards":
        pull_dashboards(client, budget)
    else:
        pull_game_endpoint(client, args.endpoint, budget, sample=args.sample)
    print(f"done; {budget[0]} of {args.max_requests} requests unused")


if __name__ == "__main__":
    main()
