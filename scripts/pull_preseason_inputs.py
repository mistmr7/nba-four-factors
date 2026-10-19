"""Pull the small set of inputs the preseason win-total module needs.

Three stats.nba.com endpoints, about 62 requests in total:

    drafthistory               one call, every draft since 1947
    leaguedashplayerbiostats   one call per season, age, height, draft slot
    commonteamroster           one call per team for the live season, the
                               current roster with birth date and experience

Raw payloads land under data/raw/preseason/ untouched; parsing happens in
nba_four_factors/preseason/inputs.py. Reruns skip files that already exist
unless --refresh is passed (use --refresh on rosters the morning of picks).

Run from the repo root:

    PYTHONPATH=. uv run python scripts/pull_preseason_inputs.py
    PYTHONPATH=. uv run python scripts/pull_preseason_inputs.py --only rosters --refresh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from nba_four_factors.api.client import Client, FetchError
from nba_four_factors.config import Endpoint

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "raw" / "preseason"
VEGAS = REPO / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"

FIRST_BIO_SEASON = 1996
LIVE_SEASON = "2026-27"

BIO_BOILER = {
    "College": "",
    "Conference": "",
    "Country": "",
    "DateFrom": "",
    "DateTo": "",
    "Division": "",
    "DraftPick": "",
    "DraftYear": "",
    "GameScope": "",
    "GameSegment": "",
    "Height": "",
    "LastNGames": "0",
    "LeagueID": "00",
    "Location": "",
    "Month": "0",
    "OpponentTeamID": "0",
    "Outcome": "",
    "PORound": "0",
    "PerMode": "Totals",
    "Period": "0",
    "PlayerExperience": "",
    "PlayerPosition": "",
    "SeasonSegment": "",
    "SeasonType": "Regular Season",
    "ShotClockRange": "",
    "StarterBench": "",
    "TeamID": "0",
    "VsConference": "",
    "VsDivision": "",
    "Weight": "",
}


def _fetch(client: Client, endpoint: str, params: dict[str, Any]) -> dict[str, Any] | None:
    shim = cast(Endpoint, SimpleNamespace(value=endpoint))
    for _attempt in range(2):
        try:
            return client.fetch(shim, params)
        except FetchError as exc:
            print(f"  {endpoint} {params.get('Season', '')} failed: {exc}")
    return None


def _save(path: Path, payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return True


def _season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[2:]}"


def pull_draft(client: Client, refresh: bool) -> None:
    f = OUT / "drafthistory.json"
    if f.exists() and not refresh:
        print("drafthistory: cached")
        return
    ok = _save(f, _fetch(client, "drafthistory", {"LeagueID": "00"}))
    print(f"drafthistory: {'ok' if ok else 'FAILED'}")


def pull_bios(client: Client, refresh: bool, last_start_year: int) -> None:
    n_ok = n_cached = 0
    for y in range(FIRST_BIO_SEASON, last_start_year + 1):
        label = _season_label(y)
        f = OUT / "bios" / f"{label.replace('-', '_')}.json"
        if f.exists() and not refresh:
            n_cached += 1
            continue
        if _save(f, _fetch(client, "leaguedashplayerbiostats", {**BIO_BOILER, "Season": label})):
            n_ok += 1
    print(f"bios: {n_ok} pulled, {n_cached} cached")


def pull_rosters(client: Client, refresh: bool, season: str) -> None:
    teams = sorted(pd.read_parquet(VEGAS, columns=["team_id"]).team_id.unique())
    n_ok = n_cached = 0
    for tid in teams:
        f = OUT / "rosters" / season.replace("-", "_") / f"{int(tid)}.json"
        if f.exists() and not refresh:
            n_cached += 1
            continue
        params = {"TeamID": int(tid), "Season": season, "LeagueID": "00"}
        if _save(f, _fetch(client, "commonteamroster", params)):
            n_ok += 1
    print(f"rosters {season}: {n_ok} pulled, {n_cached} cached, {len(teams)} teams")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["draft", "bios", "rosters"], default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--season", default=LIVE_SEASON)
    args = ap.parse_args()
    client = Client()
    last_start = int(args.season[:4]) - 1
    if args.only in (None, "draft"):
        pull_draft(client, args.refresh)
    if args.only in (None, "bios"):
        pull_bios(client, args.refresh, last_start)
    if args.only in (None, "rosters"):
        pull_rosters(client, args.refresh, args.season)


if __name__ == "__main__":
    main()
