"""Wrapper for the leaguegamelog endpoint.

Schedule truth source (locked, Section 2.4). Enumerates every game in a
given (season, season_type) pair with team-level rows.

Returns the raw parsed JSON payload. The caller is responsible for
persisting it via storage.raw.raw_season_path / save_raw.
"""

from __future__ import annotations

from typing import Any

from nba_four_factors.api.client import Client
from nba_four_factors.config import Endpoint, SeasonType, season_to_hyphen


def fetch_league_game_log(
    client: Client,
    season: str,
    season_type: SeasonType,
) -> dict[str, Any]:
    """Fetch leaguegamelog for `season` (canonical underscore form) and `season_type`.

    Parameters fixed to:
        LeagueID=00         NBA (the only league we care about)
        PlayerOrTeam=T      Team-level rows, not player-level
        Counter=0           nba_api default; no pagination
        Sorter=DATE         Chronological
        Direction=DESC      nba_api default
        DateFrom, DateTo    Unbounded (empty strings)
    """
    params: dict[str, Any] = {
        "LeagueID": "00",
        "Season": season_to_hyphen(season),
        "SeasonType": season_type.value,
        "PlayerOrTeam": "T",
        "Counter": 0,
        "Sorter": "DATE",
        "Direction": "DESC",
        "DateFrom": "",
        "DateTo": "",
    }
    return client.fetch(Endpoint.SCHEDULE, params)
