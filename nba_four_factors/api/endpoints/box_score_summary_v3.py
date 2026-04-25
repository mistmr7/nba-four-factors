"""Wrapper for the boxscoresummaryv3 endpoint.

V3 game summary. Used for 2025-26 onward (locked, Section 2.4 V2/V3
cutover). V2 Summary stopped being published 4/10/2025 per nba_api's
release note.

Schema is structurally different from V2: ArenaInfo is split from
GameSummary, LastMeeting (1 game) becomes LastFiveMeetings, SeasonSeries
is removed, and OFFICIAL_NAME is split into firstName/familyName/nameI.
All metadata our pipeline uses (line scores, inactives, referees,
schedule reconciliation fields) is present in both versions.

Returns the raw parsed JSON payload. The caller is responsible for
persisting it via storage.raw.raw_game_path / save_raw.
"""

from __future__ import annotations

from typing import Any

from nba_four_factors.api.client import Client
from nba_four_factors.config import Endpoint


def fetch_box_score_summary_v3(client: Client, game_id: str) -> dict[str, Any]:
    """Fetch boxscoresummaryv3 for `game_id`."""
    params: dict[str, Any] = {"GameID": game_id}
    return client.fetch(Endpoint.BOXSCORE_SUMMARY_V3, params)
