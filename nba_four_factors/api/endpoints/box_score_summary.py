"""Wrapper for the boxscoresummaryv3 endpoint.

Game summary: officials, inactives, line scores, and other metadata
useful for schedule reconciliation against leaguegamelog.

V3-only design (Session 5 pivot): V2 summary stopped being published
4/10/2025 per nba_api's release note, and V3 reaches cleanly back to
1997-98. The "v3" suffix lives only in the URL string; Python
identifiers drop the V2/V3 distinction since there is no longer a
choice to make.

V3 schema differs structurally from V2: ArenaInfo is split from
GameSummary, LastMeeting becomes LastFiveMeetings, SeasonSeries is
removed, and OFFICIAL_NAME splits into firstName/familyName/nameI.
All metadata our pipeline uses is present.

Returns the raw parsed JSON payload. The caller is responsible for
persisting it via storage.raw.raw_game_path / save_raw.
"""

from __future__ import annotations

from typing import Any

from nba_four_factors.api.client import Client
from nba_four_factors.config import Endpoint


def fetch_box_score_summary(client: Client, game_id: str) -> dict[str, Any]:
    """Fetch boxscoresummaryv3 for `game_id`."""
    params: dict[str, Any] = {"GameID": game_id}
    return client.fetch(Endpoint.BOXSCORE_SUMMARY, params)
