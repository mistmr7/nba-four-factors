"""Wrapper for the boxscoresummaryv2 endpoint.

Game-level metadata: arena, attendance, officials (referee assignments),
inactive players, line scores by period, last-meeting info. Used for
schedule reconciliation and anomaly investigation (locked, Section 2.4).

Single-parameter endpoint: GameID only.

Returns the raw parsed JSON payload. The caller is responsible for
persisting it via storage.raw.raw_game_path / save_raw.
"""

from __future__ import annotations

from typing import Any

from nba_four_factors.api.client import Client
from nba_four_factors.config import Endpoint


def fetch_box_score_summary(client: Client, game_id: str) -> dict[str, Any]:
    """Fetch boxscoresummaryv2 for `game_id`."""
    params: dict[str, Any] = {"GameID": game_id}
    return client.fetch(Endpoint.BOXSCORE_SUMMARY, params)
