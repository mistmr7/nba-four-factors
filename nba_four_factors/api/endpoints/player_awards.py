"""Wrapper for the playerawards endpoint.

Returns a player's career award history (All-NBA with team number, All-Defensive,
All-Star, MVP, DPOY, All-Rookie, and the weekly/monthly honors) keyed by the same
PERSON_ID we use everywhere else, so no name matching is needed.

Classic stats.nba.com resultSets shape: one result block with `headers` and
`rowSet`. The caller persists the raw JSON and parses it downstream.
"""

from __future__ import annotations

from typing import Any

from nba_four_factors.api.client import Client
from nba_four_factors.config import Endpoint


def fetch_player_awards(client: Client, player_id: int) -> dict[str, Any]:
    """Fetch the award history for a single player.

    Parameters fixed to:
        PlayerID    the NBA person id
        LeagueID=00 NBA
    """
    return client.fetch(Endpoint.PLAYER_AWARDS, {"PlayerID": int(player_id), "LeagueID": "00"})
