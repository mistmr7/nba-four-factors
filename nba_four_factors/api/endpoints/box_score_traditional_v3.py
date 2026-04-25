"""Wrapper for the boxscoretraditionalv3 endpoint.

V3 traditional box score. Used for 2025-26 onward (locked, Section 2.4
V2/V3 cutover). V2 stopped being published as of 2025-26 per nba_api's
deprecation warning.

Schema differs from V2 (camelCase column names, expanded field names like
fieldGoalsMade vs FGM), but all four-factor inputs are present:
fieldGoalsMade, fieldGoalsAttempted, threePointersMade,
freeThrowsAttempted, reboundsOffensive, turnovers, points, minutes.
The processed layer normalizes V2 and V3 to a single canonical schema.

Returns the raw parsed JSON payload. The caller is responsible for
persisting it via storage.raw.raw_game_path / save_raw.
"""

from __future__ import annotations

from typing import Any

from nba_four_factors.api.client import Client
from nba_four_factors.config import Endpoint


def fetch_box_score_traditional_v3(
    client: Client,
    game_id: str,
    start_period: str = "0",
    end_period: str = "0",
    start_range: str = "0",
    end_range: str = "0",
    range_type: str = "0",
) -> dict[str, Any]:
    """Fetch boxscoretraditionalv3 for `game_id`."""
    params: dict[str, Any] = {
        "GameID": game_id,
        "EndPeriod": end_period,
        "EndRange": end_range,
        "RangeType": range_type,
        "StartPeriod": start_period,
        "StartRange": start_range,
    }
    return client.fetch(Endpoint.BOXSCORE_TRADITIONAL_V3, params)
