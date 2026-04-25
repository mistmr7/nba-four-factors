"""Wrapper for the boxscoreadvancedv3 endpoint.

V3 advanced box score. Used for 2025-26 onward (locked, Section 2.4
V2/V3 cutover). Advanced V2 is not formally deprecated, but we cut over
with Traditional and Summary at the 2025-26 boundary for consistency.

Schema differs from V2 (camelCase column names, expanded field names like
effectiveFieldGoalPercentage vs EFG_PCT, offensiveReboundPercentage vs
OREB_PCT, turnoverRatio vs TM_TOV_PCT). All four-factor reference metrics
present in V2 are present in V3.

Returns the raw parsed JSON payload. The caller is responsible for
persisting it via storage.raw.raw_game_path / save_raw.
"""

from __future__ import annotations

from typing import Any

from nba_four_factors.api.client import Client
from nba_four_factors.config import Endpoint


def fetch_box_score_advanced_v3(
    client: Client,
    game_id: str,
    start_period: str = "0",
    end_period: str = "0",
    start_range: str = "0",
    end_range: str = "0",
    range_type: str = "0",
) -> dict[str, Any]:
    """Fetch boxscoreadvancedv3 for `game_id`."""
    params: dict[str, Any] = {
        "GameID": game_id,
        "EndPeriod": end_period,
        "EndRange": end_range,
        "RangeType": range_type,
        "StartPeriod": start_period,
        "StartRange": start_range,
    }
    return client.fetch(Endpoint.BOXSCORE_ADVANCED_V3, params)
