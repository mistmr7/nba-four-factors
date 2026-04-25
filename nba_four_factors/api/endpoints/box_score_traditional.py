"""Wrapper for the boxscoretraditionalv2 endpoint.

Traditional box score: per-player and per-team totals (points, rebounds,
assists, steals, blocks, turnovers, FG/FT attempts and makes, etc.).
Source of truth for the four-factors computation (locked, Section 2.4).

Used for seasons 1997-98 through 2024-25. For 2025-26 onward, use
fetch_box_score_traditional_v3 — V2 stopped being published as of the
2025-26 season per nba_api's deprecation notice. The cutover is handled
by config.endpoint_for_season; callers that go through the orchestration
layer don't need to choose explicitly.

Returns the raw parsed JSON payload. The caller is responsible for
persisting it via storage.raw.raw_game_path / save_raw.
"""

from __future__ import annotations

from typing import Any

from nba_four_factors.api.client import Client
from nba_four_factors.config import Endpoint


def fetch_box_score_traditional(
    client: Client,
    game_id: str,
    start_period: str = "0",
    end_period: str = "0",
    start_range: str = "0",
    end_range: str = "0",
    range_type: str = "0",
) -> dict[str, Any]:
    """Fetch boxscoretraditionalv2 for `game_id`.

    Range parameters are passed through as strings to match nba_api's
    own defaults. Callers that do not slice by period or range can leave
    everything at "0" and receive the full-game box score.
    """
    params: dict[str, Any] = {
        "GameID": game_id,
        "EndPeriod": end_period,
        "EndRange": end_range,
        "RangeType": range_type,
        "StartPeriod": start_period,
        "StartRange": start_range,
    }
    return client.fetch(Endpoint.BOXSCORE_TRADITIONAL, params)
