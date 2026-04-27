"""Wrapper for the boxscoreadvancedv3 endpoint.

Advanced box score: per-player and per-team derived metrics (Pace,
offensive/defensive ratings, eFG%, TS%, usage, PIE) for cross-validation
of the four-factor computation and for richer downstream analysis.

V3-only design (Session 5 pivot): boxscoreadvancedv2 returns '{}' from
the API for every tested historical season — empirically dead, not
just deprecated for new data. V3 reaches cleanly back to 1997-98 (e.g.
1997-98 returns Pace 97.0, ORtg 93.9, possessions 98 — internally
consistent late-90s values). The "v3" suffix lives only in the URL
string; Python identifiers drop the V2/V3 distinction.

Note (locked, Section 2.4): V3 returns both `defensiveRating` and
`estimatedDefensiveRating` (and the same pair for offensive/net). The
`estimated*` variants return implausible values (>400) for old seasons
and should be filtered or ignored for any analysis that touches
pre-2010 data. The non-estimated variants are correct.

Returns the raw parsed JSON payload. The caller is responsible for
persisting it via storage.raw.raw_game_path / save_raw.
"""

from __future__ import annotations

from typing import Any

from nba_four_factors.api.client import Client
from nba_four_factors.config import Endpoint


def fetch_box_score_advanced(
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
    return client.fetch(Endpoint.BOXSCORE_ADVANCED, params)
