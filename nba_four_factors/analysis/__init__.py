"""Analysis layer: regression analyses on the processed four-factors data.

Public API:

    load_processed
        Read processed Parquets across a season range and season type,
        concatenated into a single DataFrame.
    pivot_to_game_level
        Rotate the long format into one row per game with home and away
        features both present.
    aggregate_to_team_season
        Collapse the long format to one row per (team, season) with
        mean factors, win counts, and mean margin.

Private modules use the underscore prefix. The public API expands as
later steps in Session 11 / Session 12 land.
"""

from ._loading import read_processed as load_processed
from .pivots import aggregate_to_team_season, pivot_to_game_level

__all__ = ["aggregate_to_team_season", "load_processed", "pivot_to_game_level"]
