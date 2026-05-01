"""Read processed four-factors Parquets across a season range.

The only function in the analysis layer that touches disk for input.
Every other analysis function takes a DataFrame.

Layering:

    data/processed/{season}/{season_type_snake}.parquet
        |
        v
    read_processed(season_range, season_type) -> DataFrame

Missing seasons are silently skipped (logged at INFO). If no files match
the range at all, an empty DataFrame with the canonical 39-column schema
is returned (not None) so downstream code can pivot or aggregate without
special-casing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nba_four_factors.config import PROCESSED_DIR, SEASONS, SeasonType
from nba_four_factors.processed._naming import season_type_to_snake

logger = logging.getLogger(__name__)


_PROCESSED_SCHEMA: tuple[str, ...] = (
    "game_id",
    "game_date",
    "season",
    "season_type",
    "team_id",
    "team_abbr",
    "opp_team_id",
    "opp_abbr",
    "is_home",
    "is_neutral",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "tov",
    "pts",
    "opp_fgm",
    "opp_fga",
    "opp_fg3m",
    "opp_fg3a",
    "opp_ftm",
    "opp_fta",
    "opp_oreb",
    "opp_dreb",
    "opp_tov",
    "opp_pts",
    "off_efg_pct",
    "off_tov_pct",
    "off_orb_pct",
    "off_ft_rate",
    "def_efg_pct",
    "def_tov_pct",
    "def_orb_pct",
    "def_ft_rate",
    "margin",
)


def _expand_season_range(season_range: tuple[str, str]) -> list[str]:
    """Inclusive expansion of a (start, end) range against the canonical SEASONS tuple.

    Both endpoints must be present in SEASONS; both are included in the result.
    Raises ValueError on unknown season strings or inverted ranges.
    """
    start, end = season_range

    if start not in SEASONS:
        raise ValueError(
            f"Unknown start season {start!r}; not in SEASONS "
            f"(first={SEASONS[0]}, last={SEASONS[-1]})"
        )
    if end not in SEASONS:
        raise ValueError(
            f"Unknown end season {end!r}; not in SEASONS "
            f"(first={SEASONS[0]}, last={SEASONS[-1]})"
        )

    start_idx = SEASONS.index(start)
    end_idx = SEASONS.index(end)

    if start_idx > end_idx:
        raise ValueError(f"Inverted season range: start {start!r} comes after end {end!r}")

    return list(SEASONS[start_idx : end_idx + 1])


def read_processed(
    season_range: tuple[str, str],
    season_type: SeasonType,
) -> pd.DataFrame:
    """Concatenate per-season processed Parquets in the inclusive range.

    Parameters
    ----------
    season_range
        Inclusive (start, end) tuple in canonical underscore form, e.g.
        ``("2020_21", "2024_25")``. Both endpoints must exist in
        :data:`nba_four_factors.config.SEASONS`.
    season_type
        Which season-type slice to load. One file per season per type.

    Returns
    -------
    pd.DataFrame
        Long-format team-game rows concatenated across all matched
        seasons, with ``ignore_index=True``. Schema matches the 39-column
        processed layer. Empty DataFrame with the same column list if no
        files were found.
    """
    seasons = _expand_season_range(season_range)
    snake = season_type_to_snake(season_type)

    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for season in seasons:
        path: Path = PROCESSED_DIR / season / f"{snake}.parquet"
        if not path.exists():
            missing.append(season)
            continue
        frames.append(pd.read_parquet(path))

    if missing:
        logger.info(
            "read_processed: %d/%d seasons missing for season_type=%s: %s",
            len(missing),
            len(seasons),
            season_type.name,
            ", ".join(missing),
        )

    if not frames:
        logger.info(
            "read_processed: no files matched range=%s season_type=%s; returning empty schema",
            season_range,
            season_type.name,
        )
        return pd.DataFrame(columns=list(_PROCESSED_SCHEMA))

    return pd.concat(frames, ignore_index=True)
