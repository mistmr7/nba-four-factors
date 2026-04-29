"""End-to-end processed-layer pipeline for one (season, season_type) pair.

The one place that knows the full data flow shape:

.. code-block:: text

    raw JSON  →  tidy team-game rows  →  + opponent columns  →  + factors
                                                                    ↓
                                                          data/processed/...

Path construction is inline rather than in a ``storage/processed.py``
sibling — the processed layer has exactly one path shape so far, and the
KISS check in Session 10 §3.1 said don't add a module just for symmetry.
If a second processed-layer artifact appears (e.g., the deferred combined
master Parquet from §2.5), promote :func:`processed_path` then.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import SeasonType
from ._naming import season_type_to_snake
from .factors import add_factors
from .schedule import read_schedule

# --------------------------------------------------------------------------- #
# Path construction
# --------------------------------------------------------------------------- #
#
# Hardcoded relative root for now.  If the storage layer later exposes a
# DATA_ROOT or processed-base path (mirroring whatever raw_season_path uses
# internally), swap that in here.

_PROCESSED_ROOT: Path = Path("data/processed")


def processed_path(season: str, season_type: SeasonType) -> Path:
    """Return the Parquet path for a (season, season_type) pair.

    >>> processed_path("2024_25", SeasonType.REGULAR)
    PosixPath('data/processed/2024_25/regular_season.parquet')

    Filename uses the snake form (``regular_season``) per Session 10 §2.2.
    Underscore convention preserved for the season directory name.
    """
    return _PROCESSED_ROOT / season / f"{season_type_to_snake(season_type)}.parquet"


# --------------------------------------------------------------------------- #
# Self-join: team-side rows → team + opponent rows
# --------------------------------------------------------------------------- #
#
# The tidy schedule has one row per (game, team) with team-only stats.  To
# compute factors we need the opponent's stats on the same row.  A self-join
# on game_id where team_id != team_id pairs each row with the other team's
# row for the same game.

# Columns from the team side that get an ``opp_*`` mirror.  Listed
# explicitly so adding a new column to schedule.py doesn't silently change
# the join surface here.
_OPP_COLUMNS: list[str] = [
    "team_id",
    "team_abbr",
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
]

# Final column order (Session 10 §2.3, locked).
_FINAL_COLUMN_ORDER: list[str] = [
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
]


def _attach_opponent(team_df: pd.DataFrame) -> pd.DataFrame:
    """Self-join team-side rows so each row has both team and opponent stats.

    Implementation: build an opponent-keyed view of the team-side columns,
    inner-merge on game_id, then drop the rows where team_id == opp_team_id
    (those are a row joined with itself).  Result is the same row count as
    input — every team-game row gets its opponent's stats attached.

    Spec §2.3 names two of the opp columns specially: ``opp_team_id`` (full
    prefix) and ``opp_abbr`` (drops "team").  Handled in the rename below.
    """
    # 1. Project just the columns we want to attach as opp_*.
    opp_view = team_df[["game_id", *_OPP_COLUMNS]].copy()
    rename_map = {c: f"opp_{c}" for c in _OPP_COLUMNS}
    # Spec wants ``opp_abbr``, not ``opp_team_abbr``.
    rename_map["team_abbr"] = "opp_abbr"
    opp_view = opp_view.rename(columns=rename_map)

    # 2. Inner-merge on game_id.  Each team-game row matches BOTH rows for
    #    that game (its own and the opponent's), so this produces 2x rows.
    merged = team_df.merge(opp_view, on="game_id", how="inner")

    # 3. Drop the self-pairs.  What remains: each team-game row paired with
    #    exactly the other team's row.
    merged = merged[merged["team_id"] != merged["opp_team_id"]]

    # 4. Validate row count: should equal the input row count.  If not,
    #    something upstream is wrong (duplicate game/team rows, or game
    #    appears 3+ times).  Loud failure.
    if len(merged) != len(team_df):
        raise AssertionError(
            f"Self-join produced {len(merged)} rows; expected "
            f"{len(team_df)}.  Schedule data corruption likely."
        )

    return merged


# --------------------------------------------------------------------------- #
# Public: full pipeline
# --------------------------------------------------------------------------- #


def process_pair(season: str, season_type: SeasonType) -> Path:
    """Build the processed Parquet for one (season, season_type) pair.

    Steps (Session 10 §4):

    1. Load the saved LeagueGameLog JSON via :func:`read_schedule`.
    2. Self-join to attach opponent box stats (:func:`_attach_opponent`).
    3. Compute four factors and margin (:func:`add_factors`).
    4. Reorder columns and write Parquet.

    Idempotent: running twice produces byte-identical output (the sort in
    :func:`_tidy_schedule` ensures deterministic row order).

    Returns
    -------
    Path of the written Parquet file.

    Raises
    ------
    FileNotFoundError
        If the raw schedule JSON has not been fetched.  Caller (CLI) is
        responsible for translating this to a user-facing error pointing
        at ``backfill``.
    """
    # 1. Tidy team-side rows (~2460 for a full regular season).
    team_df = read_schedule(season, season_type)

    # 2. Attach opponent columns.
    joined = _attach_opponent(team_df)

    # 3. Compute factors + margin.
    enriched = add_factors(joined)

    # 4. Reorder to spec, sort deterministically, write.
    out = (
        enriched[_FINAL_COLUMN_ORDER]
        .sort_values(["game_date", "game_id", "team_id"])
        .reset_index(drop=True)
    )

    path = processed_path(season, season_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, engine="pyarrow", index=False)
    return path
