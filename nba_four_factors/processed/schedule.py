"""Tidy a saved LeagueGameLog JSON envelope into a per-(team, game) DataFrame.

Two functions:

* :func:`_tidy_schedule` is pure — takes the parsed JSON dict and returns a
  DataFrame.  Tests import it directly with hand-crafted fixtures, no disk.
* :func:`read_schedule`  is the I/O wrapper — composes
  :func:`load_raw` + :func:`raw_season_path` + :func:`_tidy_schedule`.

The split mirrors the existing ``orchestration/_io.py`` pattern: storage
primitives live in ``storage.raw``; this module composes them for the
processed layer.

Output schema (locked, Session 10 §2.3):

* Identity:  ``game_id``, ``game_date``, ``season``, ``season_type``,
             ``team_id``, ``team_abbr``
* Location:  ``is_home``, ``is_neutral``  (two booleans, see Session 10
             handoff for the neutral-site decision)
* Box stats: ``fgm fga fg3m fg3a ftm fta oreb dreb tov pts``

Opponent columns (``opp_*``) and the eight four-factor columns are added
later in the pipeline (self-join in :mod:`pipeline`, factor math in
:mod:`factors`).  This module deliberately stops at the team-side row.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import Endpoint, SeasonType
from ..storage.raw import load_raw, raw_season_path
from ._naming import season_type_to_snake

# --------------------------------------------------------------------------- #
# Column mapping: LeagueGameLog (uppercase) → tidy schema (snake_case)
# --------------------------------------------------------------------------- #
#
# Listed explicitly rather than auto-lowercasing.  Two reasons:
#
# 1. ``TEAM_ABBREVIATION`` becomes ``team_abbr`` (the spec name), not
#    ``team_abbreviation``.  A blanket ``.lower()`` would diverge from §2.3.
# 2. We want to drop columns that LeagueGameLog provides but we don't need
#    (TEAM_NAME, MIN, FG_PCT, FG3_PCT, FT_PCT, REB, AST, STL, BLK, PF,
#    PLUS_MINUS, VIDEO_AVAILABLE, SEASON_ID, WL).  An explicit map is the
#    natural place to make that decision visible.
#
# If LeagueGameLog ever adds or renames a column we depend on, this is the
# one place that needs updating — the rest of the pipeline runs on the tidy
# schema.

_RAW_TO_TIDY: dict[str, str] = {
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team_abbr",
    "MATCHUP": "_matchup",  # parsed into is_home/is_neutral, then dropped
    "FGM": "fgm",
    "FGA": "fga",
    "FG3M": "fg3m",
    "FG3A": "fg3a",
    "FTM": "ftm",
    "FTA": "fta",
    "OREB": "oreb",
    "DREB": "dreb",
    "TOV": "tov",
    "PTS": "pts",
}

# Final column order out of _tidy_schedule (before the pipeline self-join).
_TIDY_COLUMN_ORDER: list[str] = [
    "game_id",
    "game_date",
    "season",
    "season_type",
    "team_id",
    "team_abbr",
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
]


# --------------------------------------------------------------------------- #
# Internal: pure transform
# --------------------------------------------------------------------------- #


def _tidy_schedule(
    payload: dict[str, Any],
    *,
    season: str,
    season_type: SeasonType,
) -> pd.DataFrame:
    """Transform a parsed LeagueGameLog JSON dict into a tidy DataFrame.

    Pure function: no disk, no network, no globals.  All inputs are
    arguments; all outputs are the return value.  Tests feed a hand-crafted
    dict; production code feeds the result of :func:`load_raw`.

    Parameters
    ----------
    payload
        Parsed JSON with the standard nba_api stats envelope::

            {"resultSets": [{"name": ..., "headers": [...], "rowSet": [...]}]}
    season
        Underscore-form season (e.g. ``"2024_25"``).  Stored on every row.
    season_type
        Enum member; converted to snake form for the column value.

    Returns
    -------
    DataFrame with :data:`_TIDY_COLUMN_ORDER` columns, one row per
    (team, game).  For a full regular season this is 2460 rows.

    Raises
    ------
    ValueError
        If the envelope shape is wrong, an expected header is missing, any
        game does not appear exactly twice, or any MATCHUP has an
        unrecognised separator pattern.
    """
    # 1. Unwrap the envelope.  Defensive against ``resultSet`` (singular)
    #    in case a future endpoint version changes the key — same pattern
    #    as ``orchestration/_io.load_schedule_game_ids``.
    result_sets = payload.get("resultSets") or payload.get("resultSet")
    if not result_sets:
        raise ValueError(
            f"LeagueGameLog payload has no resultSets; " f"top-level keys: {list(payload.keys())}"
        )
    table = result_sets[0]
    headers = table["headers"]
    rows = table["rowSet"]

    # 2. Verify every column we need is present.  Loud failure on missing
    #    columns is correct here: a silent NaN-fill would propagate bad
    #    data into the analytics layer.
    missing = [c for c in _RAW_TO_TIDY if c not in headers]
    if missing:
        raise ValueError(
            f"LeagueGameLog headers missing required columns: {missing}. " f"Got headers: {headers}"
        )

    # 3. Build a DataFrame from the full rowSet, then keep only the columns
    #    we use.  Building-then-projecting (rather than projecting first) is
    #    cheaper to reason about and pandas handles it efficiently.
    df = pd.DataFrame(rows, columns=headers)
    df = df[list(_RAW_TO_TIDY.keys())].rename(columns=_RAW_TO_TIDY)

    # 4. Validate every game appears exactly twice.  This is the strongest
    #    upstream-corruption check we can run on team-level data; a missing
    #    half would silently drop the game in the self-join later.
    counts = df["game_id"].value_counts()
    bad = counts[counts != 2]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} game(s) do not appear exactly twice in the schedule. "
            f"First few offenders: {bad.head().to_dict()}"
        )

    # 5. Parse MATCHUP into is_home / is_neutral.
    #
    #    Standard game: one row has 'TEAM vs. OPP' (home), the other has
    #    'TEAM @ OPP' (away).  NBA Cup neutral-site games (~5 per season,
    #    e.g. 2024 Vegas semifinals/final) have BOTH rows as '@'.  Decision
    #    in the Session 10 handoff: model with two booleans, so neutral
    #    rows get is_home=False, is_neutral=True for both teams.
    df["_separator"] = df["_matchup"].str.split().str[1]

    # Per-game pattern of separators tells us standard vs. neutral.
    sep_sets = df.groupby("game_id")["_separator"].apply(frozenset)
    valid_standard = frozenset({"vs.", "@"})
    valid_neutral = frozenset({"@"})
    invalid = sep_sets[~sep_sets.isin([valid_standard, valid_neutral])]
    if len(invalid) > 0:
        raise ValueError(
            f"{len(invalid)} game(s) have unrecognised MATCHUP patterns. "
            f"First few offenders: {invalid.head().to_dict()}"
        )

    neutral_games = set(sep_sets[sep_sets == valid_neutral].index)
    df["is_neutral"] = df["game_id"].isin(neutral_games)
    # is_home: True only for the 'vs.' row of a standard game.  Neutral
    # rows have separator '@' so this evaluates False for them — the
    # explicit ``& ~is_neutral`` is redundant but documents intent.
    df["is_home"] = (df["_separator"] == "vs.") & ~df["is_neutral"]

    # 6. Annotate with season / season_type (snake form for storage).
    df["season"] = season
    df["season_type"] = season_type_to_snake(season_type)

    # 7. Date parsing.  pandas writes datetime64[ns] cleanly to Parquet via
    #    pyarrow; pure Python ``date`` objects round-trip less reliably.
    #    The column name remains ``game_date`` per spec.
    df["game_date"] = pd.to_datetime(df["game_date"])

    # 8. Drop scratch columns and order per schema.
    df = df.drop(columns=["_matchup", "_separator"])
    df = df[_TIDY_COLUMN_ORDER]

    # 9. Sort deterministically so two runs produce byte-identical output.
    #    (Idempotency promise from spec §4.)
    df = df.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)

    return df


# --------------------------------------------------------------------------- #
# Public: I/O wrapper
# --------------------------------------------------------------------------- #


def read_schedule(season: str, season_type: SeasonType) -> pd.DataFrame:
    """Load a saved LeagueGameLog JSON and return the tidy DataFrame.

    Composes :func:`raw_season_path` + :func:`load_raw` + :func:`_tidy_schedule`.
    This is the function pipeline.py and notebooks should call.  Tests of
    the transformation logic should call :func:`_tidy_schedule` directly
    with a fixture dict; tests of I/O composition (rare) should mock at the
    storage boundary.

    Raises
    ------
    FileNotFoundError
        If the raw JSON for the requested pair has not been fetched yet.
        Surfaced unchanged from :func:`load_raw`.
    ValueError
        Propagated from :func:`_tidy_schedule` for envelope/data issues.
    """
    path = raw_season_path(Endpoint.SCHEDULE, season, season_type)
    payload = load_raw(path)
    return _tidy_schedule(payload, season=season, season_type=season_type)
