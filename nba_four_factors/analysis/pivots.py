"""Pivot processed long-format data into game-level and team-season shapes.

Two pure functions consumed by the regression layers:

* :func:`pivot_to_game_level` rotates the per-(team, game) long format
  into one row per game with home and away features both present. Used
  by the differentials regression and home-court-advantage analysis.
* :func:`aggregate_to_team_season` collapses the long format to one row
  per (team, season) with mean factors, win counts, and mean margin.
  Used by the team-season regression that probes the offense vs defense
  decomposition.

Layering:

    data/processed/{season}/{season_type_snake}.parquet
        |
        v
    load_processed(...) -> long DataFrame (39 cols, 2 rows per game)
        |
        +--> pivot_to_game_level(df)        -> 1 row per game (wide)
        +--> aggregate_to_team_season(df)   -> 1 row per (team, season)

Both functions are pure: DataFrame in, DataFrame out, no I/O.

Column-name convention: the processed-layer suffixes ``_pct`` and
``_rate`` are dropped on the analysis-layer outputs (e.g. ``off_efg_pct``
becomes ``home_efg`` or ``off_efg``). After standardization (Session 11
§3.7) the units no longer carry useful information, and the shorter
names match the locked spec in Session 11 §6.2 and Session 12 §1.2.

Defensive defaults: empty input returns an empty DataFrame with the
canonical schema (not None). Mixing season types in
:func:`aggregate_to_team_season` raises ValueError loudly.
"""

from __future__ import annotations

import pandas as pd

_GAME_LEVEL_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_date",
    "season",
    "season_type",
    "is_neutral",
    "home_team_abbr",
    "away_team_abbr",
    "home_efg",
    "home_tov",
    "home_orb",
    "home_ftr",
    "away_efg",
    "away_tov",
    "away_orb",
    "away_ftr",
    "home_margin",
    "home_win",
)


_TEAM_SEASON_COLUMNS: tuple[str, ...] = (
    "season",
    "season_type",
    "team_abbr",
    "games",
    "wins",
    "win_pct",
    "mean_margin",
    "off_efg",
    "off_tov",
    "off_orb",
    "off_ftr",
    "def_efg",
    "def_tov",
    "def_orb",
    "def_ftr",
)


def pivot_to_game_level(
    team_game_df: pd.DataFrame,
    include_neutral: bool = False,
) -> pd.DataFrame:
    """One row per game, home and away features both present.

    Parameters
    ----------
    team_game_df
        Long-format processed-layer DataFrame, two rows per game (one
        per team). Must contain the canonical processed schema.
    include_neutral
        If False (default), neutral-site games are excluded. "Home
        advantage" is not defined for them. If True, neutral games are
        passed through with home/away assigned by alphabetical
        ``team_abbr``; ``is_neutral=True`` flags the synthetic
        assignment so downstream code can drop or special-case.

    Returns
    -------
    pd.DataFrame
        17 columns, one row per game. See :data:`_GAME_LEVEL_COLUMNS`
        for the exact order. ``home_margin = home_pts - away_pts``;
        ``home_win = (home_margin > 0).astype(int)``.

    Notes
    -----
    Defensive columns are intentionally omitted. Within a single game,
    the home team's ``def_efg_pct`` equals the away team's
    ``off_efg_pct`` by construction (each team's defensive factor is
    the opponent's offensive factor on that same game). Carrying the
    def_ columns would duplicate information. The regression layers
    that need a defensive view derive it from the away_ columns.

    Empty input returns an empty DataFrame with the canonical schema,
    not None.
    """
    if team_game_df.empty:
        return pd.DataFrame(columns=list(_GAME_LEVEL_COLUMNS))

    df = team_game_df
    if not include_neutral:
        df = df[~df["is_neutral"]]

    if df.empty:
        return pd.DataFrame(columns=list(_GAME_LEVEL_COLUMNS))

    df = df.copy()

    if include_neutral:
        neutral_mask = df["is_neutral"]
        if neutral_mask.any():
            sub = df.loc[neutral_mask].sort_values(["game_id", "team_abbr"])
            synthetic_home = sub.groupby("game_id").cumcount() == 0
            df.loc[sub.index, "is_home"] = synthetic_home.to_numpy()

    home = df[df["is_home"]]
    away = df[~df["is_home"]]

    home_view = home.rename(
        columns={
            "team_abbr": "home_team_abbr",
            "off_efg_pct": "home_efg",
            "off_tov_pct": "home_tov",
            "off_orb_pct": "home_orb",
            "off_ft_rate": "home_ftr",
            "pts": "_home_pts",
            "opp_abbr": "_home_opp_abbr",
        }
    )[
        [
            "game_id",
            "game_date",
            "season",
            "season_type",
            "is_neutral",
            "home_team_abbr",
            "home_efg",
            "home_tov",
            "home_orb",
            "home_ftr",
            "_home_pts",
            "_home_opp_abbr",
        ]
    ]

    away_view = away.rename(
        columns={
            "team_abbr": "away_team_abbr",
            "off_efg_pct": "away_efg",
            "off_tov_pct": "away_tov",
            "off_orb_pct": "away_orb",
            "off_ft_rate": "away_ftr",
            "pts": "_away_pts",
        }
    )[
        [
            "game_id",
            "away_team_abbr",
            "away_efg",
            "away_tov",
            "away_orb",
            "away_ftr",
            "_away_pts",
        ]
    ]

    merged = home_view.merge(
        away_view,
        left_on=["game_id", "_home_opp_abbr"],
        right_on=["game_id", "away_team_abbr"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(home_view):
        dropped = len(home_view) - len(merged)
        raise ValueError(
            f"pivot_to_game_level dropped {dropped} home rows during merge; "
            f"upstream data has inconsistent (game_id, opp_abbr) pairs"
        )

    merged["home_margin"] = merged["_home_pts"] - merged["_away_pts"]
    merged["home_win"] = (merged["home_margin"] > 0).astype(int)

    merged = merged.drop(columns=["_home_pts", "_away_pts", "_home_opp_abbr"])

    return merged[list(_GAME_LEVEL_COLUMNS)].reset_index(drop=True)


def aggregate_to_team_season(
    team_game_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (team, season) with mean factors plus wins and games.

    Parameters
    ----------
    team_game_df
        Long-format processed-layer DataFrame. Must contain only one
        ``season_type`` value; mixing regular season and playoffs in the
        same call is analytically wrong (different game dynamics, very
        different sample sizes per team-season). Filter the input first.

    Returns
    -------
    pd.DataFrame
        15 columns, one row per (team, season). See
        :data:`_TEAM_SEASON_COLUMNS` for the order. Factor averages are
        unweighted by minutes or possessions; each game contributes
        equally. Possession-weighted averages are an alternative worth
        exploring (Session 12 §2.15) but are not the default.

    Raises
    ------
    ValueError
        If the input contains more than one ``season_type`` value.

    Notes
    -----
    Franchise relocations within the data range produce separate rows
    per ``team_abbr`` (e.g. SEA in 2007_08 and OKC in 2008_09 each get
    their own row). This is correct: the franchise plays under different
    abbreviations in different seasons.
    """
    if team_game_df.empty:
        return pd.DataFrame(columns=list(_TEAM_SEASON_COLUMNS))

    season_types = team_game_df["season_type"].unique()
    if len(season_types) > 1:
        raise ValueError(
            f"aggregate_to_team_season requires a single season_type per call; "
            f"got {sorted(season_types.tolist())}. Filter the input by "
            f"season_type before calling."
        )

    df = team_game_df.copy()
    df["_win"] = (df["margin"] > 0).astype(int)

    grouped = df.groupby(["season", "team_abbr"], as_index=False).agg(
        season_type=("season_type", "first"),
        games=("game_id", "count"),
        wins=("_win", "sum"),
        mean_margin=("margin", "mean"),
        off_efg=("off_efg_pct", "mean"),
        off_tov=("off_tov_pct", "mean"),
        off_orb=("off_orb_pct", "mean"),
        off_ftr=("off_ft_rate", "mean"),
        def_efg=("def_efg_pct", "mean"),
        def_tov=("def_tov_pct", "mean"),
        def_orb=("def_orb_pct", "mean"),
        def_ftr=("def_ft_rate", "mean"),
    )
    grouped["win_pct"] = grouped["wins"] / grouped["games"]

    return grouped[list(_TEAM_SEASON_COLUMNS)].reset_index(drop=True)
