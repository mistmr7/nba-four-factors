"""Compute Dean Oliver's four factors and margin on a self-joined DataFrame.

Pure computation: no I/O, no globals, no surprises.  Input must already
have both team-side and opponent-side box columns (from the self-join in
:mod:`pipeline`).  Output adds nine columns and returns a new DataFrame.

Formulas (Session 10 §2.4, locked):

.. code-block:: text

    eFG%      = (FGM + 0.5 * FG3M) / FGA
    TOV%      = TOV / (FGA + 0.44 * FTA + TOV)
    ORB%      = OREB / (OREB + opp_DREB)
    FT Rate   = FTA / FGA

Defensive factors are the opponent's offensive factors *against this team*::

    def_eFG%    = (opp_FGM + 0.5 * opp_FG3M) / opp_FGA
    def_TOV%    = opp_TOV / (opp_FGA + 0.44 * opp_FTA + opp_TOV)
    def_ORB%    = opp_OREB / (opp_OREB + DREB)        # this team's DREB
    def_FT_Rate = opp_FTA / opp_FGA

Note the asymmetry in def_ORB%: the denominator is opp_OREB + *this team's*
DREB, because that's the rebound battle from the opponent's offensive end
(their offensive rebounds vs. our defensive rebounds).  Easy to flip
accidentally during refactors — comment in the code, test in the fixture.

Division-by-zero protection: any zero denominator yields NaN (not an
exception).  Implementation uses ``denominator.where(denominator != 0)``
which replaces zeros with NaN before the division — clean output, no
RuntimeWarnings.  A real game cannot have FGA=0, but we don't want
validation runs on edge-case test data to crash.
"""

from __future__ import annotations

import pandas as pd

# Columns this module reads from input.  Listed for documentation and for
# the "schema check" at the top of add_factors — fail loudly on missing
# inputs rather than producing a column of all-NaN.
_REQUIRED_TEAM_COLUMNS: list[str] = [
    "fgm",
    "fga",
    "fg3m",
    "fta",
    "oreb",
    "dreb",
    "tov",
    "pts",
]
_REQUIRED_OPP_COLUMNS: list[str] = [
    "opp_fgm",
    "opp_fga",
    "opp_fg3m",
    "opp_fta",
    "opp_oreb",
    "opp_dreb",
    "opp_tov",
    "opp_pts",
]


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add the eight four-factor columns plus margin.

    Parameters
    ----------
    df
        Per-(team, game) DataFrame already augmented with opponent columns
        via the pipeline self-join.  Must contain the columns listed in
        :data:`_REQUIRED_TEAM_COLUMNS` and :data:`_REQUIRED_OPP_COLUMNS`.

    Returns
    -------
    A new DataFrame (not a view) with nine added columns:
    ``off_efg_pct off_tov_pct off_orb_pct off_ft_rate
    def_efg_pct def_tov_pct def_orb_pct def_ft_rate margin``.

    Raises
    ------
    ValueError
        If any required input column is missing.  Loud failure beats
        producing a column of NaN that looks valid downstream.
    """
    missing = [c for c in (_REQUIRED_TEAM_COLUMNS + _REQUIRED_OPP_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_factors input missing required columns: {missing}. " f"Got: {sorted(df.columns)}"
        )

    # Don't mutate the caller's DataFrame.
    df = df.copy()

    # ----------------------------------------------------------------- #
    # Team's offensive four factors
    # ----------------------------------------------------------------- #
    df["off_efg_pct"] = (df["fgm"] + 0.5 * df["fg3m"]) / df["fga"].where(df["fga"] != 0)

    # TOV%: possessions consumed by turnovers.  Standard Oliver formula.
    tov_denom = df["fga"] + 0.44 * df["fta"] + df["tov"]
    df["off_tov_pct"] = df["tov"] / tov_denom.where(tov_denom != 0)

    # ORB%: this team's offensive rebounds divided by the rebounds available
    # at this team's offensive end (own OREB + opponent's DREB).
    orb_denom = df["oreb"] + df["opp_dreb"]
    df["off_orb_pct"] = df["oreb"] / orb_denom.where(orb_denom != 0)

    # FT Rate uses FTA, not FTM (Oliver: how often you get to the line, not
    # how often you convert).  Conversion rate is captured separately by FT%.
    df["off_ft_rate"] = df["fta"] / df["fga"].where(df["fga"] != 0)

    # ----------------------------------------------------------------- #
    # Defensive four factors = opponent's offensive factors against us
    # ----------------------------------------------------------------- #
    df["def_efg_pct"] = (df["opp_fgm"] + 0.5 * df["opp_fg3m"]) / df["opp_fga"].where(
        df["opp_fga"] != 0
    )

    opp_tov_denom = df["opp_fga"] + 0.44 * df["opp_fta"] + df["opp_tov"]
    df["def_tov_pct"] = df["opp_tov"] / opp_tov_denom.where(opp_tov_denom != 0)

    # ASYMMETRY: opponent's offensive rebounds, divided by THIS team's DREB
    # plus their OREB (the rebound battle at *their* offensive end).  Don't
    # accidentally use opp_dreb here on refactor.
    opp_orb_denom = df["opp_oreb"] + df["dreb"]
    df["def_orb_pct"] = df["opp_oreb"] / opp_orb_denom.where(opp_orb_denom != 0)

    df["def_ft_rate"] = df["opp_fta"] / df["opp_fga"].where(df["opp_fga"] != 0)

    # ----------------------------------------------------------------- #
    # Margin (kept as int because pts are ints; preserved through Parquet)
    # ----------------------------------------------------------------- #
    df["margin"] = df["pts"] - df["opp_pts"]

    return df
