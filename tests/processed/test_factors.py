"""Tests for ``nba_four_factors.processed.factors``.

The hero test (:func:`test_add_factors_hand_computed`) asserts every factor
against numbers computed from the formulas, not from the implementation.
If the implementation drifts from Oliver's definitions, this test fails;
that's the whole point.

Edge cases cover the division-by-zero behaviour required by Session 10
§2.4 ("any denominator of zero yields NaN, not an error").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_four_factors.processed.factors import add_factors

# --------------------------------------------------------------------------- #
# Hand-computed expected values (pinned, see Session 10 §7.1)
# --------------------------------------------------------------------------- #
#
# Team line:  FGM=40 FGA=85 FG3M=10 FTA=20 OREB=12 DREB=35 TOV=14 PTS=110
# Opp line:   FGM=38 FGA=88 FG3M=8  FTA=18 OREB=10 DREB=33 TOV=16 PTS=100
#
# Computed in Python (not in the implementation) — see hand-calc step at
# the top of Session 10's build log.

_EXPECTED_TEAM = {
    "off_efg_pct": 0.5294117647058824,  # (40 + 0.5*10) / 85
    "off_tov_pct": 0.12987012987012989,  # 14 / (85 + 0.44*20 + 14)
    "off_orb_pct": 0.26666666666666666,  # 12 / (12 + 33)
    "off_ft_rate": 0.23529411764705882,  # 20 / 85
    "def_efg_pct": 0.4772727272727273,  # (38 + 0.5*8) / 88
    "def_tov_pct": 0.14295925661186562,  # 16 / (88 + 0.44*18 + 16)
    "def_orb_pct": 0.2222222222222222,  # 10 / (10 + 35)
    "def_ft_rate": 0.20454545454545456,  # 18 / 88
    "margin": 10,  # 110 - 100
}


def _make_two_team_df() -> pd.DataFrame:
    """Build the post-self-join DataFrame for the §7.1 hand-computed game.

    Two rows: LAL's row paired with BOS as opponent, BOS's row paired with
    LAL as opponent.  ``add_factors`` works on this shape directly.
    """
    return pd.DataFrame(
        [
            # LAL (team) vs BOS (opp)
            {
                "team_id": 1,
                "team_abbr": "LAL",
                "fgm": 40,
                "fga": 85,
                "fg3m": 10,
                "fg3a": 30,
                "ftm": 15,
                "fta": 20,
                "oreb": 12,
                "dreb": 35,
                "tov": 14,
                "pts": 110,
                "opp_team_id": 2,
                "opp_abbr": "BOS",
                "opp_fgm": 38,
                "opp_fga": 88,
                "opp_fg3m": 8,
                "opp_fg3a": 28,
                "opp_ftm": 14,
                "opp_fta": 18,
                "opp_oreb": 10,
                "opp_dreb": 33,
                "opp_tov": 16,
                "opp_pts": 100,
            },
            # BOS (team) vs LAL (opp) — symmetric row
            {
                "team_id": 2,
                "team_abbr": "BOS",
                "fgm": 38,
                "fga": 88,
                "fg3m": 8,
                "fg3a": 28,
                "ftm": 14,
                "fta": 18,
                "oreb": 10,
                "dreb": 33,
                "tov": 16,
                "pts": 100,
                "opp_team_id": 1,
                "opp_abbr": "LAL",
                "opp_fgm": 40,
                "opp_fga": 85,
                "opp_fg3m": 10,
                "opp_fg3a": 30,
                "opp_ftm": 15,
                "opp_fta": 20,
                "opp_oreb": 12,
                "opp_dreb": 35,
                "opp_tov": 14,
                "opp_pts": 110,
            },
        ]
    )


# --------------------------------------------------------------------------- #
# Hero test: hand-computed factor values
# --------------------------------------------------------------------------- #


def test_add_factors_hand_computed():
    """Every factor matches values computed from the formulas, not the code."""
    df = add_factors(_make_two_team_df())
    lal = df.iloc[0]

    for col, expected in _EXPECTED_TEAM.items():
        assert lal[col] == pytest.approx(
            expected
        ), f"{col}: got {lal[col]!r}, expected {expected!r}"


def test_add_factors_symmetry_between_teams():
    """LAL's def_* equal BOS's off_*, and vice versa.  Sanity check that the
    self-join + factor logic doesn't accidentally swap a team and its
    opponent inside the formulas."""
    df = add_factors(_make_two_team_df())
    lal, bos = df.iloc[0], df.iloc[1]

    assert lal["off_efg_pct"] == pytest.approx(bos["def_efg_pct"])
    assert lal["off_tov_pct"] == pytest.approx(bos["def_tov_pct"])
    assert lal["off_orb_pct"] == pytest.approx(bos["def_orb_pct"])
    assert lal["off_ft_rate"] == pytest.approx(bos["def_ft_rate"])

    assert lal["def_efg_pct"] == pytest.approx(bos["off_efg_pct"])
    assert lal["def_tov_pct"] == pytest.approx(bos["off_tov_pct"])
    assert lal["def_orb_pct"] == pytest.approx(bos["off_orb_pct"])
    assert lal["def_ft_rate"] == pytest.approx(bos["off_ft_rate"])

    assert lal["margin"] == -bos["margin"]


def test_add_factors_def_orb_uses_team_dreb_not_opp_dreb():
    """Guard the asymmetry called out in factors.py.

    def_orb_pct should be opp_oreb / (opp_oreb + dreb), NOT
    opp_oreb / (opp_oreb + opp_dreb).  If a refactor flips the denominator
    to use opp_dreb, this test catches it.

    Construction: build a row where the two values would give different
    answers, then check we get the right one.
    """
    df = pd.DataFrame(
        [
            {
                "team_id": 1,
                "team_abbr": "X",
                "fgm": 40,
                "fga": 85,
                "fg3m": 10,
                "fg3a": 30,
                "ftm": 15,
                "fta": 20,
                "oreb": 12,
                "dreb": 50,
                "tov": 14,
                "pts": 110,  # dreb=50
                "opp_team_id": 2,
                "opp_abbr": "Y",
                "opp_fgm": 38,
                "opp_fga": 88,
                "opp_fg3m": 8,
                "opp_fg3a": 28,
                "opp_ftm": 14,
                "opp_fta": 18,
                "opp_oreb": 10,
                "opp_dreb": 5,
                "opp_tov": 16,
                "opp_pts": 100,  # opp_dreb=5
            }
        ]
    )
    out = add_factors(df).iloc[0]

    # Right answer:  opp_oreb / (opp_oreb + dreb)      = 10 / (10 + 50) = 0.16666...
    # Wrong answer: opp_oreb / (opp_oreb + opp_dreb)  = 10 / (10 + 5)  = 0.66666...
    assert out["def_orb_pct"] == pytest.approx(10 / 60)
    assert out["def_orb_pct"] != pytest.approx(10 / 15)


# --------------------------------------------------------------------------- #
# Edge cases: division by zero
# --------------------------------------------------------------------------- #


def test_add_factors_all_zeros_yields_nan_no_crash():
    """A row of all zeros produces NaN factors, not a RuntimeError or warning."""
    df = pd.DataFrame(
        [
            {
                "team_id": 1,
                "team_abbr": "X",
                "fgm": 0,
                "fga": 0,
                "fg3m": 0,
                "fg3a": 0,
                "ftm": 0,
                "fta": 0,
                "oreb": 0,
                "dreb": 0,
                "tov": 0,
                "pts": 0,
                "opp_team_id": 2,
                "opp_abbr": "Y",
                "opp_fgm": 0,
                "opp_fga": 0,
                "opp_fg3m": 0,
                "opp_fg3a": 0,
                "opp_ftm": 0,
                "opp_fta": 0,
                "opp_oreb": 0,
                "opp_dreb": 0,
                "opp_tov": 0,
                "opp_pts": 0,
            }
        ]
    )
    out = add_factors(df).iloc[0]

    for col in [
        "off_efg_pct",
        "off_tov_pct",
        "off_orb_pct",
        "off_ft_rate",
        "def_efg_pct",
        "def_tov_pct",
        "def_orb_pct",
        "def_ft_rate",
    ]:
        assert np.isnan(out[col]), f"{col} should be NaN for all-zero input"
    assert out["margin"] == 0


def test_add_factors_one_team_only_populates_partial():
    """If only the opponent has stats, the team's own offensive factors are
    NaN where the denominator collapses to zero, but defensive factors
    (which use opp data) populate normally.

    Specifically:
      - off_efg_pct: 0/0 = NaN (fga=0)
      - off_tov_pct: 0/0 = NaN (fga + 0.44*fta + tov = 0)
      - off_orb_pct: 0 / (0 + opp_dreb) = 0 (NOT NaN — denom is non-zero)
      - off_ft_rate: 0/0 = NaN (fga=0)
      - def_*: all populate normally from opp stats
    """
    df = pd.DataFrame(
        [
            {
                "team_id": 1,
                "team_abbr": "X",
                "fgm": 0,
                "fga": 0,
                "fg3m": 0,
                "fg3a": 0,
                "ftm": 0,
                "fta": 0,
                "oreb": 0,
                "dreb": 0,
                "tov": 0,
                "pts": 0,
                "opp_team_id": 2,
                "opp_abbr": "Y",
                "opp_fgm": 38,
                "opp_fga": 88,
                "opp_fg3m": 8,
                "opp_fg3a": 28,
                "opp_ftm": 14,
                "opp_fta": 18,
                "opp_oreb": 10,
                "opp_dreb": 33,
                "opp_tov": 16,
                "opp_pts": 100,
            }
        ]
    )
    out = add_factors(df).iloc[0]

    # Team's own offence — collapses on team-only denominators
    assert np.isnan(out["off_efg_pct"])
    assert np.isnan(out["off_tov_pct"])
    assert out["off_orb_pct"] == 0.0  # 0 / (0 + 33), valid
    assert np.isnan(out["off_ft_rate"])

    # Defensive side computes from opp's actual stats
    assert out["def_efg_pct"] == pytest.approx(42 / 88)
    assert out["def_tov_pct"] == pytest.approx(16 / (88 + 0.44 * 18 + 16))
    assert out["def_orb_pct"] == pytest.approx(10 / 10)  # opp_dreb=0 here
    assert out["def_ft_rate"] == pytest.approx(18 / 88)

    # Margin still computes
    assert out["margin"] == -100


# --------------------------------------------------------------------------- #
# Schema and contract
# --------------------------------------------------------------------------- #


def test_add_factors_does_not_mutate_input():
    """add_factors returns a new DataFrame; input is unchanged."""
    df_in = _make_two_team_df()
    df_in_copy = df_in.copy()
    _ = add_factors(df_in)
    pd.testing.assert_frame_equal(df_in, df_in_copy)


def test_add_factors_missing_required_column_raises():
    """Loud failure on missing required columns, not silent NaN."""
    df = _make_two_team_df().drop(columns=["opp_oreb"])
    with pytest.raises(ValueError, match="missing required columns"):
        add_factors(df)


def test_add_factors_adds_exactly_nine_columns():
    """Schema invariant: 8 factor columns + margin."""
    df_in = _make_two_team_df()
    df_out = add_factors(df_in)
    new_cols = set(df_out.columns) - set(df_in.columns)
    assert new_cols == {
        "off_efg_pct",
        "off_tov_pct",
        "off_orb_pct",
        "off_ft_rate",
        "def_efg_pct",
        "def_tov_pct",
        "def_orb_pct",
        "def_ft_rate",
        "margin",
    }
