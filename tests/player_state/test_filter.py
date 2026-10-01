"""Correctness tests for the scalar OU player filter.

Synthetic parameter recovery is the primary estimator test and uses no real
data. The remaining tests exercise the filter mechanics on small synthetic
panels: variance monotonicity across absences, gain trace shape around a
long absence, the leakage perturbation check, and absence reversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_four_factors.player_state.filter import (
    fit_params,
    neg_log_lik,
    obs_arrays,
    run_filter,
)

RNG = np.random.default_rng(7)

TRUE = {"lam": 0.02, "sigma2": 0.9, "c": 350.0, "mu": 12.0}


def simulate_panel(n_players=250, n_seasons=6, games_per_season=70, p_miss=0.15):
    lam, sigma2, c, mu = TRUE["lam"], TRUE["sigma2"], TRUE["c"], TRUE["mu"]
    stat = sigma2 / (2 * lam)
    rows = []
    base = pd.Timestamp("2010-11-01")
    for pid in range(n_players):
        theta = RNG.normal(mu, np.sqrt(stat))
        day = 0.0
        for s in range(n_seasons):
            day = s * 365.0 + RNG.integers(0, 5)
            prev_day = day
            for _g in range(games_per_season):
                day += float(RNG.integers(1, 4))
                dt = day - prev_day
                phi = np.exp(-lam * dt)
                theta = mu + phi * (theta - mu) + RNG.normal(0, np.sqrt(stat * (1 - phi * phi)))
                prev_day = day
                if RNG.random() < p_miss:
                    rows.append((pid, base + pd.Timedelta(days=day), 0.0, np.nan, False))
                    continue
                minutes = float(RNG.uniform(8, 40))
                y = theta + RNG.normal(0, np.sqrt(c / minutes))
                rows.append((pid, base + pd.Timedelta(days=day), minutes, y, True))
    df = pd.DataFrame(rows, columns=["person_id", "game_date", "minutes", "y", "qualifying"])
    df["game_id"] = np.arange(len(df)).astype(str)
    df["team_id"] = 0
    df["season"] = "2010_11"
    df["season_type"] = "regular_season"
    df["status"] = np.where(df["qualifying"], "played", "dnp_other")
    df["yr"] = 2010
    return df.sort_values(["person_id", "game_date"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def panel():
    return simulate_panel()


def test_synthetic_parameter_recovery(panel):
    fitted = fit_params(panel, train_through=2010)
    assert fitted["converged"]
    assert fitted["mu"] == pytest.approx(TRUE["mu"], abs=0.5)
    assert np.log(fitted["lam"]) == pytest.approx(np.log(TRUE["lam"]), abs=0.5)
    assert np.log(fitted["sigma2"]) == pytest.approx(np.log(TRUE["sigma2"]), abs=0.5)
    assert np.log(fitted["c"]) == pytest.approx(np.log(TRUE["c"]), abs=0.25)


def test_likelihood_prefers_truth_over_perturbed(panel):
    args = (*obs_arrays(panel), TRUE["mu"])
    at_truth = neg_log_lik(np.log([TRUE["lam"], TRUE["sigma2"], TRUE["c"]]), *args)
    for factor in (0.2, 5.0):
        off = neg_log_lik(np.log([TRUE["lam"] * factor, TRUE["sigma2"], TRUE["c"]]), *args)
        assert off > at_truth


def _one_player_frame(dates, minutes):
    n = len(dates)
    df = pd.DataFrame(
        {
            "person_id": 1,
            "game_date": pd.to_datetime(dates),
            "minutes": minutes,
            "game_id": [str(i) for i in range(n)],
            "team_id": 0,
            "season": "2015_16",
            "season_type": "regular_season",
            "yr": 2015,
        }
    )
    df["qualifying"] = df["minutes"] >= 5.0
    df["status"] = np.where(df["qualifying"], "played", "inactive")
    df["y"] = np.where(df["qualifying"], 14.0 + RNG.normal(0, 2, n), np.nan)
    return df


def test_variance_monotone_and_bounded_through_absence():
    dates = pd.date_range("2015-11-01", periods=40, freq="2D")
    minutes = np.array([30.0] * 10 + [0.0] * 20 + [30.0] * 10)
    df = _one_player_frame(dates, minutes)
    out = run_filter(df, TRUE)
    absent = out.iloc[10:30]
    assert (np.diff(absent.P_pred) >= -1e-9).all()
    stationary = TRUE["sigma2"] / (2 * TRUE["lam"])
    assert (out.P_pred <= stationary + 1e-6).all()


def test_gain_trace_shape_around_absence():
    dates = pd.date_range("2015-11-01", periods=40, freq="2D")
    minutes = np.array([30.0] * 15 + [0.0] * 15 + [30.0] * 10)
    df = _one_player_frame(dates, minutes)
    out = run_filter(df, TRUE)
    settled = out.gain.iloc[14]
    on_return = out.gain.iloc[30]
    later = out.gain.iloc[38]
    assert on_return > settled
    assert later < on_return


def test_leakage_perturbation_bit_identical():
    dates = pd.date_range("2015-11-01", periods=30, freq="2D")
    minutes = np.full(30, 30.0)
    df = _one_player_frame(dates, minutes)
    out1 = run_filter(df, TRUE)
    df2 = df.copy()
    df2.loc[25, "y"] = df2.loc[25, "y"] + 25.0
    out2 = run_filter(df2, TRUE)
    assert (out1.theta_pred.iloc[:26] == out2.theta_pred.iloc[:26]).all()
    assert (out1.P_pred.iloc[:26] == out2.P_pred.iloc[:26]).all()
    assert out1.theta_pred.iloc[26] != out2.theta_pred.iloc[26]


def test_absence_reverts_toward_mu_and_grows_P():
    dates = [*pd.date_range("2015-11-01", periods=20, freq="2D"), pd.Timestamp("2016-03-01")]
    minutes = np.array([34.0] * 20 + [30.0])
    df = _one_player_frame(dates, minutes)
    df.loc[:19, "y"] = 20.0
    out = run_filter(df, TRUE)
    before = out.theta_filt.iloc[19]
    on_return = out.theta_pred.iloc[20]
    assert abs(on_return - TRUE["mu"]) < abs(before - TRUE["mu"])
    assert out.P_pred.iloc[20] > out.P_pred.iloc[19]
