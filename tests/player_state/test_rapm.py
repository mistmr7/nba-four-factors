"""Tests for the RAPM fitter and the as-of schedule. All synthetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from nba_four_factors.player_state.rapm import (
    aggregate_lineup_pairs,
    build_design,
    fit_asof,
    fit_rapm,
    ratings_for_date,
    ridge_solve,
    weekly_cutoffs,
)


def planted_stints(n_players=40, n_stints=4000, seed=0, with_dates=False):
    rng = np.random.default_rng(seed)
    true = rng.normal(0, 2, n_players)
    rows = []
    for k in range(n_stints):
        h = rng.choice(n_players, 5, replace=False)
        a = rng.choice(np.setdiff1d(np.arange(n_players), h), 5, replace=False)
        poss = 20.0
        margin = true[h].sum() - true[a].sum()
        row = {
            **{f"h{i + 1}": int(p) for i, p in enumerate(h)},
            **{f"a{i + 1}": int(p) for i, p in enumerate(a)},
            "game_id": f"g{k // 30:05d}",
            "home_poss": poss,
            "away_poss": poss,
            "home_pts": poss * (110 + margin / 2 + rng.normal(0, 3)) / 100,
            "away_pts": poss * (110 - margin / 2 + rng.normal(0, 3)) / 100,
            "seconds": 300.0,
        }
        if with_dates:
            row["game_date"] = pd.Timestamp("2023-10-24") + pd.Timedelta(days=k // 30)
        rows.append(row)
    return pd.DataFrame(rows), true


def test_ridge_recovers_planted_ratings():
    st, true = planted_stints()
    fit = fit_rapm(st, lam=50.0, aggregate=False)
    r = fit.ratings.set_index("person_id")["rapm"]
    assert np.corrcoef(r.loc[np.arange(len(true))], true)[0, 1] > 0.95
    assert abs(fit.intercept - 110.0) < 2.0


def test_aggregation_does_not_change_the_fit():
    st, _ = planted_stints(n_stints=1500)
    a = fit_rapm(st, lam=50.0, aggregate=False).ratings.set_index("person_id")["rapm"]
    b = fit_rapm(st, lam=50.0, aggregate=True).ratings.set_index("person_id")["rapm"]
    assert np.allclose(a.sort_index(), b.sort_index(), atol=1e-6)
    assert len(aggregate_lineup_pairs(st)) <= len(st)


def test_prior_pulls_toward_prior_when_data_is_thin():
    X = sp.csr_matrix(np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
    y = np.array([100.0, 100.0])
    w = np.array([1.0, 1.0])
    prior = np.array([0.0, 0.0, 5.0])
    b = ridge_solve(X, y, w, lam=1e6, prior=prior)
    assert abs(b[2] - 5.0) < 1e-3


def test_prior_signs_offense_positive_defense_negated():
    st, _ = planted_stints(n_stints=200)
    prior_off = pd.Series(4.0, index=np.arange(40))
    prior_def = pd.Series(3.0, index=np.arange(40))
    fit = fit_rapm(st, lam=1e7, prior_off=prior_off, prior_def=prior_def)
    assert np.allclose(fit.ratings["orapm"], 4.0, atol=0.05)
    assert np.allclose(fit.ratings["drapm"], 3.0, atol=0.05)


def test_design_shapes_and_home_court_column():
    st = pd.DataFrame(
        [
            {
                "h1": 1,
                "h2": 2,
                "h3": 3,
                "h4": 4,
                "h5": 5,
                "a1": 6,
                "a2": 7,
                "a3": 8,
                "a4": 9,
                "a5": 10,
                "home_poss": 10.0,
                "away_poss": 5.0,
                "home_pts": 12,
                "away_pts": 4,
                "seconds": 100.0,
            }
        ]
    )
    X, y, w, _players = build_design(st)
    assert X.shape == (2, 2 + 20)
    assert y.tolist() == [120.0, 80.0] and w.tolist() == [10.0, 5.0]
    dense = X.toarray()
    assert dense[0, 1] == 1.0 and dense[1, 1] == 0.0
    assert dense[0, 2:12].sum() == 5 and dense[0, 12:].sum() == 5


def test_asof_uses_only_games_before_the_cutoff():
    st, _ = planted_stints(n_stints=3000, with_dates=True)
    cuts = weekly_cutoffs(st["game_date"], warmup_days=14, step_days=7)
    asof = fit_asof(st, cuts, lam=50.0, min_stints=100)
    first = asof["asof_date"].min()
    n_before = int((st["game_date"] < first).sum())
    assert int(asof.loc[asof["asof_date"] == first, "n_stints"].iloc[0]) == n_before
    # a rating for a game date is the latest cutoff at or before it
    pick = ratings_for_date(asof, first + pd.Timedelta(days=3))
    assert pick["asof_date"].iloc[0] == first
