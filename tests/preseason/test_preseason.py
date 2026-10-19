"""Tests for the preseason win-total lane.

All synthetic, no data directory needed. The two leakage tests are the
important ones: perturbing anything from the target season (player
minutes, single-season RAPM, team results) must not move that season's
projections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_four_factors.preseason import backtest as bt
from nba_four_factors.preseason import inputs as io
from nba_four_factors.preseason.project import Config, add_design, project_season, team_ratings

SEASONS = ["2001_02", "2002_03", "2003_04", "2004_05", "2005_06"]


def synthetic_frame(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in SEASONS:
        for team in range(4):
            for k in range(13):
                v_p = rng.normal(0, 1.5) if k < 11 else np.nan
                mpg_p = max(0.0, 34 - 2.5 * k + rng.normal(0, 3)) if k < 11 else 0.0
                rows.append(
                    {
                        "season": s,
                        "team_id": team,
                        "person_id": hash((s, team, k)) % 10_000_000,
                        "v_p": v_p,
                        "mpg_p": mpg_p,
                        "mpg_p2": mpg_p * 0.9,
                        "gp_frac_p": 0.8 if k < 11 else 0.0,
                        "new_team": bool(rng.random() < 0.2),
                        "rookie": k >= 11,
                        "age": 20 + k if k < 11 else 20,
                        "age_c": (20 + k if k < 11 else 20) - 27.0,
                        "drafted": True,
                        "log_pick": np.log(1 + k * 4),
                        "years_since_draft": float(k),
                        "mpg_S": max(0.0, mpg_p + rng.normal(0, 4)),
                        "v_S": (0 if np.isnan(v_p) else 0.5 * v_p) + rng.normal(0, 1),
                        "poss_S": 1000.0 + 300 * (11 - k),
                        "out_g1": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_season_helpers() -> None:
    assert io.prev_season("2026_27") == "2025_26"
    assert io.season_id(1999) == "1999_00"
    assert io.schedule_length("2011_12") == 66
    assert io.schedule_length("2024_25") == 82


def test_result_frame_parses_classic_shape() -> None:
    payload = {"resultSets": [{"name": "X", "headers": ["A", "B"], "rowSet": [[1, 2], [3, 4]]}]}
    df = io._result_frame(payload, "X")
    assert list(df.columns) == ["A", "B"]
    assert df.A.tolist() == [1, 3]
    assert io._result_frame(payload, "missing").empty


@pytest.mark.parametrize("value,minutes", [("model", "model"), ("raw", "naive")])
def test_shares_sum_to_five(value: str, minutes: str) -> None:
    cfg = Config(value=value, minutes=minutes)
    f = add_design(synthetic_frame(), cfg)
    p = project_season(f, "2005_06", cfg)
    sums = p.groupby("team_id").share5.sum()
    assert np.allclose(sums, 5.0)
    assert (p.mpg_hat >= 0).all()


def test_projection_ignores_target_season_outcomes() -> None:
    cfg = Config()
    f = add_design(synthetic_frame(), cfg)
    base = team_ratings(project_season(f, "2005_06", cfg)).rating.to_numpy()
    g = f.copy()
    mask = g.season == "2005_06"
    g.loc[mask, "mpg_S"] = 99.0
    g.loc[mask, "v_S"] = -50.0
    g.loc[mask, "poss_S"] = 1.0
    moved = team_ratings(project_season(g, "2005_06", cfg)).rating.to_numpy()
    assert np.allclose(base, moved)


def test_calibration_ignores_target_season_results() -> None:
    rng = np.random.default_rng(11)
    seasons = [f"{y}_{str(y + 1)[2:]}" for y in range(2000, 2008)]
    r = pd.DataFrame(
        [
            {"season": s, "team_id": t, "rating": rng.normal(0, 3)}
            for s in seasons
            for t in range(10)
        ]
    )
    r["margin"] = 1.5 * r.rating + rng.normal(0, 2, len(r))
    r["wpct"] = 0.5 + 0.03 * r.margin
    r["vpct"] = 0.5 + 0.025 * r.margin + rng.normal(0, 0.03, len(r))
    a = bt.calibrate(r)
    r2 = r.copy()
    last = r2.season == seasons[-1]
    r2.loc[last, "margin"] = 40.0
    r2.loc[last, "wpct"] = 1.0
    b = bt.calibrate(r2)
    cols = ["w_model", "w_blend"]
    assert np.allclose(a[a.season == seasons[-1]][cols], b[b.season == seasons[-1]][cols])


def test_score_counts_hits_and_drops_pushes() -> None:
    d = pd.DataFrame(
        {
            "season": ["2010_11"] * 4,
            "sched": [82] * 4,
            "vpct": [40.5 / 82, 40.5 / 82, 50.5 / 82, 41 / 82],
            "wpct": [45 / 82, 30 / 82, 55 / 82, 41 / 82],
            "w_model": [44 / 82, 44 / 82, 49 / 82, 45 / 82],
            "w_blend": [44 / 82, 44 / 82, 49 / 82, 45 / 82],
        }
    )
    s = bt.score(d, ("2010_11", "2010_11"))
    assert s["w_blend_n_0"] == 3
    assert s["w_blend_hit_0"] == pytest.approx(1 / 3)


def test_calibration_invariant_to_season_level_shift() -> None:
    rng = np.random.default_rng(5)
    seasons = [f"{y}_{str(y + 1)[2:]}" for y in range(2000, 2008)]
    r = pd.DataFrame(
        [
            {"season": s, "team_id": t, "rating": rng.normal(0, 3)}
            for s in seasons
            for t in range(10)
        ]
    )
    r["margin"] = 1.5 * (r.rating - r.groupby("season").rating.transform("mean"))
    r["wpct"] = 0.5 + 0.03 * r.margin
    r["vpct"] = 0.5 + 0.025 * r.margin
    r["team_age"] = 27.0
    a = bt.calibrate(r)
    shifted = r.copy()
    shifted.loc[shifted.season == seasons[-1], "rating"] += 4.0
    b = bt.calibrate(shifted)
    last = seasons[-1]
    assert np.allclose(a[a.season == last].w_model, b[b.season == last].w_model)


def test_availability_moves_minutes_to_teammates() -> None:
    from nba_four_factors.preseason.project import apply_availability

    proj = pd.DataFrame(
        {
            "season": ["2026_27"] * 3,
            "team_id": [1, 1, 1],
            "person_id": [10, 11, 12],
            "mpg_hat": [30.0, 20.0, 10.0],
            "v_hat": [3.0, 0.0, -1.0],
        }
    )
    out = apply_availability(proj, pd.Series({10: 41.0}))
    assert np.isclose(out.share5.sum(), 5.0)
    assert np.isclose(out.loc[out.person_id == 10, "mpg_hat"].item(), 15.0)
    assert out.loc[out.person_id == 11, "share5"].item() > 20.0 / 60.0 * 5
    full = apply_availability(proj, pd.Series(dtype=float))
    assert out.share5.mul(out.v_hat).sum() < full.share5.mul(full.v_hat).sum()


def test_name_key_folds_accents_and_suffixes() -> None:
    assert io.name_key("Kristaps Porziņģis") == io.name_key("Kristaps Porzingis")
    assert io.name_key("Jimmy Butler III") == "jimmy butler"
    assert io.name_key("Kelly Oubre Jr.") == "kelly oubre"
