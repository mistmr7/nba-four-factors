"""Scalar state-space (Kalman) team-strength model. Pure numpy.

Each team carries one hidden strength that drifts as a random walk over the
season, plus a shared home-court term. Every game is a noisy linear read of the
strength gap:

    home_margin = s[home] - s[away] + hca + noise

The filter runs once per game in date order: a predict step inflates the strength
uncertainty by the process noise Q, then an update step nudges the two teams by
the Kalman gain times the surprise (actual minus predicted margin). The prediction
recorded for each game is the one-step-ahead value formed before that game's
outcome is seen, so there is no leakage.

Each season starts fresh from the Vegas seed: a team's initial strength is its
preseason win total mapped to a points-scale net rating, with an initial variance
P0 that encodes how much the seed is trusted. The seed is therefore the filter's
prior, so this model is already cold-start aware and is directly comparable to the
seed-blended RNN and regression.

The noise levels (Q, R, P0) are tuned once on seasons before the first test year
by minimizing one-step-ahead predictive negative log likelihood, then frozen for
the expanding-window evaluation. The seed slope and home-court prior are fit on
the same pre-test block.

Run: python -m nba_four_factors.modeling.statespace_scalar
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
LINES = REPO / "data" / "features" / "vegas_game_lines.parquet"
TUNE_BEFORE = 2010
START_TEST_YEAR = 2010

_ERF = np.vectorize(math.erf)


def _phi(x, sd):
    return 0.5 * (1.0 + _ERF(x / (sd * math.sqrt(2.0))))


def load():
    m = pd.read_parquet(TABLE).dropna(subset=["home_margin", "home_win"]).copy()
    m["game_date"] = pd.to_datetime(m["game_date"])
    m["yr"] = m["season"].str[:4].astype(int)
    m = m.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return m


def seed_fit(train: pd.DataFrame):
    """Points-scale strength per preseason win total, plus the home-court prior.

    Fit the slope of realized per-game net rating on win total, and read the
    home-court prior straight off the mean home margin.
    """
    rows = []
    for _s, g in train.groupby("season"):
        diff = {}
        for t, mar in zip(g["team_id"], g["home_margin"], strict=False):
            diff.setdefault(int(t), []).append(mar)
        for t, mar in zip(g["away_team_id"], -g["home_margin"], strict=False):
            diff.setdefault(int(t), []).append(mar)
        wt = {}
        for t, w in zip(g["team_id"], g["home_win_total"], strict=False):
            wt[int(t)] = w
        for t, w in zip(g["away_team_id"], g["away_win_total"], strict=False):
            wt[int(t)] = w
        for t in diff:
            if t in wt and not np.isnan(wt[t]):
                rows.append((wt[t], float(np.mean(diff[t]))))
    a = np.array(rows)
    slope, intercept = np.polyfit(a[:, 0], a[:, 1], 1)
    mean_wt = float(
        np.nanmean(
            np.concatenate([train["home_win_total"].to_numpy(), train["away_win_total"].to_numpy()])
        )
    )
    hca = float(train["home_margin"].mean())
    return float(slope), mean_wt, hca


def run_season(g: pd.DataFrame, params, slope, mean_wt, hca):
    """Filter one season in date order, returning one-step-ahead predictions.

    Returns predicted margins, their predictive variances, actual margins, and
    actual wins, aligned to the games of g in chronological order.
    """
    Q, R, P0, P0_hca = params
    teams = sorted(set(g["team_id"].astype(int)) | set(g["away_team_id"].astype(int)))
    idx = {t: i for i, t in enumerate(teams)}
    K = len(teams)
    n = K + 1
    h = K

    wt_of = {}
    for t, w in zip(g["team_id"].astype(int), g["home_win_total"], strict=False):
        wt_of[t] = w
    for t, w in zip(g["away_team_id"].astype(int), g["away_win_total"], strict=False):
        wt_of[t] = w

    x = np.zeros(n)
    for t, i in idx.items():
        w = wt_of.get(t, mean_wt)
        x[i] = slope * ((w if not np.isnan(w) else mean_wt) - mean_wt)
    x[h] = hca
    P = np.zeros((n, n))
    P[np.arange(K), np.arange(K)] = P0
    P[h, h] = P0_hca

    pred_m, pred_v, zs, ws = [], [], [], []
    home = g["team_id"].astype(int).to_numpy()
    away = g["away_team_id"].astype(int).to_numpy()
    z_all = g["home_margin"].to_numpy(dtype=float)
    w_all = g["home_win"].to_numpy(dtype=float)

    for k in range(len(g)):
        ih, ia = idx[home[k]], idx[away[k]]
        P[np.arange(K), np.arange(K)] += Q

        Ph = P[:, ih] - P[:, ia] + P[:, h]
        s = Ph[ih] - Ph[ia] + Ph[h] + R
        mu = x[ih] - x[ia] + x[h]
        pred_m.append(mu)
        pred_v.append(s)
        zs.append(z_all[k])
        ws.append(w_all[k])

        gain = Ph / s
        e = z_all[k] - mu
        x = x + gain * e
        P = P - np.outer(gain, Ph)

    return (np.array(pred_m), np.array(pred_v), np.array(zs), np.array(ws))


def season_nll(g, params, slope, mean_wt, hca):
    pm, pv, z, _ = run_season(g, params, slope, mean_wt, hca)
    pv = np.clip(pv, 1e-6, None)
    return float(np.mean(0.5 * np.log(2 * math.pi * pv) + 0.5 * (z - pm) ** 2 / pv))


def metrics(pm, pv, z, w):
    pv = np.clip(pv, 1e-6, None)
    p = np.clip(_phi(pm, np.sqrt(pv)), 1e-6, 1 - 1e-6)
    mae = float(np.mean(np.abs(pm - z)))
    ll = float(-np.mean(w * np.log(p) + (1 - w) * np.log(1 - p)))
    acc = float(np.mean((p > 0.5) == (w > 0.5)))
    return mae, ll, acc


def tune(m, slope, mean_wt, hca):
    train = m[m["yr"] < TUNE_BEFORE]
    seasons = [g for _, g in train.groupby("season")]
    grid_Q = [0.002, 0.01, 0.03, 0.08]
    grid_R = [110.0, 130.0, 150.0, 170.0]
    grid_P0 = [4.0, 9.0, 16.0]
    best, best_nll = None, np.inf
    for Q in grid_Q:
        for R in grid_R:
            for P0 in grid_P0:
                params = (Q, R, P0, 1.0)
                nll = float(np.mean([season_nll(g, params, slope, mean_wt, hca) for g in seasons]))
                if nll < best_nll:
                    best_nll, best = nll, params
    return best, best_nll


def main():
    m = load()
    pre = m[m["yr"] < TUNE_BEFORE]
    slope, mean_wt, hca = seed_fit(pre)
    params, nll = tune(m, slope, mean_wt, hca)
    Q, R, P0, P0h = params
    print(
        f"calibrated on seasons < {TUNE_BEFORE}: slope={slope:.3f} pts/win, hca={hca:.2f}, "
        f"Q={Q}, R={R}, P0={P0}  (train NLL {nll:.4f})"
    )
    print(f"  seed trust k = R/P0 = {R / P0:.1f} games (compare the regression's ~7)\n")

    rows = []
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    for ty in range(START_TEST_YEAR, m["yr"].max() + 1):
        g = m[m["yr"] == ty]
        if len(g) < 100:
            continue
        pm, pv, z, w = run_season(g, params, slope, mean_wt, hca)
        mae, ll, _ = metrics(pm, pv, z, w)
        v = g.merge(lines, on="game_id", how="left").dropna(subset=["vegas_home_margin"])
        vmae = (
            float(np.mean(np.abs(v["home_margin"] - v["vegas_home_margin"]))) if len(v) else np.nan
        )
        rows.append((ty, len(g), mae, ll, vmae))
        print(
            f"  {ty}  n={len(g):4d}  Kalman MAE {mae:6.3f} ll {ll:.4f}  |  Vegas MAE {vmae:6.3f}",
            flush=True,
        )

    a = np.array([[r[2], r[3], r[4]] for r in rows])
    print(f"\nExpanding-window averages over {len(rows)} seasons:")
    print(f"  Kalman (scalar) : MAE {a[:,0].mean():.3f}  logloss {a[:,1].mean():.4f}")
    print(f"  Vegas line      : MAE {a[:,2].mean():.3f}")
    print(
        "  reference (same protocol): RNN+seed MAE 10.133 ll 0.6104 | M3+seed MAE 10.188 ll 0.6139"
    )
    print(
        f"  Kalman minus RNN+seed (MAE): {a[:,0].mean() - 10.133:+.3f}   "
        f"Kalman minus Vegas: {a[:,0].mean() - a[:,2].mean():+.3f}"
    )


if __name__ == "__main__":
    main()
