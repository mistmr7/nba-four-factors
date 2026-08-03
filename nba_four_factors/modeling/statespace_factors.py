"""Four-factor dynamic linear model. Pure numpy.

The scalar filter tracks one strength number per team. This richer version tracks
a four-vector per team: its latent profile in the four factors (eFG, OREB rate,
TOV rate, FT/FGA), each drifting as a random walk. Every game gives four
observations, the home-minus-away realized factor differentials, each a noisy read
of the gap between the two teams' latent profiles:

    factor_diff = f[home] - f[away] + noise   (one row per factor)

Margin is a linear readout of the latent factor gap, with weights and a home-court
term fit on training data:

    home_margin = w . (f[home] - f[away]) + hca + noise

So strength is decomposed into and tracked through the four factors instead of as
a single scalar. The state updates only from the factor observations; margin is a
pure readout, which is the honest test of whether decomposing strength into four
tracked factors beats tracking strength directly.

Factors are standardized on the pre-test block so one set of noise levels applies
across them. Each season is seeded from the Vegas win total, distributed across
the factors along the readout-weight direction so the preseason prior is
preserved. Noise levels are tuned once on seasons before the first test year and
frozen. One-step-ahead predictions only, so there is no leakage.

Run: python -m nba_four_factors.modeling.statespace_factors
"""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
LINES = REPO / "data" / "features" / "vegas_game_lines.parquet"
TUNE_BEFORE = 2010
START_TEST_YEAR = 2010
NF = 4
# Observation noise for the optional margin row. The margin scatters around a
# team's latent strength by the full game noise, not by the tiny within-game
# readout residual, so the margin read is informative but weak. Using the small
# readout residual instead makes the margin nearly a copy of the factor rows and
# the update degenerate.
MARGIN_R = 130.0

_ERF = np.vectorize(math.erf)


def _phi(x, sd):
    return 0.5 * (1.0 + _ERF(x / (sd * math.sqrt(2.0))))


def per_game_factor_diffs() -> pd.DataFrame:
    """Home-minus-away realized four-factor differentials, one row per game."""
    df = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "*/regular_season.parquet")))],
        ignore_index=True,
    )
    df = df[(~df["is_neutral"]) & (df["season"] != "1998_99")].copy()
    df["efg"] = df["off_efg_pct"]
    df["orb"] = df["off_orb_pct"]
    df["tov"] = df["off_tov_pct"]
    df["ftr"] = df["ftm"] / df["fga"]
    keep = ["game_id", "is_home", "efg", "orb", "tov", "ftr"]
    df = df[keep].replace([np.inf, -np.inf], np.nan).dropna()
    home = df[df["is_home"]].set_index("game_id")[["efg", "orb", "tov", "ftr"]]
    away = df[~df["is_home"]].set_index("game_id")[["efg", "orb", "tov", "ftr"]]
    both = home.join(away, lsuffix="_h", rsuffix="_a", how="inner")
    out = pd.DataFrame(index=both.index)
    for c in ["efg", "orb", "tov", "ftr"]:
        out[c] = both[c + "_h"] - both[c + "_a"]
    return out.reset_index()


def load():
    m = pd.read_parquet(TABLE).dropna(subset=["home_margin", "home_win"]).copy()
    m["game_date"] = pd.to_datetime(m["game_date"])
    m["yr"] = m["season"].str[:4].astype(int)
    fd = per_game_factor_diffs()
    m = m.merge(fd, on="game_id", how="inner")
    return m.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def calibrate(pre: pd.DataFrame):
    """Standardization, margin readout weights, seed slope, and home-court prior."""
    cols = ["efg", "orb", "tov", "ftr"]
    mu = pre[cols].mean().to_numpy()
    sd = pre[cols].std().to_numpy()
    Y = (pre[cols].to_numpy() - mu) / sd
    X = np.column_stack([np.ones(len(Y)), Y])
    coef, *_ = np.linalg.lstsq(X, pre["home_margin"].to_numpy(dtype=float), rcond=None)
    hca, w = float(coef[0]), coef[1:]
    resid = pre["home_margin"].to_numpy(dtype=float) - X @ coef
    margin_var = float(np.var(resid))

    rows = []
    for _s, g in pre.groupby("season"):
        diff = {}
        for t, mar in zip(g["team_id"], g["home_margin"], strict=False):
            diff.setdefault(int(t), []).append(mar)
        for t, mar in zip(g["away_team_id"], -g["home_margin"], strict=False):
            diff.setdefault(int(t), []).append(mar)
        wt = {int(t): w_ for t, w_ in zip(g["team_id"], g["home_win_total"], strict=False)}
        wt.update(
            {int(t): w_ for t, w_ in zip(g["away_team_id"], g["away_win_total"], strict=False)}
        )
        for t in diff:
            if t in wt and not np.isnan(wt[t]):
                rows.append((wt[t], float(np.mean(diff[t]))))
    a = np.array(rows)
    slope = float(np.polyfit(a[:, 0], a[:, 1], 1)[0])
    mean_wt = float(
        np.nanmean(
            np.concatenate([pre["home_win_total"].to_numpy(), pre["away_win_total"].to_numpy()])
        )
    )
    return dict(mu=mu, sd=sd, w=w, hca=hca, margin_var=margin_var, slope=slope, mean_wt=mean_wt)


def run_season(g: pd.DataFrame, params, cal, use_margin=False):
    """Filter one season. With use_margin the realized margin is added as a fifth
    observation, tied to the latent factors through the readout weights, so the
    filter corrects the factor states from the scoreboard as well as from the box
    score. Otherwise margin is a pure readout and the state sees only the factors.
    """
    Q, R, P0 = params
    w, hca, margin_var = cal["w"], cal["hca"], cal["margin_var"]
    slope, mean_wt = cal["slope"], cal["mean_wt"]
    teams = sorted(set(g["team_id"].astype(int)) | set(g["away_team_id"].astype(int)))
    idx = {t: i for i, t in enumerate(teams)}
    K = len(teams)
    n = K * NF

    wt_of = {int(t): v for t, v in zip(g["team_id"].astype(int), g["home_win_total"], strict=False)}
    wt_of.update(
        {
            int(t): v
            for t, v in zip(g["away_team_id"].astype(int), g["away_win_total"], strict=False)
        }
    )

    x = np.zeros(n)
    wnorm = float(w @ w)
    for t, i in idx.items():
        wv = wt_of.get(t, mean_wt)
        s_i = slope * ((wv if not np.isnan(wv) else mean_wt) - mean_wt)
        x[i * NF : (i + 1) * NF] = (s_i / wnorm) * w
    P = np.eye(n) * P0

    cols = ["efg", "orb", "tov", "ftr"]
    Y = (g[cols].to_numpy(dtype=float) - cal["mu"]) / cal["sd"]
    home = g["team_id"].astype(int).to_numpy()
    away = g["away_team_id"].astype(int).to_numpy()
    z_all = g["home_margin"].to_numpy(dtype=float)
    win_all = g["home_win"].to_numpy(dtype=float)
    I4 = np.eye(NF)

    pred_m, pred_v, zs, ws = [], [], [], []
    for k in range(len(g)):
        hc = idx[home[k]] * NF
        ac = idx[away[k]] * NF
        P[np.arange(n), np.arange(n)] += Q

        gap = x[hc : hc + NF] - x[ac : ac + NF]
        mu_margin = float(w @ gap + hca)
        wrow = np.zeros(n)
        wrow[hc : hc + NF] = w
        wrow[ac : ac + NF] = -w
        var_margin = float(wrow @ P @ wrow) + margin_var
        pred_m.append(mu_margin)
        pred_v.append(var_margin)
        zs.append(z_all[k])
        ws.append(win_all[k])

        PHt_f = P[:, hc : hc + NF] - P[:, ac : ac + NF]
        ef = Y[k] - gap
        if use_margin:
            PHt = np.column_stack([PHt_f, PHt_f @ w])
            top = PHt[hc : hc + NF, :] - PHt[ac : ac + NF, :]
            sm = w @ PHt[hc : hc + NF, :] - w @ PHt[ac : ac + NF, :]
            S = np.vstack([top, sm[None, :]])
            S[np.arange(NF), np.arange(NF)] += R
            S[NF, NF] += MARGIN_R
            e = np.append(ef, (z_all[k] - hca) - float(w @ gap))
        else:
            PHt = PHt_f
            S = (PHt_f[hc : hc + NF, :] - PHt_f[ac : ac + NF, :]) + R * I4
            e = ef
        Kg = np.linalg.solve(S, PHt.T).T
        x = x + Kg @ e
        P = P - Kg @ PHt.T
        P = 0.5 * (P + P.T)

    return np.array(pred_m), np.array(pred_v), np.array(zs), np.array(ws)


def season_nll(g, params, cal, use_margin=False):
    pm, pv, z, _ = run_season(g, params, cal, use_margin)
    pv = np.clip(pv, 1e-6, None)
    return float(np.mean(0.5 * np.log(2 * math.pi * pv) + 0.5 * (z - pm) ** 2 / pv))


def metrics(pm, pv, z, win):
    pv = np.clip(pv, 1e-6, None)
    p = np.clip(_phi(pm, np.sqrt(pv)), 1e-6, 1 - 1e-6)
    mae = float(np.mean(np.abs(pm - z)))
    ll = float(-np.mean(win * np.log(p) + (1 - win) * np.log(1 - p)))
    return mae, ll


def tune(m, cal, use_margin=False):
    seasons = [g for _, g in m[m["yr"] < TUNE_BEFORE].groupby("season")]
    best, best_nll = None, np.inf
    for Q in [0.002, 0.01, 0.03]:
        for R in [0.5, 1.0, 2.0, 4.0]:
            for P0 in [0.5, 1.0]:
                params = (Q, R, P0)
                nll = float(np.mean([season_nll(g, params, cal, use_margin) for g in seasons]))
                if nll < best_nll:
                    best_nll, best = nll, params
    return best, best_nll


def evaluate(m, cal, use_margin, lines):
    params, nll = tune(m, cal, use_margin)
    rows = []
    for ty in range(START_TEST_YEAR, m["yr"].max() + 1):
        g = m[m["yr"] == ty]
        if len(g) < 100:
            continue
        pm, pv, z, win = run_season(g, params, cal, use_margin)
        mae, ll = metrics(pm, pv, z, win)
        v = g.merge(lines, on="game_id", how="left").dropna(subset=["vegas_home_margin"])
        vmae = (
            float(np.mean(np.abs(v["home_margin"] - v["vegas_home_margin"]))) if len(v) else np.nan
        )
        rows.append((ty, len(g), mae, ll, vmae))
        tag = "4F-DLM+margin" if use_margin else "4F-DLM"
        print(
            f"  {ty}  n={len(g):4d}  {tag:14s} MAE {mae:6.3f} ll {ll:.4f}  |  Vegas MAE {vmae:6.3f}",
            flush=True,
        )
    a = np.array([[r[2], r[3], r[4]] for r in rows])
    return params, a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()


def main():
    m = load()
    cal = calibrate(m[m["yr"] < TUNE_BEFORE])
    print(
        f"calibrated on seasons < {TUNE_BEFORE}: readout weights (eFG,OREB,TOV,FT) = "
        f"{np.round(cal['w'], 2)} pts/sd, hca={cal['hca']:.2f}, "
        f"margin readout resid sd={math.sqrt(cal['margin_var']):.2f}\n"
    )
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    mode = os.environ.get("DLM_MODE", "factors")
    use_margin = mode == "margin"
    print(f"mode: {'factor + margin observations' if use_margin else 'factor observations only'}")
    params, mae, ll, vmae = evaluate(m, cal, use_margin, lines)

    print("\nExpanding-window averages over 16 seasons:")
    tag = "4-factor DLM (factors+margin)" if use_margin else "4-factor DLM (factors only)"
    print(f"  {tag:30s}: MAE {mae:.3f}  logloss {ll:.4f}   params Q,R,P0={params}")
    print(f"  Vegas line                    : MAE {vmae:.3f}")
    print(
        "  reference: Kalman-scalar 10.103/0.6091 | RNN+seed 10.133/0.6104 | M3+seed 10.188/0.6139"
    )


if __name__ == "__main__":
    main()
