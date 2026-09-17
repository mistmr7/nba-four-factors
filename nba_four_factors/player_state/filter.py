"""Scalar Ornstein-Uhlenbeck state-space filter over per-36 Game Score.

Tracks each player's latent current level theta with an uncertainty P that
shrinks when he plays and grows across absences. Two things distinguish this
from an exponentially weighted moving average, and both are structural:

    Observation noise scales with playing time, R_t = c / minutes. A
    38-minute night is a far more reliable read than a 6-minute night, and a
    DNP (minutes 0) yields infinite R, zero gain, and a no-op update, so
    absence is a consequence of the model rather than a branch in the code.

    The state follows an OU process indexed by calendar days, so a
    back-to-back, the All-Star break, a two-month injury, and the offseason
    are all the same mechanism. Mean reversion bounds the uncertainty at the
    stationary variance sigma2 / (2 lam): a player unseen for a year is an
    unknown NBA player, not an infinite question mark, and his estimate has
    decayed toward the league mean mu.

State transition over an elapsed gap of dt days:

    phi(dt)  = exp(-lam * dt)
    Q(dt)    = sigma2 * (1 - exp(-2 lam dt)) / (2 lam)
    theta_t  = mu + phi(dt) * (theta_prev - mu) + w,  w ~ N(0, Q(dt))

Parameters (lam, sigma2, c) are fit by maximum likelihood on the pooled
player panel, training seasons only, via the prediction error decomposition.
Never per player: a 200-minute career cannot support its own variance
estimates. mu is the minutes-weighted league mean of the observation on the
same training seasons.

Only the one-step-ahead quantities (theta_pred, P_pred) may be used as
predictive features. Filtered and smoothed series are descriptive.

Run from repo root:
    python -m nba_four_factors.player_state.filter fit --train-through 2017
    python -m nba_four_factors.player_state.filter run
    python -m nba_four_factors.player_state.filter walkforward
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "player_state"
TABLE = OUT_DIR / "player_game_score.parquet"
PARAMS = OUT_DIR / "params_scalar.json"
PRED = OUT_DIR / "player_form_predictive.parquet"
DESC = OUT_DIR / "player_form_descriptive.parquet"
WF_PARTS = OUT_DIR / "wf_parts"

MIN_QUALIFYING_MINUTES = 5.0
WINSOR = (-20.0, 60.0)


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(
        TABLE,
        columns=[
            "game_id",
            "person_id",
            "team_id",
            "season",
            "season_type",
            "game_date",
            "minutes",
            "status",
            "game_score_per36",
        ],
    )
    df["yr"] = df["season"].str[:4].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["person_id", "game_date"]).reset_index(drop=True)
    y = df["game_score_per36"].clip(*WINSOR)
    q = (df["status"] == "played") & (df["minutes"] >= MIN_QUALIFYING_MINUTES)
    df["y"] = np.where(q, y, np.nan)
    df["qualifying"] = q
    return df


def obs_arrays(df: pd.DataFrame):
    """Per-player arrays of qualifying observations for the likelihood.

    Non-qualifying rows carry no information, and the OU predict step across
    a gap composes exactly, so the likelihood chain only needs observation
    to observation transitions with the elapsed calendar days between them.
    """
    q = df[df["qualifying"]]
    pid = q["person_id"].to_numpy()
    days = (q["game_date"] - q["game_date"].min()).dt.days.to_numpy(dtype=np.float64)
    y = q["y"].to_numpy(dtype=np.float64)
    m = q["minutes"].to_numpy(dtype=np.float64)
    new_player = np.empty(len(q), dtype=bool)
    new_player[0] = True
    new_player[1:] = pid[1:] != pid[:-1]
    dt = np.empty(len(q))
    dt[0] = 0.0
    dt[1:] = days[1:] - days[:-1]
    dt[new_player] = 0.0
    return new_player, dt, y, m


def neg_log_lik(logparams, new_player, dt, y, m, mu) -> float:
    lam, sigma2, c = np.exp(logparams)
    stat = sigma2 / (2.0 * lam)
    phi = np.exp(-lam * dt)
    Q = stat * (1.0 - phi * phi)
    R = c / m
    n = len(y)
    ll = 0.0
    theta = mu
    P = stat
    log2pi = np.log(2.0 * np.pi)
    for i in range(n):
        if new_player[i]:
            theta = mu
            P = stat
        else:
            theta = mu + phi[i] * (theta - mu)
            P = phi[i] * phi[i] * P + Q[i]
        F = P + R[i]
        v = y[i] - theta
        ll += np.log(F) + v * v / F
        K = P / F
        theta = theta + K * v
        P = (1.0 - K) * P
    return 0.5 * (n * log2pi + ll)


def fit_params(df: pd.DataFrame, train_through: int) -> dict:
    from scipy.optimize import minimize

    tr = df[df["yr"] <= train_through]
    q = tr[tr["qualifying"]]
    mu = float(np.average(q["y"], weights=q["minutes"]))
    new_player, dt, y, m = obs_arrays(tr)
    x0 = np.log([0.01, 1.0, 400.0])
    res = minimize(
        neg_log_lik,
        x0,
        args=(new_player, dt, y, m, mu),
        method="L-BFGS-B",
        options={"maxiter": 200},
    )
    lam, sigma2, c = np.exp(res.x)
    out = {
        "lam": float(lam),
        "sigma2": float(sigma2),
        "c": float(c),
        "mu": mu,
        "stationary_sd": float(np.sqrt(sigma2 / (2 * lam))),
        "halflife_days": float(np.log(2) / lam),
        "train_through": train_through,
        "nll": float(res.fun),
        "n_obs": len(y),
        "converged": bool(res.success),
    }
    return out


def run_filter(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Emit one-step-ahead and filtered state at every row of the panel.

    theta_pred / P_pred at a row use information strictly before that row's
    date. Updates happen only on qualifying observations.
    """
    lam, sigma2, c, mu = p["lam"], p["sigma2"], p["c"], p["mu"]
    stat = sigma2 / (2.0 * lam)
    pid = df["person_id"].to_numpy()
    days = (df["game_date"] - df["game_date"].min()).dt.days.to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    m = df["minutes"].to_numpy(dtype=np.float64)
    qual = df["qualifying"].to_numpy()
    n = len(df)
    theta_pred = np.empty(n)
    P_pred = np.empty(n)
    theta_filt = np.empty(n)
    P_filt = np.empty(n)
    gain = np.full(n, np.nan)
    innov = np.full(n, np.nan)

    theta = mu
    P = stat
    last_pid = None
    last_day = 0.0
    for i in range(n):
        if pid[i] != last_pid:
            theta, P = mu, stat
            last_pid = pid[i]
        else:
            dt = days[i] - last_day
            phi = np.exp(-lam * dt)
            theta = mu + phi * (theta - mu)
            P = phi * phi * P + stat * (1.0 - phi * phi)
        theta_pred[i] = theta
        P_pred[i] = P
        if qual[i]:
            R = c / m[i]
            F = P + R
            K = P / F
            v = y[i] - theta
            theta = theta + K * v
            P = (1.0 - K) * P
            gain[i] = K
            innov[i] = v
        theta_filt[i] = theta
        P_filt[i] = P
        last_day = days[i]

    out = df[
        ["game_id", "person_id", "team_id", "season", "season_type", "game_date", "minutes", "status"]
    ].copy()
    out["theta_pred"] = theta_pred
    out["P_pred"] = P_pred
    out["gain"] = gain
    out["innovation"] = innov
    out["theta_filt"] = theta_filt
    out["P_filt"] = P_filt
    return out


def cmd_profile(train_through: int) -> None:
    """Profile the likelihood over lam to check identification.

    For each fixed lam on a grid spanning day-scale to multi-year halflives,
    optimize sigma2 and c and report the profiled NLL. A flat profile means
    the reversion rate is not identified by the data and the MLE point value
    should not be interpreted; a clear minimum means it is.
    """
    from scipy.optimize import minimize

    df = load_panel()
    tr = df[df["yr"] <= train_through]
    q = tr[tr["qualifying"]]
    mu = float(np.average(q["y"], weights=q["minutes"]))
    arrays = obs_arrays(tr)
    halflives = [3, 7, 15, 30, 60, 120, 240, 480, 960, 2000, 4000]
    print(f"{'halflife_d':>10s} {'lam':>10s} {'NLL':>14s} {'sigma2':>10s} {'c':>8s}")
    best = None
    for h in halflives:
        lam = np.log(2) / h

        def obj(lp, lam=lam):
            return neg_log_lik(np.concatenate([[np.log(lam)], lp]), *arrays, mu)

        r = minimize(obj, np.log([1.0, 400.0]), method="L-BFGS-B", options={"maxiter": 100})
        s2, c = np.exp(r.x)
        print(f"{h:>10d} {lam:>10.5f} {r.fun:>14.1f} {s2:>10.4f} {c:>8.0f}", flush=True)
        if best is None or r.fun < best[1]:
            best = (h, r.fun)
    print(f"\nprofile minimum at halflife ~{best[0]} days")


def cmd_fit(train_through: int) -> None:
    df = load_panel()
    t0 = time.time()
    p = fit_params(df, train_through)
    PARAMS.write_text(json.dumps(p, indent=1))
    print(json.dumps(p, indent=1))
    print(f"fit in {time.time() - t0:.0f}s -> {PARAMS}")


def cmd_run() -> None:
    df = load_panel()
    p = json.loads(PARAMS.read_text())
    out = run_filter(df, p)
    out[
        [
            "game_id",
            "person_id",
            "team_id",
            "season",
            "season_type",
            "game_date",
            "minutes",
            "status",
            "theta_pred",
            "P_pred",
            "gain",
            "innovation",
        ]
    ].to_parquet(PRED, index=False)
    out[
        ["game_id", "person_id", "game_date", "theta_filt", "P_filt"]
    ].to_parquet(DESC, index=False)
    print(f"wrote {PRED} ({len(out):,} rows)")
    print(f"wrote {DESC} (descriptive; never join into model inputs)")


def cmd_walkforward(first_test: int = 2001, time_budget: float | None = None) -> None:
    """Per-fold refits: for each test season, fit on all prior seasons and emit
    that season's predictive rows under those frozen parameters."""
    t0 = time.time()
    WF_PARTS.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    years = sorted(df["yr"].unique())
    log = {}
    logf = WF_PARTS / "params_by_fold.json"
    if logf.exists():
        log = json.loads(logf.read_text())
    for ty in [yr for yr in years if yr >= first_test]:
        part = WF_PARTS / f"pred_{ty}.parquet"
        if part.exists():
            continue
        if time_budget is not None and time.time() - t0 > time_budget:
            print("time budget reached; re-invoke to continue", flush=True)
            return
        p = fit_params(df, ty - 1)
        sub = df[df["yr"] <= ty]
        out = run_filter(sub, p)
        out = out[out["season"].str[:4].astype(int) == ty]
        out[
            [
                "game_id",
                "person_id",
                "team_id",
                "season",
                "season_type",
                "game_date",
                "minutes",
                "status",
                "theta_pred",
                "P_pred",
                "gain",
                "innovation",
            ]
        ].to_parquet(part, index=False)
        log[str(ty)] = {
            k: p[k]
            for k in ("lam", "sigma2", "c", "mu", "halflife_days", "stationary_sd")
        }
        logf.write_text(json.dumps(log, indent=1))
        print(
            f"fold {ty}: lam {p['lam']:.4f} (halflife {p['halflife_days']:.0f}d) "
            f"sigma2 {p['sigma2']:.3f} c {p['c']:.0f} mu {p['mu']:.2f} "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
    parts = sorted(WF_PARTS.glob("pred_*.parquet"))
    all_pred = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
    all_pred.to_parquet(OUT_DIR / "player_form_predictive_wf.parquet", index=False)
    print(f"walk-forward predictive table complete: {len(all_pred):,} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit", "run", "walkforward", "profile"])
    ap.add_argument("--train-through", type=int, default=2017)
    ap.add_argument("--first-test", type=int, default=2001)
    ap.add_argument("--time-budget", type=float, default=None)
    a = ap.parse_args()
    if a.command == "fit":
        cmd_fit(a.train_through)
    elif a.command == "run":
        cmd_run()
    elif a.command == "walkforward":
        cmd_walkforward(a.first_test, a.time_budget)
    elif a.command == "profile":
        cmd_profile(a.train_through)


if __name__ == "__main__":
    main()
