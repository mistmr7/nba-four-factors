"""Evaluate the scalar player filter against rolling-window baselines.

The fancy-EWMA test: a state-space filter only earns its keep if its
one-step-ahead prediction of the next per-36 Game Score beats simple rolling
means of the last L qualifying games. L is swept over 5, 10, 15, and 20,
plus season-to-date and career-to-date means.

All predictors are evaluated on identical rows: qualifying observations in
test seasons (2001 onward, matching the ladder window) where the player has
at least 20 prior qualifying career games, so every window length is fully
defined. The target is the realized per-36 Game Score of that game, which is
noisy for every predictor alike; differences in MAE reflect differences in
the quality of the underlying level estimate.

Also reports filter calibration: standardized innovations v / sqrt(P + R)
should have mean near 0 and standard deviation near 1 if the filter's
uncertainties are honest.

Run from repo root:
    python -m nba_four_factors.player_state.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .filter import MIN_QUALIFYING_MINUTES, OUT_DIR, TABLE, WINSOR

WF_PRED = OUT_DIR / "player_form_predictive_wf.parquet"
WF_PARAMS = OUT_DIR / "wf_parts" / "params_by_fold.json"
WINDOWS = (5, 10, 15, 20)
MIN_PRIOR = 20
FIRST_TEST = 2001


def build_frame() -> pd.DataFrame:
    df = pd.read_parquet(
        TABLE,
        columns=["game_id", "person_id", "season", "game_date", "minutes", "status",
                 "game_score_per36"],
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[(df.status == "played") & (df.minutes >= MIN_QUALIFYING_MINUTES)].copy()
    df["y"] = df.game_score_per36.clip(*WINSOR)
    df = df.sort_values(["person_id", "game_date"]).reset_index(drop=True)

    g = df.groupby("person_id")["y"]
    for L in WINDOWS:
        df[f"roll{L}"] = g.transform(lambda s, L=L: s.shift(1).rolling(L).mean())
    df["career"] = g.transform(lambda s: s.shift(1).expanding().mean())
    gs = df.groupby(["person_id", "season"])["y"]
    df["season_td"] = gs.transform(lambda s: s.shift(1).expanding().mean())
    df["n_prior"] = g.transform(lambda s: s.shift(1).expanding().count())

    wf = pd.read_parquet(WF_PRED, columns=["game_id", "person_id", "theta_pred", "P_pred", "minutes"])
    wf = wf.rename(columns={"minutes": "wf_minutes"})
    df = df.merge(wf, on=["game_id", "person_id"], how="inner")
    df["yr"] = df.season.str[:4].astype(int)
    return df


def report() -> None:
    df = build_frame()
    e = df[(df.yr >= FIRST_TEST) & (df.n_prior >= MIN_PRIOR)].copy()
    e = e.dropna(subset=[f"roll{L}" for L in WINDOWS] + ["career", "theta_pred"])
    print(f"evaluation rows: {len(e):,} qualifying observations, "
          f"{e.yr.min()}-{e.yr.max()}, players with >= {MIN_PRIOR} prior games")

    preds = {"filter theta_pred": "theta_pred"}
    preds.update({f"rolling mean L={L}": f"roll{L}" for L in WINDOWS})
    preds["season-to-date mean"] = "season_td"
    preds["career mean"] = "career"

    print(f"\n{'predictor':>22s} {'MAE':>8s} {'RMSE':>8s}")
    rows = []
    for name, col in preds.items():
        ae = (e[col] - e.y).abs()
        ok = ae.notna()
        mae = float(ae[ok].mean())
        rmse = float(np.sqrt(((e[col] - e.y)[ok] ** 2).mean()))
        rows.append((name, mae))
        print(f"{name:>22s} {mae:>8.4f} {rmse:>8.4f}  (n={int(ok.sum()):,})")

    best = min(rows, key=lambda r: r[1])
    print(f"\nbest: {best[0]}")

    params = json.loads(Path(WF_PARAMS).read_text())
    cs = {int(k): v["c"] for k, v in params.items()}
    e["c_fold"] = e.yr.map(cs)
    z = (e.y - e.theta_pred) / np.sqrt(e.P_pred + e.c_fold / e.minutes)
    print("\ncalibration of standardized innovations (want mean~0, sd~1):")
    print(f"  mean {z.mean():+.4f}   sd {z.std():.4f}   n={len(z):,}")

    print("\nby era (filter MAE minus best rolling-window MAE, negative = filter wins):")
    for lo, hi in [(2001, 2008), (2009, 2016), (2017, 2025)]:
        s = e[(e.yr >= lo) & (e.yr <= hi)]
        fm = (s.theta_pred - s.y).abs().mean()
        rm = min((s[f"roll{L}"] - s.y).abs().mean() for L in WINDOWS)
        cm = (s.career - s.y).abs().mean()
        print(f"  {lo}-{hi}: filter {fm:.4f}  best-roll {rm:.4f}  career {cm:.4f}  "
              f"delta(filter-roll) {fm-rm:+.4f}")

    # Noise floor and skill fraction. An oracle knowing theta exactly still
    # incurs E|eps| = sqrt(2 R_t / pi) under the Gaussian assumption, so the
    # interesting denominator is the removable error above that floor, not
    # the raw MAE. Winsorization makes the Gaussian expectation slightly
    # optimistic; the floor is an approximation and is labeled as such.
    R = e.c_fold / e.minutes
    floor = float(np.sqrt(2.0 * R / np.pi).mean())
    mae_f = float((e.theta_pred - e.y).abs().mean())
    mae_20 = float((e.roll20 - e.y).abs().mean())
    skill = (mae_20 - mae_f) / (mae_20 - floor)
    print(f"\nnoise floor (approx): MAE_floor {floor:.4f}")
    print(f"skill vs rolling-20: ({mae_20:.4f} - {mae_f:.4f}) / ({mae_20:.4f} - "
          f"{floor:.4f}) = {skill:.1%} of removable error")

    # Unrestricted table: every observation with at least one prior qualifying
    # appearance. Windows use whatever history exists; the season-to-date mean
    # falls back to the fold's league mean when empty. This is the regime
    # (rookies, returners, sparse-minutes players, early season) where
    # principled shrinkage should show its largest advantage.
    mus = {int(k): v["mu"] for k, v in params.items()}
    u = df[(df.yr >= FIRST_TEST) & (df.n_prior >= 1)].copy()
    g = u.groupby("person_id")["y"]
    for L in WINDOWS:
        u[f"roll{L}"] = g.transform(lambda s, L=L: s.shift(1).rolling(L, min_periods=1).mean())
    u["mu_fold"] = u.yr.map(mus)
    u["season_td"] = u["season_td"].fillna(u["mu_fold"])
    u = u.dropna(subset=["theta_pred", "career"])
    print(f"\nunrestricted table: {len(u):,} rows ({len(u) - len(e):,} more than the "
          f"restricted table)")
    print(f"{'predictor':>22s} {'MAE':>8s}")
    for name, col in preds.items():
        mae = float((u[col] - u.y).abs().mean())
        print(f"{name:>22s} {mae:>8.4f}")


if __name__ == "__main__":
    report()
