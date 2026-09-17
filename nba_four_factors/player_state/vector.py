"""Vector-state extension: five Game Score components, evaluated as an upgrade.

The Game Score terms partition into five components that sum exactly to the
total, so the scalar model is the aggregate of the vector model and the
comparison is clean:

    scoring     PTS + 0.4 FGM - 0.7 FGA - 0.4 (FTA - FTM)
    rebounding  0.7 OREB + 0.3 DREB
    playmaking  0.7 AST
    defense     STL + 0.7 BLK
    discipline  -0.4 PF - TOV

Version one uses a diagonal transition and diagonal observation covariance,
which makes the MLE separable: five independent scalar OU fits, one per
component, each with its own (lam, sigma2, c, mu). Fit on training seasons
(through 2017), decided on held-out seasons (2018 onward).

Decision rule, from the lane handoff: the vector version is adopted only if
the sum of its component one-step-ahead predictions beats the scalar
one-step-ahead prediction on held-out Gaussian NLL and on MAE. If it wins
one and not the other, that is reported plainly rather than papered over.

Run from repo root:
    python -m nba_four_factors.player_state.vector fit
    python -m nba_four_factors.player_state.vector evaluate
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .filter import (
    MIN_QUALIFYING_MINUTES,
    OUT_DIR,
    TABLE,
    WINSOR,
    fit_params,
    run_filter,
)

VPARAMS = OUT_DIR / "params_vector.json"
VPRED = OUT_DIR / "player_form_vector_pred.parquet"
TRAIN_THROUGH = 2017
FIRST_HOLDOUT = 2018
MIN_PRIOR = 20

COMPONENT_WINSOR = {
    "scoring": (-15.0, 45.0),
    "rebounding": (0.0, 15.0),
    "playmaking": (0.0, 15.0),
    "defense": (0.0, 12.0),
    "discipline": (-12.0, 0.0),
}


def component_frame() -> pd.DataFrame:
    df = pd.read_parquet(TABLE)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["yr"] = df["season"].str[:4].astype(int)
    m = df["minutes"].replace(0, np.nan)
    comp = pd.DataFrame(index=df.index)
    comp["scoring"] = df.pts + 0.4 * df.fgm - 0.7 * df.fga - 0.4 * (df.fta - df.ftm)
    comp["rebounding"] = 0.7 * df.oreb + 0.3 * df.dreb
    comp["playmaking"] = 0.7 * df.ast
    comp["defense"] = df.stl + 0.7 * df.blk
    comp["discipline"] = -0.4 * df.pf - df.tov
    total = comp.sum(axis=1)
    played = df.status == "played"
    err = (total[played] - df.game_score[played]).abs().max()
    if not np.isnan(err) and err > 1e-6:
        raise AssertionError(f"components do not sum to game_score, max err {err}")
    for c in comp.columns:
        df[f"c36_{c}"] = 36.0 * comp[c] / m
    return df


def panel_for(df: pd.DataFrame, comp: str) -> pd.DataFrame:
    p = df[
        ["game_id", "person_id", "team_id", "season", "season_type",
         "game_date", "minutes", "status", "yr"]
    ].copy()
    y = df[f"c36_{comp}"].clip(*COMPONENT_WINSOR[comp])
    q = (df["status"] == "played") & (df["minutes"] >= MIN_QUALIFYING_MINUTES)
    p["y"] = np.where(q, y, np.nan)
    p["qualifying"] = q
    return p.sort_values(["person_id", "game_date"]).reset_index(drop=True)


def cmd_fit() -> None:
    df = component_frame()
    params = {}
    if VPARAMS.exists():
        params = json.loads(VPARAMS.read_text())
    for comp in COMPONENT_WINSOR:
        if comp in params:
            continue
        t0 = time.time()
        p = fit_params(panel_for(df, comp), TRAIN_THROUGH)
        params[comp] = p
        VPARAMS.write_text(json.dumps(params, indent=1))
        print(
            f"{comp}: lam {p['lam']:.5f} (halflife {p['halflife_days']:.0f}d) "
            f"sigma2 {p['sigma2']:.4f} c {p['c']:.1f} mu {p['mu']:.2f} "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
    print(f"wrote {VPARAMS}")


def cmd_evaluate() -> None:
    df = component_frame()
    params = json.loads(VPARAMS.read_text())

    keys = None
    sum_pred = None
    sum_var = None
    for comp, p in params.items():
        panel = panel_for(df, comp)
        out = run_filter(panel, p)
        out = out[["game_id", "person_id", "theta_pred", "P_pred", "minutes", "status"]]
        var = out.P_pred + p["c"] / out.minutes.replace(0, np.nan)
        if keys is None:
            keys = out[["game_id", "person_id", "minutes", "status"]].copy()
            sum_pred = out.theta_pred.copy()
            sum_var = var.copy()
        else:
            sum_pred = sum_pred + out.theta_pred
            sum_var = sum_var + var
    keys["vec_pred"] = sum_pred
    keys["vec_var"] = sum_var
    keys.to_parquet(VPRED, index=False)

    scal = pd.read_parquet(
        OUT_DIR / "player_form_predictive.parquet",
        columns=["game_id", "person_id", "theta_pred", "P_pred"],
    )
    sp = json.loads((OUT_DIR / "params_scalar.json").read_text())

    base = df[["game_id", "person_id", "season", "yr", "minutes", "status",
               "game_score_per36"]].copy()
    base["y"] = base.game_score_per36.clip(*WINSOR)
    q = (base.status == "played") & (base.minutes >= MIN_QUALIFYING_MINUTES)
    base = base[q].sort_values(["person_id", "yr"])
    base["n_prior"] = base.groupby("person_id").cumcount()

    e = base.merge(keys[["game_id", "person_id", "vec_pred", "vec_var"]],
                   on=["game_id", "person_id"]).merge(
        scal, on=["game_id", "person_id"]
    )
    e["scal_var"] = e.P_pred + sp["c"] / e.minutes
    e = e[(e.yr >= FIRST_HOLDOUT) & (e.n_prior >= MIN_PRIOR)].dropna(
        subset=["vec_pred", "theta_pred", "y"]
    )
    print(f"held-out rows {FIRST_HOLDOUT}+: {len(e):,}")

    def nll(pred, var):
        return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (e.y - pred) ** 2 / var))

    mae_s = float((e.theta_pred - e.y).abs().mean())
    mae_v = float((e.vec_pred - e.y).abs().mean())
    print(f"\n{'model':>18s} {'MAE':>8s} {'NLL':>9s}")
    print(f"{'scalar filter':>18s} {mae_s:>8.4f} {nll(e.theta_pred, e.scal_var):>9.4f}")
    print(f"{'vector (sum of 5)':>18s} {mae_v:>8.4f} {nll(e.vec_pred, e.vec_var):>9.4f}")
    print(f"\nMAE delta (vector - scalar): {mae_v - mae_s:+.4f}")
    by = e.groupby("yr").apply(
        lambda x: pd.Series({
            "s": (x.theta_pred - x.y).abs().mean(),
            "v": (x.vec_pred - x.y).abs().mean(),
        }),
        include_groups=False,
    )
    print(f"seasons vector beats scalar on MAE: {int((by.v < by.s).sum())}/{len(by)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit", "evaluate"])
    a = ap.parse_args()
    if a.command == "fit":
        cmd_fit()
    else:
        cmd_evaluate()


if __name__ == "__main__":
    main()
