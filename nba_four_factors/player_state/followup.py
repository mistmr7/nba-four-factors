"""Follow-up analyses from the phase-2 review handoff.

Commands map to the handoff items:

    a2       refit on season-standardized observations; halflife robustness
    a1       covariance diagnostic for the vector NLL loss (no refit)
    a4       per-component profile likelihoods (resumable)
    bdiag    valuation correlation and top disagreements, incumbent vs filter
    brun     head-to-head arms A0-A3 on the availability walk-forward harness

Run from repo root:
    python -m nba_four_factors.player_state.followup <command>
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .filter import (
    OUT_DIR,
    fit_params,
    load_panel,
    neg_log_lik,
    obs_arrays,
    run_filter,
)

TRAIN_THROUGH = 2017
A4_LOG = OUT_DIR / "component_profiles.json"


def season_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score y within season. Each season's stats come from that season's
    qualifying observations; used only on the training block, where every
    season is training data, so no leakage arises."""
    out = df.copy()
    q = out["qualifying"]
    stats = (
        out[q]
        .groupby("season")["y"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "s_mean", "std": "s_sd"})
    )
    out = out.merge(stats, on="season", how="left")
    out["y"] = np.where(q, (out["y"] - out["s_mean"]) / out["s_sd"], np.nan)
    return out.drop(columns=["s_mean", "s_sd"])


def cmd_a2() -> None:
    df = load_panel()
    tr = df[df["yr"] <= TRAIN_THROUGH]
    z = season_standardize(tr)
    t0 = time.time()
    p = fit_params(z, TRAIN_THROUGH)
    print("season-standardized refit (train through 2017):")
    print(
        f"  halflife {p['halflife_days']:.0f}d  (raw-scale fit: 2039d)\n"
        f"  lam {p['lam']:.6f}  sigma2 {p['sigma2']:.6f}  c {p['c']:.4f}\n"
        f"  stationary sd {p['stationary_sd']:.4f} (in within-season z units)\n"
        f"  converged {p['converged']}  [{time.time() - t0:.0f}s]"
    )


def cmd_a1() -> None:
    from .filter import MIN_QUALIFYING_MINUTES, WINSOR
    from .vector import COMPONENT_WINSOR, VPARAMS, component_frame, panel_for

    df = component_frame()
    params = json.loads(VPARAMS.read_text())

    comps = list(COMPONENT_WINSOR)
    z_tr = {}
    pred = {}
    var = {}
    keys = None
    for comp in comps:
        panel = panel_for(df, comp)
        out = run_filter(panel, params[comp])
        out = out.merge(
            panel[["game_id", "person_id", "y", "qualifying", "yr"]],
            on=["game_id", "person_id"],
        )
        p = params[comp]
        v = out.P_pred + p["c"] / out.minutes.replace(0, np.nan)
        q = out.qualifying & (out.yr <= TRAIN_THROUGH)
        z_tr[comp] = ((out.y - out.theta_pred) / np.sqrt(v))[q].to_numpy()
        pred[comp] = out.theta_pred
        var[comp] = v
        if keys is None:
            keys = out[["game_id", "person_id", "minutes", "yr", "qualifying"]].copy()

    n = min(len(z) for z in z_tr.values())
    Z = np.column_stack([z_tr[c][:n] for c in comps])
    Z = Z[~np.isnan(Z).any(axis=1)]
    rho = np.corrcoef(Z.T)
    print("training-block innovation correlations:")
    print(pd.DataFrame(rho, index=comps, columns=comps).round(3).to_string())

    sd = {c: np.sqrt(var[c]) for c in comps}
    corr_var = sum(var[c] for c in comps)
    for i, ci in enumerate(comps):
        for j, cj in enumerate(comps):
            if i < j:
                corr_var = corr_var + 2.0 * rho[i, j] * sd[ci] * sd[cj]
    diag_var = sum(var[c] for c in comps)
    vec_pred = sum(pred[c] for c in comps)

    base = df[
        ["game_id", "person_id", "season", "yr", "minutes", "status", "game_score_per36"]
    ].copy()
    base["y"] = base.game_score_per36.clip(*WINSOR)
    qual = (base.status == "played") & (base.minutes >= MIN_QUALIFYING_MINUTES)
    base = base[qual].sort_values(["person_id", "yr"])
    base["n_prior"] = base.groupby("person_id").cumcount()
    keys = keys.assign(vec_pred=vec_pred, diag_var=diag_var, corr_var=corr_var)
    e = base.merge(
        keys[["game_id", "person_id", "vec_pred", "diag_var", "corr_var"]],
        on=["game_id", "person_id"],
    )
    scal = pd.read_parquet(
        OUT_DIR / "player_form_predictive.parquet",
        columns=["game_id", "person_id", "theta_pred", "P_pred"],
    )
    sp = json.loads((OUT_DIR / "params_scalar.json").read_text())
    e = e.merge(scal, on=["game_id", "person_id"])
    e["scal_var"] = e.P_pred + sp["c"] / e.minutes
    e = e[(e.yr >= 2018) & (e.n_prior >= 20)].dropna(
        subset=["vec_pred", "theta_pred", "y", "corr_var"]
    )

    def nll(pred_col, var_col):
        return float(
            np.mean(
                0.5 * np.log(2 * np.pi * e[var_col]) + 0.5 * (e.y - e[pred_col]) ** 2 / e[var_col]
            )
        )

    print(f"\nheld-out rows 2018+: {len(e):,}")
    print(f"  scalar               NLL {nll('theta_pred', 'scal_var'):.4f}")
    print(f"  vector diagonal      NLL {nll('vec_pred', 'diag_var'):.4f}")
    print(f"  vector corr-adjusted NLL {nll('vec_pred', 'corr_var'):.4f}")
    print(
        f"  mean variance: diagonal {e.diag_var.mean():.2f}  corrected {e.corr_var.mean():.2f}  "
        f"scalar {e.scal_var.mean():.2f}"
    )
    print(
        "\nverdict: corrected NLL below scalar means the covariance diagnosis is "
        "confirmed and a full-covariance fit is justified."
    )


def cmd_a4(time_budget: float | None) -> None:
    from scipy.optimize import minimize

    from .vector import COMPONENT_WINSOR, component_frame, panel_for

    t0 = time.time()
    df = component_frame()
    halflives = [30, 120, 480, 960, 2000, 4000, 8000]
    log = {}
    if A4_LOG.exists():
        log = json.loads(A4_LOG.read_text())
    for comp in COMPONENT_WINSOR:
        panel = panel_for(df, comp)
        tr = panel[panel["yr"] <= TRAIN_THROUGH]
        q = tr[tr["qualifying"]]
        mu = float(np.average(q["y"], weights=q["minutes"]))
        arrays = obs_arrays(tr)
        log.setdefault(comp, {})
        for h in halflives:
            key = str(h)
            if key in log[comp]:
                continue
            if time_budget is not None and time.time() - t0 > time_budget:
                A4_LOG.write_text(json.dumps(log, indent=1))
                print("time budget reached; re-invoke to continue", flush=True)
                return
            lam = np.log(2) / h

            def obj(lp, lam=lam, arrays=arrays, mu=mu):
                return neg_log_lik(np.concatenate([[np.log(lam)], lp]), *arrays, mu)

            r = minimize(obj, np.log([0.01, 100.0]), method="L-BFGS-B", options={"maxiter": 100})
            log[comp][key] = float(r.fun)
            A4_LOG.write_text(json.dumps(log, indent=1))
            print(f"{comp} halflife {h}d: NLL {r.fun:.1f} [{time.time() - t0:.0f}s]", flush=True)
    print("\nprofiles complete:")
    for comp, prof in log.items():
        hs = sorted(prof, key=lambda k: int(k))
        best = min(prof, key=prof.get)
        line = "  ".join(f"{h}:{prof[h] - prof[best]:.0f}" for h in hs)
        print(f"{comp:>11s} (delta-NLL vs best={best}d): {line}")


def shrunk_minutes() -> pd.DataFrame:
    """Expected minutes per player per date, mirroring the incumbent's value
    shrinkage exactly (alpha = n / (n + 12), prior-season mean fallback),
    applied to minutes instead of Game Score. This is the scale-resolution
    minutes assumption m_hat; it introduces no new minutes model, only the
    incumbent's own machinery pointed at a different column."""
    df = pd.read_parquet(
        OUT_DIR / "player_game_score.parquet",
        columns=["person_id", "season", "game_date", "minutes", "status"],
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    logs = df[(df.status == "played")].sort_values(["person_id", "season", "game_date"])
    smean = logs.groupby(["person_id", "season"])["minutes"].mean().reset_index()
    smean["next"] = smean["season"].apply(
        lambda s: f"{int(s[:4]) + 1}_{(int(s[:4]) + 1) % 100:02d}"
    )
    prior = smean[["person_id", "next", "minutes"]].rename(
        columns={"next": "season", "minutes": "prior_min"}
    )
    g = logs.groupby(["person_id", "season"])
    cnt = (g.cumcount() + 1).to_numpy()
    cum = g["minutes"].cumsum().to_numpy() / cnt
    tl = logs[["person_id", "season", "game_date"]].copy()
    tl = tl.merge(prior, on=["person_id", "season"], how="left")
    a = cnt / (cnt + 12.0)
    tl["m_hat"] = a * cum + (1 - a) * tl["prior_min"].fillna(0.0).to_numpy()
    return tl.sort_values("game_date"), prior


def filter_valuation() -> pd.DataFrame:
    """Per inactive-player-game valuation from the walk-forward filter:
    expected raw Game Score contribution theta_pred * m_hat / 36, plus the
    matching uncertainty scale sqrt(P_pred) * m_hat / 36."""
    wf = pd.read_parquet(
        OUT_DIR / "player_form_predictive_wf.parquet",
        columns=["game_id", "person_id", "game_date", "status", "theta_pred", "P_pred"],
    )
    wf = wf[wf.status == "inactive"].copy()
    wf["game_date"] = pd.to_datetime(wf["game_date"])
    tl, prior = shrunk_minutes()
    ina = pd.read_parquet(REPO_FEAT / "inactives.parquet")[
        ["game_id", "person_id", "is_home", "season"]
    ]
    ina["game_id"] = ina["game_id"].astype(str).str.zfill(10)
    wf = wf.merge(ina, on=["game_id", "person_id"], how="inner")
    wf = wf.sort_values("game_date")
    wf = pd.merge_asof(
        wf,
        tl[["person_id", "season", "game_date", "m_hat"]],
        by=["person_id", "season"],
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
    )
    wf = wf.merge(prior, on=["person_id", "season"], how="left")
    wf["m_hat"] = wf["m_hat"].fillna(wf["prior_min"]).fillna(0.0)
    wf["val"] = wf["theta_pred"] * wf["m_hat"] / 36.0
    wf["usd"] = np.sqrt(wf["P_pred"]) * wf["m_hat"] / 36.0
    return wf


REPO_FEAT = OUT_DIR.parent / "features"


def team_features() -> pd.DataFrame:
    """Per-game team-level features for every arm, on identical rows."""
    from ..features.availability_shrunk import player_gmsc, shrunk_diff

    inc = shrunk_diff(player_gmsc(), 12.0)
    inc["game_id"] = inc["game_id"].astype(str).str.zfill(10)

    wf = filter_valuation()
    agg_v = wf.groupby(["game_id", "is_home"])["val"].sum().unstack(fill_value=0.0)
    agg_u = wf.groupby(["game_id", "is_home"])["usd"].sum().unstack(fill_value=0.0)
    f = pd.DataFrame(index=agg_v.index)
    f["filt_diff"] = agg_v.get(True, 0.0) - agg_v.get(False, 0.0)
    f["u_tot"] = agg_u.get(True, 0.0) + agg_u.get(False, 0.0)
    f = f.reset_index()

    kal = pd.read_parquet(REPO_FEAT / "kalman_perfold_preds.parquet")
    kal["game_id"] = kal["game_id"].astype(str).str.zfill(10)
    lines = pd.read_parquet(REPO_FEAT / "vegas_game_lines.parquet")[
        ["game_id", "vegas_home_margin"]
    ].dropna()
    lines["game_id"] = lines["game_id"].astype(str).str.zfill(10)

    t = kal.merge(inc, on="game_id", how="left").merge(f, on="game_id", how="left")
    t = t.merge(lines, on="game_id", how="left")
    for c in ("gmscs_diff", "filt_diff", "u_tot"):
        t[c] = t[c].fillna(0.0)
    return t


def cmd_bdiag() -> None:
    wf = filter_valuation()
    from ..features.availability_shrunk import player_gmsc

    logs = player_gmsc()
    smean = logs.groupby(["person_id", "season"])["gmsc"].mean().reset_index()
    smean["next"] = smean["season"].apply(
        lambda s: f"{int(s[:4]) + 1}_{(int(s[:4]) + 1) % 100:02d}"
    )
    prior = smean[["person_id", "next", "gmsc"]].rename(columns={"next": "season", "gmsc": "prior"})
    g = logs.groupby(["person_id", "season"])
    cnt = (g.cumcount() + 1).to_numpy()
    cum = g["gmsc"].cumsum().to_numpy() / cnt
    tl = logs[["person_id", "season", "game_date"]].copy()
    tl["game_date"] = pd.to_datetime(tl["game_date"])
    tl = tl.merge(prior, on=["person_id", "season"], how="left")
    a = cnt / (cnt + 12.0)
    tl["inc_val"] = a * cum + (1 - a) * tl["prior"].fillna(0.0).to_numpy()
    tl = tl.sort_values("game_date")

    j = pd.merge_asof(
        wf.sort_values("game_date"),
        tl[["person_id", "season", "game_date", "inc_val"]],
        by=["person_id", "season"],
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
    )
    j = j.merge(prior, on=["person_id", "season"], how="left")
    j["inc_val"] = j["inc_val"].fillna(j["prior"]).fillna(0.0)
    j = j.dropna(subset=["val"])
    r = float(np.corrcoef(j.inc_val, j.val)[0, 1])
    print(
        f"valuation correlation, incumbent vs filter, {len(j):,} inactive "
        f"player-games: r = {r:.4f}"
    )
    j["dis"] = (j.val - j.inc_val).abs()
    names = pd.read_parquet(
        REPO_FEAT / "player_game_logs.parquet", columns=["person_id", "name"]
    ).drop_duplicates("person_id")
    top = j.nlargest(50, "dis").merge(names, on="person_id", how="left")
    print("\ntop disagreements (filter val vs incumbent val, raw GmSc scale):")
    print(
        top[["name", "game_date", "season", "inc_val", "val", "m_hat", "P_pred"]]
        .head(25)
        .round(2)
        .to_string(index=False)
    )


def _fit_beta(tr: pd.DataFrame, cols: list[str]) -> np.ndarray:
    A = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
    r = (tr.margin - tr.pred_m).to_numpy()
    return np.linalg.lstsq(A, r, rcond=None)[0]


def _apply(te: pd.DataFrame, coef: np.ndarray, cols: list[str]) -> np.ndarray:
    A = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
    return te.pred_m.to_numpy() + A @ coef


def cmd_brun() -> None:
    t = team_features()
    arms = {
        "A0 no feature": [],
        "A1 incumbent": ["gmscs_diff"],
        "A2 filter value": ["filt_diff"],
        "A3 filter + uncertainty": ["filt_diff", "u_tot"],
    }
    t["inter"] = t["filt_diff"] * t["u_tot"]
    arms["A3b value x uncertainty"] = ["filt_diff", "u_tot", "inter"]

    per_fold = {name: [] for name in arms}
    rows_per_fold = []
    for ty in range(2007, int(t.test_yr.max()) + 1):
        tr = t[(t.test_yr >= 2005) & (t.test_yr < ty)]
        te = t[(t.test_yr == ty)].dropna(subset=["vegas_home_margin"]).copy()
        if len(te) < 200:
            continue
        rows_per_fold.append(len(te))
        for name, cols in arms.items():
            coef = _fit_beta(tr, cols)
            pred = _apply(te, coef, cols)
            per_fold[name].append(float(np.abs(pred - te.margin).mean()))
        per_fold.setdefault("Vegas", []).append(
            float(np.abs(te.vegas_home_margin - te.margin).mean())
        )
    n_folds = len(rows_per_fold)
    print(f"folds: {n_folds} (2007-{2006 + n_folds}), rows/fold ~{int(np.mean(rows_per_fold))}")
    print(f"\n{'arm':>26s} {'mean MAE':>9s}")
    for name in [*arms, "Vegas"]:
        print(f"{name:>26s} {np.mean(per_fold[name]):>9.4f}")

    d = np.array(per_fold["A1 incumbent"]) - np.array(per_fold["A2 filter value"])
    from scipy import stats

    tstat, pval = stats.ttest_1samp(d, 0.0)
    print(
        f"\nA1 vs A2 paired across folds: mean delta {d.mean():+.4f} "
        f"(positive favors filter), t = {tstat:.2f}, p = {pval:.4f}"
    )
    print(f"folds filter better: {int((d > 0).sum())}/{len(d)}")

    te_all = t[(t.test_yr >= 2007)].dropna(subset=["vegas_home_margin"]).copy()
    a1_ae, a2_ae = [], []
    for ty in range(2007, int(t.test_yr.max()) + 1):
        tr = t[(t.test_yr >= 2005) & (t.test_yr < ty)]
        te = te_all[te_all.test_yr == ty]
        if len(te) < 200:
            continue
        a1_ae.append(np.abs(_apply(te, _fit_beta(tr, ["gmscs_diff"]), ["gmscs_diff"]) - te.margin))
        a2_ae.append(np.abs(_apply(te, _fit_beta(tr, ["filt_diff"]), ["filt_diff"]) - te.margin))
    a1_ae = np.concatenate(a1_ae)
    a2_ae = np.concatenate(a2_ae)
    diff = a1_ae - a2_ae
    rng = np.random.default_rng(11)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(
        f"game-level paired bootstrap (n={len(diff):,}): delta MAE "
        f"{diff.mean():+.4f} [95% {lo:+.4f}, {hi:+.4f}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["a2", "a1", "a4", "bdiag", "brun"])
    ap.add_argument("--time-budget", type=float, default=None)
    a = ap.parse_args()
    if a.command == "a2":
        cmd_a2()
    elif a.command == "a1":
        cmd_a1()
    elif a.command == "a4":
        cmd_a4(a.time_budget)
    elif a.command == "bdiag":
        cmd_bdiag()
    elif a.command == "brun":
        cmd_brun()


if __name__ == "__main__":
    main()
