"""Variant A test: reduced-model fallback vs the current seed-only gate.

The production blend forces alpha to 0 whenever the full feature vector
(four factors + context + shrunk recent form) is incomplete, which in
practice means roughly the first five games of each team's season are
predicted by the seed model alone.

This script tests the alternative: when recent form is missing but the
four-factor and context features exist, blend the seed with a reduced
model (factors + context, no recent form) using the same
alpha = n / (n + k) weight. The full model takes over as soon as its
features exist. Game 1 remains seed-only by necessity since every
feature is trailing.

Walk-forward, same folds and k derivation as modeling/production.py.
Reports margin MAE and win log loss overall and by games-played bucket,
on the identical row set for both strategies.

Run from repo root:
    python3 scripts/early_blend_fallback.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error

from nba_four_factors.modeling.features import _team_game_frame

REPO = Path(__file__).resolve().parents[1]

PROC = REPO / "data" / "processed"
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
VEGAS = REPO / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"

DIFF = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
CTX = ["rest_diff", "b2b_diff", "miles7d_diff"]
SINGLE = ["efg_d", "oreb_d", "tov_d", "ftmfga_d"]
W, MINP = 15, 5


def games_played() -> tuple[dict, dict]:
    """Return two maps.

    per_game: (game_id, team_id) -> team's game number at that game, 1-based.
    This is the correct count for the blend weight.

    season_total: (season, team_id) -> team's total games that season.
    This replicates the production.py bug, where the dict comprehension keyed
    by (season, team_id) is overwritten each row and keeps only the last
    game number. Included so the as-coded production behavior can be scored.
    """
    df = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "*/regular_season.parquet")))],
        ignore_index=True,
    )
    df = df[~df["is_neutral"]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_id", "season", "game_date"])
    df["gn"] = df.groupby(["team_id", "season"]).cumcount() + 1
    per_game = {
        (g, int(t)): int(n) for g, t, n in df[["game_id", "team_id", "gn"]].itertuples(index=False)
    }
    season_total = {
        (s, int(t)): int(n) for s, t, n in df[["season", "team_id", "gn"]].itertuples(index=False)
    }
    return per_game, season_total


def build():
    mt = pd.read_parquet(TABLE)
    tg = _team_game_frame().copy()
    fit = mt.dropna(subset=[*DIFF, "home_margin"])
    beta = LinearRegression().fit(fit[DIFF], fit.home_margin).coef_
    tg["C"] = tg[SINGLE].to_numpy() @ beta
    tg = tg.sort_values(["team_id", "season", "game_date"])
    g = tg.groupby(["team_id", "season"])["C"]
    roll = pd.DataFrame(
        {
            "game_id": tg["game_id"].values,
            "is_home": tg["is_home"].values,
            "base": g.transform(lambda s: s.shift(1).expanding().mean()).values,
            "fm": g.transform(lambda s: s.shift(1).rolling(W, min_periods=MINP).mean()).values,
            "fv": g.transform(lambda s: s.shift(1).rolling(W, min_periods=MINP).var(ddof=1)).values,
        }
    )
    h = roll[roll.is_home].rename(columns={"base": "hb", "fm": "hfm", "fv": "hfv"})[
        ["game_id", "hb", "hfm", "hfv"]
    ]
    a = roll[~roll.is_home].rename(columns={"base": "ab", "fm": "afm", "fv": "afv"})[
        ["game_id", "ab", "afm", "afv"]
    ]
    mt = mt.merge(h, on="game_id", how="left").merge(a, on="game_id", how="left")
    per_game, season_total = games_played()
    mt["ngn"] = [
        (per_game.get((g, int(t)), np.nan) + per_game.get((g, int(o)), np.nan)) / 2
        for g, t, o in zip(mt.game_id, mt.team_id, mt.away_team_id, strict=False)
    ]
    mt["ngn_bug"] = [
        (season_total.get((s, int(t)), np.nan) + season_total.get((s, int(o)), np.nan)) / 2
        for s, t, o in zip(mt.season, mt.team_id, mt.away_team_id, strict=False)
    ]
    return mt, tg


def run():
    mt, sg = build()
    feats_full = DIFF + CTX + ["rfd"]
    feats_red = DIFF + CTX
    seasons = sorted(mt.season.unique(), key=lambda s: int(s[:4]))
    veg_all = pd.read_parquet(VEGAS)[["season", "team_id", "win_total"]].dropna()
    rows = []
    for i in range(3, len(seasons)):
        train_set = set(seasons[:i])
        tr = mt[mt.season.isin(train_set)].copy()
        te = mt[mt.season == seasons[i]].copy()

        fm = np.concatenate([tr.hfm.dropna(), tr.afm.dropna()])
        fv = np.concatenate([tr.hfv.dropna(), tr.afv.dropna()])
        n_eff = float(np.clip(np.nanmean(fv) / np.nanvar(fm), 1, W))
        sv = max(np.nanvar(fm) - np.nanmean(fv) / n_eff, 1e-6)
        for f in (tr, te):
            wh = sv / (sv + f.hfv / n_eff)
            wa = sv / (sv + f.afv / n_eff)
            f["rfd"] = wh * (f.hfm - f.hb) - wa * (f.afm - f.ab)

        fit = tr.dropna(subset=[*DIFF, "home_margin"])
        beta_tr = LinearRegression().fit(fit[DIFF], fit.home_margin).coef_
        s = sg[sg.season.isin(train_set)].copy()
        s["r"] = s[SINGLE].to_numpy() @ beta_tr
        sig_game = float(s.groupby(["season", "team_id"])["r"].var(ddof=1).mean())
        net = sg.groupby(["season", "team_id"])["margin"].mean()
        jj = veg_all[veg_all.season.isin(train_set)].merge(
            net.reset_index().rename(columns={"margin": "net"}), on=["season", "team_id"]
        )
        a_s, b_s = np.polyfit(jj.win_total, jj.net, 1)
        sig_prior = np.var(jj.net - (a_s * jj.win_total + b_s))
        k = float(np.clip(sig_game / sig_prior, 1, 60))

        tr0 = tr.dropna(subset=["wintotal_diff", "home_margin", "home_win"])
        m0_lin = LinearRegression().fit(tr0[["wintotal_diff"]], tr0.home_margin)
        m0_clf = LogisticRegression(max_iter=1000).fit(tr0[["wintotal_diff"]], tr0.home_win)
        trf = tr.dropna(subset=[*feats_full, "home_margin", "home_win"])
        mf_lin = LinearRegression().fit(trf[feats_full], trf.home_margin)
        mf_clf = LogisticRegression(max_iter=1000).fit(trf[feats_full], trf.home_win)
        trr = tr.dropna(subset=[*feats_red, "home_margin", "home_win"])
        mr_lin = LinearRegression().fit(trr[feats_red], trr.home_margin)
        mr_clf = LogisticRegression(max_iter=1000).fit(trr[feats_red], trr.home_win)

        t = te.dropna(subset=["wintotal_diff", "home_margin", "home_win", "ngn", "ngn_bug"]).copy()
        t["p0m"] = m0_lin.predict(t[["wintotal_diff"]])
        t["p0w"] = m0_clf.predict_proba(t[["wintotal_diff"]])[:, 1]
        has_full = t[feats_full].notna().all(axis=1)
        has_red = t[feats_red].notna().all(axis=1)
        t["pfm"] = np.nan
        t["pfw"] = np.nan
        t["prm"] = np.nan
        t["prw"] = np.nan
        if has_full.any():
            t.loc[has_full, "pfm"] = mf_lin.predict(t.loc[has_full, feats_full])
            t.loc[has_full, "pfw"] = mf_clf.predict_proba(t.loc[has_full, feats_full])[:, 1]
        if has_red.any():
            t.loc[has_red, "prm"] = mr_lin.predict(t.loc[has_red, feats_red])
            t.loc[has_red, "prw"] = mr_clf.predict_proba(t.loc[has_red, feats_red])[:, 1]

        alpha_raw = np.minimum(t.ngn / (t.ngn + k), 1.0)
        alpha_bug = np.minimum(t.ngn_bug / (t.ngn_bug + k), 1.0)

        alpha_p = np.where(has_full, alpha_bug, 0.0)
        t["prod_m"] = (1 - alpha_p) * t.p0m + alpha_p * t.pfm.fillna(t.p0m)
        t["prod_w"] = (1 - alpha_p) * t.p0w + alpha_p * t.pfw.fillna(t.p0w)

        alpha_g = np.where(has_full, alpha_raw, 0.0)
        t["gated_m"] = (1 - alpha_g) * t.p0m + alpha_g * t.pfm.fillna(t.p0m)
        t["gated_w"] = (1 - alpha_g) * t.p0w + alpha_g * t.pfw.fillna(t.p0w)

        model_m = t.pfm.where(has_full, t.prm)
        model_w = t.pfw.where(has_full, t.prw)
        has_any = has_full | has_red
        alpha_f = np.where(has_any, alpha_raw, 0.0)
        t["fb_m"] = (1 - alpha_f) * t.p0m + alpha_f * model_m.fillna(t.p0m)
        t["fb_w"] = (1 - alpha_f) * t.p0w + alpha_f * model_w.fillna(t.p0w)

        t["regime"] = np.where(has_full, "full", np.where(has_red, "reduced", "seed_only"))
        rows.append(
            t[
                [
                    "season",
                    "ngn",
                    "regime",
                    "home_margin",
                    "home_win",
                    "p0m",
                    "p0w",
                    "pfm",
                    "pfw",
                    "prod_m",
                    "prod_w",
                    "gated_m",
                    "gated_w",
                    "fb_m",
                    "fb_w",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def report(r: pd.DataFrame):
    print(f"rows evaluated: {len(r):,} across {r.season.nunique()} test seasons")
    print("\nregime counts (share of games):")
    print((r.regime.value_counts(normalize=True) * 100).round(2).to_string())

    def line(name, mcol, wcol, d):
        mae = mean_absolute_error(d.home_margin, d[mcol])
        ll = log_loss(d.home_win, d[wcol], labels=[0, 1])
        print(f"    {name:22s} margin MAE {mae:.4f}   win logloss {ll:.4f}   n={len(d):,}")

    arms = [
        ("seed only", "p0m", "p0w"),
        ("prod as-coded (bug)", "prod_m", "prod_w"),
        ("gated, fixed alpha", "gated_m", "gated_w"),
        ("fallback (variant A)", "fb_m", "fb_w"),
    ]

    print("\noverall (identical rows):")
    for name, mcol, wcol in arms:
        line(name, mcol, wcol, r)

    bins = [(0, 3), (3, 5), (5, 10), (10, 15), (15, 100)]
    for lo, hi in bins:
        d = r[(r.ngn > lo) & (r.ngn <= hi)]
        if len(d) == 0:
            continue
        print(f"\ngames with avg games played in ({lo}, {hi}]:")
        for name, mcol, wcol in arms:
            line(name, mcol, wcol, d)

    diff_rows = r[r.regime == "reduced"]
    if len(diff_rows):
        print(f"\nrows where gated and fallback differ (regime = reduced, n={len(diff_rows):,}):")
        for name, mcol, wcol in arms:
            line(name, mcol, wcol, diff_rows)


def main():
    r = run()
    report(r)
    out = REPO / "data" / "features" / "early_blend_fallback_results.csv"
    r.to_csv(out, index=False)
    print(f"\nrow-level results written to {out}")


if __name__ == "__main__":
    main()
