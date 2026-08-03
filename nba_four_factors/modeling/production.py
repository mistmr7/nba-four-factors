"""Production regression: Option B composite + seed-to-model blend, evaluated.

The per-game prediction is the precision-weighted blend of two models:

    pred = (1 - alpha(n)) * M0  +  alpha(n) * M_full,   alpha(n) = n / (n + k)

M0 is the Vegas-seed model (preseason win-total differential). M_full is the
four-factor model with schedule context and the shrunk recent-form term, where
the recent-form composite is Option B (the regression-weighted strength score
C = beta . factors, in points). n is the average games the two teams have
played; early season leans on the seed, late season on the four factors. Both a
margin head and a win head are blended. k is derived from the variance
components. Reported out of sample at the game level and the season level, with
and without the pace differential.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error

from .features import _team_game_frame

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
TABLE = REPO / "data" / "features" / "modeling_table.parquet"
VEGAS = REPO / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"

DIFF = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
CTX = ["rest_diff", "b2b_diff", "miles7d_diff"]
SINGLE = ["efg_d", "oreb_d", "tov_d", "ftmfga_d"]
W, MINP = 15, 5


def games_played() -> dict:
    """(game_id, team_id) -> team's game number within its season, 1-based.

    Keyed per game. A (season, team_id) key gets overwritten every row and
    ends up holding the season total, which makes alpha nearly constant.
    """
    df = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "*/regular_season.parquet")))],
        ignore_index=True,
    )
    df = df[~df["is_neutral"]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_id", "season", "game_date"])
    df["gn"] = df.groupby(["team_id", "season"]).cumcount() + 1
    return {
        (g, int(t)): int(n) for g, t, n in df[["game_id", "team_id", "gn"]].itertuples(index=False)
    }


def build(with_pace: bool):
    mt = pd.read_parquet(TABLE)
    # Option B composite (global beta; validated equal to per-fold within 0.001).
    beta = (
        LinearRegression()
        .fit(
            mt.dropna(subset=[*DIFF, "home_margin"])[DIFF],
            mt.dropna(subset=[*DIFF, "home_margin"]).home_margin,
        )
        .coef_
    )
    tg = _team_game_frame().copy()
    tg["C"] = tg[SINGLE].to_numpy() @ beta
    g = tg.sort_values(["team_id", "season", "game_date"]).groupby(["team_id", "season"])["C"]
    roll = pd.DataFrame(
        {
            "game_id": tg.sort_values(["team_id", "season", "game_date"])["game_id"].values,
            "is_home": tg.sort_values(["team_id", "season", "game_date"])["is_home"].values,
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
    gp = games_played()
    mt["ngn"] = [
        (gp.get((g, int(t)), np.nan) + gp.get((g, int(o)), np.nan)) / 2
        for g, t, o in zip(mt.game_id, mt.team_id, mt.away_team_id, strict=False)
    ]
    mt["full_feats"] = ",".join(DIFF + CTX)
    return mt, (CTX + (["pace_diff"] if with_pace else []))


def run(with_pace: bool):
    mt, ctx = build(with_pace)
    feats_full = DIFF + ctx + ["rfd"]
    seasons = sorted(mt.season.unique(), key=lambda s: int(s[:4]))
    rows = []
    for i in range(3, len(seasons)):
        tr = mt[mt.season.isin(set(seasons[:i]))].copy()
        te = mt[mt.season == seasons[i]].copy()
        # shrinkage + k from train
        fm = np.concatenate([tr.hfm.dropna(), tr.afm.dropna()])
        fv = np.concatenate([tr.hfv.dropna(), tr.afv.dropna()])
        n_eff = float(np.clip(np.nanmean(fv) / np.nanvar(fm), 1, W))
        sv = max(np.nanvar(fm) - np.nanmean(fv) / n_eff, 1e-6)
        sg = _team_game_frame()
        for f in (tr, te):
            wh = sv / (sv + f.hfv / n_eff)
            wa = sv / (sv + f.afv / n_eff)
            f["rfd"] = wh * (f.hfm - f.hb) - wa * (f.afm - f.ab)
        # k from variance components (per fold)
        beta_tr = (
            LinearRegression()
            .fit(
                tr.dropna(subset=[*DIFF, "home_margin"])[DIFF],
                tr.dropna(subset=[*DIFF, "home_margin"]).home_margin,
            )
            .coef_
        )
        s = sg[sg.season.isin(set(seasons[:i]))].copy()
        s["r"] = s[SINGLE].to_numpy() @ beta_tr
        sig_game = float(s.groupby(["season", "team_id"])["r"].var(ddof=1).mean())
        net = sg.groupby(["season", "team_id"])["margin"].mean()
        veg = pd.read_parquet(VEGAS)[["season", "team_id", "win_total"]].dropna()
        jj = veg[veg.season.isin(set(seasons[:i]))].merge(
            net.reset_index().rename(columns={"margin": "net"}), on=["season", "team_id"]
        )
        a_s, b_s = np.polyfit(jj.win_total, jj.net, 1)
        sig_prior = np.var(jj.net - (a_s * jj.win_total + b_s))
        k = float(np.clip(sig_game / sig_prior, 1, 60))

        # M0 (seed) on wintotal_diff
        tr0 = tr.dropna(subset=["wintotal_diff", "home_margin", "home_win"])
        m0_lin = LinearRegression().fit(tr0[["wintotal_diff"]], tr0.home_margin)
        m0_clf = LogisticRegression(max_iter=1000).fit(tr0[["wintotal_diff"]], tr0.home_win)
        # M_full
        trf = tr.dropna(subset=[*feats_full, "home_margin", "home_win"])
        mf_lin = LinearRegression().fit(trf[feats_full], trf.home_margin)
        mf_clf = LogisticRegression(max_iter=1000).fit(trf[feats_full], trf.home_win)

        t = te.dropna(subset=["wintotal_diff", "home_margin", "home_win", "ngn"]).copy()
        t["p0m"] = m0_lin.predict(t[["wintotal_diff"]])
        t["p0w"] = m0_clf.predict_proba(t[["wintotal_diff"]])[:, 1]
        has = t[feats_full].notna().all(axis=1)
        t["pfm"] = np.nan
        t["pfw"] = np.nan
        if has.any():
            t.loc[has, "pfm"] = mf_lin.predict(t.loc[has, feats_full])
            t.loc[has, "pfw"] = mf_clf.predict_proba(t.loc[has, feats_full])[:, 1]
        alpha = np.where(has, np.minimum(t.ngn / (t.ngn + k), 1.0), 0.0)
        t["bm"] = (1 - alpha) * t.p0m + alpha * t.pfm.fillna(t.p0m)
        t["bw"] = (1 - alpha) * t.p0w + alpha * t.pfw.fillna(t.p0w)
        rows.append(
            t[
                [
                    "season",
                    "team_id",
                    "away_team_id",
                    "home_margin",
                    "home_win",
                    "p0m",
                    "p0w",
                    "pfm",
                    "pfw",
                    "bm",
                    "bw",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def summarize(with_pace: bool, label: str):
    r = run(with_pace)
    out = {}
    # game-level (rows where full model exists, for a fair M0/full/blend compare)
    rr = r.dropna(subset=["pfm"])
    for nm, mcol, wcol in [
        ("M0 seed", "p0m", "p0w"),
        ("M_full", "pfm", "pfw"),
        ("Blended", "bm", "bw"),
    ]:
        out[nm] = (mean_absolute_error(rr.home_margin, rr[mcol]), log_loss(rr.home_win, rr[wcol]))
    # season expected wins (blended, all games incl early) vs Vegas
    home = r[["season", "team_id", "bw"]].rename(columns={"team_id": "tid", "bw": "p"})
    away = r[["season", "away_team_id", "bw"]].rename(columns={"away_team_id": "tid"})
    away["p"] = 1 - r.bw.values
    ew = pd.concat([home, away]).groupby(["season", "tid"]).p.sum()
    sg = _team_game_frame()
    aw = sg.assign(w=(sg.margin > 0).astype(int)).groupby(["season", "team_id"]).w.sum()
    veg = (
        pd.read_parquet(VEGAS)[["season", "team_id", "win_total"]]
        .dropna()
        .set_index(["season", "team_id"])
        .win_total
    )
    idx = ew.index.intersection(aw.index).intersection(veg.index)
    season_model = (ew.reindex(idx) - aw.reindex(idx)).abs().mean()
    season_vegas = (veg.reindex(idx) - aw.reindex(idx)).abs().mean()
    print(f"\n=== {label} ===")
    print(f"  game-level (n={len(rr):,}):")
    for nm, (mae, ll) in out.items():
        print(f"    {nm:9s} margin MAE {mae:.4f}   win logloss {ll:.4f}")
    print(
        f"  season expected-wins MAE: blended {season_model:.3f}  vs Vegas preseason {season_vegas:.3f}"
    )


def main():
    summarize(False, "Production model (no pace)")
    summarize(True, "Production model (+ pace)")


if __name__ == "__main__":
    main()
