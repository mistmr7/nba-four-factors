"""Evidence runs for the Table 1 design decisions lacking Results support.

Five experiments, all walk-forward over the ladder folds (seasons sorted,
first three burn-in), each on a fixed feature-complete row set so arms are
compared on identical rows within an experiment:

    ft      fourth-factor definition: FTM/FGA vs FTA/FGA vs both (M1 form)
    form    recent form: none (M2) vs raw deviation vs shrunk deviation (M3)
    pace    pace differential as a context feature: M2 vs M2 + pace_diff
    era     era handling on M3: pooled vs trailing-10-season window vs
            recency-weighted (5-season half-life) vs post-2013 interaction
    decay   prior decay in the production blend: alpha = n/(n+k) vs faster
            (k/2), slower (2k), and a divergence-triggered discount

Run from repo root:
    python3 scripts/design_gap_experiments.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "data" / "features"
W = 15

FACTORS_FTM = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
FACTORS_FTA = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftr_d"]
CTX = ["rest_diff", "b2b_diff", "miles7d_diff"]
FORM = ["form_mean", "base_comp", "form_var", "away_form_mean", "away_base_comp", "away_form_var"]


def _fit(X, y, w=None):
    X1 = np.column_stack([np.ones(len(X)), X])
    if w is not None:
        sw = np.sqrt(w)
        X1, y = X1 * sw[:, None], y * sw
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def _pred(c, X):
    return np.column_stack([np.ones(len(X)), X]) @ c


def load():
    mt = pd.read_parquet(FEAT / "modeling_table.parquet")
    mt["yr"] = mt["season"].str[:4].astype(int)
    return mt


def folds(mt):
    seasons = sorted(mt.season.unique(), key=lambda s: int(s[:4]))
    for i in range(3, len(seasons)):
        yield set(seasons[:i]), seasons[i]


def report(name, arms, base):
    print(f"\n=== {name} ===")
    seasons_won = {a: 0 for a in arms}
    n = len(next(iter(arms.values())))
    for a, maes in arms.items():
        for j in range(n):
            if a != base and maes[j] < arms[base][j]:
                seasons_won[a] += 1
    for a, maes in arms.items():
        extra = "" if a == base else f"   beats {base}: {seasons_won[a]}/{n}"
        print(f"  {a:38s} MAE {np.mean(maes):.4f}{extra}")


def exp_ft(mt):
    cols = list(set(FACTORS_FTM + FACTORS_FTA))
    d = mt.dropna(subset=[*cols, *CTX, *FORM, "home_margin"])
    arms = {"FTM/FGA (chosen)": [], "FTA/FGA": [], "both together": []}
    for tr_set, te_s in folds(d):
        tr, te = d[d.season.isin(tr_set)], d[d.season == te_s]
        for name, X in [
            ("FTM/FGA (chosen)", FACTORS_FTM),
            ("FTA/FGA", FACTORS_FTA),
            ("both together", cols),
        ]:
            c = _fit(tr[X].to_numpy(), tr.home_margin.to_numpy())
            arms[name].append(float(np.abs(_pred(c, te[X].to_numpy()) - te.home_margin).mean()))
    report("Fourth-factor definition (M1 form)", arms, "FTM/FGA (chosen)")


def add_form(tr, te):
    fm = np.concatenate([tr.form_mean.dropna(), tr.away_form_mean.dropna()])
    fv = np.concatenate([tr.form_var.dropna(), tr.away_form_var.dropna()])
    n_eff = float(np.clip(np.nanmean(fv) / np.nanvar(fm), 1, W))
    sv = max(np.nanvar(fm) - np.nanmean(fv) / n_eff, 1e-6)
    out = []
    for f in (tr, te):
        wh = sv / (sv + f.form_var / n_eff)
        wa = sv / (sv + f.away_form_var / n_eff)
        shrunk = wh * (f.form_mean - f.base_comp) - wa * (f.away_form_mean - f.away_base_comp)
        raw = (f.form_mean - f.base_comp) - (f.away_form_mean - f.away_base_comp)
        out.append((shrunk.to_numpy(), raw.to_numpy()))
    return out


def exp_form(mt):
    d = mt.dropna(subset=[*FACTORS_FTM, *CTX, *FORM, "home_margin"])
    base_cols = FACTORS_FTM + CTX
    arms = {"no form (M2)": [], "raw deviation": [], "shrunk deviation (M3)": []}
    for tr_set, te_s in folds(d):
        tr, te = d[d.season.isin(tr_set)], d[d.season == te_s]
        (tr_s, tr_r), (te_s_, te_r) = add_form(tr, te)
        Xb_tr, Xb_te = tr[base_cols].to_numpy(), te[base_cols].to_numpy()
        y_tr, y_te = tr.home_margin.to_numpy(), te.home_margin.to_numpy()
        c = _fit(Xb_tr, y_tr)
        arms["no form (M2)"].append(float(np.abs(_pred(c, Xb_te) - y_te).mean()))
        for name, ftr, fte in [
            ("raw deviation", tr_r, te_r),
            ("shrunk deviation (M3)", tr_s, te_s_),
        ]:
            c = _fit(np.column_stack([Xb_tr, ftr]), y_tr)
            arms[name].append(float(np.abs(_pred(c, np.column_stack([Xb_te, fte])) - y_te).mean()))
    report("Recent-form reliability", arms, "shrunk deviation (M3)")


def exp_pace(mt):
    d = mt.dropna(subset=[*FACTORS_FTM, *CTX, *FORM, "pace_diff", "home_margin"])
    arms = {"without pace (M2, chosen)": [], "with pace_diff": []}
    for tr_set, te_s in folds(d):
        tr, te = d[d.season.isin(tr_set)], d[d.season == te_s]
        for name, X in [
            ("without pace (M2, chosen)", FACTORS_FTM + CTX),
            ("with pace_diff", FACTORS_FTM + CTX + ["pace_diff"]),
        ]:
            c = _fit(tr[X].to_numpy(), tr.home_margin.to_numpy())
            arms[name].append(float(np.abs(_pred(c, te[X].to_numpy()) - te.home_margin).mean()))
    report("Pace differential as a feature", arms, "without pace (M2, chosen)")


def exp_era(mt):
    d = mt.dropna(subset=[*FACTORS_FTM, *CTX, *FORM, "home_margin"])
    cols = FACTORS_FTM + CTX
    arms = {
        "pooled (chosen)": [],
        "trailing 10 seasons": [],
        "recency-weighted (5-season half-life)": [],
        "post-2013 interaction": [],
    }
    for tr_set, te_s in folds(d):
        tr, te = d[d.season.isin(tr_set)], d[d.season == te_s]
        (tr_s, _), (te_s_, _) = add_form(tr, te)
        X_tr = np.column_stack([tr[cols].to_numpy(), tr_s])
        X_te = np.column_stack([te[cols].to_numpy(), te_s_])
        y_tr, y_te = tr.home_margin.to_numpy(), te.home_margin.to_numpy()

        c = _fit(X_tr, y_tr)
        arms["pooled (chosen)"].append(float(np.abs(_pred(c, X_te) - y_te).mean()))

        recent = sorted({int(s[:4]) for s in tr_set})[-10:]
        m10 = tr.yr.isin(recent).to_numpy()
        c = _fit(X_tr[m10], y_tr[m10])
        arms["trailing 10 seasons"].append(float(np.abs(_pred(c, X_te) - y_te).mean()))

        te_yr = int(te_s[:4])
        wts = 0.5 ** ((te_yr - tr.yr.to_numpy()) / 5.0)
        c = _fit(X_tr, y_tr, w=wts)
        arms["recency-weighted (5-season half-life)"].append(
            float(np.abs(_pred(c, X_te) - y_te).mean())
        )

        mod_tr = (tr.yr >= 2013).to_numpy().astype(float)[:, None]
        mod_te = (te.yr >= 2013).to_numpy().astype(float)[:, None]
        Xi_tr = np.column_stack([X_tr, mod_tr * tr[FACTORS_FTM].to_numpy()])
        Xi_te = np.column_stack([X_te, mod_te * te[FACTORS_FTM].to_numpy()])
        c = _fit(Xi_tr, y_tr)
        arms["post-2013 interaction"].append(float(np.abs(_pred(c, Xi_te) - y_te).mean()))
    report("Era handling (M3 features)", arms, "pooled (chosen)")


def games_played() -> dict:
    """(game_id, team_id) -> the team's game number that night, from the
    processed store. Computed directly so the decay arms cover the same 25
    folds as every other experiment rather than inheriting the RNN
    universe's window."""
    df = pd.concat(
        [
            pd.read_parquet(f, columns=["game_id", "team_id", "season", "game_date", "is_neutral"])
            for f in sorted(
                glob.glob(str(REPO / "data" / "processed" / "*" / "regular_season.parquet"))
            )
        ],
        ignore_index=True,
    )
    df = df[~df.is_neutral].copy()
    df["game_date"] = pd.to_datetime(df.game_date)
    df = df.sort_values(["team_id", "season", "game_date"])
    df["gn"] = df.groupby(["team_id", "season"]).cumcount() + 1
    return {
        (g, int(t)): int(n) for g, t, n in df[["game_id", "team_id", "gn"]].itertuples(index=False)
    }


def exp_decay(mt):
    gp = games_played()
    d = mt.dropna(subset=[*FACTORS_FTM, *CTX, *FORM, "wintotal_diff", "home_margin"]).copy()
    d["ngames"] = [
        (gp.get((g, int(t)), np.nan) + gp.get((g, int(o)), np.nan)) / 2
        for g, t, o in zip(d.game_id, d.team_id, d.away_team_id, strict=False)
    ]
    d = d.dropna(subset=["ngames"])
    cols = FACTORS_FTM + CTX
    K = 7.0
    arms = {
        "alpha = n/(n+7) (chosen)": [],
        "faster forgetting (k=3.5)": [],
        "slower forgetting (k=14)": [],
        "divergence-triggered discount": [],
    }
    for tr_set, te_s in folds(d):
        tr, te = d[d.season.isin(tr_set)], d[d.season == te_s]
        (tr_s, _), (te_s_, _) = add_form(tr, te)
        X_tr = np.column_stack([tr[cols].to_numpy(), tr_s])
        X_te = np.column_stack([te[cols].to_numpy(), te_s_])
        y_tr, y_te = tr.home_margin.to_numpy(), te.home_margin.to_numpy()
        c = _fit(X_tr, y_tr)
        m3 = _pred(c, X_te)
        sc = _fit(tr.wintotal_diff.to_numpy()[:, None], y_tr)
        seed = _pred(sc, te.wintotal_diff.to_numpy()[:, None])
        n = te.ngames.to_numpy()
        for name, k in [
            ("alpha = n/(n+7) (chosen)", K),
            ("faster forgetting (k=3.5)", K / 2),
            ("slower forgetting (k=14)", K * 2),
        ]:
            al = n / (n + k)
            arms[name].append(float(np.abs((1 - al) * seed + al * m3 - y_te).mean()))
        al = n / (n + K)
        div = np.abs(m3 - seed) > 6.0
        al_adj = np.where(div, n / (n + K / 2), al)
        arms["divergence-triggered discount"].append(
            float(np.abs((1 - al_adj) * seed + al_adj * m3 - y_te).mean())
        )
    report("Prior decay in the production blend", arms, "alpha = n/(n+7) (chosen)")


def main():
    mt = load()
    exp_ft(mt)
    exp_form(mt)
    exp_pace(mt)
    exp_era(mt)
    exp_decay(mt)


if __name__ == "__main__":
    main()
