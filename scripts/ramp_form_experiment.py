"""Does the RNN's temporal ramp close the M3-to-RNN gap by itself?

Replaces M3's recent-form feature with a 20-game weighted mean of the same
per-game composite, using the attention profile's position weights (ramp
0.033 oldest to 0.106 most recent), and evaluates seed-blended on the master
table's identical rows so M3 and RNN-diff references are exact.

Arms (all M3 base features + one form deviation, seed-blended al=n/(n+7)):
    shrunk-15   production form (replication sanity check against master m3)
    flat-20     unweighted 20-game mean deviation
    ramp-20     attention-weighted 20-game mean deviation

Run from repo root:
    PYTHONPATH=. python3 scripts/ramp_form_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_four_factors.modeling.features import _add_composite, _team_game_frame
from scripts.design_gap_experiments import (
    CTX,
    FACTORS_FTM,
    FORM,
    _fit,
    _pred,
    add_form,
    games_played,
    load,
)

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "data" / "features"
K_BLEND = 7.0


def weighted_forms() -> pd.DataFrame:
    """Per (game_id, team_id): flat-20 and ramp-20 trailing composite means."""
    w = pd.read_csv(FEAT / "attention_profile.csv")["weight"].to_numpy()
    w = w / w.sum()  # oldest (pos 1) .. most recent (pos 20)

    tf = _add_composite(_team_game_frame())
    tf = tf.sort_values(["team_id", "season", "game_date"])
    rows = []
    for (_t, _s), g in tf.groupby(["team_id", "season"], sort=False):
        vals = g["composite"].to_numpy()
        gids = g["game_id"].to_numpy()
        for i in range(len(vals)):
            hist = vals[max(0, i - 20) : i]
            if len(hist) < 5:
                flat = ramp = np.nan
            else:
                ww = w[-len(hist) :]
                ramp = float((hist * (ww / ww.sum())).sum())
                flat = float(hist.mean())
            rows.append((gids[i], int(_t), flat, ramp))
    return pd.DataFrame(rows, columns=["game_id", "team_id", "flat20", "ramp20"])


def main() -> None:
    mt = load()
    wf = weighted_forms()
    home = wf.rename(columns={"flat20": "h_flat", "ramp20": "h_ramp"})
    away = wf.rename(columns={"team_id": "away_team_id", "flat20": "a_flat", "ramp20": "a_ramp"})
    mt = mt.merge(home, on=["game_id", "team_id"], how="left")
    mt = mt.merge(away, on=["game_id", "away_team_id"], how="left")

    d = mt.dropna(
        subset=[*FACTORS_FTM, *CTX, *FORM, "wintotal_diff", "h_flat", "a_flat", "home_margin"]
    ).copy()
    d["flat_dev"] = (d.h_flat - d.base_comp) - (d.a_flat - d.away_base_comp)
    d["ramp_dev"] = (d.h_ramp - d.base_comp) - (d.a_ramp - d.away_base_comp)

    gp = games_played()
    d["ngames"] = [
        (gp.get((g, int(t)), np.nan) + gp.get((g, int(o)), np.nan)) / 2
        for g, t, o in zip(d.game_id, d.team_id, d.away_team_id, strict=False)
    ]
    d = d.dropna(subset=["ngames"])

    master = pd.read_parquet(FEAT / "results_master_table.parquet")
    mrows = master.dropna(subset=["m0", "m3", "rnn_diff", "kal_m"])
    keep = set(mrows.game_id)
    cols = FACTORS_FTM + CTX

    arms = {"shrunk-15 (M3 replication)": [], "flat-20": [], "ramp-20 (RNN weights)": []}
    refs = {"master M3": [], "master RNN-diff": [], "master Kalman": []}
    seasons = sorted(d.season.unique(), key=lambda s: int(s[:4]))
    for i in range(3, len(seasons)):
        tr_set, te_s = set(seasons[:i]), seasons[i]
        tr, te = d[d.season.isin(tr_set)], d[d.season == te_s]
        te = te[te.game_id.isin(keep)]
        if te.empty:
            continue
        (tr_s, _), (te_s_, _) = add_form(tr, te)
        y_tr, y_te = tr.home_margin.to_numpy(), te.home_margin.to_numpy()
        sc = _fit(tr.wintotal_diff.to_numpy()[:, None], y_tr)
        seed = _pred(sc, te.wintotal_diff.to_numpy()[:, None])
        al = te.ngames.to_numpy() / (te.ngames.to_numpy() + K_BLEND)

        for name, ftr, fte in [
            ("shrunk-15 (M3 replication)", tr_s, te_s_),
            ("flat-20", tr.flat_dev.to_numpy(), te.flat_dev.to_numpy()),
            ("ramp-20 (RNN weights)", tr.ramp_dev.to_numpy(), te.ramp_dev.to_numpy()),
        ]:
            c = _fit(np.column_stack([tr[cols].to_numpy(), ftr]), y_tr)
            pred = _pred(c, np.column_stack([te[cols].to_numpy(), fte]))
            blended = (1 - al) * seed + al * pred
            arms[name].append(np.abs(blended - y_te))

        mm = mrows[mrows.game_id.isin(set(te.game_id))]
        refs["master M3"].append(np.abs(mm.m3 - mm.margin).to_numpy())
        refs["master RNN-diff"].append(np.abs(mm.rnn_diff - mm.margin).to_numpy())
        refs["master Kalman"].append(np.abs(mm.kal_m - mm.margin).to_numpy())

    print(f"rows evaluated: {sum(len(a) for a in arms['flat-20']):,}")
    for name, chunks in {**refs, **arms}.items():
        ae = np.concatenate(chunks)
        print(f"  {name:28s} MAE {ae.mean():.4f}")

    from scipy import stats

    sh = arms["shrunk-15 (M3 replication)"]
    rp = arms["ramp-20 (RNN weights)"]
    fold_d = np.array([s.mean() - r.mean() for s, r in zip(sh, rp, strict=True)])
    t, p = stats.ttest_1samp(fold_d, 0.0)
    print(
        f"\nramp-20 vs shrunk-15, paired by fold: delta {fold_d.mean():+.4f} "
        f"(positive favors ramp), t = {t:.2f}, p = {p:.4f}, "
        f"ramp better in {int((fold_d > 0).sum())}/{len(fold_d)} folds"
    )

    ramp_all = np.concatenate(rp)
    rnn_all = np.concatenate(refs["master RNN-diff"])
    if len(ramp_all) == len(rnn_all):
        diff = rnn_all - ramp_all
        rng = np.random.default_rng(7)
        boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(
            f"ramp-20 vs RNN-diff, game-level bootstrap: delta {diff.mean():+.4f} "
            f"(positive favors ramp) [95% CI {lo:+.4f}, {hi:+.4f}]"
        )


if __name__ == "__main__":
    main()
