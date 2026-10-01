"""Temporal-ramp transplant: does the RNN's recency weighting, moved into the
linear model, close the M3-to-RNN gap? Leak-free version.

Arms (all M3 base features + one recent-form deviation, seed-blended
al = n/(n+7), evaluated on the master table's identical rows):

    shrunk-15      production form (in-harness baseline)
    flat-20        unweighted 20-game mean deviation
    geo ramp       geometric-decay weighted 20-game mean; halflife selected
                   per fold on training data only (fit on train minus its
                   last season, scored on that last season), then refit on
                   the full training pool -- no information from the test
                   season or from any trained network touches the arm
    attention ramp the attention variant's learned lag profile applied as
                   fixed weights (descriptive reference; profile provenance
                   noted in the write-up)

Stage 1 (first run) caches the weighted form columns to
data/features/ramp_transplant_forms.parquet; stage 2 evaluates.

Run from repo root (twice if the first call hits the time budget):
    PYTHONPATH=. python3 scripts/ramp_transplant.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

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
CACHE = FEAT / "ramp_transplant_forms.parquet"
K_BLEND = 7.0
HGRID = [3, 5, 8, 12, 20, 40]
L = 20


def build_forms() -> pd.DataFrame:
    from nba_four_factors.modeling.features import _add_composite, _team_game_frame

    aw = pd.read_csv(FEAT / "attention_profile.csv")["weight"].to_numpy()
    aw = aw / aw.sum()  # index 0 = oldest of 20, index 19 = most recent
    geo = {h: 0.5 ** (np.arange(L - 1, -1, -1) / h) for h in HGRID}

    tf = _add_composite(_team_game_frame())
    tf = tf.sort_values(["team_id", "season", "game_date"])
    cols = {f"geo{h}": [] for h in HGRID}
    cols.update({"flat20": [], "attn20": [], "game_id": [], "team_id": []})
    for (_t, _s), g in tf.groupby(["team_id", "season"], sort=False):
        vals = g["composite"].to_numpy()
        gids = g["game_id"].to_numpy()
        n = len(vals)
        for i in range(n):
            hist = vals[max(0, i - L) : i]
            cols["game_id"].append(gids[i])
            cols["team_id"].append(int(_t))
            if len(hist) < 5:
                cols["flat20"].append(np.nan)
                cols["attn20"].append(np.nan)
                for h in HGRID:
                    cols[f"geo{h}"].append(np.nan)
                continue
            m = len(hist)
            cols["flat20"].append(float(hist.mean()))
            wa = aw[-m:]
            cols["attn20"].append(float((hist * (wa / wa.sum())).sum()))
            for h in HGRID:
                wg = geo[h][-m:]
                cols[f"geo{h}"].append(float((hist * (wg / wg.sum())).sum()))
    out = pd.DataFrame(cols)
    out.to_parquet(CACHE, index=False)
    return out


def main() -> None:
    t0 = time.time()
    if CACHE.exists():
        wf = pd.read_parquet(CACHE)
    else:
        wf = build_forms()
        print(f"forms cached [{time.time() - t0:.0f}s]", flush=True)

    mt = load()
    fcols = ["flat20", "attn20"] + [f"geo{h}" for h in HGRID]
    home = wf.rename(columns={c: f"h_{c}" for c in fcols})
    away = wf.rename(columns={"team_id": "away_team_id", **{c: f"a_{c}" for c in fcols}})
    mt = mt.merge(home, on=["game_id", "team_id"], how="left")
    mt = mt.merge(away, on=["game_id", "away_team_id"], how="left")

    d = mt.dropna(
        subset=[*FACTORS_FTM, *CTX, *FORM, "wintotal_diff", "h_flat20", "a_flat20", "home_margin"]
    ).copy()
    for c in fcols:
        d[f"dev_{c}"] = (d[f"h_{c}"] - d.base_comp) - (d[f"a_{c}"] - d.away_base_comp)

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

    def run_arm(tr, te, ftr, fte, seed_tr, seed_te, al):
        c = _fit(np.column_stack([tr[cols].to_numpy(), ftr]), tr.home_margin.to_numpy())
        pred = _pred(c, np.column_stack([te[cols].to_numpy(), fte]))
        return (1 - al) * seed_te + al * pred

    arms = {
        "shrunk-15": [],
        "flat-20": [],
        "geo ramp (per-fold h)": [],
        "attention ramp (fixed)": [],
    }
    refs = {"master M3": [], "master RNN-diff": [], "master Kalman": []}
    chosen_h = []
    seasons = sorted(d.season.unique(), key=lambda s: int(s[:4]))
    for i in range(3, len(seasons)):
        tr_set, te_s = seasons[:i], seasons[i]
        tr = d[d.season.isin(set(tr_set))]
        te = d[d.season == te_s]
        te = te[te.game_id.isin(keep)]
        if te.empty:
            continue
        y_tr = tr.home_margin.to_numpy()
        sc = _fit(tr.wintotal_diff.to_numpy()[:, None], y_tr)
        seed_te = _pred(sc, te.wintotal_diff.to_numpy()[:, None])
        al = te.ngames.to_numpy() / (te.ngames.to_numpy() + K_BLEND)

        # Halflife selection inside the training pool only.
        val_s = tr_set[-1]
        tr_in = tr[tr.season != val_s]
        va = tr[tr.season == val_s]
        sc_in = _fit(tr_in.wintotal_diff.to_numpy()[:, None], tr_in.home_margin.to_numpy())
        seed_va = _pred(sc_in, va.wintotal_diff.to_numpy()[:, None])
        al_va = va.ngames.to_numpy() / (va.ngames.to_numpy() + K_BLEND)
        best_h, best_mae = None, np.inf
        for h in HGRID:
            pv = run_arm(
                tr_in,
                va,
                tr_in[f"dev_geo{h}"].to_numpy(),
                va[f"dev_geo{h}"].to_numpy(),
                None,
                seed_va,
                al_va,
            )
            mae = float(np.abs(pv - va.home_margin).mean())
            if mae < best_mae:
                best_mae, best_h = mae, h
        chosen_h.append(best_h)

        (tr_sh, _), (te_sh, _) = add_form(tr, te)
        specs = [
            ("shrunk-15", tr_sh, te_sh),
            ("flat-20", tr.dev_flat20.to_numpy(), te.dev_flat20.to_numpy()),
            (
                "geo ramp (per-fold h)",
                tr[f"dev_geo{best_h}"].to_numpy(),
                te[f"dev_geo{best_h}"].to_numpy(),
            ),
            ("attention ramp (fixed)", tr.dev_attn20.to_numpy(), te.dev_attn20.to_numpy()),
        ]
        for name, ftr, fte in specs:
            pred = run_arm(tr, te, ftr, fte, None, seed_te, al)
            arms[name].append(np.abs(pred - te.home_margin.to_numpy()))

        mm = mrows[mrows.game_id.isin(set(te.game_id))]
        for rname, col in [
            ("master M3", "m3"),
            ("master RNN-diff", "rnn_diff"),
            ("master Kalman", "kal_m"),
        ]:
            refs[rname].append(np.abs(mm[col] - mm.margin).to_numpy())

    print(f"rows evaluated: {sum(len(a) for a in arms['flat-20']):,}")
    for name, chunks in {**refs, **arms}.items():
        print(f"  {name:26s} MAE {np.concatenate(chunks).mean():.4f}")
    print(f"\nchosen halflife by fold: {chosen_h}")

    from scipy import stats

    for a, b in [("geo ramp (per-fold h)", "shrunk-15"), ("attention ramp (fixed)", "shrunk-15")]:
        fd = np.array([y.mean() - x.mean() for x, y in zip(arms[a], arms[b], strict=True)])
        t, p = stats.ttest_1samp(fd, 0.0)
        print(
            f"{a} vs {b}: fold delta {fd.mean():+.4f} (positive favors first), "
            f"t = {t:.2f}, p = {p:.4f}, better in {int((fd > 0).sum())}/{len(fd)}"
        )
    for a in ["geo ramp (per-fold h)", "attention ramp (fixed)"]:
        arr = np.concatenate(arms[a])
        rnn = np.concatenate(refs["master RNN-diff"])
        diff = rnn - arr
        rng = np.random.default_rng(7)
        boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(
            f"{a} vs RNN-diff: game-level delta {diff.mean():+.4f} "
            f"(positive favors ramp) [95% CI {lo:+.4f}, {hi:+.4f}]"
        )


if __name__ == "__main__":
    main()
