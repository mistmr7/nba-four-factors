"""Shot-quality model: as-of P(make) for every shot in the possession table.

Credit engine Phase 2, brick one (Credit_Engine_Design.md). League-level
on purpose: Oliver's credit division anchors on what an average player
does with the same chance; shooter identity enters later as a separate
layer, mirroring the Thunder-project architecture.

Features, all from the possession table (no tracking, by construction):
geometry (distance, absolute angle, behind-backboard), shot subtype
grouped to a stable cross-era vocabulary, seconds into the possession
(transition proxy), chance index > 1 (putback flag), period, and a
three-point indicator. Training is walk-forward: a trailing ten-season
window ending the season before the scored one, so every probability
is as-of and era-aware.

Run from repo root:
    python -m nba_four_factors.player_state.shot_quality score 2025_26
    python -m nba_four_factors.player_state.shot_quality score all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
POSS = REPO / "data" / "player_state" / "possessions"
OUT_DIR = REPO / "data" / "player_state" / "shot_quality"


TRAIL = int(os.environ.get("SQ_TRAIL", "10"))

# Era-stable vocabulary: the feed's subtype list expands ~2017-18 from
# about nine values to 48. Groups below exist in BOTH eras so a trailing
# training window never meets unlearned categories (the 2017-18
# calibration break this replaces is documented in the design doc).
SUB_GROUPS = {
    "dunk": ["Dunk"],
    "tip": ["Tip"],
    "layup": ["Layup", "Finger Roll", "Driving"],
    "hook": ["Hook"],
    "jumper": ["Jump", "Fadeaway", "Turnaround", "Pullup", "Pull-Up", "Step Back", "Floating"],
}


def _sub_group(sub: str) -> str:
    for g, keys in SUB_GROUPS.items():
        for k in keys:
            if k in sub:
                return g
    return "other"


def extract_shots(season: str) -> pd.DataFrame:
    pos = pd.read_parquet(POSS / f"{season}_regular_season.parquet")
    rows = []
    for r in pos.itertuples(index=False):
        events = json.loads(r.events)
        t0 = float(r.start_elapsed)
        for e in events:
            if e["t"] != "shot":
                continue
            x = e.get("x")
            y = e.get("y")
            dist = e.get("dist")
            if dist is None:
                continue
            xe = 0.0 if x is None else float(x)
            ye = 0.0 if y is None else float(y)
            is_three = int(dist >= 23.7 or (abs(xe) >= 219 and ye <= 92))
            ev_e = e.get("e")
            rows.append(
                {
                    "season": season,
                    "game_id": r.game_id,
                    "shooter": e.get("pid", 0),
                    "made": int(e.get("made", 0)),
                    "three": is_three,
                    "dist": float(dist),
                    "x": x,
                    "y": y,
                    "sub": _sub_group(e.get("sub") or ""),
                    "putback": int(e.get("chance", 1) > 1),
                    "period": int(r.period),
                    "into_poss": max(0.0, float(ev_e) - t0) if ev_e is not None else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    xy_ok = df.x.notna() & df.y.notna()
    df["abs_angle"] = 0.0
    dx = df.loc[xy_ok, "x"].astype(float).abs() + 1e-9
    dy = df.loc[xy_ok, "y"].astype(float).abs()
    df.loc[xy_ok, "abs_angle"] = np.degrees(np.arctan2(dy, dx))
    df["x"] = df.x.astype(float).fillna(0.0)
    df["y"] = df.y.astype(float).fillna(0.0)
    return df


def _zone(df: pd.DataFrame) -> pd.Series:
    d = df.dist
    z = pd.Series("mid", index=df.index)
    z[d < 4] = "rim"
    z[(d >= 4) & (d < 14)] = "paint"
    z[df.three == 1] = "three"
    z[d >= 28] = "deep"
    return z


def _design(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["dist"] = df.dist.clip(0, 40)
    x["dist2"] = x.dist**2
    x["abs_angle"] = df.abs_angle
    x["three"] = df.three
    x["putback"] = df.putback
    x["into_poss"] = df.into_poss.fillna(12.0).clip(0, 24)
    x["early_poss"] = (df.into_poss.fillna(12.0) <= 6).astype(int)
    x["period"] = df.period.clip(1, 5)
    for g in [*SUB_GROUPS, "other"]:
        x[f"sub_{g}"] = (df["sub"] == g).astype(int)
    return x


def score_season(season: str, seasons_all: list[str]) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    idx = seasons_all.index(season)
    if idx == 0:
        print(f"{season}: no prior seasons, skipped")
        return
    train_seasons = seasons_all[max(0, idx - TRAIL) : idx]
    train = pd.concat([extract_shots(s) for s in train_seasons], ignore_index=True)
    test = extract_shots(season)

    y_tr = train.made.to_numpy()
    y_te = test.made.to_numpy()
    eps = 1e-6

    def ll(y, p):
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    report = {"season": season, "n_train": len(train), "n_test": len(test)}
    report["ll_const"] = ll(y_te, np.full(len(test), y_tr.mean()))

    ztr, zte = _zone(train), _zone(test)
    zone_rate = train.groupby(ztr).made.mean()
    p_zone = zte.map(zone_rate).fillna(y_tr.mean()).to_numpy()
    report["ll_zone"] = ll(y_te, p_zone)

    xtr, xte = _design(train), _design(test)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(xtr, y_tr)
    report["ll_logistic"] = ll(y_te, lr.predict_proba(xte)[:, 1])

    hgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=7,
    )
    hgb.fit(xtr, y_tr)
    p_hgb = hgb.predict_proba(xte)[:, 1]
    report["ll_hgb"] = ll(y_te, p_hgb)

    bins = np.clip((p_hgb * 20).astype(int), 0, 19)
    rel = (
        pd.DataFrame({"bin": bins, "p": p_hgb, "y": y_te})
        .groupby("bin")
        .agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    )
    w = rel.n / rel.n.sum()
    report["calib_mae"] = float((rel.pred - rel.obs).abs().mul(w).sum())

    out = test[["season", "game_id", "shooter", "made", "dist", "three", "putback"]].copy()
    out["p_make"] = p_hgb
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / f"{season}.parquet", index=False)
    pd.Series(report).to_json(OUT_DIR / f"{season}_report.json")
    print(
        f"{season}: n={len(test):,}  const {report['ll_const']:.4f}  "
        f"zone {report['ll_zone']:.4f}  logistic {report['ll_logistic']:.4f}  "
        f"hgb {report['ll_hgb']:.4f}  calibMAE {report['calib_mae']:.4f}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score"])
    ap.add_argument("season")
    args = ap.parse_args()
    seasons_all = sorted(
        p.stem.replace("_regular_season", "") for p in POSS.glob("*_regular_season.parquet")
    )
    targets = seasons_all[1:] if args.season == "all" else [args.season]
    for s in targets:
        score_season(s, seasons_all)


if __name__ == "__main__":
    main()
