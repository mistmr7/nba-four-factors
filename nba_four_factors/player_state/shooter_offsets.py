"""Shooter skill overlay: empirical-Bayes offsets on the league backbone.

Credit engine Phase 2, brick two. The backbone (shot_quality.py) is
identity-free on purpose; this layer measures each shooter's gap from
the league baseline per zone, shrunk by sample size, and applies it on
the logit scale. Mirrors the Thunder-project architecture: backbone,
then out-of-sample residual offsets with an empirical-Bayes prior.

Leakage protocol: every season parquet written by shot_quality.py was
scored by a model that never saw that season, so residuals taken from
prior seasons' output files are already out-of-sample. For a target
season, offsets are estimated from the trailing SO_TRAIL seasons of
those files and applied frozen.

Two variants per season:
    static  equal weight across the trailing window
    decay   exponential recency weighting, half-life SO_HALF_LIFE
            seasons, to test whether skill drifts at this horizon

Run from repo root:
    python -m nba_four_factors.player_state.shooter_offsets score 2025_26
    python -m nba_four_factors.player_state.shooter_offsets score all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SQ_DIR = REPO / "data" / "player_state" / "shot_quality"
OUT_DIR = REPO / "data" / "player_state" / "shooter_offsets"

TRAIL = int(os.environ.get("SO_TRAIL", "10"))
HALF_LIFE = float(os.environ.get("SO_HALF_LIFE", "3.0"))
MIN_TAU_N = 20
EPS = 1e-6

_CACHE: dict[str, pd.DataFrame] = {}


def _zone(df: pd.DataFrame) -> pd.Series:
    d = df.dist
    z = pd.Series("mid", index=df.index)
    z[d < 4] = "rim"
    z[(d >= 4) & (d < 14)] = "paint"
    z[df.three == 1] = "three"
    z[d >= 28] = "deep"
    return z


def _load(season: str) -> pd.DataFrame:
    if season not in _CACHE:
        df = pd.read_parquet(SQ_DIR / f"{season}.parquet")
        df["zone"] = _zone(df)
        _CACHE[season] = df
    return _CACHE[season]


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _offsets(train: pd.DataFrame, w: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Per shooter-zone logit offsets, shrunk toward zero.

    obs = weighted residual sum over weighted leverage sum, the one-step
    logit estimate with sampling variance lev2 / lev**2. tau2 per zone by
    method of moments over shooter-zones with MIN_TAU_N shots or more.
    Posterior mean under a N(0, tau2) prior does the shrinkage.
    """
    t = train.loc[train.shooter > 0, ["shooter", "zone", "made", "p_make"]].copy()
    t["w"] = w[train.shooter.to_numpy() > 0]
    p = t.p_make.clip(EPS, 1 - EPS)
    t["res"] = t.w * (t.made - p)
    t["lev"] = t.w * p * (1 - p)
    t["lev2"] = t.w**2 * p * (1 - p)
    g = (
        t.groupby(["shooter", "zone"])
        .agg(res=("res", "sum"), lev=("lev", "sum"), lev2=("lev2", "sum"), n=("made", "size"))
        .reset_index()
    )
    g = g[g.lev > 0].copy()
    g["obs"] = g.res / g.lev
    g["svar"] = g.lev2 / g.lev**2
    parts = []
    taus: dict[str, float] = {}
    for zone, gz in g.groupby("zone"):
        est = gz[gz.n >= MIN_TAU_N]
        tau2 = max(0.0, float((est.obs**2 - est.svar).mean())) if len(est) else 0.0
        taus[zone] = tau2
        gz = gz.copy()
        gz["off"] = gz.obs * tau2 / (tau2 + gz.svar)
        parts.append(gz)
    out = pd.concat(parts, ignore_index=True)
    return out[["shooter", "zone", "off"]], taus


def _apply(test: pd.DataFrame, offs: pd.DataFrame) -> tuple[np.ndarray, float]:
    m = test.merge(offs, on=["shooter", "zone"], how="left")
    matched = float(m.off.notna().mean())
    off = m.off.fillna(0.0).to_numpy()
    p = m.p_make.clip(EPS, 1 - EPS).to_numpy()
    logit = np.log(p / (1 - p)) + off
    return 1 / (1 + np.exp(-logit)), matched


def score_season(season: str, seasons_all: list[str]) -> None:
    idx = seasons_all.index(season)
    if idx == 0:
        print(f"{season}: no prior scored seasons, skipped")
        return
    trail = seasons_all[max(0, idx - TRAIL) : idx]
    train = pd.concat([_load(s) for s in trail], ignore_index=True)
    ages = np.concatenate([np.full(len(_load(s)), idx - seasons_all.index(s)) for s in trail])
    test = _load(season)
    y = test.made.to_numpy()

    offs_static, taus = _offsets(train, np.ones(len(train)))
    p_static, matched = _apply(test, offs_static)

    w_decay = 0.5 ** ((ages - 1) / HALF_LIFE)
    offs_decay, _ = _offsets(train, w_decay)
    p_decay, _ = _apply(test, offs_decay)

    report = {
        "season": season,
        "n_test": len(test),
        "trail_seasons": len(trail),
        "matched_share": matched,
        "ll_league": _ll(y, test.p_make.to_numpy()),
        "ll_static": _ll(y, p_static),
        "ll_decay": _ll(y, p_decay),
        "half_life": HALF_LIFE,
        "tau2": taus,
    }

    out = test[["season", "game_id", "shooter", "made", "zone"]].copy()
    out["p_league"] = test.p_make
    out["p_shooter"] = p_static
    out["p_shooter_decay"] = p_decay
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / f"{season}.parquet", index=False)
    with open(OUT_DIR / f"{season}_report.json", "w") as f:
        json.dump(report, f, indent=1)

    d_static = (report["ll_league"] - report["ll_static"]) * 1000
    d_decay = (report["ll_league"] - report["ll_decay"]) * 1000
    tau_str = " ".join(f"{z}:{np.sqrt(v):.3f}" for z, v in sorted(taus.items()))
    print(
        f"{season}: n={len(test):,}  league {report['ll_league']:.4f}  "
        f"static {report['ll_static']:.4f} (+{d_static:.1f}m)  "
        f"decay {report['ll_decay']:.4f} (+{d_decay:.1f}m)  "
        f"match {matched:.0%}  tau {tau_str}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score"])
    ap.add_argument("season")
    args = ap.parse_args()
    seasons_all = sorted(p.stem for p in SQ_DIR.glob("*.parquet"))
    targets = seasons_all[1:] if args.season == "all" else [args.season]
    for s in targets:
        score_season(s, seasons_all)


if __name__ == "__main__":
    main()
