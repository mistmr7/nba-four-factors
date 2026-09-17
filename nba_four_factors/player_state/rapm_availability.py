"""Value pregame inactives with as-of RAPM, and test the feature against Game Score.

The existing availability feature values each inactive by trailing Game Score
(features/availability_features.py, availability_shrunk.py). This module is
the same join with a different valuation: the player's as-of-date RAPM,
scaled by the share of team floor time they normally play.

    value(player, game) = rapm_asof(player, game_date) * expected_minutes / 48

rapm_asof is the latest weekly rating dated on or before the game date, which
was fitted on games strictly before that date, so the feature is known at
tip-off. expected_minutes is the player's mean minutes over their last 20
appearances before the game, across season boundaries, so a player hurt on
opening night is valued at last season's role rather than at zero.

Per game, home and away sums and their difference, in the same shape as
availability_features.parquet so the modeling code consumes it unchanged:

    data/features/availability_rapm_<prior>.parquet
        game_id, home_rapm_out, away_rapm_out, rapm_diff,
        home_rapm_rep_out, away_rapm_rep_out, rapm_rep_diff

The _rep columns subtract a replacement level (default -2.0 per 100) before
scaling, so a missing replacement-level player costs nothing.

The evaluate command repeats the availability_shrunk protocol: the four-factor
baseline, plus raw Game Score, plus RAPM, plus both, under the expanding
window from 2010, reporting out-of-sample margin MAE, log loss delta, and the
share of the Vegas gap closed in the top availability-imbalance decile.

Run from repo root:
    python -m nba_four_factors.player_state.rapm_availability build --prior zero
    python -m nba_four_factors.player_state.rapm_availability evaluate --prior zero
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FEAT = REPO / "data" / "features"
PS = REPO / "data" / "player_state"
INACTIVES = FEAT / "inactives.parquet"
GAME_SCORE = PS / "player_game_score.parquet"
REPLACEMENT_LEVEL = -2.0
TRAILING_GAMES = 20


def game_dates() -> pd.DataFrame:
    gs = pd.read_parquet(GAME_SCORE, columns=["game_id", "game_date", "season_type"])
    gs = gs[gs["season_type"] == "regular_season"].drop_duplicates("game_id")
    gs["game_date"] = pd.to_datetime(gs["game_date"])
    return gs[["game_id", "game_date"]]


def expected_minutes() -> pd.DataFrame:
    """Trailing mean minutes over the last TRAILING_GAMES appearances, as of each played game."""
    gs = pd.read_parquet(
        GAME_SCORE, columns=["person_id", "game_date", "minutes", "status", "season_type"]
    )
    gs = gs[(gs["status"] == "played") & (gs["season_type"] == "regular_season")].copy()
    gs["game_date"] = pd.to_datetime(gs["game_date"])
    gs = gs.sort_values(["person_id", "game_date"])
    gs["exp_min"] = gs.groupby("person_id")["minutes"].transform(
        lambda s: s.rolling(TRAILING_GAMES, min_periods=1).mean()
    )
    return gs[["person_id", "game_date", "exp_min"]].sort_values("game_date")


def build(prior: str, replacement_level: float = REPLACEMENT_LEVEL) -> pd.DataFrame:
    asof = pd.read_parquet(
        PS / f"rapm_asof_{prior}.parquet", columns=["person_id", "asof_date", "rapm"]
    )
    asof["asof_date"] = pd.to_datetime(asof["asof_date"])
    asof = asof.sort_values("asof_date")

    dates = game_dates()
    ina = pd.read_parquet(INACTIVES)
    ina = ina[ina["season_type"] == "regular_season"].merge(dates, on="game_id", how="inner")
    ina = ina.sort_values("game_date")

    j = pd.merge_asof(
        ina,
        asof.rename(columns={"asof_date": "game_date"}),
        by="person_id",
        on="game_date",
        direction="backward",
        allow_exact_matches=True,
    )
    j = pd.merge_asof(
        j.sort_values("game_date"),
        expected_minutes(),
        by="person_id",
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
    )
    j["rapm"] = j["rapm"].fillna(0.0)
    j["exp_min"] = j["exp_min"].fillna(0.0)
    share = j["exp_min"] / 48.0
    j["rapm_out"] = j["rapm"] * share
    j["rapm_rep_out"] = (j["rapm"] - replacement_level).clip(lower=0.0) * share

    agg = j.groupby(["game_id", "is_home"])[["rapm_out", "rapm_rep_out"]].sum().reset_index()
    home = (
        agg[agg["is_home"]].set_index("game_id")[["rapm_out", "rapm_rep_out"]].add_prefix("home_")
    )
    away = (
        agg[~agg["is_home"]].set_index("game_id")[["rapm_out", "rapm_rep_out"]].add_prefix("away_")
    )
    f = dates.set_index("game_id")[[]].join(home).join(away).fillna(0.0)
    f["rapm_diff"] = f["home_rapm_out"] - f["away_rapm_out"]
    f["rapm_rep_diff"] = f["home_rapm_rep_out"] - f["away_rapm_rep_out"]
    f = f.reset_index()
    out = FEAT / f"availability_rapm_{prior}.parquet"
    f.to_parquet(out, index=False)
    n_with = int((f["rapm_diff"].abs() > 0).sum())
    print(
        f"{out.name}  {len(f):,} games, {n_with:,} with any RAPM imbalance, mean |rapm_diff| {f['rapm_diff'].abs().mean():.3f}"
    )
    return f


def evaluate(prior: str) -> pd.DataFrame:
    from ..modeling.availability_sweep import BASE, expanding_pred, logloss

    m = (
        pd.read_parquet(FEAT / "modeling_table.parquet")
        .dropna(subset=["home_margin", "home_win"])
        .copy()
    )
    m["yr"] = m["season"].str[:4].astype(int)
    av = pd.read_parquet(FEAT / "availability_features.parquet")[["game_id", "gmsc_diff"]]
    rp = pd.read_parquet(FEAT / f"availability_rapm_{prior}.parquet")[
        ["game_id", "rapm_diff", "rapm_rep_diff"]
    ]
    lines = pd.read_parquet(FEAT / "vegas_game_lines.parquet")[["game_id", "vegas_home_margin"]]
    m = (
        m.merge(av, on="game_id", how="inner")
        .merge(rp, on="game_id", how="left")
        .merge(lines, on="game_id", how="left")
    )
    cols = [*BASE, "gmsc_diff", "rapm_diff", "rapm_rep_diff"]
    m[cols] = m[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    m = m[m["yr"] >= 2010].reset_index(drop=True)

    y = m["home_margin"].to_numpy(float)
    w = m["home_win"].to_numpy(float)
    veg = np.abs(m["vegas_home_margin"] - m["home_margin"]).to_numpy()
    ok = m["vegas_home_margin"].notna().to_numpy()

    bp, bs = expanding_pred(m, BASE)
    v = ~np.isnan(bp)
    base_ae = np.abs(bp - y)
    base_ll = logloss(bp[v], bs[v], w[v])
    rows = [
        {
            "model": "baseline four factors",
            "oos_mae": base_ae[v].mean(),
            "dll": 0.0,
            "closed_pct": np.nan,
        }
    ]

    variants = {
        "+ raw Game Score": ["gmsc_diff"],
        "+ RAPM": ["rapm_diff"],
        "+ RAPM above replacement": ["rapm_rep_diff"],
        "+ Game Score + RAPM": ["gmsc_diff", "rapm_diff"],
    }
    for name, extra in variants.items():
        p, s = expanding_pred(m, [*BASE, *extra])
        ae = np.abs(p - y)
        dd = m[extra[-1]].abs().to_numpy()
        hi = v & (dd >= np.nanquantile(dd[v], 0.9)) & ok
        b, ff, vg = base_ae[hi].mean(), ae[hi].mean(), veg[hi].mean()
        closed = 100 * (b - ff) / (b - vg) if (b - vg) > 0 else np.nan
        rows.append(
            {
                "model": name,
                "oos_mae": ae[v].mean(),
                "dll": logloss(p[v], s[v], w[v]) - base_ll,
                "closed_pct": closed,
            }
        )
    res = pd.DataFrame(rows)
    print(f"prior = {prior}, expanding window 2010+, {int(v.sum()):,} out-of-sample games")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    res.to_csv(PS / f"rapm_availability_eval_{prior}.csv", index=False)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--prior", choices=["zero", "theta"], default="zero")
        s.add_argument("--replacement-level", type=float, default=REPLACEMENT_LEVEL)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.prior, a.replacement_level)
    else:
        evaluate(a.prior)


if __name__ == "__main__":
    main()
