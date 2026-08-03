"""Reliability-shrunk availability value, and a raw-vs-shrunk comparison.

The raw feature values an absent player by their trailing season-to-date Game
Score, which is noisy in the first few games and zero for a player who has not
appeared yet (a season-opening injury, a just-traded star). We shrink toward the
player's prior-season value:

    value = alpha * trailing_this_season + (1 - alpha) * prior_season
    alpha = n / (n + k)        n = games played this season so far

so early on we lean on the established level and converge to the current level as
evidence accrues. A player with no prior season (a rookie) shrinks toward a
replacement baseline of zero.

Builds the shrunk home-minus-away differential for a given k and writes it next to
the raw features, then compares baseline vs +raw vs +shrunk under the same
expanding-window protocol as the sweep.

Run: python -m nba_four_factors.features.availability_shrunk
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..modeling.availability_sweep import BASE, expanding_pred, logloss
from .availability_features import game_dates

REPO = Path(__file__).resolve().parents[2]
FEAT = REPO / "data" / "features"


def prev_season(s: str) -> str:
    start = int(s[:4])
    return f"{start - 1}_{start % 100:02d}"


def player_gmsc():
    logs = pd.read_parquet(FEAT / "player_game_logs.parquet")
    logs = logs[(logs["season_type"] == "regular_season") & (logs["min"] > 0)].copy()
    logs["gmsc"] = (
        logs["pts"]
        + 0.4 * logs["fgm"]
        - 0.7 * logs["fga"]
        - 0.4 * (logs["fta"] - logs["ftm"])
        + 0.7 * logs["oreb"]
        + 0.3 * logs["dreb"]
        + logs["stl"]
        + 0.7 * logs["ast"]
        + 0.7 * logs["blk"]
        - 0.4 * logs["pf"]
        - logs["tov"]
    )
    logs = logs.merge(game_dates(), on="game_id", how="left").dropna(subset=["game_date"])
    return logs.sort_values(["person_id", "season", "game_date"])


def shrunk_diff(logs, k: float) -> pd.DataFrame:
    # Prior-season mean Game Score, mapped onto the following season.
    smean = logs.groupby(["person_id", "season"])["gmsc"].mean().reset_index()
    smean["next"] = smean["season"].apply(
        lambda s: f"{int(s[:4]) + 1}_{(int(s[:4]) + 1) % 100:02d}"
    )
    prior = smean[["person_id", "next", "gmsc"]].rename(columns={"next": "season", "gmsc": "prior"})

    g = logs.groupby(["person_id", "season"])
    cnt = (g.cumcount() + 1).to_numpy()
    cum = (g["gmsc"].cumsum().to_numpy()) / cnt
    tl = logs[["person_id", "season", "game_date"]].copy()
    tl = tl.merge(prior, on=["person_id", "season"], how="left")
    a = cnt / (cnt + k)
    tl["shrunk"] = a * cum + (1 - a) * tl["prior"].fillna(0.0).to_numpy()
    tl = tl.sort_values("game_date")

    dates = game_dates()
    ina = pd.read_parquet(FEAT / "inactives.parquet")
    ina = ina[ina["season_type"] == "regular_season"].merge(dates, on="game_id", how="left")
    ina = ina.dropna(subset=["game_date"]).sort_values("game_date")
    j = pd.merge_asof(
        ina,
        tl[["person_id", "season", "game_date", "shrunk"]],
        by=["person_id", "season"],
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
    )
    # No game yet this season: fall back to the prior-season value directly.
    j = j.merge(prior, on=["person_id", "season"], how="left")
    j["val"] = j["shrunk"].fillna(j["prior"]).fillna(0.0)

    agg = j.groupby(["game_id", "is_home"])["val"].sum().reset_index()
    home = agg[agg["is_home"]].set_index("game_id")["val"].rename("home")
    away = agg[~agg["is_home"]].set_index("game_id")["val"].rename("away")
    f = dates[["game_id"]].drop_duplicates().set_index("game_id").join(home).join(away).fillna(0.0)
    f["gmscs_diff"] = f["home"] - f["away"]
    return f.reset_index()[["game_id", "gmscs_diff"]]


def main():
    logs = player_gmsc()
    m = pd.read_parquet(REPO / "data" / "features" / "modeling_table.parquet")
    m = m.dropna(subset=["home_margin", "home_win"]).copy()
    m["yr"] = m["season"].str[:4].astype(int)
    av = pd.read_parquet(FEAT / "availability_features.parquet")[["game_id", "gmsc_diff"]]
    lines = pd.read_parquet(FEAT / "vegas_game_lines.parquet")[["game_id", "vegas_home_margin"]]
    m = m.merge(av, on="game_id", how="inner").merge(lines, on="game_id", how="left")
    m[[*BASE, "gmsc_diff"]] = m[[*BASE, "gmsc_diff"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def evalcols(frame, cols):
        pred, sig = expanding_pred(frame[frame["yr"] >= 2010].reset_index(drop=True), cols)
        return pred, sig

    m2010 = m[m["yr"] >= 2010].reset_index(drop=True)
    y2, w2 = m2010["home_margin"].to_numpy(float), m2010["home_win"].to_numpy(float)
    veg2 = np.abs(m2010["vegas_home_margin"] - m2010["home_margin"]).to_numpy()
    ok2 = m2010["vegas_home_margin"].notna().to_numpy()

    bp, bs = expanding_pred(m2010, BASE)
    v = ~np.isnan(bp)
    base_ae = np.abs(bp - y2)
    print(
        f"Baseline four factors      : OOS MAE {base_ae[v].mean():.4f}  ll {logloss(bp[v],bs[v],w2[v]):.4f}"
    )
    rp, rs = expanding_pred(m2010, [*BASE, "gmsc_diff"])
    print(
        f"+ raw Game Score           : OOS MAE {np.abs(rp-y2)[v].mean():.4f}  ll {logloss(rp[v],rs[v],w2[v]):.4f}"
    )

    print(
        f"\n{'k':>5s}{'OOS MAE':>9s}{'dll':>9s}   imbalance decile: base  +shrunk  Vegas  closed%"
    )
    for k in [6, 12, 20, 35]:
        sd = shrunk_diff(logs, k)
        mm = m2010.merge(sd, on="game_id", how="left")
        mm["gmscs_diff"] = mm["gmscs_diff"].fillna(0.0)
        sp, ss = expanding_pred(mm, [*BASE, "gmscs_diff"])
        ae = np.abs(sp - y2)
        dd = mm["gmscs_diff"].abs().to_numpy()
        hi = v & (dd >= np.nanquantile(dd[v], 0.9)) & ok2
        b, fft, vg = base_ae[hi].mean(), ae[hi].mean(), veg2[hi].mean()
        closed = 100 * (b - fft) / (b - vg) if (b - vg) > 0 else np.nan
        print(
            f"{k:>5d}{ae[v].mean():>9.4f}{logloss(sp[v],ss[v],w2[v])-logloss(bp[v],bs[v],w2[v]):>+9.4f}"
            f"      {b:>8.2f}{fft:>8.2f}{vg:>7.2f}{closed:>8.1f}%"
        )
    print("\n(compare raw Game Score's imbalance-decile closed%: ~66%)")


if __name__ == "__main__":
    main()
