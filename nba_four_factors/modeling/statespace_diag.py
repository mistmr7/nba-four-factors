"""Where, if anywhere, do we match or beat the closing line?

Runs the scalar Kalman filter over the test seasons, joins the per-game Vegas
closing line and the actual result, and compares margin error and win accuracy
within slices that could localize an edge or a weakness:

  - game phase (how many games each team has played)
  - Vegas spread magnitude (pick-em vs big favorite)
  - combined team quality (preseason win totals)
  - back-to-backs (a rest situation the market prices via player availability)
  - model-vs-Vegas disagreement (the biggest gaps are the injury/lineup probe)

The point is not to report a new headline number but to find structure in the gap:
is the deficit uniform, or concentrated in a few game types that point at the
missing information?

Run: python -m nba_four_factors.modeling.statespace_diag
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .statespace_scalar import LINES, START_TEST_YEAR, TUNE_BEFORE, load, run_season, seed_fit, tune


def games_played_index(m: pd.DataFrame) -> dict:
    """Chronological game number within a season for each team."""
    out = {}
    for _s, g in m.groupby("season"):
        rows = []
        for r in g.itertuples(index=False):
            rows.append((r.game_date, r.game_id, int(r.team_id)))
            rows.append((r.game_date, r.game_id, int(r.away_team_id)))
        rows.sort()
        cnt = {}
        for _d, gid, t in rows:
            cnt[t] = cnt.get(t, 0) + 1
            out[(gid, t)] = cnt[t]
    return out


def build_frame() -> pd.DataFrame:
    m = load()
    pre = m[m["yr"] < TUNE_BEFORE]
    slope, mean_wt, hca = seed_fit(pre)
    params, _ = tune(m, slope, mean_wt, hca)
    gpi = games_played_index(m)
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]

    frames = []
    for ty in range(START_TEST_YEAR, m["yr"].max() + 1):
        g = m[m["yr"] == ty]
        if len(g) < 100:
            continue
        pm, pv, z, w = run_season(g, params, slope, mean_wt, hca)
        d = g[
            [
                "game_id",
                "season",
                "team_id",
                "away_team_id",
                "home_margin",
                "home_win",
                "home_win_total",
                "away_win_total",
                "b2b_diff",
            ]
        ].copy()
        d["our_pred"] = pm
        d["our_var"] = pv
        gp_home = [
            gpi.get((gid, int(t)), 1) for gid, t in zip(d["game_id"], d["team_id"], strict=False)
        ]
        gp_away = [
            gpi.get((gid, int(t)), 1)
            for gid, t in zip(d["game_id"], d["away_team_id"], strict=False)
        ]
        d["ngames"] = 0.5 * (np.array(gp_home) + np.array(gp_away))
        frames.append(d)

    f = pd.concat(frames, ignore_index=True).merge(lines, on="game_id", how="left")
    f = f.dropna(subset=["vegas_home_margin"]).reset_index(drop=True)
    f["our_ae"] = np.abs(f["our_pred"] - f["home_margin"])
    f["veg_ae"] = np.abs(f["vegas_home_margin"] - f["home_margin"])
    f["our_win_ok"] = (f["our_pred"] > 0) == (f["home_win"] > 0.5)
    f["veg_win_ok"] = (f["vegas_home_margin"] > 0) == (f["home_win"] > 0.5)
    f["disagree"] = np.abs(f["our_pred"] - f["vegas_home_margin"])
    f["combined_wt"] = f["home_win_total"] + f["away_win_total"]
    return f


def report_slice(f, name, col, bins, labels):
    cut = pd.cut(f[col], bins=bins, labels=labels, include_lowest=True)
    print(f"\n{name}")
    print(
        f"  {'bucket':>14s}{'n':>7s}{'our MAE':>9s}{'Veg MAE':>9s}{'gap':>7s}"
        f"{'ourWin%':>9s}{'VegWin%':>9s}"
    )
    for lab in labels:
        s = f[cut == lab]
        if len(s) < 50:
            continue
        gap = s["our_ae"].mean() - s["veg_ae"].mean()
        flag = "  <= Vegas" if gap <= 0.05 else ""
        print(
            f"  {lab!s:>14s}{len(s):>7d}{s['our_ae'].mean():>9.3f}{s['veg_ae'].mean():>9.3f}"
            f"{gap:>+7.3f}{100 * s['our_win_ok'].mean():>9.1f}{100 * s['veg_win_ok'].mean():>9.1f}{flag}"
        )


def main():
    f = build_frame()
    print(f"Test games (2010-2025) with a closing line: {len(f):,}")
    print(
        f"Overall: our MAE {f['our_ae'].mean():.3f}  Vegas MAE {f['veg_ae'].mean():.3f}  "
        f"gap {f['our_ae'].mean() - f['veg_ae'].mean():+.3f}"
    )
    print(
        f"         our win acc {100 * f['our_win_ok'].mean():.1f}%  "
        f"Vegas win acc {100 * f['veg_win_ok'].mean():.1f}%"
    )
    print(
        f"         games we and Vegas pick a different winner: "
        f"{100 * np.mean((f['our_pred'] > 0) != (f['vegas_home_margin'] > 0)):.1f}%"
    )

    report_slice(
        f,
        "By game phase (avg games played by the two teams)",
        "ngames",
        [0, 10, 25, 50, 200],
        ["1-10", "11-25", "26-50", "51+"],
    )
    report_slice(f, "By Vegas spread magnitude (points)", None, None, None) if False else None
    f["abs_spread"] = np.abs(f["vegas_home_margin"])
    report_slice(
        f,
        "By Vegas spread magnitude (points)",
        "abs_spread",
        [0, 3, 6, 10, 100],
        ["0-3", "3-6", "6-10", "10+"],
    )
    report_slice(
        f,
        "By combined preseason win totals (team quality)",
        "combined_wt",
        [0, 70, 82, 94, 200],
        ["<70", "70-82", "82-94", "94+"],
    )

    print("\nBack-to-backs (either team on a back-to-back)")
    for lab, mask in [("neither b2b", f["b2b_diff"] == 0), ("a team on b2b", f["b2b_diff"] != 0)]:
        s = f[mask]
        print(
            f"  {lab:>14s}{len(s):>7d}{s['our_ae'].mean():>9.3f}{s['veg_ae'].mean():>9.3f}"
            f"{s['our_ae'].mean() - s['veg_ae'].mean():>+7.3f}"
        )

    print("\nBy how much we disagree with Vegas (decile of |our_pred - vegas|)")
    f["dec"] = pd.qcut(f["disagree"], 10, labels=False)
    for dq in [0, 4, 8, 9]:
        s = f[f["dec"] == dq]
        name = {0: "closest", 4: "median", 8: "9th", 9: "most (top 10%)"}[dq]
        print(
            f"  {name:>14s}{len(s):>7d}{s['our_ae'].mean():>9.3f}{s['veg_ae'].mean():>9.3f}"
            f"{s['our_ae'].mean() - s['veg_ae'].mean():>+7.3f}"
            f"   our win {100 * s['our_win_ok'].mean():.1f}%  Veg win {100 * s['veg_win_ok'].mean():.1f}%"
        )


if __name__ == "__main__":
    main()
