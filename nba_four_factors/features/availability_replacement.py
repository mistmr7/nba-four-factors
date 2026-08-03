"""Replacement-aware and timing-aware availability features.

Two ideas the simple value-out feature misses.

Replacement quality. Losing a star hurts less if a good backup steps in. Instead of
summing the value that is OUT, we value the rotation that actually plays. For each
team-game we take the top-9 trailing Game Scores of the full roster (everyone, as
if healthy) and the top-9 of the AVAILABLE roster (inactives removed); the
difference, depletion, is the value lost net of whoever replaces it. A deep team
shows a small depletion because the next man up is nearly as good.

Timing. A chronic absence is partly already baked into a team's trailing four-factor
stats (recent games were played without the player), so its marginal value-out is
double-counting; a fresh scratch is new information. We split the value-out into
fresh (player missed few of the team's recent games) and chronic by a missed-game
streak.

All values are trailing and as-of (prior games only), so the features are causal.
Output diffs (home minus away) written to availability_replacement.parquet, with an
inline expanding-window test against the shrunk Game-Score feature.

Run: python -m nba_four_factors.features.availability_replacement
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .availability_features import game_dates
from .availability_shrunk import player_gmsc

REPO = Path(__file__).resolve().parents[2]
FEAT = REPO / "data" / "features"
PROC = REPO / "data" / "processed"
K = 9
FRESH_MAX = 2  # missed this-many-or-fewer of the team's recent games => fresh


def _next_season(s):
    y = int(s[:4]) + 1
    return f"{y}_{(y + 1) % 100:02d}"


def trailing():
    logs = player_gmsc()  # played, regular season, gmsc, game_date, sorted
    g = logs.groupby(["person_id", "season"])
    cnt = (g.cumcount() + 1).to_numpy()
    csum = g["gmsc"].cumsum().to_numpy()
    logs["cum"] = csum / cnt  # inclusive (for as-of of inactives)
    with np.errstate(invalid="ignore", divide="ignore"):
        logs["prior"] = np.where(cnt > 1, (csum - logs["gmsc"].to_numpy()) / (cnt - 1), np.nan)
    sm = g["gmsc"].mean().reset_index()
    sm["season"] = sm["season"].apply(_next_season)
    logs = logs.merge(sm.rename(columns={"gmsc": "ps"}), on=["person_id", "season"], how="left")
    logs["avail_val"] = logs["prior"].fillna(logs["ps"]).fillna(0.0)
    return logs


def build():
    logs = trailing()
    dates = game_dates()
    ina = pd.read_parquet(FEAT / "inactives.parquet")
    ina = ina[ina["season_type"] == "regular_season"].merge(dates, on="game_id", how="left")
    ina = ina.dropna(subset=["game_date"]).sort_values("game_date")

    asof = logs[["person_id", "season", "game_date", "cum"]].sort_values("game_date")
    j = pd.merge_asof(
        ina,
        asof,
        by=["person_id", "season"],
        on="game_date",
        direction="backward",
        allow_exact_matches=False,
    )
    sm = logs.groupby(["person_id", "season"])["gmsc"].mean().reset_index()
    sm["season"] = sm["season"].apply(_next_season)
    j = j.merge(sm.rename(columns={"gmsc": "ps"}), on=["person_id", "season"], how="left")
    j["out_val"] = j["cum"].fillna(j["ps"]).fillna(0.0)

    # ---- replacement: top-K of full vs available rosters ----
    av = logs[["game_id", "is_home", "avail_val"]].rename(columns={"avail_val": "val"})
    av["available"] = True
    out = j[["game_id", "is_home", "out_val"]].rename(columns={"out_val": "val"})
    out["available"] = False
    long = pd.concat([av, out], ignore_index=True)
    long = long.sort_values(["game_id", "is_home", "val"], ascending=[True, True, False])
    long["frank"] = long.groupby(["game_id", "is_home"]).cumcount()
    fullK = long[long["frank"] < K].groupby(["game_id", "is_home"])["val"].sum()
    avl = long[long["available"]].copy()
    avl["arank"] = avl.groupby(["game_id", "is_home"]).cumcount()
    availK = avl[avl["arank"] < K].groupby(["game_id", "is_home"])["val"].sum()
    dep = (fullK - availK).rename("dep").reset_index()

    # ---- timing: fresh vs chronic value-out via missed-game streak ----
    streak = _missed_streaks()
    j = j.merge(streak, on=["game_id", "person_id"], how="left")
    j["fresh"] = j["out_val"] * (j["missed_streak"].fillna(0) <= FRESH_MAX)
    j["chronic"] = j["out_val"] * (j["missed_streak"].fillna(99) > FRESH_MAX)
    tim = j.groupby(["game_id", "is_home"])[["fresh", "chronic"]].sum().reset_index()

    feat = dep.merge(tim, on=["game_id", "is_home"], how="outer")
    home = (
        feat[feat["is_home"]].set_index("game_id")[["dep", "fresh", "chronic"]].add_prefix("home_")
    )
    away = (
        feat[~feat["is_home"]].set_index("game_id")[["dep", "fresh", "chronic"]].add_prefix("away_")
    )
    g0 = dates[["game_id"]].drop_duplicates().set_index("game_id")
    f = g0.join(home).join(away).fillna(0.0).reset_index()
    for m in ["dep", "fresh", "chronic"]:
        f[m + "_diff"] = f["home_" + m] - f["away_" + m]
    f.to_parquet(FEAT / "availability_replacement.parquet", index=False)
    print(
        f"availability_replacement.parquet  {len(f):,} games  "
        f"(mean |dep_diff| {f['dep_diff'].abs().mean():.2f}, "
        f"|fresh| {f['fresh_diff'].abs().mean():.2f}, |chronic| {f['chronic_diff'].abs().mean():.2f})"
    )
    return f


def _missed_streaks() -> pd.DataFrame:
    """For each inactive occurrence, how many of the team's immediately prior games
    the player also missed (a chronic-absence proxy)."""
    proc = pd.concat(
        [pd.read_parquet(p) for p in glob.glob(str(PROC / "*/regular_season.parquet"))],
        ignore_index=True,
    )
    proc["game_date"] = pd.to_datetime(proc["game_date"])
    sched = (
        proc[["season", "team_id", "game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["team_id", "season", "game_date"])
    )
    sched["gn"] = sched.groupby(["team_id", "season"]).cumcount()
    logs = pd.read_parquet(FEAT / "player_game_logs.parquet")
    logs = logs[(logs["season_type"] == "regular_season") & (logs["min"] > 0)]
    played = logs[["game_id", "team_id", "person_id"]].merge(
        sched[["game_id", "team_id", "gn"]], on=["game_id", "team_id"], how="left"
    )
    ina = pd.read_parquet(FEAT / "inactives.parquet")
    ina = ina[ina["season_type"] == "regular_season"][["game_id", "team_id", "person_id", "season"]]
    ina = ina.merge(sched[["game_id", "team_id", "gn"]], on=["game_id", "team_id"], how="left")

    played_idx = {}
    for t, p, gn in zip(played["team_id"], played["person_id"], played["gn"], strict=False):
        played_idx.setdefault((int(t), int(p)), set()).add(gn)
    rows = []
    for gid, t, p, gn in zip(
        ina["game_id"], ina["team_id"], ina["person_id"], ina["gn"], strict=False
    ):
        if pd.isna(gn):
            rows.append((gid, int(p), 0))
            continue
        s = played_idx.get((int(t), int(p)), set())
        k = int(gn) - 1
        streak = 0
        while k >= 0 and k not in s:
            streak += 1
            k -= 1
        rows.append((gid, int(p), streak))
    return pd.DataFrame(rows, columns=["game_id", "person_id", "missed_streak"])


def _test(f):
    from ..modeling.availability_sweep import BASE, expanding_pred
    from .availability_shrunk import shrunk_diff

    TABLE = FEAT / "modeling_table.parquet"
    LINES = FEAT / "vegas_game_lines.parquet"
    m = pd.read_parquet(TABLE).dropna(subset=["home_margin", "home_win"]).copy()
    m["yr"] = m["season"].str[:4].astype(int)
    sh = shrunk_diff(player_gmsc(), 12.0)
    lines = pd.read_parquet(LINES)[["game_id", "vegas_home_margin"]]
    m = m.merge(f[["game_id", "dep_diff", "fresh_diff", "chronic_diff"]], on="game_id", how="left")
    m = m.merge(sh, on="game_id", how="left").merge(lines, on="game_id", how="left")
    cols = [*BASE, "gmscs_diff", "dep_diff", "fresh_diff", "chronic_diff"]
    m[cols] = m[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    m = m[m["yr"] >= 2010].reset_index(drop=True)
    y = m["home_margin"].to_numpy(float)
    veg = np.abs(m["vegas_home_margin"] - m["home_margin"]).to_numpy()
    ok = m["vegas_home_margin"].notna().to_numpy()
    bp, bs = expanding_pred(m, BASE)
    v = ~np.isnan(bp)
    base_ae = np.abs(bp - y)
    d = m["gmscs_diff"].abs().to_numpy()
    hi = v & (d >= np.nanquantile(d[v], 0.9)) & ok
    print(
        f"\nbaseline four factors OOS MAE {base_ae[v].mean():.4f} | Vegas {veg[v&ok].mean():.4f} "
        f"(hi-absence Vegas {veg[hi].mean():.4f})"
    )
    print(f"{'+ feature(s)':34s}{'OOS MAE':>9s}{'dMAE':>9s}{'hiDec':>8s}{'closed%':>9s}")
    print("-" * 70)
    for name, c in [
        ("shrunk gmsc (best so far)", ["gmscs_diff"]),
        ("replacement depletion", ["dep_diff"]),
        ("shrunk gmsc + depletion", ["gmscs_diff", "dep_diff"]),
        ("fresh + chronic (split out)", ["fresh_diff", "chronic_diff"]),
        ("shrunk gmsc + fresh + chronic", ["gmscs_diff", "fresh_diff", "chronic_diff"]),
        ("depletion + fresh + chronic", ["dep_diff", "fresh_diff", "chronic_diff"]),
    ]:
        pred, sig = expanding_pred(m, BASE + c)
        ae = np.abs(pred - y)
        b, ff, vg = base_ae[hi].mean(), ae[hi].mean(), veg[hi].mean()
        closed = 100 * (b - ff) / (b - vg) if (b - vg) > 0 else np.nan
        print(
            f"{name:34s}{ae[v].mean():>9.4f}{ae[v].mean()-base_ae[v].mean():>+9.4f}{ff:>8.3f}{closed:>8.1f}%"
        )


def main():
    f = build()
    _test(f)


if __name__ == "__main__":
    main()
