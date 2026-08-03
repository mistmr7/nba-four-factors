"""
EDA 08: Playoff home court advantage over time, isolated.

Purpose
This notebook asks one question from several angles: how large is home court
advantage in the playoffs, how has it changed across 1997-98 to the present,
and how much of it is actually home court rather than something correlated with
hosting (better teams host, road teams travel). It also uses the 2019-20 bubble
as a natural control: those playoffs were played at a single neutral site with
no crowd and no travel, so comparing bubble HCA to the surrounding years
estimates the part of home advantage that the crowd and travel supply.

Sections
S1. Build the one-row-per-game home-perspective frame. Drop neutral-site games
    (the 2019-20 bubble) from the main trend; keep them aside for S5.
S2. Playoff HCA over time. Home win rate and home margin by season, with a
    trend fit. Regular season overlaid for context.
S3. Playoff versus regular-season HCA in the same year. Is the playoff crowd
    worth more than the regular-season one, and is the gap moving.
S4. Four-factor decomposition. Which of the four factors does the home team
    actually win, and is the free-throw-rate edge (the officiating and crowd
    tell) trending.
S5. The 2020 bubble natural experiment. HCA with no crowd and no travel versus
    the neighboring seasons.
S6. Travel. How much of the home margin tracks the visitor's recent travel
    load rather than home court itself.
S7. Seed and team quality. Higher seeds host, so regress the home margin on the
    regular-season win gap to separate home court from being the better team.
S8. Era synthesis table.

Run order
Set the paths in CONFIG if your layout differs, then run top to bottom. It
reads every processed playoffs parquet present, so once the 2025-26 playoffs
are pulled and processed they enter every figure and table automatically with
no code change. Figures are written to figures/eda_08.
"""

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thesis_style import apply_thesis_style

apply_thesis_style()

# %%
# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
TRAVEL_PATH = Path(__file__).resolve().parents[1] / "data" / "features" / "travel.parquet"
FIG_DIR = Path(__file__).resolve().parent / "figures" / "eda_08"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Seasons short enough that raw-win counts mislead; flagged, not dropped.
SHORT_SEASONS = {"1998_99", "2011_12"}
BUBBLE_SEASON = "2019_20"

# Era buckets for the synthesis table, by season start year.
ERA_BOUNDS = [(1997, 2004), (2005, 2012), (2013, 2019), (2020, 2026)]


def season_start_year(season: str) -> int:
    return int(str(season)[:4])


# %%
# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------


def load_layer(season_type: str) -> pd.DataFrame:
    """Concatenate every processed parquet of one season_type across seasons."""
    files = sorted(PROCESSED_DIR.glob(f"*/{season_type}.parquet"))
    if not files:
        raise FileNotFoundError(f"No {season_type}.parquet under {PROCESSED_DIR}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["start_year"] = df["season"].map(season_start_year)
    return df


playoffs = load_layer("playoffs")
regular = load_layer("regular_season")
travel = pd.read_parquet(TRAVEL_PATH)

print(f"Playoff team-rows: {len(playoffs):,}  seasons: {playoffs['season'].nunique()}")
print(f"Latest playoff season present: {sorted(playoffs['season'].unique())[-1]}")
print("Reminder: this updates automatically once 2025-26 playoffs are processed.")


# %%
# ----------------------------------------------------------------------
# S1. Home-perspective frame. One row per game, the home team's view.
# ----------------------------------------------------------------------


def home_frame(df: pd.DataFrame, drop_neutral: bool = True) -> pd.DataFrame:
    """Return one row per game from the home team's perspective.

    margin is the home team's point margin. The four-factor differentials are
    signed so positive favors the home team. Neutral-site games are dropped by
    default since there is no real home court there.
    """
    h = df[df["is_home"]].copy()
    if drop_neutral:
        h = h[~h["is_neutral"]]
    h["efg_diff"] = h["off_efg_pct"] - h["def_efg_pct"]
    h["oreb_diff"] = h["off_orb_pct"] - h["def_orb_pct"]
    h["tov_diff"] = h["def_tov_pct"] - h["off_tov_pct"]  # positive: home forces more TOs
    h["ftr_diff"] = h["off_ft_rate"] - h["def_ft_rate"]
    h["home_win"] = (h["margin"] > 0).astype(int)
    return h


po = home_frame(playoffs)
rs = home_frame(regular)
print(f"Playoff games (non-neutral): {len(po):,}   Regular-season games: {len(rs):,}")


# %%
# ----------------------------------------------------------------------
# S2. Playoff HCA over time
# ----------------------------------------------------------------------


def by_season(h: pd.DataFrame) -> pd.DataFrame:
    g = h.groupby(["season", "start_year"], as_index=False).agg(
        n=("margin", "size"),
        home_win_pct=("home_win", "mean"),
        home_margin=("margin", "mean"),
    )
    return g.sort_values("start_year").reset_index(drop=True)


po_season = by_season(po)
rs_season = by_season(rs)

# Trend fit on playoff home win pct, excluding the crowdless bubble year.
trend = po_season[po_season["season"] != BUBBLE_SEASON]
slope_wp, intercept_wp = np.polyfit(trend["start_year"], trend["home_win_pct"], 1)
slope_mg, _ = np.polyfit(trend["start_year"], trend["home_margin"], 1)

print("Playoff HCA over time:")
print(f"  mean home win pct (ex-bubble): {trend['home_win_pct'].mean():.3f}")
print(f"  mean home margin  (ex-bubble): {trend['home_margin'].mean():+.2f}")
print(f"  home win pct trend: {slope_wp * 10:+.3f} per decade")
print(f"  home margin trend:  {slope_mg * 10:+.2f} points per decade")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, label in (
    (axes[0], "home_win_pct", "Home win rate"),
    (axes[1], "home_margin", "Home margin (pts)"),
):
    ax.plot(po_season["start_year"], po_season[col], marker="o", color="#534AB7", label="playoffs")
    ax.plot(
        rs_season["start_year"],
        rs_season[col],
        marker="o",
        alpha=0.5,
        color="#4C78A8",
        label="regular season",
    )
    b = po_season[po_season["season"] == BUBBLE_SEASON]
    if len(b):
        ax.scatter(
            b["start_year"], b[col], color="#D62728", zorder=5, label="2020 bubble (no crowd)"
        )
    ax.set_title(f"Playoff {label} by season")
    ax.set_xlabel("Season start year")
    ax.set_ylabel(label)
    ax.legend()
axes[0].axhline(0.5, color="#888780", ls="--", lw=0.8)
axes[1].axhline(0.0, color="#888780", ls="--", lw=0.8)
fig.tight_layout()
fig.savefig(FIG_DIR / "2_hca_over_time.png", dpi=130)
plt.close(fig)


# %%
# ----------------------------------------------------------------------
# S3. Playoff versus regular-season HCA, same year
# ----------------------------------------------------------------------
cmp = po_season.merge(rs_season, on=["season", "start_year"], suffixes=("_po", "_rs"))
cmp["margin_gap"] = cmp["home_margin_po"] - cmp["home_margin_rs"]
cmp["winpct_gap"] = cmp["home_win_pct_po"] - cmp["home_win_pct_rs"]
cmp_nb = cmp[cmp["season"] != BUBBLE_SEASON]
print("Playoff minus regular-season HCA (same year, ex-bubble):")
print(f"  mean home-margin gap: {cmp_nb['margin_gap'].mean():+.2f} pts")
print(f"  mean home-win-pct gap: {cmp_nb['winpct_gap'].mean():+.3f}")
gap_slope, _ = np.polyfit(cmp_nb["start_year"], cmp_nb["margin_gap"], 1)
print(f"  gap trend: {gap_slope * 10:+.2f} pts per decade")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(cmp["start_year"], cmp["margin_gap"], color="#1D9E75", alpha=0.8)
ax.axhline(0, color="#333", lw=0.8)
ax.set_title("Playoff home margin minus regular-season home margin, by year")
ax.set_xlabel("Season start year")
ax.set_ylabel("Margin gap (pts)")
fig.tight_layout()
fig.savefig(FIG_DIR / "3_playoff_vs_regular_gap.png", dpi=130)
plt.close(fig)


# %%
# ----------------------------------------------------------------------
# S4. Four-factor decomposition of the home edge
# ----------------------------------------------------------------------
factors = ["efg_diff", "oreb_diff", "tov_diff", "ftr_diff"]
print("Mean home-team four-factor edge in the playoffs (ex-bubble):")
for f in factors:
    print(f"  {f:10s}: {po[f].mean():+.4f}")

# Which factor edge most drives the home margin. Standardize for comparable
# coefficients, then least-squares regress margin on the four diffs.
X = po[factors].to_numpy()
Xz = (X - X.mean(axis=0)) / X.std(axis=0)
A = np.column_stack([np.ones(len(Xz)), Xz])
coef, *_ = np.linalg.lstsq(A, po["margin"].to_numpy(), rcond=None)
print("Standardized drivers of home margin (pts per 1 SD of the edge):")
for name, c in zip(factors, coef[1:], strict=False):
    print(f"  {name:10s}: {c:+.2f}")

# Trend of the free-throw-rate edge, the crowd/officiating tell.
ftr_season = po.groupby("start_year")["ftr_diff"].mean().reset_index()
ftr_nb = ftr_season[ftr_season["start_year"] != season_start_year(BUBBLE_SEASON)]
ftr_slope, _ = np.polyfit(ftr_nb["start_year"], ftr_nb["ftr_diff"], 1)
print(f"  home free-throw-rate edge trend: {ftr_slope * 10:+.4f} per decade")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ftr_season["start_year"], ftr_season["ftr_diff"], marker="o", color="#EF9F27")
ax.axhline(0, color="#888780", ls="--", lw=0.8)
ax.set_title("Home free-throw-rate edge in the playoffs by season")
ax.set_xlabel("Season start year")
ax.set_ylabel("Home FTR minus away FTR")
fig.tight_layout()
fig.savefig(FIG_DIR / "4_ftr_edge_trend.png", dpi=130)
plt.close(fig)


# %%
# ----------------------------------------------------------------------
# S5. The 2020 bubble natural experiment
# The processed layer flags every bubble game neutral and drops the home flag,
# so the designated home team is recovered from the raw schedule MATCHUP field
# ("vs." marks the home side). The bubble's designated home kept the procedural
# perks (last change, ball, bench) but had no crowd and no travel, so its win
# rate against the neighboring crowd-and-travel seasons isolates the share of
# home court that the building, not the bracket, supplies.
# ----------------------------------------------------------------------
RAW_DIR = Path("~/Projects/nba-four-factors/data/raw/leaguegamelog").expanduser()


def designated_home(season: str) -> pd.DataFrame:
    """game_id -> home team_abbr, parsed from the raw playoff MATCHUP strings."""
    import json

    path = RAW_DIR / season / "playoffs.json"
    if not path.exists():
        return pd.DataFrame(columns=["game_id", "home_abbr"])
    rs = json.loads(path.read_text())["resultSets"][0]
    cols = rs["headers"]
    gi, mi = cols.index("GAME_ID"), cols.index("MATCHUP")
    ti = cols.index("TEAM_ABBREVIATION")
    rows = [{"game_id": r[gi], "home_abbr": r[ti]} for r in rs["rowSet"] if "vs." in r[mi]]
    return pd.DataFrame(rows)


neighbors = po_season[po_season["start_year"].between(2017, 2022)]
dh = designated_home(BUBBLE_SEASON)
bubble_rows = playoffs[(playoffs["season"] == BUBBLE_SEASON)].merge(dh, on="game_id", how="inner")
bub = bubble_rows[bubble_rows["team_abbr"] == bubble_rows["home_abbr"]]
print("2020 bubble (neutral site, no crowd, no travel):")
if len(bub):
    print(f"  bubble designated-home win pct: {(bub['margin'] > 0).mean():.3f}")
    print(f"  bubble designated-home margin:  {bub['margin'].mean():+.2f}")
else:
    print("  bubble home designation unavailable (raw schedule not found).")
print(f"  neighboring 2017-2022 mean home win pct: {neighbors['home_win_pct'].mean():.3f}")
print(f"  neighboring 2017-2022 mean home margin:  {neighbors['home_margin'].mean():+.2f}")
print("  The drop from neighbors to bubble estimates the crowd-plus-travel share of HCA.")


# %%
# ----------------------------------------------------------------------
# S6. Travel. Does the visitor's recent travel load inflate the home margin?
# ----------------------------------------------------------------------
tcols = ["game_id", "team_id", "miles_7d", "days_rest", "is_b2b", "leg_miles"]
home_t = travel[tcols].rename(columns={c: f"home_{c}" for c in tcols[2:]})
away_t = travel[tcols].rename(
    columns={"team_id": "opp_team_id", **{c: f"away_{c}" for c in tcols[2:]}}
)
pt = po.merge(home_t, on=["game_id", "team_id"], how="left").merge(
    away_t, on=["game_id", "opp_team_id"], how="left"
)
pt["away_extra_miles_7d"] = pt["away_miles_7d"] - pt["home_miles_7d"]
pt["away_rest_deficit"] = pt["home_days_rest"] - pt["away_days_rest"]
sub = pt.dropna(subset=["away_extra_miles_7d", "away_rest_deficit"])
Z = np.column_stack(
    [
        np.ones(len(sub)),
        (sub["away_extra_miles_7d"] - sub["away_extra_miles_7d"].mean())
        / sub["away_extra_miles_7d"].std(),
        (sub["away_rest_deficit"] - sub["away_rest_deficit"].mean())
        / sub["away_rest_deficit"].std(),
    ]
)
tcoef, *_ = np.linalg.lstsq(Z, sub["margin"].to_numpy(), rcond=None)
print("Travel and the home margin (playoffs):")
print(f"  baseline home margin net of travel terms (intercept): {tcoef[0]:+.2f} pts")
print(f"  per 1 SD extra visitor 7-day miles: {tcoef[1]:+.2f} pts")
print(f"  per 1 SD home rest advantage:       {tcoef[2]:+.2f} pts")


# %%
# ----------------------------------------------------------------------
# S7. Seed and team quality. Separate home court from being the better team.
# ----------------------------------------------------------------------
reg_wins = (
    regular[~regular["is_neutral"]]
    .assign(win=lambda d: (d["margin"] > 0).astype(int))
    .groupby(["season", "team_id"], as_index=False)["win"]
    .sum()
    .rename(columns={"win": "reg_wins"})
)
po_q = po.merge(reg_wins, on=["season", "team_id"], how="left").merge(
    reg_wins.rename(columns={"team_id": "opp_team_id", "reg_wins": "opp_reg_wins"}),
    on=["season", "opp_team_id"],
    how="left",
)
po_q["win_gap"] = po_q["reg_wins"] - po_q["opp_reg_wins"]
sub2 = po_q.dropna(subset=["win_gap"])
W = np.column_stack([np.ones(len(sub2)), sub2["win_gap"].to_numpy()])
wcoef, *_ = np.linalg.lstsq(W, sub2["margin"].to_numpy(), rcond=None)
print("Home margin versus regular-season win gap (host minus visitor):")
print(f"  pure home court, equal teams (intercept): {wcoef[0]:+.2f} pts")
print(f"  per extra regular-season win of quality:  {wcoef[1]:+.3f} pts")
print(f"  raw mean home margin for comparison:      {po['margin'].mean():+.2f} pts")


# %%
# ----------------------------------------------------------------------
# S8. Era synthesis table
# ----------------------------------------------------------------------
def era_label(yr: int) -> str:
    for lo, hi in ERA_BOUNDS:
        if lo <= yr <= hi:
            return f"{lo}-{hi}"
    return "other"


po_nb = po[po["season"] != BUBBLE_SEASON].copy()
po_nb["era"] = po_nb["start_year"].map(era_label)
rs_nb = rs.copy()
rs_nb["era"] = rs_nb["start_year"].map(era_label)
era_tbl = po_nb.groupby("era").agg(
    games=("margin", "size"),
    po_home_win=("home_win", "mean"),
    po_home_margin=("margin", "mean"),
)
era_tbl["rs_home_margin"] = rs_nb.groupby("era")["margin"].mean()
era_tbl["playoff_premium"] = era_tbl["po_home_margin"] - era_tbl["rs_home_margin"]
print("Era synthesis (bubble excluded):")
print(era_tbl.round(3).to_string())

print("\nDone. Figures in", FIG_DIR)
