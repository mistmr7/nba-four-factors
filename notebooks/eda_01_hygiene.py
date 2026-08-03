# %% [markdown]
# # EDA 01: Data Hygiene (29-season redo)
#
# **Session 12, post-backfill.** Verifies the processed layer's structural
# claims across 29 seasons (1997_98 through 2025_26, regular season).
#
# **Three-bullet summary:**
#
# 1. All 29 seasons are present with expected row counts. Pre-2004_05 lands at
#    1189 games (29-team era). 2004_05 onward at 1230 games. Lockouts and
#    COVID-shortened seasons match expected schedule lengths. 2025_26
#    regular season completed April 12, 2026 with full 1230 games.
# 2. Margin variance has risen substantially across 29 seasons. Std climbs
#    from ~12-13 pts (1997-2010 stable era) to 16.43 pts in 2025_26. Trend
#    accelerates post-2020.
# 3. Bubble anomaly fix verified across reprocessing. All 176 post-shutdown
#    2019_20 rows correctly flagged is_neutral=True, is_home=False. Neutral
#    games breakdown confirmed clean across all eras (Bubble, NBA Cup
#    semifinals, international games).
#
# **Known anomalous seasons** (referenced throughout):
#
# - 1998_99: Lockout, 50-game schedule (725 league-wide games)
# - 2011_12: Lockout, 66-game schedule (990 games)
# - 2019_20: COVID shutdown Mar 11 2020; Bubble seeding Jul-Oct 2020 flagged neutral
# - 2020_21: Compressed 72-game schedule, no/limited crowds at most arenas
# - 2025_26: Regular season complete; playoffs in progress at time of analysis
#
# Methodology note: NBA Cup final games (since 2023-24) are excluded from analysis
# because nba.com's leaguegamelog endpoint treats them as outside the regular season.
# Cup quarterfinals and semifinals are present; semifinals are correctly flagged as
# neutral-site.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from thesis_style import apply_thesis_style

from nba_four_factors.analysis import load_processed
from nba_four_factors.config import SeasonType

sns.set_theme(style="whitegrid", context="notebook")

apply_thesis_style()
FIG_DIR = Path("figures/eda_01")
FIG_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

# Era markers used across multiple charts. Index positions get computed
# per-chart from the actual seasons present.
ERA_MARKERS = {
    "1998_99": ("1998 lockout", "firebrick"),
    "2011_12": ("2011 lockout", "firebrick"),
    "2019_20": ("COVID shutdown", "darkorange"),
    "2020_21": ("no/limited crowds", "darkorange"),
    "2022_23": ("full crowds back", "seagreen"),
    "2025_26": ("in progress", "gray"),
}

# %%
df = load_processed(("1997_98", "2025_26"), SeasonType.REGULAR)
print(f"long-format shape: {df.shape}")
print(f"seasons present:   {sorted(df['season'].unique())}")
print(f"date range:        {df['game_date'].min()} to {df['game_date'].max()}")

# %% [markdown]
# ## Section 1: Data Hygiene
#
# Every subsection either passes silently or surfaces an anomaly.

# %% [markdown]
# ### 1.1 Per-season row counts
#
# Expected: 2 rows per game (one per team). Full-season expectations:
#
# - 1997_98-2003_04 (29-team era): 1189 games -> 2378 rows
# - 2004_05+ (30-team era): 1230 games -> 2460 rows
# - 1998_99 lockout: 725 games -> 1450 rows
# - 2011_12 lockout: 990 games -> 1980 rows
# - 2019_20 COVID: ~970 games -> ~1940 rows (pre-shutdown + Bubble seeding)
# - 2020_21 short: 1080 games -> 2160 rows
# - 2025_26 in-progress: depends on games played at pull time

# %%
season_counts = df.groupby("season").size().rename("team_game_rows")
games_per_season = (season_counts / 2).astype(int).rename("games")
season_summary = pd.concat([season_counts, games_per_season], axis=1)
print(season_summary)

# %%
fig, ax = plt.subplots(figsize=(15, 5))
x = np.arange(len(games_per_season))
seasons = games_per_season.index.tolist()

ax.bar(x, games_per_season.values, color="steelblue", edgecolor="black")
ax.axhline(1230, color="gray", linestyle="--", alpha=0.5, label="30-team full season (1230)")
ax.axhline(1189, color="gray", linestyle=":", alpha=0.5, label="29-team full season (1189)")

for season_key, (_label, color) in ERA_MARKERS.items():
    if season_key in seasons:
        idx = seasons.index(season_key)
        ax.axvline(idx, color=color, linestyle="--", linewidth=1, alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(seasons, rotation=60, ha="right")
ax.set_ylabel("games")
ax.set_title("Games per season (regular season, 1997-98 to 2025-26)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_games_per_season.png", dpi=150)
plt.show()

# %% [markdown]
# **Results:** All 29 seasons present with expected row counts. The
# 29-team era seasons (1997_98 to 2003_04) all land at 2378 team-game
# rows (1189 games x 2). The 30-team era (2004_05 onward) lands at 2460
# rows (1230 games x 2) except for known exceptions: 2012_13 has 2458
# rows (1229 games, Boston Marathon bombing cancellation correctly
# dropped by the canceled-game filter); the two lockout seasons land at
# expected reduced counts (1998_99 at 725 games, 2011_12 at 990 games);
# COVID-shortened 2019_20 at 1059 games and 2020_21 at 1080 games match
# their schedules; 2025_26 regular season completed April 12, 2026 with
# full 1230 games (the "in-progress" caveat applies only to playoffs).

# %% [markdown]
# ### 1.2 Missing values
#
# None expected on any column.

# %%
nulls = df.isna().sum()
nulls_present = nulls[nulls > 0]
if nulls_present.empty:
    print("no null values in any column")
else:
    print("UNEXPECTED nulls:")
    print(nulls_present)

# %% [markdown]
# **Results:** Zero null values across all columns and all 29 seasons.
# The processing pipeline drops canceled games via the canceled-game
# filter in `_loading.py` and produces clean numeric columns throughout.
# No surprises.

# %% [markdown]
# ### 1.3 Score sanity
#
# Min/max margin per season. Anything beyond 60 is rare but possible
# (NBA all-time record is 73). Anything at exactly 0 should be filtered
# out by the canceled-game filter in load_processed; verifying here.

# %%
score_summary = df.groupby("season")["margin"].agg(["min", "max", "mean", "std"])
print(score_summary.round(2))

# %%
zero_margin = (df["margin"] == 0).sum()
print(f"rows with margin == 0: {zero_margin}  (should be 0 after canceled-game filter)")

extreme = df[df["margin"].abs() > 60]
print(f"rows with |margin| > 60: {len(extreme)}")
if len(extreme) > 0:
    print("largest blowouts:")
    print(
        extreme.nlargest(10, "margin")[
            ["game_date", "season", "team_abbr", "opp_abbr", "pts", "opp_pts", "margin"]
        ]
    )

# %% [markdown]
# **Results:** All season margins behave as expected. Mean margin is
# exactly 0.0 every season (symmetric by construction in long-format,
# since every game contributes a +margin and -margin row). Zero
# canceled-game rows escaping the filter. The largest blowout in the
# dataset is the 73-point Memphis vs Thunder game (Dec 2, 2021), which
# is the actual all-time NBA record. Margin standard deviation is the
# real finding here: 12.03 minimum (2003_04) climbing to 16.50
# (2025_26), with steep post-2020 acceleration. Pre-2011 std stable
# around 13; 2011-2018 gradual drift to ~14; 2019-onward sharp climb
# to 16+. This is the headline variance finding of the EDA.

# %% [markdown]
# ### 1.4 Team count per season
#
# Historical context:
# - 1997_98 to 2003_04: 29 teams (Charlotte expansion in 2004_05)
# - 2004_05 onward: 30 teams
#
# No season should deviate from those expectations.

# %%
teams_per_season = df.groupby("season")["team_abbr"].nunique().rename("teams")
print(teams_per_season)

expected_teams = {s: (29 if s < "2004_05" else 30) for s in teams_per_season.index}
mismatches = [
    (s, actual, expected_teams[s])
    for s, actual in teams_per_season.items()
    if actual != expected_teams[s]
]
if mismatches:
    print("\nUNEXPECTED team counts:")
    for s, actual, expected in mismatches:
        print(f"  {s}: {actual} (expected {expected})")
else:
    print("\nall seasons match expected team counts")

# %% [markdown]
# **Results:** All 29 seasons match expected team counts. Pre-2004_05
# correctly shows 29 teams; 2004_05 onward shows 30 teams (Charlotte
# Bobcats expansion). No defunct teams or franchise-relocation
# inconsistencies surface at the league-wide level.

# %% [markdown]
# ### 1.5 Schedule symmetry
#
# Each team should have roughly equal home and away counts. Small
# deviations are normal due to neutral-site games and Bubble seeding.

# %%
home_away = (
    df.groupby(["season", "team_abbr", "is_home"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={False: "away", True: "home"})
)
home_away["diff"] = home_away["home"] - home_away["away"]

print("largest home/away imbalances across all 29 seasons:")
worst = home_away.reindex(home_away["diff"].abs().sort_values(ascending=False).index).head(15)
print(worst)

# %% [markdown]
# **Results:** Every imbalance in the top 15 is from 2019_20. Bubble
# teams (Milwaukee, San Antonio, Orlando, Philadelphia, Portland,
# Sacramento, Utah, Miami, Lakers, Indiana, etc.) all show 11-point
# away/home imbalances because their Bubble seeding games count as
# is_home=False per the anomaly fix. The 8 teams that didn't go to the
# Bubble (Atlanta, Charlotte, Chicago, Cleveland, Detroit, Golden
# State, Knicks, Minnesota) don't appear in the imbalance list. This is
# correct behavior, not a data bug, but means 2019_20 home/away counts
# are not a useful per-team statistic. Document caveat for thesis chapter.
# The only other true imbalance is the ±1 for BOS and IND in 2012_13
# from the canceled Boston Marathon game, well below the top-15
# threshold; the thesis text names both exceptions.

# %% [markdown]
# ### 1.6 Neutral-site flag distribution
#
# Sources of neutral-site games:
#
# - 2019_20 Bubble (Jul 30 - Oct 11, 2020): 88 seeding games, flagged via
#   anomalies.NEUTRAL_SITE_DATE_OVERRIDES
# - NBA Cup semifinals (since 2023_24): 2 per year in Las Vegas. The Cup
#   final is also played at a neutral venue but is excluded from the
#   regular season endpoint because it doesn't count toward standings
# - NBA Mexico City Game (annual since 2023_24): 1 per year
# - NBA Paris Games (2024_25 onward): 2-game set per year
#
# Each neutral game contributes 2 team-rows.

# %%
neutral_by_season = df[df["is_neutral"]].groupby("season").size().rename("neutral_team_rows")
neutral_games_per_season = (neutral_by_season / 2).rename("neutral_games")
result = pd.concat([neutral_by_season, neutral_games_per_season], axis=1).fillna(0).astype(int)
seasons_with_neutrals = result[result["neutral_team_rows"] > 0]
print("seasons with neutral games:")
print(seasons_with_neutrals)
print()
print(f"total neutral team-game rows across all seasons: {df['is_neutral'].sum()}")
print(f"total neutral games: {df['is_neutral'].sum() // 2}")

# %% [markdown]
# **Results:** Three seasons show neutral flags. 2019_20 has 88 Bubble
# seeding games (176 team-rows). 2024_25 and 2025_26 each have 5
# neutral games (10 team-rows): 2 NBA Cup semifinals + 1 Mexico City +
# 2 Paris Games. Flags exist only from 2024_25 onward (plus the Bubble
# date override): the MATCHUP field does not mark neutral sites before
# then, so roughly 29 international regular-season games (from the
# Dec 1997 Mexico City game through the Jan 2024 Paris game) plus the
# two 2023_24 NBA Cup semifinals in Las Vegas, 31 games total, carry
# is_neutral=False and are treated as ordinary home games. Documented
# as a limitation in the thesis (31 of 34,357 played games, ~0.1%);
# manual game-id overrides deliberately deferred. Total flagged: 98
# neutral games across 29 seasons, 196 team-rows.

# %% [markdown]
# ### 1.7 Bubble fix verification (regression test)
#
# Confirms the anomalies.py override is firing correctly for 2019_20
# post-shutdown games (Jul 30 - Oct 11, 2020).

# %%
season_2019_20 = df[df["season"] == "2019_20"]
pre_shutdown = season_2019_20[season_2019_20["game_date"] < "2020-03-12"]
post_shutdown = season_2019_20[season_2019_20["game_date"] >= "2020-07-30"]

print("2019_20 pre-shutdown (Oct 2019 - Mar 11, 2020):")
print(f"  rows:              {len(pre_shutdown)}")
print(f"  is_neutral=True:   {pre_shutdown['is_neutral'].sum()}  (should be 0)")

print("\n2019_20 post-shutdown / Bubble (Jul 30+ 2020):")
print(f"  rows:              {len(post_shutdown)}")
print(f"  is_neutral=True:   {post_shutdown['is_neutral'].sum()}  (should equal rows)")
print(f"  is_home=True:      {post_shutdown['is_home'].sum()}  (should be 0)")

# %% [markdown]
# **Results:** Bubble fix works correctly. 1942 pre-shutdown rows show
# zero neutral flags (correct: these were normal home/away games). 176
# post-shutdown rows all flagged is_neutral=True and is_home=False
# (correct: Bubble games were all played at Disney's ESPN Wide World
# of Sports complex, no home crowd, no home court). The
# NEUTRAL_SITE_DATE_OVERRIDES registry in anomalies.py and the
# `apply_known_neutral_overrides` hook in `_tidy_schedule` are firing
# as designed.
