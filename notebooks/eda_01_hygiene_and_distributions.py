# %% [markdown]
# # EDA 01: Data Hygiene and Marginal Distributions
#
# **Session 12, Day 2.** Verifying the processed layer's structural
# claims and looking at marginal distributions of the four factors
# across 13 seasons (2012_13 through 2024_25, regular season).
#
# **Three-bullet summary** (filled in after running):
#
# 1. _(TBD)_
# 2. _(TBD)_
# 3. _(TBD)_

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from nba_four_factors.analysis import (
    load_processed,
    pivot_to_game_level,
)
from nba_four_factors.config import SeasonType

sns.set_theme(style="whitegrid", context="notebook")
FIG_DIR = Path("figures/eda_01")
FIG_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

# %%
df = load_processed(("2012_13", "2024_25"), SeasonType.REGULAR)
print(f"long-format shape: {df.shape}")
print(f"seasons present:   {sorted(df['season'].unique())}")
print(f"date range:        {df['game_date'].min()} to {df['game_date'].max()}")

# %% [markdown]
# ## Section 1: Data Hygiene
#
# Per Session 12 §2.1. Each subsection either passes silently or
# surfaces an anomaly. The Bubble flag check at the end is the
# high-value verification.

# %% [markdown]
# ### 1.1 Per-season row counts
#
# Expected: ~2460 team-game rows per full season (1230 games x 2).
# 2019_20 short due to COVID shutdown (~1940 rows).
# 2020_21 short due to compressed 72-game schedule (~2160 rows).

# %%
season_counts = df.groupby("season").size().rename("team_game_rows")
games_per_season = (season_counts / 2).rename("games")
season_summary = pd.concat([season_counts, games_per_season], axis=1)
print(season_summary)

# %%
fig, ax = plt.subplots(figsize=(11, 4))
games_per_season.plot.bar(ax=ax, color="steelblue", edgecolor="black")
ax.axhline(1230, color="gray", linestyle="--", alpha=0.6, label="full season (1230)")
ax.set_ylabel("games")
ax.set_title("Games per season (regular season only)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "01_games_per_season.png", dpi=150)
plt.show()

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
# ### 1.3 Score sanity
#
# Margin should be bounded. Anything beyond 60 is rare but possible
# (the all-time NBA record is 73, Memphis vs OKC, Dec 2021).
# Anything below 1 in absolute value would mean a tie, which is
# impossible under modern OT rules.

# %%
score_summary = df.groupby("season")["margin"].agg(["min", "max", "mean", "std"])
print(score_summary)

# %%
zero_margin = (df["margin"] == 0).sum()
print(f"\nrows with margin == 0: {zero_margin}")
extreme_margin = df[df["margin"].abs() > 50]
print(f"rows with |margin| > 50: {len(extreme_margin)}")
if len(extreme_margin) > 0:
    print("largest blowouts:")
    print(
        extreme_margin.nlargest(5, "margin")[
            ["game_date", "team_abbr", "opp_abbr", "pts", "opp_pts", "margin"]
        ]
    )

# %% [markdown]
# ### 1.4 Team count per season
#
# Should be exactly 30 every season since 2004_05 (Charlotte expansion).

# %%
teams_per_season = df.groupby("season")["team_abbr"].nunique().rename("teams")
print(teams_per_season)
assert (teams_per_season == 30).all(), "season(s) without exactly 30 teams"

# %% [markdown]
# ### 1.5 Schedule symmetry
#
# Each team should have approximately equal home and away game counts
# in a season. Small deviations are normal (Bubble, neutral-site games).

# %%
home_away = (
    df.groupby(["season", "team_abbr", "is_home"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={False: "away", True: "home"})
)
home_away["diff"] = home_away["home"] - home_away["away"]
worst_imbalance = home_away.reindex(
    home_away["diff"].abs().sort_values(ascending=False).index
).head(10)
print("largest home/away imbalances:")
print(worst_imbalance)

# %% [markdown]
# ### 1.6 Neutral-site flag distribution
#
# Expected: ~5 per season since 2023_24 (NBA Cup semifinals/final in
# Las Vegas), 0 before that, occasional Mexico City and London games
# scattered earlier. Each neutral game contributes 2 team-rows.

# %%
neutral_by_season = df[df["is_neutral"]].groupby("season").size().rename("neutral_team_rows")
neutral_games_per_season = (neutral_by_season / 2).rename("neutral_games")
print(pd.concat([neutral_by_season, neutral_games_per_season], axis=1).fillna(0).astype(int))

# %% [markdown]
# ### 1.7 Bubble check (high-value verification)
#
# **2020_21** was the season immediately AFTER the Bubble; the Bubble
# was the playoffs portion of **2019_20** at the end of the regular
# season's interruption. Checking both seasons for neutral-flag
# behavior:
#
# - 2019_20 regular season pre-shutdown games: home/away normal
# - 2019_20 regular season post-shutdown ("seeding games" in Bubble):
#   were these flagged neutral? They were played at Disney with no fans.
# - 2020_21 regular season: full schedule, mostly empty arenas, but
#   played at home venues.
#
# If 2019_20 seeding games are NOT flagged neutral, that's a discovery
# that affects HCA computations for that season specifically.

# %%
season_2019_20 = df[df["season"] == "2019_20"]
print("2019_20 regular season:")
print(f"  total team-game rows:  {len(season_2019_20)}")
print(f"  is_neutral=True rows:  {(season_2019_20['is_neutral']).sum()}")
print(
    f"  date range:            {season_2019_20['game_date'].min()} to {season_2019_20['game_date'].max()}"
)

# Bubble seeding games started July 30, 2020. Anything in 2019_20 played
# from late July onward was at Disney.
post_shutdown = season_2019_20[season_2019_20["game_date"] >= "2020-07-30"]
print("\n2019_20 post-shutdown games (Bubble seeding games):")
print(f"  team-game rows:        {len(post_shutdown)}")
print(f"  is_neutral=True rows:  {(post_shutdown['is_neutral']).sum()}")
print(f"  pct flagged neutral:   {post_shutdown['is_neutral'].mean():.2%}")

# %%
season_2020_21 = df[df["season"] == "2020_21"]
print("2020_21 regular season:")
print(f"  total team-game rows:  {len(season_2020_21)}")
print(f"  is_neutral=True rows:  {(season_2020_21['is_neutral']).sum()}")
print(
    f"  date range:            {season_2020_21['game_date'].min()} to {season_2020_21['game_date'].max()}"
)

# %% [markdown]
# **Interpretation note** (fill in after running):
#
# - If 2019_20 post-shutdown games are flagged neutral: data is correct,
#   HCA pipelines will naturally exclude them.
# - If NOT flagged neutral: those ~88 games per team will contribute to
#   the 2019_20 HCA estimate as if they were normal home games. The
#   2019_20 HCA number then needs an asterisk in the thesis chapter,
#   or a separate exclusion filter. Note this in the three-bullet
#   summary at the top.

# %% [markdown]
# ## Section 2: Marginal Distributions
#
# Per Session 12 §2.2. Histograms of the four offensive factors per
# season, plus margin. Calibration check that factor calculations are
# producing physically plausible numbers and a first visual on whether
# the modernization story is in the data.

# %% [markdown]
# ### 2.1 Offensive four factors: per-season summary stats
#
# Expectations:
# - eFG% centers around 0.50, drifting up over 13 years
# - TOV% around 0.13, slow decline
# - ORB% around 0.22 to 0.28, declining as 3-point shooting rises
# - FT rate around 0.20 to 0.25, year-to-year fluctuation no strong trend

# %%
factor_cols = ["off_efg_pct", "off_tov_pct", "off_orb_pct", "off_ft_rate"]
season_factor_means = df.groupby("season")[factor_cols].mean().round(4)
print(season_factor_means)

# %% [markdown]
# ### 2.2 Distributions: violin per season per factor
#
# Each violin shows the distribution of team-game values within a
# season. Looking for centering, spread, and trend across seasons.

# %%
factor_labels = {
    "off_efg_pct": "Effective FG%",
    "off_tov_pct": "Turnover %",
    "off_orb_pct": "Offensive Rebound %",
    "off_ft_rate": "FT Rate (FTA/FGA)",
}

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for ax, col in zip(axes.ravel(), factor_cols, strict=False):
    sns.violinplot(
        data=df,
        x="season",
        y=col,
        ax=ax,
        inner="quartile",
        linewidth=0.8,
        cut=0,
    )
    ax.set_title(factor_labels[col])
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Offensive four factors: per-season distributions", y=1.00, fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_offensive_factor_violins.png", dpi=150)
plt.show()

# %% [markdown]
# ### 2.3 Defensive four factors: same shape, mirrored expectations
#
# By construction, league-wide def_X distribution mirrors league-wide
# off_X distribution (every team's defense is the league's offense
# against them). Verifying this holds and the dtypes are clean.

# %%
def_factor_cols = ["def_efg_pct", "def_tov_pct", "def_orb_pct", "def_ft_rate"]
print("league-wide means by season (offensive vs defensive):")
comparison = pd.concat(
    [
        df.groupby("season")[factor_cols].mean().add_prefix(""),
        df.groupby("season")[def_factor_cols].mean().add_prefix(""),
    ],
    axis=1,
)
print(comparison.round(4))

# %%
print("league-wide difference (off mean minus def mean) per season:")
for off_col, def_col in zip(factor_cols, def_factor_cols, strict=False):
    diff = (df.groupby("season")[off_col].mean() - df.groupby("season")[def_col].mean()).abs().max()
    print(f"  {off_col} vs {def_col}: max abs diff = {diff:.6f}")

# %% [markdown]
# Differences should be effectively zero (well below 1e-10) since by
# construction `mean(off_X) == mean(def_X)` league-wide. Anything
# larger means there is asymmetric data filtering somewhere upstream.

# %% [markdown]
# ### 2.4 Margin distribution
#
# Roughly normal, centered slightly above zero (HCA), std dev around
# 13 to 14 points. Q-Q plot vs normal looks for fat tails (blowouts).

# %%
games = pivot_to_game_level(df)
print(f"game-level shape after pivot: {games.shape}")
print("\nhome margin summary stats:")
print(games["home_margin"].describe())
print(f"\nhome win rate: {games['home_win'].mean():.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(games["home_margin"], bins=60, kde=True, ax=axes[0], color="steelblue")
axes[0].axvline(0, color="black", linestyle="--", alpha=0.6, label="zero")
axes[0].axvline(
    games["home_margin"].mean(),
    color="firebrick",
    linestyle="-",
    alpha=0.8,
    label=f"mean = {games['home_margin'].mean():.2f}",
)
axes[0].set_title("Home margin distribution (all seasons pooled)")
axes[0].set_xlabel("home margin (pts)")
axes[0].legend()


stats.probplot(games["home_margin"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot vs normal")

plt.tight_layout()
plt.savefig(FIG_DIR / "03_margin_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ### 2.5 Per-season margin std (competitiveness over time)
#
# Has the league become more or less competitive game-to-game? Lower
# std = more parity in single-game outcomes. Higher std = more blowouts
# and close games coexisting.

# %%
margin_std_by_season = games.groupby("season")["home_margin"].agg(["mean", "std"])
print(margin_std_by_season.round(2))

# %%
fig, ax = plt.subplots(figsize=(11, 4))
margin_std_by_season["std"].plot(ax=ax, marker="o", color="darkorange")
ax.set_ylabel("std of home margin (pts)")
ax.set_title("Single-game margin variance by season (proxy for competitive parity)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_margin_std_by_season.png", dpi=150)
plt.show()

# %% [markdown]
# ## Three-bullet summary
#
# After running the cells above, fill these in (also at the top of
# the notebook):
#
# 1. _Hygiene status: pass / fail. Did anything unexpected surface?_
# 2. _Bubble flagging: were 2019_20 post-shutdown games marked neutral?_
# 3. _Modernization visible? eFG% trend and ORB% trend in the violins?_
# %%
