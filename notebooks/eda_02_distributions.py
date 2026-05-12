# %% [markdown]
# # EDA 02: Marginal Distributions of the Four Factors (29-season)
#
# **Session 12, post-backfill.** Marginal distributions of the four
# offensive and defensive factors across 29 seasons (1997_98 to 2025_26,
# regular season). Calibration check that factor calculations land at
# physically plausible values, plus first visual on whether the
# modernization story is present in the raw data.
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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy import stats

from nba_four_factors.analysis import load_processed, pivot_to_game_level
from nba_four_factors.config import SeasonType

sns.set_theme(style="whitegrid", context="notebook")
FIG_DIR = Path("figures/eda_02")
FIG_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

ERA_MARKERS = {
    "1998_99": ("1998 lockout", "firebrick"),
    "2011_12": ("2011 lockout", "firebrick"),
    "2019_20": ("COVID shutdown", "darkorange"),
    "2020_21": ("no/limited crowds", "darkorange"),
    "2022_23": ("full crowds back", "seagreen"),
    "2025_26": ("post-COVID", "gray"),
}

# %%
df = load_processed(("1997_98", "2025_26"), SeasonType.REGULAR)
print(f"long-format shape: {df.shape}")
print(f"seasons: {df['season'].nunique()}")

# %% [markdown]
# ## Section 1: Offensive four factors per season
#
# Expectations (canonical NBA values):
#
# - **eFG%**: centers around 0.50, rising over time
# - **TOV%**: ~0.13, slow decline expected
# - **ORB%**: ~0.22-0.28, declining as 3-pt shooting rises
# - **FT rate**: ~0.20-0.30, declining over time

# %%
factor_cols = ["off_efg_pct", "off_tov_pct", "off_orb_pct", "off_ft_rate"]
season_factor_means = df.groupby("season")[factor_cols].mean().round(4)
print(season_factor_means)

# %% [markdown]
# ## Section 2: Per-season violin distributions
#
# Each violin shows the within-season distribution of team-game values.
# Looking for: centering, spread, era inflections, and clear directional
# drift across 29 seasons.

# %%
# %%


factor_labels = {
    "off_efg_pct": "Effective FG%",
    "off_tov_pct": "Turnover %",
    "off_orb_pct": "Offensive Rebound %",
    "off_ft_rate": "FT Rate (FTA/FGA)",
}

fig, axes = plt.subplots(2, 2, figsize=(20, 11))
seasons_ordered = sorted(df["season"].unique())

for ax, col in zip(axes.ravel(), factor_cols, strict=False):
    sns.violinplot(
        data=df,
        x="season",
        y=col,
        order=seasons_ordered,
        ax=ax,
        inner="quartile",
        linewidth=0.6,
        cut=0,
    )
    ax.set_title(factor_labels[col], fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=70, labelsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

fig.suptitle(
    "Offensive four factors: per-season distributions (1997_98 to 2025_26)", y=1.00, fontsize=14
)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_offensive_factor_violins.png", dpi=150)
plt.show()
# %% [markdown]
# ### 2b: FT Rate using FTM/FGA (alternative formulation)
#
# Dean Oliver's original four-factors paper used FTM/FGA (free throws
# MADE per FGA), not FTA/FGA. The two capture different things:
#
# - **FTA/FGA**: aggressiveness in drawing fouls and getting to the line.
#   Decorrelated from FT shooting skill.
# - **FTM/FGA**: aggressiveness AND FT shooting skill rolled into one
#   number. What actually contributes to scoring.
#
# The processed layer currently stores FTA/FGA (per Session 10 §2.4).
# For comparison, we compute FTM/FGA here and inspect the trend.
# If the per-season distributions look qualitatively similar, the
# choice between the two doesn't materially affect downstream
# regressions; if they diverge, we should pick consciously.

# %%
df["off_ftm_rate"] = df["ftm"] / df["fga"].where(df["fga"] != 0)

correlation_per_season = (
    df.groupby("season")[["off_ft_rate", "off_ftm_rate"]].corr().iloc[0::2, -1].droplevel(1)
)
print("Per-season correlation between FTA/FGA and FTM/FGA:")
print(correlation_per_season.round(4))
print(
    f"\nleague-wide correlation across all rows: {df['off_ft_rate'].corr(df['off_ftm_rate']):.4f}"
)

# %%
fig, ax = plt.subplots(figsize=(15, 6))
seasons_ordered = sorted(df["season"].unique())

sns.violinplot(
    data=df,
    x="season",
    y="off_ftm_rate",
    order=seasons_ordered,
    ax=ax,
    inner="quartile",
    linewidth=0.6,
    cut=0,
    color="steelblue",
)
ax.set_title("FT Rate (FTM/FGA): per-season distributions (1997_98 to 2025_26)", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("FTM / FGA")
ax.tick_params(axis="x", rotation=70, labelsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "01b_ftm_rate_violins.png", dpi=150)
plt.show()

# %%
season_means_compare = df.groupby("season")[["off_ft_rate", "off_ftm_rate"]].mean().round(4)
print("Per-season means: FTA/FGA vs FTM/FGA")
print(season_means_compare)

# %% [markdown]
# ### 2c: How often FTA > FGA (and FTM > FGA)?
#
# The FT rate violin (§2 bottom-right) shows team-game values reaching
# above 0.8 in earlier seasons and still above 0.5 recently. This isn't
# necessarily anomalous: FGA doesn't increment on a fouled shot that
# misses, so a team that draws shooting fouls aggressively (and misses
# while fouled) can plausibly accumulate FTA > FGA in a single game.
#
# Worth quantifying how common that is, and whether it shifts across
# the 29-season span. Hypothesis: as FGAs-per-game fall (modernization,
# longer 3-point attempts, less rim attacking) and free-throw rates
# fall too, the relative frequency of FTA > FGA could move either way.

# %%
ftm_gt_fga = (df["ftm"] > df["fga"]).sum()
fta_gt_fga = (df["fta"] > df["fga"]).sum()
ftm_eq_fga = (df["ftm"] == df["fga"]).sum()
fta_eq_fga = (df["fta"] == df["fga"]).sum()

print(f"Total team-game rows: {len(df)}")
print()
print(f"FTM > FGA:   {ftm_gt_fga:>6} rows  ({100 * ftm_gt_fga / len(df):.3f}%)")
print(f"FTM == FGA:  {ftm_eq_fga:>6} rows  ({100 * ftm_eq_fga / len(df):.3f}%)")
print(f"FTA > FGA:   {fta_gt_fga:>6} rows  ({100 * fta_gt_fga / len(df):.3f}%)")
print(f"FTA == FGA:  {fta_eq_fga:>6} rows  ({100 * fta_eq_fga / len(df):.3f}%)")

# %%
by_season = df.groupby("season").apply(
    lambda g: pd.Series(
        {
            "games": len(g),
            "ftm_gt_fga": int((g["ftm"] > g["fga"]).sum()),
            "fta_gt_fga": int((g["fta"] > g["fga"]).sum()),
            "pct_ftm_gt": 100 * (g["ftm"] > g["fga"]).mean(),
            "pct_fta_gt": 100 * (g["fta"] > g["fga"]).mean(),
        }
    ),
    include_groups=False,
)
print(by_season.round(2))

# %%
fig, ax = plt.subplots(figsize=(15, 5))
seasons_list = by_season.index.tolist()
x = np.arange(len(seasons_list))

ax.plot(
    x,
    by_season["pct_fta_gt"].values,
    marker="o",
    linewidth=2.0,
    color="firebrick",
    label="FTA > FGA",
)
ax.plot(
    x,
    by_season["pct_ftm_gt"].values,
    marker="s",
    linewidth=2.0,
    color="steelblue",
    label="FTM > FGA",
)

for season_key, (_label, color) in ERA_MARKERS.items():
    if season_key in seasons_list:
        idx = seasons_list.index(season_key)
        ax.axvline(idx, color=color, linestyle="--", linewidth=1, alpha=0.3)

ax.set_xticks(x)
ax.set_xticklabels(seasons_list, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("% of team-game rows")
ax.set_title("Frequency of FTA > FGA and FTM > FGA per season (1997_98 to 2025_26)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_fta_exceeds_fga_frequency.png", dpi=150)
plt.show()
# %% [markdown]
# ## Section 3: Defensive factors mirror check
#
# By construction, league-wide def_X distribution mirrors league-wide
# off_X distribution: every team's defensive eFG% IS the league's
# offensive eFG% against that team. So league-wide mean(off_X) per
# season should equal league-wide mean(def_X) to numerical precision.
#
# Catches asymmetric data filtering or computation bugs.

# %%
def_factor_cols = ["def_efg_pct", "def_tov_pct", "def_orb_pct", "def_ft_rate"]

print("Max |off_X - def_X| per season (should all be ~0):")
for off_col, def_col in zip(factor_cols, def_factor_cols, strict=False):
    diff = (df.groupby("season")[off_col].mean() - df.groupby("season")[def_col].mean()).abs().max()
    status = "OK" if diff < 1e-6 else "PROBLEM"
    print(f"  {off_col} vs {def_col}: max abs diff = {diff:.2e}  [{status}]")

# %% [markdown]
# ## Section 4: Margin distribution + Q-Q plot (all seasons pooled)
#
# Pooled across 29 seasons. Expected: bimodality with peaks around
# ±6-9 and a notch near zero (end-of-game foul mechanics + no-ties rule).
# Q-Q tests approximate normality with mild fat tails from blowouts.

# %%
games = pivot_to_game_level(df)
print(f"game-level shape after pivot: {games.shape}")
print("\nhome margin summary stats (all 29 seasons pooled):")
print(games["home_margin"].describe().round(3))
print(f"\nhome win rate (pooled): {games['home_win'].mean():.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

bin_edges = np.arange(-75, 76, 2)
sns.histplot(games["home_margin"], bins=bin_edges, kde=True, ax=axes[0], color="steelblue")
axes[0].axvline(0, color="black", linestyle="--", alpha=0.6, label="zero")
axes[0].axvline(
    games["home_margin"].mean(),
    color="firebrick",
    linestyle="-",
    alpha=0.8,
    label=f"mean = {games['home_margin'].mean():.2f}",
)
axes[0].set_title("Home margin distribution (29 seasons pooled, 2-pt bins)")
axes[0].set_xlabel("home margin (pts)")
axes[0].legend()

stats.probplot(games["home_margin"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot vs normal")

plt.tight_layout()
plt.savefig(FIG_DIR / "02_margin_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 5: Per-season margin std (competitiveness over time)
#
# From the data hygiene step we noticed std rising substantially,
# especially post-2020. This visualizes the trend across all 29 seasons.

# %%
margin_std_by_season = games.groupby("season")["home_margin"].agg(["mean", "std", "count"])
print(margin_std_by_season.round(3))

# %%
fig, ax = plt.subplots(figsize=(15, 5))
x = np.arange(len(margin_std_by_season))
seasons_list = margin_std_by_season.index.tolist()

ax.plot(
    x,
    margin_std_by_season["std"].values,
    marker="o",
    markersize=7,
    linewidth=2.0,
    color="darkorange",
)

for season_key, (label, color) in ERA_MARKERS.items():
    if season_key in seasons_list:
        idx = seasons_list.index(season_key)
        ax.axvline(idx, color=color, linestyle="--", linewidth=1, alpha=0.4)
        ax.text(
            idx,
            ax.get_ylim()[1] * 0.95 if "lockout" in label else ax.get_ylim()[1] * 0.88,
            label,
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
            color=color,
            alpha=0.7,
        )

ax.set_xticks(x)
ax.set_xticklabels(seasons_list, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("std of home margin (pts)")
ax.set_title("Single-game margin variance by season (1997_98 to 2025_26)")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_margin_std_by_season.png", dpi=150)
plt.show()
# %%
# %% [markdown]
# ### 5b: Mean HCA vs single-game std (dual-axis)
#
# Two findings from §5 visualized together: HCA is declining while
# single-game standard deviation is rising. The "X shape" emerging
# post-2010 is the headline finding of the EDA work.

# %%
fig, ax_mean = plt.subplots(figsize=(15, 6))

x = np.arange(len(margin_std_by_season))
seasons_list = margin_std_by_season.index.tolist()
seasons_display = [s.replace("_", "-") for s in seasons_list]

color_mean = "steelblue"
color_std = "darkorange"

mean_vals = margin_std_by_season["mean"].values
std_vals = margin_std_by_season["std"].values

ax_mean.plot(x, mean_vals, marker="o", markersize=7, linewidth=2.0, color=color_mean)
ax_mean.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax_mean.set_ylabel("Average Home Court Advantage (pts)", color=color_mean, fontsize=11)
ax_mean.tick_params(axis="y", labelcolor=color_mean)
ax_mean.set_ylim(-0.5, 5.5)

ax_std = ax_mean.twinx()
ax_std.plot(x, std_vals, marker="s", markersize=7, linewidth=2.0, color=color_std)
ax_std.set_ylabel("Standard Deviation of Margin (pts)", color=color_std, fontsize=11)
ax_std.tick_params(axis="y", labelcolor=color_std)
ax_std.set_ylim(11.5, 17.5)

era_annotations = {
    "1998_99": "1998-99 Lockout",
    "2011_12": "2011-12 Lockout",
    "2019_20": "COVID shutdown",
    "2020_21": "Empty arenas",
    "2022_23": "Take-foul rule",
}

for season_key, label_text in era_annotations.items():
    if season_key in seasons_list:
        idx = seasons_list.index(season_key)
        color = ERA_MARKERS[season_key][1] if season_key in ERA_MARKERS else "gray"
        ax_mean.axvline(idx, color=color, linestyle="--", linewidth=1, alpha=0.4)
        ax_mean.annotate(
            label_text,
            xy=(idx, 5.5),
            xytext=(idx - 0.12, 5),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            alpha=0.85,
            rotation=90,
        )

ax_mean.set_xticks(x)
ax_mean.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax_mean.set_title(
    "Home Court Advantage Decline and Margin Variability Rise\n1997-98 to 2025-26",
    fontsize=13,
    pad=20,
)

plt.tight_layout()
plt.savefig(FIG_DIR / "04_hca_mean_vs_std_dual_axis.png", dpi=150)
plt.show()
# %%
