# %% [markdown]
# # EDA 02: Marginal Distributions of the Four Factors (29-season)
#
# **Session 12, post-backfill.** Marginal distributions of the four
# offensive and defensive factors across 29 seasons (1997_98 to 2025_26,
# regular season). Calibration check that factor calculations land at
# physically plausible values, plus first visual on whether the
# modernization story is present in the raw data.
#
# **Three-bullet summary:**
#
# 1. Modernization story confirmed and refined. eFG% rose 8pp across 29
#    seasons (0.479 to 0.547) with sharpest acceleration 2014-2019
#    (Warriors era). ORB% shows a U-shape: long decline 1997-2021 then
#    clear reversal 2022-2026 back to ~2010-11 levels. FT rate fell 27%
#    from 1997 baseline (0.337 to 0.246). TOV% slow gradual decline (no
#    era inflection).
# 2. FTA/FGA and FTM/FGA correlate at 0.94 across all 29 seasons with
#    stable correlation. The ~12% independent variance in FTM/FGA
#    represents team FT shooting skill, conceptually distinct from
#    "aggressiveness in drawing fouls". Decision, as corrected in the
#    2b Results block below: the study uses makes-based FTM/FGA, chosen
#    by the Session 14 out-of-sample head-to-head over the attempt-based
#    version initially preferred here. FT rate is bounded in
#    practice: FTA exceeds FGA only once in 68,716 team-game rows
#    (Hack-a-Shaq game, Nov 19, 1999).
# 3. Headline visualization: dual-axis HCA mean vs std by season. "X
#    shape" emerges post-2010. HCA declining (3.4 to 1.7) while std
#    rising (12.5 to 16.4). Signal shrinking while noise rises. This is
#    the centerpiece chart for the thesis introduction.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy import stats
from thesis_style import apply_thesis_style

from nba_four_factors.analysis import load_processed, pivot_to_game_level
from nba_four_factors.config import SeasonType

sns.set_theme(style="whitegrid", context="notebook")

apply_thesis_style()
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
# **Results:** All four factors show meaningful era trends across 29
# seasons. eFG% climbs from 0.479 (1997_98) to 0.547 (2025_26), an
# 8-percentage-point swing. The climb is gradual 1997-2013, accelerates
# sharply 2014-2019 (Warriors dynasty era), then continues modest
# growth post-2020. TOV% drifts down from 0.145 to 0.127 across the
# era, smallest movement of the four. ORB% shows a striking
# non-monotonic pattern: gradual decline 1997-2003 from 0.286 toward
# 0.270, leveling off 2004-2011, fast decline 2012-2021 to 0.220
# minimum, then clear reversal 2022-2026 back to 0.258 (near 2010_11
# levels). FT rate falls 27% from 0.337 (1997_98) to 0.246 (2024_25)
# with a slight uptick to 0.267 in 2025_26. The 2025_26 ORB% and FT
# rate upticks may reflect tactical shifts post-Thunder/Pacers Finals
# (crash the offensive glass + accept transition fouls).

# %% [markdown]
# ## Section 2: Per-season violin distributions
#
# Each violin shows the within-season distribution of team-game values.
# Looking for: centering, spread, era inflections, and clear directional
# drift across 29 seasons.

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
# **Results:** The violins confirm the per-season means trend visually
# and add texture. eFG% violins widen in recent years, suggesting the
# gap between worst and best team-games is growing (by 2025_26 upper
# whiskers reach 80%, lower 30%). TOV% violins narrow slightly in
# recent years, indicating more consistent ball security across the
# league. ORB% violins show the U-shape: high medians 1997-2003 (28-30%),
# bottoming out 2019-2021, recent climb back to 26%. The lower tails
# (teams completely abandoning offensive rebounding) were fatter during
# the decline era; in 2025_26 the bottom tails are tighter, suggesting
# the league-wide shift back toward crashing the glass. FT rate violins
# in 2025_26 are much tighter than 1997-2005, meaning teams cluster
# closer to league average than they used to. Outlier "high FT rate"
# teams are rarer now.

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
# The processed layer stores the FTA/FGA rate column plus raw FTM/FTA
# counts; the modeling feature layer computes FTM/FGA from the stored
# makes. For comparison, we compute FTM/FGA here and inspect the trend.
# Resolution: the choice was made consciously and empirically. The
# Session 14 out-of-sample head-to-head selected FTM/FGA as the
# study's fourth factor; see the corrected Results block below.

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
# **Results:** FTA/FGA and FTM/FGA correlate at 0.9415 league-wide
# across all 29 seasons. Per-season correlations are tightly clustered
# between 0.929 and 0.946 with no era drift. This means the FT
# shooting skill component (FTM/FGA - FTA/FGA gap) has roughly
# constant relative size across all 29 seasons. The two formulations
# capture distinct concepts: FTA/FGA measures aggressiveness in
# drawing fouls; FTM/FGA combines aggressiveness with shooting skill.
# The 0.94 correlation means 88% shared variance, leaving 12% as
# independent FT shooting skill.
#
# Initial decision recorded here: use attempt-based FTA/FGA, on the
# reasoning that it isolates aggressiveness in drawing fouls, treating
# FT shooting as a separate finishing skill, so folding shooting skill
# into the factor blurs what it is meant to isolate. Note the section
# header above is the accurate attribution: Oliver's original
# formulation was makes-based FT/FGA, so the initial preference here
# was a deliberate departure from Oliver, not fidelity to him.
#
# CORRECTION (Session 14, supersedes the above): the out-of-sample
# head-to-head in modeling/cv_vegas.py reversed this decision. On
# matched games with a validated closing spread, swapping makes-based
# FTM/FGA in for attempt-based FTA/FGA improved margin MAE (10.144 vs
# 10.156 for the four-factor models; 10.092 vs 10.105 for the full
# models) and win log loss (0.6127 vs 0.6134; 0.6097 vs 0.6105),
# consistently across test seasons. The 12% independent FT shooting
# skill variance carries predictive signal, so the study uses FTM/FGA
# as the fourth factor. See docs/Session14_vegas_headtohead.md.
#
# The 2025_26 slight uptick is present in both
# formulations but more pronounced in FTA/FGA, suggesting the increase
# is driven by aggressiveness, not shooting skill.

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
# **Results:** FT rate is bounded in practice despite being unbounded
# in theory. Across all 68,716 team-game rows across 29 seasons, FTA
# exceeds FGA exactly once: November 19, 1999, Lakers vs Bulls (LAL 103,
# CHI 95), where the Lakers attempted 64 free throws against 63 field
# goals. This is the canonical Hack-a-Shaq game: peak-prime Shaquille
# O'Neal (career 50% FT shooter, near-automatic from the field) being
# sent to the line repeatedly by the post-Jordan-retirement Bulls (17-65
# that year). Lakers shot 43/64 from the line (67%, well below league
# average, consistent with a Shaq-heavy free throw distribution). FTM
# > FGA never happens. Practical implication: the four-factor framework
# treats FT rate as effectively in [0, 1] for all real games, and the
# theoretical upper bound doesn't constrain the analysis.

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
# **Results:** Mirror check passes cleanly. eFG%, TOV%, and FT rate
# show exact zero difference (computed from team-side stats only, so
# the league-wide aggregation is symmetric by construction). ORB% shows
# 5.55e-17 max difference, which is float64 machine epsilon and arises
# because ORB% involves a division using opponent-side DREB data
# (different order of operations in the per-row computation introduces
# tiny rounding noise that survives at the seventeenth decimal place).
# Confirms the processed-layer factor computation is symmetric and
# unbiased. Useful for thesis methodology: the symmetric regression in
# Session 13.3 assumes mean(off_X) = mean(def_X) league-wide, and this
# verification provides empirical support.

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
# **Results:** Pooled mean home margin across 29 seasons = 2.77 points.
# This is the single-number HCA estimate across the full modern era,
# cleaner than the canonical "3-point" folklore figure that circulates
# in basketball literature. Pooled home win rate ≈ 0.586 (home team
# wins about 59% of all games). Bimodality persists at full-era scale:
# clear peaks around ±5-9 with a notch near zero, exactly the shape
# predicted by end-of-game foul mechanics (trailing teams intentionally
# fouling to force misses, leading to widening rather than narrowing
# margins late) plus the no-ties rule (overtime resolves to a non-zero
# margin). Right peak is slightly higher than left peak, reflecting
# HCA shifting the entire distribution rightward. Q-Q plot shows the
# distribution is approximately normal through roughly ±3σ with
# mild fat tails beyond that (blowout games more frequent than
# normal would predict, on both ends). For regression purposes, OLS
# standard errors will be approximately valid but slightly underestimate
# prediction-interval coverage at the extremes. Document caveat in
# methodology section.

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

# %% [markdown]
# **Results:** Two distinct eras emerge clearly. Era 1 (1997_98 to
# 2017_18, 21 seasons): std stable around 12-13.7, lowest values
# 2003_04 (12.03) and 2004_05 (12.26). This contradicts the casual
# narrative that variance has been rising since the late 90s; it was
# stable for 21 years. Era 2 (2018_19 to 2025_26, 8 seasons): std
# drifts upward gradually from 13.7 to 16.4, coinciding with the
# three-point revolution and pace acceleration.  The 2022_23 anomaly
# is notable: std drops from 15.3 (2021_22) to 13.7 then rebounds to
# 15.6 (2023_24). The 2022_23 dip coincides with the take-foul rule
# introduction (penalty for transition fouls), suggesting stricter
# officiating reduced blowouts temporarily before tactics adjusted.
# The 2020_21 jump (no/limited crowds) is a real natural-experiment
# signal: empty arenas produced genuinely more random outcomes.

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

# %% [markdown]
# **Results:** The two findings are inversely correlated and the visual
# tells the story compactly. From 1997-98 to 2025-26, HCA mean declines
# from ~3.4 to ~1.7 while margin std rises from ~12.5 to ~16.4. The
# crossover begins around 2011 (matching the variance-rise era from §5)
# and accelerates 2019 onward. Three notable per-season values: 2002_03
# is the all-time HCA peak (3.88 points); 2020_21 is the all-time HCA
# trough (0.94 points, the empty-arenas natural experiment); 2024_25
# is the lowest non-pandemic HCA (1.69) with 2025_26 essentially
# matching it (1.73). The thesis-chapter framing: "Across 29 seasons,
# home court advantage in the NBA has declined from roughly 3.4 points
# (pre-2011 era) to roughly 1.8 points (2023-2026 era), while the
# variance of single-game margins has simultaneously increased from
# ~12.5 points std to ~16.4 points std. Home advantage has become both
# smaller in absolute terms and harder to detect against rising
# game-to-game noise." This dual-effect framing is substantially
# stronger than either finding alone and is the centerpiece chart for
# the thesis introduction.

# %%
