# %% [markdown]
# # EDA 03: Cross-Factor Relationships (29-season)
#
# **Session 12, post-EDA 02.** Correlation structure between the four
# offensive factors, their defensive mirrors, and the off-vs-def
# within-team-game joint distribution. Plus per-season correlation
# drift for every offensive-factor pair across 29 seasons.
#
# The four-factor framework implicitly treats the factors as roughly
# independent dimensions of team performance. They are not. eFG% and
# ORB% are mechanically anticorrelated because a made field goal
# produces no offensive rebound opportunity. Beyond mechanics, era-level
# style shifts (the 3-point revolution, glass-crashing trends) can
# amplify or dampen these correlations over time.
#
# This notebook quantifies both: the baseline correlation structure
# pooled across 29 seasons, and the drift over time. Within-season
# correlations are computed at the team-game level (not at the
# team-season-average level); those measure different things and
# within-season is the right one for understanding factor independence
# in regression.
#
# **Three-bullet summary** (filled in after running):
#
# 1. _(TBD)_
# 2. _(TBD)_
# 3. _(TBD)_

# %%
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from nba_four_factors.analysis import load_processed
from nba_four_factors.config import SeasonType

sns.set_theme(style="whitegrid", context="notebook")
FIG_DIR = Path("figures/eda_03")
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

OFF_COLS = ["off_efg_pct", "off_tov_pct", "off_orb_pct", "off_ft_rate"]
DEF_COLS = ["def_efg_pct", "def_tov_pct", "def_orb_pct", "def_ft_rate"]

FACTOR_LABELS = {
    "off_efg_pct": "Off eFG%",
    "off_tov_pct": "Off TOV%",
    "off_orb_pct": "Off ORB%",
    "off_ft_rate": "Off FT rate",
    "def_efg_pct": "Def eFG%",
    "def_tov_pct": "Def TOV%",
    "def_orb_pct": "Def ORB%",
    "def_ft_rate": "Def FT rate",
}

# %%
df = load_processed(("1997_98", "2025_26"), SeasonType.REGULAR)
print(f"long-format shape: {df.shape}")
print(f"seasons: {df['season'].nunique()}")
print(
    f"rows per season range: [{df.groupby('season').size().min()}, {df.groupby('season').size().max()}]"
)

# %% [markdown]
# ## Section 1: Pooled offensive 4x4 correlation matrix
#
# All team-game rows pooled across 29 seasons. Pearson correlation.
#
# Expected signal direction (sign predictions, not magnitudes):
#
# - **eFG% vs ORB%**: negative. Two physical drivers, both pushing
#   the same direction.
#   - Mechanical: a made FG produces no offensive rebound chance.
#     Higher eFG% mechanically lowers ORB%.
#   - Style: teams that shoot more 3s tend to crash the glass less
#     (long rebounds, transition-defense priority). 3-point-heavy
#     teams have higher eFG% and lower ORB%.
# - **eFG% vs FT rate**: ambiguous sign. Older era favored inside
#   scoring (higher FT rate AND moderate eFG%); modern era favors
#   3-point shooting (higher eFG% AND lower FT rate). Pooled across
#   29 seasons these two effects could cancel.
# - **eFG% vs TOV%**: weakly negative. Decision-making quality
#   couples them slightly.
# - **TOV% vs ORB%**, **TOV% vs FT rate**, **ORB% vs FT rate**:
#   expected near zero.
#
# With 68k+ rows, almost any nonzero correlation is "statistically
# significant" by p-value. Focus on effect size:
# `|r| > 0.10` is real signal, `|r| < 0.05` is noise-tier.

# %%
off_corr = df[OFF_COLS].corr()
print("Pooled offensive correlation matrix (29 seasons):")
print(off_corr.round(4))

# %%
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    off_corr,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-0.5,
    vmax=0.5,
    square=True,
    cbar_kws={"label": "Pearson r"},
    xticklabels=[FACTOR_LABELS[c] for c in OFF_COLS],
    yticklabels=[FACTOR_LABELS[c] for c in OFF_COLS],
    ax=ax,
)
ax.set_title("Offensive four-factor correlation matrix\n29 seasons pooled (1997-98 to 2025-26)")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_pooled_offensive_correlation.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 2: Pooled defensive 4x4 correlation matrix (mirror check)
#
# By construction the universe of def_X values is a permutation of the
# universe of off_X values (each game contributes both teams to both
# pools), so the def-vs-def correlation matrix should equal the
# off-vs-off matrix to numerical precision. This is a sanity check
# analogous to EDA 02 §3, extended from marginals to joint distributions.

# %%
def_corr = df[DEF_COLS].corr()
print("Pooled defensive correlation matrix:")
print(def_corr.round(4))

diff_matrix = off_corr.values - def_corr.values
print(f"\nMax abs diff between off and def correlation matrices: {np.abs(diff_matrix).max():.2e}")
status = "OK" if np.abs(diff_matrix).max() < 1e-6 else "PROBLEM"
print(f"Mirror check: [{status}]")

# %%
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    def_corr,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-0.5,
    vmax=0.5,
    square=True,
    cbar_kws={"label": "Pearson r"},
    xticklabels=[FACTOR_LABELS[c] for c in DEF_COLS],
    yticklabels=[FACTOR_LABELS[c] for c in DEF_COLS],
    ax=ax,
)
ax.set_title("Defensive four-factor correlation matrix\n29 seasons pooled (mirror of Section 1)")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_pooled_defensive_correlation.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 3: Off vs Def 8x8 within-team-game correlation matrix
#
# The 8x8 matrix combines Sections 1, 2, and the off-vs-def cross-block.
# The upper-left 4x4 quadrant is the offensive correlation from §1.
# The lower-right 4x4 is the defensive correlation from §2.
# The off-diagonal 4x4 quadrants are the new structure: within a single
# team-game, how does this team's offense correlate with the opponent's
# offense (which is this team's defense)?
#
# Candidate mechanisms for off-vs-def correlation in a single team-game.
# Note: §3 cannot discriminate between these. Each would predict the
# same observed correlation pattern. Mechanism identification requires
# data not in the four-factor box-score pipeline (ref crew assignments,
# play-by-play, possession counts, etc.).
#
# 1. Pace coupling: both teams play in the same possession-paced game.
#    Higher pace can lift both teams' shooting (transition opportunities)
#    or both teams' turnover rates (faster decisions).
# 2. Officiating crew: per-game referee scalar affects both teams'
#    FT rates and possibly TOV% (call frequency).
# 3. Behavioral coupling: physical play begets physical play. One
#    team's aggression draws matching aggression, lifting both teams'
#    FT rates regardless of refs.
# 4. Score-state coupling: blowouts produce garbage-time stats that
#    affect both sides asymmetrically.
# 5. Opponent-quality coupling: strong defenses on both sides
#    produce simultaneous low-eFG games without any pace or ref
#    involvement.
# 6. Style-matchup coupling: certain matchup types produce certain
#    joint statistical signatures (e.g., a slow grind-it-out team
#    playing a pace-and-space team pulls both away from their averages).
#
# Diagonal cross-block entries (off_efg vs def_efg, off_tov vs def_tov,
# etc.) measure whether good shooting/turnover/etc. nights for one team
# coincide with the same for its opponent. Positive diagonal means yes
# (some per-game scalar is at work); negative would mean opponents
# trade off; near zero would mean within-game independence.

# %%
all_cols = OFF_COLS + DEF_COLS
full_corr = df[all_cols].corr()
print("Off-vs-Def 8x8 within-team-game correlation matrix:")
print(full_corr.round(4))

# %%
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    full_corr,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-0.5,
    vmax=0.5,
    square=True,
    cbar_kws={"label": "Pearson r"},
    xticklabels=[FACTOR_LABELS[c] for c in all_cols],
    yticklabels=[FACTOR_LABELS[c] for c in all_cols],
    ax=ax,
)
ax.hlines([4], 0, 8, colors="black", linewidth=1.5)
ax.vlines([4], 0, 8, colors="black", linewidth=1.5)
ax.set_title(
    "All 8 factors: within-team-game correlation matrix\n29 seasons pooled (1997-98 to 2025-26)"
)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_off_vs_def_full_correlation.png", dpi=150)
plt.show()

# %% [markdown]
# ### 3b: Off-vs-Def cross-block isolated
#
# Cleaner view of just the 4x4 cross-block. Each cell is (this team's
# offense factor X) correlated with (opponent's offense factor Y, which
# is this team's defense factor Y) within the same game row.

# %%
cross_block = full_corr.loc[OFF_COLS, DEF_COLS]
print("Off-vs-Def cross-block (4x4):")
print(cross_block.round(4))

# %%
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cross_block,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-0.2,
    vmax=0.2,
    square=True,
    cbar_kws={"label": "Pearson r"},
    xticklabels=[FACTOR_LABELS[c] for c in DEF_COLS],
    yticklabels=[FACTOR_LABELS[c] for c in OFF_COLS],
    ax=ax,
)
ax.set_title(
    "Off vs Def cross-block: this team's offense vs same-game opponent offense\n29 seasons pooled, tight color scale"
)
ax.set_xlabel("opponent factor (this team's defense)")
ax.set_ylabel("this team's offensive factor")
plt.tight_layout()
plt.savefig(FIG_DIR / "03b_off_vs_def_cross_block.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 4: eFG% vs ORB% deep dive
#
# The headline negative correlation. Scatter of all team-game rows
# (subsampled for plotting), with the pooled OLS fit overlaid. The
# slope is the rate at which ORB% drops per unit of eFG% gain; the
# physical interpretation is "fraction of available offensive rebounds
# lost per percentage-point gain in scoring efficiency."

# %%
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(df), size=10000, replace=False)
df_sample = df.iloc[sample_idx]

fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(
    data=df_sample,
    x="off_efg_pct",
    y="off_orb_pct",
    alpha=0.15,
    s=10,
    ax=ax,
    color="steelblue",
    edgecolor=None,
)

slope, intercept = np.polyfit(df["off_efg_pct"], df["off_orb_pct"], 1)
x_line = np.linspace(df["off_efg_pct"].quantile(0.001), df["off_efg_pct"].quantile(0.999), 100)
ax.plot(
    x_line,
    slope * x_line + intercept,
    color="firebrick",
    linewidth=2,
    label=f"OLS fit: orb = {slope:.3f} * efg + {intercept:.3f}",
)

r_pool = df["off_efg_pct"].corr(df["off_orb_pct"])
ax.set_title(f"eFG% vs ORB% scatter (10k sample, 29 seasons pooled)\nPearson r = {r_pool:.4f}")
ax.set_xlabel("Offensive eFG%")
ax.set_ylabel("Offensive ORB%")
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_efg_vs_orb_scatter.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 5: Per-season correlation drift, all 6 offensive pairs
#
# Within-season correlations computed at the team-game level
# (group by season, then `.corr()` on each group), plotted across
# 29 seasons. Small-multiples grid, 2 rows by 3 columns, one panel
# per pair.
#
# Two questions the chart should answer:
#
# 1. Is the four-factor framework's assumption of approximate factor
#    independence stable, or has it drifted with the 3-point revolution?
# 2. Specifically for eFG% vs ORB%, is the mechanical floor masked,
#    matched, or amplified by the style trend?
#
# Y-axis is fixed at [-0.5, 0.3] across all panels so eyeballing the
# magnitude across pairs is honest. Era markers carry forward from
# EDA 02.


# %%
def _within_season_corr(df_in: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    return df_in.groupby("season").apply(
        lambda g, a=col_a, b=col_b: g[a].corr(g[b]),
        include_groups=False,
    )


pair_corr = {}
for col_a, col_b in combinations(OFF_COLS, 2):
    pair_corr[f"{col_a}_vs_{col_b}"] = _within_season_corr(df, col_a, col_b)

corr_drift = pd.DataFrame(pair_corr)
print("Per-season within-season correlations, all offensive-factor pairs:")
print(corr_drift.round(4))

# %%
seasons_ordered = sorted(df["season"].unique())
seasons_display = [s.replace("_", "-") for s in seasons_ordered]
x = np.arange(len(seasons_ordered))

fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
pairs_list = list(combinations(OFF_COLS, 2))

for ax, (col_a, col_b) in zip(axes.ravel(), pairs_list, strict=False):
    key = f"{col_a}_vs_{col_b}"
    series = corr_drift[key].reindex(seasons_ordered)
    ax.plot(x, series.values, marker="o", markersize=5, linewidth=1.8, color="steelblue")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    for season_key, (_, color) in ERA_MARKERS.items():
        if season_key in seasons_ordered:
            idx = seasons_ordered.index(season_key)
            ax.axvline(idx, color=color, linestyle="--", linewidth=0.8, alpha=0.3)

    ax.set_title(f"{FACTOR_LABELS[col_a]} vs {FACTOR_LABELS[col_b]}", fontsize=11)
    ax.set_ylim(-0.5, 0.3)

for ax in axes[-1, :]:
    ax.set_xticks(x[::3])
    ax.set_xticklabels(
        [seasons_display[i] for i in range(0, len(seasons_display), 3)],
        rotation=70,
        ha="right",
        fontsize=8,
    )

for ax in axes[:, 0]:
    ax.set_ylabel("Pearson r")

fig.suptitle(
    "Per-season correlation drift: offensive four-factor pairs\n1997-98 to 2025-26, within-season team-game level",
    fontsize=13,
    y=1.00,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_per_season_correlation_drift.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 6: eFG% vs ORB% drift, spotlight
#
# The single most interesting pair from §5 gets its own chart with full
# season labels and era annotations.
#
# Interpretation grid (what each shape would mean):
#
# - **Flat negative line near -0.30 across all 29 seasons**: the
#   mechanical floor dominates, style shifts are weak. Four-factor
#   independence assumption is moderately violated but stably so.
# - **Trending more negative over time**: style and mechanics
#   compounding. The 3-point revolution made the negative coupling
#   stronger, not just changed where the marginals sit.
# - **Trending less negative (toward zero) over time**: style and
#   mechanics partially canceling, possibly because higher-eFG% teams
#   also crash the glass aggressively in the modern era (recent ORB%
#   reversal from EDA 02 §1 hints at this).
# - **U-shape mirroring the ORB% U-shape from EDA 02 §1**: same
#   tactical-era story as the ORB% marginal trend, expressed at the
#   joint-distribution level.

# %%
efg_orb_series = corr_drift["off_efg_pct_vs_off_orb_pct"].reindex(seasons_ordered)

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(x, efg_orb_series.values, marker="o", markersize=7, linewidth=2.0, color="firebrick")
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

y_top = ax.get_ylim()[1]
for season_key, (label, color) in ERA_MARKERS.items():
    if season_key in seasons_ordered:
        idx = seasons_ordered.index(season_key)
        ax.axvline(idx, color=color, linestyle="--", linewidth=1, alpha=0.4)
        ax.annotate(
            label,
            xy=(idx, y_top),
            xytext=(idx - 0.12, y_top * 0.97 if y_top > 0 else y_top * 1.03),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            alpha=0.85,
            rotation=90,
        )

ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("Pearson r (within-season, team-game level)")
ax.set_title(
    "eFG% vs ORB% correlation drift across 29 seasons\nMechanical floor plus era-level style shift"
)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_efg_vs_orb_drift.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 7: Headline numbers for the writeup
#
# Compact print summary. Useful for paste-into-thesis-chapter and for
# the three-bullet summary at the top of this notebook.

# %%
print("Pooled (29-season) offensive correlations, key pairs:")
print(f"  eFG% vs ORB%:    r = {df['off_efg_pct'].corr(df['off_orb_pct']):+.4f}")
print(f"  eFG% vs TOV%:    r = {df['off_efg_pct'].corr(df['off_tov_pct']):+.4f}")
print(f"  eFG% vs FT rate: r = {df['off_efg_pct'].corr(df['off_ft_rate']):+.4f}")
print(f"  ORB% vs TOV%:    r = {df['off_orb_pct'].corr(df['off_tov_pct']):+.4f}")
print(f"  ORB% vs FT rate: r = {df['off_orb_pct'].corr(df['off_ft_rate']):+.4f}")
print(f"  TOV% vs FT rate: r = {df['off_tov_pct'].corr(df['off_ft_rate']):+.4f}")
print()

print("eFG% vs ORB% within-season correlation, first/middle/last 3 seasons:")
print(efg_orb_series.head(3).round(4))
print("...")
print(efg_orb_series.iloc[len(efg_orb_series) // 2 - 1 : len(efg_orb_series) // 2 + 2].round(4))
print("...")
print(efg_orb_series.tail(3).round(4))
print()

print(
    f"Min within-season r (eFG% vs ORB%): {efg_orb_series.min():+.4f} in {efg_orb_series.idxmin()}"
)
print(
    f"Max within-season r (eFG% vs ORB%): {efg_orb_series.max():+.4f} in {efg_orb_series.idxmax()}"
)
print(f"Range: {efg_orb_series.max() - efg_orb_series.min():.4f}")

# %% [markdown]
# ## Followups parked for later
#
# Not done here, worth doing eventually:
#
# 1. **Playoff version of this entire notebook.** Cross-factor
#    correlations could shift in playoffs (slower pace, tighter
#    defenses, smaller sample). Copy this file to
#    `eda_03b_cross_factor_playoffs.py` and change the `load_processed`
#    call to `SeasonType.PLAYOFFS`.
# 2. **Across-teams (season-average) correlations.** Within-season
#    team-game correlations and across-teams season-average
#    correlations are different objects. Within-season captures
#    mechanics + within-season style; across-teams captures pure
#    strategy/style. Worth doing as a supplementary table if the
#    regression chapter wants to discuss factor independence at the
#    team-strategy level.
# 3. **Per-season drift of off-vs-def cross-block pairs.** Section 5
#    only does within-offensive-factor pairs. The 16 off-vs-def
#    cross-block pairs (e.g. off_efg vs def_efg over time) could show
#    pace-related drift worth flagging.
# 4. **Significance-corrected coloring on the cross-block heatmap.**
#    With 68k+ rows, p-values are useless. Effect-size thresholds are
#    used in plain text in the markdown; could be visualized with a
#    custom diverging-with-deadzone colormap on §3b.
# 5. **Bluesky / blog post: mechanism discrimination for the off-vs-def
#    cross-block.** Out of scope for the thesis (too granular, requires
#    data not in this pipeline). Full sketch lives in
#    `docs/blog_ideas.md` entry #1.
