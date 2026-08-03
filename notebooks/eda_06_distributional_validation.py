# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EDA 06: Season win-pct distribution
#
# Section 1 of the win-probability methodology EDA.
#
# Tests three things:
#
# 1. Within-season team win-pcts are approximately normally distributed.
# 2. The standard deviation of within-season win-pct is stable across the 29-season window,
#    or whether per-era treatment is required.
# 3. The z-score framework on win-pct is empirically supported.
#
# Excluded:
#
# - 1998-99 (50-game lockout, no Vegas market in source data)
# - 2025-26 (current season, in progress)
#
# Reference: see eda/eda_specs/season_win_probability_eda.md

# %%
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from thesis_style import apply_thesis_style

from nba_four_factors.analysis import load_processed
from nba_four_factors.config import SeasonType

sns.set_theme(style="whitegrid", context="notebook")

apply_thesis_style()

warnings.filterwarnings(
    "ignore",
    message="As of SciPy 1.17",
    category=FutureWarning,
)

# %% [markdown]
# ## Configuration

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VEGAS_PARQUET = PROJECT_ROOT / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"
FIG_DIR = PROJECT_ROOT / "notebooks" / "figures" / "eda_06"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_SEASONS = {"1998_99", "2025_26"}

GAMES_PER_SEASON = {
    "1997_98": 82,
    "1999_00": 82,
    "2000_01": 82,
    "2001_02": 82,
    "2002_03": 82,
    "2003_04": 82,
    "2004_05": 82,
    "2005_06": 82,
    "2006_07": 82,
    "2007_08": 82,
    "2008_09": 82,
    "2009_10": 82,
    "2010_11": 82,
    "2011_12": 66,
    "2012_13": 82,
    "2013_14": 82,
    "2014_15": 82,
    "2015_16": 82,
    "2016_17": 82,
    "2017_18": 82,
    "2018_19": 82,
    "2019_20": 72,
    "2020_21": 72,
    "2021_22": 82,
    "2022_23": 82,
    "2023_24": 82,
    "2024_25": 82,
}

ERA_TWO_WAY = {
    "pre_2012": lambda s: int(s.split("_")[0]) < 2012,
    "post_2012": lambda s: int(s.split("_")[0]) >= 2012,
}

ERA_THREE_WAY = {
    "pre_modernization": lambda s: int(s.split("_")[0]) < 2013,
    "analytics_era": lambda s: 2013 <= int(s.split("_")[0]) < 2021,
    "post_bubble": lambda s: int(s.split("_")[0]) >= 2021,
}

# %% [markdown]
# ## Load and prepare data

# %%
df_vegas = pd.read_parquet(VEGAS_PARQUET)
print(f"Loaded {len(df_vegas)} team-season rows from Vegas dataset.")
print(f"Seasons in raw dataset: {df_vegas['season'].nunique()}.")

# %%
df = df_vegas[~df_vegas["season"].isin(EXCLUDED_SEASONS)].copy()
df = df.dropna(subset=["actual_wins"])

df["games_played"] = df["season"].map(GAMES_PER_SEASON)
unmapped = df[df["games_played"].isna()]
if not unmapped.empty:
    raise ValueError(f"Seasons without games_played mapping: {sorted(unmapped['season'].unique())}")

df["win_pct"] = df["actual_wins"] / df["games_played"]

print(f"After exclusions: {len(df)} team-season rows across {df['season'].nunique()} seasons.")
print()
print("Win-pct summary:")
print(df["win_pct"].describe().round(4))

# %% [markdown]
# Sanity check that win-pct lands in [0, 1] and the per-season mean is ~0.500.

# %%
print(f"win_pct min: {df['win_pct'].min():.4f}")
print(f"win_pct max: {df['win_pct'].max():.4f}")
print()

season_means = df.groupby("season")["win_pct"].mean()
print("Per-season win-pct means (should all be ~0.500 in balanced seasons):")
print(season_means.round(4).to_string())

# %% [markdown]
# ## Chart 1A: Pooled win-pct histogram

# %%
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.arange(0.0, 1.025, 0.025)
counts, edges, _ = ax.hist(
    df["win_pct"],
    bins=bins,
    edgecolor="white",
    linewidth=0.5,
    alpha=0.75,
    color="steelblue",
    label="observed",
)

mu = df["win_pct"].mean()
sigma = df["win_pct"].std()
x = np.linspace(0, 1, 400)
pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
pdf_scaled = pdf * len(df) * 0.025
ax.plot(
    x,
    pdf_scaled,
    color="firebrick",
    linewidth=2,
    label=f"normal fit (mu={mu:.3f}, sigma={sigma:.3f})",
)

ax.axvline(0.500, linestyle="--", color="black", alpha=0.6, label="mean = 0.500")

ax.set_xlim(0, 1)
ax.set_xlabel("Team season win-pct")
ax.set_ylabel("Count of team-seasons")
ax.set_title(f"Pooled win-pct distribution ({df['season'].nunique()} seasons, n={len(df)})")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "1a_pooled_histogram.png", dpi=150)
plt.show()

# %% [markdown]
# ## Chart 1B: Q-Q plot against normal

# %%
fig, ax = plt.subplots(figsize=(8, 8))

stats.probplot(df["win_pct"], dist="norm", plot=ax)

ax.set_title("Q-Q plot: pooled win-pct vs normal")
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Ordered values")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "1b_qq_plot.png", dpi=150)
plt.show()

# %% [markdown]
# ## Chart 1C: Per-season histograms, faceted small multiples

# %%
seasons = sorted(df["season"].unique())
n_seasons = len(seasons)
n_cols = 5
n_rows = int(np.ceil(n_seasons / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True, sharey=True)
axes = axes.flatten()

bins_small = np.arange(0.0, 1.05, 0.05)

for i, season in enumerate(seasons):
    ax = axes[i]
    sub = df[df["season"] == season]["win_pct"]
    ax.hist(sub, bins=bins_small, edgecolor="white", linewidth=0.5, color="steelblue", alpha=0.75)
    ax.axvline(0.500, linestyle="--", color="black", alpha=0.4)
    ax.set_title(season.replace("_", "-"), fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

for j in range(n_seasons, len(axes)):
    axes[j].axis("off")

fig.suptitle("Per-season win-pct distributions", y=1.00)
fig.tight_layout()
fig.savefig(FIG_DIR / "1c_per_season_facets.png", dpi=150)
plt.show()

# %% [markdown]
# ## Chart 1D: sigma_winpct across seasons

# %%
season_stats = (
    df.groupby("season")
    .agg(
        mean_winpct=("win_pct", "mean"),
        sigma_winpct=("win_pct", "std"),
        n_teams=("win_pct", "count"),
    )
    .reset_index()
    .sort_values("season")
)

season_stats["start_year"] = season_stats["season"].str.slice(0, 4).astype(int)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    season_stats["start_year"],
    season_stats["sigma_winpct"],
    marker="o",
    color="steelblue",
    linewidth=1.5,
)

pooled_sigma = df["win_pct"].std()
ax.axhline(
    pooled_sigma,
    linestyle="--",
    color="firebrick",
    alpha=0.6,
    label=f"pooled sigma = {pooled_sigma:.3f}",
)

ax.axvline(2013, linestyle=":", color="gray", alpha=0.6, label="era cut (2013)")
ax.axvline(2021, linestyle=":", color="gray", alpha=0.6, label="era cut (2021)")

upper = pooled_sigma * 1.15
lower = pooled_sigma * 0.85
ax.fill_between(
    [season_stats["start_year"].min(), season_stats["start_year"].max()],
    [lower, lower],
    [upper, upper],
    color="firebrick",
    alpha=0.1,
    label="+/- 15% of pooled",
)

ax.set_xlabel("Season start year")
ax.set_ylabel("sigma of within-season win-pct")
ax.set_title("Within-season sigma of win-pct across 29 seasons")
ax.legend(loc="best")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "1d_sigma_across_seasons.png", dpi=150)
plt.show()

# %% [markdown]
# ## Statistics: per-season and pooled


# %%
def summarize_group(group: pd.DataFrame, label: str) -> dict:
    wp = group["win_pct"]
    n = len(wp)
    out = {
        "label": label,
        "n": n,
        "mean": wp.mean(),
        "sigma": wp.std(),
        "skew": stats.skew(wp),
        "kurtosis": stats.kurtosis(wp),
        "p05": wp.quantile(0.05),
        "p25": wp.quantile(0.25),
        "p50": wp.quantile(0.50),
        "p75": wp.quantile(0.75),
        "p95": wp.quantile(0.95),
    }
    if n >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(wp)
        out["shapiro_stat"] = shapiro_stat
        out["shapiro_p"] = shapiro_p
    if n >= 8:
        ad_result = stats.anderson(wp, dist="norm")
        out["anderson_stat"] = ad_result.statistic
        out["anderson_crit_5pct"] = ad_result.critical_values[2]
    return out


per_season = pd.DataFrame([summarize_group(df[df["season"] == s], s) for s in seasons])
print("Per-season summary (rounded):")
print(per_season.round(4).to_string(index=False))

# %%
pooled_stats = summarize_group(df, "pooled")
print()
print("Pooled summary:")
for k, v in pooled_stats.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# %% [markdown]
# ## Era-split: pre-2012 vs post-2012

# %%
df["era_2way"] = df["season"].apply(
    lambda s: "pre_2012" if ERA_TWO_WAY["pre_2012"](s) else "post_2012"
)
df["era_3way"] = df["season"].apply(
    lambda s: next(name for name, fn in ERA_THREE_WAY.items() if fn(s))
)

era_two_summary = pd.DataFrame(
    [summarize_group(df[df["era_2way"] == era], era) for era in ["pre_2012", "post_2012"]]
)
print("Two-way era split:")
print(era_two_summary.round(4).to_string(index=False))

# %%
era_three_summary = pd.DataFrame(
    [
        summarize_group(df[df["era_3way"] == era], era)
        for era in ["pre_modernization", "analytics_era", "post_bubble"]
    ]
)
print()
print("Three-way era split:")
print(era_three_summary.round(4).to_string(index=False))

# %% [markdown]
# ## Sigma stability assessment

# %%
sigmas = season_stats["sigma_winpct"].values
pooled = pooled_stats["sigma"]
within_15pct = ((sigmas >= pooled * 0.85) & (sigmas <= pooled * 1.15)).mean()
print(f"Pooled sigma_winpct: {pooled:.4f}")
print(f"Season-level sigma range: [{sigmas.min():.4f}, {sigmas.max():.4f}]")
print(f"Range as fraction of pooled: [{sigmas.min()/pooled:.3f}, {sigmas.max()/pooled:.3f}]")
print(f"Fraction of seasons within +/- 15% of pooled: {within_15pct:.1%}")

era_two_sigmas = era_two_summary.set_index("label")["sigma"]
era_two_diff = abs(era_two_sigmas["pre_2012"] - era_two_sigmas["post_2012"])
era_two_pct = era_two_diff / pooled
print()
print(f"Two-way era sigma gap: {era_two_diff:.4f} ({era_two_pct:.1%} of pooled)")

era_three_sigmas = era_three_summary.set_index("label")["sigma"]
era_three_range = era_three_sigmas.max() - era_three_sigmas.min()
era_three_pct = era_three_range / pooled
print(f"Three-way era sigma range: {era_three_range:.4f} ({era_three_pct:.1%} of pooled)")

# %% [markdown]
# **Section 1 Results:**
#
# - Within-season team win-pcts are approximately normally distributed
#   across the 27-season window. Pooled mean = 0.4996 (essentially
#   0.500 as the zero-sum balance requires), pooled sigma = 0.1516
#   across 804 team-seasons.
# - Distribution shape: pooled skew = -0.21 (mild left lean), kurtosis
#   = -0.67 (mildly platykurtic, slightly flatter peak than normal).
#   Visually bell-shaped in pooled histogram (Chart 1A) and per-season
#   facets (Chart 1C). Q-Q plot (Chart 1B) tracks linear from -2 to +2
#   quantiles, with mild deviation only in the extreme tails (the
#   bounded-distribution artifact since win-pct is capped at 0.000 and
#   1.000).
# - Sigma stability is excellent. Per-season sigma range is [0.122,
#   0.189], with 92.6% of seasons falling within +/- 15% of pooled
#   sigma. The 2022-23 season is the tightest at 0.122; the 1997-98
#   season is the widest at 0.189. Both are within reasonable bounds.
# - Era splits confirm pooled sigma is appropriate. Two-way era gap
#   (pre-2012 0.1533 vs post-2012 0.1500): 0.0033 absolute, 2.2% of
#   pooled. Three-way era range (pre-modernization 0.1532, analytics
#   era 0.1517, post-bubble 0.1465): 0.0067 absolute, 4.4% of pooled.
#   Both well below the +/- 15% threshold that would require per-era
#   treatment.
# - Normality tests reject at p<0.05 for the pooled distribution and
#   most individual seasons. With pooled n=804, Shapiro-Wilk and
#   Anderson-Darling detect any real-world deviation from platonic
#   normal; this is expected and does NOT indicate the framework is
#   broken. The Q-Q plot's visual fit from -2 to +2 quantiles is the
#   methodologically relevant evidence, and that fit is clean.
# - **Decision locked:** Use pooled sigma_winpct = 0.1516 for z-scoring
#   team strength across all 27 seasons. No per-season or per-era
#   treatment needed.
# - The Phi-based win-probability mapping (downstream of this z-score
#   construction) is empirically supported. Section 7 will test whether
#   Phi fits empirical win rates better than logistic.

# %% [markdown]
# ## Section 2: Per-game four-factor differential distributions
#
# Section 1 established that within-season team win-pct is approximately
# normal with stable pooled sigma. The model also z-scores the four
# factor differentials and feeds them as features. The same
# distributional assumption applies, and needs the same validation.
#
# This section loads the box-score data (separate from the Vegas
# team-season data used in Section 1) and tests approximate normality
# of the per-game team-minus-opponent differential for each factor.
#
# Sign convention: positive means the team played better at that factor.
#
# - eFG% differential:     own off_efg_pct - own def_efg_pct
# - TOV% differential:     own def_tov_pct - own off_tov_pct
# - ORB% differential:     own off_orb_pct - own def_orb_pct
# - FT rate differential:  own off_ft_rate - own def_ft_rate
#
# (TOV is sign-flipped: lower TOV is better, so opponent's TOV minus
# own TOV makes positive = good. The other three are already positive-
# is-good in their natural off-minus-def form.)

# %%

df_box = load_processed(("1997_98", "2025_26"), SeasonType.REGULAR)
df_box = df_box[~df_box["season"].isin(EXCLUDED_SEASONS)].copy()

print(f"Box-score long-format shape after exclusions: {df_box.shape}")
print(f"Seasons covered: {df_box['season'].nunique()}")

# %%
df_ff = df_box.copy()

df_ff["efg_diff"] = df_ff["off_efg_pct"] - df_ff["def_efg_pct"]
df_ff["tov_diff"] = df_ff["def_tov_pct"] - df_ff["off_tov_pct"]
df_ff["orb_diff"] = df_ff["off_orb_pct"] - df_ff["def_orb_pct"]
df_ff["ft_diff"] = df_ff["off_ft_rate"] - df_ff["def_ft_rate"]

FACTOR_DIFFS = ["efg_diff", "tov_diff", "orb_diff", "ft_diff"]
FACTOR_LABELS = {
    "efg_diff": "eFG% differential",
    "tov_diff": "TOV% differential (opp - own)",
    "orb_diff": "ORB% differential",
    "ft_diff": "FT rate differential",
}

print(f"Per-game team-game rows for four-factor diagnostics: {len(df_ff):,}")

# %%
print("Per-game four-factor differential summary statistics (pooled):")
print("-" * 70)
ff_stats = []
for col in FACTOR_DIFFS:
    series = df_ff[col].dropna()
    stat_row = {
        "factor": FACTOR_LABELS[col],
        "n": len(series),
        "mean": series.mean(),
        "sigma": series.std(),
        "skew": stats.skew(series),
        "kurtosis": stats.kurtosis(series),
        "p05": series.quantile(0.05),
        "p95": series.quantile(0.95),
    }
    ff_stats.append(stat_row)
    print(f"  {FACTOR_LABELS[col]}")
    print(f"    n={stat_row['n']:,}, mean={stat_row['mean']:+.4f}, sigma={stat_row['sigma']:.4f}")
    print(f"    skew={stat_row['skew']:+.3f}, kurtosis={stat_row['kurtosis']:+.3f}")
    print(f"    [5th, 95th] = [{stat_row['p05']:+.4f}, {stat_row['p95']:+.4f}]")
print("-" * 70)

ff_stats_df = pd.DataFrame(ff_stats)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, FACTOR_DIFFS, strict=False):
    series = df_ff[col].dropna()
    ax.hist(
        series,
        bins=60,
        density=True,
        alpha=0.75,
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
    )
    x_range = np.linspace(series.min(), series.max(), 200)
    fitted_normal = stats.norm.pdf(x_range, series.mean(), series.std())
    ax.plot(x_range, fitted_normal, color="firebrick", linewidth=1.5, label="Fitted normal")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_title(FACTOR_LABELS[col])
    ax.set_xlabel("Differential")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=9)
fig.suptitle("Per-game four-factor differentials: pooled distributions", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "2a_pergame_factor_histograms.png", dpi=150)
plt.show()

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, FACTOR_DIFFS, strict=False):
    series = df_ff[col].dropna()
    stats.probplot(series, dist="norm", plot=ax)
    ax.set_title(FACTOR_LABELS[col])
    ax.get_lines()[0].set_markersize(2)
    ax.get_lines()[0].set_markerfacecolor("steelblue")
    ax.get_lines()[0].set_markeredgecolor("steelblue")
fig.suptitle("Per-game four-factor differentials: Q-Q plots vs normal", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "2b_pergame_factor_qqplots.png", dpi=150)
plt.show()

# %%
print("Shapiro-Wilk normality tests (caveated: large-n rejects any minor deviation)")
print("-" * 70)
for col in FACTOR_DIFFS:
    series = df_ff[col].dropna()
    sample_size = min(len(series), 5000)
    sample = series.sample(n=sample_size, random_state=42) if len(series) > sample_size else series
    stat_val, p_val = stats.shapiro(sample)
    print(f"  {FACTOR_LABELS[col]}: stat={stat_val:.4f}, p={p_val:.4f}")
    print(
        f"    (n={sample_size:,} subsample; p<0.05 rejects normality but skew/kurtosis are the better guide)"
    )
print("-" * 70)

# %% [markdown]
# **Section 2 Results:**
#
# - All four factor differentials have mean and skew of exactly 0.000
#   by construction (within each game, the differential is symmetric
#   across the two teams; pooling team-game rows zeros out both
#   first moments). The diagnostic of interest is kurtosis.
# - Three of four factors have kurtosis within +/- 0.05 of normal:
#   eFG% (+0.032), TOV% (+0.048), ORB% (-0.023). These are
#   essentially textbook normal at the per-game level.
# - FT rate differential is mildly heavy-tailed: kurtosis +0.506.
#   The Shapiro-Wilk test on a 5,000-row subsample rejects normality
#   only for FT rate (p<0.0001); the other three factors pass
#   (p between 0.10 and 0.59). Reading: FT rate has slightly fatter
#   tails than the other three factors, consistent with foul-rate
#   variance being higher than possession-outcome variance.
# - Pooled sigma values (per-game scale): eFG% 0.091, TOV% 0.047,
#   ORB% 0.107, FT rate 0.136. These are the raw within-game
#   differential standard deviations, the input to z-scoring.
# - For the model: z-scoring the per-game differentials is justified
#   for all four factors. FT rate's slight heavy-tailedness is mild
#   enough that direct z-scoring is still defensible; the model's
#   regularization handles the modest extra tail mass.

# %% [markdown]
# ## Section 3: Rolling-mean four-factor differentials
#
# The model uses rolling-window features, not raw per-game values. The
# central limit theorem says the mean of N draws from any distribution
# approaches normal as N grows. So the rolling means the model actually
# uses should look notably more normal than the raw per-game
# differentials in Section 2.
#
# Canonical window: 15 games. This is the default the model will use,
# pending the shifting-window response curve experiment.
#
# Sub-section 3b sweeps the window from 5 to 40 games and tracks how
# the distributional shape changes, providing empirical evidence for
# how window choice affects rolling-mean normality.

# %%
CANONICAL_WINDOW = 15

df_ff_sorted = df_ff.sort_values(["team_abbr", "season", "game_date"]).reset_index(drop=True)

rolling_groups = df_ff_sorted.groupby(["team_abbr", "season"], group_keys=False)
for col in FACTOR_DIFFS:
    rolled_col = f"{col}_roll{CANONICAL_WINDOW}"
    df_ff_sorted[rolled_col] = rolling_groups[col].transform(
        lambda s: s.rolling(window=CANONICAL_WINDOW, min_periods=CANONICAL_WINDOW).mean()
    )

roll_cols = [f"{col}_roll{CANONICAL_WINDOW}" for col in FACTOR_DIFFS]
df_ff_roll = df_ff_sorted.dropna(subset=roll_cols).copy()
print(
    f"Rolling-mean rows (window={CANONICAL_WINDOW} games, after dropping incomplete windows): {len(df_ff_roll):,}"
)

# %%
print(f"Rolling {CANONICAL_WINDOW}-game four-factor differential summary statistics (pooled):")
print("-" * 70)
for col in FACTOR_DIFFS:
    roll_col = f"{col}_roll{CANONICAL_WINDOW}"
    series = df_ff_roll[roll_col].dropna()
    raw_stats_row = ff_stats_df[ff_stats_df["factor"] == FACTOR_LABELS[col]].iloc[0]
    roll_sigma = series.std()
    roll_skew = stats.skew(series)
    roll_kurt = stats.kurtosis(series)
    print(f"  {FACTOR_LABELS[col]}")
    print(
        f"    rolling sigma={roll_sigma:.4f} (vs raw {raw_stats_row['sigma']:.4f}, ratio {roll_sigma/raw_stats_row['sigma']:.3f})"
    )
    print(f"    rolling skew={roll_skew:+.3f} (vs raw {raw_stats_row['skew']:+.3f})")
    print(f"    rolling kurtosis={roll_kurt:+.3f} (vs raw {raw_stats_row['kurtosis']:+.3f})")
print("-" * 70)
print("Expected: rolling sigma smaller than raw (lower variance of means);")
print("rolling skew and kurtosis closer to zero (CLT smoothing).")

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, FACTOR_DIFFS, strict=False):
    roll_col = f"{col}_roll{CANONICAL_WINDOW}"
    series = df_ff_roll[roll_col].dropna()
    ax.hist(
        series,
        bins=60,
        density=True,
        alpha=0.75,
        color="seagreen",
        edgecolor="white",
        linewidth=0.4,
    )
    x_range = np.linspace(series.min(), series.max(), 200)
    fitted_normal = stats.norm.pdf(x_range, series.mean(), series.std())
    ax.plot(x_range, fitted_normal, color="firebrick", linewidth=1.5, label="Fitted normal")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_title(FACTOR_LABELS[col])
    ax.set_xlabel(f"Rolling {CANONICAL_WINDOW}-game mean differential")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=9)
fig.suptitle(
    f"Rolling {CANONICAL_WINDOW}-game mean four-factor differentials: pooled distributions",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "3a_rolling15_factor_histograms.png", dpi=150)
plt.show()

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, FACTOR_DIFFS, strict=False):
    roll_col = f"{col}_roll{CANONICAL_WINDOW}"
    series = df_ff_roll[roll_col].dropna()
    stats.probplot(series, dist="norm", plot=ax)
    ax.set_title(FACTOR_LABELS[col])
    ax.get_lines()[0].set_markersize(2)
    ax.get_lines()[0].set_markerfacecolor("seagreen")
    ax.get_lines()[0].set_markeredgecolor("seagreen")
fig.suptitle(
    f"Rolling {CANONICAL_WINDOW}-game mean differentials: Q-Q plots vs normal", fontsize=12
)
plt.tight_layout()
plt.savefig(FIG_DIR / "3b_rolling15_factor_qqplots.png", dpi=150)
plt.show()

# %% [markdown]
# ### 3b: Window-size sweep
#
# How do the distributional diagnostics change as window size varies?
# Sweep window in {5, 10, 15, 20, 30, 40}. For each window, compute the
# rolling-mean differential for each factor and report sigma, skew, and
# kurtosis.
#
# Expected pattern (CLT predictions):
# - sigma decreases monotonically with window size.
# - skew and kurtosis move toward zero as window grows.

# %%
WINDOW_SWEEP = [5, 10, 15, 20, 30, 40]

sweep_rows = []
for window in WINDOW_SWEEP:
    sweep_groups = df_ff_sorted.groupby(["team_abbr", "season"], group_keys=False)
    for col in FACTOR_DIFFS:
        rolled = sweep_groups[col].transform(
            lambda s, w=window: s.rolling(window=w, min_periods=w).mean()
        )
        series = rolled.dropna()
        sweep_rows.append(
            {
                "window": window,
                "factor": FACTOR_LABELS[col],
                "n": len(series),
                "sigma": series.std(),
                "skew": stats.skew(series),
                "kurtosis": stats.kurtosis(series),
            }
        )

sweep_df = pd.DataFrame(sweep_rows)
print("Window-size sweep: how distributional shape changes with window size")
print("-" * 70)
for factor in [FACTOR_LABELS[c] for c in FACTOR_DIFFS]:
    factor_sweep = sweep_df[sweep_df["factor"] == factor].copy()
    print(f"\n  {factor}")
    print(f"    {'window':>8} {'n':>8} {'sigma':>8} {'skew':>8} {'kurtosis':>10}")
    for _, row in factor_sweep.iterrows():
        print(
            f"    {int(row['window']):>8} {int(row['n']):>8,} {row['sigma']:>8.4f} {row['skew']:>+8.3f} {row['kurtosis']:>+10.3f}"
        )
print("-" * 70)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sweep_colors = ["steelblue", "darkorange", "seagreen", "firebrick"]
for col_idx, (metric, ylabel) in enumerate(
    [("sigma", "Sigma"), ("skew", "Skew"), ("kurtosis", "Kurtosis")]
):
    ax = axes[col_idx]
    for factor_idx, col in enumerate(FACTOR_DIFFS):
        factor = FACTOR_LABELS[col]
        factor_sweep = sweep_df[sweep_df["factor"] == factor].copy()
        ax.plot(
            factor_sweep["window"],
            factor_sweep[metric],
            marker="o",
            markersize=6,
            linewidth=1.5,
            color=sweep_colors[factor_idx],
            label=factor,
        )
    ax.set_xlabel("Window size (games)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs window size")
    if metric != "sigma":
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
    if col_idx == 2:
        ax.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "3c_window_sweep_diagnostics.png", dpi=150)
plt.show()

# %% [markdown]
# **Section 3 Results:**
#
# - Rolling 15-game means shrink sigma by a factor of ~0.378 across
#   all four factors. Theoretical CLT prediction for independent
#   draws is 1/sqrt(15) = 0.258. The observed 0.378 is larger,
#   indicating consecutive team-games are NOT independent. Teams
#   have persistent strength within a season, so rolling means
#   smooth less than independent-draws math would predict.
# - The effective independent sample size for a 15-game window is
#   approximately (0.091/0.034)^2 = ~7 games. That is the
#   team-quality-persistence signal: a 15-game window contains the
#   information of about 7 truly independent observations.
# - Rolling kurtosis stays close to zero for eFG (-0.10), TOV
#   (+0.06), and ORB (+0.12). The CLT smoothing works as expected
#   for these three factors.
# - FT rate kurtosis stays elevated (+0.35) even at the rolling-mean
#   level, declining only modestly from +0.51 raw. This means the
#   fat tails aren't just noise; some teams persistently differ
#   from league average in foul-drawing rate, producing rolling
#   means that have heavier tails than the other factors. Worth
#   noting in methodology as a property of FT rate as a feature.
# - Window-size sweep confirms the CLT pattern: sigma decreases
#   monotonically from W=5 (0.047 for eFG) to W=40 (0.029). Larger
#   windows mean tighter feature distributions but lose responsiveness
#   to recent performance changes. The shifting-window model
#   experiment will navigate this tradeoff empirically.
# - Rolling skew shifts away from zero for ORB (becomes increasingly
#   negative with window size) and eFG (becomes increasingly positive).
#   This is the autocorrelation-driven version of "teams that win
#   stay winning, teams that lose stay losing" showing up in the
#   feature distributions at longer averaging windows. Mild; not
#   methodology-breaking.

# %% [markdown]
# ## Section 4: Per-season sigma stability for each factor differential
#
# Section 1 established pooled sigma works for win-pct (92.6% of
# seasons within +/- 15% of pooled). The same question for each of the
# four factor differentials: is the per-season sigma stable enough
# across seasons to justify pooled sigma, or does at least one factor
# need per-era treatment?
#
# Two-way era split (pre-2012 / post-2012) and three-way era split
# (pre-modernization 1997-2013 / analytics-era 2013-2020 / post-bubble
# 2021-2026) are both tested. The decision about pooled vs per-era is
# left to the reader of the printed output, matching Section 1's style.

# %%
ff_seasons = sorted(df_ff["season"].unique())

print("Per-season sigma stability for each factor differential")
print("-" * 70)
sigma_stability = {}
for col in FACTOR_DIFFS:
    per_season_sigma = df_ff.groupby("season")[col].std()
    per_season_sigma = per_season_sigma.reindex(ff_seasons).dropna()
    pooled_sigma_col = df_ff[col].std()
    within_15pct = (
        (per_season_sigma >= 0.85 * pooled_sigma_col)
        & (per_season_sigma <= 1.15 * pooled_sigma_col)
    ).sum()
    sigma_stability[col] = {
        "pooled_sigma": pooled_sigma_col,
        "per_season_min": per_season_sigma.min(),
        "per_season_max": per_season_sigma.max(),
        "min_ratio": per_season_sigma.min() / pooled_sigma_col,
        "max_ratio": per_season_sigma.max() / pooled_sigma_col,
        "n_seasons": len(per_season_sigma),
        "n_within_15pct": within_15pct,
        "pct_within_15pct": within_15pct / len(per_season_sigma) * 100,
        "per_season": per_season_sigma,
    }
    sd = sigma_stability[col]
    print(f"\n  {FACTOR_LABELS[col]}")
    print(f"    Pooled sigma: {sd['pooled_sigma']:.4f}")
    print(f"    Per-season sigma range: [{sd['per_season_min']:.4f}, {sd['per_season_max']:.4f}]")
    print(f"    Range as fraction of pooled: [{sd['min_ratio']:.3f}, {sd['max_ratio']:.3f}]")
    print(
        f"    Seasons within +/- 15% of pooled: {sd['n_within_15pct']} / {sd['n_seasons']} ({sd['pct_within_15pct']:.1f}%)"
    )
print("-" * 70)

# %%
PRE_2012_SEASONS = [s for s in ff_seasons if s <= "2011_12"]
POST_2012_SEASONS = [s for s in ff_seasons if s >= "2012_13"]

PRE_MOD_SEASONS = [s for s in ff_seasons if s <= "2012_13"]
ANALYTICS_SEASONS = [s for s in ff_seasons if "2013_14" <= s <= "2019_20"]
POST_BUBBLE_SEASONS = [s for s in ff_seasons if s >= "2021_22"]

print("Two-way era sigma comparison:")
print("-" * 70)
print(f"  {'factor':<32} {'pre-2012':>10} {'post-2012':>10} {'gap':>8} {'gap_pct':>10}")
for col in FACTOR_DIFFS:
    pre_sigma = df_ff[df_ff["season"].isin(PRE_2012_SEASONS)][col].std()
    post_sigma = df_ff[df_ff["season"].isin(POST_2012_SEASONS)][col].std()
    gap = post_sigma - pre_sigma
    pooled_sigma_col = sigma_stability[col]["pooled_sigma"]
    gap_pct = abs(gap) / pooled_sigma_col * 100
    print(
        f"  {FACTOR_LABELS[col]:<32} {pre_sigma:>10.4f} {post_sigma:>10.4f} {gap:>+8.4f} {gap_pct:>9.2f}%"
    )
print("-" * 70)

print("\nThree-way era sigma comparison:")
print("-" * 70)
print(f"  {'factor':<32} {'pre-mod':>10} {'analytics':>10} {'post-bubble':>12}")
for col in FACTOR_DIFFS:
    pre_sigma = df_ff[df_ff["season"].isin(PRE_MOD_SEASONS)][col].std()
    ana_sigma = df_ff[df_ff["season"].isin(ANALYTICS_SEASONS)][col].std()
    pb_sigma = df_ff[df_ff["season"].isin(POST_BUBBLE_SEASONS)][col].std()
    print(f"  {FACTOR_LABELS[col]:<32} {pre_sigma:>10.4f} {ana_sigma:>10.4f} {pb_sigma:>12.4f}")
print("-" * 70)

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
season_idx = np.arange(len(ff_seasons))
season_labels = [s.replace("_", "-") for s in ff_seasons]
for ax, col in zip(axes.flat, FACTOR_DIFFS, strict=False):
    sd = sigma_stability[col]
    per_season = sd["per_season"].reindex(ff_seasons)
    ax.plot(
        season_idx, per_season.values, marker="o", markersize=5, linewidth=1.2, color="steelblue"
    )
    ax.axhline(
        sd["pooled_sigma"],
        color="firebrick",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Pooled sigma = {sd['pooled_sigma']:.4f}",
    )
    ax.axhline(
        sd["pooled_sigma"] * 1.15,
        color="firebrick",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
        label="+/- 15% bounds",
    )
    ax.axhline(
        sd["pooled_sigma"] * 0.85, color="firebrick", linestyle=":", linewidth=0.8, alpha=0.5
    )
    ax.set_xticks(season_idx[::3])
    ax.set_xticklabels(
        [season_labels[i] for i in season_idx[::3]], rotation=60, ha="right", fontsize=8
    )
    ax.set_ylabel("Per-season sigma")
    ax.set_title(FACTOR_LABELS[col])
    ax.legend(loc="best", fontsize=8)
fig.suptitle("Per-season sigma stability for four-factor differentials", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "4a_per_season_sigma_stability.png", dpi=150)
plt.show()

# %% [markdown]
# **Section 4 Results:**
#
# - Three of four factor differentials show clean pooled-sigma
#   stability: eFG% (27/27 seasons within +/- 15% of pooled,
#   range [0.96, 1.06]), TOV% (27/27 within +/- 15%, range
#   [0.95, 1.09]), ORB% (27/27 within +/- 15%, range [0.92, 1.06]).
#   For these three factors, pooled sigma is the right methodology
#   choice with no era stratification needed.
# - FT rate differential is the exception: only 21 of 27 seasons
#   within +/- 15% of pooled, with a per-season range of [0.83, 1.20].
#   The two-way era gap (pre-2012 0.147 vs post-2012 0.123) is
#   17.47% of pooled, well above the 15% threshold that the other
#   three factors clear easily.
# - The three-way era split for FT rate shows monotone compression:
#   pre-modernization 0.146 to analytics era 0.128 to post-bubble
#   0.114. FT rate variance has been shrinking across the league
#   for two decades. This is consistent with EDA 04 §6 (FT rate
#   declined 21% over 29 seasons) and EDA 04 §11 (uniform
#   coefficient shrinkage in the four-factor regression).
# - Methodology decision: pooled sigma for eFG, TOV, and ORB
#   differentials; per-era (pre-2012 / post-2012) sigma for FT rate
#   differential. The era cut aligns with the 2012-13 inflection
#   point established in EDA 04 (3PA acceleration) and EDA 05 (HCA
#   modernization decline). Using the same era cut for FT rate
#   z-scoring keeps the methodology internally consistent.
# - For the model: when computing FT rate features, z-score against
#   the pre-2012 sigma for games in pre-2012 seasons and the
#   post-2012 sigma for games in post-2012 seasons. The other three
#   factors z-score against pooled sigma uniformly.

# %% [markdown]
# ## Section 5: Cross-factor independence at the team-game level
#
# The model treats the four factor differentials as separate features.
# This is justified if the differentials are approximately independent
# at the team-game level. If they are strongly correlated, the model
# may benefit from interaction terms or principal components rather
# than direct features.
#
# EDA 03 found cross-block coupling (one team's offensive eFG
# correlates with the opposing team's offensive eFG in the same game).
# This is a different test: within a single team-game, do this team's
# four differentials correlate with each other?
#
# Reference points for interpreting correlation magnitudes:
# - |r| < 0.20: weak; factors approximately independent.
# - 0.20 <= |r| < 0.50: moderate; direct features still defensible
#   but worth noting in methodology.
# - |r| >= 0.50: strong; consider interaction terms or PCA.

# %%
ff_corr_matrix = df_ff[FACTOR_DIFFS].corr()
print("Cross-factor differential correlation matrix (pooled):")
print("-" * 70)
labeled_corr = ff_corr_matrix.rename(index=FACTOR_LABELS, columns=FACTOR_LABELS)
print(labeled_corr.round(4))
print("-" * 70)

max_off_diag = ff_corr_matrix.where(~np.eye(len(FACTOR_DIFFS), dtype=bool)).abs().max().max()
print(f"\nMaximum off-diagonal |r|: {max_off_diag:.4f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
era_labels_5 = [
    "Pre-modernization (1997-2013)",
    "Analytics era (2013-2020)",
    "Post-bubble (2021-2026)",
]
era_lists = [PRE_MOD_SEASONS, ANALYTICS_SEASONS, POST_BUBBLE_SEASONS]
for ax, era_name, era_list in zip(axes, era_labels_5, era_lists, strict=False):
    era_df = df_ff[df_ff["season"].isin(era_list)]
    era_corr = era_df[FACTOR_DIFFS].corr()
    short_labels = ["eFG", "TOV", "ORB", "FT"]
    sns.heatmap(
        era_corr,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"{era_name}\nn={len(era_df):,} team-games")
fig.suptitle("Cross-factor differential correlations by era", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "5a_cross_factor_corr_by_era.png", dpi=150)
plt.show()

# %% [markdown]
# **Section 5 Results:**
#
# - Maximum pairwise correlation between any two factor differentials
#   is |r| = 0.197 (eFG vs TOV). All six pairs are in the "weak
#   coupling" band (|r| < 0.20). Three pairs are essentially
#   independent (|r| < 0.10): eFG-FT (+0.029), ORB-TOV (-0.052),
#   ORB-FT (-0.027).
# - The eFG-TOV mild negative coupling (-0.197) reflects the
#   within-game possession structure: a possession ends in either
#   a shot (eFG counts) or a turnover (TOV counts), so teams
#   doing well at one tend to be very slightly worse at the other.
#   This is the differential-scale echo of EDA 03's within-game
#   "two pairs" finding (eFG and TOV as alternative possession
#   outcomes; ORB and FT as additive secondary opportunities).
# - The four factor differentials can be treated as approximately
#   independent features. No interaction terms or principal
#   components needed in the baseline model specification. The
#   regression model fits four direct features; the RNN's attention
#   mechanism receives them as separate input dimensions.
# - Era-stratified correlation matrices (chart 5a) show the
#   pattern is stable across pre-modernization, analytics era,
#   and post-bubble. The weak coupling is not a modernization-era
#   artifact and persists in all three eras with similar magnitudes.
# - For the model: standard z-scoring of each factor independently
#   is justified. Methodology section can state "factor differentials
#   are approximately independent at the team-game level
#   (max pairwise |r| = 0.197) and are treated as separate model
#   features."
