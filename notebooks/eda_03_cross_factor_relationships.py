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
# **Three-bullet summary:**
#
# 1. **Cross-block coupling exists and is dominated by 3PA rate.**
#    Within-team-game off-vs-def factor couplings are nonzero for all
#    four diagonals (FT +0.175, eFG +0.137, TOV +0.124, ORB +0.092).
#    Joint pace + 3PA controls explain 72% of the eFG coupling and 65%
#    of ORB, but only 14% of TOV and 20% of FT. The 3-point revolution
#    is the dominant within-game coupling mechanism; pace is largely
#    redundant given 3PA (except for ORB, where it adds 13pp). TOV and
#    FT couplings remain mostly unexplained and need auxiliary
#    mechanisms (officiating, behavioral, score-state).
# 2. **The four factors decompose into two pairs.** eFG and TOV show
#    polar cross-block coupling (alternative possession outcomes,
#    mutually exclusive at the possession level); ORB and FT show
#    uniform coupling (additive secondary opportunities from rim
#    aggression). Suggests a two-axis reframing of the four-factor
#    framework: primary efficiency (eFG, TOV) and secondary
#    opportunity (ORB, FT).
# 3. **Factor relationships are not era-stable, except where physics
#    holds them fixed.** eFG vs FT rate decoupled slowly and
#    monotonically across 29 seasons (+0.08 in 1997-98 to ~0.00 by
#    2025-26) as 3-point volume replaced inside-out scoring. But the
#    within-season eFG vs ORB coupling stayed flat and noisy at
#    ~-0.075 with no trend: the mechanical floor (a make produces no
#    offensive rebound) is physics and does not drift. The pooled eFG
#    vs ORB correlation of -0.151 is roughly half genuine within-game
#    coupling and half era-trend aggregation artifact.

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

seasons_ordered = sorted(df["season"].unique())
seasons_display = [s.replace("_", "-") for s in seasons_ordered]
x = np.arange(len(seasons_ordered))

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
#
# **Results:**
#
# - eFG% vs ORB% at -0.151 is the largest entry. Expected sign,
#   modest magnitude. Mechanical floor plus 3-point-shooting style
#   coupling, both pushing negative.
# - TOV% vs FT rate at +0.142 is the surprise. Same magnitude as eFG
#   vs ORB, opposite sign. Both factors tag rim-aggression offenses
#   (drives produce both fouls drawn and turnovers from contested
#   possessions). Not predicted in advance.
# - Other four pairs sit in [-0.07, +0.07], noise-tier to weak.
# - Four-factor independence assumption survives, barely. Two
#   entanglements worth flagging in methodology: eFG vs ORB and
#   TOV vs FT rate.

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
#
# **Results:**
#
# - Mirror passes within float64 epsilon (max abs diff 1.08e-14).
# - Confirms the box-score-derived four-factor data is fully symmetric
#   at the joint-distribution level (a stronger check than EDA 02 §3's
#   marginal mirror).
# - Substantive defensive content lives in the off-vs-def cross-block
#   (§3), not in def-vs-def correlations. Defensive correlations as a
#   standalone object are a redundant view of offensive correlations.

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
#
# **Results:**
#
# - All four cross-block diagonals positive: FT rate (+0.175),
#   eFG (+0.137), TOV (+0.124), ORB (+0.092). Some per-game scalar
#   couples both teams' rates symmetrically.
# - off_efg vs def_tov at -0.146 is the largest off-diagonal cross-pair,
#   comparable in magnitude to the eFG diagonal coupling. Unexpected
#   sign (this team shoots well = opponent turns it over less); not
#   predicted in advance.
# - Cross-block structure is non-uniform: different factor-pairs show
#   different coupling patterns. A single per-game scalar wouldn't
#   produce this; the coupling mechanism is multi-dimensional.
# - Mechanism unidentified by §3 alone. The pace test in §3c is one
#   piece of mechanism discrimination; full discrimination would
#   need data outside this pipeline (ref crew, play-by-play). See
#   `docs/blog_ideas.md` entry #1.

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
# is this team's defense factor Y) within the same game row. The tighter
# color scale (-0.2 to +0.2) makes the structure of the off-vs-def
# coupling visible in a way the wider 8x8 scale doesn't.
#
# What to look for: are the diagonal entries the largest (one per-game
# scalar story), or is there off-diagonal structure (multi-dimensional
# coupling)? Are there 2x2 sub-block patterns that suggest the four
# factors group naturally?
#
# **Results:**
#
# - Cross-block decomposes into two qualitatively different 2x2
#   sub-blocks. Not predicted in advance; surfaced visually from the
#   heatmap.
# - Top-left 2x2 (eFG, TOV vs def_eFG, def_TOV): polar coupling.
#   Same-factor entries positive (diagonal), different-factor entries
#   negative (anti-diagonal). Consistent with eFG and TOV being
#   alternative possession outcomes that are mutually exclusive at the
#   possession level.
# - Bottom-right 2x2 (ORB, FT vs def_ORB, def_FT): uniform coupling.
#   All four entries positive. Consistent with ORB and FT being
#   additive secondary opportunities (rim aggression generates both)
#   that aren't zero-sum within a game.
# - Suggests the four factors decompose into two natural pairs:
#   primary efficiency outcomes (eFG, TOV) and secondary opportunities
#   (ORB, FT). Could justify a two-axis reframing of the four-factor
#   framework. See `docs/blog_ideas.md` entry #2 for the full sketch.

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
# ## Section 3c: Pace coupling test
#
# §3 found that all four cross-block diagonals are positive, implying
# some per-game scalar lifts both teams' rate stats together. Pace is
# the most commonly cited candidate: possessions per game is a per-game
# scalar by construction, since both teams play in the same paced game
# and share possessions within 1 or 2.
#
# This section runs the pace test from `docs/blog_ideas.md` entry #1:
# compute pace per team-game, correlate it with each of the 8 factor
# columns, then partial out pace from the cross-block diagonals and see
# if they shrink. Big shrinkage means pace was doing most of the work.
# Small shrinkage means pace is one of multiple coupling mechanisms and
# the other candidates (refereeing, behavior, score state, opponent
# quality, style-matchup) explain most of the structure.
#
# Pace formula: `possessions = FGA + 0.44 * FTA + TOV - OREB`.
# Per-team estimate. Within a single NBA game both teams' possession
# counts agree within 1 or 2 (possessions alternate), so the per-team-row
# pace is a clean proxy for the game-level scalar.
#
# Note: the four factors are already pace-neutral rates by construction
# (eFG% is per shot, TOV% is per possession, ORB% is per opportunity,
# FT rate is per shot). Any nonzero correlation between pace and a factor
# is empirical style coupling, not a mechanical normalization artifact.
#
# **Results (synthesis across 3c.1-3c.5):**
#
# - Pace alone is a weak explainer of cross-block coupling (largest
#   single shrinkage 37% for ORB; near zero for TOV and FT).
# - 3PA rate is a strong explainer for eFG (72%) and ORB (52%),
#   modest for FT (18%), small but real for TOV (11%).
# - Joint pace + 3PA controls explain 72% of eFG, 65% of ORB, 14% of
#   TOV, and 20% of FT cross-block coupling.
# - Pace is redundant given 3PA for eFG, TOV, and FT (joint shrinkage
#   ~= 3PA-alone shrinkage). For ORB, pace adds 13 percentage points
#   of independent explanatory power on top of 3PA.
# - The 3-point revolution is the dominant within-game coupling
#   mechanism for the four-factor framework. This is a substantive
#   addition to the methodology section of the thesis.
# - TOV and FT cross-block couplings remain 80%+ unexplained after
#   both pace and 3PA controls. These require auxiliary mechanism
#   identification (officiating, behavioral, score-state) that the
#   four-factor framework doesn't provide.
# - For full mechanism-discrimination roadmap and testable
#   hypotheses, see `docs/blog_ideas.md` entries #1 (mechanism
#   discrimination) and #3 (3PA as within-game coupler).

# %%
required_pace_cols = ["fga", "fta", "tov", "oreb"]
missing_cols = [c for c in required_pace_cols if c not in df.columns]
if missing_cols:
    print(f"Required columns not found: {missing_cols}")
    print(f"Available columns: {sorted(df.columns.tolist())}")
    raise KeyError(f"Cannot compute pace, missing: {missing_cols}")

df["poss"] = df["fga"] + 0.44 * df["fta"] + df["tov"] - df["oreb"]
print(f"Pace summary across {len(df):,} team-games:")
print(df["poss"].describe().round(2))

# %% [markdown]
# ### 3c.1: Per-season pace trend
#
# Sanity check that pace evolved across the 29 seasons as expected.
# Full pace-trend analysis lives in EDA 04 (time trends); this is the
# minimum needed to confirm pace is meaningful here.
#
# **Results:**
#
# - Three-era structure: dead-ball era start (1997-2003, ~93-96),
#   grinding era (2003-2012, flat at ~93-95), pace-and-space adoption
#   (2013-2017, rise to ~97-100), modern fast era (2018-2026, stable
#   at ~101-103).
# - Biggest single-season jump: 2017-18 (100.15) to 2018-19 (103.10),
#   a +3 possession step. Coincides with the 14-second shot clock
#   reset after offensive rebounds (introduced 2018-19) and intensified
#   freedom-of-movement enforcement. Rule changes visible in the data.
# - Highest-pace season is 2019-20 (103.45), the COVID-disrupted year.
#   Possibly Bubble-related (concentrated games, short rotations) or
#   schedule-composition artifact. Footnote candidate, not investigated
#   here.
# - Pace is meaningful across the 29-season window: 10-point spread
#   between slowest and fastest seasons. Pace-control analyses in
#   3c.2-3c.4 are not testing a static variable.
# - Sanity check on max-pace outliers: top-5 rows (poss > 136) all
#   appear as same-date team pairs and are confirmed multi-OT games
#   (e.g., 2019-03-01 ATL-CHI 4-OT). Volume scaling matches expected
#   OT-minute multipliers. No data anomalies.

# %%
pace_by_season = df.groupby("season")["poss"].mean().reindex(seasons_ordered)
print("Mean possessions per team-game by season:")
print(pace_by_season.round(2))

# %%
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(x, pace_by_season.values, marker="o", markersize=7, linewidth=2.0, color="darkgreen")

for season_key, (_, color) in ERA_MARKERS.items():
    if season_key in seasons_ordered:
        idx = seasons_ordered.index(season_key)
        ax.axvline(idx, color=color, linestyle="--", linewidth=1, alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("Mean possessions per team-game")
ax.set_title("Pace trend across 29 seasons (1997-98 to 2025-26)")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_pace_by_season.png", dpi=150)
plt.show()

# %% [markdown]
# ### 3c.2: Pace correlations with all 8 factors
#
# How does pace covary with each rate-normalized factor? The factors are
# already pace-neutral by construction, so any nonzero correlation reveals
# empirical style coupling: high-pace games are systematically different
# from low-pace games in ways the rate normalization doesn't capture.
#
# Predictions (sign only, magnitudes are empirical):
#
# - **eFG vs pace**: positive. High-pace games include transition shots,
#   which are more efficient than half-court shots.
# - **TOV vs pace**: positive. Rushed decisions in transition produce
#   more turnovers per possession.
# - **ORB vs pace**: ambiguous. Transition possessions produce fewer
#   long misses to crash; but high-pace teams may be self-selected as
#   athletic and capable of crashing.
# - **FT rate vs pace**: negative. Half-court sets generate more shooting
#   fouls than transition; high-pace games have fewer half-court sets.
#
# Same predictions apply to def_X factors by data symmetry. The
# off/def split is plotted in two colors so any asymmetry is visible.
#
# **Results:**
#
# - Sign predictions held for eFG (+) and FT (-). TOV vs pace at
#   +0.008 is essentially zero, contradicting the "rushed decisions"
#   prediction; consistent with Mike's TOV cancellation theory (pace
#   shortens possessions while rushed decisions increase errors, net
#   effect zero).
# - ORB vs pace at -0.18 came out clearly negative. Consistent with
#   high-pace games producing more long misses (transition rushed
#   shots, more 3PA) that are harder to crash on the offensive glass.
# - Off-def asymmetry in pace correlations: off_efg vs pace = +0.141
#   but def_efg vs pace = +0.210. Similar 7pp gap for ORB (-0.182 vs
#   -0.197). Not a bug; pace is computed per team-row, not per game,
#   so off vs def perspectives differ. Substantively: when your team
#   plays fast, the opponent's eFG rises slightly MORE than your own,
#   suggesting fast-paced teams "transmit" pace to opponents and give
#   up disproportionate transition opportunities. Worth a footnote
#   in the thesis methodology, not central.

# %%
pace_correlations = pd.Series({col: df["poss"].corr(df[col]) for col in OFF_COLS + DEF_COLS})
print("Pearson correlation of pace with each factor:")
print(pace_correlations.round(4))

# %%
fig, ax = plt.subplots(figsize=(12, 5))
labels = [FACTOR_LABELS[c] for c in pace_correlations.index]
colors = ["steelblue"] * 4 + ["firebrick"] * 4
bars = ax.bar(labels, pace_correlations.values, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Pearson r with pace (poss per team-game)")
ax.set_title("Pace coupling with the 8 factors\n29 seasons pooled (1997-98 to 2025-26)")
ax.tick_params(axis="x", rotation=20, labelsize=9)

for bar, val in zip(bars, pace_correlations.values, strict=False):
    offset = 0.01 if val >= 0 else -0.025
    ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:+.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "08_pace_factor_correlations.png", dpi=150)
plt.show()

# %% [markdown]
# ### 3c.3: The pace test, partial-correlation cross-block diagonals
#
# Compute cross-block diagonal correlations (off_X vs def_X) both with
# and without controlling for pace.
#
# Partial correlation formula:
#
#     r(X, Y | Z) = (r(X,Y) - r(X,Z) * r(Y,Z)) / sqrt((1 - r(X,Z)^2)(1 - r(Y,Z)^2))
#
# Interpretation of the shrinkage column:
#
# - 100% shrinkage: pace fully accounts for the coupling.
# - 0% shrinkage: pace contributes nothing.
# - Negative shrinkage: pace was suppressing the underlying coupling,
#   which would be a surprising result worth investigating.
#
# This is the strongest piece of mechanism discrimination in the whole
# notebook. Whatever the numbers come out to, write them up in the §7
# headline summary and reference them in the thesis methodology section.
#
# **Results (the headline finding of the notebook):**
#
# - FT rate diagonal (+0.175, the largest cross-block coupling) has
#   essentially zero pace shrinkage (-0.25%). Pace is eliminated as
#   the dominant mechanism for FT-rate cross-coupling.
# - TOV diagonal (+0.124) has zero pace shrinkage (0.02%). Pace
#   contributes nothing to TOV cross-coupling. When one team turns it
#   over more, the opponent also turns it over more, regardless of
#   pace. Pushes hard toward behavioral / style-matchup explanations.
# - eFG diagonal (+0.137) has 19% pace shrinkage. Pace explains about
#   a fifth of eFG cross-coupling. The remaining 81% needs another
#   mechanism. Smaller pace contribution than predicted.
# - ORB diagonal (+0.092) has 37% pace shrinkage, the largest of the
#   four. Within-game ORB coupling is substantially a pace-mediated
#   story: fewer misses in high-pace games means fewer rebounding
#   opportunities for both teams.
# - Pace is NOT the dominant per-game scalar explaining cross-block
#   coupling. The four-factor framework's independence assumption
#   fails in structured ways that pace alone does not explain.
# - Predicted shrinkages (40-60% eFG, 30-50% TOV, small FT) were
#   partially wrong. Real story: pace matters most for ORB (least
#   predicted) and barely at all for TOV and FT (most predicted).
# - Thesis methodology implication: pace control in regression
#   addresses ~20% of the eFG cross-block coupling, ~37% of ORB, and
#   essentially nothing for TOV or FT. Other mechanisms (officiating,
#   behavioral, score-state, opponent quality, style-matchup) need
#   investigation. See `docs/blog_ideas.md` entry #1 for the
#   testable-hypotheses framework.


# %%
def partial_correlation(df_in: pd.DataFrame, x: str, y: str, z: str) -> float:
    r_xy = df_in[x].corr(df_in[y])
    r_xz = df_in[x].corr(df_in[z])
    r_yz = df_in[y].corr(df_in[z])
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))


pace_test_rows = []
for off_col, def_col in zip(OFF_COLS, DEF_COLS, strict=False):
    raw = df[off_col].corr(df[def_col])
    partial = partial_correlation(df, off_col, def_col, "poss")
    shrinkage = (raw - partial) / raw if raw != 0 else 0.0
    pace_test_rows.append(
        {
            "pair": f"{FACTOR_LABELS[off_col]} vs {FACTOR_LABELS[def_col]}",
            "raw_r": raw,
            "partial_r_given_pace": partial,
            "shrinkage_pct": shrinkage * 100,
        }
    )

pace_test = pd.DataFrame(pace_test_rows)
print("Cross-block diagonals: raw vs partial-given-pace")
print(pace_test.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

# %% [markdown]
# ### 3c.4: Pace test extended to the off-diagonal cross-pairs
#
# The diagonal entries are the most natural test cases, but the
# off-diagonal cross-pairs (e.g. off_efg vs def_tov at -0.146 in §3)
# also showed structure. Run the same partial-correlation test on all
# 16 cross-block entries to see which couplings survive pace control.
#
# **Results:** Not analyzed in detail in this pass. The diagonal
# entries are covered more thoroughly by §3c.5's joint pace + 3PA
# table. The unique content here is the 12 off-diagonal cross-pairs
# (e.g. off_efg vs def_tov at -0.146). The before/after heatmaps are
# saved at `figures/eda_03/09_pace_test_cross_block.png`; if the
# off-diagonal pace shrinkage becomes relevant (e.g. for the
# mechanism-discrimination blog post), read the shrinkage off that
# figure or extend §3c.5's joint test to all 16 cross-block cells.

# %%
all_cross_rows = []
for off_col in OFF_COLS:
    for def_col in DEF_COLS:
        raw = df[off_col].corr(df[def_col])
        partial = partial_correlation(df, off_col, def_col, "poss")
        all_cross_rows.append(
            {
                "off": FACTOR_LABELS[off_col],
                "def": FACTOR_LABELS[def_col],
                "raw_r": raw,
                "partial_r": partial,
            }
        )

all_cross = pd.DataFrame(all_cross_rows)
raw_pivot = all_cross.pivot(index="off", columns="def", values="raw_r").reindex(
    index=[FACTOR_LABELS[c] for c in OFF_COLS],
    columns=[FACTOR_LABELS[c] for c in DEF_COLS],
)
partial_pivot = all_cross.pivot(index="off", columns="def", values="partial_r").reindex(
    index=[FACTOR_LABELS[c] for c in OFF_COLS],
    columns=[FACTOR_LABELS[c] for c in DEF_COLS],
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(
    raw_pivot,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-0.2,
    vmax=0.2,
    square=True,
    cbar_kws={"label": "Pearson r"},
    ax=axes[0],
)
axes[0].set_title("Cross-block raw correlations")
axes[0].set_xlabel("opponent factor (this team's defense)")
axes[0].set_ylabel("this team's offensive factor")

sns.heatmap(
    partial_pivot,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-0.2,
    vmax=0.2,
    square=True,
    cbar_kws={"label": "Partial r given pace"},
    ax=axes[1],
)
axes[1].set_title("Cross-block partial correlations given pace")
axes[1].set_xlabel("opponent factor (this team's defense)")
axes[1].set_ylabel("")

fig.suptitle(
    "Pace test on the full cross-block: before and after partialing out pace\n29 seasons pooled",
    fontsize=13,
    y=1.02,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "09_pace_test_cross_block.png", dpi=150)
plt.show()

# %% [markdown]
# ### 3c.5 Sidequest: 3-point attempt rate as a mechanism candidate
#
# §3c.3 found that pace explains the cross-block coupling unevenly:
# 37% of ORB, 19% of eFG, essentially nothing for TOV and FT. The
# remaining unexplained coupling needs another mechanism.
#
# A natural candidate: 3-point attempt rate, 3PA/FGA. Hypothesis:
# both teams' shot selection in a single game shares era-level
# strategic context. In modern games, both teams shoot more 3s. In
# 2003 games, neither does. Within a single game, the 3PA shares
# correlate moderately because pace, score state, and shared modern
# era strategy push them together.
#
# Mechanistic predictions:
#
# - eFG cross-coupling: 3PA-rate control should shrink the +0.137
#   coupling substantially. 3s have higher eFG by weight, so games
#   where both teams shoot many 3s produce co-elevated eFG.
# - ORB cross-coupling: 3PA-rate control should shrink the +0.092
#   coupling substantially beyond pace. 3PA misses are long rebounds,
#   harder to crash; 3-point heavy games have fewer ORB opportunities
#   for both teams.
# - TOV cross-coupling: 3PA-rate control probably doesn't matter much.
#   Mike's theory: pace shortens possessions (lowers TOV) but rushed
#   decisions increase errors (raises TOV); they cancel. 3PA rate is
#   only weakly related to this cancellation.
# - FT cross-coupling: 3PA-rate control might shrink slightly. More
#   3s means less rim attacking means lower FT rate. But the +0.175
#   coupling is mostly officiating/behavioral (working hypothesis),
#   not strategic mix.
#
# The joint partial test (control for both pace AND 3PA) tells us
# whether 3PA adds explanatory power on top of pace, or whether the
# two are so correlated that controlling for one is essentially
# controlling for the other.
#
# This sidequest is exploratory material for `docs/blog_ideas.md`
# entry #3 (3PA as a within-game coupling mechanism).
#
# **Results (the deepest finding of the notebook):**
#
# - 3PA rate is the dominant explainer of cross-block coupling for
#   every factor diagonal: 72% of eFG, 52% of ORB, 11% of TOV, 18%
#   of FT. Larger than pace's single-variable shrinkage in every
#   pair.
# - eFG headline: 72% of the +0.137 cross-block coupling is
#   explained by shared within-game 3PA rate. The four-factor
#   framework's "independent eFG" assumption fails primarily because
#   teams in the same game share era-level shot-mix strategy. Clean,
#   identifiable mechanism rather than a mystery.
# - ORB is unique: pace adds independent explanatory power on top of
#   3PA. 3PA-alone shrinkage 52%, joint shrinkage 65%, gap of 13pp.
#   Best mechanistic candidate for the independent pace effect is
#   2-point miss dynamics (transition rushed shots vs half-court set
#   shots). Worth a Session 13 regression sidebar.
# - For eFG, TOV, and FT, joint shrinkage essentially equals 3PA-alone
#   shrinkage. Pace is redundant once 3PA is controlled for these
#   three diagonals.
# - Pace and 3PA correlation is +0.379. Moderate. They rose together
#   over 29 seasons but the joint test cleanly separates their
#   contributions.
# - TOV diagonal: 86% of the +0.124 coupling remains unexplained
#   after both pace and 3PA are controlled. Pushes hard toward
#   behavioral or style-matchup explanations. The 3PA contribution
#   of 11% is small but non-zero.
# - FT diagonal: 80% of the +0.175 coupling remains unexplained.
#   Officiating crew and behavioral matching remain the leading
#   unidentified candidates.
# - Refined thesis methodology paragraph (replaces the §3c.3 draft):
#   "Within-team-game cross-block couplings between offensive and
#   defensive factors are non-zero for all four factor diagonals.
#   Joint partial-correlation tests with pace and 3PA rate as
#   controls explain 72% of the eFG coupling and 65% of the ORB
#   coupling, but only 14% of TOV and 20% of FT. The eFG and ORB
#   couplings are well-explained by shared within-game variables
#   (pace and shot mix); the TOV and FT couplings require auxiliary
#   mechanisms (behavioral matching, officiating crew, score state)
#   for which the four-factor framework alone does not provide
#   controls."
# - Predictions check: eFG 3PA shrinkage was a major underestimate
#   (predicted 40-50%, got 72%). The analytics-revolution-as-coupling
#   story is stronger than expected. ORB, TOV, and FT predictions
#   were close to actual.

# %%
required_3pa_cols = ["fg3a", "fga"]
missing_3pa = [c for c in required_3pa_cols if c not in df.columns]
if missing_3pa:
    print(f"Required columns not found: {missing_3pa}")
    print(f"Available columns: {sorted(df.columns.tolist())}")
    raise KeyError(f"Cannot compute 3PA rate, missing: {missing_3pa}")

df["fg3a_rate"] = df["fg3a"] / df["fga"].where(df["fga"] != 0)
print(f"3PA/FGA summary across {len(df):,} team-games:")
print(df["fg3a_rate"].describe().round(4))
print(f"\nPearson r between pace and 3PA/FGA: {df['poss'].corr(df['fg3a_rate']):+.4f}")

# %%
fg3a_by_season = df.groupby("season")["fg3a_rate"].mean().reindex(seasons_ordered)
print("\nMean 3PA/FGA by season:")
print(fg3a_by_season.round(4))

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(x, fg3a_by_season.values, marker="o", markersize=7, linewidth=2.0, color="purple")

for season_key, (_, color) in ERA_MARKERS.items():
    if season_key in seasons_ordered:
        idx = seasons_ordered.index(season_key)
        ax.axvline(idx, color=color, linestyle="--", linewidth=1, alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("Mean 3PA / FGA")
ax.set_title("3-point attempt rate trend across 29 seasons (1997-98 to 2025-26)")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_3pa_rate_by_season.png", dpi=150)
plt.show()

# %% [markdown]
# Sanity check on the pace-vs-3PA correlation above. If pace and
# 3PA-rate are highly correlated (e.g. r > 0.7), partialing out one
# is nearly equivalent to partialing out the other, and the joint
# test won't add much. If they're moderately correlated (0.3-0.6),
# they're related but distinguishable, and the joint test will tell
# us their independent contributions.


# %%
def partial_correlation_multi(df_in: pd.DataFrame, x: str, y: str, z_cols: list) -> float:
    """
    Partial correlation of x and y given multiple controls in z_cols,
    via OLS residualization. Equivalent to single-z partial_correlation()
    when len(z_cols) == 1, with negligible numerical drift.
    """
    n = len(df_in)
    Z = np.column_stack([np.ones(n), df_in[z_cols].values])
    x_vals = df_in[x].values
    y_vals = df_in[y].values

    b_x, _, _, _ = np.linalg.lstsq(Z, x_vals, rcond=None)
    b_y, _, _, _ = np.linalg.lstsq(Z, y_vals, rcond=None)

    x_resid = x_vals - Z @ b_x
    y_resid = y_vals - Z @ b_y
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


threepa_test_rows = []
for off_col, def_col in zip(OFF_COLS, DEF_COLS, strict=False):
    raw = df[off_col].corr(df[def_col])
    partial_pace = partial_correlation(df, off_col, def_col, "poss")
    partial_3pa = partial_correlation(df, off_col, def_col, "fg3a_rate")
    partial_both = partial_correlation_multi(df, off_col, def_col, ["poss", "fg3a_rate"])
    threepa_test_rows.append(
        {
            "pair": f"{FACTOR_LABELS[off_col]} vs {FACTOR_LABELS[def_col]}",
            "raw_r": raw,
            "partial_pace": partial_pace,
            "partial_3pa": partial_3pa,
            "partial_both": partial_both,
            "pace_shrink_pct": (raw - partial_pace) / raw * 100 if raw != 0 else 0,
            "3pa_shrink_pct": (raw - partial_3pa) / raw * 100 if raw != 0 else 0,
            "both_shrink_pct": (raw - partial_both) / raw * 100 if raw != 0 else 0,
        }
    )

threepa_test = pd.DataFrame(threepa_test_rows)
print("\nCross-block diagonals: pace, 3PA rate, and joint controls")
print(threepa_test.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

# %% [markdown]
# Read the table left-to-right:
#
# - `partial_pace`: §3c.3 result, repeated here for comparison.
# - `partial_3pa`: same test with 3PA/FGA as the control.
# - `partial_both`: joint control, both pace and 3PA/FGA partialed out.
# - `pace_shrink_pct`: fraction of raw coupling pace alone explains.
# - `3pa_shrink_pct`: fraction 3PA alone explains.
# - `both_shrink_pct`: fraction the two TOGETHER explain.
#
# The interesting comparisons:
#
# 1. `3pa_shrink` vs `pace_shrink` for each pair. Larger is the
#    bigger driver.
# 2. `both_shrink` vs the individual shrinkages. If both_shrink is
#    roughly equal to the larger single shrinkage, pace and 3PA are
#    explaining the same coupling (redundant controls). If both_shrink
#    is substantially larger, pace and 3PA explain independent
#    portions (additive controls).

# %% [markdown]
# ## Section 4: eFG% vs ORB% deep dive
#
# The headline negative correlation. Scatter of all team-game rows
# (subsampled for plotting), with the pooled OLS fit overlaid. The
# slope is the rate at which ORB% drops per unit of eFG% gain; the
# physical interpretation is "fraction of available offensive rebounds
# lost per percentage-point gain in scoring efficiency."
#
# **Results:**
#
# - OLS fit: `orb = -0.174 * efg + 0.347`. Per percentage point of
#   eFG% gain, ORB% drops 0.17 percentage points. A 10pp eFG%
#   improvement predicts a 1.7pp ORB% decrease.
# - Pearson r = -0.151, matches §1 (sanity check).
# - Variance explained: r^2 = 0.023. The eFG-ORB anticorrelation is
#   statistically robust but weak in effect size at the team-game
#   level. The mechanical floor operates at the possession level and
#   is heavily smoothed by aggregating ~85 shots per team-game.
# - The cloud is wide and roughly elliptical; the negative tilt is
#   real but visually subtle. Single team-game's ORB% is poorly
#   predicted by its eFG%. Cross-factor coupling is much cleaner at
#   the season-aggregate level (test deferred to Session 13).
# - Methodological footnote: visible horizontal striping at ORB% near
#   33% is a discrete-count artifact (small-integer numerators over
#   moderate-integer denominators producing recurring rational
#   fractions like 10/30, 8/24, etc.). Not a data anomaly; just the
#   nature of rate calculations from integer box-score components.

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
#
# **Results:**
#
# - **eFG vs FT rate is the cleanest era-level drift in the notebook.**
#   Monotone decline from +0.06 to +0.09 in 1997-2003 down to -0.02 to
#   0.00 in 2018-2026. Visually verified at full size in the §5 grid:
#   a slow, persistent, monotone-in-aggregate decoupling (total drop
#   ~0.11 over 29 seasons, ~0.0038 per year), swamped by noise at any
#   short timescale but unmistakable across the full series. No
#   reversion spikes. Unlike eFG vs ORB, this pair has a direction.
#   The 3-point revolution decoupled the two factors: 1997 era's
#   "high eFG = inside-out scoring = high FT rate" linkage gave way to
#   2025 era's "high eFG = 3-point volume, which doesn't draw fouls."
#   The last three seasons (2023-24 through 2025-26) sit flat near
#   zero rather than continuing down: the decoupling appears to have
#   stalled, not reversed. Too noisy to call, but worth noting against
#   the midrange-comeback hypothesis. A four-factor regression fit on
#   1997-2010 data would treat eFG and FT rate as positively coupled;
#   a 2018-2026 fit would treat them as independent.
# - **eFG vs TOV has a mild recent decline.** Earlier table reading
#   called this "no monotone trend," and across the full 29 seasons
#   that holds (noisy, range +0.03 to +0.18, 2013-14's +0.18 an
#   outlier). But the full-size grid panel shows the last three
#   seasons (2023-24 through 2025-26) declining cleanly from ~0.07 to
#   ~0.04, the most coherent move in the panel. Not strong enough to
#   call a trend, but possibly related to the midrange-comeback story
#   (midrange jumpers are low-turnover possessions, which would shift
#   the eFG-TOV relationship). Flag for EDA 04 follow-up.
# - **eFG vs ORB is a noisy, roughly flat series with no secular
#   trend.** Within-season mean around -0.075, year-to-year swings of
#   0.05-0.10 that swamp any drift. The 2025-26 reading (-0.068) is
#   statistically indistinguishable from 1997-98 (-0.038) given the
#   noise. The 3-point revolution did not change the within-game
#   eFG-ORB coupling. (Earlier read of a U-shape from cherry-picked
#   table values did not survive the §6 full-series chart.)
# - **Pooled vs within-season gap for eFG vs ORB.** Pooled r = -0.151
#   (§1), within-season mean ~ -0.075. The pooled value is twice the
#   within-season average. Since the within-season series is flat,
#   the entire extra negativity in the pooled value is era-trend
#   artifact: eFG rose and ORB fell as separate marginal trends, and
#   pooling all team-games captures that across-season covariance
#   (Simpson's-paradox-flavored aggregation). The within-game
#   coupling itself is era-invariant at roughly -0.075 (mechanical
#   floor is physics; it doesn't drift). This sharpens the §3c.5
#   caveat: 3PA rate, being era-correlated, is partly a modernity
#   proxy rather than a clean within-game mechanism. The honest
#   version of §3c.5 needs a within-season conditional test.
# - **The other four pairs are stable across 29 seasons.** eFG vs TOV
#   ranges +0.03 to +0.18 with no monotone trend (2013-14's +0.18 is
#   an outlier). TOV vs FT rate is consistently around +0.10
#   (rim-aggression coupling, stable across eras). TOV vs ORB and ORB
#   vs FT rate are noise-tier in every season.
# - **Thesis implication.** The four-factor framework's factor
#   relationships are NOT stable across the 29-year window. The
#   regression chapter must either fit era-specific models, include
#   year fixed effects, or include 3PA-rate controls. The 1997-2026
#   pooled four-factor regression would mask the eFG-FT decoupling
#   and the eFG-ORB U-shape.


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
    ax.set_ylim(-0.2, 0.2)

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
#
# **Results:**
#
# - None of the predicted shapes hold. The series is noisy and
#   roughly flat: mean around -0.075, year-to-year swings of
#   0.05-0.10 that swamp any trend. No monotone drift, no U-shape.
# - Two single-season negative spikes (1998-99 at -0.12, 2022-23 at
#   -0.14), both reverting immediately the following season. No
#   shared cause; most parsimonious explanation is noise. The
#   2022-23 spike is specifically NOT a take-foul-rule effect, since
#   a rule effect would persist rather than bounce back to -0.05.
# - Three single-season near-zero spikes (2008-09, 2018-19, 2019-20
#   all near -0.03 to -0.04), also reverting.
# - The only mild structural feature: 2012-2018 is a six-season band
#   consistently below -0.085, the most sustained stretch of stronger
#   coupling. Weak, and bracketed by the 2008-09 near-zero spike.
# - Headline interpretation: the within-game eFG-ORB coupling is
#   era-invariant at roughly -0.075. The mechanical floor (a make
#   produces no offensive rebound) is physics and does not drift; the
#   style component apparently does not drift either. The 3-point
#   revolution reshaped the marginal distributions of eFG and ORB
#   (EDA 02) but left their within-game joint coupling unchanged.
# - This is a cleaner finding than a trend would have been: it
#   isolates the pooled -0.151 (§1) as half genuine within-game
#   coupling and half era-trend aggregation artifact.

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
# filling in the three-bullet summary at the top of this notebook.
#
# **Results:** the three-bullet summary at the top of the notebook is
# this section's deliverable. After running §7, write the three
# bullets up top by selecting from across the Results blocks in
# §1-§6.

# %%
print("=" * 70)
print("SECTION 7: HEADLINE NUMBERS FOR THE WRITEUP")
print("=" * 70)

print("\n--- Pooled (29-season) offensive correlations, all 6 pairs ---")
print(f"  eFG% vs ORB%:    r = {df['off_efg_pct'].corr(df['off_orb_pct']):+.4f}")
print(f"  eFG% vs TOV%:    r = {df['off_efg_pct'].corr(df['off_tov_pct']):+.4f}")
print(f"  eFG% vs FT rate: r = {df['off_efg_pct'].corr(df['off_ft_rate']):+.4f}")
print(f"  ORB% vs TOV%:    r = {df['off_orb_pct'].corr(df['off_tov_pct']):+.4f}")
print(f"  ORB% vs FT rate: r = {df['off_orb_pct'].corr(df['off_ft_rate']):+.4f}")
print(f"  TOV% vs FT rate: r = {df['off_tov_pct'].corr(df['off_ft_rate']):+.4f}")

print("\n--- Cross-block diagonals (within-team-game off_X vs def_X) ---")
for off_col, def_col in zip(OFF_COLS, DEF_COLS, strict=False):
    r = df[off_col].corr(df[def_col])
    print(f"  {FACTOR_LABELS[off_col]:<12} vs {FACTOR_LABELS[def_col]:<12}: r = {r:+.4f}")

print("\n--- Pace + 3PA mechanism test (cross-block diagonals) ---")
print("  Shrinkage = fraction of raw coupling explained by the control")
for _, row in threepa_test.iterrows():
    print(
        f"  {row['pair']:<28} "
        f"pace {row['pace_shrink_pct']:+6.1f}%  "
        f"3PA {row['3pa_shrink_pct']:+6.1f}%  "
        f"both {row['both_shrink_pct']:+6.1f}%"
    )

print("\n--- eFG% vs FT rate decoupling (era-level drift) ---")
efg_ft_series = corr_drift["off_efg_pct_vs_off_ft_rate"].reindex(seasons_ordered)
print(f"  First 3 seasons mean: {efg_ft_series.head(3).mean():+.4f}")
print(f"  Last 3 seasons mean:  {efg_ft_series.tail(3).mean():+.4f}")
print(
    f"  Total drift: {efg_ft_series.tail(3).mean() - efg_ft_series.head(3).mean():+.4f} over 29 seasons"
)

print("\n--- eFG% vs ORB% within-season stability ---")
print(f"  Pooled r (all team-games): {df['off_efg_pct'].corr(df['off_orb_pct']):+.4f}")
print(f"  Within-season mean:        {efg_orb_series.mean():+.4f}")
print(
    f"  Within-season min / max:   {efg_orb_series.min():+.4f} ({efg_orb_series.idxmin()}) / {efg_orb_series.max():+.4f} ({efg_orb_series.idxmax()})"
)
print(f"  Within-season std:         {efg_orb_series.std():.4f}")
print("  Within-season series is flat/noisy with no trend; the pooled")
print("  value's extra negativity is era-trend aggregation artifact.")

print("\n--- Pace and 3PA descriptive ---")
print(f"  Mean pace (poss per team-game): {df['poss'].mean():.2f}")
print(f"  Pace correlation with 3PA rate: {df['poss'].corr(df['fg3a_rate']):+.4f}")
print(
    f"  3PA rate 1997-98 / 2025-26:     {fg3a_by_season.iloc[0]:.4f} / {fg3a_by_season.iloc[-1]:.4f}"
)
print("=" * 70)

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
# 2. **Within-season conditional 3PA test.** §3c.5 found 3PA explains
#    72% of the pooled eFG cross-block coupling, but §5/§6 showed the
#    within-season eFG-ORB coupling is flat while the pooled value is
#    inflated by era trend. Since 3PA rate is era-correlated, the
#    §3c.5 result is partly capturing era trend, not pure within-game
#    mechanism. The honest test: partial correlation conditioning on
#    3PA rate computed within each season, then averaged. Would
#    separate the within-game 3PA mechanism from the era-proxy effect.
# 3. **Across-teams (season-average) correlations.** Within-season
#    team-game correlations and across-teams season-average
#    correlations are different objects. Within-season captures
#    mechanics + within-season style; across-teams captures pure
#    strategy/style. Worth doing as a supplementary table if the
#    regression chapter wants to discuss factor independence at the
#    team-strategy level.
# 4. **Per-season drift of off-vs-def cross-block pairs.** Section 5
#    only does within-offensive-factor pairs. The 16 off-vs-def
#    cross-block pairs (e.g. off_efg vs def_efg over time) could show
#    pace-related or 3PA-related drift worth flagging.
# 5. **eFG-FT spotlight chart.** The eFG vs FT rate decoupling was
#    visually verified in the §5 grid but does not have a dedicated
#    spotlight figure. If the midrange-comeback blog post gets
#    written, generate the eFG-FT spotlight then as a figure for
#    that post (with the recent plateau called out).
# 6. **eFG-TOV recent decline.** The §5 grid showed eFG vs TOV
#    declining cleanly over the last three seasons (~0.07 to ~0.04).
#    Possibly related to the midrange comeback (midrange jumpers are
#    low-turnover possessions). Worth a closer look in EDA 04.
# 7. **ORB 2-point-miss decomposition.** §3c.5 found pace adds 13pp
#    of independent explanatory power on top of 3PA for the ORB
#    cross-block coupling. Best mechanistic candidate is 2-point miss
#    dynamics (transition rushed shots vs half-court set shots).
#    Decompose ORB into 2pt-miss-ORB and 3pt-miss-ORB, test whether
#    the independent pace effect concentrates in the 2pt component.
#    Session 13 regression sidebar.
#
# Blog post material surfaced during this notebook is captured in
# `docs/blog_ideas.md` entries #1 (mechanism discrimination), #2 (the
# two-pair structure), and #3 (3PA as within-game coupler, plus the
# player-driven midrange-comeback subhypothesis). The pooled-vs-
# within-season distinction from §5/§6 is also flagged there as a
# standalone methodological-post candidate.
