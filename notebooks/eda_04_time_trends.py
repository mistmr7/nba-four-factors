# %% [markdown]
# # EDA 04: Time Trends (29-season)
#
# **Session 12, post-EDA 03.** The consolidated NBA modernization-story
# notebook. EDA 02 looked at per-season factor distributions (violins,
# marginal shapes). EDA 03 looked at relationships between factors
# (cross-block coupling, pace and 3PA mechanisms). EDA 04 looks at
# league-wide trends over time: how each metric's central tendency has
# moved across 29 seasons.
#
# The critical additions beyond EDA 02 are the two metrics that are not
# part of the four factors and would otherwise never get a dedicated
# trend chart: 3-point attempt rate and pace. These are the actual
# engines of the modernization story; the four factors are downstream
# of them.
#
# This notebook is fully self-contained. It reproduces the four-factor
# per-season trends (which overlap with EDA 02 §2.1) cleanly rather than
# depending on EDA 02, so a thesis reader can use EDA 04 as the single
# modernization-story reference.
#
# Regular season only, matching EDA 02 and EDA 03.
#
# Carry-forward note: EDA 03 already computed pace by season (§3c.1) and
# 3PA rate by season (§3c.5) as sidequests. EDA 04 reproduces both as
# first-class figures with full era annotations.
#
# **Three-bullet summary:**
#
# 1. **3PA rate is the master modernization variable.** Five of the
#    six trends are tightly coupled to it at the season-trend level:
#    eFG% (r = +0.970), pace (+0.957), TOV% (-0.912), FT rate
#    (-0.890), margin variance (+0.876). The 4-factor framework's
#    metrics are largely downstream signatures of one underlying shift
#    (the 3-point revolution), not independent trends. eFG% rose
#    +0.067 from 1997-98 to 2025-26; 3PA rate nearly tripled from
#    0.160 to 0.416 over the same window. This is the season-trend
#    echo of EDA 03 §3c.5's within-game finding that 3PA explained
#    72% of the eFG cross-block coupling.
# 2. **ORB% is the subheader: it tracked the master variable for 23
#    seasons then broke from it.** ORB% bottomed at 0.220 in 2020-21
#    and has recovered to 0.258 in 2025-26, +0.038 in just 5 seasons
#    of monotone increase. The era-split correlation against 3PA
#    reverses sign: -0.953 (decline 1997-2021) to +0.748 (recovery
#    2021-2026). The 2025-26 season specifically shows three
#    independent signals of a possible midrange-strategy comeback
#    (3PA dipped, ORB% continued rising, FT rate jumped +0.021, the
#    largest single-year FT-rate increase in two decades). Whether
#    this is a few outlier teams (HOU/Durant, SAC/DeRozan,
#    PHX/Booker) or a broader shift is testable but parked
#    (followup #3).
# 3. **Era-split regression: the framework still fits, but with
#    smaller weights and modest ORB share shift.** Decline-era and
#    recovery-era four-factor regressions on win_pct have nearly
#    equal R^2 (0.927 vs 0.939), but every standardized coefficient
#    shrunk in the recovery era (typically 25-45%). Win_pct variance
#    has compressed; teams are bunched closer in season records.
#    ORB% offensive-weight share rose modestly (+1.4pp), FT rate's
#    dropped most (-1.9pp). eFG% dominates at 50% share in both eras,
#    above the canonical Dean Oliver 40%. These shifts inform the
#    Session 13 regression spec: pooled 1997-2026 fits would mask
#    real era-dependent structure.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from nba_four_factors.analysis import load_processed, pivot_to_game_level
from nba_four_factors.config import SeasonType

sns.set_theme(style="whitegrid", context="notebook")
FIG_DIR = Path("figures/eda_04")
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

FACTOR_LABELS = {
    "off_efg_pct": "Effective FG%",
    "off_tov_pct": "Turnover %",
    "off_orb_pct": "Offensive Rebound %",
    "off_ft_rate": "FT Rate (FTA/FGA)",
}

# %%
df = load_processed(("1997_98", "2025_26"), SeasonType.REGULAR)
print(f"long-format shape: {df.shape}")
print(f"seasons: {df['season'].nunique()}")

seasons_ordered = sorted(df["season"].unique())
seasons_display = [s.replace("_", "-") for s in seasons_ordered]
x = np.arange(len(seasons_ordered))


def add_era_markers(ax, annotate: bool = False) -> None:
    """
    Draw the standard era marker vertical lines on a season-indexed axis.
    Set annotate=True to also write rotated era labels near the top.
    """
    y_top = ax.get_ylim()[1]
    for season_key, (label, color) in ERA_MARKERS.items():
        if season_key in seasons_ordered:
            idx = seasons_ordered.index(season_key)
            ax.axvline(idx, color=color, linestyle="--", linewidth=1, alpha=0.4)
            if annotate:
                ax.annotate(
                    label,
                    xy=(idx, y_top),
                    xytext=(idx - 0.12, y_top),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=color,
                    alpha=0.85,
                    rotation=90,
                )


def season_trend_plot(
    series: pd.Series, ylabel: str, title: str, color: str, fname: str, percent: bool = False
) -> None:
    """
    Standard single-metric season trend line chart with era markers.
    series is indexed by season key, reindexed to seasons_ordered.
    """
    vals = series.reindex(seasons_ordered)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, vals.values, marker="o", markersize=7, linewidth=2.0, color=color)
    add_era_markers(ax, annotate=True)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.show()


# %% [markdown]
# ## Section 2: League-wide eFG% by season
#
# The headline four-factor modernization metric. Per-season mean of
# off_efg_pct.
#
# Prediction: rise from ~0.50 toward ~0.55+ over the window. EDA 02 §2.1
# already established the rough shape; this is the clean thesis-figure
# version. Watch the last few seasons: if the midrange comeback is real
# (3PA rate dipped in 2025-26, see §3), eFG% growth could be stalling,
# since midrange 2s are lower-eFG than 3s.
#
# **Results:**
#
# - eFG% rose substantially across the 29-season window: 0.480 in
#   1997-98 to 0.547 in 2025-26, a +0.067 absolute gain (about 14%
#   relative). The curve shape closely tracks 3PA rate (§3); the §9c
#   season-level correlation matrix quantifies it at r = +0.970, the
#   single strongest pair in the matrix.
# - Interpretation: eFG% is largely downstream of 3PA adoption. As
#   teams shifted shot mix toward 3s (worth 1.5x in the eFG formula),
#   league eFG% rose mechanically, not necessarily because shooting
#   itself improved. The 2pt/3pt eFG decomposition (followup #4) would
#   separate "shot-mix shift" from "shooting got better." The r =
#   0.970 leaves only ~3% of variance unexplained by 3PA at the
#   season-trend level; almost all of the eFG% rise is mix-shift.
# - This is the season-level echo of EDA 03 §3c.5's within-game
#   finding that 3PA rate explained 72% of the eFG cross-block
#   coupling. The same master variable shows up at both the
#   within-game and the season-trend level.

# %%
efg_by_season = df.groupby("season")["off_efg_pct"].mean()
print("Mean off_efg_pct by season:")
print(efg_by_season.reindex(seasons_ordered).round(4))

# %%
season_trend_plot(
    efg_by_season,
    ylabel="League mean eFG%",
    title="League-wide eFG% by season (1997-98 to 2025-26)",
    color="steelblue",
    fname="01_efg_by_season.png",
    percent=True,
)

# %% [markdown]
# ## Section 3: 3-point attempt rate by season
#
# The actual three-point revolution metric: FG3A / FGA. This is the most
# dramatic chart in the notebook. Reproduces EDA 03 §3c.5's series as a
# first-class figure.
#
# Prediction: dramatic monotone rise from ~0.16 in 1997-98 to ~0.42 in
# 2024-25, with a small dip to ~0.42 in 2025-26 (the first non-monotonic
# move in the series, and the data anchor for the player-driven
# midrange-comeback hypothesis from `docs/blog_ideas.md` entry #3).
#
# **Results:**
#
# - The most dramatic trend in the notebook. 3PA/FGA rose from 0.160
#   in 1997-98 to 0.421 in 2024-25, then dipped to 0.416 in 2025-26.
#   Nearly a tripling of three-point shot share over 29 seasons.
# - The 2025-26 dip is the first non-monotonic move in the series.
#   Whether it is a one-season blip or the start of a sustained
#   reversal is the open question. The §3b year-over-year chart shows
#   it as a single negative bar after a long run of positive bars.
# - The dip is the data anchor for the player-driven midrange-comeback
#   hypothesis (blog_ideas.md entry #3): the league-wide drop may be
#   a few outlier teams with elite midrange scorers (HOU/Durant,
#   SAC/DeRozan, PHX/Booker), not a uniform shift. Testing that
#   requires per-team dispersion analysis (followup #3).
# - 3PA rate is the master variable of the modernization story.
#   eFG% (§2), TOV% (§4), pace (§7), and margin variance (§8) all
#   read as downstream fingerprints of this one trend.

# %%
required_3pa_cols = ["fg3a", "fga"]
missing_3pa = [c for c in required_3pa_cols if c not in df.columns]
if missing_3pa:
    print(f"Required columns not found: {missing_3pa}")
    print(f"Available columns: {sorted(df.columns.tolist())}")
    raise KeyError(f"Cannot compute 3PA rate, missing: {missing_3pa}")

df["fg3a_rate"] = df["fg3a"] / df["fga"].where(df["fga"] != 0)
fg3a_by_season = df.groupby("season")["fg3a_rate"].mean()
print("Mean 3PA/FGA by season:")
print(fg3a_by_season.reindex(seasons_ordered).round(4))

# %%
season_trend_plot(
    fg3a_by_season,
    ylabel="League mean 3PA / FGA",
    title="3-point attempt rate by season (1997-98 to 2025-26)",
    color="purple",
    fname="02_3pa_rate_by_season.png",
    percent=True,
)

# %% [markdown]
# ### 3b: 3PA rate year-over-year change
#
# The level chart shows the hockey stick. The first-difference chart
# shows where the acceleration and deceleration happened: which specific
# seasons saw the fastest adoption, and whether the recent dip is a
# one-season blip or the start of a sustained reversal.

# %%
fg3a_yoy = fg3a_by_season.reindex(seasons_ordered).diff()
print("Year-over-year change in 3PA/FGA:")
print(fg3a_yoy.round(4))

fig, ax = plt.subplots(figsize=(15, 5))
colors = ["firebrick" if v < 0 else "seagreen" for v in fg3a_yoy.fillna(0).values]
ax.bar(x, fg3a_yoy.fillna(0).values, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
add_era_markers(ax)
ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("YoY change in 3PA / FGA")
ax.set_title("3-point attempt rate: year-over-year change (1997-98 to 2025-26)")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_3pa_rate_yoy_change.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 4: TOV% by season
#
# Per-season mean of off_tov_pct.
#
# Prediction: slow decline. TOV% is the quietest of the four factors
# historically. Worth checking whether the decline is monotone or has
# structure (era inflections, lockout-season anomalies).
#
# **Results:**
#
# - TOV% dropped substantially across the window: 0.145 in 1997-98 to
#   0.127 in 2025-26, a -0.018 absolute decline (about 12% relative).
#   Larger than the "slow decline" prediction implied. The decline
#   tracks 3PA rate inversely: §9c reports r = -0.912.
# - Working hypothesis: two mechanisms point the same direction.
#   Quicker possessions in the high-pace modern game leave less time
#   per possession to commit a turnover, and the shift away from
#   interior touches means fewer turnovers in the crowded paint.
# - Note this is the season-trend complement to EDA 03's finding that
#   the TOV cross-block coupling was the LEAST explained by pace and
#   3PA (essentially zero pace shrinkage, 11% 3PA shrinkage). Not a
#   contradiction: the league-wide LEVEL of TOV% moved with the 3PA
#   era, but the within-game COUPLING between two opposing teams' TOV
#   rates did not. Different questions, both true.

# %%
tov_by_season = df.groupby("season")["off_tov_pct"].mean()
print("Mean off_tov_pct by season:")
print(tov_by_season.reindex(seasons_ordered).round(4))

# %%
season_trend_plot(
    tov_by_season,
    ylabel="League mean TOV%",
    title="League-wide TOV% by season (1997-98 to 2025-26)",
    color="darkorange",
    fname="04_tov_by_season.png",
    percent=True,
)

# %% [markdown]
# ## Section 5: ORB% by season
#
# Per-season mean of off_orb_pct.
#
# Prediction: the conventional story is steady decline as a corollary of
# more 3-pointers (long misses are harder to crash, transition-defense
# priority rises). But EDA 02 §1 already flagged a U-shape: ORB% bottomed
# out around 2022 and started recovering. This section confirms and
# quantifies that U-shape. The U-shape is a genuine finding because the
# "corollary of more threes" story predicts monotone decline and the
# data contradicts it for the last 3-4 seasons.
#
# **Results:**
#
# - ORB% shows a clean U-shape (technically a checkmark: a long
#   23-season decline followed by a sharp 5-season recovery).
#   1997-98 was the peak at 0.312 (max season). The minimum was
#   2020-21 at 0.220, exactly when COVID disruption peaked and 3PA
#   share was rising fastest. 2025-26 has recovered to 0.258, a
#   gain of +0.038 off the bottom in just 5 seasons. Five
#   consecutive seasons of monotone increase; this is not noise.
# - The decline phase tracks the 3PA rise: long three-point misses
#   are harder to crash, and transition-defense priority rises as
#   offenses spread the floor. That story explains the first ~23
#   seasons.
# - The recovery phase contradicts that story. Teams are crashing
#   the offensive glass more again in the last 5 seasons despite
#   3PA rate staying high (and only dipping marginally in 2025-26).
#   The "more threes means less offensive rebounding" mechanism
#   cannot explain a recovery while threes stay elevated.
# - Subheader to the master-variable story: every other modernization
#   metric tracked 3PA rate cleanly. ORB% tracked it through 2020-21
#   and then broke from it. The §9c matrix detects this implicitly,
#   ORB% x margin std = -0.585 is the single lowest-magnitude
#   correlation in the entire matrix because ORB%'s U-shape runs
#   against margin std's monotone rise. The single full-window
#   correlation in §9c (ORB% x 3PA at -0.838) blends the long
#   decline with the recent reversal; the §5b era-split corrects
#   for this.
# - Open question raised during review: does transition offense
#   follow the same shape as the ORB recovery? Two opposite causal
#   stories (offensive-glass crashing generates transition the other
#   way, vs teams crash because they have gotten better at transition
#   defense and can afford to). Transition data would distinguish
#   these. Not measurable from box-score columns; requires pbpstats
#   play-type data. Parked as a sidequest, see followups and
#   blog_ideas.md.

# %%
orb_by_season = df.groupby("season")["off_orb_pct"].mean()
print("Mean off_orb_pct by season:")
print(orb_by_season.reindex(seasons_ordered).round(4))

print(f"\nMin ORB% season: {orb_by_season.idxmin()} at {orb_by_season.min():.4f}")
print(f"Max ORB% season: {orb_by_season.idxmax()} at {orb_by_season.max():.4f}")
print(
    f"2025-26 vs the min: {orb_by_season.reindex(seasons_ordered).iloc[-1] - orb_by_season.min():+.4f}"
)

# %%
season_trend_plot(
    orb_by_season,
    ylabel="League mean ORB%",
    title="League-wide ORB% by season (1997-98 to 2025-26)",
    color="seagreen",
    fname="05_orb_by_season.png",
    percent=True,
)

# %% [markdown]
# ### 5b: ORB% x 3PA correlation, era-split
#
# The §9c matrix reports the full-window ORB% x 3PA correlation at
# -0.838. But §5 just established that ORB% is U-shaped: a long
# decline through 2020-21 followed by a 5-season recovery. The single
# full-window correlation blends the decline (where ORB and 3PA moved
# strongly inversely) with the recovery (where ORB rose despite 3PA
# staying high). The blend dilutes the negative coupling.
#
# Split the correlation by era to put a number on the reversal:
# decline window (1997-98 through 2020-21, 24 seasons) versus recovery
# window (2021-22 through 2025-26, 5 seasons).
#
# Caveat: 5 seasons is a tiny correlation sample. The recovery-era
# correlation has wide uncertainty and should be read as direction
# only, not magnitude.
#
# **Results:**
#
# - Decline-only window (1997-98 to 2020-21, 24 seasons): r = -0.953.
#   Essentially as strong a negative coupling as 24 noisy data points
#   can produce. The mechanical "more threes means less offensive
#   rebounding" story is correct for this phase.
# - Full window (29 seasons): r = -0.838. The summary value, attenuated
#   by the recovery.
# - Recovery-only window (2021-22 to 2025-26, 5 seasons): r = +0.748.
#   Positive. The relationship between ORB% and 3PA rate has reversed
#   sign in the recovery era. Both metrics are trending up together
#   for the first time in the 29-season window.
# - Caveat: 5 data points. The +0.748 magnitude is fragile (a single
#   season landing differently would shift it substantially) but the
#   direction is solid because both series are monotone upward across
#   the 5 seasons.
# - Implication: the full-window -0.838 in §9c is not just hiding
#   attenuation; it is hiding a sign flip. ORB% has gone from
#   anticorrelated with 3PA (decline era) to positively correlated
#   with 3PA (recovery era). The 9c matrix detects this implicitly
#   through the ORB row's unusually weak correlations (especially
#   ORB% x margin std at -0.585, the matrix's lowest off-diagonal
#   magnitude), but the era-split makes the reversal explicit.

# %%
decline_seasons_subset = [s for s in seasons_ordered if s <= "2020_21"]
recovery_seasons_subset = [s for s in seasons_ordered if s >= "2021_22"]

orb_decline = orb_by_season.reindex(decline_seasons_subset)
fg3a_decline = fg3a_by_season.reindex(decline_seasons_subset)
orb_recovery = orb_by_season.reindex(recovery_seasons_subset)
fg3a_recovery = fg3a_by_season.reindex(recovery_seasons_subset)

r_full = orb_by_season.reindex(seasons_ordered).corr(fg3a_by_season.reindex(seasons_ordered))
r_decline = orb_decline.corr(fg3a_decline)
r_recovery = orb_recovery.corr(fg3a_recovery)

print("ORB% x 3PA rate correlation by window:")
print(f"  Full window   (1997-98 to 2025-26, {len(seasons_ordered)} seasons): r = {r_full:+.4f}")
print(
    f"  Decline only  (1997-98 to 2020-21, {len(decline_seasons_subset)} seasons): r = {r_decline:+.4f}"
)
print(
    f"  Recovery only (2021-22 to 2025-26, {len(recovery_seasons_subset)} seasons): r = {r_recovery:+.4f}"
)
print(f"\nAttenuation: full-window correlation is {r_full - r_decline:+.4f} less negative")
print("than decline-only, quantifying how much the 5-season recovery diluted")
print("the long-term inverse coupling.")

# %% [markdown]
# ## Section 6: FT rate by season
#
# Per-season mean of off_ft_rate (FTA/FGA, the canonical formulation;
# EDA 02 §2b explored the FTM/FGA alternative).
#
# Prediction: year-to-year fluctuation, no strong secular trend. This is
# the null-result section. Confirming "no trend" is still worth doing,
# and it provides a contrast against the dramatic 3PA and eFG trends.
#
# **Results: the no-trend prediction was wrong.**
#
# - FT rate declined substantially across the window: 0.336 in
#   1997-98 to 0.267 in 2025-26, a -0.069 absolute decline (about
#   21% relative). Comparable in magnitude to the ORB% decline.
#   §9c reports the correlation against 3PA at -0.890, one of the
#   strongest negative trend correlations in the matrix. Not a null
#   result.
# - The trend has more structure than the other modernization
#   metrics. There is a mid-2000s spike (peak 0.339 in 2005-06)
#   during the slow-down/grinding era when half-court interior
#   offense dominated and drew more shooting fouls. Then a steady
#   decline through 2020-21 (0.250) as the perimeter shift reduced
#   foul-drawing. Then a recent uptick: 0.267 in 2025-26, up from
#   0.246 in 2024-25.
# - **The 2025-26 jump is large.** +0.021 in one season, the biggest
#   single-year FT rate increase in over 20 years. It is the third
#   independent 2025-26 signal pointing toward the midrange-comeback
#   hypothesis (the others being the 3PA dip in §3 and the ORB
#   recovery acceleration in §5). More interior shooting means more
#   drives into contact, which lifts FT rate. Three coincident
#   signals from three independent metrics in the same season is
#   stronger evidence than any one alone.
# - This is also the season-trend complement to EDA 03 §5's
#   within-season eFG-FT decoupling finding. Same underlying
#   mechanism (the perimeter shift reduces foul-drawing), visible at
#   both the trend level here and the correlation-drift level in
#   EDA 03.

# %%
ft_by_season = df.groupby("season")["off_ft_rate"].mean()
print("Mean off_ft_rate by season:")
print(ft_by_season.reindex(seasons_ordered).round(4))

print(f"\nFT rate range across 29 seasons: [{ft_by_season.min():.4f}, {ft_by_season.max():.4f}]")
print(f"Spread: {ft_by_season.max() - ft_by_season.min():.4f}")

# %%
season_trend_plot(
    ft_by_season,
    ylabel="League mean FT rate (FTA/FGA)",
    title="League-wide FT rate by season (1997-98 to 2025-26)",
    color="firebrick",
    fname="06_ft_rate_by_season.png",
    percent=True,
)

# %% [markdown]
# ## Section 7: Pace by season
#
# Reproduces EDA 03 §3c.1 as a first-class figure.
#
# Pace formula: `possessions = FGA + 0.44 * FTA + TOV - OREB`, per-season
# mean.
#
# Important caveat on units: `extraEDAfun.md` specifies "possessions per
# 48 minutes," but the processed layer does not currently carry a
# minutes-played column, so this section computes possessions per
# team-game instead. For non-overtime games the two are identical; for
# OT games, per-game slightly overcounts relative to per-48. OT games are
# under 5% of the data, so the trend shape is unaffected, but the
# absolute number is not directly comparable to published NBA "pace"
# figures (which are per-48). The §10 headline numbers label this
# explicitly as "possessions per team-game" to avoid confusion.
#
# If a minutes column is added to the processed layer later, switch to
# true per-48 by dividing poss by (minutes / 48) in the cell below.
#
# **Results:**
#
# - Pace rose with 3PA rate across the window. Mean possessions per
#   team-game went from 93.7 in 1997-98 to 102.6 in 2025-26.
# - Confirms EDA 03 §3c.1's three-era structure: dead-ball era start
#   (1997-2003, ~93-96), grinding era (2003-2012, flat at ~93-95),
#   pace-and-space adoption (2013-2017, rise to ~97-100), modern fast
#   era (2018-2026, stable at ~101-103). The biggest single-season
#   jump remains 2017-18 to 2018-19, coinciding with the 14-second
#   shot-clock reset rule.
# - Pace and 3PA rose together. EDA 03 §3c.5 measured their
#   correlation at +0.379 at the team-game level; the §9c season-level
#   trend matrix shows the trend co-movement is much higher (the two
#   league-wide series are near-collinear over the window).
# - Units caveat carries into §10: this is possessions per team-game,
#   not the published per-48 "pace" metric.

# %%
required_pace_cols = ["fga", "fta", "tov", "oreb"]
missing_pace = [c for c in required_pace_cols if c not in df.columns]
if missing_pace:
    print(f"Required columns not found: {missing_pace}")
    print(f"Available columns: {sorted(df.columns.tolist())}")
    raise KeyError(f"Cannot compute pace, missing: {missing_pace}")

minutes_candidates = ["min", "mp", "minutes", "min_played"]
minutes_col = next((c for c in minutes_candidates if c in df.columns), None)
if minutes_col is not None:
    print(f"Minutes column found ({minutes_col}). True per-48 pace is available.")
    print("To switch: poss_per48 = df['poss'] / (df[minutes_col] / 48)")
else:
    print("No minutes column in processed layer. Using possessions per team-game.")

df["poss"] = df["fga"] + 0.44 * df["fta"] + df["tov"] - df["oreb"]
pace_by_season = df.groupby("season")["poss"].mean()
print("\nMean possessions per team-game by season:")
print(pace_by_season.reindex(seasons_ordered).round(2))

# %%
season_trend_plot(
    pace_by_season,
    ylabel="Mean possessions per team-game",
    title="Pace by season (possessions per team-game, 1997-98 to 2025-26)",
    color="darkgreen",
    fname="07_pace_by_season.png",
    percent=False,
)

# %% [markdown]
# ## Section 8: Margin variance by season
#
# Per-season standard deviation of game margin. The "has the league
# become more or less competitive" question.
#
# EDA 02 §5 touched this (the X-shape finding: HCA declining while margin
# std rising). This section is the clean standalone version. Unlike
# every other section in this notebook, it requires the game-level pivot
# rather than working at team-game level, because margin is a
# game-level quantity.
#
# Prediction: from EDA 02 §5, single-game margin std has been rising,
# especially post-2010. Confirm and quantify.
#
# **Results:**
#
# - Single-game margin standard deviation increased substantially:
#   13.35 in 1997-98 to 16.43 in 2025-26, a +3.08 absolute gain
#   (about 23% relative). Confirms the EDA 02 §5 observation.
# - §9c reports the season-trend correlation against 3PA at +0.876,
#   strong but not as tight as eFG and pace's correlations with 3PA.
#   Margin std is downstream of the modernization trend but with
#   more noise than the in-game factors.
# - Mechanism: 3-point shooting is a higher-value, lower-probability
#   shot than the interior shots it replaced. As it became the
#   dominant offensive strategy, game-to-game outcomes became more
#   variable, because the variance of a three-heavy offense is
#   structurally higher than a two-heavy one. More variance per
#   possession compounds into more variance per game.
# - This connects to EDA 02 §5's X-shape finding (home court advantage
#   declining while margin std rising). EDA 04 §8 isolates the
#   margin-std half of that X as a clean standalone trend.
# - Followup: decompose whether rising variance is driven by more
#   blowouts, fewer close games, or both. The std alone does not
#   distinguish these; it needs the margin distribution shape per
#   season (followup #5).

# %%
games = pivot_to_game_level(df)
print(f"game-level shape after pivot: {games.shape}")

margin_std_by_season = games.groupby("season")["home_margin"].std()
margin_mean_by_season = games.groupby("season")["home_margin"].mean()
print("\nPer-season home margin std and mean:")
margin_summary = pd.DataFrame(
    {
        "margin_std": margin_std_by_season,
        "margin_mean": margin_mean_by_season,
    }
).reindex(seasons_ordered)
print(margin_summary.round(3))

# %%
season_trend_plot(
    margin_std_by_season,
    ylabel="Std of home margin (pts)",
    title="Single-game margin variance by season (1997-98 to 2025-26)",
    color="indianred",
    fname="08_margin_std_by_season.png",
    percent=False,
)

# %% [markdown]
# ## Section 9: The combined modernization chart
#
# The payoff section. Two views of all six metrics together.
#
# 9a is a small-multiples grid in true units: the safer thesis figure,
# easier to read each trend on its own scale.
#
# 9b is a z-scored overlay: all six series standardized to mean 0 and
# std 1, plotted on one axis. The blog-post figure, with the visual
# punch of "everything moving together" (or not).
#
# **Results:**
#
# - The small-multiples grid (§9a) and z-scored overlay (§9b) both
#   show the unifying thread: five of the six metrics are downstream
#   fingerprints of 3PA adoption. eFG% tracks it directly (r =
#   +0.970, the strongest pair in the matrix), TOV% inversely
#   (-0.912), pace directly (+0.957), margin variance directly
#   (+0.876), FT rate strongly inversely (-0.890). Only ORB%'s
#   recent U-shape recovery breaks cleanly from the 3PA story.
# - 3PA x Pace at +0.957 is near-collinear at the season level.
#   At the team-game level (EDA 03 §3c.5) the same pair correlated
#   at +0.379, much smaller. The aggregation pulls out their shared
#   secular trend. Methodological implication for Session 13: at the
#   season-aggregation resolution, pace and 3PA cannot be cleanly
#   separated in regression. Year fixed effects or a per-game
#   resolution are needed to distinguish their effects.
# - The single weakest off-diagonal correlation in the entire 7x7
#   matrix is ORB% x margin std at -0.585. Every other cross-pair
#   sits in the |r| = 0.78 to 0.97 range. The unusually low ORB
#   correlation is the matrix's signature of the U-shape: margin std
#   rose monotonically while ORB% U-turned, so the line-of-best-fit
#   summary blurs. Reading low correlations in this matrix as
#   "structural break" rather than "weak coupling" is the correct
#   interpretation.
# - Headline framing for the thesis: the NBA modernization story is
#   largely one variable (3PA rate) with multiple downstream
#   signatures, not several independent trends. ORB% is the
#   subheader: it tracked the master variable for 23 seasons then
#   broke from it. This is the season-trend echo of EDA 03 §3c.5's
#   within-game finding that 3PA rate dominated the cross-block
#   factor coupling.
# - Caveat: the season-level correlations in §9c are inflated by
#   shared monotone trend (29 near-monotone data points; near-
#   collinearity is the default for any two trending series). They
#   are descriptive support for the visual co-movement story, not a
#   causal claim. The honest version of the unifying-thread argument
#   leans on mechanism (the eFG formula weights 3s at 1.5x; quicker
#   possessions reduce TOV opportunities; etc.) rather than on the
#   correlation values alone. The §11 era-split regression preview
#   pushes this further by testing whether the factor weights
#   themselves changed across eras.

# %%
trend_series = {
    "eFG%": efg_by_season,
    "3PA rate": fg3a_by_season,
    "TOV%": tov_by_season,
    "ORB%": orb_by_season,
    "FT rate": ft_by_season,
    "Pace (poss/team-game)": pace_by_season,
}
trend_colors = {
    "eFG%": "steelblue",
    "3PA rate": "purple",
    "TOV%": "darkorange",
    "ORB%": "seagreen",
    "FT rate": "firebrick",
    "Pace (poss/team-game)": "darkgreen",
}

# %%
fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
for ax, (name, series) in zip(axes.ravel(), trend_series.items(), strict=False):
    vals = series.reindex(seasons_ordered)
    ax.plot(x, vals.values, marker="o", markersize=5, linewidth=1.8, color=trend_colors[name])
    add_era_markers(ax)
    ax.set_title(name, fontsize=12)

for ax in axes[-1, :]:
    ax.set_xticks(x[::3])
    ax.set_xticklabels(
        [seasons_display[i] for i in range(0, len(seasons_display), 3)],
        rotation=70,
        ha="right",
        fontsize=8,
    )

fig.suptitle(
    "The NBA modernization story: six metrics, 1997-98 to 2025-26\nSmall multiples, true units",
    fontsize=14,
    y=1.00,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "09a_modernization_grid.png", dpi=150)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(16, 7))
for name, series in trend_series.items():
    vals = series.reindex(seasons_ordered)
    z = (vals - vals.mean()) / vals.std()
    ax.plot(
        x, z.values, marker="o", markersize=4, linewidth=1.8, color=trend_colors[name], label=name
    )

ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
add_era_markers(ax)
ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("Z-score (standardized within metric)")
ax.set_title("The NBA modernization story: six metrics z-scored on one axis\n1997-98 to 2025-26")
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "09b_modernization_zscored.png", dpi=150)
plt.show()

# %% [markdown]
# ### 9c: Season-level trend correlation matrix
#
# The §9b overlay makes co-movement visible; this makes it citable.
# Each metric is a 29-point series (one value per season). The
# correlation matrix below is the pairwise Pearson correlation of
# those season-level series, NOT the within-game team-level
# correlations from EDA 03. A high value here means "these two
# league-wide trends moved together across 29 seasons."
#
# This directly supports the unifying-thread framing: if 3PA rate
# correlates strongly with eFG%, TOV%, pace, and margin variance at
# the season level, the modernization story is one variable (3PA
# adoption) with multiple downstream fingerprints rather than several
# independent trends.
#
# Caveat: 29 data points, and most of these series are near-monotone
# over the window, so almost any two will correlate highly just from
# shared trend. This is descriptive support for the visual story, not
# a causal claim. Spurious-correlation-of-trending-series applies.

# %%
trend_df = pd.DataFrame(
    {name: series.reindex(seasons_ordered) for name, series in trend_series.items()}
)
trend_df["margin std"] = margin_std_by_season.reindex(seasons_ordered)

trend_corr = trend_df.corr()
print("Season-level trend correlation matrix (29-point series):")
print(trend_corr.round(3))

# %%
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    trend_corr,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    vmin=-1.0,
    vmax=1.0,
    square=True,
    cbar_kws={"label": "Pearson r (season-level series)"},
    ax=ax,
)
ax.set_title(
    "Season-level trend correlations, 1997-98 to 2025-26\nHow the seven modernization metrics co-moved across 29 seasons"
)
plt.tight_layout()
plt.savefig(FIG_DIR / "09c_trend_correlation_matrix.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 10: Headline numbers for the writeup
#
# Compact printout. Each metric's first-season value, last-season value,
# and total change. Paste-ready for the thesis chapter.
#
# **Results:** the three-bullet summary at the top of the notebook is
# this section's deliverable. After running §10, write the three bullets
# up top by selecting from across the Results blocks in §2-§9.

# %%
print("=" * 70)
print("SECTION 10: HEADLINE NUMBERS FOR THE WRITEUP")
print("=" * 70)
print(f"\n{'metric':<26} {'1997-98':>10} {'2025-26':>10} {'change':>10}")
print("-" * 60)
for name, series in trend_series.items():
    vals = series.reindex(seasons_ordered)
    first = vals.iloc[0]
    last = vals.iloc[-1]
    print(f"{name:<26} {first:>10.4f} {last:>10.4f} {last - first:>+10.4f}")

print(
    f"\n{'margin std':<26} {margin_std_by_season.reindex(seasons_ordered).iloc[0]:>10.4f} "
    f"{margin_std_by_season.reindex(seasons_ordered).iloc[-1]:>10.4f} "
    f"{margin_std_by_season.reindex(seasons_ordered).iloc[-1] - margin_std_by_season.reindex(seasons_ordered).iloc[0]:>+10.4f}"
)

print("\nNote: pace is possessions per team-game, NOT per-48. Not directly")
print("comparable to published NBA pace figures. See §7 caveat.")
print("=" * 70)

# %% [markdown]
# ## Section 11: Era-split four-factor regression preview
#
# §5 found ORB% reversed in the recovery era (2021-22 onward) while
# every other metric continued tracking 3PA. Question raised during
# review: did ORB%'s importance in the four-factor framework actually
# change in regression weights, or is the reversal purely descriptive?
#
# Test: fit the canonical four-factor regression separately on the
# decline era (1997-98 through 2020-21) and the recovery era
# (2021-22 through 2025-26). Compare standardized coefficients across
# eras. If ORB%'s weight rose in the recovery era, that supports the
# "ORB became a strategy again" reading. If it stayed flat, the
# reversal is real at the trend level but did not propagate to the
# regression weighting that determines win-pct dependence.
#
# This is a PREVIEW. Full regression methodology (era-specific fits
# vs year fixed effects vs 3PA controls, plus the Dean Oliver
# weighting tests) is Session 13 work. This section is here only
# because the §5 reversal motivated the specific ORB-weight question
# and the test is small enough to fit as a teaser.
#
# Specification: team-season aggregation (mean of each factor across
# the season per team, win count per team-season). OLS regression of
# win_pct on 8 standardized factors (4 offensive, 4 defensive). The
# coefficients answer: per one-standard-deviation increase in this
# factor (within this era), what is the expected change in win_pct?
# Standardization makes coefficients directly comparable across eras
# and across factors of different natural scales.
#
# Sample-size caveat upfront: decline era has ~24 seasons x 30 teams
# = ~720 team-seasons; recovery era has ~5 seasons x 30 teams = ~150
# team-seasons. Recovery era coefficients have substantially wider
# confidence intervals; read direction and rough magnitude, not
# precision.
#
# **Results:**
#
# - R^2 nearly identical across eras: 0.927 (decline) vs 0.939
#   (recovery). The four-factor framework still explains win_pct
#   nearly as well in the modern era. Framework is not broken.
# - **Uniform coefficient shrinkage** is the deepest finding. Every
#   offensive coefficient and every defensive coefficient is smaller
#   in absolute value in the recovery era than in the decline era,
#   typically by 25-45%. Per one-standard-deviation factor difference,
#   modern teams gain noticeably less win_pct than 1997-2020 teams
#   did. The four factors mattered MORE before, not just differently.
#   Combined with R^2 staying nearly equal, this implies win_pct
#   variance has compressed: teams' season records are bunched closer
#   together now, so smaller coefficients fit the data. League is
#   either more competitive or more homogenized in factor variance
#   (or both). This belongs in followups for deeper investigation.
# - ORB% strategy hypothesis: direction confirmed, magnitude modest.
#   ORB% share among the four offensive factors rose from 18.64% to
#   20.03%, +1.39 percentage points. Matches the "ORB became a
#   strategy again" reading but does not reach the 3-5pp threshold
#   that would have been a substantive shift. The level-trend
#   recovery (ORB% rose +0.038 absolute) is much bigger than the
#   regression-weight recovery (+1.4pp share). Two readings remain:
#   (a) early-adopter teams driving the ORB recovery (HOU, SAC, PHX,
#   etc) are getting the win-pct benefit, but the league-wide
#   average weight is moderated by teams that have not shifted, or
#   (b) the strategy is mostly noise: teams crash the glass more for
#   reasons unrelated to winning. Distinguishing these needs per-team
#   analysis (followup #3).
# - FT rate share dropped most among the four offensive factors:
#   9.87% to 8.02%, -1.85 percentage points. Largest share change.
#   Speculative drivers include the 2021-22 NBA rule changes
#   targeting non-basketball foul-drawing moves (which would reduce
#   FT rate as a controllable strategic lever), or denominator
#   composition shifts as 3PA share kept rising. Not tested here.
# - eFG% dominates both eras at 50% share, well above the canonical
#   Dean Oliver 40%. The 4-factor framework consistently underweights
#   eFG% based on the past 29 seasons of regression evidence. FT
#   rate is similarly overweighted in the canonical framework
#   (15% canonical vs 8-10% measured). This is a direct
#   updating-the-framework finding for Session 13's regression
#   chapter.
# - Caveat: 150 team-seasons in the recovery era is thin. Per-pair
#   SE(change) approx 0.004; observed changes in coefficients run
#   0.003 to 0.05, so the larger changes are clearly real but the
#   smaller ones are borderline. Share-percentage differences below
#   ~1pp should be read as noise.
# - Methodological implication for Session 13: era-specific fits
#   reveal substantial coefficient shifts. A pooled 1997-2026
#   regression would mask both the uniform shrinkage and the
#   ORB/FT share shifts. The Session 13 regression spec needs to
#   handle this (era-specific models, year fixed effects, or
#   modernity-proxy controls).

# %%
score_candidates = ["pts", "points", "score", "team_pts"]
opp_score_candidates = ["opp_pts", "opp_points", "opp_score", "opponent_pts"]
margin_candidates = ["margin", "plus_minus", "pm", "team_margin"]

score_col = next((c for c in score_candidates if c in df.columns), None)
opp_score_col = next((c for c in opp_score_candidates if c in df.columns), None)
margin_col = next((c for c in margin_candidates if c in df.columns), None)

if margin_col:
    print(f"Using margin column: {margin_col}")
    df["_won"] = (df[margin_col] > 0).astype(int)
elif score_col and opp_score_col:
    print(f"Using score columns: {score_col} vs {opp_score_col}")
    df["_won"] = (df[score_col] > df[opp_score_col]).astype(int)
else:
    print(f"Available columns: {sorted(df.columns.tolist())}")
    raise KeyError(
        "Need margin or score columns to compute wins. Adjust the candidate lists above."
    )

# %%
all_factor_cols = [*OFF_COLS, "def_efg_pct", "def_tov_pct", "def_orb_pct", "def_ft_rate"]

team_season = (
    df.groupby(["season", "team_abbr"])
    .agg(
        **{col: (col, "mean") for col in all_factor_cols},
        wins=("_won", "sum"),
        games=("_won", "count"),
    )
    .reset_index()
)
team_season["win_pct"] = team_season["wins"] / team_season["games"]

print(f"Team-season rows: {len(team_season)}")
print(f"Mean games per team-season: {team_season['games'].mean():.1f}")
print(f"Win_pct range: [{team_season['win_pct'].min():.3f}, {team_season['win_pct'].max():.3f}]")

# %%
era_decline = team_season[team_season["season"] <= "2020_21"].copy()
era_recovery = team_season[team_season["season"] >= "2021_22"].copy()
print(f"Decline era team-seasons (1997-98 to 2020-21): {len(era_decline)}")
print(f"Recovery era team-seasons (2021-22 to 2025-26): {len(era_recovery)}")


def fit_standardized_ols(data: pd.DataFrame, factor_cols: list, target_col: str) -> tuple:
    """
    Fit OLS with z-scored predictors. Returns standardized coefficients,
    standard errors, and R-squared. Standardization is within the data
    passed in (so each era is standardized against its own scale).
    """
    X = data[factor_cols].values
    y = data[target_col].values
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_z = (X - X_mean) / X_std
    Z = np.column_stack([np.ones(len(X_z)), X_z])
    b, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    y_pred = Z @ b
    residuals = y - y_pred
    n, k = Z.shape
    sigma2 = (residuals**2).sum() / (n - k)
    cov = sigma2 * np.linalg.inv(Z.T @ Z)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (residuals**2).sum() / ((y - y.mean()) ** 2).sum()
    coef = pd.Series(b[1:], index=factor_cols)
    stderr = pd.Series(se[1:], index=factor_cols)
    return coef, stderr, r2


coef_decline, se_decline, r2_decline = fit_standardized_ols(era_decline, all_factor_cols, "win_pct")
coef_recovery, se_recovery, r2_recovery = fit_standardized_ols(
    era_recovery, all_factor_cols, "win_pct"
)

print(f"\nDecline era R^2: {r2_decline:.4f}")
print(f"Recovery era R^2: {r2_recovery:.4f}")

# %%
comparison = pd.DataFrame(
    {
        "decline coef": coef_decline,
        "decline SE": se_decline,
        "recovery coef": coef_recovery,
        "recovery SE": se_recovery,
        "change (recovery - decline)": coef_recovery - coef_decline,
    }
)
print("\nStandardized coefficients (change in win_pct per 1 SD of factor):")
print(comparison.round(4))


def oliver_share(coef_series: pd.Series, off_cols: list) -> pd.Series:
    off_abs = coef_series[off_cols].abs()
    total = off_abs.sum()
    return (off_abs / total * 100).round(2)


share_decline = oliver_share(coef_decline, OFF_COLS)
share_recovery = oliver_share(coef_recovery, OFF_COLS)

share_df = pd.DataFrame(
    {
        "decline era %": share_decline,
        "recovery era %": share_recovery,
        "change pp": share_recovery - share_decline,
        "Dean Oliver canonical %": [40.0, 25.0, 20.0, 15.0],
    },
    index=OFF_COLS,
)

print("\nOffensive factor weight share (% of total |coefficient|), Dean Oliver framing:")
print(share_df.round(2))
print("\nReads: did each factor's relative importance among the four offensive")
print("factors change between eras? Particularly: did ORB%'s share rise?")

# %% [markdown]
# ## Followups parked for later
#
# Not done here, worth doing eventually:
#
# 1. **True per-48 pace.** Requires a minutes-played column in the
#    processed layer. The §7 defensive check will flag it automatically
#    if one gets added. Switch is a one-line change.
# 2. **Playoff time trends.** This notebook is regular season only. A
#    playoff version would show whether the modernization trends hold
#    at the same pace in the postseason or whether playoff basketball
#    lags the regular-season trend.
# 3. **Per-team trend dispersion (the midrange-comeback test).** §3
#    shows the league-wide 3PA rate dipping in 2025-26. The
#    player-driven midrange-comeback hypothesis (`docs/blog_ideas.md`
#    entry #3) predicts this is a few outlier teams, not a uniform
#    shift. Test: per-team-season 3PA rate, across-team standard
#    deviation by season, and identify whether HOU/SAC/PHX dominate
#    the low-3PA tail in recent seasons.
# 4. **Two-point eFG vs three-point eFG decomposition.** `extraEDAfun.md`
#    §10. Decompose eFG% into a 2-point component (FG2M/FG2A) and a
#    3-point component (1.5*FG3M/FG3A) to see whether the eFG% rise is
#    driven by shot-mix shift toward 3s or by shooting itself getting
#    better. Belongs here or in a dedicated shot-mix notebook.
# 5. **Margin variance decomposition.** §8 shows the trend. The
#    follow-up question is whether rising margin variance is driven by
#    more blowouts, fewer close games, or both. Requires looking at the
#    margin distribution shape per season, not just the std.
# 6. **Transition offense vs the ORB U-shape.** §5 found ORB%
#    recovering in recent seasons against the "more threes means less
#    offensive rebounding" story. Open question: does transition
#    offense follow the same shape? Two opposite causal stories
#    (offensive-glass crashing generates transition the other way, vs
#    teams crash because they have gotten better at transition defense
#    and can afford to). Transition data would distinguish them. Not
#    measurable from box-score columns; requires pbpstats play-type
#    tagging. Pairs naturally with the corner-threes and play-type
#    work in blog_ideas.md entry #3.
# 7. **Uniform coefficient shrinkage in the four-factor regression.**
#    §11 found that ALL eight standardized factor coefficients shrunk
#    in absolute magnitude in the recovery era (typically 25-45%
#    drops), while R^2 stayed essentially constant. Combined, this
#    implies win_pct variance has compressed: teams are bunched
#    closer together in season records now. Open questions: (a) is
#    this driven by more competitive parity at the talent level, by
#    factor-variance homogenization across teams, by external sources
#    of variance (load management, lineup volatility, schedule
#    standardization), or some combination? (b) Does the four-factor
#    framework lose explanatory power faster than alternative
#    frameworks (single-axis metrics like net rating, advanced metrics
#    like RAPM)? Session 13 should consider whether to fit
#    region-of-recent-data models with full diagnostics, not just
#    era-specific coefficient comparisons.
