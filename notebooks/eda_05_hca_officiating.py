# %% [markdown]
# # EDA 05: Home Court Advantage Descriptive + Officiating (29-season)
#
# **Session 12, post-EDA 04.** The thesis-centerpiece notebook. HCA is
# the headline empirical finding of the regression chapter, and EDA 05
# is the descriptive groundwork that makes the Session 13 regression
# results readable.
#
# EDA 02 §5 touched HCA in the dual-axis HCA-decline-vs-margin-std-rise
# chart. EDA 04 §8 isolated the margin-std half cleanly. EDA 05 isolates
# the HCA half cleanly and goes deeper: HCA decomposed by competitive
# context and by month, the 2020-21 no-crowds natural experiment, the
# crowd-restoration trajectory, and the officiating FT-rate-gap
# contribution. All descriptive; conditional HCA (regression intercept
# after factors absorbed) is Session 13.4's `hca.py` deliverable.
#
# Both regular-season and playoff data are loaded. HCA shows up
# differently in playoffs (smaller samples, higher stakes, different
# travel patterns), and a thesis chapter on HCA without that
# comparison would be incomplete. Sections run regular-season as the
# primary track with playoff lines overlaid or noted in parallel.
#
# Scope cut: officiating analysis is FT-rate gap only. Without a
# personal-fouls column in the processed data, foul disparity cannot
# be measured directly. With the advanced ingest (parked, see
# `docs/blog_ideas.md`), this would expand to TS-pct gap, off-rating
# gap, and def-rating gap as separable HCA mechanism candidates.
#
# **Three-bullet summary:**
#
# 1. **The regular-season HCA decline is a multi-phase structural
#    story, not a COVID story.** Across 29 seasons, HCA fell from
#    +3.24 pts (pre-modernization baseline, 1997-2013) to +1.73 pts
#    (2025-26), a -1.51 pt total decline that decomposes into three
#    structural components: -0.68 pts during the analytics revolution
#    (2013-2020, 45% of total), -0.60 pts as a post-bubble shortfall
#    vs the analytics-era baseline (40% of total), and an additional
#    -0.23 pts of within-post-bubble drift through 2025-26 (15% of
#    total). COVID disruption was -1.62 pts at the Bubble floor but
#    mostly reverted with crowd restoration. The persistent
#    post-bubble shortfall indicates HCA decline is ongoing, not a
#    temporary COVID effect.
# 2. **Playoff HCA is robust to all of this.** Across the same 29
#    seasons, playoff HCA stayed essentially unchanged: +4.42 pre-
#    modernization, +4.23 in the analytics era, +4.40 post-bubble.
#    The 2020-21 Bubble dropped to +3.34 (small sample, 85 games),
#    then snapped back. Meanwhile regular-season HCA dropped 39%
#    (3.24 to 1.96). The reg-vs-playoff gap GREW from +1.18 pts pre-
#    modernization to +2.44 pts post-bubble. Whatever mechanisms
#    drive the regular-season decline (officiating shifts, charter
#    flights, sports science, 3PA-revolution effects on foul-drawing)
#    don't operate in high-stakes playoff games. Caveat: ~1 pt of
#    playoff HCA is structural false-positive from the higher seed
#    hosting more games per series, so true playoff HCA is closer to
#    ~3-3.4 pts, but the era-stability of that level is the more
#    interesting finding than the absolute number.
# 3. **Officiating (FT-rate gap proxy) accounts for ~24% of season-
#    level HCA variance** (r = +0.485). The home FT-rate advantage
#    exists in every single one of 29 seasons (pooled +0.0106, or
#    3.72% above road) and has roughly halved (first 3 seasons mean
#    +0.0102, last 3 seasons mean +0.0055). The 2020-21 spike during
#    no-crowds (+0.0112) suggests refereeing baseline is only
#    partially crowd-driven. Officiating is a meaningful but partial
#    mechanism; structural factors beyond officiating account for
#    the other ~75% of HCA variance. The Bubble natural experiment
#    suggests crowds alone account for ~63% of analytics-era HCA,
#    leaving ~37% irreducible to non-crowd factors (travel,
#    refereeing baseline, home-team familiarity).

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from thesis_style import apply_thesis_style

from nba_four_factors.analysis import load_processed, pivot_to_game_level
from nba_four_factors.config import SeasonType

sns.set_theme(style="whitegrid", context="notebook")

apply_thesis_style()
FIG_DIR = Path("figures/eda_05")
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

# %% [markdown]
# ## Section 1: Data load with is_neutral filter
#
# Two loads: regular-season and playoffs. Each filtered to remove
# neutral-flagged games (NBA Cup games from 2023-24 onward, Mexico City
# / London / Paris one-offs, and the 2020-21 Bubble games that were
# officially designated neutral). Without this filter, neutral games
# would dilute HCA toward zero by including games where home-court
# advantage definitionally doesn't apply.
#
# The 2020-21 regular season is a special case worth noting up front:
# games WERE played in home arenas (just with limited/no crowds), so
# they are marked is_home=True and is_neutral=False. The filter
# preserves them. They function as the natural-experiment data: home-
# designated games without home crowds, which §6 analyzes directly.

# %%
df_reg = load_processed(("1997_98", "2025_26"), SeasonType.REGULAR)
df_po = load_processed(("1997_98", "2025_26"), SeasonType.PLAYOFFS)

print(f"Regular season long-format shape: {df_reg.shape}")
print(f"Playoffs long-format shape: {df_po.shape}")

neutral_reg = df_reg["is_neutral"].sum()
neutral_po = df_po["is_neutral"].sum()
print("\nNeutral-flagged team-game rows (will be filtered out):")
print(f"  Regular season: {neutral_reg} ({100 * neutral_reg / len(df_reg):.2f}%)")
print(f"  Playoffs: {neutral_po} ({100 * neutral_po / len(df_po):.2f}%)")

df_reg = df_reg[~df_reg["is_neutral"]].copy()
df_po = df_po[~df_po["is_neutral"]].copy()
print("\nAfter filter:")
print(f"  Regular season: {df_reg.shape}")
print(f"  Playoffs: {df_po.shape}")

seasons_ordered = sorted(df_reg["season"].unique())
seasons_display = [s.replace("_", "-") for s in seasons_ordered]
x = np.arange(len(seasons_ordered))


def add_era_markers(ax, annotate: bool = False) -> None:
    """
    Draw standard era marker vertical lines on a season-indexed axis.
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


# %% [markdown]
# ## Section 2: Per-season raw HCA in margin points
#
# The headline chart. Mean home margin by season, plotted with era
# markers. Both regular season and playoffs.
#
# Pivot to game-level first since margin is a per-game quantity, not a
# per-team-game one (the long-format has each game twice).
#
# Prediction: regular-season HCA roughly 3.0 pts pre-COVID, drops to
# ~0.9 in 2020-21 (already known from EDA 04 §8), partial recovery to
# ~1.7 by 2025-26. Playoff HCA noisier (smaller sample per season) but
# conventionally similar in magnitude or slightly higher. The thesis
# point: HCA in 2025-26 is still about half its pre-COVID value
# despite full crowds being back since 2022-23.
#
# **Results:**
#
# - Regular-season HCA range across 29 seasons: 0.944 (2020-21, Bubble
#   floor) to 3.884 (2002-03). Most seasons sit in the 2.5-3.5 range
#   through 2012-13, then drift to 2.1-2.7 through 2019-20, suggesting
#   an inflection around 2012-13 (analyzed in §7b). 2025-26 is +1.727
#   pts, still well below pre-modernization values.
# - Playoff HCA noisier as predicted (60-87 games per playoff bracket
#   per season vs ~1,200 regular-season games). Range: 2.063 (2016-17)
#   to 8.081 (2007-08). 2019-20 and 2025-26 are NaN because the
#   is_neutral filter removed the Orlando Bubble playoffs entirely and
#   2025-26 playoffs hadn't started at data pull.
# - Playoff HCA averages higher than regular-season HCA in nearly
#   every season. Pre-COVID mean: 4.05 pts (vs 3.06 reg season).
#   Post-COVID mean: 4.19 pts (vs 1.99 reg season). The playoff
#   advantage has grown post-COVID because regular-season HCA fell
#   while playoff HCA held steady. §7c quantifies this further.
# - Methodological caveat applies to all playoff numbers: about 1 pt
#   of "playoff HCA" is structural false-positive from the higher
#   seed hosting more games per series (4 of 7 in a typical 7-game
#   series). True playoff HCA is roughly 1 pt lower than reported.
#   Session 13.4's hca.py will quantify this via regression
#   conditioning on team quality.
# - No season had negative HCA. Even at the 2020-21 Bubble floor,
#   home teams still averaged +0.944 pts. HCA is structurally
#   resilient, and the value at the Bubble floor functions as the
#   approximate "irreducible HCA when crowds are removed but other
#   home-court factors remain."

# %%
games_reg = pivot_to_game_level(df_reg)
games_po = pivot_to_game_level(df_po)
print(f"Regular season game-level shape: {games_reg.shape}")
print(f"Playoff game-level shape: {games_po.shape}")

# %%
hca_reg = games_reg.groupby("season")["home_margin"].mean().reindex(seasons_ordered)
hca_po = games_po.groupby("season")["home_margin"].mean().reindex(seasons_ordered)

print("Per-season mean home margin (regular season):")
print(hca_reg.round(3))
print("\nPer-season mean home margin (playoffs):")
print(hca_po.round(3))

# %%
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(
    x,
    hca_reg.values,
    marker="o",
    markersize=7,
    linewidth=2.0,
    color="steelblue",
    label="Regular season",
)
ax.plot(
    x,
    hca_po.values,
    marker="s",
    markersize=6,
    linewidth=1.5,
    color="firebrick",
    alpha=0.7,
    label="Playoffs",
)
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
add_era_markers(ax, annotate=True)
ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("Mean home margin (pts)")
ax.set_title(
    "Home Court Advantage by season: regular season vs playoffs\n1997-98 to 2025-26, neutral games excluded"
)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_hca_by_season.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 3: Per-season HCA in home win rate terms
#
# The same story expressed as P(home win). Useful for thesis writeup
# because win rate is more intuitive to general readers than margin
# points.
#
# Prediction: regular-season home win rate ~0.58 pre-COVID, drops to
# ~0.50-0.51 in 2020-21, recovers to ~0.54 by 2025-26. Playoff home
# win rate conventionally similar or slightly higher.
#
# **Results:**
#
# - Regular-season home win rate hovered around 0.59-0.62 through
#   2012-13, then drifted to 0.57-0.59 through 2019-20, dropped to
#   0.544 in 2020-21 (Bubble), and recovered partially to 0.554 in
#   2025-26. The pattern mirrors the margin-based HCA story closely.
# - Notably, the home win rate never dropped to the no-HCA baseline
#   of 0.500. Even in the no-crowds 2020-21 Bubble, home teams won
#   54.4% of games. The "irreducible HCA when crowds are removed"
#   shows up here as ~4 percentage points above pure chance.
# - Playoff home win rate ranges widely (0.557 to 0.744) due to small
#   sample. Pre-COVID mean ~0.64; post-COVID mean ~0.58. The win-rate
#   data hints at a small playoff HCA reduction post-COVID that the
#   margin-based view didn't show. May be sample-size noise; the
#   margin-based playoff HCA actually rose slightly. Both quantities
#   tell consistent stories within their measurement noise.

# %%
games_reg["home_win"] = (games_reg["home_margin"] > 0).astype(int)
games_po["home_win"] = (games_po["home_margin"] > 0).astype(int)

home_winrate_reg = games_reg.groupby("season")["home_win"].mean().reindex(seasons_ordered)
home_winrate_po = games_po.groupby("season")["home_win"].mean().reindex(seasons_ordered)

print("Per-season home win rate (regular season):")
print(home_winrate_reg.round(4))
print("\nPer-season home win rate (playoffs):")
print(home_winrate_po.round(4))

# %%
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(
    x,
    home_winrate_reg.values,
    marker="o",
    markersize=7,
    linewidth=2.0,
    color="steelblue",
    label="Regular season",
)
ax.plot(
    x,
    home_winrate_po.values,
    marker="s",
    markersize=6,
    linewidth=1.5,
    color="firebrick",
    alpha=0.7,
    label="Playoffs",
)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="No HCA baseline")
add_era_markers(ax, annotate=True)
ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("Home win rate")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.set_title(
    "Home win rate by season: regular season vs playoffs\n1997-98 to 2025-26, neutral games excluded"
)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_home_winrate_by_season.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 4: HCA decomposed by competitive context
#
# Is HCA primarily a close-game effect, or does it scale with blowout
# magnitude? Conventional wisdom says HCA matters most in close games
# where crowd energy and travel fatigue tip outcomes; blowouts are
# decided by talent gaps that swamp HCA. The data will say.
#
# Close games defined as |final margin| <= 10. Threshold chosen because
# it's the standard NBA "close game" definition; 5 is too tight (small
# sample after filter), 15 would include too many drift-toward-blowout
# games.
#
# **Results:**
#
# - HCA is overwhelmingly a close-game phenomenon. Pooled across 29
#   seasons, regular-season HCA in close games (|margin| <= 10) is
#   +0.670 pts; in blowouts (|margin| > 10) it is +5.508 pts. At
#   first glance the blowout number looks larger, but interpret the
#   sign carefully: blowouts are by definition large margins, and
#   when the home team is the winner of the blowout, the margin is
#   large and positive. The blowout HCA reflects the asymmetry of
#   "home teams winning blowouts" vs "road teams winning blowouts."
# - Close-game share is 56.6% in regular season and 54.0% in
#   playoffs, so most games are close games and HCA's main effect
#   on outcomes comes through this bucket.
# - Playoff close-game HCA is +0.867 pts (slightly higher than reg
#   season +0.670), consistent with the broader finding that playoff
#   HCA is more robust.
# - The post-COVID drop in HCA shows up in BOTH close-game and
#   blowout buckets. Close-game HCA dropped from ~0.7-1.0 pts pre-2013
#   to ~0.5-0.6 pts in recent seasons; blowout HCA dropped from
#   ~5-8 pts pre-COVID to ~3-4 pts post-COVID. The decline is broad,
#   not localized to one competitive context.
# - 2025-26 close-game HCA is +0.169 pts and blowout HCA is +3.160
#   pts, both meaningfully below historical norms. Consistent with
#   the post-bubble drift hypothesis from §7b.

# %%
close_threshold = 10
games_reg["is_close"] = games_reg["home_margin"].abs() <= close_threshold
games_po["is_close"] = games_po["home_margin"].abs() <= close_threshold

print(f"Close-game share (|margin| <= {close_threshold}):")
print(f"  Regular season: {games_reg['is_close'].mean():.3f}")
print(f"  Playoffs: {games_po['is_close'].mean():.3f}")

# %%
hca_close_reg = (
    games_reg[games_reg["is_close"]]
    .groupby("season")["home_margin"]
    .mean()
    .reindex(seasons_ordered)
)
hca_blowout_reg = (
    games_reg[~games_reg["is_close"]]
    .groupby("season")["home_margin"]
    .mean()
    .reindex(seasons_ordered)
)

print("\nMean home margin by competitive context (regular season):")
print(
    pd.DataFrame(
        {
            "close (|m|<=10)": hca_close_reg,
            "blowout (|m|>10)": hca_blowout_reg,
        }
    ).round(3)
)

print("\nPooled across 29 seasons:")
print(
    f"  Close-game HCA (regular season): {games_reg[games_reg['is_close']]['home_margin'].mean():+.3f}"
)
print(
    f"  Blowout HCA (regular season):    {games_reg[~games_reg['is_close']]['home_margin'].mean():+.3f}"
)
print(
    f"  Close-game HCA (playoffs):       {games_po[games_po['is_close']]['home_margin'].mean():+.3f}"
)
print(
    f"  Blowout HCA (playoffs):          {games_po[~games_po['is_close']]['home_margin'].mean():+.3f}"
)

# %%
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(
    x,
    hca_close_reg.values,
    marker="o",
    markersize=6,
    linewidth=1.8,
    color="steelblue",
    label="Close games (|margin| <= 10)",
)
ax.plot(
    x,
    hca_blowout_reg.values,
    marker="s",
    markersize=6,
    linewidth=1.8,
    color="firebrick",
    label="Blowouts (|margin| > 10)",
)
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
add_era_markers(ax)
ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("Mean home margin (pts)")
ax.set_title(
    "Home Court Advantage by competitive context (regular season only)\nClose vs blowout games, 1997-98 to 2025-26"
)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_hca_by_competitive_context.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 5: HCA by month of season
#
# Travel fatigue accumulates through the season; HCA could be higher in
# late season. Also seasonal effects (Christmas Day games are heavily
# marketed and home teams might be hyped; January back-to-backs are
# brutal for road teams).
#
# Calendar-month grouping is the simplest. Different seasons start in
# different months, so "first month of season" is conflated across
# schedules with different start dates. Calendar month accepts this
# blurring; the alternative (weeks since season start) is cleaner but
# substantially more code. Calendar month suffices for a descriptive
# view.
#
# Results follow after the chart prints, since the small-sample
# annotation depends on the printout.

# %%
games_reg["game_date"] = pd.to_datetime(games_reg["game_date"])
games_reg["calendar_month"] = games_reg["game_date"].dt.month

month_names = {
    10: "Oct",
    11: "Nov",
    12: "Dec",
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
}

hca_by_month = games_reg.groupby("calendar_month")["home_margin"].agg(["mean", "std", "count"])
hca_by_month = hca_by_month.reindex([10, 11, 12, 1, 2, 3, 4, 5, 6]).dropna()
print("HCA by calendar month (regular season pooled across 29 seasons):")
print(hca_by_month.round(3))

normal_total = hca_by_month["count"].sum()
small_threshold = normal_total / len(hca_by_month) * 0.2
small_sample_months = hca_by_month[hca_by_month["count"] < small_threshold].index.tolist()
print(
    f"\nMonths with anomalously small game counts (<20% of mean bucket size): {[month_names[m] for m in small_sample_months]}"
)
print("These months are dominated by lockout/COVID seasons (which had compressed schedules)")
print("and should not be interpreted as meaningful late-season HCA signals.")

# %% [markdown]
# **Results:**
#
# - Calendar-month HCA does NOT show the travel-fatigue pattern
#   predicted. If road fatigue accumulated through the season, HCA
#   should rise toward April. Instead, late-season HCA (April +2.58)
#   is similar to early-season (Oct +2.30) and mid-season (Jan
#   +2.87, Feb +2.57, Mar +2.93).
# - November is the highest at +3.04 pts. Possible explanations:
#   teams still settling rotations (more home-team consistency
#   advantage over road teams figuring things out), more back-to-
#   backs early in season, or just noise.
# - May has only 179 games vs 5,000+ for normal in-season months
#   (and 3,617 for April). Almost all May games come from the
#   2020-21 compressed COVID schedule. The May value of +1.83
#   reflects the low-HCA Bubble era, not a "late-season HCA"
#   effect. Flagged in gray on the chart with the small-sample
#   warning.
# - The travel-fatigue hypothesis at calendar-month resolution
#   isn't supported. A finer-grained test would look at
#   back-to-back games specifically or day-of-week patterns
#   (followup #4), where travel-fatigue mechanisms would show up
#   more cleanly than calendar month does.

# %%
fig, ax = plt.subplots(figsize=(11, 6))
month_labels = [month_names[m] for m in hca_by_month.index]
colors = ["lightgray" if m in small_sample_months else "steelblue" for m in hca_by_month.index]
bars = ax.bar(month_labels, hca_by_month["mean"].values, color=colors, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(
    games_reg["home_margin"].mean(),
    color="firebrick",
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    label=f"29-season mean ({games_reg['home_margin'].mean():.2f})",
)

for bar, count, month_idx in zip(
    bars, hca_by_month["count"].values, hca_by_month.index, strict=False
):
    label_color = "darkred" if month_idx in small_sample_months else "black"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        -0.35,
        f"n={int(count):,}",
        ha="center",
        va="top",
        fontsize=9,
        color=label_color,
    )

ax.set_ylabel("Mean home margin (pts)")
ax.set_title(
    "HCA by calendar month (regular season pooled, 1997-98 to 2025-26)\nGray bars = months dominated by anomalous seasons; treat with caution"
)
ax.legend(loc="best")
ax.set_ylim(bottom=ax.get_ylim()[0] - 0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_hca_by_month.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 6: The 2020-21 no-crowds natural experiment
#
# This is the cleanest test of "crowds cause HCA" available in the data.
# The 2020-21 regular season was played with empty or near-empty arenas
# through most of the season, but games were still played in the home
# team's arena (is_home=True, is_neutral=False). That's the natural-
# experiment data: home-designated games without home crowds.
#
# Prediction: if crowds drive HCA, 2020-21 HCA should be near zero.
# Actual value from §2: mean home margin in 2020-21 = ~0.94 pts, which
# is well below the pre-COVID ~3 pts norm but not zero. Reading:
# crowds explain most of HCA but not all of it. The ~30% residual is
# attributable to other home-court factors (travel, sleep, familiarity,
# refereeing baseline).
#
# This section quantifies the comparison cleanly and frames it for the
# thesis chapter.
#
# **Results:**
#
# - 2020-21 (Bubble floor) HCA: +0.944 pts vs pre-modernization
#   baseline of +3.06 pts (1997-98 to 2018-19 mean, the wide-window
#   comparison) or vs analytics-era baseline of +2.55 pts (the
#   narrower 2013-2020 comparison that controls for the modernization
#   decline).
# - Using the analytics-era as the appropriate baseline (since the
#   modernization decline was already underway pre-COVID), the 2020-21
#   Bubble HCA represents a -1.6 pt drop from baseline, of which:
#   * Implied crowd contribution: -1.6 pts (about 63% of analytics-
#     era HCA explained by crowds when they're removed)
#   * Non-crowd residual: +0.944 pts (about 37% of HCA still present
#     in the no-crowds condition)
# - Note: the wide-window pre-COVID comparison (pre-2018-19 baseline
#   of +3.06 pts) yields a different implied crowd contribution
#   (~69%). The choice of baseline matters: it depends on whether
#   you're asking "how much HCA did crowds explain in the early-NBA
#   era" or "how much HCA did crowds explain in the immediate
#   pre-COVID era." The analytics-era baseline is more apples-to-apples
#   for current discussions because the analytics-era HCA was already
#   reduced by other modernization factors.
# - The non-crowd residual (~0.94 pts) is what's irreducible to
#   crowd presence: travel fatigue, refereeing baseline beyond crowd
#   influence, home-team familiarity with sightlines/rims, the
#   accumulated familiarity advantages of playing in one's own
#   facility. This is roughly the "structural" or "non-crowd" HCA.

# %%
pre_covid_seasons = [s for s in seasons_ordered if s <= "2018_19"]
post_covid_seasons = [s for s in seasons_ordered if s >= "2022_23"]

pre_covid_hca = games_reg[games_reg["season"].isin(pre_covid_seasons)]["home_margin"].mean()
covid_2020_hca = games_reg[games_reg["season"] == "2020_21"]["home_margin"].mean()
post_covid_hca = games_reg[games_reg["season"].isin(post_covid_seasons)]["home_margin"].mean()

n_pre = len(games_reg[games_reg["season"].isin(pre_covid_seasons)])
n_2020 = len(games_reg[games_reg["season"] == "2020_21"])
n_post = len(games_reg[games_reg["season"].isin(post_covid_seasons)])

print(f"Pre-COVID (1997-98 to 2018-19, {len(pre_covid_seasons)} seasons, {n_pre} games):")
print(f"  Mean HCA: {pre_covid_hca:+.3f} pts")
print(f"\n2020-21 (the no-crowds natural experiment, {n_2020} games):")
print(f"  Mean HCA: {covid_2020_hca:+.3f} pts")
print(f"  Ratio to pre-COVID: {covid_2020_hca / pre_covid_hca * 100:.1f}%")
print(
    f"  Residual not explained by crowds: {covid_2020_hca:+.3f} pts ({covid_2020_hca / pre_covid_hca * 100:.1f}% of pre-COVID baseline)"
)
print(
    f"  Implied crowd contribution: {pre_covid_hca - covid_2020_hca:+.3f} pts ({(1 - covid_2020_hca / pre_covid_hca) * 100:.1f}% of pre-COVID HCA)"
)
print(
    f"\nPost-COVID full-crowds era (2022-23 to 2025-26, {len(post_covid_seasons)} seasons, {n_post} games):"
)
print(f"  Mean HCA: {post_covid_hca:+.3f} pts")
print(f"  Ratio to pre-COVID: {post_covid_hca / pre_covid_hca * 100:.1f}%")

# %% [markdown]
# ## Section 7: The crowd-restoration trajectory
#
# 2020-21 (no crowds) into 2021-22 (partial) into 2022-23 (full crowds
# back) and onward to 2025-26. Conventional prediction: HCA recovers
# proportionally as crowds return. Actual trajectory needs inspection.
#
# Prediction (from EDA 04 numbers): HCA bounced from 0.94 in 2020-21
# to ~1.7 in 2025-26, but did NOT fully recover to the pre-COVID ~3.0
# baseline. Recovery is partial. Possible explanations:
#
# 1. Refereeing patterns changed during the bubble/empty-arenas era
#    and did not snap back to pre-COVID patterns.
# 2. Travel-side-of-HCA shrunk with charter flight standardization
#    and improved sports-science practices that reduce road fatigue.
# 3. Sample-period noise; 3 years post-restoration may not be enough
#    to see a fully steady-state HCA.
#
# This section produces the chart that goes in the thesis chapter.
#
# **Results:**
#
# - The crowd-restoration trajectory: 2019-20 (last pre-COVID, +2.17),
#   2020-21 (Bubble, +0.94), 2021-22 (partial crowds, +1.72), 2022-23
#   (full crowds back, +2.50), 2023-24 (+2.15), 2024-25 (+1.69),
#   2025-26 (+1.73).
# - Recovery is partial and non-monotonic. The 2022-23 peak of +2.50
#   represents the high point of restoration but never reaches the
#   pre-modernization baseline of +3.24 nor the analytics-era +2.55.
# - From 2022-23 onward, HCA drifted DOWN despite continuously full
#   crowds. By 2024-25 and 2025-26 the values (+1.69 and +1.73) are
#   meaningfully below the 2022-23 restoration peak. The continued
#   drift suggests structural factors are pulling HCA down
#   independent of the crowd-presence variable. This is §7b's
#   "post-bubble drift" finding.
# - 2025-26 is 43.6% below the wide pre-COVID baseline (+3.06) and
#   32.2% below the analytics-era baseline (+2.55). The shortfall
#   is real either way.

# %%
recovery_window = ["2019_20", "2020_21", "2021_22", "2022_23", "2023_24", "2024_25", "2025_26"]
recovery_data = hca_reg.loc[recovery_window]
print("HCA trajectory through crowd restoration:")
for s in recovery_window:
    season_display = s.replace("_", "-")
    print(f"  {season_display}: {recovery_data[s]:+.3f} pts")

print(f"\nPre-COVID baseline (1997-98 to 2018-19 mean): {pre_covid_hca:+.3f}")
print(
    f"Recovery shortfall (2025-26 vs pre-COVID): {hca_reg['2025_26'] - pre_covid_hca:+.3f} pts ({(1 - hca_reg['2025_26'] / pre_covid_hca) * 100:.1f}% below baseline)"
)

# %%
fig, ax = plt.subplots(figsize=(12, 6))
recovery_x = np.arange(len(recovery_window))
recovery_labels = [s.replace("_", "-") for s in recovery_window]
ax.plot(
    recovery_x, recovery_data.values, marker="o", markersize=10, linewidth=2.5, color="steelblue"
)
ax.axhline(
    pre_covid_hca,
    color="firebrick",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8,
    label=f"Pre-COVID baseline ({pre_covid_hca:+.2f})",
)
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="No HCA")

ax.annotate(
    "Bubble era\n(no crowds)",
    xy=(1, recovery_data["2020_21"]),
    xytext=(0.5, recovery_data["2020_21"] - 0.5),
    fontsize=10,
    ha="center",
    color="darkorange",
)
ax.annotate(
    "Full crowds\nback",
    xy=(3, recovery_data["2022_23"]),
    xytext=(3, recovery_data["2022_23"] + 0.5),
    fontsize=10,
    ha="center",
    color="seagreen",
)

ax.set_xticks(recovery_x)
ax.set_xticklabels(recovery_labels, fontsize=11)
ax.set_ylabel("Mean home margin (pts)")
ax.set_title(
    "HCA crowd-restoration trajectory (regular season)\nThe partial recovery: HCA returned but not to pre-COVID levels"
)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_hca_crowd_restoration.png", dpi=150)
plt.show()

# %% [markdown]
# ### 7b: Four-era HCA decomposition
#
# The recovery chart shows the COVID-era trajectory, but discussion
# during review surfaced a deeper inflection: HCA appears to have
# started declining around 2012-13, coincident with the 3PA rate
# acceleration. The pre-COVID HCA baseline isn't a single value
# across 22 seasons; it's an old-NBA value followed by an analytics-
# era decline. A simple "pre-COVID vs Bubble vs post" split obscures
# this by averaging across two distinct pre-COVID phases.
#
# Four-era decomposition:
#
# 1. **Pre-modernization (1997-98 to 2012-13):** Stable HCA, the "old
#    NBA" before the 3PA revolution accelerated.
# 2. **Analytics revolution / 3PA spike (2013-14 to 2019-20):** HCA
#    already declining alongside the 3PA-rate acceleration that began
#    2012-13. Most of the pre-COVID HCA decline lives here.
# 3. **Bubble floor (2020-21):** No-crowds natural experiment.
# 4. **Post-bubble (2021-22 to 2025-26):** Crowds restored, HCA partly
#    recovered but did not return to analytics-era values, let alone
#    pre-modernization. Within this era there's a within-bucket pattern
#    worth flagging: initial restoration peak (2021-22 to 2022-23,
#    +2.11 pts) followed by continued drift (2023-24 onward), even
#    with crowds at full capacity.

# %%
era_buckets = {
    "Pre-modernization (1997-98 to 2012-13)": [s for s in seasons_ordered if s <= "2012_13"],
    "Analytics revolution / 3PA spike (2013-14 to 2019-20)": [
        s for s in seasons_ordered if "2013_14" <= s <= "2019_20"
    ],
    "Bubble floor (2020-21)": ["2020_21"],
    "Post-bubble (2021-22 to 2025-26)": [s for s in seasons_ordered if s >= "2021_22"],
}

print("Four-era HCA decomposition:")
print("-" * 70)
era_means = {}
for label, season_list in era_buckets.items():
    era_games = games_reg[games_reg["season"].isin(season_list)]
    mean_hca = era_games["home_margin"].mean()
    n_games = len(era_games)
    n_seasons = len(season_list)
    era_means[label] = mean_hca
    print(f"  {label}")
    print(f"    Mean HCA: {mean_hca:+.3f} pts (n = {n_games:,} games, {n_seasons} seasons)")
print("-" * 70)

pre_mod_hca = era_means["Pre-modernization (1997-98 to 2012-13)"]
analytics_era_hca = era_means["Analytics revolution / 3PA spike (2013-14 to 2019-20)"]
post_bubble_hca = era_means["Post-bubble (2021-22 to 2025-26)"]

print("\nDecomposition of total HCA decline:")
print(f"  Old-NBA baseline (pre-modernization):  {pre_mod_hca:+.3f} pts")
print(f"  Most recent (2025-26):                 {hca_reg['2025_26']:+.3f} pts")
print(f"  Total decline:                         {hca_reg['2025_26'] - pre_mod_hca:+.3f} pts")
print()
print("Component breakdown:")
print(f"  Analytics-era decline:                 {analytics_era_hca - pre_mod_hca:+.3f} pts")
print(f"    (Pre-mod {pre_mod_hca:.2f} to analytics era {analytics_era_hca:.2f})")
print(
    f"    {(analytics_era_hca - pre_mod_hca) / (hca_reg['2025_26'] - pre_mod_hca) * 100:.1f}% of total decline"
)
print()
print(
    f"  COVID disruption (mostly reverted):    {covid_2020_hca - analytics_era_hca:+.3f} pts at peak"
)
print("    Most of the COVID shock was eroded by crowd restoration")
print()
print(f"  Post-bubble structural shortfall:      {post_bubble_hca - analytics_era_hca:+.3f} pts")
print(f"    (Analytics era {analytics_era_hca:.2f} to post-bubble {post_bubble_hca:.2f})")
print(
    f"    {(post_bubble_hca - analytics_era_hca) / (hca_reg['2025_26'] - pre_mod_hca) * 100:.1f}% of total decline"
)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
era_labels = list(era_buckets.keys())
era_values = [era_means[label] for label in era_labels]
era_colors = ["steelblue", "cornflowerblue", "darkorange", "firebrick"]
bars = ax.bar(range(len(era_labels)), era_values, color=era_colors, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(
    pre_mod_hca,
    color="steelblue",
    linestyle=":",
    linewidth=1,
    alpha=0.5,
    label=f"Pre-modernization baseline ({pre_mod_hca:+.2f})",
)

for bar, value in zip(bars, era_values, strict=False):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.05,
        f"{value:+.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

short_labels = [
    "Pre-mod\n(1997-2013)",
    "Analytics era\n(2013-2020)",
    "Bubble floor\n(2020-21)",
    "Post-bubble\n(2021-2026)",
]
ax.set_xticks(range(len(era_labels)))
ax.set_xticklabels(short_labels, fontsize=10)
ax.set_ylabel("Mean home margin (pts)")
ax.set_title(
    "Four-era HCA decomposition\nMost of the HCA decline pre-dates COVID; restoration didn't return HCA to analytics-era values"
)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "05b_hca_four_era_decomposition.png", dpi=150)
plt.show()

# %% [markdown]
# ### 7b-supplement: Within-post-bubble restoration vs drift
#
# The post-bubble bucket averages 5 seasons (2021-22 through 2025-26),
# but those 5 seasons aren't homogeneous. Restoration peak (2021-22,
# 2022-23) is the high; drift (2023-24, 2024-25, 2025-26) is the low.
# Worth quantifying as a sub-finding because it answers "does HCA
# continue falling after crowds restored" without making it its own
# top-level era.

# %%
restoration_seasons = ["2021_22", "2022_23"]
drift_seasons = ["2023_24", "2024_25", "2025_26"]

restoration_games = games_reg[games_reg["season"].isin(restoration_seasons)]
drift_games = games_reg[games_reg["season"].isin(drift_seasons)]

restoration_mean = restoration_games["home_margin"].mean()
drift_mean = drift_games["home_margin"].mean()

print("Within post-bubble breakdown:")
print(
    f"  Restoration peak (2021-22 to 2022-23, {len(restoration_games):,} games): {restoration_mean:+.3f} pts"
)
print(f"  Drift            (2023-24 to 2025-26, {len(drift_games):,} games): {drift_mean:+.3f} pts")
print(f"  Continued decline within post-bubble: {drift_mean - restoration_mean:+.3f} pts")
print("  Crowds were at full capacity throughout this drift.")

# %% [markdown]
# ### 7c: Regular season vs playoff HCA, era-by-era
#
# The regular-season HCA story is clear: structural decline starting
# 2013, COVID amplification, partial restoration, continued drift.
# Now check whether playoffs show the same pattern. Conventional
# expectation: playoffs amplify HCA, so all eras should show higher
# playoff HCA than regular season, but the trajectory should match
# (declining over time if modernization affects both contexts).
#
# Alternative hypothesis: playoff HCA is robust to whatever drives
# the regular-season decline. Playoff intensity, stakes, and lower-
# stakes-context-irrelevance (charter flights matter less when both
# teams are charter, sports science matters less in high-arousal
# states) might mean modernization doesn't reduce playoff HCA.

# %%
playoff_era_means = {}
print("Playoff HCA by era (excluding NaN seasons with no qualifying playoff games):")
print("-" * 70)
for label, season_list in era_buckets.items():
    era_games_po = games_po[games_po["season"].isin(season_list)]
    if len(era_games_po) > 0:
        mean_hca_po = era_games_po["home_margin"].mean()
        n_games_po = len(era_games_po)
        playoff_era_means[label] = mean_hca_po
        print(f"  {label}")
        print(f"    Mean playoff HCA: {mean_hca_po:+.3f} pts (n = {n_games_po:,} games)")
    else:
        print(f"  {label}: no playoff games in this era")
print("-" * 70)

print("\nComparison: regular season vs playoff HCA per era")
print(f"{'Era':<50} {'Reg':>8} {'Playoff':>9} {'Gap':>8}")
print("-" * 80)
for label in era_buckets:
    reg_val = era_means[label]
    po_val = playoff_era_means.get(label, float("nan"))
    gap = po_val - reg_val if not np.isnan(po_val) else float("nan")
    print(f"{label:<50} {reg_val:>+8.3f} {po_val:>+9.3f} {gap:>+8.3f}")

print()
print("Caveat: playoff HCA includes a structural ~1 pt false-positive bias from")
print("the higher seed hosting more games per series (e.g., 4-of-7 in a 7-game")
print("series). Subtracting that, 'true' playoff HCA is probably ~1 pt lower")
print("than reported. The era-to-era TREND in playoff HCA is still informative")
print("regardless of the offset.")

# %% [markdown]
# ## Section 8: Officiating descriptive - home vs road FT rate
#
# Pooled across all team-games, what's the average FT rate when a team
# plays at home vs on the road? Same teams, different venues. The home
# FT advantage is the difference.
#
# The mechanism candidate: if home teams shoot more free throws at home
# than on the road, that's consistent with home-court refereeing bias
# (or with home teams playing differently when at home in ways that
# draw more fouls). Either way, it's a measurable contribution to HCA.
#
# Scope cut: this is FT-rate gap analysis only. With advanced data
# (parked, see `docs/blog_ideas.md`), this section would expand to TS-
# pct gap (which adds shooting efficiency to the foul-drawing story),
# off-rating gap (per-possession scoring advantage at home), and def-
# rating gap (per-possession defending advantage at home). The four
# components together would decompose HCA into separable mechanism
# candidates. With traditional data alone, only FT-rate gap is
# directly measurable.
#
# **Results:**
#
# - Pooled across 29 regular seasons, home FT rate is 0.2955 vs road
#   0.2849, a gap of +0.0106 absolute (or +3.72% relative). Real and
#   consistent home-team advantage in FT-drawing.
# - Pooled across 29 playoff seasons, the gap is larger: home 0.3258
#   vs road 0.3035, gap +0.0224 absolute (+7.37% relative). Roughly
#   double the regular-season gap. Could reflect:
#   * Officials are more home-biased in higher-stakes playoff games
#   * Home teams attack the basket more in playoffs (drawing more
#     fouls structurally, not via ref bias)
#   * The structural seeding-host effect (better teams play more home
#     games AND draw more fouls because they're better)
#   * Some combination
# - This section is purely descriptive. The trend in this gap over
#   time (§9) and its correlation with HCA decline (§9b) provide the
#   suggestive evidence for officiating's role.

# %%
home_ft_pooled = df_reg[df_reg["is_home"]]["off_ft_rate"].mean()
road_ft_pooled = df_reg[~df_reg["is_home"]]["off_ft_rate"].mean()
gap_pooled = home_ft_pooled - road_ft_pooled

print("Pooled across 29 seasons, regular season, neutral excluded:")
print(f"  Home FT rate (FTA/FGA): {home_ft_pooled:.4f}")
print(f"  Road FT rate (FTA/FGA): {road_ft_pooled:.4f}")
print(f"  Home-road gap:          {gap_pooled:+.4f}")
print(f"  Relative gap: {gap_pooled / road_ft_pooled * 100:+.2f}% above road")

home_ft_po = df_po[df_po["is_home"]]["off_ft_rate"].mean()
road_ft_po = df_po[~df_po["is_home"]]["off_ft_rate"].mean()
gap_po = home_ft_po - road_ft_po

print("\nPlayoffs (pooled across 29 seasons):")
print(f"  Home FT rate: {home_ft_po:.4f}")
print(f"  Road FT rate: {road_ft_po:.4f}")
print(f"  Home-road gap: {gap_po:+.4f}")
print(f"  Relative gap: {gap_po / road_ft_po * 100:+.2f}% above road")

# %% [markdown]
# ## Section 9: FT-rate gap trend over time
#
# Has the home FT advantage shrunk across 29 seasons? If yes, and the
# shrinkage aligns temporally with the HCA decline from §2, that's
# evidence (but NOT proof) that some of HCA's decline runs through
# officiating.
#
# Methodologically, this is the same observational-equivalence problem
# we hit in EDA 03 §3's cross-block mechanism discrimination. Two
# series that decline together over 29 years correlate strongly just
# because both are trending. The temporal coincidence is suggestive but
# not causal. A rigorous test would need ref-crew data (parked, see
# `docs/blog_ideas.md` entry #1).
#
# What this section can establish:
# - Whether the FT-rate gap exists at all (yes, from §8 pooled).
# - Whether it's stable or shrinking over time.
# - If shrinking, whether the shrinkage tracks the HCA decline.
#
# **Results:**
#
# - The FT-rate gap is positive in every single one of 29 seasons.
#   Home teams have a consistent FT-drawing advantage that has never
#   disappeared, even in 2020-21 (no crowds; gap was actually +0.0112,
#   slightly above the 29-season mean of +0.0106).
# - The gap has gradually shrunk: first 3 seasons (1997-98 to 1999-00)
#   mean +0.0102, last 3 seasons (2023-24 to 2025-26) mean +0.0055.
#   Roughly halved over 29 seasons.
# - The 2020-21 spike in the gap (+0.0112 during the no-crowds season)
#   is counterintuitive: if crowd influence drives the home FT
#   advantage, removing crowds should shrink the gap, not leave it
#   stable. This is consistent with refereeing baseline (independent
#   of crowd intimidation) being a substantial part of the FT-rate
#   gap, or with home teams playing more aggressively (driving more)
#   when there's no crowd-pressure effect.
# - Season-level correlation between HCA and FT-rate gap is
#   r = +0.4854, r^2 = 0.236. The FT-rate gap accounts for about 24%
#   of HCA's season-to-season variance.
# - Subject to the monotone-trend confound (both series decline, so
#   their correlation is partially driven by shared trend). But
#   r = 0.49 is moderate enough that the correlation isn't purely
#   shared-trend; there's real co-variation. The "staggering
#   agreement" observation from the §9b z-scored overlay is borne
#   out: the two declines do track each other across 29 seasons.
# - Methodological framing for the thesis: officiating (proxied by
#   FT-rate gap) is a meaningful but partial driver of HCA decline.
#   Other contributors must account for ~75% of HCA variance.
#   Charter flights, sports science, the 3PA revolution's effect on
#   foul-drawing strategy, and refereeing patterns beyond FT-rate
#   are all plausible candidates that this descriptive analysis
#   cannot distinguish.

# %%
home_ft_by_season = (
    df_reg[df_reg["is_home"]].groupby("season")["off_ft_rate"].mean().reindex(seasons_ordered)
)
road_ft_by_season = (
    df_reg[~df_reg["is_home"]].groupby("season")["off_ft_rate"].mean().reindex(seasons_ordered)
)
ft_gap_by_season = home_ft_by_season - road_ft_by_season

ft_gap_summary = pd.DataFrame(
    {
        "home_ft": home_ft_by_season,
        "road_ft": road_ft_by_season,
        "gap": ft_gap_by_season,
    }
)
print("Home vs road FT rate by season:")
print(ft_gap_summary.round(4))

# %%
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(x, ft_gap_by_season.values, marker="o", markersize=7, linewidth=2.0, color="darkgreen")
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(
    ft_gap_by_season.mean(),
    color="firebrick",
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    label=f"29-season mean ({ft_gap_by_season.mean():+.4f})",
)
add_era_markers(ax)
ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("Home FT rate - Road FT rate")
ax.set_title(
    "FT rate gap (home minus road) by season\nThe officiating-mediated component of HCA, 1997-98 to 2025-26"
)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_ft_gap_by_season.png", dpi=150)
plt.show()

# %% [markdown]
# ### 9b: HCA vs FT-rate gap on the same axis
#
# If the FT-rate gap has been shrinking and HCA has been shrinking,
# overlaying them (z-scored, same x-axis) tells us whether the
# trajectories are consistent with officiating being a partial cause
# of the HCA decline.

# %%
hca_z = (hca_reg - hca_reg.mean()) / hca_reg.std()
ft_gap_z = (ft_gap_by_season - ft_gap_by_season.mean()) / ft_gap_by_season.std()

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(
    x,
    hca_z.values,
    marker="o",
    markersize=6,
    linewidth=1.8,
    color="steelblue",
    label="HCA (regular season, z-scored)",
)
ax.plot(
    x,
    ft_gap_z.values,
    marker="s",
    markersize=6,
    linewidth=1.8,
    color="darkgreen",
    label="FT-rate gap (z-scored)",
)
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
add_era_markers(ax)
ax.set_xticks(x)
ax.set_xticklabels(seasons_display, rotation=70, ha="right", fontsize=9)
ax.set_ylabel("Z-score")
ax.set_title("HCA and FT-rate gap z-scored on one axis\nDo the two declines track each other?")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_hca_and_ft_gap_zscored.png", dpi=150)
plt.show()

r_hca_ft_gap = hca_reg.corr(ft_gap_by_season)
print(f"\nSeason-level correlation, HCA vs FT-rate gap (29 seasons): r = {r_hca_ft_gap:+.4f}")
print("Caveat: same monotone-trend confound as EDA 04 §9c. Two declining series")
print("will correlate strongly just from shared trend; this is suggestive of")
print("officiating's role in HCA decline, not proof of it.")

# %% [markdown]
# ## Section 10: Headline numbers for the writeup
#
# Compact print summary. HCA values at the key reference points, plus
# the home FT advantage trend. Paste-ready for the thesis chapter.
#
# **Results:** the three-bullet summary at the top of the notebook is
# this section's deliverable. After running §10, write the three
# bullets up top by selecting from across the Results blocks in §2-§9.

# %%
print("=" * 70)
print("SECTION 10: HEADLINE NUMBERS FOR THE WRITEUP")
print("=" * 70)

print("\n--- Reference HCA values (regular season, four-era framework) ---")
print(f"  Pre-modernization baseline (1997-98 to 2012-13):  {pre_mod_hca:+.3f} pts")
print(f"  Analytics era (2013-14 to 2019-20):               {analytics_era_hca:+.3f} pts")
print(f"  Bubble floor (2020-21):                           {covid_2020_hca:+.3f} pts")
print(f"  Post-bubble (2021-22 to 2025-26):                 {post_bubble_hca:+.3f} pts")
print(f"  2025-26 (most recent):                            {hca_reg['2025_26']:+.3f} pts")

print(
    f"\n--- Decline decomposition (total: {hca_reg['2025_26'] - pre_mod_hca:+.3f} pts from pre-mod baseline) ---"
)
analytics_decline = analytics_era_hca - pre_mod_hca
post_bubble_shortfall = post_bubble_hca - analytics_era_hca
total_decline = hca_reg["2025_26"] - pre_mod_hca
print(
    f"  Analytics-era decline (pre-COVID):                {analytics_decline:+.3f} pts ({analytics_decline / total_decline * 100:.1f}% of total)"
)
print(
    f"  Post-bubble shortfall vs analytics era:           {post_bubble_shortfall:+.3f} pts ({post_bubble_shortfall / total_decline * 100:.1f}% of total)"
)
print(
    f"  COVID disruption (Bubble peak): {covid_2020_hca - analytics_era_hca:+.3f} pts but mostly reverted by restoration"
)

print("\n--- HCA in home win rate terms ---")
pre_mod_winrate = home_winrate_reg.loc[[s for s in seasons_ordered if s <= "2012_13"]].mean()
analytics_winrate = home_winrate_reg.loc[
    [s for s in seasons_ordered if "2013_14" <= s <= "2019_20"]
].mean()
post_bubble_winrate = home_winrate_reg.loc[[s for s in seasons_ordered if s >= "2021_22"]].mean()
print(f"  Pre-modernization baseline mean:  {pre_mod_winrate:.4f}")
print(f"  Analytics era mean:               {analytics_winrate:.4f}")
print(f"  2020-21 (Bubble):                 {home_winrate_reg['2020_21']:.4f}")
print(f"  Post-bubble mean:                 {post_bubble_winrate:.4f}")
print(f"  2025-26 (most recent):            {home_winrate_reg['2025_26']:.4f}")

print("\n--- Crowd contribution implied by Bubble natural experiment ---")
print(f"  Analytics-era HCA (the appropriate baseline):  {analytics_era_hca:+.3f} pts")
print(f"  2020-21 Bubble HCA (no crowds):                {covid_2020_hca:+.3f} pts")
print(
    f"  Implied crowd contribution:                    {analytics_era_hca - covid_2020_hca:+.3f} pts ({(1 - covid_2020_hca / analytics_era_hca) * 100:.1f}% of analytics-era HCA)"
)
print(
    f"  Non-crowd residual at Bubble floor:            {covid_2020_hca:+.3f} pts ({covid_2020_hca / analytics_era_hca * 100:.1f}% of analytics-era HCA)"
)

print("\n--- Within post-bubble: restoration vs drift ---")
print(
    f"  Restoration (2021-22 to 2022-23): {restoration_mean:+.3f} pts ({len(restoration_games):,} games)"
)
print(f"  Drift (2023-24 to 2025-26):       {drift_mean:+.3f} pts ({len(drift_games):,} games)")
print(
    f"  Continued decline post-restoration: {drift_mean - restoration_mean:+.3f} pts (crowds remained full)"
)

print("\n--- Playoff vs regular season HCA (pooled) ---")
print(f"  Regular season HCA (pooled):       {games_reg['home_margin'].mean():+.3f} pts")
print(f"  Playoff HCA (pooled):              {games_po['home_margin'].mean():+.3f} pts")
print("  Caveat: ~1 pt of playoff HCA is structural false-positive from")
print("  the higher seed hosting more games per series.")

print("\n--- Close vs blowout HCA (pooled, regular season) ---")
print(
    f"  Close-game HCA (|margin|<=10): {games_reg[games_reg['is_close']]['home_margin'].mean():+.3f}"
)
print(
    f"  Blowout HCA (|margin|>10):     {games_reg[~games_reg['is_close']]['home_margin'].mean():+.3f}"
)

print("\n--- FT-rate gap (officiating descriptive) ---")
print(f"  Pooled home-road FT-rate gap: {gap_pooled:+.4f}")
print(f"  First 3 seasons mean: {ft_gap_by_season.head(3).mean():+.4f}")
print(f"  Last 3 seasons mean:  {ft_gap_by_season.tail(3).mean():+.4f}")
print(f"  Total drift: {ft_gap_by_season.tail(3).mean() - ft_gap_by_season.head(3).mean():+.4f}")
print(f"  Season-level correlation with HCA: r = {r_hca_ft_gap:+.4f}")
print(f"  r^2 = {r_hca_ft_gap ** 2:.3f}, so FT-rate gap accounts for")
print(f"  ~{r_hca_ft_gap ** 2 * 100:.0f}% of season-level HCA variance.")

print("=" * 70)

# %% [markdown]
# ## Followups parked for later
#
# Not done here, worth doing eventually:
#
# 1. **Conditional HCA via regression.** Session 13.4's `hca.py`
#    deliverable. Compare raw HCA (this notebook's §2) to HCA after
#    the four factors are controlled for. If conditional HCA is much
#    smaller than raw HCA, factors are absorbing most of HCA; if
#    conditional HCA tracks raw HCA, factors don't explain HCA and
#    the HCA story has to be about something outside the framework.
# 2. **Per-team HCA dispersion.** Some teams have HCA = 5+ pts, others
#    have HCA near 0. The league-average is hiding heterogeneity.
#    Per-team-season HCA, ranked, would show the spread. Pairs with
#    the altitude and timezone followups below.
# 3. **Altitude and timezone effects.** Denver and Utah are famously
#    hard road games (altitude); cross-coast travel with 3-hour shifts
#    is brutal. Approximate from team_abbr until Session 12-13 travel
#    features land. Could be its own substantive section in the thesis.
# 4. **Day-of-week HCA.** Back-to-back away games are killers; HCA
#    might be higher when the away team is in their second-of-a-back-
#    to-back. Computable from game_date with some date arithmetic.
# 5. **Playoff HCA decomposition.** §2 plots regular-season and
#    playoff HCA on the same chart. Sample size limits per-season
#    interpretation in playoffs (~85 games per playoff bracket), but
#    pooled across eras the playoff HCA is testable against the
#    regular-season HCA. Open question: is playoff HCA larger or
#    smaller than regular-season HCA, and how has the gap moved?
# 6. **Officiating crew fixed effects on HCA.** The right way to test
#    "do refs favor home teams" is to include ref crew as a fixed
#    effect in a regression. Needs ref-crew data scraping from NBA
#    officiating reports. Pairs with `docs/blog_ideas.md` entry #1.
# 7. **HCA decomposition via advanced data (parked, all four pieces).**
#    With advanced ingest (`docs/blog_ideas.md` "Studies parked
#    pending advanced data ingest"), §8 expands from FT-rate gap alone
#    to FT-rate gap + TS-pct gap + off-rating gap + def-rating gap.
#    The four-component decomposition separates HCA into offensive
#    home advantage, defensive home advantage, and a foul-drawing
#    component within offense. Highest-leverage advanced study.
# 8. **2019-20 Bubble playoffs as a second natural experiment.** The
#    2019-20 playoffs were played entirely in Orlando with all games
#    designated neutral, so they're filtered out of this notebook's
#    main analysis. But they constitute a second no-crowds dataset
#    distinct from 2020-21's regular-season home-marked games. A
#    sidebar comparison would strengthen the §6 finding by checking
#    whether the implied crowd contribution is consistent across both
#    natural experiments.
