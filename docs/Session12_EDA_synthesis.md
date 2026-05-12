# Session 12 EDA Synthesis

Status: **End of weekend, May 11. EDA 01 and EDA 02 complete on 29-season range. Sections 3+ of `extraEDAfun.md` pending. Section 13 spec ready but blocked pending EDA completion.**

This document consolidates findings from completed EDA work and serves as the handoff for the remaining EDA sections from `extraEDAfun.md`. Methodology notes from the weekend are captured below.

## Where we are

29 seasons of clean processed data (1997-98 to 2025-26 regular season). All three V3 box-score endpoints at 100% completeness. 68,716+ team-game rows. Bubble anomaly fix in place. Two of the planned EDA notebooks complete.

The current plan is to walk through `extraEDAfun.md` section by section, deciding whether to execute or skip each. Then start Session 13 (regression layer per `Session13.md`).

## Completed EDA notebooks

* `notebooks/eda_01_hygiene.py` — Data hygiene
* `notebooks/eda_02_distributions.py` — Marginal distributions

Figures saved to `notebooks/figures/eda_01/` and `notebooks/figures/eda_02/` (both gitignored).

---

## EDA 01: Data hygiene (DONE on 29 seasons)

### Sections covered (mapped to `extraEDAfun.md` Section 1)

1.1 Per-season row counts
1.2 Missing values
1.3 Score sanity
1.4 Team count per season (with 29-team era caveat for pre-2004_05)
1.5 Schedule symmetry
1.6 Neutral-site flag distribution
1.7 Bubble fix verification

### Findings

1. **All 29 seasons present with expected row counts.** Pre-2004_05 lands at 2378 team-game rows (29 teams x 82 games x 2). 2004_05+ lands at 2460 (30 teams). Lockouts (1998_99 at 725 games, 2011_12 at 990) and COVID (2019_20 at 1059, 2020_21 at 1080) match expected schedule lengths. 2025_26 regular season completed April 12, 2026 with full 1230 games (the "in-progress" caveat applies only to playoffs).

2. **No null values in any column across all seasons.** The processing pipeline drops canceled games (filter in `_loading.py`) and produces clean numeric columns throughout.

3. **Margin variance has risen substantially.** Single-game margin std ranges from 12.03 (2003_04 minimum) to 16.50 (2025_26 maximum). Pre-2011 era stable around 13. Post-2020 acceleration is steep.

4. **2019_20 home/away counts unreliable for per-team analysis** due to Bubble: Bubble teams played 11+ more away games than home games in the 2019_20 totals.

5. **Neutral games breakdown across the modern era** is clean:
   - 2019_20: 88 Bubble seeding games
   - 2024_25 onward: 5 neutral games per year = 2 NBA Cup semifinals + 1 Mexico City + 2 Paris Games (NBA Cup final excluded from regular-season endpoint by league rule)

6. **Bubble fix verified.** All 176 post-shutdown 2019_20 team-game rows correctly flagged `is_neutral=True`, `is_home=False`.

### Methodology footnotes

* Pre-2004_05 used 29-team league. Charlotte Bobcats expansion was 2004_05.
* 2012_13 has 1229 games (one less than 1230) due to Boston Marathon bombing cancellation, canceled-game filter correctly drops the 0-0 row.
* NBA Cup final excluded from `leaguegamelog` regular season endpoint because it doesn't count toward standings. Worth a methodology footnote in the thesis chapter.

---

## EDA 02: Marginal distributions (DONE on 29 seasons)

### Sections covered (mapped to `extraEDAfun.md` Section 2)

2.1 Offensive four factors per-season summary stats
2.2 Per-season violin distributions for offensive factors
2.3 Defensive factors mirror check
2.4 Margin distribution + Q-Q plot (pooled)
2.5 Per-season margin std (with era markers)
Plus: 2b FTM/FGA comparison, 2c FTA>FGA frequency, 5b dual-axis HCA vs std

### Findings

1. **eFG% rose 8 percentage points across 29 seasons.** From 0.479 (1997-98) to 0.547 (2025-26). Acceleration concentrated 2014-15 to 2019-20 (Warriors dynasty era).

2. **ORB% has a U-shape, not monotonic decline.** Long decline 1997-2021 (0.286 to 0.220), clear reversal 2022-2026 (back to 0.258, near 2010-11 levels). This is one of the most interesting findings, possibly driven by tactical shifts post-Thunder/Pacers Finals.

3. **TOV% gradual decline.** Smallest movement (~14.5% to ~12.5%), no era inflection.

4. **FT rate (FTA/FGA) fell 27% across the era.** From 0.337 to 0.246. Slight uptick to 0.267 in 2025_26.

5. **FTA/FGA and FTM/FGA correlate at 0.94** across all 29 seasons (stable correlation). The 6% of independent variance in FTM/FGA represents team FT shooting skill. Decision: use FTA/FGA per Oliver's original framework. Document choice in methodology section.

6. **FT rate is bounded in practice.** Across 68,716 team-game rows, FTA > FGA occurred exactly once: November 19, 1999, Lakers vs Bulls (Hack-a-Shaq game). FT rate is unbounded in theory but effectively capped below 1.0 in real games.

7. **Defensive mirror check verified.** Max |mean(off_X) - mean(def_X)| per season is 0 for eFG%, TOV%, FT rate, and 5.55e-17 for ORB% (float64 epsilon). Symmetric construction confirmed.

8. **Pooled mean home margin (29 seasons) = 2.77.** Single-number HCA estimate. Cleaner than canonical "3-point" figure. Bimodal margin distribution with peaks ~±5-9 confirmed (end-of-game foul mechanics + no-ties rule).

9. **Q-Q plot shows approximate normality with mild fat tails** confined to ~±4 sigma. OLS standard errors approximately valid but slightly underestimate prediction-interval coverage at extremes. Document caveat in methodology section.

10. **Headline visualization: dual-axis HCA mean vs std by season.** "X shape" emerging post-2010. HCA declining (3.4 to 1.7) while std rising (12.5 to 16.4). Signal shrinking, noise rising. This is the centerpiece chart for the thesis introduction.

### Era inflections visible across all factor charts

Era markers used consistently: 1998-99 lockout, 2011-12 lockout, 2019-20 COVID shutdown, 2020-21 empty arenas, 2022-23 take-foul rule introduction, 2025-26 in-progress.

Three distinct eras emerge: pre-2011 (stable on most metrics), 2011-2018 (gradual modernization drift), 2019-onward (sharp acceleration on multiple dimensions).

---

## Remaining EDA from `extraEDAfun.md`

Each section below is flagged with a recommendation. Decide per-section at kickoff.

### Section 3: Cross-factor relationships — RECOMMEND DO

Correlation matrices for offensive factors, defensive factors, and off-vs-def within-game. The famous eFG% vs ORB% negative correlation (can't grab an offensive rebound on a make) is worth verifying in your data.

Effort: 60-90 min. New file: `notebooks/eda_03_cross_factor_relationships.py`.

Worth doing because the four-factor framework assumes the factors are roughly independent. If they're heavily correlated, that's a methodology limitation worth flagging.

### Section 4: Time trends — RECOMMEND DO

League-wide trends across 29 seasons: eFG%, 3PA rate (the bigger modernization story), TOV%, ORB%, FT rate, margin variance, pace.

Effort: 90-120 min. New file: `notebooks/eda_04_time_trends.py`.

Some overlap with EDA 02 §2.1 (per-season factor means) but with critical additions:
- 3-point attempt rate (FG3A / FGA) — the actual three-point revolution metric
- Pace computation (possessions per 48 minutes)
- These are not in the four factors and would be missed otherwise

Worth doing as the modernization-story chart with everything in one place.

### Section 5: HCA descriptive — RECOMMEND DO

The thesis-centerpiece descriptive analysis. Three sub-analyses:
- Per-season raw HCA with era markers (29-season version of the chart we built for 13 seasons on Saturday)
- Home win rate per season (same shape expressed as probability)
- HCA decomposed by close-vs-blowout, using **win rate not margin** (methodological correction from Sunday's discussion)
- HCA by month (travel-fatigue hypothesis)
- 2020-21 Bubble baseline as natural experiment

Effort: 90-120 min. New file: `notebooks/eda_05_hca_descriptive.py`.

This is genuinely needed before regression. The conditional-HCA chart from Session 13.4 will complement this, not replace it. Both belong in the thesis chapter.

### Section 6: Playoff vs regular season — RECOMMEND DEFER

The 29-season playoff dataset is in place but the comparison can wait until after Session 13. Reason: the regression machinery in Session 13.1 will make playoff comparisons more rigorous than descriptive averaging would.

Possible exception: a quick "are playoff four-factor distributions different from regular season?" check is cheap to do as part of EDA 04. Could add as a section in eda_04 if curious.

### Section 7: "Is the home crowd back?" — RECOMMEND DEFER TO SESSION 13.4

Conditional HCA (regression intercept after factors absorbed) by season is exactly what Session 13.4's `hca.py` and chart produces. Don't do it twice descriptively first.

### Section 8: Officiating's contribution to HCA — RECOMMEND DO (as part of EDA 05)

Home vs road FT rate. Foul disparity. The "FT rate gap" trend. These are descriptive analyses that complement Section 5's HCA work. Doing both together in one notebook makes sense.

Effort: adds ~30-60 min to EDA 05.

### Section 9: Dynastic teams — RECOMMEND SKIP for EDA, REVISIT FOR THESIS NARRATIVE

Top-5 and bottom-5 team-seasons by net rating with factor profiles. Compelling for thesis narrative but not analytically necessary for the regression work. Save for thesis-chapter writing phase.

### Section 10: 3-point revolution decomposition — RECOMMEND DO (as Day 4 sidequest)

Decompose eFG% rise into "more 3s" vs "better 3-point shooting" vs "better 2-point shooting." The hypothesis (most of the rise comes from mix shift, not skill) is testable from the box-score columns. If true, it's a real thesis-chapter aside.

Effort: 60-90 min. Could be a separate notebook `eda_06_three_point_decomposition.py` or a section in EDA 04.

### Section 11: Game-context HCA — RECOMMEND SKIP or PICK ONE

Month, day-of-week, altitude, time-zone. Interesting but rabbit-hole-prone. Pick at most one, probably altitude (Denver/Utah natural experiment, simple to compute). Save the rest for thesis-chapter aside writing.

### Section 12: Factor importance stability — DEFER, IT'S SESSION 13.2

This is literally the per-season coefficient stability visualization specified in Session 13.2. Don't do it descriptively first.

### Section 13: Matchup-level prediction — DEFER, IT'S SESSION 13.2 AND 13.4

The cross-validation harness in Session 13.2 covers this.

### Section 14: "Four factors are not enough" diagnostic — DEFER TO POST-SESSION 13

Residual analysis after regression is fit. Can't do this descriptively; needs the fitted model.

### Section 15: 58/42 stress test — DEFER, IT'S SESSION 13.3

The team-season regression in Session 13.3 produces this.

---

## Recommended remaining EDA path

Three new notebooks plus an optional fourth:

1. **EDA 03: Cross-factor relationships** (60-90 min)
2. **EDA 04: Time trends + maybe pace + playoffs comparison aside** (90-120 min)
3. **EDA 05: HCA descriptive + officiating's HCA contribution** (120-180 min)
4. **EDA 06: 3-point revolution decomposition** (60-90 min, optional)

Total: 4.5-8 hours depending on optional inclusions.

Decide per-section at kickoff. If energy fades, can stop after EDA 05 and have a complete-enough descriptive picture for Session 13 to proceed.

---

## Methodology notes locked from this weekend's discussions

These are decisions made during EDA 01 / EDA 02 that should carry forward consistently.

1. **OREB% formula**: Use Oliver's original `OREB / (OREB + opp_DREB)`, the "share of available offensive rebounds" formulation. Not OREB/Poss. The processed layer matches this convention. Document choice in methodology section.

2. **FT rate formula**: Use FTA/FGA (Oliver's original). FTM/FGA correlates at 0.94 with FTA/FGA and the 6% independent variance is "FT shooting skill," which is conceptually distinct from the "aggressiveness" the four-factor framework wants to capture.

3. **Standard deviation vs variance**: Report std (in points, comparable to mean margin) not variance (squared units). Loose usage of "variance" in conversation has been corrected; the thesis chapter uses "standard deviation" or "std" precisely.

4. **Era handling**: Include lockout and COVID seasons with asterisk annotations on charts rather than excluding them. Era markers consistent across charts: 1998-99 lockout, 2011-12 lockout, 2019-20 COVID shutdown, 2020-21 empty arenas, 2022-23 take-foul rule, 2025-26 in-progress.

5. **2025-26 handling**: Regular season completed April 12, 2026. Treat as a normal complete season for regular-season EDA. The "in-progress" caveat applies only to playoffs.

6. **Bubble fix carries forward**: `NEUTRAL_SITE_DATE_OVERRIDES` in `anomalies.py` flags 2019-20 post-shutdown games correctly. Verified across reprocessing.

7. **Cup final exclusion**: NBA Cup final is excluded from `leaguegamelog` regular-season endpoint by league rule (doesn't count toward standings). Cup semifinals and group-stage games DO appear and are correctly flagged as neutral when applicable. Document in methodology section.

8. **Section 4 (close-game HCA) rewrite**: The original "HCA in close vs blowout games using mean margin" decomposition is mathematically truncated (the close-game filter caps the dependent variable). Replace with **home win rate in close games**, which is unbounded by the margin filter. To be applied in EDA 05.

9. **Chart conventions**:
   - Y-axis percentages use `PercentFormatter` not raw decimals
   - Y-axis rates (FT rate, OREB%) use decimals since they're not bounded percentages
   - Season labels use hyphens (1997-98) in display, underscores (1997_98) in code
   - Era markers always include text annotations (not just vertical lines)

10. **Aggregate vs per-season metrics**: When reporting comparisons, mean of per-season values (each season weighted equally) is preferred over pooled-across-games. Each season is one observation; generalization-across-seasons is the right framing.

---

## Pending action items (for next session)

### Recommended order

1. **Walk through EDA 03 / 04 / 05** per section above. Decide go/skip on each at kickoff.
2. **Update this synthesis doc** with findings from each completed EDA section.
3. **Commit EDA work** as separate commits per notebook.
4. **Start Session 13.1** using `Session13.md` as the spec.

### Cleanup candidates (low priority)

* Bump `DEFAULT_TIMEOUT_SECONDS` in `nba_four_factors/api/client.py` from 30 to 60 seconds. Root cause of the historical backfill saga.
* Include exception class/message in the "game fetch failed" WARNING log line.
* Promote `scripts/fill_gaps.py` patterns into orchestration as a proper `backfill --gap-fill-only` mode.

### Deferred to Session 13 or later

* Conditional HCA chart (Session 13.4)
* Coefficient stability visualization (Session 13.2)
* Cross-validation across A/B/C strategies (Session 13.2)
* All of Sections 7, 11, 12, 13, 14, 15 from `extraEDAfun.md`
* Day 4 sidequest options not chosen (3-point revolution decomposition if not done as EDA 06)
* Day 5 synthesis writeup for thesis chapter

---

## Files added this weekend

* `nba_four_factors/analysis/pivots.py`
* `tests/analysis/test_pivots.py`
* `nba_four_factors/analysis/__init__.py` (updated)
* `nba_four_factors/anomalies.py` (NEUTRAL_SITE_DATE_OVERRIDES added)
* `nba_four_factors/processed/schedule.py` (anomaly override hook in `_tidy_schedule`)
* `tests/processed/test_schedule.py` (Bubble fix tests)
* `known_anomalies.md` (Bubble fix documented + historical backfill gaps documented)
* `notebooks/eda_01_hygiene.py`
* `notebooks/eda_02_distributions.py`
* `scripts/check_missing_games.py`
* `scripts/fill_gaps.py`
* `data/processed/*` (15 historical seasons newly processed + 2019_20 reprocessed + 2025_26)
* `data/raw/*` (15 historical seasons fetched, gaps closed)
* `docs/Session13.md` (regression layer spec, 13.1 detailed)
* `docs/Session12_EDA_synthesis.md` (this document)

## Final state

Dataset: 29 complete seasons, 100% coverage across all three V3 endpoints.
EDA: 01 and 02 complete with substantive findings.
Spec: Session 13 plan ready, blocked pending Section 3+ EDA completion.
Cleanup: Documented for future sessions.

Next sit-down: open this document and `extraEDAfun.md` side by side, walk through Sections 3 onward.
