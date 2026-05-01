# extraEDAfun.md

A structured EDA plan for the four-factors thesis, with the boring-but-important
stuff first and the genuinely interesting "what if" explorations second. The
goal is to know the data cold before any regression gets fit, and to surface
findings the formal model might miss.

We have 10 seasons (2015_16 through 2024_25) of clean processed Parquets.
~24,000 team-game rows for regular season, ~1,700 for playoffs. The four
factors are computed both for offense and defense, plus margin and home flag.

## Part I: The grown-up EDA

Things you'd do before any regression in any analysis project. Skip none of
these. They're the foundation.

### 1. Data hygiene

* **Per-season row counts.** Confirm 2019_20 is short (~970 games × 2 teams),
  2020_21 is also short (~1,080 × 2). Plot them as a bar chart of
  games-per-season. Any weird outlier seasons? COVID two-fers should be the
  only oddities.
* **Missing values.** None expected since the processing pipeline drops bad
  rows, but verify. `df.isna().sum()` per column.
* **Score sanity.** Min and max margin per season. Anything above 60 is suspect
  but possible (the famous Sacramento 64-point loss). Anything below 1 or
  exactly 0 means data error (NBA games can't tie).
* **Team count per season.** 30 every season, no exceptions.
* **Schedule symmetry.** Each team should have roughly equal home and away
  game counts. If 2020_21 has any weird bubble-related distribution, surface it.
* **Neutral-site flag distribution.** How many neutral games per season?
  Should be ~5/season for NBA Cup (started 2023_24), 0 before, occasional
  Mexico City / London games scattered earlier. The 2020_21 Bubble was
  technically all neutral but the ingest pipeline may or may not have flagged
  them. **Worth checking, this affects every HCA computation downstream.**

### 2. Marginal distributions

* **Each of the 4 offensive factors, histogram + summary stats per season.**
  eFG% should center around 0.50-0.55, drifting up over the 10 years.
  TOV% around 0.13-0.14. ORB% around 0.22-0.28 (declining as 3-point shooting
  rises). FT rate around 0.20-0.25.
* **Same for defensive (which is opponent's offensive).** The 8 columns are
  paired: each team's defensive eFG% is the opponent's offensive eFG%, by
  construction. They should have the same league-wide distribution.
* **Margin distribution.** Roughly normal, centered slightly above 0 (HCA),
  std dev around 13-14. Q-Q plot vs normal — looks for fat tails.

### 3. Cross-factor relationships

* **Correlation matrix of the 4 offensive factors.** They're not independent.
  Famously: eFG% and ORB% are negatively correlated (you can't grab an
  offensive rebound on a make). TOV% mostly independent. FT rate mostly
  independent.
* **Same for defensive factors.** Similar pattern.
* **Off vs def correlations.** Within a single team-game, your offense and
  defense are anticorrelated through pace and randomness. Worth a heatmap.

### 4. Time trends

* **League-wide eFG% by season.** The headline NBA modernization story.
  Should rise from ~0.50 in 2015_16 to ~0.55+ in 2024_25. **Plot this.**
* **3-point attempt rate (FG3A / FGA) by season.** Even more dramatic rise.
* **TOV% by season.** Slow decline.
* **ORB% by season.** Steady decline (corollary of more 3-pointers).
* **FT rate by season.** Some year-to-year fluctuation, no strong trend.
* **Margin variance by season.** Has the league become more or less
  competitive? Look at season-level std of game margin.
* **Pace.** Computed from the box score columns (FGA + 0.44*FTA + TOV - OREB
  is a rough possession estimate). Pace has risen since 2015. Quantify.

### 5. The home court advantage descriptive

This is the headline finding of the thesis. Need to know it cold before
the regression confirms it.

* **Per-season raw HCA.** Mean home margin per season, plot as a line.
  Should show ~3.0 pre-COVID, drop to ~1.5-1.8 post-COVID, with a possible
  recovery in 2023_24 / 2024_25.
* **Home win rate per season.** Same shape, expressed as a probability.
  Should be ~0.58 pre-COVID, ~0.54 post-COVID.
* **HCA decomposed.** Within each season:
  - HCA in non-blowout games (margin within 10) — is the effect concentrated
    in close games?
  - HCA by month — does it strengthen later in the season as travel fatigue
    accumulates?
* **2020_21 Bubble baseline.** All games were neutral. If the Bubble HCA is
  ~0 (as expected) and we still see pre-COVID HCA of ~3, that's the
  cleanest possible isolation of "real" HCA. Check this. It's the
  natural experiment.

### 6. Playoff vs regular season comparison

* **Same 4-factor distributions.** Playoffs are typically ~2-3 percentage
  points lower on eFG% (better defenses), slightly lower TOV%, similar
  ORB%, similar FT rate. Confirm.
* **HCA in playoffs vs regular season.** Conventional wisdom: playoff HCA
  is bigger (rabid crowd, higher stakes). Empirically, often the gap is
  smaller than people think. Check.
* **Playoff sample size warnings.** ~85 games per playoff bracket means
  any per-season effect has wide confidence intervals. Pre-register this
  before running comparisons so we don't over-interpret noise.

## Part II: The fun stuff

These are exploratory questions that go beyond what the thesis chapter
formally needs. Some will produce findings worth reporting; some will be
dead ends. Both are useful. Doing all of these is a few weeks of work,
so prioritize based on what catches your eye.

### 7. The "is the home crowd back?" question

The 2020_21 Bubble removed crowds entirely. 2021_22 was partial-capacity.
2022_23 onward is full crowds. **HCA recovery should track crowd
restoration. Does it?**

* Plot conditional HCA (regression intercept after factors absorbed) by
  season, with vertical lines at 2020 (Bubble) and 2022 (full crowds back).
* If the HCA *didn't* fully recover post-2022, that's a finding. Possible
  explanations: rule changes, refereeing, schedule changes.
* The 2024_25 data point matters here. Even one more datapoint past
  2023_24 helps.

### 8. Officiating's contribution to HCA

* **FT rate by home vs road.** Home teams shoot more free throws. Always
  have. By how much in your data?
* **Foul disparity.** From PF column: home teams commit fewer fouls.
  Quantify.
* **The "FT rate gap" trend.** Has the home-FT advantage shrunk? It might
  account for a measurable fraction of the conditional HCA decline.
* This is an unsexy but very real candidate explanation for the post-COVID
  HCA decline. Would make a great thesis chapter aside.

### 9. The dynastic teams

* Who's the best team-season in the data by each factor? By win_pct?
  (Probably 2016_17 Warriors or 2015_16 Warriors-with-the-record.)
* Who's the worst? (Bunch of candidates — 2020_21 Rockets, etc.)
* Pick the top-5 and bottom-5 team-seasons by net rating
  (mean_margin / pace * 100 or similar). Look at their factor profiles.
  What pattern of strengths and weaknesses defines elite vs awful?
* This isn't a regression question, it's a descriptive one. But it
  produces compelling thesis-chapter narratives.

### 10. The 3-point revolution as a four-factors story

* **eFG% can rise from "more 3s" or "better shooting." Decompose.** The
  efg formula: `(FGM + 0.5 * FG3M) / FGA`. Compute the "two-pointer eFG%"
  (just `FG2M / FG2A`) and the "three-pointer eFG%" (`1.5 * FG3M / FG3A`)
  separately. Plot both by season. Hypothesis: most of the eFG% rise comes
  from the *mix* shifting toward 3s, not from 2-point or 3-point shooting
  itself getting better.
* If true, that means the four-factor framework is partially obscuring
  the actual story. Worth a thesis chapter discussion.

### 11. Game-context HCA

Conditional on factors, the HCA is ~3 pts pre-COVID. But within a single
season, is it constant?

* HCA by month-of-season. Travel fatigue?
* HCA by day-of-week. Back-to-back games are death for the road team —
  but does a Wednesday HCA differ from a Saturday HCA?
* HCA by altitude. Denver. Utah. The classic. Compute mean home margin
  for high-altitude teams vs everyone else. Real or myth?
* HCA by time-zone-difference. East coast team plays west coast team
  with a 3-hour shift, who wins more often? (You'll have a Session 12-13
  travel feature for this. Until then, approximate from `team_abbr`.)

### 12. The "factor importance" stability question

If you fit the same regression on each season's data, how much do the
coefficients move year to year? The thesis chapter wants to claim "eFG%
became more important." That requires *real* coefficient drift, not
estimation noise.

* Bootstrap each per-season regression and report 95% CIs on each
  coefficient. Plot β with CI bands by season.
* If the eFG% coefficient's CI in 2015_16 doesn't overlap 2024_25's CI,
  the trend is real. If they overlap, even significant trends in the
  pooled-with-year-interaction model have to be reported with caution.

### 13. The matchup-level prediction game

This isn't EDA, it's a stretch goal. **Save for Session 13.**

* Given two teams' team-season factor averages, predict the outcome of
  a specific game between them. Standardize each team's factors against
  the league mean for that season. Difference them. Apply learned weights.
* Calibration: across the 10 seasons, does a model trained on one season
  predict the next season's games well? This is the prediction layer
  question and belongs in Session 13.
* But you can preview it here: train on 2015_16-2022_23, test on 2023_24,
  see how it does. The result will inform the Session 13 spec.

### 14. The "four factors are not enough" diagnostic

Eight features explain X% of game-margin variance. What does the residual
look like?

* Plot residuals from the symmetric regression vs `home_team_abbr`. Some
  teams might systematically over- or under-perform the four-factor
  prediction. (Coaching? Lineup tactics? Referee favoritism?)
* Plot residuals vs `game_date`. Any seasonal drift?
* Plot residuals vs game-total points (pace proxy). Does the model do
  worse on high-pace games?
* This is "what's the 4-factor model missing" — and it's the natural
  setup for a Session 13 conversation about whether to add features
  beyond the canonical four.

### 15. A specific 58/42 stress test

The team-season regression is supposed to recover something close to the
published 58/42 offense/defense split. What if we vary the spec?

* Fit on win_pct (default), then on net_rating, then on margin per game.
  Do the splits agree?
* Fit on each season separately (30 obs, 8 features — underpowered but
  fittable with regularization). Plot the off/def share by season.
  Was 58/42 always the answer, or has it drifted?
* Fit only on regular season vs only on playoffs. Different? (Probably
  noisier in playoffs but interesting if defense's share rises.)
* If multiple specs agree on a number near 58/42, that's robust. If they
  diverge wildly, the original 58/42 finding may be more spec-dependent
  than the literature acknowledges.

## Part III: Order of operations

A reasonable cadence to actually do this:

**Day 1 (1-2 hours):** Sections 1-2. Data hygiene and marginals. End the
session knowing the data has no surprises and you've seen every column's
distribution.

**Day 2 (2-3 hours):** Sections 4-5. The headline finding of the whole
thesis is HCA, so spend real time here. Build the per-season HCA trend
chart. Look at the Bubble. Look at the post-2020 recovery.

**Day 3 (1-2 hours):** Section 6 + section 8. Playoffs and officiating.
These are both potential "thesis chapter aside" material.

**Day 4 onward (open-ended):** Whichever of sections 7-15 catches your
attention. None of these are required for the thesis, but each could
produce a finding worth a paragraph or a chart.

## Part IV: Outputs to produce

For each EDA section, save:

* A Jupyter notebook in `notebooks/eda_NN_<topic>.ipynb` with the analysis
* PNG charts saved to `notebooks/figures/` (gitignored or committed depending
  on your preference)
* A bullet list of findings at the top of each notebook (so future-you
  can scan without re-running)

Don't write the notebooks like a tutorial. Write them like a working
research notebook: lots of tangents, half-finished sidequests, discarded
hypotheses with notes on why they were discarded. The mess is the value.

## Part V: One last thing

If any single section produces a finding that surprises you — really
surprises you, not just "huh, neat" — pause and investigate. Surprising
findings in clean data are either:

1. A bug in the upstream pipeline (unlikely; you've tested heavily, but
   possible)
2. A real phenomenon nobody else has noticed (rare but possible)
3. A misreading on your part of what the column actually represents
   (most likely)

Each is worth pursuing. The thesis benefits from clear documentation of
each surprise, what you investigated, and what conclusion you reached.
That's how analytical chemistry trained you to work, and the discipline
transfers cleanly to sports analytics.

The MIT Sloan crowd notices when an analyst respects their data. This is
how.
