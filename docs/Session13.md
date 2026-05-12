# Session 13: Inference Layer (Regression Analysis)

Status: **Draft. Sub-session 13.1 fully specified; 13.2, 13.3, 13.4 are stubs to be filled in as findings inform design.**

This is the analytical heart of the thesis. Builds the regression machinery on top of the analysis-layer foundation from Sessions 11-12, runs cross-validation to compare strategies, and produces the headline coefficient and HCA charts.

Session 13 is structured as **four sub-sessions** (Structure B from Session 12 discussion). Each sub-session is roughly one day of focused work. The detailed spec for each sub-session is written immediately before that work begins, so findings from earlier sub-sessions can inform later ones.

## Overview

| Sub-session | Topic | Status | Estimated effort |
|-------------|-------|--------|------------------|
| 13.1 | `regression.py` + `differentials.py` + tests | Specified below | 3-3.5 hrs |
| 13.2 | Cross-validation + coefficient stability | Stub | 3-4 hrs |
| 13.3 | `symmetric.py` + `team_season.py` + tests | Stub | 2-3 hrs |
| 13.4 | `hca.py` + `charts.py` + CLI integration | Stub | 3-4 hrs |
| **Total** | | | **11-14 hours** |

## Methodology decisions locked from Session 12 discussion

These apply across all four sub-sessions:

1. **Differentials as primary focus**, with symmetric and team-season as proper modules each getting their own analyses
2. **Both target variables**: linear regression on `home_margin`, logistic regression on `home_win`
3. **No crowd-presence indicator**: let the data speak; don't preemptively flag COVID-era seasons as special
4. **Confidence intervals**:
   - Per-season fits: analytical OLS standard errors from `statsmodels` (default)
   - Bootstrap as a sanity check on 3-5 representative seasons
   - Bootstrap on aggregate held-out MAE values in cross-validation (24 per-season MAEs resampled with replacement, 1000 iterations)
5. **Comparison metric**: mean of per-season MAEs (not pooled across games), so each season contributes equally and we see generalization-across-seasons rather than per-game error
6. **Feature scaling**: Run BOTH Option X (pooled mean/std from training set) and Option Y (per-season mean/std) for comparison. Option X vs Y comparison is itself a methodological finding worth reporting

## Cross-session conventions

* All sorts deterministic (pandas default `kind="stable"`)
* CV uses fixed `random_state=42` everywhere
* Output CSVs round-trip identically across runs
* Charts not asserted byte-stable (matplotlib jitter); test only that files are produced
* `statsmodels` (not sklearn) for fitting, since analytical standard errors are needed
* `sklearn.metrics` for evaluation metrics (AUC, log-loss)

---

# Session 13.1: Foundation Regression Machinery + Differentials

Status: **Draft, pending kickoff.**

## 1. Goal

Build `nba_four_factors/analysis/regression.py` with shared regression machinery (standardization, linear fit, logistic fit, in-sample metrics) plus `nba_four_factors/analysis/differentials.py` with the differentials-specific feature builder and per-season fitting logic.

Plus tests, plus updates to `analysis/__init__.py` for the public API.

This is the analytical heart of the regression layer. Everything in 13.2 (cross-validation), 13.3 (symmetric + team-season), and 13.4 (HCA + charts + orchestration) builds on these two modules.

## 2. Context: where Session 12 left off

29 seasons of clean processed data (1997-98 to 2025-26, regular season). 68,716+ team-game rows. Bubble anomaly fix in place. EDA 01 (hygiene) and EDA 02 (distributions) complete. Headline findings from EDA:

* HCA mean has declined from ~3.4 in late-90s to ~1.7 in 2024-26
* Single-game margin std has risen from ~12.5 (1997-2010 stable era) to ~16.4 in 2025-26
* The pre-2011 era is stable in variance; the 2011+ era shows the rise
* Three eras emerging in the data: pre-2011 (stable), 2011-2018 (gradual drift), 2019-onward (sharp acceleration)
* End-of-game foul mechanics produce bimodality in the margin distribution

The regression work in this sub-session is about turning these descriptive findings into quantitative claims about which factors drive HCA and how their weights have evolved.

## 3. Scope (locked)

### 3.1 Two modules

**`regression.py`** (shared machinery): pure functions for fitting OLS linear, fitting unregularized logistic, standardizing features, and extracting coefficients with analytical standard errors. The three regression flavors (differentials, symmetric, team-season) all delegate to these.

**`differentials.py`** (first flavor): builds home-minus-away feature differentials from the pivoted game-level DataFrame, runs per-season regressions for both linear (margin target) and logistic (win target), returns coefficient tables and metrics.

The other two flavors (`symmetric.py`, `team_season.py`) are deferred to Session 13.3.

### 3.2 What's NOT in 13.1

Explicitly deferred to other sub-sessions:

* Cross-validation (13.2)
* 29-curves visualization (13.2)
* `symmetric.py` and `team_season.py` (13.3)
* `hca.py` and `charts.py` (13.4)
* Orchestration / CLI integration (13.4)

The temptation to do "just a quick chart at the end" should be resisted. Charts are 13.4 territory and have their own design considerations.

## 4. Module structure

```
nba_four_factors/analysis/regression.py        ~120 lines
nba_four_factors/analysis/differentials.py     ~100 lines
nba_four_factors/analysis/__init__.py          updated public API
tests/analysis/test_regression.py              ~150 lines
tests/analysis/test_differentials.py           ~100 lines
```

Both files are pure functions. DataFrame in, DataFrame/dict out. No I/O. Tests use synthetic in-memory fixtures, no disk dependencies beyond optional smoke tests against real processed data.

## 5. Detailed design

### 5.1 `regression.py`

Five pure functions:

```python
def standardize(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score each column. Return (Xz, means, stds) so the same
    standardization can be reapplied to held-out data."""

def apply_standardization(
    X: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    """Apply a previously-computed standardization to new data.
    Used in cross-validation to scale test data with training means/stds.
    Raises ValueError if any X column not in means."""

def fit_linear(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Fit OLS linear regression with statsmodels.
    Returns a dict with:
        coefs: pd.Series (feature -> coefficient) including intercept
        std_errs: pd.Series (feature -> analytical SE) including intercept
        in_sample_r2: float
        in_sample_mae: float
        n_obs: int
        residuals: pd.Series (per-row residual, same index as y)
    """

def fit_logistic(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Fit unregularized logistic regression with statsmodels.
    penalty=None equivalent (use Logit with no regularization).
    Returns a dict with:
        coefs: pd.Series (feature -> coefficient) including intercept
        std_errs: pd.Series (feature -> analytical SE) including intercept
        in_sample_auc: float
        in_sample_accuracy: float (at 0.5 threshold)
        in_sample_log_loss: float
        n_obs: int
        predicted_probs: pd.Series (per-row predicted P(y=1))
    """

def ci_95(
    coefs: pd.Series,
    std_errs: pd.Series,
) -> pd.DataFrame:
    """Build a coefficient table with 95% CIs.
    Returns DataFrame with columns: coef, std_err, ci_lower, ci_upper.
    Index is feature names (matches coefs.index).
    """
```

**Design notes:**

* `statsmodels` (not sklearn) for both linear and logistic. Reason: we need analytical standard errors, which sklearn does NOT provide cleanly. statsmodels' OLS and Logit classes give us coefficients AND standard errors out of the box.
* `fit_linear` uses `statsmodels.api.OLS(y, sm.add_constant(X))`. The `add_constant` is required for an intercept term.
* `fit_logistic` uses `statsmodels.api.Logit(y, sm.add_constant(X))`. No regularization. Convergence settings default; bump `maxiter` if needed during testing.
* `standardize` z-scores using sample std (Bessel's correction, ddof=1) for consistency with pandas defaults. Returns the means/stds so they can be reapplied to held-out test sets in cross-validation.
* `ci_95` uses 1.96 as the z-multiplier, not a t-distribution lookup. At n=1230 games per season, the difference is negligible.
* All functions return new objects; nothing mutates the input.

**Dependencies to add to `pyproject.toml`:**

```bash
uv add statsmodels scikit-learn
```

`sklearn` for metrics (AUC, log-loss); we don't use it for fitting.

### 5.2 `differentials.py`

Three functions, building on `regression.py`:

```python
def build_differential_features(
    game_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute home-minus-away differentials for the four factors.

    Input: game-level DataFrame from pivot_to_game_level, with columns
    home_efg, home_tov, home_orb, home_ftr, away_efg, away_tov,
    away_orb, away_ftr, plus identifiers and home_margin/home_win.

    Output: DataFrame with the same row order, containing columns:
        game_id, game_date, season, season_type,
        efg_diff, tov_diff, orb_diff, ftr_diff,
        home_margin, home_win
    """

def fit_differentials_one_season(
    game_df: pd.DataFrame,
    season: str,
) -> dict:
    """Fit BOTH linear (margin) and logistic (win) for one season.

    Calls build_differential_features internally.
    Standardizes the four differentials within this season's data.
    Returns a dict:
        season: str
        n_games: int
        linear: dict (from regression.fit_linear)
        logistic: dict (from regression.fit_logistic)
        means: pd.Series (standardization means)
        stds: pd.Series (standardization stds)
    """

def fit_differentials_all_seasons(
    games_df: pd.DataFrame,
) -> dict[str, dict]:
    """Fit per-season differentials regressions for all seasons in input.

    Returns a dict keyed by season string, each value being a fit_dict
    from fit_differentials_one_season.

    Seasons are processed in chronological order.
    """
```

**Design notes:**

* `build_differential_features` is responsible for the rename and the subtraction. Inputs come from `pivot_to_game_level`. The function does NOT modify the input DataFrame.
* `efg_diff = home_efg - away_efg`. Same for the other three.
* `fit_differentials_one_season` is the workhorse. It standardizes within the season (per Session 11 §3.7), fits both regressions, and packages everything. Standardization happens AFTER differentials are computed, on the differential columns.
* `fit_differentials_all_seasons` is a thin wrapper. Iterates seasons, calls the per-season function. Returns a dict for easy lookup.
* The dict structure (not a DataFrame) is deliberate: each season's fit has variable internal structure (linear coefs, logistic coefs, std errs, metrics). A nested dict is the right shape. Cross-validation code in 13.2 will iterate this dict.

### 5.3 Public API update

`analysis/__init__.py` adds:

```python
from .regression import (
    standardize,
    apply_standardization,
    fit_linear,
    fit_logistic,
    ci_95,
)
from .differentials import (
    build_differential_features,
    fit_differentials_one_season,
    fit_differentials_all_seasons,
)
```

`__all__` updated correspondingly.

## 6. Tests

### 6.1 `tests/analysis/test_regression.py`

Cover the shared machinery thoroughly because everything else depends on it.

**`standardize` tests:**
* Each output column has mean ~0 and std ~1
* Returned means/stds match per-column input means/stds
* Empty input returns empty DataFrame with correct schema
* Single-row input handled gracefully (std becomes 0; document expected behavior, probably return NaN columns and let downstream handle)

**`apply_standardization` tests:**
* Applying training-data means/stds to identical test data produces zero z-scores
* Mismatched columns raise ValueError
* Handles new data with different means/stds correctly (re-centers and rescales using training stats, not new-data stats)

**`fit_linear` tests:**
* Synthetic dataset y = 2*x1 - 3*x2 + N(0, 0.1) on 500 obs: coefs land within tolerance (β1 ~= 2 +/- 0.05, β2 ~= -3 +/- 0.05)
* Synthetic dataset where y is exactly linear in X: residuals are zero
* R^2 for perfect linear data is ~1.0; for pure noise is ~0
* Standard errors are returned and are positive
* n_obs is correct
* Empty input raises a clear error (don't let it produce NaN coefs silently)

**`fit_logistic` tests:**
* Synthetic dataset where P(y=1) is strongly determined by X: AUC > 0.9
* Synthetic dataset where y is random: AUC ~= 0.5, accuracy ~= 0.5
* Predicted probs are bounded [0, 1]
* In-sample log-loss is computed correctly (verify against sklearn.metrics.log_loss for a known case)

**`ci_95` tests:**
* Output DataFrame has correct columns (coef, std_err, ci_lower, ci_upper)
* ci_upper - ci_lower = 2 * 1.96 * std_err for each row
* Coefficient with zero SE produces ci_lower == ci_upper == coef
* Negative coefs produce sensible CIs (signs match coef)

### 6.2 `tests/analysis/test_differentials.py`

**`build_differential_features` tests:**
* Output has correct columns (efg_diff, tov_diff, orb_diff, ftr_diff, game_id, game_date, season, season_type, home_margin, home_win)
* Differential math: a hand-constructed 2-game DataFrame produces expected diffs (e.g., home_efg=0.55, away_efg=0.50 -> efg_diff=0.05)
* Row order preserved from input
* Empty input returns empty DataFrame with correct schema

**`fit_differentials_one_season` tests:**
* End-to-end on a synthetic 100-game fixture for a single season:
  - Returns dict with expected keys (season, n_games, linear, logistic, etc.)
  - n_games matches input row count
  - linear['coefs'] has 5 entries (4 features + const)
  - logistic['coefs'] has 5 entries
  - means and stds are returned for the 4 differential features
* Synthetic data where home team systematically dominates: intercept in linear fit is positive (captures HCA)
* Synthetic data where home_win is perfectly determined by efg_diff: logistic AUC ~= 1.0

**`fit_differentials_all_seasons` tests:**
* 3-season synthetic fixture produces 3 entries in the output dict
* Each entry has the expected structure
* Seasons are processed independently (no cross-season contamination)
* Empty input returns empty dict

### 6.3 Test fixture strategy

Build a `make_synthetic_games_df` factory in `tests/analysis/conftest.py` that produces game-level DataFrames matching what `pivot_to_game_level` would output. Parameters: n_seasons, n_games_per_season, optional bias toward home team (for HCA testing), optional factor correlation with margin (for coefficient testing).

Tests then use this factory rather than reading from disk. This keeps tests fast (no Parquet I/O) and deterministic (controllable random seeds).

## 7. Implementation order

Three passes:

1. **`regression.py` first.** Build the five functions and write all their tests. Get them passing. This is the foundation; everything else depends on the standard errors being computed correctly.

2. **`differentials.py` second.** Build the three functions on top of the verified `regression.py`. Write the tests. Get them passing.

3. **`__init__.py` and smoke test.** Update the public API. Run a quick real-data smoke test (load 2024_25, fit differentials, inspect coefficients).

Resist the temptation to write any analysis or visualization code in this session. The temptation will be strong because seeing the coefficients for the first time is exciting. Save that for 13.2.

## 8. Smoke test plan

After tests pass, this should work:

```bash
uv run python -c "
from nba_four_factors.analysis import (
    load_processed,
    pivot_to_game_level,
    fit_differentials_one_season,
    ci_95,
)
from nba_four_factors.config import SeasonType

df = load_processed(('2024_25', '2024_25'), SeasonType.REGULAR)
games = pivot_to_game_level(df)
fit = fit_differentials_one_season(games, '2024_25')

print(f'season: {fit[\"season\"]}')
print(f'n_games: {fit[\"n_games\"]}')
print()
print('Linear fit (margin):')
linear_table = ci_95(fit['linear']['coefs'], fit['linear']['std_errs'])
print(linear_table.round(3))
print(f'R^2: {fit[\"linear\"][\"in_sample_r2\"]:.3f}')
print(f'MAE: {fit[\"linear\"][\"in_sample_mae\"]:.3f}')
print()
print('Logistic fit (win):')
logistic_table = ci_95(fit['logistic']['coefs'], fit['logistic']['std_errs'])
print(logistic_table.round(3))
print(f'AUC: {fit[\"logistic\"][\"in_sample_auc\"]:.3f}')
"
```

**Expected output (roughly):**

* Linear coefficients in standardized units. β_efg should be positive and largest (most predictive). β_tov should be negative (turnovers hurt margin). β_orb positive, β_ftr positive.
* The intercept (const) should be positive (~1.5-2.0 in margin points), reflecting conditional HCA for 2024_25.
* R^2 in the 0.20-0.30 range. The four factors explain ~25% of single-game margin variance — the other 75% is residual noise.
* MAE around 9-10 points (consistent with margin std of ~16 and R^2 of 0.25).
* Logistic AUC around 0.70-0.75. Predicts wins better than coin flip, not perfectly.

If these numbers land, the foundation is solid and we're ready for 13.2.

## 9. Open questions for review

1. **Use `statsmodels` for fitting or implement OLS by hand from numpy?** Default: statsmodels. Reasons: SE computation is built in, the API is well-tested, no need to reinvent. Risk: another dependency. Mitigation: statsmodels is mature and widely-used.

2. **Should `fit_linear` return predicted values too?** Currently returns residuals, from which predicted = y - residuals. Cross-validation in 13.2 will need predicted values. We could compute them on demand from the coefficient table + input X, or return them. Default: don't return in 13.1; the prediction function lives in 13.2 where it's actually used.

3. **What about heteroscedasticity-robust SEs?** statsmodels supports HC0, HC1, HC2, HC3 robust standard errors. Probably worth using for the thesis methodology. Default: regular SEs in 13.1, add HC3 option in 13.2 if heteroscedasticity testing surfaces a need.

4. **Convergence handling for logistic on weird seasons?** Lockout seasons (1998_99, 2011_12) and short seasons (2020_21) have fewer games. Possible numerical issues. Default: trust statsmodels to handle; add explicit error handling if it surfaces during testing.

5. **MAE on the original margin scale or on the standardized scale?** The linear regression is fit on standardized features but predicts raw margin in points. MAE should be in points (raw scale) so it's interpretable. Default: yes, MAE in points. Code needs to be explicit about this since the coefficients are in standardized units while the target is not.

## 10. Estimated effort

* `regression.py`: 90 minutes (writing + tests + iteration)
* `differentials.py`: 60 minutes
* Public API + smoke test: 15 minutes
* Buffer for debugging: 30 minutes
* **Total: ~3-3.5 hours of focused work**

Bigger than original Session 12 estimate for pivots, smaller than the spec might suggest because the test patterns are now familiar.

## 11. Findings from 13.1

_To be filled in after sub-session completion. Likely topics: which coefficients had the cleanest fits, which seasons had convergence issues, what the typical R^2 lands at, any methodology surprises._

---

# Session 13.2: Cross-Validation + Coefficient Stability Visualization

Status: **Stub. To be fully specified after 13.1 completes.**

## 1. Goal

Use the regression machinery from 13.1 to:
1. Compare strategies (A, C, B-rolling-3, B-exp-weighted) x (Option X, Y scaling) = 8 combinations via leave-one-season-out cross-validation
2. Produce the 29-curves coefficient stability visualization

## 2. High-level scope

Two main analyses:

### 2.1 Cross-validation harness

Build `nba_four_factors/analysis/cross_validation.py` (or similar). LOSOCV across 24 valid test seasons (1999_00 through 2025_26). Per-season MAE values, bootstrap CI on aggregate mean.

Strategies to compare:
* **A**: pooled across all training seasons
* **C**: era-matched (pre-2011 / 2011-2018 / 2019-onward)
* **B-rolling-3**: last 3 training seasons before test season
* **B-exp-weighted**: exponentially-weighted across all training seasons (λ=0.85 default, may tune)

Scalings:
* **Option X**: pooled mean/std from training set
* **Option Y**: per-season mean/std within training set

Compare on both linear (MAE in points) and logistic (log-loss, AUC).

### 2.2 Coefficient stability visualization

29 per-season fits, each with 4 coefficients and analytical CIs. Plot β_efg, β_tov, β_orb, β_ftr across 29 seasons with 95% CI ribbons. 4 panels.

## 3. Outputs

* `data/analysis/cross_validation/results_linear.csv` — per-strategy per-season MAE
* `data/analysis/cross_validation/results_logistic.csv` — per-strategy per-season log-loss
* `data/analysis/cross_validation/aggregate_with_ci.csv` — bootstrap CI on aggregate
* `charts/coefficient_history_29seasons.png` — 4-panel coefficient evolution
* `charts/cv_strategy_comparison.png` — per-season + aggregate visualization

## 4. Open methodology questions for build-time

* Bootstrap iteration count (1000 default, may need fewer if compute slow)
* What to do with 2019_20 (Bubble) in B-rolling-3 if it appears in the training window for a test season — probably treat as normal but flag
* How to report the Option X vs Y comparison (side-by-side or merged)
* Whether to include 2025_26 in test set or hold out as future-data validation

## 5. Detailed spec

_To be written before 13.2 kickoff._

## 6. Findings from 13.2

_To be filled in after sub-session completion._

---

# Session 13.3: Symmetric and Team-Season Regressions

Status: **Stub. To be fully specified after 13.2 completes.**

## 1. Goal

Build the remaining two regression flavors using the `regression.py` machinery from 13.1.

## 2. High-level scope

### 2.1 `symmetric.py`

Per-season regression with all 8 factors (off_efg, off_tov, off_orb, off_ftr, def_efg, def_tov, def_orb, def_ftr) as predictors. Both linear (margin) and logistic (win) targets.

Key methodological feature: the per-game paired structure forces β_off = -β_def to numerical precision (mathematical artifact, not signal). The useful output is `|β_off| + |β_def|` per factor per season — total factor weight.

### 2.2 `team_season.py`

One regression on all team-season rows (one row per team per season, not per-game). Predicts win_pct from 8 team-season averages. ~750 rows total.

Methodologically distinct from differentials and symmetric because:
* No per-season fitting (single global model)
* Symmetry isn't enforced (sample is one row per team per season)
* Where the 58/42 offense/defense split would be visible

## 3. Open methodology questions for build-time

* Should `team_season.py` use 8 features or 4 differentials at team-season level?
* How to compute weight_shares correctly: |β_off| / (|β_off| + |β_def|) per factor pair?
* Should we run a per-season variant of the team-season regression too (30 obs per season, underpowered)?
* Whether to include CV on the team-season regression (only ~25 train seasons vs 1 held-out, smaller sample makes CV less informative)

## 4. Detailed spec

_To be written before 13.3 kickoff._

## 5. Findings from 13.3

_To be filled in after sub-session completion._

---

# Session 13.4: HCA Computation + Charts + CLI Integration

Status: **Stub. To be fully specified after 13.3 completes.**

## 1. Goal

Final analysis-layer modules: HCA computation, all four headline charts, and CLI orchestration. Completes the analysis layer.

## 2. High-level scope

### 2.1 `hca.py`

Compute raw HCA (per-season mean home margin, neutral games excluded) and conditional HCA (per-season intercept from differentials regression). The chart layer plots both on the same axes.

### 2.2 `charts.py`

Four headline charts, each a pure function (DataFrame in, file out):

* `plot_hca_trend` — raw + conditional HCA, both lines on one chart
* `plot_differentials_coef_trends` — 4 panels, per-season β over time with CI ribbons
* `plot_symmetric_total_weights` — |β_off| + |β_def| per factor per season
* `plot_team_season_weight_shares` — off vs def share per factor, bar chart

### 2.3 `orchestration/analyze.py`

CLI driver:
```bash
uv run python -m nba_four_factors.cli analyze --regression all
uv run python -m nba_four_factors.cli analyze --regression differentials --no-charts
```

### 2.4 `cli.py` updates

Add `analyze` subcommand following the existing pattern.

## 3. Outputs

```
data/analysis/regression/
  differentials/coefficients_linear.csv
  differentials/coefficients_logistic.csv
  differentials/metrics.csv
  symmetric/coefficients_linear.csv
  symmetric/coefficients_logistic.csv
  symmetric/metrics.csv
  symmetric/total_weights.csv
  team_season/coefficients_linear.csv
  team_season/metrics.csv
  team_season/weight_shares.csv
charts/
  hca_trend.png
  differentials_coef_trends.png
  symmetric_total_weights.png
  team_season_weight_shares.png
```

## 4. Open methodology questions for build-time

* Chart styling conventions (color palette, era markers, font choices)
* DPI for thesis-quality output (150 default; may want 300 for print)
* Whether to add interactive (plotly/bokeh) versions for exploration vs static PNG for thesis
* CLI flag structure (one big `--regression all` vs per-flavor)

## 5. Detailed spec

_To be written before 13.4 kickoff._

## 6. Findings from 13.4

_To be filled in after sub-session completion._

---

# Final synthesis (post-Session 13)

_To be written after all four sub-sessions complete. Topics:_

* _Which strategy won the cross-validation comparison and by how much_
* _Did Option X vs Y scaling matter?_
* _How did per-season coefficients drift across the 29-season range?_
* _Did the 58/42 offense/defense split reproduce?_
* _Conditional HCA trend: does it follow the same shape as raw HCA, or differ?_
* _What's the headline number for the thesis chapter?_
