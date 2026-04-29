# Session 11: Inference Layer (Regression Analysis)

Status: **Draft, pending review.** Locked sections marked.

## 1. Goal

Build the inference layer: per-season and per-team-season regressions that
produce the headline thesis findings about which four factors matter, by how
much, and how their weights have changed across 25 seasons. Read from the
Session 10 processed Parquets; write coefficient tables, model fit metrics,
and trend charts.

The deliverables are scientific findings (coefficient estimates, HCA trend,
offense/defense weight decomposition) and the analytical scaffolding to
reproduce them deterministically. This is NOT a prediction layer (deferred
to Session 13).

## 2. Context: what came before, what comes after

```
Session 10  →  data/processed/{season}/{season_type}.parquet
                (39 columns, one row per (game, team), the four factors)
                              |
                              v
Session 11  →  nba_four_factors/analysis/    ← THIS SESSION
                regression: differentials, symmetric, team-season aggregation
                outputs: coefficients, metrics, HCA trend
                              |
                              v
Session 12  →  nba_four_factors/processed/travel_rest.py
                add days_rest, traveled, opp_days_rest, opp_traveled
                to the processed Parquet (schema extension)
                              |
                              v
Session 13  →  nba_four_factors/prediction/
                exp-weighted rolling features, factor caps, composite dw,
                logistic regression with calibration, season-win projections
```

Sessions 12 and 13 are described in §10 of this spec for context. Their
specs will be drafted in their own sessions.

## 3. Scope (locked)

### 3.1 Three regressions

Three distinct regression analyses, each answering a different question:

* **Differentials regression** (per-season, game-level)
  Question: how much does a 1-SD home-minus-away advantage in each factor
  predict home margin and home win? What's the residual home-court
  advantage after these are absorbed?
  Predictors: `efg_diff`, `tov_diff`, `orb_diff`, `ftr_diff` (home minus
  away). Targets: home margin (linear), home win (logistic).
  Notes: this is the rebuilt cleaner version of `regression_differentials.py`.

* **Symmetric regression** (per-season, game-level)
  Question: when both team and opponent factors are in the model, what are
  the standardized total weights of each factor?
  Predictors: all 8 factors (`off_*` and `def_*`). Targets: margin (linear),
  win (logistic).
  Notes: this is the rebuilt cleaner version of `regression_analysis.py`.
  Critical: per-game data is paired by construction, which forces
  `β_off = -β_def` exact to numerical precision. This is mathematical
  symmetry, not signal. The output is preserved for the
  `|β_off| + |β_def|` total-weight comparison only. The regression cannot
  decompose offense vs defense at this aggregation level.

* **Team-season aggregation regression** (one row per team per season)
  Question: at the team-season level, with paired-game symmetry broken,
  what fraction of total predictive weight is offensive vs defensive?
  Predictors: 8 team-season averages of off_* and def_* factors.
  Targets: win count (linear) or win % (linear). Logistic doesn't apply.
  Notes: this is the attempt to recover something like the published
  ~58/42 offense/defense split. Outcome-honest: if the regression produces
  a clean split, report it; if it doesn't, report what it does produce.

### 3.2 Output target

```
data/analysis/
  regression/
    differentials/
      coefficients_linear_rs.csv
      coefficients_linear_po.csv
      coefficients_logistic_rs.csv
      coefficients_logistic_po.csv
      metrics_rs.csv
      metrics_po.csv
    symmetric/
      coefficients_linear_rs.csv
      coefficients_linear_po.csv
      coefficients_logistic_rs.csv
      coefficients_logistic_po.csv
      metrics_rs.csv
      metrics_po.csv
    team_season/
      coefficients_linear.csv
      metrics.csv
      weight_shares.csv
charts/
  hca_trend.png                    ← raw + conditional, both lines
  differentials_coef_trends.png    ← 4 panels, per-season β over time
  symmetric_total_weights.png      ← total |β_off| + |β_def| per factor
  team_season_weight_shares.png    ← off vs def share per factor
```

`data/analysis/` is gitignored (derived from `data/processed/`). `charts/`
is gitignored too; published thesis figures get committed elsewhere.

### 3.3 Codebase location (locked)

Inside the package: `nba_four_factors/analysis/`. Same layering discipline
as `processed/`: pure functions in narrowly-scoped modules, I/O wrappers
where needed, a CLI driver in `orchestration/analyze.py`, tests in
`tests/analysis/`.

CLI invocation:

```bash
uv run python -m nba_four_factors.cli analyze --regression differentials --season-type regular
```

```bash
uv run python -m nba_four_factors.cli analyze --regression all
```

### 3.4 Season scope

All 25 seasons present in `data/processed/`. The analysis layer reads what's
there; if a season's processed file doesn't exist, that season is silently
omitted from the output (logged at INFO level).

Regular season and playoffs are handled as separate runs, not pooled. They
have different sample sizes per season (~1230 vs ~80 games) and arguably
different game dynamics; pooling would obscure both.

Play-in is excluded from regression analyses. Sample size is too small per
season (~10 games) and the games are structurally a hybrid of late-regular
and early-playoff dynamics. Including them would be a methodological
distraction.

### 3.5 Pooled vs per-season

Per-season is the primary output. 25 separate per-season fits per regression,
producing the "β over time" tables and charts that visualize drift.

A pooled fit (one regression on all 25 seasons combined) is also produced
for each regression as a robustness summary. Two flavors:

* **Naive pool**: all rows combined, one set of coefficients with standard
  errors (clustered by season for inference)
* **Year-trend pool**: one set of coefficients plus interaction with a
  centered year variable, to test "is the trend significant" formally

The per-season fits are the visualization. The pooled fits are the inference
the thesis chapter will cite for claims like "the eFG% coefficient
significantly increased from era X to era Y."

### 3.6 Train/test methodology

Both flavors reported, in separate columns of the metrics CSVs:

* **In-sample fit metrics** (R², AUC, log-loss): computed on the same data
  the model was trained on. These are the existing published numbers.
  Useful for inference (coefficient estimation), with no overfitting risk
  at 4-8 features and 1000+ observations per season. Honest about what
  they are (training metrics, not held-out).

* **Held-out CV metrics** (5-fold CV R², CV AUC): cross-validated estimates
  of generalization error. These are the numbers the thesis chapter cites
  for any prediction-flavored claim.

Both reported because they answer different questions. A common mistake is
to report only training metrics and call them "model performance"; another
common mistake is to report only CV metrics and use them for inference.

### 3.7 Standardization

Per-season z-scoring within each regression. This means:

* Within a single season's regression, all four (or eight) features are
  zero-mean, unit-variance
* Coefficients are then reported in 1-SD units, comparable across factors
* Coefficients are NOT directly comparable across seasons in raw efg-points
  units; "1-SD eFG% in 2024" is a different raw number than in 2001
* For cross-season trend charts, the 1-SD units are still meaningful
  because they represent "1-SD relative to that season's own variance"

This matches the existing convention in `regression_analysis.py` and
`regression_differentials.py`. Documented in the module docstring.

### 3.8 Regularization

* Linear: no regularization (`LinearRegression`, no penalty).
* Logistic: explicit `penalty=None`. The existing code uses `C=1e6` as a
  workaround for sklearn's defaults; we'll use the modern explicit syntax.

Justification: 4-8 features and ~1000-2500 observations per season give a
features-per-row ratio that doesn't need regularization for stable estimation.
For a thesis chapter, regularization adds an extra knob to defend without
changing conclusions.

If pooled regressions encounter numerical issues with 25 seasons of data
(unlikely), we'll add a minimal Ridge penalty and document.

## 4. Decisions to lock during build

These are TODOs to confirm with Mike during the build session, not before.
Surfacing them now so they're not surprises.

### 4.1 Charting library

`matplotlib` is in the existing code. Continuing with it for consistency. If
the thesis wants paper-grade figures eventually, that's a separate session.

Open subdecisions: shared color palette across charts, font choice, figure
DPI for thesis-quality output. Will land sensible defaults during build.

### 4.2 HCA — raw vs conditional, both reported

Decided: chart the HCA trend showing both raw and conditional, on the same
axes, with both lines clearly labeled. The thesis chapter can cite either or
both; the chart makes the methodological choice transparent rather than
hiding it.

* **Raw HCA**: per-season mean of home margin (unconditional)
* **Conditional HCA**: per-season intercept of the differentials linear
  regression (after factors absorbed)

### 4.3 Team-season regression: wins or win%?

Default: win %. Cleaner unit (bounded 0-1), shorter seasons (2020-21 was 72
games) don't bias the regression toward shorter-season teams.

If win count is needed for some downstream comparison, easy to add as a
secondary regression. Default produces win %.

### 4.4 Logistic accuracy threshold

Default: 0.5. Used only for accuracy reporting; AUC is the primary metric.
Threshold choice doesn't affect AUC or log-loss.

## 5. Module structure

Mirrors `processed/`. Pure functions split from I/O. Each module owns one
analytical concern.

```
nba_four_factors/analysis/
  __init__.py              ← public API exports
  _loading.py              ← read processed Parquets, filter by season_type,
                             concat across seasons, basic dropna
  pivots.py                ← team-game → game-level (home/away on one row),
                             team-game → team-season aggregations
  regression.py            ← shared regression machinery: standardize, fit,
                             extract coefficients and metrics
  differentials.py         ← differentials-specific feature build + fit
  symmetric.py             ← symmetric-specific (uses processed columns
                             directly; no feature build needed)
  team_season.py           ← team-season aggregation + regression
  hca.py                   ← raw + conditional HCA computation
  charts.py                ← all plotting functions, one per chart, pure
                             (DataFrame in, file out)
nba_four_factors/orchestration/
  analyze.py               ← CLI driver: dispatch to the three regressions,
                             write all outputs
tests/analysis/
  conftest.py              ← shared fixtures for analysis tests
  test_loading.py
  test_pivots.py
  test_regression.py       ← machinery tests with synthetic data
  test_differentials.py    ← end-to-end on small fixture
  test_symmetric.py        ← antisymmetry verification, total-weight extraction
  test_team_season.py      ← end-to-end on small fixture
  test_hca.py
  test_charts.py           ← smoke tests: chart files produced, axes labeled
  test_cli_analyze.py      ← CLI dispatch tests
```

### 5.1 Public API

```python
from nba_four_factors.analysis import (
    load_processed,            # _loading.read across seasons + filter
    pivot_to_game_level,       # pivots
    aggregate_to_team_season,  # pivots
    fit_differentials,         # differentials
    fit_symmetric,             # symmetric
    fit_team_season,           # team_season
    compute_hca,               # hca
    plot_hca_trend,            # charts
    plot_differentials_coef_trends,
    plot_symmetric_total_weights,
    plot_team_season_weight_shares,
)
```

### 5.2 Pure-function discipline

Following Session 10's pattern:

* `_loading.read_processed(season_range, season_type)` is the only function
  that touches disk for input. Every analysis function takes a DataFrame.
* `charts.plot_*(df, output_path)` is the only function that touches disk
  for output.
* Everything else is `DataFrame in, DataFrame out` (or scalar/dict out).

This makes the functions trivial to test with hand-constructed fixtures and
keeps the inner logic free of "did the file load correctly" concerns.

## 6. File-by-file plan

### 6.1 `_loading.py`

```python
def read_processed(
    season_range: tuple[str, str],
    season_type: SeasonType,
) -> pd.DataFrame:
    """Concatenate per-season processed Parquets in the range."""
```

* Globs `data/processed/{season}/{season_type_snake}.parquet` for every
  season in the inclusive range.
* Concatenates with `pd.concat(ignore_index=True)`.
* Logs at INFO any season whose file is missing (silent skip, not error).
* Returns an empty DataFrame with the right schema if no files match.

### 6.2 `pivots.py`

```python
def pivot_to_game_level(team_game_df: pd.DataFrame) -> pd.DataFrame:
    """One row per game, home and away features both present.

    Includes columns: game_id, game_date, season,
                      home_team, away_team, is_neutral,
                      home_efg, home_tov, home_orb, home_ftr,
                      away_efg, away_tov, away_orb, away_ftr,
                      home_margin, home_win.
    """

def aggregate_to_team_season(team_game_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season) with averages of all 8 factors plus
    wins, games, win_pct, mean_margin.

    Includes columns: season, team_abbr, games, wins, win_pct,
                      mean_margin,
                      off_efg, off_tov, off_orb, off_ftr,
                      def_efg, def_tov, def_orb, def_ftr.
    """
```

`pivot_to_game_level` handles the home/away split correctly. Two design
notes:

* Neutral-site games (`is_neutral=True`) have no home team. They're either
  excluded from pivots (default, since "home advantage" is not defined) or
  included as `is_neutral=True` rows that downstream regression silently
  drops. Default: exclude. Documented in docstring.
* The merge key is `(game_id, away_team)` to prevent cross-game contamination.

`aggregate_to_team_season` is straightforward groupby. Care taken to not
average across `season_type` if both regular and playoffs exist in the input.
Caller filters first.

### 6.3 `regression.py`

```python
def fit_linear(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit OLS, return coefficients (Series indexed by feature name),
    intercept, in-sample R², 5-fold CV R²."""

def fit_logistic(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit unregularized logistic, return coefficients, intercept,
    in-sample AUC and accuracy, 5-fold CV AUC."""

def standardize(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score each column. Returns (Xz, means, stds)."""
```

Shared machinery. The three regression-specific modules delegate to these.

### 6.4 `differentials.py`

```python
def build_differential_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute home-minus-away differentials for the four factors.
    Returns DataFrame with columns: efg_diff, tov_diff, orb_diff, ftr_diff."""

def fit_differentials(
    game_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-season differentials regression.
    Returns (linear_coefs, logistic_coefs, metrics) all indexed by season."""
```

Per-season loop calls `regression.fit_linear` and `fit_logistic` on the
differential features. Output DataFrames are wide format with one row per
season.

### 6.5 `symmetric.py`

```python
def fit_symmetric(
    team_game_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-season symmetric (8-feature) regression.
    Returns (linear_coefs, logistic_coefs, metrics) all indexed by season."""

def total_factor_weights(linear_coefs: pd.DataFrame) -> pd.DataFrame:
    """For each season and each factor, return |β_off| + |β_def|.
    These totals are the meaningful output of the symmetric regression
    (the off/def split itself is a structural artifact)."""
```

Includes a docstring warning about the antisymmetry. Tests assert that
`β_off ≈ -β_def` to numerical precision — that's the proof the regression
is correctly capturing the structural symmetry, and a regression test for
any future refactor.

### 6.6 `team_season.py`

```python
def fit_team_season(
    team_season_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single regression on all team-season rows.

    Returns (coefficients, metrics).  Coefficients are 8 features
    (off + def), one row.  Metrics include in-sample R², 5-fold CV R²,
    and per-factor weight shares.
    """

def weight_shares(coefs: pd.Series) -> pd.DataFrame:
    """For each factor pair, compute |β_off| / (|β_off| + |β_def|).
    Returns DataFrame with columns: factor, off_share, def_share."""
```

The team-season regression is the place where 58/42 (or whatever it actually
is) gets recovered. One regression on all 25-season-worth of team-season
rows, ~750 rows total (30 teams × 25 seasons, minus a few defunct teams
in early years).

Alternative: per-season team-season regression (one regression per season,
on 30 team-season rows each). Statistically underpowered (30 obs, 8
features) but visualizes drift if needed. Default is single pooled
regression; per-season as an option.

### 6.7 `hca.py`

```python
def compute_hca(
    team_game_df: pd.DataFrame,
    differentials_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Per-season HCA, both raw and conditional.

    Returns DataFrame indexed by season with columns:
        raw_hca         : mean home margin for that season
        conditional_hca : differentials regression intercept
    """
```

Raw HCA filters out neutral-site games before averaging. Conditional HCA
just reads from the differentials metrics. This is mostly bookkeeping;
the actual computation is one line each.

### 6.8 `charts.py`

```python
def plot_hca_trend(hca_df: pd.DataFrame, out_path: Path) -> Path: ...
def plot_differentials_coef_trends(coefs: pd.DataFrame, out_path: Path) -> Path: ...
def plot_symmetric_total_weights(totals: pd.DataFrame, out_path: Path) -> Path: ...
def plot_team_season_weight_shares(shares: pd.DataFrame, out_path: Path) -> Path: ...
```

Pure functions: DataFrame in, file out. All four return the path written so
the CLI can present them. No state, no globals.

Style decisions (defaults, easily overridden):

* Figure size: 14 wide × 7 tall for trend charts; 10 × 6 for single panels
* DPI: 150 for default output (sharp on screen, decent for paper)
* Color palette: existing one from `regression_analysis.py` (warm offense,
  cool defense)
* Era markers: vertical dashed line at 2015 (three-point inflection) and
  2019 (COVID), light gray, low alpha
* Annotations: factor name + season label on first/last points

### 6.9 `orchestration/analyze.py`

```python
def run_analyze(args: argparse.Namespace) -> int: ...
```

Dispatches based on `args.regression`. For each requested regression:

1. Load processed data via `_loading.read_processed`
2. Pivot or aggregate as needed
3. Fit
4. Write CSVs to `data/analysis/regression/<flavor>/`
5. Optionally write charts to `charts/`
6. Log summary to stdout

`--regression all` runs all three.

### 6.10 CLI changes (`cli.py`)

Add the `analyze` subparser following the existing pattern:

```python
def _add_analyze_subparser(subparsers) -> None:
    az = subparsers.add_parser(
        "analyze",
        help="Run regression analyses on the processed layer.",
    )
    az.add_argument(
        "--regression",
        choices=["differentials", "symmetric", "team_season", "all"],
        required=True,
    )
    az.add_argument(
        "--season-type",
        choices=["regular", "playoffs"],
        default="regular",
    )
    az.add_argument(
        "--season-range",
        metavar="YYYY_YY:YYYY_YY",
        default=None,
        help="Inclusive range; default is all seasons present.",
    )
    az.add_argument("--no-charts", action="store_true")
```

`team_season` ignores `--season-type` (uses regular season only by
construction). Driver flags this in INFO logging if the user passes one.

## 7. Tests

### 7.1 Loading

* `read_processed` returns concatenated DataFrames matching expected schema
* Missing seasons silently skipped, INFO logged
* Empty result is a properly-shaped empty DataFrame, not None

### 7.2 Pivots

* `pivot_to_game_level` produces correct row count: input rows / 2 (modulo
  neutral-site exclusions)
* `home_team` and `away_team` correctly assigned for standard games
* Neutral-site games excluded by default; included if `include_neutral=True`
* Hand-computed differentials match expected values on a 2-team fixture
* `aggregate_to_team_season` produces 30 rows per season (modulo defunct
  teams), correct win counts, correct factor averages

### 7.3 Regression machinery

* `fit_linear` on a deterministic synthetic dataset produces known
  coefficients (synthetic: y = 2*x1 - 3*x2 + ε; assert β within tolerance)
* `fit_logistic` analogously
* `standardize` produces zero-mean unit-variance columns; means and stds
  returned correctly
* CV metrics computed and returned

### 7.4 Differentials

* End-to-end on a 2-season fixture: per-season coefficients DataFrame
  has correct shape (2 rows, expected columns)
* HCA intercept is positive on a fixture where home teams have systematic
  margin advantage
* All metric columns present and within sane ranges

### 7.5 Symmetric

* End-to-end produces shape-correct outputs
* **Antisymmetry verification test**: assert that for every season and
  every factor, `linear_coefs[off_X] + linear_coefs[def_X] == 0` to within
  1e-6. This is the structural property; a regression test for future
  refactors.
* `total_factor_weights` produces non-negative values

### 7.6 Team-season

* Aggregation produces correct team-season averages on small fixture
* Regression runs on a synthetic 30-team-season fixture
* Weight shares sum to 1.0 per factor pair
* Per-season variant produces 25 rows when given 25 seasons

### 7.7 HCA

* Raw HCA: hand-computed mean margin on small fixture matches expected
* Neutral-site games correctly excluded
* Conditional HCA correctly extracted from differentials metrics

### 7.8 Charts

* Each plot function produces a file at the requested path
* Files are non-empty and PNG (magic bytes check)
* Smoke test only — no pixel-level comparison

### 7.9 CLI

* `analyze --regression differentials` dispatches to `run_analyze`
* `--regression all` triggers all three
* Invalid combinations error correctly
* `--no-charts` skips chart generation but still writes CSVs

## 8. Determinism and reproducibility

* All sorts deterministic (pandas default sort, plus explicit
  `kind="stable"` on the per-season groupby)
* CV uses fixed `random_state=42` everywhere
* Output CSVs round-trip identically across runs (test asserts MD5 stable)
* Charts not asserted byte-stable (matplotlib has font-rendering jitter
  across machines); test only that files are produced

## 9. Worked example: end-to-end after Session 11 ships

```bash
uv run python -m nba_four_factors.cli analyze --regression all
```

Expected output structure:

```
data/analysis/regression/differentials/coefficients_linear_rs.csv  (25 rows × 4 features + 2 metric cols)
data/analysis/regression/differentials/metrics_rs.csv               (25 rows × ~5 metric cols)
data/analysis/regression/symmetric/coefficients_linear_rs.csv       (25 rows × 8 features + intercept + R²)
data/analysis/regression/team_season/coefficients_linear.csv         (1 row × 8 features + R² + CV-R²)
data/analysis/regression/team_season/weight_shares.csv               (4 rows: factor, off_share, def_share)

charts/hca_trend.png                          (raw + conditional, both lines)
charts/differentials_coef_trends.png          (4 panels)
charts/symmetric_total_weights.png            (4 lines, totals over time)
charts/team_season_weight_shares.png          (single bar chart, off/def split)
```

Sanity checks the user can run after:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('data/analysis/regression/differentials/metrics_rs.csv', index_col='season')
print('seasons:', len(df))
print('mean conditional HCA:', df['lin_intercept'].mean().round(3))
print('post-COVID conditional HCA:', df.loc['2019-20':'2024-25', 'lin_intercept'].mean().round(3))
"
```

The expected output should land near the published findings: pre-COVID
conditional HCA around 3.0, post-COVID dropping to ~1.8. If those don't
reproduce, the regression diverges from the prior implementation in some
identifiable way and gets investigated before merge.

## 10. Look-ahead

### 10.1 Session 12: Travel and rest features

Goal: extend the processed Parquet schema with `days_rest`, `traveled`,
`opp_days_rest`, `opp_traveled`. These are needed by the prediction layer
(Session 13) and useful as covariates in any inference layer extension.

Approach:

* New module `nba_four_factors/processed/travel_rest.py` exposing
  `add_travel_rest(team_game_df) -> team_game_df` (pure function, four new
  columns added).
* `MetroMap` constant carrying franchise-to-metro mappings, including
  historical relocations (NJN, SEA, VAN, NOH/NOK with 2005-06 Katrina
  handling).
* Vectorized implementation using `groupby("team_abbr").shift(1)` rather
  than the Python-loop approach in the existing `travel_rest.py`. The
  loop is too slow on the 25-season dataset.
* First-game handling: `days_rest = NaN` (not capped at 7 like the existing
  code), with a separate boolean column `is_first_game_of_season` for
  models that want to treat that case explicitly. Cleaner than encoding two
  meanings ("genuinely 7 days rest" vs "first game") into one column.
* Self-join to attach `opp_*` versions, mirroring the existing `opp_*`
  pattern in the Session 10 schema.
* Pipeline change: `pipeline.process_pair` calls `add_travel_rest` after
  `add_factors`. New columns appended to existing schema (additive change,
  not a rewrite).
* Tests: hand-computed `days_rest` and `traveled` on small 5-game-per-team
  fixture; metro-map handles all known historical cases; Katrina year
  2005-06 NOK location is OKC, not NOLA.

Open decisions: whether to ALSO surface back-to-back flag, road-trip-length,
`prev_game_metro` for diagnostics. Recommendation: minimum viable for
Session 12 (the four core columns), additions in a Session 12.5 if
the prediction layer needs them.

### 10.2 Session 13: Prediction layer (win probability)

Goal: rebuild the existing `win_probability_v3.py` with the same general
approach, on the new clean data foundation, with proper modular structure
and tests.

Approach:

* New top-level module `nba_four_factors/prediction/` with submodules:
  - `features.py` — exp-weighted rolling features over multiple windows,
    cumulative features, capped net rating
  - `outliers.py` — 3-sigma factor caps, composite z-score downweighting
  - `model.py` — train, predict, season projection, calibration
  - `evaluation.py` — full-season MAE, checkpoint MAE, per-season breakdown
* CLI subcommand `predict` with subcommands for train, predict, evaluate.
* Outputs to `data/prediction/`:
  - `game_predictions.parquet` (game-level win probabilities)
  - `season_checkpoints.parquet` (projected wins at game checkpoints)
  - `feature_importance.csv` (coefficient table)
  - `model_config.json` (hyperparameters, caps, training seasons)
* Hyperparameters initially mirror v3: λ_window=0.9, λ_cumul=0.95,
  windows=[5, 10, 20], MARGIN_CAP=25, FACTOR_SIGMA=3.0, COMP_SIGMA=2.0,
  COMP_WEIGHT=0.25, LR_C=0.1.
* Beta weights for composite computation come from the team-season
  regression in Session 11 (replacing the hardcoded `BETA = {...}` from
  v3).

Critical dependency: travel/rest features from Session 12 are required
inputs (`rest_diff`, `travel_adv`, `home_int`, `traveled`, `opp_traveled`).

Open decisions for Session 13:

* Do we include 2025-26 in training, or hold it out as a true future
  validation set?
* Calibration approach: Platt scaling, isotonic regression, or
  uncalibrated? v3 was uncalibrated; we should test calibration impact
  on log-loss and Brier.
* What "feature importance" actually means for the thesis chapter — is
  this a "build a working predictor" session or "extract methodological
  insight from prediction performance" session? Different scope
  implications.

These get resolved in the Session 13 spec.

## 11. Open questions for review

Before lock-in:

1. Is the `data/analysis/` output directory the right location, or should
   regression outputs go somewhere else (e.g. `analysis/output/`)? Defaults
   above; trivial to change.

2. Should the team-season regression use 8 features, or differentials at the
   team-season level (4 features: team's mean off_X minus mean def_X)?
   Different scientific question. Default above is 8 features (where the
   58/42 split, if it exists, would be visible). The 4-feature version is a
   reasonable secondary analysis but not the primary deliverable.

3. CSV column naming: keep `lin_efg_diff` style prefix from existing code,
   or simplify to `efg_diff` and put model type in the filename? Current
   plan keeps the prefix for compatibility with existing outputs.

4. The chart for `team_season_weight_shares.png`: stacked bar (one bar per
   factor, off and def stacks) or grouped bar (4 factor groups, off and def
   side-by-side)? Default: grouped bar, easier to read.

5. Anything specific the thesis chapter needs that's not in the deliverable
   list above?

## 12. Files to be created (preview)

```
nba_four_factors/analysis/__init__.py            ~30 lines
nba_four_factors/analysis/_loading.py            ~50 lines
nba_four_factors/analysis/pivots.py              ~120 lines
nba_four_factors/analysis/regression.py          ~80 lines
nba_four_factors/analysis/differentials.py       ~70 lines
nba_four_factors/analysis/symmetric.py           ~70 lines
nba_four_factors/analysis/team_season.py         ~80 lines
nba_four_factors/analysis/hca.py                 ~40 lines
nba_four_factors/analysis/charts.py              ~250 lines
nba_four_factors/orchestration/analyze.py        ~80 lines
nba_four_factors/cli.py                          ~50 line addition

tests/analysis/conftest.py                       ~150 lines
tests/analysis/test_loading.py                   ~50 lines
tests/analysis/test_pivots.py                    ~120 lines
tests/analysis/test_regression.py                ~80 lines
tests/analysis/test_differentials.py             ~80 lines
tests/analysis/test_symmetric.py                 ~70 lines
tests/analysis/test_team_season.py               ~70 lines
tests/analysis/test_hca.py                       ~50 lines
tests/analysis/test_charts.py                    ~40 lines
tests/analysis/test_cli_analyze.py               ~80 lines

Total: roughly 1,700 lines of production code, 1,100 lines of tests.
```

Bigger than Session 10. Consequence of three regressions plus charting
plus the methodological care (CV, both HCA flavors, antisymmetry tests)
that the thesis chapter needs.
