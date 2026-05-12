# Session 12: Pivots (`nba_four_factors/analysis/pivots.py`)

Status: **Draft, pending kickoff.**

This continues Session 11. Step 1 (loading) is done and merged. This session
covers Step 2: the two pivots that turn the long-format processed
Parquets into the shapes the regression layers want.

## 1. Goal

Build `nba_four_factors/analysis/pivots.py` with two pure functions:

* `pivot_to_game_level(team_game_df)` — one row per game, home and away
  features both present on the row. Used by the differentials regression
  and HCA computation.
* `aggregate_to_team_season(team_game_df)` — one row per (team, season),
  averages of all 8 factors plus wins, games, win_pct, mean_margin. Used
  by the team-season regression that's where the offense vs defense split
  question gets answered.

Plus tests, plus an updated public API in `analysis/__init__.py`.

## 2. Context: where Step 1 left off

Step 1 produced:

* `nba_four_factors/analysis/__init__.py` exporting `load_processed`
* `nba_four_factors/analysis/_loading.py` with `read_processed` and
  `_expand_season_range`
* 12 passing tests in `tests/analysis/test_loading.py`
* Verified against real 10 seasons (2015_16 through 2024_25) of processed
  data. `load_processed` returns a 23,958-row regular-season DataFrame
  and a 1,668-row playoff DataFrame across the full range.

`pivots.py` is the next module in the §5 module structure of the original
Session 11 spec. Reread `docs/Session11.md` §6.2 for the original design,
this handoff updates and refines it based on what we learned from real data.

## 3. Scope

### 3.1 `pivot_to_game_level`

Long-format input: 2 rows per game (one per team). Wide-format output:
1 row per game with home and away features both columns on the row.

**Signature:**

```python
def pivot_to_game_level(
    team_game_df: pd.DataFrame,
    include_neutral: bool = False,
) -> pd.DataFrame:
    """One row per game, home and away features both present."""
```

**Output columns:**

```
game_id, game_date, season, season_type, is_neutral,
home_team_abbr, away_team_abbr,
home_efg, home_tov, home_orb, home_ftr,
away_efg, away_tov, away_orb, away_ftr,
home_margin, home_win
```

(Renames from the underlying schema: `off_efg_pct` becomes `home_efg`/`away_efg`
depending on which row, etc. The "off_" prefix is meaningless once we've
rotated home and away onto a single row, so we drop it.)

**Design notes:**

* `is_neutral=True` rows are excluded by default. "Home advantage" is
  undefined for neutral games. The flag is preserved on the output for
  downstream code that might want to distinguish, but neutrals are
  excluded from the regression by default. Setting `include_neutral=True`
  passes them through. In neutral games, the row marked `is_home=True` in
  the input becomes the "home" side of the output; this is arbitrary but
  consistent with how the upstream `pipeline.py` assigns the flag.
* The merge key is `(game_id, away_team_abbr)`. Using only `game_id` would
  produce a self-join cross product. The `away_team_abbr` constraint forces
  each home row to merge with the specific away row that completes the same
  game.
* `home_margin` is computed from the home team's `pts - opp_pts`.
* `home_win` is `home_margin > 0` cast to int.
* Ties: NBA games can't end in ties, but if one ever appears in the data
  (data error), `home_win` will be 0. Acceptable; the data is wrong, the
  regression will tolerate one bad row.

### 3.2 `aggregate_to_team_season`

Long-format input: many rows per (team, season). Output: one row per
(team, season) with averages of all 8 factors plus win counts and margins.

**Signature:**

```python
def aggregate_to_team_season(
    team_game_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (team, season) with mean factors plus wins and games."""
```

**Output columns:**

```
season, season_type, team_abbr,
games, wins, win_pct, mean_margin,
off_efg, off_tov, off_orb, off_ftr,
def_efg, def_tov, def_orb, def_ftr
```

**Design notes:**

* Caller is responsible for filtering by `season_type` first. The
  aggregation does NOT group by `season_type`; mixing regular and playoff
  rows into the same team-season average would be analytically wrong.
  The function asserts the input has only one unique `season_type` value
  and raises ValueError otherwise. (Friendly error so a misuse fails
  loudly.)
* Averages are unweighted by minutes or possessions; each game contributes
  equally. This matches the "home court advantage" tradition of treating
  each game as one observation. Possession-weighted averages are an
  alternative worth exploring in `extraEDAfun.md` but are NOT the default.
* `wins` is the count of games where `margin > 0`. `win_pct` is
  `wins / games`. The team-season regression in §3.3 of Session 11 uses
  `win_pct` as the target.
* Defunct teams: handled implicitly. Seattle SuperSonics (SEA) had their
  last season in 2007_08 and won't appear in our 2015_16+ data. New
  Orleans franchise relocations (NOH/NOK/NOP) within our range produce
  separate `team_abbr` rows by season — that's correct behavior, since
  the abbr changes ARE the data. (If we wanted franchise-stable rows
  we'd need a franchise mapping, which Session 13's `MetroMap` will provide
  but isn't needed here.)

### 3.3 Public API update

`analysis/__init__.py` adds:

```python
from .pivots import pivot_to_game_level, aggregate_to_team_season
```

`__all__` updated correspondingly.

## 4. Module structure

```
nba_four_factors/analysis/pivots.py            ~120 lines
tests/analysis/test_pivots.py                  ~150 lines
```

`pivots.py` has no I/O. Pure DataFrame in, DataFrame out. Tests use the
`make_processed_tree` factory from `tests/analysis/conftest.py` plus a
new in-memory builder for cases where reading from disk isn't needed.

## 5. Tests

The test file should cover:

### 5.1 `pivot_to_game_level`

* Output row count equals input row count / 2 for non-neutral input
* `home_team_abbr` and `away_team_abbr` correctly assigned
* Neutral-site games excluded by default; row count drops by 1 per
  neutral game when there are any
* `include_neutral=True` retains them
* Hand-computed `home_margin` matches expected on a 2-team fixture
* Hand-computed `home_win` matches (1 if home margin positive, 0 else)
* The merge key prevents cross-game contamination — a synthetic fixture
  with two games' home rows and verifying no row mixes them
* Empty input produces empty output with the right schema

### 5.2 `aggregate_to_team_season`

* 30 teams × N seasons produces 30N rows on the standard fixture
* `wins`, `games`, `win_pct` arithmetic correct on hand-computed fixture
* Factor averages match hand-computed values
* Mixing season types raises ValueError
* Empty input produces empty DataFrame with the right schema
* Defunct/relocated teams produce separate rows per `team_abbr` (no
  silent franchise merging)

## 6. Implementation order

Two tight passes:

1. **Pivot first** — write `pivot_to_game_level` plus its tests; verify
   on real 2024_25 data via REPL. The home/away merge key is the
   subtle bit; nail it before moving on.
2. **Then aggregation** — `aggregate_to_team_season` is mostly groupby +
   agg, simpler logic, but the season_type guard rail is important.

Then update `__init__.py`, run pytest, smoke test against real data.

## 7. Smoke test plan

Once both functions exist:

```bash
uv run python -c "
from nba_four_factors.analysis import (
    load_processed, pivot_to_game_level, aggregate_to_team_season
)
from nba_four_factors.config import SeasonType

df = load_processed(('2015_16', '2024_25'), SeasonType.REGULAR)
print('long:', df.shape)

games = pivot_to_game_level(df)
print('game-level:', games.shape)
print('home win rate:', games['home_win'].mean().round(3))
print('mean home margin:', games['home_margin'].mean().round(3))

ts = aggregate_to_team_season(df)
print('team-season:', ts.shape)
print('avg games/team-season:', ts['games'].mean().round(1))
print('avg win_pct:', ts['win_pct'].mean().round(3))
"
```

Expected:

* `long:` around `(23958, 39)`
* `game-level:` around `(11979, 16)` — half the long count, modulo neutrals
* `home win rate:` around `0.55-0.59` — the headline HCA finding
* `mean home margin:` around `2.0-3.0` — pre-COVID closer to 3, post-COVID closer to 1.8
* `team-season:` around `(300, 14)` — 30 teams × 10 seasons
* `avg games/team-season:` around 78-82 (lower for 2019_20 COVID and 2020_21)
* `avg win_pct:` 0.500 by construction

If those land, pivots are good and Step 2 is done.

## 8. Open questions for review

1. **Should `pivot_to_game_level` also keep the raw `pts` and `opp_pts`?**
   Currently it doesn't, only `home_margin`. Differentials regression
   doesn't need them. HCA doesn't need them. Anything that wants raw points
   can recompute. Default: don't include.

2. **Should `aggregate_to_team_season` also include `mean_margin_at_home`
   and `mean_margin_on_road`?** Useful for any HCA-by-team analysis. Not
   in the Session 11 spec. Probably belongs in `extraEDAfun.md`.

3. **How to handle 2020_21's reduced 72-game schedule and 2019_20's
   COVID-truncated ~970 games?** The aggregation handles them implicitly
   (each team gets fewer games, win_pct is still meaningful). The
   regression will weight all team-seasons equally. If we want
   game-count-weighted regression as a robustness check, that's a
   `team_season.py` decision, not a `pivots.py` decision.

## 9. Look-ahead

After `pivots.py` lands:

* Step 3: `regression.py` (shared machinery — `fit_linear`, `fit_logistic`,
  `standardize`, CV)
* Step 4: `differentials.py` + `symmetric.py` + `team_season.py` (the three
  flavors)
* Step 5: `hca.py` and `charts.py`
* Step 6: orchestrator + CLI

These all build on `_loading.py` and `pivots.py`. They're the analytical
heart of Session 11. With pivots done, the rest is largely well-bounded
modules of regression-machinery-plus-output-tables.

## Pause point (end of weekend, May 11)

### What shipped this session

1. **Step 2: `pivots.py`** committed. Two pure functions
   (`pivot_to_game_level`, `aggregate_to_team_season`) plus tests, plus
   updated public API in `analysis/__init__.py`. Smoke test against real
   data confirmed expected shapes and `home_win_rate` ≈ 0.55-0.59 across
   13 seasons (2012_13-2024_25).

2. **Bubble anomaly fix** (Session 12.5 work, done early). 2019_20 Bubble
   seeding and playoff games (Jul 30 - Oct 11, 2020) were not flagged
   `is_neutral=True` by the processed pipeline because MATCHUP was scheduled
   normally. Added `NEUTRAL_SITE_DATE_OVERRIDES` to `anomalies.py` and an
   `apply_known_neutral_overrides` call in `_tidy_schedule`. Tests added,
   `known_anomalies.md` updated. Re-processed 2019_20 regular and
   playoffs. Verified: 176 post-shutdown team-game rows now correctly
   carry `is_neutral=True`, `is_home=False`.

3. **Historical backfill complete.** Raw data and processed parquets for
   1997_98 through 2025_26 regular season and playoffs. All three V3
   box-score endpoints at 100% completeness (`scripts/check_missing_games.py`
   reports zeros across the board). 68,716+ team-game rows on disk.

4. **EDA notebooks (Day 2 and partial Day 3).**
   - `notebooks/eda_01_hygiene_and_distributions.py`: data hygiene plus
     marginal distributions. Three findings: modernization visible in
     factor trends, single-game margin std rising ~20% over 13 seasons,
     and end-of-game bimodality in margin distribution explained by
     intentional-foul mechanics and no-ties rule.
   - `notebooks/eda_02_hca.py`: per-season HCA. Headline finding:
     2024_25 mean home margin = 1.69 pts, the lowest non-pandemic value
     in the (then-13-season) range. Pre-COVID baseline ≈ 2.7 (not 3.0
     as previously assumed in the field).

### Open work

1. **Re-run `eda_02_hca.py` on the expanded 29-season range.** Just
   change the `load_processed` season-range argument. The thesis-chapter
   centerpiece chart goes from 13 dots to 29.

2. **Section 4 rewrite.** The original "close vs blowout HCA in margin
   units" decomposition is mathematically truncated by the
   `|margin| <= 10` filter. Replace with **home win RATE in close
   games**, which is the unbounded metric that actually carries
   HCA-from-late-game-effects signal. Code patch is in the chat
   history.

3. **Day 4 sidequest selection.** Top candidates per session plan:
   §2.10 (3-point revolution decomposition) and §2.8 (officiating's
   HCA contribution). One sidequest, not two.

4. **Day 5 synthesis.** Update three-bullet summaries in both EDA
   notebooks. Lightweight writeup. Decide what carries into Session 13
   regression spec.

### Session 12.5 candidates (post-EDA, pre-Session 13)

- Bump `DEFAULT_TIMEOUT_SECONDS` in `api/client.py` from 30 to 60.
  Historical V3 box-score calls were slow enough to time out at 30s,
  producing the long backfill saga. This is a one-line fix.
- Include exception class/message in the "game fetch failed" WARNING
  log line. Currently uninformative for debugging.
- Promote `scripts/fill_gaps.py` patterns into orchestration as a
  proper `backfill --gap-fill-only` mode, with checkpoint-aware retry
  of previously-failed game IDs.

### Files added/modified this session

- `nba_four_factors/analysis/pivots.py`
- `nba_four_factors/analysis/__init__.py`
- `nba_four_factors/anomalies.py` (Bubble overrides)
- `nba_four_factors/processed/schedule.py` (anomaly override hook)
- `tests/analysis/test_pivots.py`
- `tests/processed/test_schedule.py` (Bubble fix tests)
- `known_anomalies.md` (Bubble fix documented)
- `notebooks/eda_01_hygiene_and_distributions.py`
- `notebooks/eda_02_hca.py`
- `scripts/check_missing_games.py`
- `scripts/fill_gaps.py`
- `data/processed/` (15 new historical season parquets + 2019_20 re-processed + 2025_26)
- `data/raw/` (15 historical seasons fetched, gaps closed)
