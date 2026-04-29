# Session 10 — Processed layer (raw → four-factors)

Status: **Draft, pending review.** Locked sections marked.

## 1. Goal

Read raw `LeagueGameLog` JSON files from disk and produce per-(season,
season_type) Parquet files containing one row per (game, team) with Dean
Oliver's four factors computed for both the team and its opponent.

This wraps up "data collection." After Session 10:

* Raw layer (Sessions 4–9) → fetches and persists JSON from the API
* **Processed layer (Session 10) → tidies raw JSON into analysis-ready Parquet**
* Validation (Session 11) → spot-checks data quality
* EDA (Session 12) → first analytical use of the processed data

## 2. Scope (locked)

### 2.1 Endpoints used

`LeagueGameLog` only.

`LeagueGameLog` already contains every box-score field the four-factor
formulas need (FGM, FGA, FG3M, FTA, OREB, DREB, TOV, PTS) at team granularity.
The box-score endpoints (Traditional/Advanced/Summary) provide additional
detail (per-player breakdowns, pace-adjusted ratings, officials/inactives)
that this session does not need. They will be integrated into the processed
layer in Session 13 as a *schema extension* — additional columns on the same
table, not a parallel table.

### 2.2 Output target

```
data/processed/
    2024_25/
        regular_season.parquet
        playoffs.parquet
        play_in.parquet            # only for seasons >= 2020_21
    2025_26/
        regular_season.parquet
        ...
```

* Per (season, season_type) file
* Underscore convention preserved (`2024_25` not `2024-25`)
* `data/processed/` is gitignored (derived from raw)

### 2.3 Schema

One row per (game, team). 2460 rows for a full regular season (1230 games × 2 teams).

| Column | Type | Source | Notes |
|---|---|---|---|
| `game_id` | str | LeagueGameLog GAME_ID | 10-char zero-padded |
| `game_date` | date | LeagueGameLog GAME_DATE | parsed to date type |
| `season` | str | function arg | underscore form, `2024_25` |
| `season_type` | str | function arg | `regular_season`, `playoffs`, `play_in` |
| `team_id` | int | LeagueGameLog TEAM_ID | NBA team ID |
| `team_abbr` | str | LeagueGameLog TEAM_ABBREVIATION | `LAL`, `BOS`, etc. |
| `opp_team_id` | int | self-join | opponent's TEAM_ID |
| `opp_abbr` | str | self-join | opponent's abbreviation |
| `is_home` | bool | derived from MATCHUP | `vs.` = home, `@` = away |
| `fgm`, `fga`, `fg3m`, `fg3a`, `ftm`, `fta`, `oreb`, `dreb`, `tov`, `pts` | int | LeagueGameLog | team's box stats |
| `opp_fgm`, `opp_fga`, ..., `opp_pts` | int | self-join | opponent's box stats (full set) |
| `off_efg_pct` | float | computed | team's effective FG% |
| `off_tov_pct` | float | computed | team's turnover rate |
| `off_orb_pct` | float | computed | team's offensive rebound % |
| `off_ft_rate` | float | computed | team's free-throw rate (FTA/FGA) |
| `def_efg_pct` | float | computed | opponent's eFG% against team |
| `def_tov_pct` | float | computed | opponent's TOV% against team |
| `def_orb_pct` | float | computed | opponent's ORB% against team |
| `def_ft_rate` | float | computed | opponent's FT rate against team |
| `margin` | int | computed | `pts - opp_pts` |

All column names are snake_case.

### 2.4 Four factors — formulas (locked)

Per Dean Oliver, *Basketball on Paper* (Brassey's, 2004):

```
eFG%      = (FGM + 0.5 * FG3M) / FGA
TOV%      = TOV / (FGA + 0.44 * FTA + TOV)
ORB%      = OREB / (OREB + opp_DREB)
FT Rate   = FTA / FGA
```

**FT Rate uses FTA, not FTM.** Oliver's original specifies attempts; the
intent is to measure how often a team gets to the line, not how often they
convert. (Conversion is captured separately by FT%.) This matches the old
project's convention.

Defensive factors are computed as the opponent's offensive factors *against
this team*:

```
def_eFG%     = (opp_FGM + 0.5 * opp_FG3M) / opp_FGA
def_TOV%     = opp_TOV / (opp_FGA + 0.44 * opp_FTA + opp_TOV)
def_ORB%     = opp_OREB / (opp_OREB + DREB)
def_FT_Rate  = opp_FTA / opp_FGA
```

Division-by-zero protection: any denominator of zero yields `NaN`, not an
error. (A team with zero field-goal attempts is impossible in a real game,
but we don't want validation runs to crash on edge-case test data.)

### 2.5 What's deferred

* Travel/rest computation (days_rest, traveled flags) — Session 14 (features)
* Cross-endpoint reconciliation — Session 13
* Player-level box scores — Session 13
* Pace-adjusted ratings (ORtg, DRtg, NetRtg) — Session 13
* Combined master Parquet across all seasons — derivable from per-season
  files via `pd.concat`, not worth maintaining as a separate artifact

## 3. Module layout

```
nba_four_factors/processed/
    __init__.py
    schedule.py        # JSON → tidy team-game DataFrame (no joins yet)
    factors.py         # add the 8 four-factor columns + margin
    pipeline.py        # orchestrate: schedule → join → factors → write
nba_four_factors/cli.py
    # New subcommand: `process --season YYYY_YY --season-type regular`
```

Four files, each with a single responsibility:

* **`schedule.py`** — Read a saved `LeagueGameLog` JSON and return a tidy
  per-(team, game) DataFrame. Pure I/O + envelope unwrapping.
* **`factors.py`** — Given a DataFrame with team and opponent box columns,
  add the 8 four-factor columns + margin. Pure computation, no I/O.
* **`pipeline.py`** — Compose schedule + self-join + factors + write. The
  one place that knows the full pipeline shape.
* **CLI subcommand** — Expose pipeline as `process` on the existing CLI.
  Same argparse pattern as `backfill` and `incremental`.

### 3.1 KISS / DRY checks

**Could `schedule.py` and `factors.py` merge?** No. Schedule is I/O,
factors are computation. Merging entangles testability — you couldn't unit
test the formulas without disk access.

**Could `pipeline.py` merge into the CLI?** No. CLI is argparse plumbing;
pipeline is the data flow. Keeping them separate means you can call
`pipeline.process_pair()` from a notebook without going through argparse.

**Are we duplicating `_io.load_schedule_game_ids` from the orchestration
layer?** Partially. That function returns `list[str]` of game IDs;
`schedule.py` returns the full DataFrame. The DataFrame is a strict
superset, so `_io.load_schedule_game_ids` could be re-implemented as
`schedule.read_schedule_dataframe(...)["game_id"].unique().tolist()`.
**Decision: leave both as-is for now.** They serve different layers
(orchestration vs processed), and reusing across layer boundaries
introduces coupling that's worse than the duplication. Revisit if a third
caller appears.

## 4. Pipeline flow

For each `(season, season_type)` pair:

```
1. Read JSON
   data/raw/leaguegamelog/{season}_{season_type}.json
   → unwrap resultSets[0].rowSet
   → 2460 rows, snake_case columns
                  ↓
2. Self-join on game_id
   → still 2460 rows
   → each row now has both team's and opponent's box stats
                  ↓
3. Compute four factors
   → +9 columns (8 factors + margin)
                  ↓
4. Write parquet
   data/processed/{season}/{season_type}.parquet
```

All four steps are deterministic and idempotent. Running `process --season
2024_25 --season-type regular` twice produces byte-identical output.

## 5. Failure modes

| Failure | Behavior |
|---|---|
| Source JSON missing | Raise `FileNotFoundError` immediately. No silent skip. |
| Source JSON malformed (no `resultSets`, missing columns) | Raise `ValueError` with a clear message. |
| Self-join produces unexpected row count (≠ 2 × game count) | Raise `AssertionError`. Indicates schedule data corruption. |
| Division-by-zero in factor formulas | Return `NaN` for that cell. No crash. |
| Output directory doesn't exist | Create it (mkdir parents=True). |
| Output file exists | Overwrite. Idempotency requires it. |

The CLI subcommand handles `FileNotFoundError` with a clear message about
running `backfill` first, and exits non-zero. Other exceptions propagate
with full traceback (these are bugs, not user error).

## 6. CLI

```
uv run python -m nba_four_factors.cli process \
    --season YYYY_YY \
    --season-type regular|playoffs|play_in
```

* `--season` required, single season only (no `--season-range` like backfill)
* `--season-type` required
* No `--dry-run` (operations are local-disk only, ~seconds, no API calls)
* No `--reverse` (single pair, no ordering concern)

Bulk processing across many seasons is out of scope; if needed, shell loop:

```bash
for season in 2020_21 2021_22 2022_23 2023_24 2024_25; do
    for st in regular_season playoffs; do
        uv run python -m nba_four_factors.cli process \
            --season "$season" --season-type "$st"
    done
done
```

## 7. Tests

### 7.1 `factors.py` — hand-computed expected output

One test fixture: a single game with hand-computed expected values for all
8 factors and margin. Written from the formulas, not from the
implementation. If the implementation drifts, this test catches it.

Example (illustrative, exact numbers TBD):

```
Team:     FGM=40, FGA=85, FG3M=10, FTA=20, OREB=12, DREB=35, TOV=14, PTS=110
Opp:      FGM=38, FGA=88, FG3M=8,  FTA=18, OREB=10, DREB=33, TOV=16, PTS=100

eFG%   = (40 + 5) / 85       = 0.5294
TOV%   = 14 / (85 + 8.8 + 14) = 0.1300
ORB%   = 12 / (12 + 33)      = 0.2667
FT_R   = 20 / 85             = 0.2353
margin = 110 - 100           = 10
```

Plus edge cases:

* All zeros → all NaN, no crash
* Only one team has stats → defensive side correctly populates from opponent

### 7.2 `schedule.py` — hand-crafted JSON fixture

Small JSON fixture mimicking the `LeagueGameLog` envelope, with 4 rows
(2 games, 2 teams each). Tests:

* Correct row count
* All required columns present and snake_case
* `is_home` correctly derived from MATCHUP string
* Date parsing yields actual `date` objects, not strings

### 7.3 `pipeline.py` — end-to-end with fixture data

One integration-style test that runs the whole pipeline against the fixture
JSON, writes a tmp Parquet, reads it back, asserts schema and row count.

### 7.4 CLI subcommand tests

Mirror the existing CLI test pattern: argparse parsing, dispatch routing,
error paths (missing required args, invalid season_type).

## 8. Open questions

None. (Filled in during pre-spec discussion.)

## 9. TODO

* Hand-compute the test fixture values for `factors.py` (need to do this
  before writing the test).
* Verify that the `LeagueGameLog` JSON for 2024-25 actually contains all
  10 expected box columns. (We assume this from the old project's success
  with the same endpoint, but worth confirming on real data before writing
  `schedule.py`.)
