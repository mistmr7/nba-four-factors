# Data Collection: End-to-End Walkthrough

A guided tour of how raw NBA data flows from the public API through the system
and lands as analysis-ready Parquet files. Reads top-to-bottom; each section
builds on the previous one. Companion to `docs/orchestration.md` (which goes
deep on the orchestration policy specifically) and `docs/session_10.md` (which
specifies the processed layer as built).

Audience: future-you, returning to this project after a six-month gap and
needing to remember how it works without re-reading every file.

## 1. Overview

The pipeline has four layers, each with one job:

```
                +---------------------+
                |    NBA Stats API    |   stats.nba.com (public)
                +----------+----------+
                           |
                           v
                +---------------------+
                |    API client       |   nba_four_factors/api/
                |  rate limit, fetch  |
                +----------+----------+
                           |
                           v
                +---------------------+
                |    Storage layer    |   nba_four_factors/storage/
                |  raw JSON on disk   |   data/raw/
                +----------+----------+
                           |
                           v
                +---------------------+
                |  Orchestration      |   nba_four_factors/orchestration/
                |  what to fetch when |   manifests/, data/checkpoints/
                +----------+----------+
                           |
                           v
                +---------------------+
                |  Processed layer    |   nba_four_factors/processed/
                |  factors + Parquet  |   data/processed/
                +---------------------+
```

The split is deliberate. Each layer changes for a different reason:

* The **API client** changes when NBA Stats changes its rate limits, error
  shapes, or response envelopes.
* The **storage layer** changes when we want to move data (e.g. swap local
  disk for S3) or change the on-disk layout.
* The **orchestration layer** changes when we revise *what* we fetch and
  *how we resume* (e.g. circuit breakers, checkpoint format, manifest schema).
* The **processed layer** changes when we revise the analytical schema or
  factor formulas.

These rarely move together. Coupling them in one module would mean every
small change ripples across concerns it doesn't actually touch.

## 2. What we're collecting

Three distinct kinds of data, all from `stats.nba.com`:

### 2.1 Schedule (LeagueGameLog)

Per (season, season_type), one HTTP call returns a JSON envelope listing every
game played, with each row being a (game, team) combination. A full regular
season returns 2460 rows (1230 games multiplied by 2 teams). The
`LeagueGameLog` envelope includes team-level box stats inline:
FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, TOV, PTS.

This is the primary data source. Everything in the four-factors processed
layer comes from it.

### 2.2 Box scores (Traditional, Advanced, Summary v3)

Per game (one HTTP call per game, per endpoint), more granular detail:

* **Traditional v3** gives per-player box stats: minutes, FG made/attempted,
  rebounds split by offensive/defensive, assists, steals, blocks, turnovers,
  fouls, plus_minus, points.
* **Advanced v3** gives pace-adjusted ratings: offensive/defensive rating,
  net rating, true shooting %, assist %, turnover ratio, possessions, pace.
* **Summary v3** gives game metadata: officials, inactives, attendance,
  arena, lead changes, times tied.

These are not used by the four-factors analysis (yet). They're being collected
as a foundation for downstream work: player-level analyses, pace-adjusted
ratings, audit/comparison work between team and player aggregations.

For a 25-season backfill that's roughly 30,000 games, three endpoints each:
~90,000 HTTP calls. The orchestration layer's job is to make that resumable.

### 2.3 Season type taxonomy

Three season types per year:

* `REGULAR` (`Regular Season` in the NBA API parameter; `regular_season` in
  the processed Parquet column values)
* `PLAY_IN` (`PlayIn` in the API; `play_in` in Parquet)
* `PLAYOFFS` (`Playoffs` in the API; `playoffs` in Parquet)

Play-In didn't exist before 2019-20. Calling the API with PlayIn for older
seasons returns an empty result set, not an error.

Naming inconsistency between the API (`Regular Season` with a space, `PlayIn`
without) is real and lives in `config.SeasonType.value`. The processed layer
hides this with a small helper module (`processed/_naming.py`); see Session 10
spec for the full reasoning.

## 3. Layer 1: The API client

`nba_four_factors/api/` contains:

* `client.py` — thin HTTP wrapper around `stats.nba.com`, returns parsed dict
* `rate_limiter.py` — token bucket; throttles to roughly 1 request/second to
  avoid being blocked
* `endpoints/` — one file per endpoint, each exposing a `fetch_<endpoint>`
  function with a typed signature

### 3.1 What the client guarantees

Calling `fetch_league_game_log(client, season, season_type)`:

* Issues an HTTP GET with the right query params (the seasons-and-types
  serialization is endpoint-specific; the function handles it)
* Sleeps if needed to stay under the rate limit
* Retries on transient failures (5xx, certain 4xx, network blips) with
  exponential backoff
* Raises `RetriesExhaustedError` when retries don't recover the call
* Returns the parsed JSON dict on success (no validation; that's a higher
  layer's concern)

The client deliberately does NOT:

* Decide *whether* to fetch (orchestration's job)
* Save anything to disk (storage's job)
* Interpret the payload schema (processed layer's job)

### 3.2 Why rate limiting matters

`stats.nba.com` will silently start returning empty rowsets, weird HTTP
codes, or just timing out if you fire too many requests too fast. There's
no documented rate limit. ~1 req/sec has been stable in practice. The
backfill of 90k+ box scores will take roughly a day at this rate, which
sounds slow but is the actual constraint.

The token bucket lives in `rate_limiter.py`. It's process-wide, so even if
the orchestration layer parallelized requests within a process, the limiter
would serialize them. We do not parallelize across processes; running two
backfills simultaneously would defeat the limiter.

### 3.3 Error model

* Recoverable (retry): network errors, 5xx responses, 429s
* Unrecoverable (raise to orchestration): invalid season strings, malformed
  responses that survive parsing, exhausted retries
* The orchestration drivers catch `RetriesExhaustedError` and record the
  failure in the checkpoint; they don't catch other exceptions

## 4. Layer 2: Storage

`nba_four_factors/storage/`:

* `raw.py` — path construction and JSON read/write for raw payloads
* `manifest.py` — per-pair manifest (what was fetched, when, what's missing)
* `checkpoint.py` — resumable progress state for in-flight backfills

### 4.1 On-disk layout

```
data/
  raw/
    leaguegamelog/
      2024_25/
        regular_season.json
        play_in.json
        playoffs.json
      2025_26/
        regular_season.json
    boxscoretraditionalv3/
      0022400001.json
      0022400002.json
      ...
    boxscoreadvancedv3/
      0022400001.json
      ...
    boxscoresummaryv3/
      0022400001.json
      ...
  checkpoints/
    2024_25_regular_season.json
    ...
manifests/
  2024_25_regular_season.json
  ...
```

* Schedule files: per (season, season_type)
* Box score files: per (endpoint, game_id), 10-character zero-padded ID
* Checkpoints: per (season, season_type), accumulate during a backfill,
  cleared once the manifest is written
* Manifests: per (season, season_type), the durable record of what got
  fetched and what didn't

`data/raw/` and `data/processed/` are gitignored. `manifests/` is committed.
This is intentional: raw JSON and derived Parquet can always be recreated
from the API; manifests document what was successfully fetched and when, which
matters for reproducibility but is small enough to commit.

### 4.2 Why JSON for raw

Three reasons:

1. **Identity to API**: storing exactly what the API returned means we can
   re-derive everything downstream from on-disk data without re-hitting the
   API. If a future analysis needs a column we ignored today, it's already
   there.
2. **Diffability**: text-based format means `git diff` (during development)
   and `jq` exploration both work.
3. **Schema flexibility**: nba_api's V3 endpoints have a hierarchical
   envelope that doesn't flatten cleanly to a single tabular schema; storing
   JSON defers that decision to the processed layer.

Raw files are not the smallest possible representation. A full season of
schedule data is ~390KB. 30k box scores at ~50KB each is ~1.5GB per endpoint.
The whole raw layer for a 25-season backfill is around ~5GB. Acceptable for
local development and small-cluster work.

### 4.3 Manifests

Each manifest is a JSON document (`manifests/<season>_<season_type>.json`)
with this minimal schema:

```json
{
  "season": "2024-25",
  "season_type": "Regular Season",
  "pulled_at": "2026-04-29T18:43:12Z",
  "expected_game_count": 1230,
  "actual_game_count": 1228,
  "game_ids": ["0022400001", "0022400002", ...],
  "missing_or_failed": ["0022400845", "0022401123"]
}
```

* `season` and `season_type` use the human form (matches `SeasonType.value`)
  for legibility
* `expected_game_count` is the number of distinct game IDs in the schedule
* `actual_game_count` is games where ALL THREE box-score endpoints succeeded
  (intersection, not union — a game with traditional and advanced but missing
  summary isn't fully ingested)
* `missing_or_failed` is the diff. Re-running the same backfill picks up
  exactly these games

The manifest is written once at the END of a (season, season_type) pair.
While a backfill is mid-flight, ground truth lives in the checkpoint file,
not the manifest.

### 4.4 Checkpoints

Per-pair JSON file at `data/checkpoints/<season>_<season_type>.json`. Tracks
in-progress state:

```json
{
  "schedule_done": true,
  "box_scores": {
    "boxscoretraditionalv3": {
      "completed": ["0022400001", "0022400002", ...],
      "failed": ["0022400845"]
    },
    "boxscoreadvancedv3": {
      "completed": [...],
      "failed": [...]
    },
    "boxscoresummaryv3": {
      "completed": [...],
      "failed": [...]
    }
  }
}
```

Updated incrementally as games complete. If the process dies (network drop,
power outage, deliberate Ctrl-C), the next run reads this checkpoint and
skips everything in `completed`, retrying everything in `failed`.

Once the manifest is written for the pair, the checkpoint can be deleted.
In practice it's left in place so re-running the backfill is a fast no-op.

## 5. Layer 3: Orchestration

`nba_four_factors/orchestration/`:

* `_io.py` — package-internal storage-binding shim. Composes the narrow verbs
  in the API and storage layers into the four operations the policy layer
  actually needs. Tests fake this single surface instead of patching across
  endpoints and storage.
* `_common.py` — shared helpers for both drivers (`is_pair_done`, season
  range expansion, checkpoint loaders)
* `historical.py` — backfill driver: one-shot historical sweep over a season
  range, full resume semantics, circuit breakers
* `incremental.py` — nightly delta driver: re-fetch schedule for one season,
  pull any pending box scores
* `process.py` — processed-layer driver (Session 10): one (season,
  season_type) pair → one Parquet file

### 5.1 Backfill (`historical.py`)

The big one. Used to populate the raw layer from cold. Processes (season,
season_type) pairs sequentially. For each pair:

1. Fetch schedule (one API call, populates `data/raw/leaguegamelog/...`)
2. Load schedule's game IDs from disk
3. For each box-score endpoint, for each game, fetch and save
4. Update the per-pair checkpoint after each game
5. Once the pair is complete, write the manifest, optionally clear the
   checkpoint

Resume semantics:

* Re-running the same backfill skips pairs already manifested
* Within a pair, re-running skips games already in `completed`, retries
  games in `failed`
* If `--max-consecutive-failures N` is hit, the driver stops mid-pair to
  avoid hammering the API during an outage; checkpoint preserves progress

Defaults: newest-first season order (so a half-finished backfill at least
has the recent stuff), all season types, all four endpoints. Override with
`--season-range`, `--season-type`, `--endpoint`, `--reverse`.

### 5.2 Incremental (`incremental.py`)

The nightly driver. Used while a season is in progress. For one season:

1. Re-fetch the schedule (overwrites the existing JSON if any)
2. Diff: schedule game IDs minus what we have in `completed` for each box
   endpoint
3. Fetch the missing ones
4. Update checkpoint and (if no failures) manifest

Idempotent. Running it twice in a row is a no-op the second time. Designed
to be cheap and safe enough for an unattended cron job.

Why a separate driver instead of a flag on backfill: backfill's resume and
circuit-breaker logic are tuned for sweeping ~90k games; incremental's
job is "what changed since yesterday." Different defaults, different
optimal failure modes, easier to reason about as separate scripts.

### 5.3 Process (`process.py`, Session 10)

Local-disk only, no API surface. Per (season, season_type) pair:

1. Read raw schedule JSON
2. Tidy into one row per (game, team)
3. Self-join to attach opponent stats
4. Compute four factors and margin
5. Write Parquet to `data/processed/<season>/<season_type>.parquet`

Idempotent and deterministic (sort + write produces byte-identical output
across runs). No checkpoint needed — runs in seconds. See `docs/session_10.md`
for the full spec.

### 5.4 Why three drivers, not one

Could be one CLI command with mode flags. Three reasons we kept them
separate:

* Different failure semantics (backfill needs circuit breakers, incremental
  doesn't, process can't fail in ways that need recovery)
* Different defaults (backfill wants newest-first across all endpoints;
  incremental wants today's season only; process is single-pair)
* Easier to test in isolation

## 6. Layer 4: Processed (Session 10)

`nba_four_factors/processed/` is fully covered by `docs/session_10.md`.
Brief recap for completeness:

* `_naming.py` — `season_type_to_snake` and `snake_to_human` helpers
* `schedule.py` — `_tidy_schedule` (pure transformation) and `read_schedule`
  (I/O wrapper). The tidy function is module-internal but importable for
  testing.
* `factors.py` — `add_factors(df) → df` adds the eight Oliver factors plus
  margin. Pure, no side effects.
* `pipeline.py` — `process_pair(season, season_type) → Path` orchestrates
  load → tidy → self-join → factors → write.

Output schema is locked at 39 columns, one row per (game, team), 2460 rows
per full regular season. See `docs/session_10.md` §2.3 for the column list.

## 7. The user-facing CLI

One Python module entry point:

```bash
uv run python -m nba_four_factors.cli <subcommand> [options]
```

Three subcommands.

### 7.1 backfill

Sweep historical data into the raw layer.

```bash
uv run python -m nba_four_factors.cli backfill --help
```

Flags:

* `--season-range YYYY_YY:YYYY_YY` (default: all seasons)
* `--reverse` — chronological order; default is newest-first
* `--season-type regular play_in playoffs` (default: all three; space-separated list)
* `--endpoint schedule traditional advanced summary` (default: all four)
* `--dry-run` — print plan, no API calls
* `--max-consecutive-failures N` — circuit breaker, default 10

Common patterns:

```bash
uv run python -m nba_four_factors.cli backfill --season-range 2024_25:2024_25 --endpoint schedule
```

```bash
uv run python -m nba_four_factors.cli backfill --season-range 2020_21:2024_25 --season-type regular playoffs --endpoint schedule
```

```bash
uv run python -m nba_four_factors.cli backfill --season-range 2000_01:2024_25
```

The first command is fast (one API call). The second is fast for the
schedule-only subset (a few dozen calls). The third is the full sweep,
roughly a day.

### 7.2 incremental

Nightly delta pull. Defaults to current season.

```bash
uv run python -m nba_four_factors.cli incremental
```

```bash
uv run python -m nba_four_factors.cli incremental --season 2025_26 --dry-run
```

Cron-friendly. Re-fetches schedule, picks up new games, retries previously
failed ones.

### 7.3 process

Build the processed Parquet for one (season, season_type) pair.

```bash
uv run python -m nba_four_factors.cli process --season 2024_25 --season-type regular
```

No range argument by design (Session 10 §6). For bulk processing, loop in
the shell:

```bash
for season in 2020_21 2021_22 2022_23 2023_24 2024_25; do
    for st in regular playoffs; do
        uv run python -m nba_four_factors.cli process --season "$season" --season-type "$st"
    done
done
```

Each call takes single-digit seconds. The whole loop runs in under a minute.

## 8. Worked example: from cold start to processed Parquet

The end-to-end flow for a single season pair, executed step by step.
Useful as a sanity check after a fresh checkout, or as a recovery procedure
after `data/raw/` was nuked.

### Step 1: install

```bash
git clone <repo>
cd nba-four-factors
uv sync
```

### Step 2: fetch the schedule

```bash
uv run python -m nba_four_factors.cli backfill --season-range 2024_25:2024_25 --endpoint schedule
```

What happens:

* Driver computes the work plan: one pair (2024_25 × regular), one endpoint
  (schedule), one API call
* Rate limiter allows the call immediately (token bucket starts full)
* `fetch_league_game_log` issues the GET, retries if needed, returns a dict
* `save_raw` writes `data/raw/leaguegamelog/2024_25/regular_season.json`
* Checkpoint updated: `schedule_done = true`
* Since no box scores were requested, the manifest's `actual_game_count`
  will be 0, but the schedule JSON is on disk

You can confirm:

```bash
ls -la data/raw/leaguegamelog/2024_25/
```

```bash
jq '.resultSets[0].rowSet | length' data/raw/leaguegamelog/2024_25/regular_season.json
```

Expected: 2460 (1230 games × 2 team rows).

### Step 3: process to Parquet

```bash
uv run python -m nba_four_factors.cli process --season 2024_25 --season-type regular
```

What happens (no API calls, all local):

* `read_schedule` loads the JSON, calls `_tidy_schedule` to flatten the
  envelope into a per-(game, team) DataFrame
* `_attach_opponent` self-joins on `game_id` to add `opp_*` columns
* `add_factors` computes the eight factors plus margin
* `to_parquet` writes the result

Sanity check:

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/processed/2024_25/regular_season.parquet')
print('rows:', len(df))
print('cols:', len(df.columns))
print('margin sum:', df['margin'].sum())
"
```

Expected: 2460 rows, 39 columns, margin sum exactly 0.

### Step 4: bulk processing (the steady state)

Once schedule has been backfilled for many seasons, processing them all
takes about a minute:

```bash
for season in 2000_01 2001_02 2002_03 2003_04 2004_05 2005_06 2006_07 \
              2007_08 2008_09 2009_10 2010_11 2011_12 2012_13 2013_14 \
              2014_15 2015_16 2016_17 2017_18 2018_19 2019_20 2020_21 \
              2021_22 2022_23 2023_24 2024_25; do
    for st in regular playoffs; do
        uv run python -m nba_four_factors.cli process --season "$season" --season-type "$st"
    done
done
```

Skips `play_in` for pre-2019-20 seasons (the JSON file would not exist; the
driver prints a friendly hint and exits 1, which the loop ignores).

## 9. Recovery procedures

### 9.1 "I deleted data/raw/ by accident"

Re-run the backfill. The orchestration layer will refetch everything that's
not already on disk.

```bash
uv run python -m nba_four_factors.cli backfill
```

This is a long operation. ~24 hours for the full 25-season box-score
backfill at 1 req/sec. Schedule-only is fast (~5 minutes for everything).

### 9.2 "A backfill died mid-flight"

Re-run the same command. The checkpoint files in `data/checkpoints/` will
be picked up automatically. Pairs already manifested are skipped; pairs in
flight resume where they left off.

```bash
uv run python -m nba_four_factors.cli backfill --season-range 2020_21:2024_25
```

If a particular game has been failing repeatedly, the manifest will list it
in `missing_or_failed`. To force a retry of just that game, edit the
checkpoint to remove it from the `failed` list (so the driver re-tries).
This is a manual recovery procedure; nothing automates it because the
common case is "bad data permanently" (game cancelled/voided/forfeited)
which the manifest correctly reports.

### 9.3 "I want to refetch a single (season, season_type) from scratch"

```bash
rm -rf data/raw/leaguegamelog/2024_25/regular_season.json
rm -f data/checkpoints/2024_25_regular_season.json
rm -f manifests/2024_25_regular_season.json
uv run python -m nba_four_factors.cli backfill --season-range 2024_25:2024_25 --season-type regular
```

### 9.4 "I want to wipe data/processed/ and rebuild"

Just rerun the process loop. The processed layer is a pure function of
`data/raw/`; deleting `data/processed/` loses nothing.

```bash
rm -rf data/processed/
```

Then the bulk processing loop from §8.4.

## 10. Known anomalies

See `known_anomalies.md` for the live list. Categories you should expect:

* **Pre-2003 advanced box scores**: the `boxscoreadvancedv3` endpoint
  returns sparse or empty data for some pre-2002 games. Manifests record
  these as failures; downstream analyses must handle missing rows.
* **2019-20 bubble games**: scheduled at neutral sites in Orlando.
  `MATCHUP` may show `vs.` or `@` inconsistently across game halves; the
  processed layer's `is_neutral` flag catches it but downstream analyses
  must be careful with HCA computation for that season.
* **NBA Cup neutral-site games (2023-24+)**: ~5 games per season at Las
  Vegas. Both rows have `@`, both are flagged `is_neutral=True,
  is_home=False`.
* **2020-21 shortened season**: 72-game season, not 82. Total team-game
  rows = 2160, not 2460. Valid; just different.
* **Voided / forfeited games**: rare. Tend to surface as "game in schedule
  but no box score available." Permanent failures, recorded in
  `missing_or_failed`.

## 11. What's NOT in scope (yet)

These are conscious omissions, slated for later sessions:

* **Player-level box scores from `boxscoretraditionalv3`** — the JSON is
  collected but not surfaced in the processed Parquet. Per-player analyses
  will be a separate processed-layer extension (probably Session 13ish).
* **Pace-adjusted ratings from `boxscoreadvancedv3`** — collected but not
  surfaced. Adding to the processed layer requires deciding on join keys
  (game-team granularity exists; offensive/defensive rating already roughly
  derivable from the four factors plus pace).
* **Officials, attendance, lead changes from `boxscoresummaryv3`** —
  collected but unused. Most likely useful for narrative chapters in the
  thesis, not the four-factors regression.
* **Travel and rest features** — `days_rest`, `traveled`, `opp_*` versions.
  Slated for Session 12. The current processed Parquet does not include
  these; the prediction layer (Session 13) needs them.
* **Combined master Parquet** — concatenation of all per-(season,
  season_type) files into one. Convenient for downstream analyses;
  deferred until something explicitly benefits from it.
* **Cross-endpoint reconciliation** — verifying that fields appearing in
  multiple endpoints (FGM, FGA, etc. show up in LeagueGameLog AND
  boxscoretraditionalv3) actually agree. Useful as data-quality scaffolding;
  no current usage demands it. Slated for later session.

## 12. Files referenced by section

| Section | Relevant files |
|---|---|
| 3 (API client) | `nba_four_factors/api/client.py`, `rate_limiter.py`, `endpoints/*.py` |
| 4 (Storage) | `nba_four_factors/storage/raw.py`, `manifest.py`, `checkpoint.py` |
| 5 (Orchestration) | `nba_four_factors/orchestration/_io.py`, `_common.py`, `historical.py`, `incremental.py`, `process.py` |
| 6 (Processed) | `nba_four_factors/processed/__init__.py`, `_naming.py`, `schedule.py`, `factors.py`, `pipeline.py` |
| 7 (CLI) | `nba_four_factors/cli.py` |
| 9 (Recovery) | `data/checkpoints/`, `manifests/` |

## 13. Glossary

* **Pair** — shorthand for `(season, season_type)`. The unit of work for
  backfill and process.
* **Game ID** — 10-character zero-padded string, e.g. `"0022400001"`. The
  leading `00` indicates regular season; `004` indicates playoffs; `005`
  indicates play-in. Treat as opaque strings, never integers.
* **Endpoint** — one of the four NBA stats API surfaces we consume:
  schedule (LeagueGameLog), boxscoretraditionalv3, boxscoreadvancedv3,
  boxscoresummaryv3.
* **Snake form** vs **human form** — `regular_season` vs `Regular Season`.
  Snake form lives in Parquet column values and file paths; human form
  lives in manifests and matches the NBA API parameter strings. Convert
  with `processed.season_type_to_snake` and `snake_to_human`.
* **Backfill** — historical sweep populating `data/raw/` from cold. Long
  operation, full resume semantics.
* **Incremental** — nightly delta pull for the current season. Short, safe,
  cron-friendly.
* **Process** — local-disk operation: raw JSON → processed Parquet. No API.
* **Manifest** — per-pair JSON record of what was fetched and what wasn't.
  Source of truth for "is this pair done."
* **Checkpoint** — per-pair in-flight progress state during a backfill.
  Cleared (or stale-but-harmless) once the manifest is written.
