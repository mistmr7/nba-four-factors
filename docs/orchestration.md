# NBA Four Factors — Orchestration Spec

**Status:** Locked, Session 8 (2026-04-28).
**Scope:** Orchestration layer — historical backfill driver and incremental update driver. Covers iteration model, flush cadence, failure handling, "done" definition, CLI surface, and module layout.
**Out of scope (deferred to Session 9+):** Logging and observability format, implementation code, processed-layer normalization, manifest content schema details.
**Companion documents:** Read alongside the Session 6 and Session 7→8 handoff documents for upstream context (checkpoint module API, V3-only architecture, raw/manifest storage conventions).

---

## 1. Background

The `storage/checkpoint.py` module from Session 7 is the in-memory state machine that tracks ingestion progress per `(season, season_type)` tuple. The orchestrator is the policy layer that drives that state machine: it decides which tuples to process, in what order, with what failure semantics, and when to flush state to disk.

There are two drivers because their use cases differ in shape:

- **Historical backfill** is a one-shot sweep over the full date range (1997-98 through 2025-26 by default). The total work is approximately 321,000 HTTP calls (~107k games × 3 box-score endpoints, plus ~90 schedule calls). At a 30 calls/min rate limit, the wall-clock floor is ~178 hours, realistically 250+ hours including retries and overhead. The run will span multiple sessions and machine restarts. **Resumability is the dominant design constraint.**
- **Incremental update** is a low-overhead nightly-cron-style run that pulls new games from the current season since the last update. The total work is small (a handful of games per night during the season, zero during the offseason). **Fast no-op-when-nothing-new is the dominant design constraint.**

---

## 2. Architecture

```
CLI (cli.py)
  └── argparse subcommands: backfill, incremental
      └── dispatches to:
          orchestration/
          ├── historical.py    (backfill driver)
          ├── incremental.py   (nightly driver)
          └── _common.py       (shared helpers: stripe iteration, circuit breaker, dry-run printer)

Orchestration consumes existing layers without duplication:
  ├── api/client.py + api/rate_limiter.py    (HTTP + 30/min sliding window + 5× exp backoff)
  ├── api/endpoints/{schedule,box_score_*}.py (V3 wrappers)
  ├── storage/raw.py                         (per-game + per-season JSON persistence)
  ├── storage/manifest.py                    (committed proof-of-pull artifacts)
  └── storage/checkpoint.py                  (in-memory progress, flushed to data/checkpoints/)
```

The orchestrator owns flush cadence (β I/O model, locked Session 7); checkpoint mutators are pure-functional on the in-memory dict. The orchestrator never duplicates rate-limiting or retry logic — it consumes those from `api/rate_limiter.py` as-is.

---

## 3. Iteration Model

### 3.1 Endpoint-stripe order (within a `(season, season_type)`)

**Decision: Striped (endpoint-major).** Schedule first, then traditional, then advanced, then summary.

```
for endpoint in [SCHEDULE, BOXSCORE_TRADITIONAL, BOXSCORE_ADVANCED, BOXSCORE_SUMMARY]:
    for game_id in pending_games_for(endpoint):
        fetch(endpoint, game_id)
```

Schedule is mandatory-first because game IDs come from it; box-score endpoints can't iterate without the schedule's game list. Within the box-score stripes, **traditional → advanced → summary** is the locked order: traditional carries the four-factor inputs (the canonical computation source), so a mid-stripe crash leaves the most-useful endpoint complete and the auxiliary endpoints empty.

**Rationale for striped over interleaved (game-major):**

- *Retry efficiency on transient failures.* A network blip during one stripe flags game IDs in `failed[that_endpoint]` only; on resume, only those re-fetch. Interleaved would mark the same game failed across all three endpoints simultaneously, forcing redundant re-fetches when only one stripe was actually affected.
- *Clean partial-run shapes.* `is_endpoint_complete(...)` per-endpoint maps 1:1 to actual stripe completion status. No per-game ambiguity until the stripe finishes.
- *Aligned with one-shot use case.* The historical backfill is meant to terminate, not iterate. Optimizing for "you can do exploratory work mid-backfill" is unnecessary because EDA and processed-layer work come *after* the raw layer is done.

The rejected alternative (interleaved / game-major) would have meant fetching all three box-score endpoints for game N before moving to game N+1. Its main advantage was mid-backfill usability (every completed game is fully populated across all endpoints), but the use case for that doesn't apply here.

**Iteration order is local to `historical.py`.** `incremental.py` revisits the choice — the per-night loop is small enough that interleaved-vs-striped is effectively a wash, and incremental is not the place to optimize.

### 3.2 Within-season order (across season_types)

**Decision: regular_season → play_in → playoffs.**

This matches the actual NBA season sequence: the play-in tournament occurs at the end of the regular season and seeds the final 4 playoff slots.

**Play-in skipping.** For seasons before the play-in tournament was instituted, `play_in` is skipped entirely — no checkpoint file, no schedule fetch attempted, no entry in the targeted set. Skip threshold is `PLAY_IN_FIRST_SEASON` (see TODO §12.1).

### 3.3 Cross-season order (across seasons in the targeted range)

**Decision: User-specifiable via `--reverse` flag. Default is newest-first.**

Newest-first defaults expose the most-likely-needed data first — current-season games are what any sanity check or smoke test would reach for during/after a partial run. Newest-first also surfaces V3-era-specific bugs before sinking many hours of API budget into 1997-98 data.

The `--reverse` flag flips to chronological. Chronological is occasionally useful (e.g., to surface 1990s-era data anomalies first), so the override exists. Naming is mildly awkward — `--reverse` flips the *default*, which is itself reverse-of-natural — but the alternatives (`--chronological`, `--oldest-first`) are more verbose. Locked as `--reverse`.

---

## 4. Flush Cadence

**Decision: Flush after every successful checkpoint mutation. No override.**

Every successful `mark_game_complete`, `mark_game_failed`, `mark_schedule_complete`, or `mark_schedule_failed` triggers an immediate `save_checkpoint(...)` to `data/checkpoints/<season>_<season_type_slug>.json`.

**Rationale:**

- *I/O cost is negligible at this scale.* A flush is one local file write of a few KB; sub-millisecond on SSD. Total flush time across a 250-hour run is on the order of single-digit minutes. Order-of-magnitude analysis: 321k flushes × ~5 ms ≈ 27 minutes across 250 hours of wall-clock = 0.18% overhead.
- *Crash window is zero re-fetched games.* The mental model is exact: "if it's in the checkpoint, it's done."
- *Smaller spec and impl surface.* No timer code, no batching code, no override knobs to test or document.
- *Future-proofing is cheap.* If I/O ever becomes a bottleneck, batching can be added in the orchestrator without touching `checkpoint.py`. Easy to add later, hard to retract a knob no one's using. YAGNI consistent with the project's existing posture.

**Hidden assumption:** the disk where `data/checkpoints/` lives is local and fast. A network-mounted `data/` would make per-mutation flush expensive. Not a current concern; flag if `data/` is ever relocated.

---

## 5. Failure Handling

### 5.1 Per-call retries

Owned by `api/rate_limiter.py` and `api/client.py` (Session 4). 5× exponential backoff with decorrelated jitter on 429 / 5xx / timeouts; non-retryable 4xx fails fast. **The orchestrator does not duplicate retry logic.** It consumes the existing client and reacts to its terminal exceptions (`RetriesExhaustedError`, `NonRetryableStatusError`).

### 5.2 Per-game failure (post-retry-exhausted)

When the client raises a terminal exception for a game-level fetch:

1. Orchestrator catches the exception.
2. Calls `mark_game_failed(checkpoint, endpoint, game_id)`.
3. Logs the failure (level: WARNING; format deferred to Session 9).
4. Continues to the next game in the stripe.
5. Increments the run-level consecutive-failure counter (§5.4).

The game is now in `failed[endpoint]`. On the next `historical` invocation, `pending_games(checkpoint, endpoint, all_game_ids)` returns the failed game alongside genuinely never-attempted ones (per Session 7 lock: "failed are pending"). It will be retried.

**No "permanently failed" state exists in the checkpoint.** Permanently-failing games belong in `known_anomalies.md` (separate Session 9+ workstream); the checkpoint stays single-purpose.

### 5.3 Per-schedule failure

Same pattern, different state transition: `mark_schedule_failed(checkpoint)`. Schedule failure aborts the current `(season, season_type)` — box-score stripes can't iterate without a schedule. Orchestrator continues to the next `(season, season_type)` pair.

### 5.4 Run-level circuit breaker

**Decision: Abort run after 10 consecutive failures. Counter resets on any success.**

Triggered by 10 consecutive `mark_*_failed` calls with no intervening `mark_*_complete`.

On trip:

1. Orchestrator catches its own circuit-breaker exception.
2. Final `save_checkpoint(...)` (already up-to-date from per-mutation flushes; this is paranoid double-flush insurance).
3. Log: `"Circuit breaker tripped after 10 consecutive failures. Last failure: <endpoint> <game_id>. Run aborted; rerun `historical` to resume."`
4. Exit with non-zero status code.

**Rationale.** During a sustained `stats.nba.com` outage, the rate limiter dutifully retries each game 5× and exhausts before declaring failure. Without a circuit breaker, the orchestrator would burn 30 calls/min × hours of failed-but-attempted-anyway calls before noticing. 10 consecutive failures is well past "transient blip" (which scatters successes between failures) and well before "burned a meaningful chunk of the budget." Resumable: per-mutation flushes have already saved state.

The threshold is overridable via `--max-consecutive-failures` for debugging or for users who want to ride out a long outage.

The rejected alternative — a percentage-based threshold over a sliding window — handles intermittent flakiness more nuanced-ly but introduces a windowed counter that needs its own spec. YAGNI for v1.

---

## 6. Rate Limiting

**Decision: Use the existing global rate limiter (`api/rate_limiter.py`) without modification. No per-endpoint sub-limits.**

If per-endpoint patterns ever emerge in production (e.g., advanced is rate-limit-touchy in practice), the fix lives in `rate_limiter.py`, not duplicated in orchestration. The orchestrator is rate-limiter-agnostic by design.

---

## 7. "Done" Definition

### 7.1 Per-`(season, season_type)`

A `(season, season_type)` pair is done when all of the following hold:

1. `is_schedule_complete(checkpoint)` is True.
2. For each box-score endpoint E in {traditional, advanced, summary}: `is_endpoint_complete(checkpoint, E, expected_count)` is True (lenient: `len(completed) + len(failed) == expected_count`).

`expected_count` is derived at runtime from the schedule (the expected game count for that `(season, season_type)`, per Session 7's "schedule is truth" lock — never duplicated as a stored field).

Lenient completeness pairs with the retry-on-next-run failure model: a single permanently-stuck game shouldn't block the orchestrator from reporting a `(season, season_type)` as done. Permanent failures are surfaced through `known_anomalies.md`, not through `is_endpoint_complete`'s return value.

### 7.2 Run-level

The run is done when every `(season, season_type)` pair in the targeted set is done. The targeted set is determined by CLI flags:

- *Default:* all 29 seasons (1997-98 through 2025-26) × all applicable season_types (`regular_season` and `playoffs` always; `play_in` only for seasons ≥ `PLAY_IN_FIRST_SEASON`).
- `--season-range` and `--season-type` filters reduce the targeted set.
- `--endpoint` filter does not change the targeted `(season, season_type)` set; it changes which endpoints are processed *within* each pair.

### 7.3 Skip-already-done semantics

At startup, the orchestrator checks done-status for each `(season, season_type)` in the targeted set. Already-done pairs are skipped entirely — no schedule re-fetch, no stripe iteration, no API calls. Reruns of `historical` over a finished span are fast no-ops.

(Edge case for future consideration: forced re-fetch of a completed `(season, season_type)` is not currently supported. Workaround is to delete the checkpoint file manually. A `--force` flag can be added later if a use case appears; YAGNI now.)

---

## 8. CLI Surface

### 8.1 Subcommands

argparse-based, two top-level subcommands:

```
python -m nba_four_factors.cli backfill ...
python -m nba_four_factors.cli incremental ...
```

argparse chosen over typer/click because the project hasn't pulled in a CLI library and stdlib coverage is sufficient for this surface.

### 8.2 `backfill`

```
python -m nba_four_factors.cli backfill \
    [--season-range YYYY_YY:YYYY_YY] \
    [--reverse] \
    [--season-type {regular,playoffs,play_in} ...] \
    [--endpoint {schedule,traditional,advanced,summary} ...] \
    [--dry-run] \
    [--max-consecutive-failures N]
```

| Flag | Type | Default | Behavior |
|------|------|---------|----------|
| `--season-range` | string `YYYY_YY:YYYY_YY` | all 29 seasons | inclusive on both ends |
| `--reverse` | bool | false (newest-first) | flips to chronological |
| `--season-type` | nargs='+' | all three | choices: `regular`, `playoffs`, `play_in` |
| `--endpoint` | nargs='+' | all four | choices: `schedule`, `traditional`, `advanced`, `summary` |
| `--dry-run` | bool | false | print planned work, no API calls |
| `--max-consecutive-failures` | int | 10 | circuit breaker threshold |

**Argument-to-enum mapping.** User-facing names are short (`traditional`); the corresponding `Endpoint` enum members are longer (`Endpoint.BOXSCORE_TRADITIONAL`). The CLI parser maps short names to enum values inside `cli.py` before handing off to the orchestrator. Same for `--season-type` (`regular` → `SeasonType.REGULAR`, etc.).

### 8.3 `incremental`

```
python -m nba_four_factors.cli incremental \
    [--season YYYY_YY] \
    [--dry-run]
```

| Flag | Type | Default | Behavior |
|------|------|---------|----------|
| `--season` | string `YYYY_YY` | current season (computed from today) | single season only |
| `--dry-run` | bool | false | print planned work, no API calls |

Incremental is intentionally narrow: no `--season-range` (always one season), no `--endpoint` filter (always all four), no `--max-consecutive-failures` override (the loop is small enough that the default is fine).

### 8.4 `--endpoint` filter and schedule dependency

If `--endpoint` is given without `schedule` *and* the schedule is not complete for any targeted `(season, season_type)`, the orchestrator rejects the run before any API calls:

```
Error: schedule must be complete before targeting box-score endpoints in isolation.
       Affected: <season> <season_type>
       Rerun with `--endpoint schedule` (or omit --endpoint) first.
```

Strict over implicit. The error message is self-correcting — the user's next command is in the message.

### 8.5 Deliberately omitted flags

- `--config` / config file: YAGNI, consistent with hardcoded data paths.
- `--checkpoint-dir`: path scheme is locked at `data/checkpoints/`; debugging via symlinks if ever needed.
- `--verbose` / `-v`: deferred to Session 9 observability spec.
- `--force`: deferred until a use case appears (see §7.3 edge case).

---

## 9. Module Layout

```
nba_four_factors/
├── cli.py                      # argparse dispatch, calls into orchestration
├── orchestration/
│   ├── __init__.py
│   ├── historical.py           # backfill driver
│   ├── incremental.py          # nightly update driver
│   └── _common.py              # shared helpers (stripe iteration, circuit breaker, dry-run printer)
├── api/                        # existing
├── storage/                    # existing
└── ...

tests/
├── unit/
│   ├── test_cli.py             # argparse parsing, subcommand routing, --endpoint strict-reject
│   ├── test_orchestration_historical.py
│   ├── test_orchestration_incremental.py
│   └── test_orchestration_common.py
└── integration/                # existing endpoint integration tests; orchestration adds none initially
```

**Two driver modules, not one.** Backfill and incremental share *some* helpers (stripe iteration, schedule-then-box-score sequencing, circuit breaker, dry-run printer) but their top-level shape is meaningfully different — backfill is a sweep over a multi-dimensional space with resume semantics; incremental is a single-season delta detector with an early-exit. A single file would either be 600+ lines with two distinct halves, or it'd force an over-general abstraction trying to unify them. Two files keeps each one focused.

**`_common.py` underscore-prefixed.** Package-internal, not a public API. The Session 7 cross-module-private mitigation applies to *cross-package* private imports; within-package underscore is the standard Python convention for "internal to this package." If a helper later needs to be called from outside `orchestration/`, promote it.

**`cli.py` is the entry point and argument parser; orchestration is the engine.** Clean separation:

- Tests of CLI parsing live in `test_cli.py` and don't need orchestration mocks.
- Tests of orchestration logic live in `test_orchestration_*.py` and don't need to construct argparse namespaces — they call the underlying functions directly with explicit arguments.

**Dependency injection, mirroring existing patterns.** Orchestration unit tests use injected fakes for the rate limiter, HTTP client, and checkpoint storage — the same DI pattern that `api/rate_limiter.py` (injectable clock/sleep/random) and the storage modules already use. No real API calls in unit tests.

---

## 10. Control Flow (Pseudocode)

For Session 9 implementation reference. Pseudocode, not real Python — exact exception types, function naming, dry-run formatting, and ordering of flush relative to log are impl details.

```python
def run_backfill(args):
    targets = compute_targeted_set(args)  # respects --season-range, --season-type, default ordering, --reverse
    if args.dry_run:
        print_dry_run_plan(targets, args.endpoint)
        return

    consecutive_failures = 0

    for season, season_type in targets:
        ckpt = load_checkpoint(season, season_type) or initialize_checkpoint(season, season_type)

        if is_pair_done(ckpt, season, season_type, endpoints=args.endpoint):
            log_info("skip: already done", season=season, season_type=season_type)
            continue

        # schedule stripe
        if "schedule" in args.endpoint and not is_schedule_complete(ckpt):
            mark_schedule_started(ckpt)
            save_checkpoint(season, season_type, ckpt)
            try:
                fetch_and_persist_schedule(season, season_type)
                mark_schedule_complete(ckpt)
                consecutive_failures = 0
            except RetriesExhaustedError:
                mark_schedule_failed(ckpt)
                consecutive_failures += 1
                save_checkpoint(season, season_type, ckpt)
                if consecutive_failures >= args.max_consecutive_failures:
                    abort_run(ckpt, season, season_type)
                continue  # to next (season, season_type); box-score stripes can't run without schedule
            save_checkpoint(season, season_type, ckpt)
        elif not is_schedule_complete(ckpt) and any_box_score_in(args.endpoint):
            reject_run(
                "schedule must be complete before targeting box-score endpoints in isolation",
                season=season,
                season_type=season_type,
            )

        # box-score stripes
        all_game_ids = load_game_ids_from_schedule(season, season_type)
        expected_count = len(all_game_ids)

        for endpoint in [BOXSCORE_TRADITIONAL, BOXSCORE_ADVANCED, BOXSCORE_SUMMARY]:
            if endpoint not in args.endpoint:
                continue
            if is_endpoint_complete(ckpt, endpoint, expected_count):
                continue  # skip already-complete stripes on resume
            for game_id in pending_games(ckpt, endpoint, all_game_ids):
                try:
                    fetch_and_persist_game(endpoint, game_id)
                    mark_game_complete(ckpt, endpoint, game_id)
                    consecutive_failures = 0
                except RetriesExhaustedError:
                    mark_game_failed(ckpt, endpoint, game_id)
                    consecutive_failures += 1
                    save_checkpoint(season, season_type, ckpt)
                    if consecutive_failures >= args.max_consecutive_failures:
                        abort_run(ckpt, season, season_type)
                    continue
                save_checkpoint(season, season_type, ckpt)  # every-mutation flush

        write_manifest(season, season_type, ckpt)


def run_incremental(args):
    season = args.season or current_season()

    for season_type in [REGULAR, PLAY_IN, PLAYOFFS]:
        if season_type == PLAY_IN and season < PLAY_IN_FIRST_SEASON:
            continue

        ckpt = load_checkpoint(season, season_type) or initialize_checkpoint(season, season_type)

        # always re-fetch schedule for incremental (current-season games appear daily)
        fetch_and_persist_schedule(season, season_type)
        mark_schedule_complete(ckpt)  # idempotent if already complete
        save_checkpoint(season, season_type, ckpt)

        all_game_ids = load_game_ids_from_schedule(season, season_type)
        if not all_game_ids:
            continue  # season_type doesn't have games yet (e.g., playoffs not started)

        for endpoint in [BOXSCORE_TRADITIONAL, BOXSCORE_ADVANCED, BOXSCORE_SUMMARY]:
            for game_id in pending_games(ckpt, endpoint, all_game_ids):
                try:
                    fetch_and_persist_game(endpoint, game_id)
                    mark_game_complete(ckpt, endpoint, game_id)
                except RetriesExhaustedError:
                    mark_game_failed(ckpt, endpoint, game_id)
                save_checkpoint(season, season_type, ckpt)
```

Notes for Session 9 impl:

- `compute_targeted_set` lives in `_common.py` — it consumes raw CLI args and produces an ordered list of `(season, season_type)` tuples respecting the §3.2 within-season order, the §3.3 cross-season order, and the play-in skip rule.
- `is_pair_done` and `any_box_score_in` are thin helpers in `_common.py`.
- `current_season()` is a helper that maps today's date to the canonical season string. Likely lives in `config.py` next to `SEASONS`. See §12.3.
- The pseudocode shows `save_checkpoint` after every successful mutation and after every failure; a small refactor in impl could consolidate into a context-manager wrapper, but the spec doesn't require it.

---

## 11. Out of Scope (Deferred)

- **Logging format and levels.** What goes to stdout vs file, structured vs text, log rotation. The spec uses informal `log_info(...)` placeholders; these get formalized when observability is specced in Session 9+.
- **Progress reporting / ETA estimation.** A 250-hour run benefits from an accurate ETA. Trivial v1 (calls remaining × ~2 sec) is sufficient; v2 could use a windowed average to handle the rate limiter's actual behavior including retries.
- **Manifest content schema.** The orchestrator writes a manifest at end-of-`(season, season_type)` per existing manifest conventions (Session 4). Exact field set and ordering deferred until impl.
- **Implementation.** Modules, tests, gates. Sessions 9–10 likely.
- **Processed-layer normalization** and downstream layers (feature engineering, four-factor computation, modeling, EDA). Out of scope entirely.
- **Anomaly registry (`known_anomalies.md`).** Referenced in §5.2 as the home for permanently-failing games. The file format and update workflow are a separate spec discussion (carried gap from Session 6).

---

## 12. TODOs Before Session 9 Implementation Begins

### 12.1 Verify `PLAY_IN_FIRST_SEASON`

The play-in tournament was instituted for the 2020-21 NBA season. The 2019-20 bubble had a single play-in series that may or may not be flagged as `PlayIn` in `leaguegamelog`'s schedule data.

*Verification:* query `leaguegamelog` for `season=2019-20 season_type=PlayIn` and `season=2020-21 season_type=PlayIn`. The first season returning a non-empty row set is `PLAY_IN_FIRST_SEASON`. Likely value: `2020_21`.

Add the constant to `config.py` if it doesn't already exist. Both the `--season-type play_in` flag and the within-season ordering depend on it.

### 12.2 Confirm `Endpoint` enum string-value mapping for CLI choices

User-facing `--endpoint` choices are `schedule`, `traditional`, `advanced`, `summary`. The `Endpoint` enum members are `SCHEDULE`, `BOXSCORE_TRADITIONAL`, `BOXSCORE_ADVANCED`, `BOXSCORE_SUMMARY` with values like `"leaguegamelog"`, `"boxscoretraditionalv3"`, etc. The mapping table lives in `cli.py`. Verify no naming collisions and decide whether to expose the mapping as a small dict literal or via a helper function.

### 12.3 Design `current_season()` helper

The `incremental` driver's default `--season` value computes from today's date. Need a helper that maps `date(today)` → canonical season string (e.g., `"2025_26"` for any date between October 2025 and June 2026, with offseason behavior to define).

*Open sub-question:* what does `current_season()` return during the offseason (July–September)? The previous completed season (so `incremental` is idempotent on completed data) or the upcoming season (so the schedule fetch starts working as soon as it's published)? Lean: previous completed season, with an explicit `--season 2026_27` override available once the new schedule is published. Lock at impl time.

Likely lives in `config.py` next to `SEASONS`.

---

*End of spec. Ready for Session 9 implementation.*
