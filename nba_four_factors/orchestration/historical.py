"""Historical backfill driver.

One-shot sweep of the targeted ``(season, season_type)`` set, processing each
pair through schedule → traditional → advanced → summary stripes (§3.1).
Resumability is the dominant design constraint (§1): every successful
checkpoint mutation flushes immediately to disk (§4 β I/O), so a crash
re-fetches at most zero games.

Entry point: :func:`run_backfill`.

Exit codes:

* ``0`` — success (run completed, or all targeted pairs already done)
* ``1`` — circuit breaker tripped (§5.4); rerun ``backfill`` to resume
* ``2`` — validation reject (§8.4); schedule incomplete and only box-score
          endpoints targeted
"""

from __future__ import annotations

import logging
import sys

from ..api.client import Client, RetriesExhaustedError
from ..config import Endpoint
from ..storage.checkpoint import (
    initialize_checkpoint,
    is_endpoint_complete,
    is_schedule_complete,
    load_checkpoint,
    mark_game_complete,
    mark_game_failed,
    mark_schedule_complete,
    mark_schedule_failed,
    mark_schedule_started,
    pending_games,
    save_checkpoint,
)
from ._common import (
    BOX_SCORE_ENDPOINTS,
    CircuitBreakerTripped,
    any_box_score_in,
    compute_targeted_set,
    is_pair_done,
    print_dry_run_plan,
)
from ._io import (
    fetch_and_save_game,
    fetch_and_save_schedule,
    load_schedule_game_ids,
    write_pair_manifest,
)

log = logging.getLogger(__name__)


# Internal sentinel used by :func:`_run_schedule_stripe` to tell the caller
# whether to proceed to box-score stripes for this pair.  Cleaner than
# re-checking ``is_schedule_complete`` after the call.
_SCHEDULE_OK = "ok"
_SCHEDULE_ALREADY_DONE = "already-done"
_SCHEDULE_FAILED = "failed"


def run_backfill(args) -> int:
    """Execute the historical backfill per the spec.

    :param args: argparse namespace from ``cli.py``.  Required attributes:

        * ``season_range``: ``str | None`` (e.g. ``"2010_11:2024_25"``)
        * ``reverse``: ``bool``
        * ``season_type``: ``list[SeasonType]``
        * ``endpoint``: ``list[Endpoint]``
        * ``dry_run``: ``bool``
        * ``max_consecutive_failures``: ``int``

    :returns: process exit code; see module docstring.
    """
    targets = compute_targeted_set(
        season_range=args.season_range,
        season_types=args.season_type,
        reverse=args.reverse,
    )

    if args.dry_run:
        print_dry_run_plan(targets, args.endpoint)
        return 0

    # One Client per run: shares the rate limiter across thousands of calls,
    # which is the whole point of having a Client at all.  Lifecycle is
    # implicit (no context manager) — Client is stateless w.r.t. external
    # resources beyond what GC handles.
    client = Client()

    consecutive_failures = 0
    threshold: int = args.max_consecutive_failures

    # Track the most recent (season, season_type, ckpt) so the circuit-breaker
    # handler can do a paranoid final flush even if the trip happens deep in
    # a helper.  Initialized to None so a trip before the first iteration
    # (impossible in practice, but defensive) is harmless.
    last_pair: tuple[str, object, object] | None = None

    try:
        for season, season_type in targets:
            ckpt = load_checkpoint(season, season_type) or initialize_checkpoint(
                season, season_type
            )
            last_pair = (season, season_type, ckpt)

            if is_pair_done(ckpt, season, season_type, endpoints=args.endpoint):
                log.info(
                    "skip already-done pair: %s %s",
                    season,
                    season_type.name.lower(),
                )
                continue

            # ----- schedule stripe (§3.1) -----
            schedule_state, consecutive_failures = _run_schedule_stripe(
                ckpt,
                season,
                season_type,
                args,
                client,
                consecutive_failures=consecutive_failures,
                threshold=threshold,
            )
            if schedule_state == _SCHEDULE_FAILED:
                # Schedule fetch hit retry exhaustion.  Box-score stripes
                # can't iterate without a schedule; advance to next pair.
                # (Circuit breaker, if tripped, will already have raised.)
                continue

            # ----- §8.4 strict-reject -----
            # If we're not requesting schedule and the schedule isn't already
            # complete, and we ARE requesting box-score endpoints, refuse.
            if (
                Endpoint.SCHEDULE not in args.endpoint
                and not is_schedule_complete(ckpt)
                and any_box_score_in(args.endpoint)
            ):
                _print_strict_reject(season, season_type)
                return 2

            # ----- box-score stripes (§3.1) -----
            if not any_box_score_in(args.endpoint):
                # Schedule-only run; nothing else to do for this pair.
                write_pair_manifest(season, season_type, ckpt)
                continue

            all_game_ids = load_schedule_game_ids(season, season_type)
            expected_count = len(all_game_ids)

            for endpoint in BOX_SCORE_ENDPOINTS:
                if endpoint not in args.endpoint:
                    continue
                if is_endpoint_complete(ckpt, endpoint, expected_count):
                    # Resume case: already-complete stripe, skip to next.
                    continue
                consecutive_failures = _run_box_score_stripe(
                    ckpt,
                    season,
                    season_type,
                    endpoint,
                    all_game_ids,
                    client,
                    consecutive_failures=consecutive_failures,
                    threshold=threshold,
                )

            write_pair_manifest(season, season_type, ckpt)

    except CircuitBreakerTripped as exc:
        # §5.4 step 2: paranoid double-flush.  Per-mutation flushes have
        # already saved state; this is insurance against e.g. a torn write
        # in the immediately-prior save_checkpoint call.
        if last_pair is not None:
            season, season_type, ckpt = last_pair
            try:
                save_checkpoint(season, season_type, ckpt)
            except Exception:
                log.exception("paranoid final flush failed")
        log.error(
            "Circuit breaker tripped after %d consecutive failures. "
            "Last failure: %s. Run aborted; rerun `historical` to resume.",
            exc.threshold,
            exc.what,
        )
        return 1

    return 0


# --------------------------------------------------------------------------- #
# Stripe helpers
# --------------------------------------------------------------------------- #


def _run_schedule_stripe(
    ckpt,
    season: str,
    season_type,
    args,
    client: Client,
    *,
    consecutive_failures: int,
    threshold: int,
) -> tuple[str, int]:
    """Process the schedule stripe for one pair.

    Returns ``(state, new_consecutive_failures)`` where ``state`` is one of
    :data:`_SCHEDULE_OK`, :data:`_SCHEDULE_ALREADY_DONE`,
    :data:`_SCHEDULE_FAILED`.

    Raises :class:`CircuitBreakerTripped` if the threshold is hit.
    """
    if Endpoint.SCHEDULE not in args.endpoint:
        # Schedule wasn't requested for this run.  Whether the schedule is
        # complete or not, we don't touch it here; the strict-reject in the
        # caller handles the "schedule incomplete + box-score requested"
        # case before we proceed to box-score stripes.
        return _SCHEDULE_OK, consecutive_failures

    if is_schedule_complete(ckpt):
        return _SCHEDULE_ALREADY_DONE, consecutive_failures

    mark_schedule_started(ckpt)
    save_checkpoint(season, season_type, ckpt)

    try:
        fetch_and_save_schedule(client, season, season_type)
    except RetriesExhaustedError:
        mark_schedule_failed(ckpt)
        save_checkpoint(season, season_type, ckpt)
        consecutive_failures += 1
        log.warning(
            "schedule fetch failed: %s %s",
            season,
            season_type.name.lower(),
        )
        if consecutive_failures >= threshold:
            raise CircuitBreakerTripped(
                threshold,
                what=f"schedule {season} {season_type.name.lower()}",
            ) from None
        return _SCHEDULE_FAILED, consecutive_failures

    mark_schedule_complete(ckpt)
    save_checkpoint(season, season_type, ckpt)
    return _SCHEDULE_OK, 0  # success resets the counter


def _run_box_score_stripe(
    ckpt,
    season: str,
    season_type,
    endpoint: Endpoint,
    all_game_ids,
    client: Client,
    *,
    consecutive_failures: int,
    threshold: int,
) -> int:
    """Process one box-score stripe (§3.1) and return the new failure counter.

    Raises :class:`CircuitBreakerTripped` if the threshold is hit.
    """
    for game_id in pending_games(ckpt, endpoint, all_game_ids):
        try:
            fetch_and_save_game(client, endpoint, game_id)
        except RetriesExhaustedError:
            mark_game_failed(ckpt, endpoint, game_id)
            save_checkpoint(season, season_type, ckpt)
            consecutive_failures += 1
            log.warning(
                "game fetch failed: %s %s",
                endpoint.name.lower(),
                game_id,
            )
            if consecutive_failures >= threshold:
                raise CircuitBreakerTripped(
                    threshold,
                    what=f"{endpoint.name.lower()} {game_id}",
                ) from None
            continue

        mark_game_complete(ckpt, endpoint, game_id)
        save_checkpoint(season, season_type, ckpt)
        consecutive_failures = 0  # success resets

    return consecutive_failures


def _print_strict_reject(season: str, season_type) -> None:
    """Emit the §8.4 strict-reject message to stderr."""
    msg = (
        "Error: schedule must be complete before targeting box-score "
        "endpoints in isolation.\n"
        f"       Affected: {season} {season_type.name.lower()}\n"
        "       Rerun with `--endpoint schedule` (or omit --endpoint) first."
    )
    print(msg, file=sys.stderr)
