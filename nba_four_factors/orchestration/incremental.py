"""Nightly incremental update driver.

Pulls new games from the current season since the last update.  Total work is
small (handful of games per night during the season, zero during the
offseason).  Optimized for fast no-op-when-nothing-new (§1).

Differences from the historical driver (deliberate, §3.1, §8.3, §10):

* Single season only (default: ``current_season()``); no ``--season-range``.
* All four endpoints always, no ``--endpoint`` filter.
* Schedule is *always* re-fetched — current-season games appear daily, so the
  schedule on disk goes stale every game day.  ``mark_schedule_complete`` is
  idempotent, so this is cheap.
* No circuit breaker.  Per-night surface is small enough that failures show
  up in logs and the operator just reruns; a windowed counter would be
  YAGNI-violating overhead.

Entry point: :func:`run_incremental`.
"""

from __future__ import annotations

import contextlib
import logging

from ..api.client import Client, RetriesExhaustedError
from ..config import (
    PLAY_IN_FIRST_SEASON,
    SeasonType,
    current_season,
)
from ..storage.checkpoint import (
    initialize_checkpoint,
    is_schedule_complete,
    load_checkpoint,
    mark_game_complete,
    mark_game_failed,
    mark_schedule_complete,
    mark_schedule_started,
    pending_games,
    save_checkpoint,
)
from ._common import BOX_SCORE_ENDPOINTS, SEASON_TYPE_ORDER
from ._io import (
    fetch_and_save_game,
    fetch_and_save_schedule,
    load_schedule_game_ids,
)

log = logging.getLogger(__name__)


def run_incremental(args) -> int:
    """Execute the nightly incremental update.

    :param args: argparse namespace from ``cli.py``.  Required attributes:

        * ``season``: ``str | None`` — single season; ``None`` means
          :func:`current_season`.
        * ``dry_run``: ``bool``

    :returns: process exit code (always ``0`` — incremental tolerates
        per-pair failures and just logs them).
    """
    season: str = args.season or current_season()

    if args.dry_run:
        _print_dry_run(season)
        return 0

    # One Client per run; shares the rate limiter across all season_types.
    client = Client()

    for season_type in SEASON_TYPE_ORDER:
        if season_type is SeasonType.PLAY_IN and season < PLAY_IN_FIRST_SEASON:
            continue

        ckpt = load_checkpoint(season, season_type) or initialize_checkpoint(season, season_type)

        # Always re-fetch the schedule — current-season games appear daily.
        # If this fails, skip box-score stripes for tonight; tomorrow's run
        # will retry.
        try:
            fetch_and_save_schedule(client, season, season_type)
        except RetriesExhaustedError:
            log.warning(
                "incremental schedule fetch failed: %s %s",
                season,
                season_type.name.lower(),
            )
            continue

        # The schedule mutators enforce a strict state machine in which
        # `complete` is terminal, so a blunt mark_schedule_complete crashes on
        # any season already marked complete by a prior backfill (the common
        # case). incremental re-fetches the entire schedule each run, so the
        # schedule is genuinely complete afterward: skip the transition when it
        # is already complete, and otherwise route through `in_progress` first
        # so the complete transition is legal. The start call is guarded
        # because a checkpoint left mid-run is already in progress and cannot
        # be started again.
        if not is_schedule_complete(ckpt):
            # a checkpoint already in progress raises; proceed straight to complete
            with contextlib.suppress(ValueError):
                mark_schedule_started(ckpt)
            mark_schedule_complete(ckpt)
            save_checkpoint(season, season_type, ckpt)

        all_game_ids = load_schedule_game_ids(season, season_type)
        if not all_game_ids:
            # Season-type doesn't have games yet (e.g., playoffs not started,
            # or play-in first season but tournament hasn't run).  Fast no-op.
            continue

        for endpoint in BOX_SCORE_ENDPOINTS:
            for game_id in pending_games(ckpt, endpoint, all_game_ids):
                try:
                    fetch_and_save_game(client, endpoint, game_id)
                except RetriesExhaustedError:
                    mark_game_failed(ckpt, endpoint, game_id)
                    save_checkpoint(season, season_type, ckpt)
                    log.warning(
                        "incremental game fetch failed: %s %s",
                        endpoint.name.lower(),
                        game_id,
                    )
                    continue

                mark_game_complete(ckpt, endpoint, game_id)
                save_checkpoint(season, season_type, ckpt)

    return 0


def _print_dry_run(season: str) -> None:
    """Print the incremental dry-run plan (§8.3)."""
    types_to_visit = [
        st
        for st in SEASON_TYPE_ORDER
        if not (st is SeasonType.PLAY_IN and season < PLAY_IN_FIRST_SEASON)
    ]
    type_labels = ", ".join(st.name.lower() for st in types_to_visit)
    print(f"DRY RUN: incremental for season {season}")
    print(f"DRY RUN: would re-fetch schedule for {type_labels}")
    print(
        "DRY RUN: would pull box-score (traditional, advanced, summary) " "for any pending games."
    )
    print("DRY RUN: no API calls were made.")
