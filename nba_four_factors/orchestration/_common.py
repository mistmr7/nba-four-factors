"""Shared orchestration helpers.

Package-internal (underscore-prefixed per §9). Provides:

* :class:`CircuitBreakerTripped` — internal control-flow exception raised by
  the run-level circuit breaker (§5.4).
* :func:`compute_targeted_set` — builds the ordered list of
  ``(season, season_type)`` tuples for a backfill, respecting cross-season
  order (§3.3), within-season order (§3.2), and the play-in skip rule.
* :func:`is_pair_done` — skip-already-done check (§7.1, §7.3) modulated by
  the ``--endpoint`` filter.
* :func:`any_box_score_in` — predicate for "any box-score endpoint
  requested" (used by §8.4 strict-reject).
* :func:`print_dry_run_plan` — deterministic ``--dry-run`` printer.

Anything called from outside the ``orchestration`` package should be promoted
out of this module per the underscore convention.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from ..config import (
    PLAY_IN_FIRST_SEASON,
    SEASONS,
    Endpoint,
    SeasonType,
)
from ..storage.checkpoint import is_endpoint_complete, is_schedule_complete

log = logging.getLogger(__name__)


# Ordered tuple of box-score endpoints, §3.1: traditional → advanced → summary.
# Locked because traditional carries the four-factor inputs; mid-stripe crashes
# leave the most-useful endpoint complete and the auxiliary endpoints empty.
BOX_SCORE_ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint.BOXSCORE_TRADITIONAL,
    Endpoint.BOXSCORE_ADVANCED,
    Endpoint.BOXSCORE_SUMMARY,
)

# Within-season order, §3.2.
SEASON_TYPE_ORDER: tuple[SeasonType, ...] = (
    SeasonType.REGULAR,
    SeasonType.PLAY_IN,
    SeasonType.PLAYOFFS,
)


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


class CircuitBreakerTripped(Exception):
    """Raised when the consecutive-failure threshold is hit (§5.4).

    Caught at the top of :func:`historical.run_backfill` for clean shutdown
    (final paranoid flush, log, exit non-zero).  Not used by the incremental
    driver — its per-night surface is small enough that a circuit breaker
    isn't warranted (see spec §10 incremental pseudocode).
    """

    def __init__(self, threshold: int, *, what: str) -> None:
        self.threshold = threshold
        self.what = what
        super().__init__(
            f"Circuit breaker tripped after {threshold} consecutive failures. "
            f"Last failure: {what}."
        )


# --------------------------------------------------------------------------- #
# Target-set construction (§3.2, §3.3)
# --------------------------------------------------------------------------- #


def parse_season_range(spec: str) -> tuple[str, str]:
    """Parse a ``--season-range`` value of form ``YYYY_YY:YYYY_YY``.

    Returns the ``(start, end)`` pair, both inclusive.  Does not validate that
    the seasons exist in :data:`SEASONS`; that's :func:`compute_targeted_set`'s
    job.

    :raises ValueError: on malformed input.
    """
    if ":" not in spec:
        raise ValueError(f"--season-range must be 'YYYY_YY:YYYY_YY', got {spec!r}")
    start, end = spec.split(":", 1)
    return start.strip(), end.strip()


def compute_targeted_set(
    *,
    season_range: str | None,
    season_types: Iterable[SeasonType],
    reverse: bool,
) -> list[tuple[str, SeasonType]]:
    """Build the ordered list of ``(season, season_type)`` pairs for a backfill.

    Ordering rules (locked, §3.2 and §3.3):

    * cross-season:  newest-first by default; ``--reverse`` flips to chronological.
    * within-season: ``regular_season → play_in → playoffs``.
    * play-in skip:  for ``season < PLAY_IN_FIRST_SEASON``, the ``play_in``
      slot is omitted entirely (no checkpoint, no entry in the set).

    :param season_range: optional ``YYYY_YY:YYYY_YY`` string; ``None`` means
        all of :data:`SEASONS`.
    :param season_types: which season_types the user requested (a subset of
        :class:`SeasonType`).
    :param reverse: ``True`` selects chronological (oldest-first); ``False``
        is newest-first (the default).

    :raises ValueError: if ``season_range`` references seasons not present
        in :data:`SEASONS`.
    """
    if season_range is None:
        seasons = list(SEASONS)
    else:
        start, end = parse_season_range(season_range)
        if start not in SEASONS or end not in SEASONS:
            raise ValueError(f"--season-range endpoints must be in SEASONS; got {start}..{end}")
        # SEASONS is canonically oldest-first; clamp inclusively.
        i, j = SEASONS.index(start), SEASONS.index(end)
        if i > j:
            i, j = j, i
        seasons = list(SEASONS[i : j + 1])

    # Cross-season order.  SEASONS is oldest→newest, so we reverse by default
    # (newest-first per §3.3) and un-reverse on --reverse.
    if not reverse:
        seasons.reverse()

    requested_types = set(season_types)
    # Within-season order is fixed regardless of the user's flag order.
    ordered_types = [st for st in SEASON_TYPE_ORDER if st in requested_types]

    pairs: list[tuple[str, SeasonType]] = []
    for season in seasons:
        for st in ordered_types:
            if st is SeasonType.PLAY_IN and season < PLAY_IN_FIRST_SEASON:
                continue  # §3.2 play-in skip
            pairs.append((season, st))
    return pairs


# --------------------------------------------------------------------------- #
# Done-status check (§7.1, §7.3)
# --------------------------------------------------------------------------- #


def any_box_score_in(endpoints: Iterable[Endpoint]) -> bool:
    """True if any of the requested endpoints is a box-score endpoint."""
    requested = set(endpoints)
    return any(e in requested for e in BOX_SCORE_ENDPOINTS)


def is_pair_done(
    ckpt,
    season: str,
    season_type: SeasonType,
    *,
    endpoints: Iterable[Endpoint],
) -> bool:
    """Are all the endpoints requested for *this run* complete on this pair?

    Per §7.1, modulated by the ``--endpoint`` filter (§7.3): we only care
    about the endpoints the current run actually targets.  A run with
    ``--endpoint schedule`` skips a pair as soon as schedule is complete,
    even if the box-scores haven't been touched yet.

    Lenient completeness for box-score stripes: a pair counts as done when
    ``len(completed) + len(failed) == expected_count`` for each requested
    box-score endpoint.  Permanent failures are surfaced via
    ``known_anomalies.md`` (out of scope), not via this predicate.

    Performs a local file read (``load_game_ids_from_schedule``) only when
    box-score endpoints are requested *and* the schedule is already complete;
    no API calls.
    """
    requested = set(endpoints)

    if Endpoint.SCHEDULE in requested and not is_schedule_complete(ckpt):
        return False

    box_score_targets = [e for e in BOX_SCORE_ENDPOINTS if e in requested]
    if not box_score_targets:
        return True

    if not is_schedule_complete(ckpt):
        # Box-score stripes can't be complete without a schedule.
        return False

    # Deferred import to avoid a circular dependency: _io imports the api
    # endpoint modules at module-load time, and we want _common.py to remain
    # safely importable from anywhere (including conftest before fixtures
    # have replaced the api layer).
    from ._io import load_schedule_game_ids

    expected_count = len(load_schedule_game_ids(season, season_type))
    return all(
        is_endpoint_complete(ckpt, endpoint, expected_count) for endpoint in box_score_targets
    )


# --------------------------------------------------------------------------- #
# Dry-run printer (§8.2)
# --------------------------------------------------------------------------- #


def print_dry_run_plan(
    pairs: list[tuple[str, SeasonType]],
    endpoints: Iterable[Endpoint],
) -> None:
    """Print the planned work without making any API calls.

    Output is intentionally plain text — formal observability format is
    deferred to Session 9+ (§11).
    """
    endpoint_labels = ", ".join(_endpoint_cli_name(e) for e in endpoints)
    print(f"DRY RUN: {len(pairs)} (season, season_type) pair(s) targeted")
    print(f"DRY RUN: endpoints: {endpoint_labels}")
    for season, season_type in pairs:
        print(f"  {season}  {season_type.name.lower()}")
    print("DRY RUN: no API calls were made.")


def _endpoint_cli_name(e: Endpoint) -> str:
    """Map an Endpoint enum member back to its short CLI name (§8.2).

    Mirrors the cli.py mapping but kept here so dry-run output doesn't have
    to reach into cli.py.  See §12.2 TODO; if the canonical mapping moves
    to a helper in ``config.py``, this should delegate there.
    """
    return {
        Endpoint.SCHEDULE: "schedule",
        Endpoint.BOXSCORE_TRADITIONAL: "traditional",
        Endpoint.BOXSCORE_ADVANCED: "advanced",
        Endpoint.BOXSCORE_SUMMARY: "summary",
    }.get(e, e.name.lower())
