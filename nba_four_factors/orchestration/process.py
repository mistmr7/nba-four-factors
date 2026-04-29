"""CLI driver for the ``process`` subcommand (Session 10).

Thin wrapper around :func:`processed.process_pair`.  Unlike the historical
and incremental drivers, ``process`` has no checkpointing, no API calls,
no dry-run distinction (operations are local-disk only and run in
seconds).  This module exists for two reasons:

1. **Pattern consistency.**  ``cli.py`` dispatches via
   ``run_<command>(args)``; deviating for one subcommand creates a
   special case in the dispatcher.
2. **Future hook.**  If ``process`` ever needs progress logging,
   per-season summary stats, or batch handling, this is where that goes
   without touching the CLI surface.

The driver translates one user-facing failure mode (raw JSON missing) into
a friendly message and a non-zero exit, per Session 10 §5.  Other
exceptions propagate with full traceback — they're bugs, not user error.
"""

from __future__ import annotations

import argparse
import logging

from ..processed import process_pair

logger = logging.getLogger(__name__)


def run_process(args: argparse.Namespace) -> int:
    """Build the processed Parquet for one (season, season_type) pair.

    ``args`` must have ``season`` (str, e.g. ``"2024_25"``) and
    ``season_type`` (a :class:`SeasonType` enum member after
    :func:`cli._translate_args` has run).

    Returns the process exit code: ``0`` on success, ``1`` if the raw
    schedule JSON is missing.
    """
    try:
        path = process_pair(args.season, args.season_type)
    except FileNotFoundError as e:
        # The most common user-facing failure: they ran ``process`` before
        # ``backfill`` finished for this pair.  Translate to a clear hint.
        logger.error(
            "Raw schedule JSON not found for %s %s: %s. "
            "Run `backfill --season-range %s:%s --season-type %s` first.",
            args.season,
            args.season_type.name.lower(),
            e,
            args.season,
            args.season,
            args.season_type.name.lower(),
        )
        return 1

    logger.info("Wrote processed Parquet: %s", path)
    return 0
