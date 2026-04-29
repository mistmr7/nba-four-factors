"""Command-line entry point.

Three top-level subcommands:

    python -m nba_four_factors.cli backfill ...
    python -m nba_four_factors.cli incremental ...
    python -m nba_four_factors.cli process ...        # added Session 10

This module is intentionally a thin shim: it parses arguments, maps short CLI
strings to enum members (§8.2), and dispatches to orchestration drivers.  All
control-flow logic lives in :mod:`nba_four_factors.orchestration`.

argparse over typer/click because the project hasn't pulled in a CLI library
and stdlib coverage is sufficient for this surface (§8.1).
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Endpoint, SeasonType
from .orchestration.historical import run_backfill
from .orchestration.incremental import run_incremental
from .orchestration.process import run_process

# --------------------------------------------------------------------------- #
# CLI-name → enum mapping (§8.2, §12.2 TODO)
# --------------------------------------------------------------------------- #

# User-facing names are short; enum members are longer.  See §12.2 for the
# open verification step on the Endpoint enum's underlying string values.
ENDPOINT_CLI_TO_ENUM: dict[str, Endpoint] = {
    "schedule": Endpoint.SCHEDULE,
    "traditional": Endpoint.BOXSCORE_TRADITIONAL,
    "advanced": Endpoint.BOXSCORE_ADVANCED,
    "summary": Endpoint.BOXSCORE_SUMMARY,
}
ENDPOINT_CLI_CHOICES: list[str] = list(ENDPOINT_CLI_TO_ENUM.keys())

SEASON_TYPE_CLI_TO_ENUM: dict[str, SeasonType] = {
    "regular": SeasonType.REGULAR,
    "play_in": SeasonType.PLAY_IN,
    "playoffs": SeasonType.PLAYOFFS,
}
SEASON_TYPE_CLI_CHOICES: list[str] = list(SEASON_TYPE_CLI_TO_ENUM.keys())


# --------------------------------------------------------------------------- #
# Parser construction
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with the three subcommands.

    Exposed (rather than constructed inline in ``main``) so tests can call it
    directly without instantiating sys.argv — see §9 test layout note.
    """
    parser = argparse.ArgumentParser(
        prog="python -m nba_four_factors.cli",
        description="NBA four-factors raw-layer ingestion driver.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_backfill_subparser(subparsers)
    _add_incremental_subparser(subparsers)
    _add_process_subparser(subparsers)

    return parser


def _add_backfill_subparser(subparsers) -> None:
    """Register the ``backfill`` subcommand (§8.2)."""
    bf = subparsers.add_parser(
        "backfill",
        help="One-shot historical sweep over a season range.",
        description=(
            "Sweep the targeted (season, season_type) set, fetching schedule "
            "and box-score endpoints with full resume semantics."
        ),
    )
    bf.add_argument(
        "--season-range",
        metavar="YYYY_YY:YYYY_YY",
        default=None,
        help="Inclusive season range (default: all seasons).",
    )
    bf.add_argument(
        "--reverse",
        action="store_true",
        help="Iterate chronologically (oldest-first); default is newest-first.",
    )
    bf.add_argument(
        "--season-type",
        nargs="+",
        choices=SEASON_TYPE_CLI_CHOICES,
        default=list(SEASON_TYPE_CLI_CHOICES),
        help="Which season_types to process (default: all three).",
    )
    bf.add_argument(
        "--endpoint",
        nargs="+",
        choices=ENDPOINT_CLI_CHOICES,
        default=list(ENDPOINT_CLI_CHOICES),
        help="Which endpoints to process (default: all four).",
    )
    bf.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work; make no API calls.",
    )
    bf.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=10,
        metavar="N",
        help="Circuit-breaker threshold (§5.4); default 10.",
    )


def _add_incremental_subparser(subparsers) -> None:
    """Register the ``incremental`` subcommand (§8.3)."""
    inc = subparsers.add_parser(
        "incremental",
        help="Nightly delta pull for the current (or specified) season.",
        description=(
            "Re-fetch schedule and pull any pending box-score data for one "
            "season.  Intentionally narrow surface (§8.3)."
        ),
    )
    inc.add_argument(
        "--season",
        metavar="YYYY_YY",
        default=None,
        help="Single season (default: computed from today's date).",
    )
    inc.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work; make no API calls.",
    )


def _add_process_subparser(subparsers) -> None:
    """Register the ``process`` subcommand (Session 10).

    Builds the processed-layer Parquet for one (season, season_type) pair.
    Single-pair only — bulk processing is a shell loop, per Session 10 §6.
    No --dry-run (local-disk only, runs in seconds, no API surface to
    protect against accidental load).
    """
    pr = subparsers.add_parser(
        "process",
        help="Build processed four-factors Parquet for one (season, season_type).",
        description=(
            "Read saved LeagueGameLog JSON and produce a per-(game, team) "
            "Parquet with Dean Oliver's four factors computed for both team "
            "and opponent.  Idempotent and deterministic."
        ),
    )
    pr.add_argument(
        "--season",
        metavar="YYYY_YY",
        required=True,
        help="Single season (e.g. 2024_25).  No range; loop in shell for bulk.",
    )
    pr.add_argument(
        "--season-type",
        choices=SEASON_TYPE_CLI_CHOICES,
        required=True,
        help="One of: regular, play_in, playoffs.",
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def _translate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Convert CLI string choices to enum members in-place.

    argparse stores ``choices`` as the raw strings.  The orchestration layer
    works in enums, so we translate here at the boundary — keeps orchestration
    tests free of CLI parsing concerns (§9).

    ``season_type`` is polymorphic across subcommands: ``backfill`` accepts a
    list (``nargs="+"``); ``process`` accepts a single value.  We dispatch on
    type to handle both shapes without leaking subcommand awareness into the
    translator.
    """
    if getattr(args, "endpoint", None) is not None:
        args.endpoint = [ENDPOINT_CLI_TO_ENUM[s] for s in args.endpoint]
    if getattr(args, "season_type", None) is not None:
        if isinstance(args.season_type, list):
            args.season_type = [SEASON_TYPE_CLI_TO_ENUM[s] for s in args.season_type]
        else:
            args.season_type = SEASON_TYPE_CLI_TO_ENUM[args.season_type]
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point.  ``argv`` defaults to ``sys.argv[1:]`` per argparse."""
    # Logging format is informal here; formal observability deferred to the
    # Session 9+ logging spec (§11).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    args = _translate_args(args)

    if args.command == "backfill":
        return run_backfill(args)
    if args.command == "incremental":
        return run_incremental(args)
    if args.command == "process":
        return run_process(args)

    # Unreachable: argparse would have errored on an unknown subcommand
    # because ``required=True`` is set on the subparsers.  Defensive only.
    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
