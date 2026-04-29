"""Tests for ``nba_four_factors.cli``.

Covers argparse parsing, the CLI-string-to-enum translation step (§8.2),
and dispatch to the orchestration drivers.  The driver functions
themselves are mocked here — their behavior is exercised in
test_orchestration_historical.py and test_orchestration_incremental.py.
"""

from __future__ import annotations

import pytest

from nba_four_factors import cli
from nba_four_factors.config import Endpoint, SeasonType

# --------------------------------------------------------------------------- #
# build_parser / argparse defaults
# --------------------------------------------------------------------------- #


def test_backfill_subcommand_default_arguments():
    parser = cli.build_parser()
    args = parser.parse_args(["backfill"])

    assert args.command == "backfill"
    assert args.season_range is None
    assert args.reverse is False
    assert args.dry_run is False
    assert args.max_consecutive_failures == 10
    # argparse stores the raw CLI strings; translation happens later
    assert set(args.season_type) == {"regular", "play_in", "playoffs"}
    assert set(args.endpoint) == {"schedule", "traditional", "advanced", "summary"}


def test_incremental_subcommand_default_arguments():
    parser = cli.build_parser()
    args = parser.parse_args(["incremental"])

    assert args.command == "incremental"
    assert args.season is None
    assert args.dry_run is False


def test_backfill_max_consecutive_failures_is_int():
    parser = cli.build_parser()
    args = parser.parse_args(["backfill", "--max-consecutive-failures", "5"])
    assert args.max_consecutive_failures == 5
    assert isinstance(args.max_consecutive_failures, int)


def test_backfill_reverse_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["backfill", "--reverse"])
    assert args.reverse is True


def test_backfill_season_range_passes_through_unparsed():
    parser = cli.build_parser()
    args = parser.parse_args(["backfill", "--season-range", "2010_11:2024_25"])
    # CLI does not parse the range; that's compute_targeted_set's job
    assert args.season_range == "2010_11:2024_25"


# --------------------------------------------------------------------------- #
# Enum translation (§8.2)
# --------------------------------------------------------------------------- #


def test_translate_args_endpoint_strings_become_enum_members():
    parser = cli.build_parser()
    raw = parser.parse_args(["backfill", "--endpoint", "schedule", "traditional"])
    args = cli._translate_args(raw)
    assert args.endpoint == [Endpoint.SCHEDULE, Endpoint.BOXSCORE_TRADITIONAL]


def test_translate_args_season_type_strings_become_enum_members():
    parser = cli.build_parser()
    raw = parser.parse_args(["backfill", "--season-type", "regular", "playoffs"])
    args = cli._translate_args(raw)
    assert args.season_type == [SeasonType.REGULAR, SeasonType.PLAYOFFS]


def test_translate_args_preserves_user_order():
    """The translation step must not reorder; canonical ordering is the
    orchestrator's job (§3.2)."""
    parser = cli.build_parser()
    raw = parser.parse_args(["backfill", "--endpoint", "summary", "schedule", "advanced"])
    args = cli._translate_args(raw)
    assert args.endpoint == [
        Endpoint.BOXSCORE_SUMMARY,
        Endpoint.SCHEDULE,
        Endpoint.BOXSCORE_ADVANCED,
    ]


def test_translate_args_incremental_has_no_enum_lists():
    """Incremental doesn't have --endpoint or --season-type, so translation
    must be a no-op for it (no AttributeError)."""
    parser = cli.build_parser()
    raw = parser.parse_args(["incremental"])
    args = cli._translate_args(raw)
    assert args.command == "incremental"


# --------------------------------------------------------------------------- #
# main() dispatch
# --------------------------------------------------------------------------- #


def test_main_dispatches_to_backfill(monkeypatch):
    seen = {}

    def fake_backfill(args):
        seen["called"] = True
        seen["dry_run"] = args.dry_run
        seen["endpoint"] = args.endpoint
        return 0

    monkeypatch.setattr("nba_four_factors.cli.run_backfill", fake_backfill)
    monkeypatch.setattr(
        "nba_four_factors.cli.run_incremental",
        lambda args: pytest.fail("wrong driver dispatched"),
    )

    rc = cli.main(["backfill", "--dry-run"])

    assert rc == 0
    assert seen["called"] is True
    assert seen["dry_run"] is True
    # Translated to enums by the time the driver sees them
    assert all(isinstance(e, Endpoint) for e in seen["endpoint"])


def test_main_dispatches_to_incremental(monkeypatch):
    seen = {}

    def fake_incremental(args):
        seen["called"] = True
        seen["season"] = args.season
        return 0

    monkeypatch.setattr("nba_four_factors.cli.run_incremental", fake_incremental)
    monkeypatch.setattr(
        "nba_four_factors.cli.run_backfill",
        lambda args: pytest.fail("wrong driver dispatched"),
    )

    rc = cli.main(["incremental", "--season", "2024_25"])

    assert rc == 0
    assert seen["called"] is True
    assert seen["season"] == "2024_25"


def test_main_propagates_driver_exit_code(monkeypatch):
    monkeypatch.setattr("nba_four_factors.cli.run_backfill", lambda args: 42)
    assert cli.main(["backfill"]) == 42


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #


def test_main_invalid_subcommand_errors():
    with pytest.raises(SystemExit):
        cli.main(["bogus"])


def test_main_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        cli.main([])


def test_invalid_endpoint_choice_errors():
    with pytest.raises(SystemExit):
        cli.main(["backfill", "--endpoint", "preseason"])


def test_invalid_season_type_choice_errors():
    with pytest.raises(SystemExit):
        cli.main(["backfill", "--season-type", "summer_league"])


def test_max_consecutive_failures_non_int_errors():
    with pytest.raises(SystemExit):
        cli.main(["backfill", "--max-consecutive-failures", "ten"])


def test_process_subcommand_required_arguments():
    """`process` requires both --season and --season-type."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["process"])
    with pytest.raises(SystemExit):
        parser.parse_args(["process", "--season", "2024_25"])
    with pytest.raises(SystemExit):
        parser.parse_args(["process", "--season-type", "regular"])


def test_process_subcommand_parses_required_arguments():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "process",
            "--season",
            "2024_25",
            "--season-type",
            "regular",
        ]
    )
    assert args.command == "process"
    assert args.season == "2024_25"
    # argparse stores the raw string; translation happens later
    assert args.season_type == "regular"


def test_process_subcommand_invalid_season_type_errors():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "process",
                "--season",
                "2024_25",
                "--season-type",
                "preseason",
            ]
        )


# --------------------------------------------------------------------------- #
# Enum translation: process passes a SCALAR, not a list
# --------------------------------------------------------------------------- #


def test_translate_args_process_season_type_becomes_single_enum():
    """process subcommand has a single --season-type, not nargs='+'.
    _translate_args must produce a scalar enum, not a list."""
    parser = cli.build_parser()
    raw = parser.parse_args(
        [
            "process",
            "--season",
            "2024_25",
            "--season-type",
            "playoffs",
        ]
    )
    args = cli._translate_args(raw)
    assert args.season_type == SeasonType.PLAYOFFS
    assert not isinstance(args.season_type, list)


def test_translate_args_backfill_season_type_still_a_list():
    """Regression: backfill's --season-type uses nargs='+', should still
    translate to a list of enums after the polymorphic branch was added."""
    parser = cli.build_parser()
    raw = parser.parse_args(["backfill", "--season-type", "regular", "playoffs"])
    args = cli._translate_args(raw)
    assert args.season_type == [SeasonType.REGULAR, SeasonType.PLAYOFFS]
    assert isinstance(args.season_type, list)


# --------------------------------------------------------------------------- #
# main() dispatches to run_process
# --------------------------------------------------------------------------- #


def test_main_dispatches_to_process(monkeypatch):
    seen = {}

    def fake_process(args):
        seen["called"] = True
        seen["season"] = args.season
        seen["season_type"] = args.season_type
        return 0

    monkeypatch.setattr("nba_four_factors.cli.run_process", fake_process)
    monkeypatch.setattr(
        "nba_four_factors.cli.run_backfill",
        lambda args: pytest.fail("wrong driver dispatched"),
    )
    monkeypatch.setattr(
        "nba_four_factors.cli.run_incremental",
        lambda args: pytest.fail("wrong driver dispatched"),
    )

    rc = cli.main(["process", "--season", "2024_25", "--season-type", "regular"])

    assert rc == 0
    assert seen["called"] is True
    assert seen["season"] == "2024_25"
    # Translated to a SeasonType enum (not a list) by the time the driver sees it
    assert seen["season_type"] == SeasonType.REGULAR


def test_main_propagates_process_exit_code(monkeypatch):
    """If run_process returns 1 (e.g., raw JSON missing), main returns 1."""
    monkeypatch.setattr("nba_four_factors.cli.run_process", lambda args: 1)
    rc = cli.main(
        [
            "process",
            "--season",
            "2024_25",
            "--season-type",
            "regular",
        ]
    )
    assert rc == 1
