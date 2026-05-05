"""Data integrity check across raw, processed, and cross-layer reconciliation.

Run from repo root:
    uv run python scripts/integrity_check.py

Reports findings without modifying anything. Exit 0 if clean, 1 if any
issues found.

Six categories of checks:

1. Raw schedule completeness: each season has expected schedule files
2. Raw box-score completeness: every game in every schedule has all
   three box-score JSONs on disk. Scope-limited: only pairs where the
   processed Parquet ALSO exists are considered "in scope" for this
   check, since pairs without processed Parquets are by definition
   in-progress or deferred.
3. Processed layer completeness: each processed Parquet's row count
   matches 2 x games in the schedule for that (season, season_type)
4. Schema integrity: 39 canonical columns, correct dtypes, expected
   non-null columns are non-null
5. Cross-layer reconciliation: game IDs in processed Parquets exist
   as box-score files
6. Within-row sanity: identity invariants on the processed data
   (mathematical bounds, not basketball-plausibility bounds)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from nba_four_factors.config import (
    PROCESSED_DIR,
    RAW_DIR,
    SEASONS,
    Endpoint,
    SeasonType,
)
from nba_four_factors.storage.raw import (
    raw_game_path,
    raw_season_path,
    season_type_slug,
)

CANONICAL_SCHEMA: tuple[str, ...] = (
    "game_id",
    "game_date",
    "season",
    "season_type",
    "team_id",
    "team_abbr",
    "opp_team_id",
    "opp_abbr",
    "is_home",
    "is_neutral",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "tov",
    "pts",
    "opp_fgm",
    "opp_fga",
    "opp_fg3m",
    "opp_fg3a",
    "opp_ftm",
    "opp_fta",
    "opp_oreb",
    "opp_dreb",
    "opp_tov",
    "opp_pts",
    "off_efg_pct",
    "off_tov_pct",
    "off_orb_pct",
    "off_ft_rate",
    "def_efg_pct",
    "def_tov_pct",
    "def_orb_pct",
    "def_ft_rate",
    "margin",
)

KNOWN_CANCELED_GAMES: frozenset[str] = frozenset(
    {
        "0021201214",
    }
)

BOX_ENDPOINTS = (
    Endpoint.BOXSCORE_TRADITIONAL,
    Endpoint.BOXSCORE_ADVANCED,
    Endpoint.BOXSCORE_SUMMARY,
)


class Findings:
    def __init__(self) -> None:
        self.issues: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []

    def issue(self, category: str, message: str) -> None:
        self.issues.append((category, message))

    def warn(self, category: str, message: str) -> None:
        self.warnings.append((category, message))

    @property
    def clean(self) -> bool:
        return not self.issues


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _game_ids_in_schedule(season: str, season_type: SeasonType) -> set[str]:
    path = raw_season_path(Endpoint.SCHEDULE, season, season_type)
    if not path.exists():
        return set()
    with path.open() as f:
        payload = json.load(f)
    result_sets = payload.get("resultSets") or payload.get("resultSet") or []
    if not result_sets:
        return set()
    headers = result_sets[0]["headers"]
    rows = result_sets[0]["rowSet"]
    gid_idx = headers.index("GAME_ID")
    return {row[gid_idx] for row in rows}


def _processed_path(season: str, st: SeasonType) -> Path:
    return PROCESSED_DIR / season / f"{season_type_slug(st)}.parquet"


def check_raw_schedules(
    seasons: list[str], findings: Findings
) -> dict[tuple[str, SeasonType], set[str]]:
    section("[1] Raw schedule completeness")

    schedule_games: dict[tuple[str, SeasonType], set[str]] = {}

    for season in seasons:
        for st in SeasonType:
            path = raw_season_path(Endpoint.SCHEDULE, season, st)
            if not path.exists():
                if st == SeasonType.PLAY_IN and season < "2020_21":
                    continue
                schedule_games[(season, st)] = set()
                print(f"  MISSING: {season} {st.name.lower():8s} schedule")
                continue
            gids = _game_ids_in_schedule(season, st)
            schedule_games[(season, st)] = gids
            print(f"  OK:      {season} {st.name.lower():8s} schedule ({len(gids)} games)")
            if not gids:
                findings.warn("schedule", f"{season} {st.name} has zero games in schedule")

    return schedule_games


def check_raw_box_completeness(
    schedule_games: dict[tuple[str, SeasonType], set[str]],
    findings: Findings,
) -> None:
    """Check 2: every scheduled game (in scope) has all three box-score files."""
    section("[2] Raw box-score completeness (scoped to processed pairs)")

    in_scope_games: set[str] = set()
    skipped_pairs: list[str] = []

    for (season, st), gids in schedule_games.items():
        if _processed_path(season, st).exists():
            in_scope_games.update(gids)
        elif gids:
            skipped_pairs.append(f"{season} {st.name.lower()}")

    print(f"  In-scope games (pairs with processed Parquets): {len(in_scope_games)}")
    if skipped_pairs:
        print("  Skipped pairs (no processed Parquet, not in scope):")
        for p in sorted(skipped_pairs):
            print(f"    - {p}")

    for endpoint in BOX_ENDPOINTS:
        missing = [
            gid for gid in sorted(in_scope_games) if not raw_game_path(endpoint, gid).exists()
        ]
        if missing:
            print(f"  {endpoint.value}: {len(missing)} missing")
            for gid in missing[:10]:
                print(f"    - {gid}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
            findings.issue(
                "box_completeness",
                f"{endpoint.value}: {len(missing)} in-scope games missing on disk",
            )
        else:
            print(f"  {endpoint.value}: all {len(in_scope_games)} in-scope games present")


def check_processed_completeness(
    schedule_games: dict[tuple[str, SeasonType], set[str]],
    findings: Findings,
) -> dict[tuple[str, SeasonType], pd.DataFrame]:
    """Check 3: row count = 2 x scheduled games for each processed Parquet."""
    section("[3] Processed layer completeness")

    processed: dict[tuple[str, SeasonType], pd.DataFrame] = {}
    missing_pairs: list[str] = []

    for (season, st), gids in sorted(schedule_games.items()):
        path = _processed_path(season, st)
        if not path.exists():
            if not gids:
                continue
            missing_pairs.append(f"{season} {st.name.lower()}")
            continue

        df = pd.read_parquet(path)
        processed[(season, st)] = df

        expected = 2 * len(gids)
        actual = len(df)

        if actual == expected:
            print(f"  OK:      {season} {st.name.lower():8s} {actual} rows ({len(gids)} games)")
        else:
            diff = actual - expected
            symbol = "OVER" if diff > 0 else "UNDER"
            print(
                f"  {symbol}:    {season} {st.name.lower():8s} {actual} rows, expected {expected} (diff {diff:+d})"
            )
            findings.issue(
                "processed_count",
                f"{season} {st.name}: row count {actual}, expected {expected}",
            )

    if missing_pairs:
        print("\n  Pairs with schedules but no processed Parquet (informational):")
        for p in sorted(missing_pairs):
            print(f"    - {p}")
        for p in missing_pairs:
            findings.warn(
                "processed_pending", f"{p}: schedule exists but processed Parquet not yet built"
            )

    return processed


def check_schema(processed: dict, findings: Findings) -> None:
    section("[4] Schema integrity")

    NEVER_NULL_COLS = (
        "game_id",
        "game_date",
        "season",
        "season_type",
        "team_id",
        "team_abbr",
        "opp_team_id",
        "opp_abbr",
        "is_home",
        "is_neutral",
        "margin",
    )

    issues_found = 0
    for (season, st), df in sorted(processed.items()):
        label = f"{season} {st.name.lower()}"

        actual_cols = tuple(df.columns)
        if actual_cols != CANONICAL_SCHEMA:
            extra = set(actual_cols) - set(CANONICAL_SCHEMA)
            missing = set(CANONICAL_SCHEMA) - set(actual_cols)
            if extra or missing:
                print(f"  SCHEMA: {label}: extra={sorted(extra)} missing={sorted(missing)}")
                findings.issue("schema", f"{label}: schema mismatch")
                issues_found += 1
                continue
            else:
                print(f"  ORDER:  {label}: same columns, different order")
                findings.warn("schema_order", f"{label}: column order differs from canonical")

        nulls = {col: df[col].isna().sum() for col in NEVER_NULL_COLS if df[col].isna().any()}
        if nulls:
            print(f"  NULLS:  {label}: {nulls}")
            findings.issue("schema_nulls", f"{label}: nulls in {list(nulls.keys())}")
            issues_found += 1

    if issues_found == 0:
        print(
            f"  OK: all {len(processed)} processed Parquets have correct schema and no critical nulls"
        )


def check_cross_layer(processed: dict, findings: Findings) -> None:
    section("[5] Cross-layer reconciliation")

    issues_found = 0
    for (season, st), df in sorted(processed.items()):
        label = f"{season} {st.name.lower()}"
        unique_gids = set(df["game_id"].unique())

        missing_per_endpoint = {}
        for endpoint in BOX_ENDPOINTS:
            missing = [gid for gid in unique_gids if not raw_game_path(endpoint, gid).exists()]
            if missing:
                missing_per_endpoint[endpoint.value] = len(missing)

        if missing_per_endpoint:
            print(
                f"  ORPHAN: {label}: processed game IDs missing box-score files: {missing_per_endpoint}"
            )
            findings.issue("cross_layer", f"{label}: orphan game IDs in processed")
            issues_found += 1

    if issues_found == 0:
        print("  OK: all processed game IDs have backing box-score files")


def check_within_row_sanity(processed: dict, findings: Findings) -> None:
    """Check 6: identity invariants on the processed rows.

    Mathematical bounds only ([0, 1] for percentages, non-negative for
    rates). Basketball-plausibility outliers are real and not data errors.
    """
    section("[6] Within-row sanity")

    issues_found = 0

    for (season, st), df in sorted(processed.items()):
        label = f"{season} {st.name.lower()}"
        problems: list[str] = []

        self_pairs = (df["team_id"] == df["opp_team_id"]).sum()
        if self_pairs:
            problems.append(f"{self_pairs} rows with team_id == opp_team_id")

        margin_check = (df["pts"] - df["opp_pts"]) != df["margin"]
        bad_margin = margin_check.sum()
        if bad_margin:
            problems.append(f"{bad_margin} rows where margin != pts - opp_pts")

        gid_counts = df.groupby("game_id").size()
        not_paired = (gid_counts != 2).sum()
        if not_paired:
            problems.append(f"{not_paired} game_ids without exactly 2 rows")

        for col in (
            "off_efg_pct",
            "off_tov_pct",
            "off_orb_pct",
            "def_efg_pct",
            "def_tov_pct",
            "def_orb_pct",
        ):
            out_of_bounds = ((df[col] < 0.0) | (df[col] > 1.0)).sum()
            if out_of_bounds:
                problems.append(f"{out_of_bounds} rows with {col} outside [0, 1]")

        for col in ("off_ft_rate", "def_ft_rate"):
            out_of_bounds = (df[col] < 0.0).sum()
            if out_of_bounds:
                problems.append(f"{out_of_bounds} rows with {col} negative")

        zero_margins_mask = df["margin"] == 0
        unexpected_zero_margins = zero_margins_mask & ~df["game_id"].isin(KNOWN_CANCELED_GAMES)
        n_unexpected = int(unexpected_zero_margins.sum())
        n_expected = int(zero_margins_mask.sum() - unexpected_zero_margins.sum())

        if n_expected:
            print(
                f"  {label}: noted {n_expected} canceled-game rows (documented; see known_anomalies.md)"
            )
        if n_unexpected:
            problems.append(
                f"{n_unexpected} unexpected rows with margin == 0 "
                f"(NBA games can't tie; not in known_anomalies)"
            )

        if problems:
            print(f"  {label}:")
            for p in problems:
                print(f"    - {p}")
            findings.issue("within_row", f"{label}: {len(problems)} sanity issues")
            issues_found += 1

    if issues_found == 0:
        print(f"  OK: all {len(processed)} processed Parquets pass identity invariants")


def main() -> int:
    findings = Findings()

    seasons_present = [s for s in SEASONS if (RAW_DIR / "leaguegamelog" / s).is_dir()]
    print(f"Seasons present at raw layer: {len(seasons_present)}")
    print(f"  {', '.join(seasons_present)}")

    schedule_games = check_raw_schedules(seasons_present, findings)
    check_raw_box_completeness(schedule_games, findings)
    processed = check_processed_completeness(schedule_games, findings)
    check_schema(processed, findings)
    check_cross_layer(processed, findings)
    check_within_row_sanity(processed, findings)

    section("Summary")

    print(f"\nFindings: {len(findings.issues)} issues, {len(findings.warnings)} warnings")

    if findings.warnings:
        print("\nWarnings (informational):")
        for category, msg in findings.warnings:
            print(f"  [{category}] {msg}")

    if findings.issues:
        print("\nIssues (blocking):")
        for category, msg in findings.issues:
            print(f"  [{category}] {msg}")
        return 1

    print("\nAll integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
