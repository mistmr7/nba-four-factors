"""Check for missing box-score files versus what the schedules say should exist.

Compares game IDs present in each season's leaguegamelog schedule against
the box-score files on disk. Reports gaps per (season, season_type, endpoint).

Usage:
    uv run python scripts/check_missing_games.py
    uv run python scripts/check_missing_games.py --verbose
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from nba_four_factors.config import RAW_DIR, SEASONS, Endpoint


def schedule_game_ids(season: str, season_type_snake: str) -> set[str]:
    """Read a saved leaguegamelog JSON and return the set of game IDs.

    Returns empty set if the schedule file doesn't exist.
    """
    path = RAW_DIR / Endpoint.SCHEDULE.value / season / f"{season_type_snake}.json"
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


def boxscore_game_ids(endpoint: str) -> set[str]:
    """Return the set of game IDs present as files for an endpoint."""
    base = RAW_DIR / endpoint
    if not base.exists():
        return set()
    return {p.stem for p in base.iterdir() if p.suffix == ".json"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every missing game_id, not just per-season counts",
    )
    args = parser.parse_args()

    box_endpoints = {
        "traditional": Endpoint.BOXSCORE_TRADITIONAL.value,
        "advanced": Endpoint.BOXSCORE_ADVANCED.value,
        "summary": Endpoint.BOXSCORE_SUMMARY.value,
    }

    box_present = {name: boxscore_game_ids(ep) for name, ep in box_endpoints.items()}

    season_types = [("regular", "regular_season"), ("playoffs", "playoffs")]

    total_missing_by_endpoint: dict[str, int] = defaultdict(int)
    missing_detail: dict[tuple[str, str, str], list[str]] = {}

    header = (
        f"{'season':<10}"
        f"{'type':<10}"
        f"{'scheduled':>10}"
        f"{'miss_trad':>10}"
        f"{'miss_adv':>10}"
        f"{'miss_sum':>10}"
    )
    print(header)
    print("-" * len(header))

    for season in SEASONS:
        for st_label, st_snake in season_types:
            scheduled = schedule_game_ids(season, st_snake)
            if not scheduled:
                continue

            row = f"{season:<10}{st_label:<10}{len(scheduled):>10}"
            for endpoint_name in ("traditional", "advanced", "summary"):
                missing = scheduled - box_present[endpoint_name]
                row += f"{len(missing):>10}"
                if missing:
                    total_missing_by_endpoint[endpoint_name] += len(missing)
                    missing_detail[(season, st_label, endpoint_name)] = sorted(missing)
            print(row)

    print()
    print("Total missing across all seasons:")
    for name in ("traditional", "advanced", "summary"):
        print(f"  {name:<12} {total_missing_by_endpoint[name]}")

    if args.verbose and missing_detail:
        print()
        print("Detail (every missing game_id):")
        for (season, st, ep), gids in sorted(missing_detail.items()):
            print(f"  {season} {st} {ep}: {gids}")


if __name__ == "__main__":
    main()
