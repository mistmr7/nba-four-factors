"""Fetch box-score gaps where the schedule lists a game but the file is missing.

The historical backfill orchestrator marks a (season, season_type) pair
done once each box-score endpoint has been attempted for every scheduled
game, regardless of outcome. Games that failed during outages land in
the checkpoint's `failed` list and are not retried on subsequent
backfill invocations.

This script bypasses the orchestrator entirely: it compares each season's
schedule against the box-score files on disk, fetches whatever's missing
via nba_api, saves through the same storage helpers the orchestrator
uses, and updates the checkpoint files to reflect successful fetches.

Idempotent: skips files already on disk. Rate-limited via the existing
RateLimiter.

Usage:
    uv run python scripts/fill_gaps.py
    uv run python scripts/fill_gaps.py --dry-run
    uv run python scripts/fill_gaps.py --max-games 100

After completion, scripts/check_missing_games.py should show zeros (or
much smaller numbers) and the checkpoint files will list the newly
fetched games under `completed` rather than `failed`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from nba_api.stats.endpoints import (
    boxscoreadvancedv3,
    boxscoresummaryv3,
    boxscoretraditionalv3,
)

from nba_four_factors.api.rate_limiter import RateLimiter
from nba_four_factors.config import RAW_DIR, SEASONS, Endpoint, SeasonType
from nba_four_factors.storage.checkpoint import checkpoint_path
from nba_four_factors.storage.raw import exists_raw, raw_game_path, save_raw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fill_gaps")


_SEASON_TYPE_TO_SNAKE = {
    SeasonType.REGULAR: "regular_season",
    SeasonType.PLAYOFFS: "playoffs",
    SeasonType.PLAY_IN: "play_in",
}

_BOX_ENDPOINTS = (
    Endpoint.BOXSCORE_TRADITIONAL,
    Endpoint.BOXSCORE_ADVANCED,
    Endpoint.BOXSCORE_SUMMARY,
)

_FETCHERS = {
    Endpoint.BOXSCORE_TRADITIONAL: lambda gid: boxscoretraditionalv3.BoxScoreTraditionalV3(
        game_id=gid, timeout=60
    ).get_dict(),
    Endpoint.BOXSCORE_ADVANCED: lambda gid: boxscoreadvancedv3.BoxScoreAdvancedV3(
        game_id=gid, timeout=60
    ).get_dict(),
    Endpoint.BOXSCORE_SUMMARY: lambda gid: boxscoresummaryv3.BoxScoreSummaryV3(
        game_id=gid, timeout=60
    ).get_dict(),
}

_EXPECTED_KEY = {
    Endpoint.BOXSCORE_TRADITIONAL: "boxScoreTraditional",
    Endpoint.BOXSCORE_ADVANCED: "boxScoreAdvanced",
    Endpoint.BOXSCORE_SUMMARY: "boxScoreSummary",
}


def _read_schedule_game_ids(season: str, season_type: SeasonType) -> set[str]:
    """Return the set of game IDs in a season's leaguegamelog JSON."""
    snake = _SEASON_TYPE_TO_SNAKE[season_type]
    path = RAW_DIR / Endpoint.SCHEDULE.value / season / f"{snake}.json"
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


def _boxscore_game_ids_on_disk(endpoint: Endpoint) -> set[str]:
    base = RAW_DIR / endpoint.value
    if not base.exists():
        return set()
    return {p.stem for p in base.iterdir() if p.suffix == ".json"}


def _compute_gaps() -> list[tuple[Endpoint, str, str, SeasonType]]:
    """Return [(endpoint, game_id, season, season_type)] for every missing file.

    Iterates every (season, season_type) with a schedule on disk, diffs
    schedule game_ids against the files present for each box-score endpoint.
    """
    box_present = {ep: _boxscore_game_ids_on_disk(ep) for ep in _BOX_ENDPOINTS}

    gaps: list[tuple[Endpoint, str, str, SeasonType]] = []
    for season in SEASONS:
        for season_type in (SeasonType.REGULAR, SeasonType.PLAYOFFS):
            scheduled = _read_schedule_game_ids(season, season_type)
            if not scheduled:
                continue
            for endpoint in _BOX_ENDPOINTS:
                missing = scheduled - box_present[endpoint]
                for gid in sorted(missing):
                    gaps.append((endpoint, gid, season, season_type))
    return gaps


def _update_checkpoint(
    season: str,
    season_type: SeasonType,
    endpoint: Endpoint,
    fetched_ids: list[str],
) -> None:
    """Move newly fetched game IDs from `failed` to `completed` in the checkpoint."""
    path = checkpoint_path(season, season_type)
    if not path.exists():
        return

    with path.open() as f:
        ckpt = json.load(f)

    box = ckpt.get("box_scores", {}).get(endpoint.value)
    if box is None:
        return

    completed = set(box.get("completed", []))
    failed = set(box.get("failed", []))

    for gid in fetched_ids:
        completed.add(gid)
        failed.discard(gid)

    box["completed"] = sorted(completed)
    box["failed"] = sorted(failed)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(ckpt, f, indent=2)
    tmp.replace(path)


def _validate_payload(endpoint: Endpoint, payload: dict) -> bool:
    return _EXPECTED_KEY[endpoint] in payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work, make no API calls, write nothing.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Stop after this many successful fetches (for testing).",
    )
    args = parser.parse_args(argv)

    logger.info("computing gaps...")
    gaps = _compute_gaps()
    logger.info("total gaps found: %d", len(gaps))

    if args.dry_run:
        by_endpoint: dict[str, int] = {}
        for endpoint, _, _, _ in gaps:
            by_endpoint[endpoint.value] = by_endpoint.get(endpoint.value, 0) + 1
        for ep, n in sorted(by_endpoint.items()):
            logger.info("  %s: %d", ep, n)
        return 0

    rate_limiter = RateLimiter()
    fetched = 0
    skipped = 0
    failed: list[tuple[Endpoint, str, str]] = []
    fetched_by_pair: dict[tuple[str, SeasonType, Endpoint], list[str]] = {}

    for endpoint, game_id, season, season_type in gaps:
        target = raw_game_path(endpoint, game_id)

        if exists_raw(target):
            skipped += 1
            continue

        if args.max_games is not None and fetched >= args.max_games:
            logger.info("reached --max-games=%d, stopping", args.max_games)
            break

        rate_limiter.acquire()
        try:
            payload = _FETCHERS[endpoint](game_id)
        except Exception as e:
            failed.append((endpoint, game_id, f"{type(e).__name__}: {e}"))
            logger.warning("FETCH FAILED: %s %s: %s", endpoint.value, game_id, e)
            continue

        if not _validate_payload(endpoint, payload):
            failed.append((endpoint, game_id, "missing expected data key"))
            logger.warning(
                "MALFORMED: %s %s: keys=%s",
                endpoint.value,
                game_id,
                list(payload.keys()),
            )
            continue

        save_raw(target, payload)
        fetched += 1
        key = (season, season_type, endpoint)
        fetched_by_pair.setdefault(key, []).append(game_id)

        if fetched % 25 == 0:
            logger.info(
                "progress: fetched=%d failed=%d remaining=~%d",
                fetched,
                len(failed),
                len(gaps) - fetched - skipped - len(failed),
            )

    logger.info("updating checkpoints for %d (pair, endpoint) groups", len(fetched_by_pair))
    for (season, season_type, endpoint), ids in fetched_by_pair.items():
        _update_checkpoint(season, season_type, endpoint, ids)
        logger.info(
            "  %s %s %s: +%d completed",
            season,
            season_type.name.lower(),
            endpoint.value,
            len(ids),
        )

    logger.info(
        "DONE: fetched=%d skipped=%d failed=%d",
        fetched,
        skipped,
        len(failed),
    )

    if failed:
        logger.warning("FAILED GAMES (candidates for ANOMALOUS_GAME_IDS):")
        for endpoint, gid, reason in failed:
            logger.warning("  %s %s: %s", endpoint.value, gid, reason)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
