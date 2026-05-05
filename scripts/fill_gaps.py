"""One-off: fetch the 50 box-score gaps left over from the 2015-2020 backfill.

Bypasses the orchestrator (which considers the affected season pairs
'already done' even though specific games are missing) and writes
JSON files in the exact format and path scheme that storage.raw uses.

Idempotent: skips files already on disk. Rate-limited via the existing
RateLimiter so the run plays nice with nba.com.

Run:
    uv run python scripts/fill_gaps.py
    uv run python scripts/fill_gaps.py --dry-run

After completion, the JSON files land where the processed pipeline
expects them. Manifests are NOT updated; if downstream code relies on
manifest-counted-vs-disk equality, that's a separate fix.
"""

from __future__ import annotations

import argparse
import logging
import sys

from nba_api.stats.endpoints import (
    boxscoreadvancedv3,
    boxscoresummaryv3,
    boxscoretraditionalv3,
)

from nba_four_factors.api.rate_limiter import RateLimiter
from nba_four_factors.config import Endpoint
from nba_four_factors.storage.raw import exists_raw, raw_game_path, save_raw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fill_gaps")


GAPS: list[tuple[Endpoint, str]] = [
    (Endpoint.BOXSCORE_ADVANCED, "0021300734"),
    (Endpoint.BOXSCORE_ADVANCED, "0021300735"),
    (Endpoint.BOXSCORE_ADVANCED, "0021300736"),
    (Endpoint.BOXSCORE_ADVANCED, "0021300737"),
    (Endpoint.BOXSCORE_ADVANCED, "0021300738"),
    (Endpoint.BOXSCORE_ADVANCED, "0021300739"),
    (Endpoint.BOXSCORE_ADVANCED, "0021300740"),
    (Endpoint.BOXSCORE_ADVANCED, "0021300742"),
    (Endpoint.BOXSCORE_ADVANCED, "0021300746"),
    (Endpoint.BOXSCORE_ADVANCED, "0021300752"),
    (Endpoint.BOXSCORE_SUMMARY, "0021200037"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200024"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200025"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200041"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200043"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200073"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200091"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200092"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200093"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200213"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200668"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200694"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200695"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200722"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200753"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200885"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200886"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200914"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200932"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200955"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022200984"),
    (Endpoint.BOXSCORE_TRADITIONAL, "0022201031"),
    (Endpoint.BOXSCORE_SUMMARY, "0042300224"),
    (Endpoint.BOXSCORE_SUMMARY, "0042300225"),
    (Endpoint.BOXSCORE_SUMMARY, "0042300314"),
]

_FETCHERS = {
    Endpoint.BOXSCORE_TRADITIONAL: lambda gid: boxscoretraditionalv3.BoxScoreTraditionalV3(
        game_id=gid, timeout=30
    ).get_dict(),
    Endpoint.BOXSCORE_ADVANCED: lambda gid: boxscoreadvancedv3.BoxScoreAdvancedV3(
        game_id=gid, timeout=30
    ).get_dict(),
    Endpoint.BOXSCORE_SUMMARY: lambda gid: boxscoresummaryv3.BoxScoreSummaryV3(
        game_id=gid, timeout=30
    ).get_dict(),
}


def _validate_payload(endpoint: Endpoint, payload: dict) -> bool:
    """Cheap sanity check: response has the expected top-level data key.

    The payloads consistently have 'meta' plus an endpoint-specific data key.
    For traditional that's 'boxScoreTraditional', etc. We don't validate
    structure beyond that here.
    """
    expected = {
        Endpoint.BOXSCORE_TRADITIONAL: "boxScoreTraditional",
        Endpoint.BOXSCORE_ADVANCED: "boxScoreAdvanced",
        Endpoint.BOXSCORE_SUMMARY: "boxScoreSummary",
    }
    return expected[endpoint] in payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work, make no API calls, write nothing.",
    )
    args = parser.parse_args(argv)

    rate_limiter = RateLimiter()

    fetched = 0
    skipped = 0
    failed: list[tuple[Endpoint, str, str]] = []

    for endpoint, game_id in GAPS:
        target = raw_game_path(endpoint, game_id)

        if exists_raw(target):
            skipped += 1
            logger.info("skip (already on disk): %s %s", endpoint.value, game_id)
            continue

        if args.dry_run:
            logger.info("would fetch: %s %s -> %s", endpoint.value, game_id, target)
            continue

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
        logger.info("fetched: %s %s", endpoint.value, game_id)

    logger.info(
        "DONE: fetched=%d skipped=%d failed=%d total=%d",
        fetched,
        skipped,
        len(failed),
        len(GAPS),
    )

    if failed:
        logger.warning("FAILED GAMES:")
        for endpoint, gid, reason in failed:
            logger.warning("  %s %s: %s", endpoint.value, gid, reason)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
