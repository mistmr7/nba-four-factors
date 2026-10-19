"""Pull Barttorvik player-season stats (college, 2008 onward) for the preseason lane.

One CSV per season from barttorvik.com/getadvstats.php, saved untouched
under data/raw/preseason/barttorvik/<year>.csv (year is the season's
ending year, so 2015 is 2014-15). The file has no header; column names
are assigned in nba_four_factors/preseason/college.py. Reruns skip years
already on disk unless --refresh is passed (use it on the current year).

Run from the repo root:

    PYTHONPATH=. uv run python scripts/pull_barttorvik.py
    PYTHONPATH=. uv run python scripts/pull_barttorvik.py --years 2026 --refresh
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "raw" / "preseason" / "barttorvik"
URL = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"
FIRST_YEAR = 2008
PAUSE_SECONDS = 2.0


def pull(year: int, refresh: bool) -> str:
    f = OUT / f"{year}.csv"
    if f.exists() and not refresh:
        return "cached"
    req = urllib.request.Request(URL.format(year=year), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    if len(body) < 10_000:
        return f"suspiciously small ({len(body)} bytes), not saved"
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_bytes(body)
    time.sleep(PAUSE_SECONDS)
    return f"ok ({len(body) // 1024} KB)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    years = args.years or list(range(FIRST_YEAR, args.last + 1))
    for y in years:
        print(f"{y}: {pull(y, args.refresh)}")


if __name__ == "__main__":
    main()
