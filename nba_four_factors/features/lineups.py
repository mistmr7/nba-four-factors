"""Pull 5-man lineup stats per season (leaguedashlineups).

Season-level dashboard, one call per season per measure, so the whole history is a
few dozen calls. We pull Advanced (net, offensive, defensive rating, the headline
lineup signal) and Base (minutes, raw shooting) for 5-man units, concatenated with
a SEASON column. This is the lighter, pre-aggregated route to lineup-level
information, short of full play-by-play.

Same local-run pattern as the clutch and awards pulls (nba_api typed endpoint,
manual pause, needs network and Python 3.11+).

Run: python -m nba_four_factors.features.lineups
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FEAT = REPO / "data" / "features"
START, END = 2007, 2025  # lineup data is reliable from the late 2000s on
SLEEP = 1.5
SEASON_TYPE = "Regular Season"


def seasons():
    return [f"{y}_{(y + 1) % 100:02d}" for y in range(START, END + 1)]


def hyphen(s):
    return f"{s[:4]}-{s[5:]}"


def pull():
    from nba_api.stats.endpoints import leaguedashlineups

    buckets = {"adv": [], "base": []}
    for s in seasons():
        sh = hyphen(s)
        for key, measure, per in [
            ("adv", "Advanced", "Per100Possessions"),
            ("base", "Base", "PerGame"),
        ]:
            try:
                df = leaguedashlineups.LeagueDashLineups(
                    season=sh,
                    season_type_all_star=SEASON_TYPE,
                    measure_type_detailed_defense=measure,
                    per_mode_detailed=per,
                    group_quantity=5,
                ).get_data_frames()[0]
                df["SEASON"] = s
                buckets[key].append(df)
            except Exception as exc:
                print(f"  WARN {s} {key}: {type(exc).__name__} {exc}", flush=True)
            time.sleep(SLEEP)
        print(f"  {s} done", flush=True)

    FEAT.mkdir(parents=True, exist_ok=True)
    for key, frames in buckets.items():
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out.to_parquet(FEAT / f"lineups_{key}.parquet", index=False)
            print(f"lineups_{key}.parquet  {len(out):,} rows  ({out['SEASON'].nunique()} seasons)")


def main():
    pull()


if __name__ == "__main__":
    main()
