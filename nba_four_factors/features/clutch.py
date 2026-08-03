"""Pull NBA clutch-time splits (last 5 minutes, score within 5) per season.

These are the real clutch-time stats, not the final-margin proxy used in the
league_clutch one-off. The data comes from the league dashboard endpoints, which
return every player (or team) for a season in a single call, so the whole history
is a few dozen calls, a couple of minutes.

We pull both Base (scoring, shooting, wins/losses, plus-minus) and Advanced (net
rating, true shooting, usage) for players and teams, and concatenate across seasons
with a SEASON column. Downstream analysis builds the clutch leaderboards.

Implementation note: the leaguedash endpoints take 40+ filter parameters. Rather
than hand-maintain that surface against our generic client, we use nba_api's typed
dashboard classes (correct defaults) with a manual pause for rate limiting. Needs
network and the project Python (3.11+), so it runs on your machine.

Run: python -m nba_four_factors.features.clutch            # regular season
     python -m nba_four_factors.features.clutch playoffs   # playoff clutch -> *_po
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FEAT = REPO / "data" / "features"
START, END = 2005, 2025  # season start years (2005-06 .. 2025-26)
SLEEP = 1.5  # seconds between calls (rate-limit courtesy)

# Pass "playoffs" to pull playoff clutch instead of regular season. Playoff output
# is written to clutch_*_po.parquet so it never overwrites the regular-season files.
MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "regular"
SEASON_TYPE = "Playoffs" if MODE == "playoffs" else "Regular Season"
SUFFIX = "_po" if MODE == "playoffs" else ""


def seasons():
    return [f"{y}_{(y + 1) % 100:02d}" for y in range(START, END + 1)]


def hyphen(s):
    return f"{s[:4]}-{s[5:]}"


def _pull(cls, season_hyphen, measure):
    return cls(
        season=season_hyphen,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense=measure,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def pull():
    from nba_api.stats.endpoints import leaguedashplayerclutch, leaguedashteamclutch

    buckets = {"player_base": [], "player_adv": [], "team_base": [], "team_adv": []}
    specs = [
        ("player_base", leaguedashplayerclutch.LeagueDashPlayerClutch, "Base"),
        ("player_adv", leaguedashplayerclutch.LeagueDashPlayerClutch, "Advanced"),
        ("team_base", leaguedashteamclutch.LeagueDashTeamClutch, "Base"),
        ("team_adv", leaguedashteamclutch.LeagueDashTeamClutch, "Advanced"),
    ]
    for s in seasons():
        sh = hyphen(s)
        for key, cls, measure in specs:
            try:
                df = _pull(cls, sh, measure)
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
            out.to_parquet(FEAT / f"clutch_{key}{SUFFIX}.parquet", index=False)
            print(
                f"clutch_{key}{SUFFIX}.parquet  {len(out):,} rows  ({out['SEASON'].nunique()} seasons)"
            )


def summary():
    p = FEAT / f"clutch_player_base{SUFFIX}.parquet"
    if not p.is_file():
        print("no clutch parquet yet; run the pull first")
        return
    d = pd.read_parquet(p)
    cols = [
        c for c in d.columns if c in ("PLAYER_NAME", "GP", "W", "L", "PTS", "FG_PCT", "PLUS_MINUS")
    ]
    print("\nclutch_player_base columns:", list(d.columns)[:20])
    jb = d[(d.get("PLAYER_ID") == 1627759)] if "PLAYER_ID" in d.columns else d.iloc[0:0]
    if len(jb):
        print("\nJaylen Brown clutch rows (sample):")
        print(jb[cols].head(12).to_string(index=False))


def main():
    pull()
    summary()


if __name__ == "__main__":
    main()
