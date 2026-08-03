"""Pull and parse player season accolades (All-NBA, All-Star, All-Defensive, ...).

Two stages:

  pull   For every player who appears in our game logs or inactive lists, fetch
         the playerawards endpoint and save the raw JSON to
         data/raw/playerawards/<person_id>.json. Idempotent: a player whose file
         already exists is skipped, so the run resumes cleanly after an
         interruption. This stage needs network access and the project's Python
         (3.11+), so it runs on your machine, like the other pulls.

  parse  Read every raw file and build data/features/accolades.parquet, one row
         per (person_id, season, award), with an All-NBA / All-Defensive tier
         where applicable. Pure pandas; no network.

Run both:        python -m nba_four_factors.features.awards
Parse only:      python -m nba_four_factors.features.awards --parse

Seasons are normalized to our underscore form (the API's "2022-23" -> "2022_23").
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from nba_four_factors.config import RAW_DIR
from nba_four_factors.storage.raw import load_raw, save_raw

REPO = Path(__file__).resolve().parents[2]
FEAT = REPO / "data" / "features"
RAW = RAW_DIR / "playerawards"

# Description -> the accolade class we keep. Everything else (weekly/monthly
# honors, conference awards) is ignored for now.
AWARD_CLASS = {
    "All-NBA": "all_nba",
    "All-Defensive Team": "all_defensive",
    "NBA All-Star": "all_star",
    "NBA Most Valuable Player": "mvp",
    "NBA Defensive Player of the Year": "dpoy",
    "All-Rookie Team": "all_rookie",
    "NBA Rookie of the Year": "roy",
    "NBA Most Improved Player": "mip",
    "Sixth Man of the Year": "sixth_man",
}


def _norm_season(s) -> str | None:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if "-" in s and len(s) >= 7:
        start = s[:4]
        return f"{start}_{(int(start) + 1) % 100:02d}" if start.isdigit() else None
    if s.isdigit() and len(s) == 4:
        y = int(s)
        return f"{y - 1}_{y % 100:02d}"
    return None


def _player_ids() -> list[int]:
    ids = set()
    for name in ("player_game_logs.parquet", "inactives.parquet"):
        p = FEAT / name
        if p.is_file():
            ids.update(int(x) for x in pd.read_parquet(p)["person_id"].dropna().unique())
    return sorted(ids)


def pull():
    from nba_four_factors.api.client import Client
    from nba_four_factors.api.endpoints.player_awards import fetch_player_awards

    ids = _player_ids()
    client = Client()
    done = skipped = 0
    for i, pid in enumerate(ids, 1):
        path = RAW / f"{pid}.json"
        if path.is_file():
            skipped += 1
            continue
        try:
            save_raw(path, fetch_player_awards(client, pid))
            done += 1
        except Exception as exc:
            print(f"  WARN {pid}: {type(exc).__name__} {exc}", flush=True)
        if i % 100 == 0:
            print(f"  {i}/{len(ids)}  fetched {done}, skipped {skipped}", flush=True)
    print(f"pull done: fetched {done}, skipped {skipped}, total players {len(ids)}")


def parse():
    rows = []
    for f in sorted(RAW.glob("*.json")):
        try:
            d = load_raw(f)
        except Exception:
            continue
        rs = d.get("resultSets") or []
        if not rs:
            continue
        block = rs[0]
        idx = {h: i for i, h in enumerate(block.get("headers", []))}
        if "DESCRIPTION" not in idx:
            continue
        for r in block.get("rowSet", []):
            desc = r[idx["DESCRIPTION"]]
            cls = AWARD_CLASS.get(desc)
            if cls is None:
                continue
            season = _norm_season(r[idx["SEASON"]]) if "SEASON" in idx else None
            if season is None:
                continue
            tier = r[idx["ALL_NBA_TEAM_NUMBER"]] if "ALL_NBA_TEAM_NUMBER" in idx else None
            try:
                tier = int(tier) if tier not in (None, "", "NULL") else None
            except (ValueError, TypeError):
                tier = None
            rows.append((int(r[idx["PERSON_ID"]]), season, cls, tier))

    df = pd.DataFrame(rows, columns=["person_id", "season", "award", "tier"]).drop_duplicates()
    FEAT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEAT / "accolades.parquet", index=False)
    print(
        f"accolades.parquet  {len(df):,} rows  "
        f"({df['person_id'].nunique():,} players, {df['season'].nunique()} seasons)"
    )
    print("\nrows per award class:")
    print(df["award"].value_counts().to_string())
    print("\nAll-NBA by tier:")
    print(df[df["award"] == "all_nba"]["tier"].value_counts().sort_index().to_string())


def main():
    if "--parse" in sys.argv:
        parse()
        return
    pull()
    parse()


if __name__ == "__main__":
    main()
