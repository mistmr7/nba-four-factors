"""Travel and schedule-fatigue features.

Travel is modeled as a continuous burden, not a relabeled home/away flag.
For each team's chronological game sequence within a season it computes the
great-circle distance flown since the previous game, days of rest, the
back-to-back flag, the signed timezone shift (eastward is positive and
circadian-harder), a trailing seven-day mileage total (the "home off a long
road trip" fatigue proxy), and a signed homestand / road-trip streak length.

These are deliberately distinct from the ``is_home`` indicator: a team can be
home and heavily travelled (just back from a western swing) or away and barely
travelled (an intra-metro game). That independent variation is what lets a
regression estimate a travel coefficient separately from a home-court one.

Design notes
* Venue of a game is the home team's arena, so for a team's row the venue is
  its own arena when ``is_home`` else the opponent's arena.
* Co-located teams (LAL/LAC identical, BKN/NYK a few miles apart) yield a
  near-zero leg automatically, capturing the "same metro, no real travel" case
  without a special list.
* Travel runs continuously across a season, regular season into play-in and
  playoffs, because fatigue does not reset at the playoff boundary. It does
  reset between seasons: a season opener has no previous game, so its leg is
  set to 0.0 (teams begin rested at home base) and ``days_rest`` is left null
  and flagged by ``is_season_opener``.
* Neutral-site games (``is_neutral``) are played at neither team's arena. The
  true venue is not stored in the processed layer, so the home arena is used
  as an approximation and the row is flagged. Across 1997-2026 these are under
  0.3 percent of games; drop them if exactness matters.

Output is one row per ``(game_id, team_id)`` so it merges into either the
two-row-per-game layout or a one-row home-perspective layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .arenas import ARENAS

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FEATURES_DIR = REPO_ROOT / "data" / "features"

_EARTH_RADIUS_MILES = 3958.7613

# Columns the feature builder needs from the processed layer.
_INPUT_COLS = [
    "game_id",
    "game_date",
    "season",
    "season_type",
    "team_id",
    "team_abbr",
    "opp_abbr",
    "is_home",
    "is_neutral",
]

_OUTPUT_COLS = [
    "game_id",
    "team_id",
    "season",
    "season_type",
    "game_date",
    "team_abbr",
    "venue_abbr",
    "venue_lat",
    "venue_lon",
    "is_home",
    "is_neutral",
    "is_season_opener",
    "leg_miles",
    "days_rest",
    "is_b2b",
    "tz_shift",
    "miles_7d",
    "streak",
]


def haversine_miles(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance in statute miles between two points.

    Vectorized over numpy arrays. Returns a float for scalar input.
    """
    lat1r, lon1r, lat2r, lon2r = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def _attach_venue(frame: pd.DataFrame) -> pd.DataFrame:
    """Add venue abbreviation, coordinates, and standard UTC offset per row."""
    out = frame.copy()
    venue_abbr = np.where(out["is_home"].to_numpy(), out["team_abbr"], out["opp_abbr"])
    out["venue_abbr"] = venue_abbr
    lat, lon, off = [], [], []
    for ab in venue_abbr:
        arena = ARENAS.get(ab)
        if arena is None:
            raise KeyError(f"No arena entry for abbreviation {ab!r}. Update arenas.py.")
        lat.append(arena.lat)
        lon.append(arena.lon)
        off.append(arena.utc_offset)
    out["venue_lat"] = lat
    out["venue_lon"] = lon
    out["_venue_off"] = off
    return out


def _signed_streak(is_home: np.ndarray) -> np.ndarray:
    """Signed consecutive home/road streak.

    +k on the kth game of a homestand, -k on the kth game of a road trip. The
    streak counts the current game inclusively.
    """
    streak = np.empty(len(is_home), dtype=int)
    run = 0
    prev: bool | None = None
    for i, home in enumerate(is_home):
        home_b = bool(home)
        if prev is None or home_b != prev:
            run = 1
        else:
            run += 1
        streak[i] = run if home_b else -run
        prev = home_b
    return streak


def _per_team_season(group: pd.DataFrame) -> pd.DataFrame:
    """Compute travel features for one team within one season, date-ordered."""
    g = group.sort_values("game_date").reset_index(drop=True)

    prev_lat = g["venue_lat"].shift(1)
    prev_lon = g["venue_lon"].shift(1)
    prev_off = g["_venue_off"].shift(1)
    prev_date = g["game_date"].shift(1)

    leg = haversine_miles(prev_lat, prev_lon, g["venue_lat"], g["venue_lon"])
    g["is_season_opener"] = prev_date.isna()
    # Opener: no previous game, treat as rested at home base.
    g["leg_miles"] = np.where(g["is_season_opener"], 0.0, leg)

    days_between = (g["game_date"] - prev_date).dt.days
    g["days_rest"] = days_between - 1
    g["is_b2b"] = days_between == 1
    g.loc[g["is_season_opener"], "days_rest"] = np.nan
    g.loc[g["is_season_opener"], "is_b2b"] = False

    g["tz_shift"] = (g["_venue_off"] - prev_off).fillna(0).astype(int)

    # Trailing seven-day mileage, current leg included, by calendar date.
    miles = g.set_index("game_date")["leg_miles"].rolling("7D").sum()
    g["miles_7d"] = miles.to_numpy()

    g["streak"] = _signed_streak(g["is_home"].to_numpy())
    return g


def build_travel_features(processed: pd.DataFrame) -> pd.DataFrame:
    """Build travel features from a processed per-(team, game) frame.

    Pure transform. ``processed`` must contain ``_INPUT_COLS``. Returns one row
    per ``(game_id, team_id)`` with ``_OUTPUT_COLS``.
    """
    missing = set(_INPUT_COLS) - set(processed.columns)
    if missing:
        raise KeyError(f"processed frame missing columns: {sorted(missing)}")

    work = processed[_INPUT_COLS].copy()
    work["game_date"] = pd.to_datetime(work["game_date"])
    work = _attach_venue(work)

    pieces = [_per_team_season(grp) for _, grp in work.groupby(["team_id", "season"], sort=False)]
    out = pd.concat(pieces, ignore_index=True)
    return out[_OUTPUT_COLS].sort_values(["season", "game_date", "team_id"]).reset_index(drop=True)


def load_processed_all() -> pd.DataFrame:
    """Load every processed season file (regular season and playoffs)."""
    frames = []
    for path in sorted(PROCESSED_DIR.glob("*/*.parquet")):
        frames.append(pd.read_parquet(path, columns=_INPUT_COLS))
    if not frames:
        raise FileNotFoundError(f"No processed parquet files under {PROCESSED_DIR}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Build travel features across all seasons and write to data/features/."""
    processed = load_processed_all()
    features = build_travel_features(processed)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "travel.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Wrote {len(features):,} rows to {out_path}")


if __name__ == "__main__":
    main()
