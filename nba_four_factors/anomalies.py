"""Game-level anomaly registry for the four-factors pipeline.

Two registries:

* ``ANOMALOUS_GAME_IDS``: games excluded entirely from four-factors
  aggregates. Currently empty. See known_anomalies.md "Specific game
  anomalies".

* ``NEUTRAL_SITE_DATE_OVERRIDES``: per-season date ranges of games
  that nba.com schedules with normal home/away MATCHUP fields but
  were actually played at neutral venues. The processed layer applies
  these overrides via :func:`apply_known_neutral_overrides`. See
  known_anomalies.md "COVID-19 disruptions" for the 2019-20 Bubble
  case.

Do not add entries here without also documenting them in
known_anomalies.md.
"""

from __future__ import annotations

import pandas as pd

ANOMALOUS_GAME_IDS: frozenset[str] = frozenset()


NEUTRAL_SITE_DATE_OVERRIDES: dict[str, tuple[tuple[str, str], ...]] = {
    "2019_20": (("2020-07-30", "2020-10-11"),),
}


def apply_known_neutral_overrides(
    df: pd.DataFrame,
    *,
    season: str,
) -> pd.DataFrame:
    """Force is_neutral=True for games inside known neutral date ranges.

    Some games are scheduled by nba.com with normal vs./@ MATCHUP fields
    but were actually played at neutral venues. The 2019-20 Bubble at
    Disney World is the canonical case: all seeding, play-in, and
    playoff games from Jul 30 to Oct 11, 2020 were played at neutral
    arenas with no crowds, but the API records them with normal
    home/away designations.

    Parameters
    ----------
    df
        Tidy-schedule DataFrame with ``game_date`` (datetime),
        ``is_neutral``, and ``is_home`` columns populated by
        :func:`processed.schedule._tidy_schedule`.
    season
        Underscore-form season identifier (e.g. ``"2019_20"``).

    Returns
    -------
    A new DataFrame (not a view). For games inside override ranges,
    ``is_neutral`` is set to True and ``is_home`` is set to False to
    match the invariant the MATCHUP-based detection already enforces
    for neutral games. Games outside the ranges and seasons not listed
    in :data:`NEUTRAL_SITE_DATE_OVERRIDES` pass through unchanged.
    """
    if season not in NEUTRAL_SITE_DATE_OVERRIDES:
        return df

    df = df.copy()

    for start_str, end_str in NEUTRAL_SITE_DATE_OVERRIDES[season]:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        mask = (df["game_date"] >= start) & (df["game_date"] <= end)
        df.loc[mask, "is_neutral"] = True
        df.loc[mask, "is_home"] = False

    return df
