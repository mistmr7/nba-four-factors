"""Era-accurate arena reference table.

Maps each team abbreviation as it appears in the processed layer to the
location of that franchise's home arena and the arena's standard UTC offset.

The processed ``team_abbr`` is already era-accurate, so relocations and
renames are encoded by the abbreviation itself and need no season logic:

* ``NJN`` -> New Jersey (Meadowlands), ``BKN`` -> Brooklyn (Barclays)
* ``SEA`` -> Seattle, ``OKC`` -> Oklahoma City
* ``VAN`` -> Vancouver, ``MEM`` -> Memphis
* ``CHH`` / ``CHA`` -> Charlotte (both eras)
* ``NOH`` / ``NOP`` -> New Orleans, ``NOK`` -> Oklahoma City (the 2005-06 and
  2006-07 Hurricane Katrina relocation seasons)

Co-located teams share arena coordinates (LAL/LAC) or sit within a few miles
(BKN/NYK), so an intra-metro "road" game produces a near-zero travel leg
automatically. No special-case metro list is needed.

UTC offsets are standard (non-DST). Daylight saving is intentionally ignored:
the whole continental US shifts together, so the *direction* of a timezone
change (the quantity that matters for circadian load) is unaffected, and
Arizona's lack of DST is captured by giving Phoenix the mountain offset.
"""

from __future__ import annotations

from typing import NamedTuple


class Arena(NamedTuple):
    """Home-arena location for one team abbreviation."""

    city: str
    lat: float
    lon: float
    utc_offset: int  # standard offset in hours, e.g. Eastern is -5


# Abbreviation -> Arena. Coordinates are the arena (or arena's metro center)
# to the nearest few hundred meters, which is far finer than inter-city travel
# legs require.
ARENAS: dict[str, Arena] = {
    "ATL": Arena("Atlanta", 33.7573, -84.3963, -5),
    "BOS": Arena("Boston", 42.3662, -71.0621, -5),
    "BKN": Arena("Brooklyn", 40.6826, -73.9754, -5),
    "NJN": Arena("East Rutherford", 40.8118, -74.0682, -5),
    "CHA": Arena("Charlotte", 35.2251, -80.8392, -5),
    "CHH": Arena("Charlotte", 35.2251, -80.8392, -5),
    "CHI": Arena("Chicago", 41.8807, -87.6742, -6),
    "CLE": Arena("Cleveland", 41.4965, -81.6882, -5),
    "DAL": Arena("Dallas", 32.7905, -96.8104, -6),
    "DEN": Arena("Denver", 39.7487, -105.0077, -7),
    "DET": Arena("Detroit", 42.3411, -83.0553, -5),
    "GSW": Arena("San Francisco", 37.7680, -122.3877, -8),
    "HOU": Arena("Houston", 29.7508, -95.3621, -6),
    "IND": Arena("Indianapolis", 39.7639, -86.1555, -5),
    "LAC": Arena("Los Angeles", 34.0430, -118.2673, -8),
    "LAL": Arena("Los Angeles", 34.0430, -118.2673, -8),
    "MEM": Arena("Memphis", 35.1382, -90.0506, -6),
    "VAN": Arena("Vancouver", 49.2778, -123.1089, -8),
    "MIA": Arena("Miami", 25.7814, -80.1870, -5),
    "MIL": Arena("Milwaukee", 43.0451, -87.9172, -6),
    "MIN": Arena("Minneapolis", 44.9795, -93.2760, -6),
    "NOH": Arena("New Orleans", 29.9490, -90.0821, -6),
    "NOP": Arena("New Orleans", 29.9490, -90.0821, -6),
    "NOK": Arena("Oklahoma City", 35.4634, -97.5151, -6),
    "NYK": Arena("New York", 40.7505, -73.9934, -5),
    "OKC": Arena("Oklahoma City", 35.4634, -97.5151, -6),
    "ORL": Arena("Orlando", 28.5392, -81.3839, -5),
    "PHI": Arena("Philadelphia", 39.9012, -75.1720, -5),
    "PHX": Arena("Phoenix", 33.4457, -112.0712, -7),
    "POR": Arena("Portland", 45.5316, -122.6668, -8),
    "SAC": Arena("Sacramento", 38.5802, -121.4997, -8),
    "SAS": Arena("San Antonio", 29.4270, -98.4375, -6),
    "SEA": Arena("Seattle", 47.6221, -122.3540, -8),
    "TOR": Arena("Toronto", 43.6435, -79.3791, -5),
    "UTA": Arena("Salt Lake City", 40.7683, -111.9011, -7),
    "WAS": Arena("Washington", 38.8981, -77.0209, -5),
}
