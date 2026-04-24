"""Game-level anomaly registry for the four-factors pipeline.

Games listed in ANOMALOUS_GAME_IDS are excluded from four-factors
aggregates. See known_anomalies.md at repo root for documentation
of each entry's rationale. Do not add entries here without also
documenting them in known_anomalies.md.
"""

from __future__ import annotations

# Currently empty. Populate as concrete game_id values are identified
# during schedule reconciliation. Each addition must be accompanied by
# an entry in known_anomalies.md under "Specific game anomalies"
# explaining what the anomaly is and why exclusion is the right handling.
ANOMALOUS_GAME_IDS: frozenset[str] = frozenset()
