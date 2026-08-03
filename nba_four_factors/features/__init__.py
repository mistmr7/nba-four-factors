"""Feature-engineering layer.

Modules here turn the processed per-(team, game) four-factor frames into
model-ready features. Each module is self-contained and writes a parquet to
``data/features/`` keyed by ``(game_id, team_id)`` so features merge cleanly
into either the two-row-per-game layout or a one-row home-perspective layout.
"""

from __future__ import annotations
