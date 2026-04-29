"""Processed layer: raw LeagueGameLog JSON → tidy four-factors Parquet.

Public API:

* :func:`read_schedule`   — load and tidy a saved schedule JSON.
* :func:`add_factors`     — add the 8 four-factor columns + margin.
* :func:`process_pair`    — full pipeline: load → tidy → join → factors → write.

Naming helpers for the project's "human-readable + computer-readable"
convention (Session 10 decision):

* :func:`season_type_to_snake` — ``SeasonType.REGULAR`` → ``"regular_season"``.
* :func:`snake_to_human`       — ``"regular_season"`` → ``"Regular Season"``.

The Parquet column ``season_type`` stores the snake form; the manifest
layer (see ``orchestration/_io.write_pair_manifest``) stores the human
form.  Different audiences, different defaults; convert with the helpers.
"""

from __future__ import annotations

# Helpers come from a separate private module rather than being defined
# inline here, to keep the imports below from forming a cycle (the
# submodules need these helpers at import time).
from ._naming import season_type_to_snake, snake_to_human
from .factors import add_factors
from .pipeline import process_pair
from .schedule import read_schedule

__all__ = [
    "add_factors",
    "process_pair",
    "read_schedule",
    "season_type_to_snake",
    "snake_to_human",
]
