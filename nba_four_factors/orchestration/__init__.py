"""Orchestration layer.

Drives the per-(season, season_type) state machine owned by
``storage/checkpoint.py``.  Two top-level entry points:

    run_backfill(args)     -- one-shot historical sweep (historical.py)
    run_incremental(args)  -- nightly delta pull (incremental.py)

Both are dispatched from ``cli.py``.  See orchestration spec §2 for the
architecture overview and §10 for the control-flow pseudocode this layer
implements.
"""

from .historical import run_backfill
from .incremental import run_incremental

__all__ = ["run_backfill", "run_incremental"]
