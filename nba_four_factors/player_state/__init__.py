"""Player-state lane: player-level form tracking beneath the team-level models.

This subpackage is isolated from the thesis pipeline by design. It reads the
existing raw and processed stores and writes only to data/player_state/.
Nothing here modifies the team-level Parquet store, its manifest, or its
checkpoints, and nothing here may weaken the no-leakage claims of Chapter 4.

Rungs, mirroring the team-level ladder one level down:
    1. Hollinger Game Score fact table (gamescore.py, deterministic transform)
    2. Scalar OU state-space filter on per-36 Game Score (filter.py)
    3. Vector-state extension, adopted only if it earns its place
"""
