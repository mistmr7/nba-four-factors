"""Aggregate possession-parser validation across every built season.

Runs possessions.validate in quiet mode per season and writes one row
per season to data/features/possession_validation.csv. Run AFTER a full
`build all` so every season's parquet reflects the current parser.

    PYTHONPATH=. python3 scripts/possession_validation_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_four_factors.player_state.possessions import OUT_DIR, validate

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "features" / "possession_validation.csv"


def main() -> None:
    rows = []
    for f in sorted(OUT_DIR.glob("*_regular_season.parquet")):
        season = f.name.replace("_regular_season.parquet", "")
        rows.append(validate(season, quiet=True))
        r = rows[-1]
        print(
            f"{season}: pts exact {r['exact_pts_all']:.4f}  "
            f"oreb exact {r['exact_oreb']:.4f}  corr {r['poss_corr']:.3f}  "
            f"anomalies {r['anomalies']}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    worst = df.sort_values("exact_pts_all").head(3)[["season", "exact_pts_all"]]
    print("weakest seasons by points reconciliation:")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
