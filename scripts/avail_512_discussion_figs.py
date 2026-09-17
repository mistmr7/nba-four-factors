"""Discussion figures: gap to Vegas by closing-line size and by game number.

Reads data/features/avail_512_where_we_miss.parquet (built by
scripts/avail_512_where_we_miss.py).

Outputs:
    figures/modeling/gap_by_line_size.png
    figures/modeling/gap_by_game_number.png

Run from repo root:
    python3 scripts/avail_512_discussion_figs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "data" / "features"
FIGS = REPO / "figures" / "modeling"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 300,
})

YLIM = (0.0, 0.42)


def gap_bars(d, key, order, xlabel, fname):
    g = d.groupby(key, observed=True).agg(
        avail=("ae1", "mean"), vegas=("aev", "mean")).reindex(order)
    gap = g.avail - g.vegas
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(range(len(order)), gap, color="#1f6fb4", width=0.62)
    for i, v in enumerate(gap):
        ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=9, color="#333333")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE gap to Vegas (points)")
    fig.tight_layout()
    fig.savefig(FIGS / fname)
    plt.close(fig)
    print(f"wrote {fname}")


def main() -> None:
    d = pd.read_parquet(FEAT / "avail_512_where_we_miss.parquet")
    gap_bars(d, "line_b", ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"],
             "Absolute Vegas closing line (points)", "gap_by_line_size.png")
    gap_bars(d, "gn_b", ["1-10", "11-20", "21-30", "31-40", "41-50",
                         "51-60", "61-70", "71+"],
             "Season game number (mean of both teams)", "gap_by_game_number.png")


if __name__ == "__main__":
    main()
