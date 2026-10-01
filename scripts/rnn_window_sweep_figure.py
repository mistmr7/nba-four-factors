"""Window-sweep figure for the RNN results section.

Best validation margin MAE per window length, organized from the joint
randomized encoder search (all knobs sampled together; this is the
best-per-window view of the same 138 runs behind the architecture table).

Output:
    figures/modeling/rnn_window_sweep.png

Run from repo root:
    python3 scripts/rnn_window_sweep_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "features" / "rnn_tuning_results.csv"
OUT = REPO / "figures" / "modeling" / "rnn_window_sweep.png"

BLUE = "#4C78A8"
GRAY = "#888780"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)


def main() -> None:
    r = pd.read_csv(CSV)
    best = r.groupby("seq_len").val_margin_mae.min()

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(best.index, best.values, "-o", color=BLUE, lw=1.8, ms=6)
    sel = 20
    ax.plot([sel], [best[sel]], "o", color=BLUE, ms=11, mfc="none", mew=1.6)
    for x, y in best.items():
        va, off = ("bottom", 6) if x != 15 else ("bottom", 6)
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, off),
            ha="center",
            va=va,
            fontsize=9,
        )
    ax.annotate(
        "selected",
        (sel, best[sel]),
        textcoords="offset points",
        xytext=(0, -16),
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    ax.set_xticks(list(best.index))
    ax.set_xlabel("window length L (prior games per team)")
    ax.set_ylabel("best validation margin MAE")
    lo, hi = best.min(), best.max()
    ax.set_ylim(lo - 0.03, hi + 0.05)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
