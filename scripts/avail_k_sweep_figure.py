"""Figure for the shrinkage-constant sweep promised in Methods 4.7.

Numbers from nba_four_factors.features.availability_shrunk (expanding-window
regression base, 2010-2025 test seasons):

    k    OOS MAE   gap closed, top imbalance decile
    6    10.3641   71.7%
    12   10.3639   73.9%   (chosen)
    20   10.3649   73.5%
    35   10.3672   72.4%
    raw  10.3708   66.4%   (no shrinkage reference)

Output: figures/modeling/avail_k_sweep.png
Run from repo root: python3 scripts/avail_k_sweep_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
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

K = [6, 12, 20, 35]
MAE = [10.3641, 10.3639, 10.3649, 10.3672]
CLOSED = [71.7, 73.9, 73.5, 72.4]
RAW_MAE = 10.3708
RAW_CLOSED = 66.4


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    ax1.plot(K, MAE, marker="o", ms=4, color="#1f6fb4")
    ax1.axhline(RAW_MAE, color="#888888", lw=1.0, ls="--")
    ax1.text(34, RAW_MAE - 0.0006, "raw (no shrinkage)", ha="right",
             fontsize=8.5, color="#666666")
    ax1.plot([12], [10.3639], marker="o", ms=8, mfc="none", color="#b03a2e")
    ax1.set_xticks(K)
    ax1.set_xlabel("Shrinkage constant k")
    ax1.set_ylabel("Out-of-sample margin MAE")
    ax1.set_title("Overall accuracy", fontsize=10)

    ax2.plot(K, CLOSED, marker="o", ms=4, color="#1f6fb4")
    ax2.axhline(RAW_CLOSED, color="#888888", lw=1.0, ls="--")
    ax2.text(34, RAW_CLOSED + 0.4, "raw (no shrinkage)", ha="right",
             fontsize=8.5, color="#666666")
    ax2.plot([12], [73.9], marker="o", ms=8, mfc="none", color="#b03a2e")
    ax2.set_xticks(K)
    ax2.set_ylim(60, 78)
    ax2.set_xlabel("Shrinkage constant k")
    ax2.set_ylabel("Gap closed, top imbalance decile (%)")
    ax2.set_title("Where it matters", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGS / "avail_k_sweep.png")
    plt.close(fig)
    print("wrote avail_k_sweep.png")


if __name__ == "__main__":
    main()
