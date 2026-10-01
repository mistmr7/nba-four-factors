"""Prior-retention sweep figure: forced full takeover vs the uninterrupted blend.

Alpha is set to 1 for all games at or beyond the takeover point T, discarding
the preseason prior entirely from that point on. Values from the handoff
sensitivity experiment (appendix Table Y3): every takeover before roughly
game 30 increases error; beyond 30 differences sit within noise.

Output:
    figures/modeling/prior_takeover_sweep.png

Run from repo root:
    python3 scripts/prior_takeover_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "modeling" / "prior_takeover_sweep.png"

BLUE = "#4C78A8"
RED = "#C44E52"
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

T = [8, 10, 15, 20, 30, 41]
MAE = [9.8719, 9.8613, 9.8461, 9.8402, 9.8338, 9.8289]
BLEND = 9.8332


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(T, MAE, "-o", color=BLUE, lw=1.8, ms=5.5, label="full takeover at game T")
    ax.axhline(BLEND, color=RED, ls="--", lw=1.4, label=f"uninterrupted blend ({BLEND:.4f})")
    for x, y in zip(T, MAE, strict=False):
        off = 6 if y >= BLEND else -15
        ax.annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, off),
            ha="center",
            fontsize=8.5,
        )
    ax.set_ylim(min(MAE) - 0.006, max(MAE) + 0.006)
    ax.set_xticks(T)
    ax.set_xlabel("takeover point T (average games played)")
    ax.set_ylabel("overall margin MAE")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
