"""Learned attention profile over the 20-game window.

Renders the average attention weight per window position saved by
rnn/attention.py (data/features/attention_profile.csv) in the thesis house
style, with the uniform-weight reference line.

Output:
    figures/modeling/attention_profile.png

Run from repo root:
    python3 scripts/attention_profile_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "features" / "attention_profile.csv"
OUT = REPO / "figures" / "modeling" / "attention_profile.png"

BLUE = "#4C78A8"
GRAY = "#888780"
RED = "#C44E52"

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
    p = pd.read_csv(CSV)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(p.position, p.weight, "-o", color=BLUE, lw=1.8, ms=4.5,
            label="learned attention weight")
    ax.axhline(1.0 / 20.0, color=RED, ls="--", lw=1.2,
               label="uniform weight (1/20)")
    new = p.weight.iloc[-1]
    old = p.weight.iloc[0]
    ax.annotate(f"{new:.3f}", (20, new), textcoords="offset points",
                xytext=(-2, 7), ha="center", fontsize=9)
    ax.annotate(f"{old:.3f}", (1, old), textcoords="offset points",
                xytext=(2, -13), ha="center", fontsize=9)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("position in the 20-game window (1 = oldest, 20 = most recent)")
    ax.set_ylabel("mean attention weight")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
