"""Figures for Results 5.12 from the avail_512 evidence CSVs.

Outputs:
    figures/modeling/avail_decile_mae.png
    figures/modeling/avail_gap_closed_by_season.png

Run from repo root:
    python3 scripts/avail_512_figures.py
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

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 300,
    }
)


def fig_decile() -> None:
    d = pd.read_csv(FEAT / "avail_512_decile.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(
        d.dec, d.kalman, marker="o", ms=4, color="#444444", label="Kalman filter (no availability)"
    )
    ax.plot(d.dec, d.avail, marker="s", ms=4, color="#1f6fb4", label="Kalman + availability")
    ax.plot(d.dec, d.vegas, marker="^", ms=4, color="#b03a2e", label="Vegas closing line")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decile of absence-value imbalance (1 = balanced, 10 = most imbalanced)")
    ax.set_ylabel("Margin MAE (points)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "avail_decile_mae.png")
    plt.close(fig)
    print("wrote avail_decile_mae.png")


def fig_season() -> None:
    s = pd.read_csv(FEAT / "avail_512_season.csv")
    overall = 100 * (s.kalman.mean() - s.avail.mean()) / (s.kalman.mean() - s.vegas.mean())
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(s.test_yr, s.closed_pct, color="#1f6fb4", width=0.72)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.axhline(overall, color="#b03a2e", lw=1.0, ls="--")
    ax.text(2006.7, overall + 1.5, f"overall {overall:.0f}%", color="#b03a2e", fontsize=9)
    for lo, hi, y in [(2007, 2012, 5.5), (2013, 2019, 17.3), (2020, 2025, 50.9)]:
        ax.plot([lo - 0.36, hi + 0.36], [y, y], color="#444444", lw=1.2)
        ax.text((lo + hi) / 2, y + 2.0, f"{y:.0f}%", ha="center", fontsize=9, color="#444444")
    ax.set_xticks(range(2007, 2026, 3))
    ax.set_xlabel("Test season")
    ax.set_ylabel("Share of Kalman-to-Vegas gap closed (%)")
    fig.tight_layout()
    fig.savefig(FIGS / "avail_gap_closed_by_season.png")
    plt.close(fig)
    print("wrote avail_gap_closed_by_season.png")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_decile()
    fig_season()
