"""Figures for the early-blend experiment and the alpha bug fix.

Reads data/features/early_blend_fallback_results.csv written by
scripts/early_blend_fallback.py and produces three figures in
figures/modeling/.

Run from repo root:
    python3 scripts/early_blend_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "features" / "early_blend_fallback_results.csv"
OUT = REPO / "figures" / "modeling"

BLUE = "#4C78A8"
GREEN = "#1D9E75"
ORANGE = "#E8A33D"
RED = "#C44E52"
GRAY = "#888780"

K = 7.3

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)


def fig_alpha_ramp():
    n = np.linspace(0, 40, 400)
    alpha = n / (n + K)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(0, 5, color=GRAY, alpha=0.15, label="seed-only (recent-form feature unavailable)")
    ax.plot(n, alpha, color=BLUE, lw=2.5, label="alpha = n / (n + k), k = 7.3")
    for gx, note in [(1, "12%"), (K, "50%"), (15, "67%"), (30, "80%")]:
        gy = gx / (gx + K)
        ax.plot([gx], [gy], "o", color=BLUE, ms=6)
        ax.annotate(note, (gx, gy), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax.set_xlabel("average games played (n)")
    ax.set_ylabel("weight on the model, alpha")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 40)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "early_blend_alpha_ramp.png")
    plt.close(fig)


def fig_bucket_mae(r: pd.DataFrame):
    bins = [(0, 3), (3, 5), (5, 10), (10, 15), (15, 100)]
    labels = ["(0,3]", "(3,5]", "(5,10]", "(10,15]", "(15+]"]
    arms = [
        ("seed only", "p0m", GRAY),
        ("blended model", "gated_m", BLUE),
        ("early reduced-model blend", "fb_m", GREEN),
    ]
    maes = {name: [] for name, _, _ in arms}
    ns = []
    for lo, hi in bins:
        d = r[(r.ngn > lo) & (r.ngn <= hi)]
        ns.append(len(d))
        for name, col, _ in arms:
            maes[name].append((d.home_margin - d[col]).abs().mean())
    x = np.arange(len(bins))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (name, _, color) in enumerate(arms):
        ax.bar(x + (i - 1) * w, maes[name], w, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb}\nn={n:,}" for lb, n in zip(labels, ns, strict=False)])
    ax.set_xlabel("average games played going into the game")
    ax.set_ylabel("margin MAE (points)")
    lo = min(min(v) for v in maes.values())
    hi = max(max(v) for v in maes.values())
    ax.set_ylim(lo - 0.15, hi + 0.15)
    ax.legend()
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "early_blend_bucket_mae.png")
    plt.close(fig)


def fig_threshold_sweep(r: pd.DataFrame):
    r = r.copy()
    r["ae_gated"] = (r.home_margin - r.gated_m).abs()
    r["ae_seed"] = (r.home_margin - r.p0m).abs()
    ts = list(range(6, 16))
    maes = []
    for t in ts:
        ae = np.where(r.ngn >= t, r.ae_gated, r.ae_seed)
        maes.append(float(np.mean(ae)))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ts, maes, "-o", color=BLUE, lw=2)
    ax.axvspan(6.5, 10.5, color=GRAY, alpha=0.12, label="flat region")
    ax.set_xlabel("minimum games played before the model receives weight (T)")
    ax.set_ylabel("overall margin MAE (points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "early_blend_threshold_sweep.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(RESULTS)
    fig_alpha_ramp()
    fig_bucket_mae(r)
    fig_threshold_sweep(r)
    for f in [
        "early_blend_alpha_ramp.png",
        "early_blend_bucket_mae.png",
        "early_blend_threshold_sweep.png",
    ]:
        print(OUT / f)


if __name__ == "__main__":
    main()
