"""Shared thesis figure style for all EDA and modeling figures.

Usage, immediately after any seaborn set_theme call (or after imports):

    from thesis_style import apply_thesis_style
    apply_thesis_style()

What it enforces, so every figure matches the thesis:
- Serif font to match the document body text
- No top or right spines, muted grid, frameless legends
- The project palette as the default color cycle
- 300 dpi on every save, regardless of the dpi passed at the call site
- No figure suptitles, and no axes title on single-panel figures.
  Multi-panel figures keep their per-axes titles, which act as panel
  labels. Colorbars and other thin axes are ignored when deciding
  whether a figure is single-panel. Figure captions in the thesis
  carry the description instead.
"""

from __future__ import annotations

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
from cycler import cycler

BLUE = "#4C78A8"
GREEN = "#1D9E75"
ORANGE = "#EF9F27"
PURPLE = "#534AB7"
RED = "#D62728"
GRAY = "#888780"
PALETTE = [BLUE, GREEN, ORANGE, PURPLE, RED, GRAY]

_RC = {
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


def _main_panel_count(fig) -> int:
    """Count visually distinct full-size panels, ignoring colorbars and twins."""
    positions = set()
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        bounds = ax.get_position().bounds
        if bounds[2] < 0.12 or bounds[3] < 0.12:
            continue
        positions.add(tuple(round(v, 2) for v in bounds))
    return max(len(positions), 1)


def _strip_titles(fig) -> None:
    sup = getattr(fig, "_suptitle", None)
    if sup is not None:
        sup.set_visible(False)
    if _main_panel_count(fig) == 1:
        for ax in fig.axes:
            ax.set_title("")


def apply_thesis_style() -> None:
    try:
        import seaborn as sns

        sns.set_theme(style="white", palette=PALETTE)
    except ImportError:
        pass
    plt.rcParams.update(_RC)
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)

    if getattr(matplotlib.figure.Figure.savefig, "_thesis_wrapped", False):
        return

    original = matplotlib.figure.Figure.savefig

    def savefig(self, *args, **kwargs):
        _strip_titles(self)
        kwargs["dpi"] = 300
        return original(self, *args, **kwargs)

    savefig._thesis_wrapped = True
    matplotlib.figure.Figure.savefig = savefig
