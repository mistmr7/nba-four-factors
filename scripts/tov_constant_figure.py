"""Season-by-season counted free-throw constant, and paragraph numbers.

For each season, classifies every free throw in the play-by-play data
by trip type and computes the terminal-trip rate: trips that end a
possession (ordinary 2-shot and 3-shot trips) divided by total FTA.
And-ones, technicals, flagrants with ball retained, and clear-path
trips contribute FTA but no termination, which is why the rate sits
below 0.5.

Also prints the sensitivity numbers for the thesis paragraph:
swapping the modern counted rate in for 0.44 in the TOV% formula,
measured on the game-level TOV% differential.

Outputs:
    figures/modeling/tov_constant_by_season.png
    data/features/tov_constant_by_season.csv

Run from repo root:
    python3 scripts/tov_constant_figure.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PBP = REPO / "data" / "raw" / "playbyplay"
OUT_FIG = REPO / "figures" / "modeling"
OUT_CSV = REPO / "data" / "features" / "tov_constant_by_season.csv"

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


def counted_k_by_season() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(PBP / "*_regular_season.parquet"))):
        season = Path(f).name.replace("_regular_season.parquet", "")
        d = pd.read_parquet(f, columns=["actionType", "subType"])
        s = d.loc[d.actionType == "Free Throw", "subType"].fillna("")
        fta = len(s)
        if fta == 0:
            continue
        terminal = (s == "Free Throw 1 of 2").sum() + (s == "Free Throw 1 of 3").sum()
        rows.append(
            {
                "season": season,
                "fta": fta,
                "terminal_trips": int(terminal),
                "counted_k": terminal / fta,
                "and_one_1of1": int((s == "Free Throw 1 of 1").sum()),
                "technical_fta": int(s.str.contains("Technical").sum()),
                "flagrant_cp_fta": int(s.str.contains("Flagrant|Clear Path").sum()),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_numbers() -> None:
    files = sorted(glob.glob(str(REPO / "data" / "processed" / "*" / "regular_season.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df.pts + df.opp_pts > 0].copy()

    def tovd(k: float) -> pd.Series:
        t = df.tov / (df.fga + k * df.fta + df.tov)
        o = df.opp_tov / (df.opp_fga + k * df.opp_fta + df.opp_tov)
        return o - t

    base, alt = tovd(0.44), tovd(0.419)
    ch = (alt - base) * 100
    print("sensitivity of the TOV% differential, 0.44 vs counted modern 0.419:")
    print(f"  correlation      {np.corrcoef(base, alt)[0, 1]:.5f}")
    print(f"  differential SD  {base.std() * 100:.2f} pp")
    print(f"  mean |change|    {ch.abs().mean():.3f} pp")
    print(f"  max |change|     {ch.abs().max():.3f} pp")


def main() -> None:
    ks = counted_k_by_season()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    ks.to_csv(OUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(ks))
    ax.plot(x, ks.counted_k, "-o", color=BLUE, ms=4.5, lw=1.8, label="counted terminal-trip rate")
    ax.axhline(0.44, color=RED, ls="--", lw=1.4, label="literature constant (0.44)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [s.replace("_", "-") for s in ks.season], rotation=60, ha="right", fontsize=8
    )
    ax.set_ylabel("terminal FT trips per FTA")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "tov_constant_by_season.png")
    plt.close(fig)

    print(ks[["season", "counted_k"]].round(4).to_string(index=False))
    print(f"\npooled counted k: {ks.terminal_trips.sum() / ks.fta.sum():.4f}")
    print(f"range: {ks.counted_k.min():.4f} to {ks.counted_k.max():.4f}")
    print(f"figure: {OUT_FIG / 'tov_constant_by_season.png'}")
    print()
    sensitivity_numbers()


if __name__ == "__main__":
    main()
