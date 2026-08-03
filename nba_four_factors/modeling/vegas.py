"""Vegas preseason win total to team-strength mapping (the M0 seed).

Default answer to Session 14 open question 2. Converts a preseason over/under
win total into a prior team strength on the points scale by fitting, across all
team-seasons with a market line, realized regular-season net rating (average
point margin per game) on the preseason win total. The fitted line lets a team
be seeded at game one with a points-scale strength, and a game's seed margin is
then home_strength minus away_strength plus home court.

A wins-on-line fit is reported alongside as a calibration check (how well the
market predicted realized wins).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
VEGAS = REPO / "data" / "vegas" / "nba_preseason_win_totals_1997_2026.parquet"
FIGDIR = REPO / "figures" / "modeling"
OUT = REPO / "data" / "features" / "vegas_seed_mapping.json"


def realized_team_seasons() -> pd.DataFrame:
    """Per team-season realized wins, games, and net rating from box scores."""
    files = sorted(glob.glob(str(PROC / "*/regular_season.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[~df["is_neutral"]]
    g = df.assign(w=(df["margin"] > 0).astype(int)).groupby(["season", "team_id"])
    return g.agg(wins=("w", "sum"), games=("w", "size"), net=("margin", "mean")).reset_index()


def fit_mapping() -> dict:
    real = realized_team_seasons()
    veg = pd.read_parquet(VEGAS)[["season", "team_id", "win_total", "actual_wins"]]
    d = real.merge(veg, on=["season", "team_id"], how="inner").dropna(subset=["win_total"])

    # Strength mapping: net rating on preseason win total.
    bn = np.polyfit(d["win_total"], d["net"], 1)
    net_pred = np.polyval(bn, d["win_total"])
    r2_net = 1 - np.sum((d["net"] - net_pred) ** 2) / np.sum((d["net"] - d["net"].mean()) ** 2)

    # Calibration: realized wins on preseason line.
    bw = np.polyfit(d["win_total"], d["wins"], 1)
    mae_line = float(np.mean(np.abs(d["wins"] - d["win_total"])))

    mapping = {
        "net_on_wintotal": {"slope": float(bn[0]), "intercept": float(bn[1]), "r2": float(r2_net)},
        "wins_on_wintotal": {"slope": float(bw[0]), "intercept": float(bw[1])},
        "vegas_win_total_mae": mae_line,
        "n_team_seasons": int(len(d)),
        "note": "strength_prior_net = slope*win_total + intercept; points-scale seed for M0",
    }

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(d["win_total"], d["net"], s=10, alpha=0.4, color="#4C78A8")
    xs = np.linspace(d["win_total"].min(), d["win_total"].max(), 50)
    ax[0].plot(
        xs,
        np.polyval(bn, xs),
        color="#D62728",
        label=f"net = {bn[0]:.3f}*line {bn[1]:+.2f} (R2={r2_net:.2f})",
    )
    ax[0].set_title("Strength seed: realized net rating vs preseason win total")
    ax[0].set_xlabel("Preseason win total")
    ax[0].set_ylabel("Realized net rating (pts/gm)")
    ax[0].legend()
    ax[1].scatter(d["win_total"], d["wins"], s=10, alpha=0.4, color="#1D9E75")
    ax[1].plot([20, 70], [20, 70], color="#888780", ls="--", label="identity")
    ax[1].set_title(f"Calibration: realized wins vs line (Vegas MAE {mae_line:.2f})")
    ax[1].set_xlabel("Preseason win total")
    ax[1].set_ylabel("Realized wins")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "vegas_seed_mapping.png", dpi=130)
    plt.close(fig)

    OUT.write_text(json.dumps(mapping, indent=2))
    return mapping


def main() -> None:
    m = fit_mapping()
    print("Vegas seed mapping (default for open question 2):")
    print(
        f"  strength_prior_net = {m['net_on_wintotal']['slope']:.4f} * win_total "
        f"{m['net_on_wintotal']['intercept']:+.3f}   (R2={m['net_on_wintotal']['r2']:.3f})"
    )
    print(f"  one extra win on the line ~ {m['net_on_wintotal']['slope']:.3f} pts/gm of net rating")
    print(
        f"  Vegas line MAE vs realized wins: {m['vegas_win_total_mae']:.2f} wins "
        f"(n={m['n_team_seasons']} team-seasons)"
    )
    print(f"  saved coefficients to {OUT}")


if __name__ == "__main__":
    main()
