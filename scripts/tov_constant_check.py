"""Critical examination of the 0.44 free-throw constant in TOV% and pace.

The folklore possession formula poss = FGA + 0.44*FTA - OREB + TOV
inherits its 0.44 from Oliver-era literature. Two questions:

1. What constant does the data actually imply? Both teams in a game
   consume the same number of possessions, so the implied k is the one
   that makes the two teams' possession estimates agree game by game.
   Least squares over the game-level differences gives k-hat, pooled
   and per season. Honest caveat: this estimates the k that minimizes
   within-game asymmetry of the ESTIMATES, so it absorbs any other
   asymmetric box-score quirks (team rebounds, end-of-period effects).
   It is a diagnostic for the constant, not a play-by-play count.

2. Does the choice matter for the study? Recompute TOV% and its
   game differential under k = 0.44 and k = k-hat and compare.

Findings (2026-07, all processed regular-season games):
    pooled k-hat = 0.477 (SE 0.002), per-season range 0.443-0.497.

SUPERSEDED as evidence about the constant (see scripts/
tov_constant_figure.py): direct play-by-play counting of terminal
FT trips gives the true trip-level rate, 0.418-0.445 per season,
pooled 0.432 across 1.64M FTA, validating the literature 0.44.
The symmetry fit above measures a different estimand, the MARGINAL
termination rate per additional FTA of differential (~0.47, since
marginal free throws are predominantly ordinary two-shot trips,
confirmed by a counted within-game differential slope of 0.467 on
2023-24 play-by-play). Average vs marginal, not error vs truth.
Sensitivity either way: swapping 0.432 for 0.44 moves the TOV%
differential corr 0.999998, mean 0.010 pp, max 0.072 pp against a
4.74 pp SD. Retained 0.44, consistent with the counted rate and
the literature.

Run from repo root:
    python3 scripts/tov_constant_check.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(REPO / "data" / "processed" / "*" / "regular_season.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df.pts + df.opp_pts > 0].copy()


def khat(dB: pd.Series, dF: pd.Series) -> float:
    return float(-np.sum(dB * dF) / np.sum(dF * dF))


def main() -> None:
    df = load()
    home = df[df.is_home].copy()
    dB = (home.fga - home.oreb + home.tov) - (home.opp_fga - home.opp_oreb + home.opp_tov)
    dF = home.fta - home.opp_fta

    k_pool = khat(dB, dF)
    resid = dB + k_pool * dF
    se = float(np.sqrt(np.sum(resid**2) / (len(dB) - 1) / np.sum(dF * dF)))
    print(f"pooled k-hat = {k_pool:.4f} (SE {se:.4f}), n = {len(home):,} games")

    per = (
        home.assign(dB=dB, dF=dF)
        .groupby("season")[["dB", "dF"]]
        .apply(lambda g: khat(g.dB, g.dF))
        .round(3)
    )
    print("per-season k-hat:")
    print(per.to_string())

    def tovpct(k: float, prefix: str = "") -> pd.Series:
        p = prefix
        return df[f"{p}tov"] / (df[f"{p}fga"] + k * df[f"{p}fta"] + df[f"{p}tov"])

    t44, tk = tovpct(0.44), tovpct(k_pool)
    d44 = tovpct(0.44, "opp_") - t44
    dk = tovpct(k_pool, "opp_") - tk
    print(f"team-game TOV% corr (0.44 vs k-hat): {np.corrcoef(t44, tk)[0, 1]:.6f}")
    print(f"mean level shift: {(tk - t44).mean() * 100:.3f} pp")
    print(f"differential corr: {np.corrcoef(d44, dk)[0, 1]:.6f}")
    print(f"differential SD: {d44.std() * 100:.2f} pp")
    print(f"max |differential change|: {(dk - d44).abs().max() * 100:.3f} pp")


if __name__ == "__main__":
    main()
