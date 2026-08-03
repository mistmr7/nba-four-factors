"""Build Siamese per-game sequence tensors for the RNN. Pure numpy/pandas.

For each game (home perspective) we assemble the two teams' recent-game
histories and the pre-game context:

* home_seq, away_seq: each team's last L games BEFORE this game, as vectors of
  the 8 four-factor values (team and opponent eFG, TOV, OREB, FT/FGA) plus pace.
  Pace is the last channel so it can be dropped for the no-pace experiment.
  Shorter histories are left-padded with zeros and flagged by a mask.
* ctx: pre-game schedule differentials (rest, back-to-back, 7-day travel),
  all home-minus-away and known before tip-off.
* targets: home point margin and home win.

Sequences hold only games strictly before the predicted game, so there is no
leakage of the current outcome. Saved to data/features/rnn_sequences.npz with a
season array so the trainer can split by season.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
TABLE = REPO / "data" / "features" / "modeling_table.parquet"

# Sequence feature set. Default is the 8 four factors + pace. Set RAW_SEQ=1 to
# instead feed the raw box-score counts (team and opponent), letting the network
# learn its own representation rather than trusting the four-factor summary.
SEQ_FEATS_FF = [
    "off_efg_pct",
    "def_efg_pct",
    "off_tov_pct",
    "def_tov_pct",
    "off_orb_pct",
    "def_orb_pct",
    "off_ftmfga",
    "def_ftmfga",
    "pace",
]
SEQ_FEATS_RAW = [
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "tov",
    "pts",
    "opp_fgm",
    "opp_fga",
    "opp_fg3m",
    "opp_fg3a",
    "opp_ftm",
    "opp_fta",
    "opp_oreb",
    "opp_dreb",
    "opp_tov",
    "opp_pts",
]
RAW = bool(os.environ.get("RAW_SEQ"))
SEQ_FEATS = SEQ_FEATS_RAW if RAW else SEQ_FEATS_FF
OUT = REPO / "data" / "features" / ("rnn_sequences_raw.npz" if RAW else "rnn_sequences.npz")
# Context now includes the season-to-date four-factor summary (the regression's
# inductive bias) so the RNN only has to add temporal signal on top.
SCHED_FEATS = ["rest_diff", "b2b_diff", "miles7d_diff"]
SUMMARY_FEATS = ["d_efg_d", "d_oreb_d", "d_tov_d", "d_ftmfga_d"]
CTX_FEATS = SCHED_FEATS + SUMMARY_FEATS
L = 20  # sequence length (recent games)


def team_game_features() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(PROC / "*/regular_season.parquet")))],
        ignore_index=True,
    )
    df = df[(~df["is_neutral"]) & (df["season"] != "1998_99")].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["off_ftmfga"] = df["ftm"] / df["fga"]
    df["def_ftmfga"] = df["opp_ftm"] / df["opp_fga"]
    df["pace"] = 0.5 * (
        (df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"])
        + (df["opp_fga"] + 0.44 * df["opp_fta"] - df["opp_oreb"] + df["opp_tov"])
    )
    # Drop games with missing/degenerate four-factor stats (e.g., zero FGA) so a
    # single bad row cannot poison a team's history sequence.
    df[SEQ_FEATS] = df[SEQ_FEATS].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=SEQ_FEATS)
    return df.sort_values(["team_id", "season", "game_date"])


def build():
    tf = team_game_features()
    F = len(SEQ_FEATS)
    # Per (game_id, team_id): the team's prior-L-game history matrix + mask.
    hist, mask = {}, {}
    for (_t, _s), g in tf.groupby(["team_id", "season"], sort=False):
        arr = g[SEQ_FEATS].to_numpy(dtype=np.float32)
        gids = g["game_id"].to_numpy()
        for j, gid in enumerate(gids):
            h = np.zeros((L, F), dtype=np.float32)
            m = np.zeros(L, dtype=np.float32)
            prior = arr[max(0, j - L) : j]
            if len(prior):
                h[L - len(prior) :] = prior
                m[L - len(prior) :] = 1.0
            hist[(gid, int(_t))] = h
            mask[(gid, int(_t))] = m

    mt = pd.read_parquet(TABLE).dropna(subset=["home_margin", "home_win"]).copy()
    # Impute missing context/summary (early-season games) to neutral 0 so they are
    # kept; the seed blend handles their weak signal at prediction time.
    mt[CTX_FEATS] = mt[CTX_FEATS].fillna(0.0)
    mt["wintotal_diff"] = mt["wintotal_diff"].fillna(0.0)
    # Games played so far (avg of the two teams), for the seed blend alpha=n/(n+k).
    gp = {}
    for (_t, _s), g in tf.groupby(["team_id", "season"], sort=False):
        for j, gid in enumerate(g["game_id"].to_numpy()):
            gp[(gid, int(_t))] = j + 1

    # Recent-form pieces, for the full-production-regression comparison on the
    # same split (shrunk recent-form term, computed train-only at eval).
    form_cols = [
        "form_mean",
        "base_comp",
        "form_var",
        "away_form_mean",
        "away_base_comp",
        "away_form_var",
    ]
    mt[form_cols] = mt[form_cols].fillna(0.0)

    H, A, HM, AM, C, ym, yw, seas, wt, ng, gid, FM = [], [], [], [], [], [], [], [], [], [], [], []
    for r in mt.itertuples(index=False):
        kh, ka = (r.game_id, int(r.team_id)), (r.game_id, int(r.away_team_id))
        if kh not in hist or ka not in hist:
            continue
        # Skip games where a team has no prior games this season: the sequence is
        # fully masked, and a masked average over zero steps is 0/0 -> NaN (which
        # can even hang a GPU kernel). The seed covers these games in production.
        if mask[kh].sum() == 0 or mask[ka].sum() == 0:
            continue
        H.append(hist[kh])
        A.append(hist[ka])
        HM.append(mask[kh])
        AM.append(mask[ka])
        C.append([getattr(r, c) for c in CTX_FEATS])
        ym.append(r.home_margin)
        yw.append(r.home_win)
        seas.append(r.season)
        wt.append(r.wintotal_diff)
        ng.append(0.5 * (gp.get(kh, 1) + gp.get(ka, 1)))
        gid.append(r.game_id)
        FM.append([getattr(r, c) for c in form_cols])
    np.savez_compressed(
        OUT,
        home_seq=np.array(H, dtype=np.float32),
        away_seq=np.array(A, dtype=np.float32),
        home_mask=np.array(HM, dtype=np.float32),
        away_mask=np.array(AM, dtype=np.float32),
        ctx=np.array(C, dtype=np.float32),
        y_margin=np.array(ym, dtype=np.float32),
        y_win=np.array(yw, dtype=np.float32),
        season=np.array(seas),
        wintotal_diff=np.array(wt, dtype=np.float32),
        ngames=np.array(ng, dtype=np.float32),
        game_id=np.array(gid),
        form=np.array(FM, dtype=np.float32),
        form_feats=np.array(
            [
                "form_mean",
                "base_comp",
                "form_var",
                "away_form_mean",
                "away_base_comp",
                "away_form_var",
            ]
        ),
        seq_feats=np.array(SEQ_FEATS),
        ctx_feats=np.array(CTX_FEATS),
        n_sched=len(SCHED_FEATS),
        L=L,
    )
    print(f"Saved {len(H):,} games to {OUT}")
    print(f"  home_seq {np.array(H).shape}  ctx {np.array(C).shape} (3 schedule + 4 summary)")
    print(f"  ctx features: {CTX_FEATS}")
    print(
        f"  also saved wintotal_diff (seed) and ngames (blend); seasons {pd.Series(seas).nunique()}"
    )


if __name__ == "__main__":
    build()
