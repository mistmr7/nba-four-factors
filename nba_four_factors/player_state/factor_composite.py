"""Individual four-factor composite: price the factor channels in points.

Rebuilt 2026-10-02 from the Credit_Engine_Design.md spec after the v0
pricing code turned out never to have been persisted. The v0 harness
numbers ledgered that day are NOT reproduced by this file; it is the
traceable replacement.

Identity behind every channel. A team factor F = N / D is a sum of
player numerators over a team denominator, so

    F_team - F_lg = sum_i (n_i - F_lg * d_i) / D_team

where d_i is the player's slice of the denominator. Volume share times
rate deviation, summed over the roster, recovers the team deviation
exactly. Multiplying by the factor's accounting weight (points of
margin per unit of factor) converts each slice to points.

Offense, own team factor:
    efg   n = FGM + 0.5*3PM      d = FGA                    D = team FGA
    ft    n = FTM                d = FGA                    D = team FGA
    tov   n = TOV, priced < 0    d = FGA + 0.44*FTA + TOV   D = team sum of d
    oreb  n = OREB               d = on-floor chances / 5   D = team chances
Defense, opponent factor:
    dreb  n = DREB               d = on-floor chances / 5   D = team chances
    stl   n = STL                d = defensive possessions  D = opp used
    blk   n = BLK * p_blk        d = opp FGA faced          D = opp FGA
    sf    n = SF * k_sf, < 0     d = opp FGA faced          D = opp FGA

p_blk: league FG% on 2PT attempts within 8 ft, the make value a block
erases. k_sf: league FTM per shooting foul (and-ones, two- and three-shot
trips counted from FT subtypes). Both as-of, trailing window.

As-of discipline: every league baseline is a trailing 365-day window
strictly before the game date (one full season of games, leak-free).
Weights are the accounting regression refit on seasons before the target.

Known holes, declared: no creation or usage credit (assists, shot
creation), no defended eFG (the dominant defensive term), forced
turnovers other than steals unattributed, bonus fouls unattributed.

Feed defect worked around: the possession parser drops fouls that occur
while no possession is open (possessions.py records a foul only when
cur is not None), so the persisted shooting_fouls channel holds about a
third of real shooting fouls (2025-26: 8,246 of 25,617). This module
recounts shooting fouls per player-game from the raw play-by-play.

Run from repo root:
    python -m nba_four_factors.player_state.factor_composite weights 2025_26
    python -m nba_four_factors.player_state.factor_composite build 2025_26
    python -m nba_four_factors.player_state.factor_composite validate 2025_26
    python -m nba_four_factors.player_state.factor_composite leaders 2025_26 --n 50
    python -m nba_four_factors.player_state.factor_composite passthrough 2025_26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
RAW_PBP = REPO / "data" / "raw" / "playbyplay"
CHANNELS = REPO / "data" / "player_state" / "factor_channels"
ROSTERS = REPO / "data" / "player_state" / "rosters"
GMSC = REPO / "data" / "player_state" / "player_game_score.parquet"
OUT_DIR = REPO / "data" / "player_state" / "factor_composite"
FEAT = REPO / "data" / "features"

# Table 4 M1 predictive weights (Final_Audit_Handoff.md), FT factor on FTA/FGA.
# Used only for the sensitivity variant.
M1_WEIGHTS = {"efg": 106.8, "tov": 107.5, "orb": 36.0, "ft": 22.8}

CHANNEL_COLS = ["c_efg", "c_ft", "c_tov", "c_oreb", "c_dreb", "c_stl", "c_blk", "c_sf"]
WINDOW_DAYS = 365


def _seasons() -> list[str]:
    return sorted(p.name for p in PROC.iterdir() if p.is_dir() and p.name[:4].isdigit())


def _prior(season: str) -> str | None:
    ss = _seasons()
    i = ss.index(season)
    return ss[i - 1] if i > 0 else None


def fit_weights(seasons: list[str]) -> dict[str, float]:
    """Accounting regression: margin on same-game factor differentials.

    Table 1 definitions: eFG, TOV% forced minus committed, ORB%, FTM/FGA.
    One row per non-neutral game (home perspective).
    """
    frames = []
    for s in seasons:
        df = pd.read_parquet(PROC / s / "regular_season.parquet")
        frames.append(df[df.is_home & ~df.is_neutral])
    d = pd.concat(frames, ignore_index=True)
    x = pd.DataFrame(
        {
            "efg": d.off_efg_pct - d.def_efg_pct,
            "tov": d.def_tov_pct - d.off_tov_pct,
            "orb": d.off_orb_pct - d.def_orb_pct,
            "ft": d.ftm / d.fga - d.opp_ftm / d.opp_fga,
        }
    )
    ok = x.notna().all(axis=1) & d.margin.notna()
    x, y = x[ok], d.margin[ok].to_numpy(dtype=float)
    a = np.column_stack([x.to_numpy(dtype=float), np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    r2 = 1 - resid.var() / y.var()
    out = {
        k: round(float(b), 3)
        for k, b in zip(["efg", "tov", "orb", "ft", "intercept"], beta, strict=True)
    }
    out["r2"] = round(float(r2), 4)
    out["n_games"] = int(len(x))
    return out


def _pbp_extras(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shooting fouls per player-game, plus per-game league ingredients."""
    cols = [
        "gameId",
        "actionType",
        "subType",
        "personId",
        "shotDistance",
        "shotResult",
        "shotValue",
    ]
    pbp = pd.read_parquet(RAW_PBP / f"{season}_regular_season.parquet", columns=cols)
    pbp["gameId"] = pbp.gameId.astype(str)
    is_sf = (pbp.actionType == "Foul") & (pbp.subType == "Shooting")
    sf = (
        pbp[is_sf]
        .groupby(["gameId", "personId"])
        .size()
        .rename("sf_raw")
        .reset_index()
        .rename(columns={"gameId": "game_id", "personId": "person_id"})
    )
    sf["person_id"] = sf.person_id.astype(int)
    ft = pbp[pbp.actionType == "Free Throw"]
    rim = pbp[(pbp.actionType.isin(["Made Shot", "Missed Shot"])) & (pbp.shotValue == 2)]
    rim = rim[rim.shotDistance <= 8]
    g = pd.DataFrame(
        {
            "sf_n": is_sf.groupby(pbp.gameId).sum(),
            "ft_1of1": (ft.subType == "Free Throw 1 of 1").groupby(ft.gameId).sum(),
            "ft_1of3": (ft.subType == "Free Throw 1 of 3").groupby(ft.gameId).sum(),
            "rim_fga": rim.groupby("gameId").size(),
            "rim_fgm": (rim.shotResult == "Made").groupby(rim.gameId).sum(),
        }
    ).fillna(0.0)
    g.index.name = "game_id"
    return sf, g.reset_index()


def _load_season(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ch = pd.read_parquet(CHANNELS / f"{season}.parquet")
    ch["game_id"] = ch.game_id.astype(str)
    ro = pd.read_parquet(
        ROSTERS / f"{season}_regular_season.parquet", columns=["game_id", "person_id", "team_id"]
    )
    ro["game_id"] = ro.game_id.astype(str)
    ro = ro.drop_duplicates(["game_id", "person_id"])
    box = pd.read_parquet(
        PROC / season / "regular_season.parquet", columns=["game_id", "game_date"]
    )
    box["game_id"] = box.game_id.astype(str)
    dates = box.drop_duplicates("game_id")
    sf, g = _pbp_extras(season)
    ch = ch.merge(ro, on=["game_id", "person_id"], how="left")
    ch = ch.merge(sf, on=["game_id", "person_id"], how="left")
    ch["sf_raw"] = ch.sf_raw.fillna(0.0)
    ch = ch.merge(dates, on="game_id", how="left")
    g = g.merge(dates, on="game_id", how="left")
    return ch, g


def _baselines(ch: pd.DataFrame, g: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
    """League rates over the trailing window strictly before each target date."""
    c = ch.assign(used=ch.fga + 0.44 * ch.fta + ch.tov)
    num_cols = [
        "fga",
        "fgm_w",
        "fta",
        "ftm",
        "tov",
        "used",
        "oreb",
        "dreb",
        "oreb_chances",
        "dreb_chances",
        "stl",
        "def_poss",
        "blk",
        "sf_raw",
        "opp_fga_faced",
    ]
    daily = c.groupby("game_date")[num_cols].sum()
    gd = g.groupby("game_date")[["sf_n", "ft_1of1", "ft_1of3", "rim_fga", "rim_fgm"]].sum()
    daily = daily.join(gd, how="outer").fillna(0.0).sort_index()
    cs = daily.cumsum()
    dates = daily.index.values
    zero = pd.Series(0.0, index=cs.columns)

    def window(d: np.datetime64) -> pd.Series:
        hi = np.searchsorted(dates, d, side="left") - 1
        lo = np.searchsorted(dates, d - np.timedelta64(WINDOW_DAYS, "D"), side="left") - 1
        top = cs.iloc[hi] if hi >= 0 else zero
        bot = cs.iloc[lo] if lo >= 0 else zero
        return top - bot

    rows = {}
    for d in pd.Series(targets.unique()).sort_values():
        w = window(np.datetime64(d))
        fta_per_sf = (w.ft_1of1 + 3 * w.ft_1of3 + 2 * (w.sf_n - w.ft_1of1 - w.ft_1of3)) / w.sf_n
        ft_pct = w.ftm / w.fta
        rows[d] = {
            "efg_lg": w.fgm_w / w.fga,
            "ftr_lg": w.ftm / w.fga,
            "ftar_lg": w.fta / w.fga,
            "tov_lg": w.tov / w.used,
            "orb_lg": w.oreb / (w.oreb_chances / 5),
            "drb_lg": w.dreb / (w.dreb_chances / 5),
            "stl_lg": w.stl / w.def_poss,
            "blk_lg": w.blk / w.opp_fga_faced,
            "sf_lg": w.sf_raw / w.opp_fga_faced,
            "p_blk": w.rim_fgm / w.rim_fga,
            "k_sf": fta_per_sf * ft_pct,
            "k_sf_att": fta_per_sf,
            "window_games": float(w.sf_n > 0),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "game_date"
    return out.reset_index()


def build(season: str) -> pd.DataFrame:
    prior = _prior(season)
    if prior is None:
        raise SystemExit(f"{season}: needs a prior season for the trailing window and weights")
    w = fit_weights([s for s in _seasons() if s < season])
    cur, g_cur = _load_season(season)
    old, g_old = _load_season(prior)
    both = pd.concat([old, cur], ignore_index=True)
    g_both = pd.concat([g_old, g_cur], ignore_index=True)
    base = _baselines(both, g_both, cur.game_date)

    df = cur.merge(base, on="game_date", how="left")
    df["used"] = df.fga + 0.44 * df.fta + df.tov
    team = df.groupby(["game_id", "team_id"])[["fga", "used", "oreb_chances", "dreb_chances"]].sum()
    team["T_orbch"] = team.oreb_chances / 5
    team["T_drbch"] = team.dreb_chances / 5
    team = team.rename(columns={"fga": "T_fga", "used": "T_used"})[
        ["T_fga", "T_used", "T_orbch", "T_drbch"]
    ]
    team = team.reset_index()
    opp = team[["game_id", "team_id", "T_fga", "T_used"]].rename(
        columns={"team_id": "opp_team_id", "T_fga": "O_fga", "T_used": "O_used"}
    )
    df = df.merge(team, on=["game_id", "team_id"], how="left")
    pair = team[["game_id", "team_id"]].merge(opp, on="game_id")
    pair = pair[pair.team_id != pair.opp_team_id]
    df = df.merge(pair, on=["game_id", "team_id"], how="left")

    def price(wts: dict[str, float], ft_num: str, ft_rate: str, k: str) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["c_efg"] = wts["efg"] * (df.fgm_w - df.efg_lg * df.fga) / df.T_fga
        out["c_ft"] = wts["ft"] * (df[ft_num] - df[ft_rate] * df.fga) / df.T_fga
        out["c_tov"] = -wts["tov"] * (df.tov - df.tov_lg * df.used) / df.T_used
        out["c_oreb"] = wts["orb"] * (df.oreb - df.orb_lg * df.oreb_chances / 5) / df.T_orbch
        out["c_dreb"] = wts["orb"] * (df.dreb - df.drb_lg * df.dreb_chances / 5) / df.T_drbch
        out["c_stl"] = wts["tov"] * (df.stl - df.stl_lg * df.def_poss) / df.O_used
        out["c_blk"] = wts["efg"] * df.p_blk * (df.blk - df.blk_lg * df.opp_fga_faced) / df.O_fga
        out["c_sf"] = -wts["ft"] * df[k] * (df.sf_raw - df.sf_lg * df.opp_fga_faced) / df.O_fga
        return out.fillna(0.0)

    acct = price(w, "ftm", "ftr_lg", "k_sf")
    m1 = price(M1_WEIGHTS, "fta", "ftar_lg", "k_sf_att")
    for col in CHANNEL_COLS:
        df[col] = acct[col]
    df["ff"] = acct[CHANNEL_COLS].sum(axis=1)
    df["ff_m1"] = m1[CHANNEL_COLS].sum(axis=1)
    df["c_dreb_m1"] = m1["c_dreb"]
    for k_, v in w.items():
        df[f"w_{k_}"] = v
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / f"{season}.parquet", index=False)
    print(f"{season}: weights {w}")
    print(f"{season}: {len(df):,} player-games priced; team_id missing {df.team_id.isna().sum()}")
    return df


def validate(season: str) -> None:
    """Roster identities and the margin check.

    1. Per team-game, summed channel contributions must equal the priced
       team deviation computed straight from team sums (catches any
       denominator slip).
    2. Team composite vs actual margin: positive slope, partial fit, since
       the composite sees only the attributable slice of each factor.
    """
    df = pd.read_parquet(OUT_DIR / f"{season}.parquet")
    t = df.groupby(["game_id", "team_id"]).agg(
        c_efg=("c_efg", "sum"),
        fgm_w=("fgm_w", "sum"),
        fga=("fga", "sum"),
        efg_lg=("efg_lg", "first"),
        w_efg=("w_efg", "first"),
        ff=("ff", "sum"),
    )
    direct = t.w_efg * (t.fgm_w / t.fga - t.efg_lg)
    print(f"  efg identity max |diff|: {(t.c_efg - direct).abs().max():.2e}")
    box = pd.read_parquet(
        PROC / season / "regular_season.parquet", columns=["game_id", "team_id", "margin"]
    )
    box["game_id"] = box.game_id.astype(str)
    j = t.reset_index().merge(box, on=["game_id", "team_id"])
    r = np.corrcoef(j.ff, j.margin)[0, 1]
    slope = np.polyfit(j.ff, j.margin, 1)[0]
    print(f"  team composite vs margin: r {r:.3f}, slope {slope:.3f}, n {len(j):,}")
    sf_kept = df.shooting_fouls.sum() / df.sf_raw.sum()
    print(f"  parser shooting_fouls / raw PBP: {sf_kept:.3f}")
    b = df.drop_duplicates("game_date").sort_values("game_date")
    print("  baseline ranges over the season:")
    for c in ["efg_lg", "tov_lg", "orb_lg", "drb_lg", "p_blk", "k_sf"]:
        print(f"    {c:7s} {b[c].min():.4f} .. {b[c].max():.4f}")


def leaders(season: str, n: int = 50, dreb_scale: float = 1.0) -> pd.DataFrame:
    """Season-total leaderboard.

    dreb_scale shrinks DREB credit toward its measured team pass-through
    (see passthrough); 1.0 is the spec as written. Descriptive only, not
    an adoption decision.
    """
    df = pd.read_parquet(OUT_DIR / f"{season}.parquet")
    if dreb_scale != 1.0:
        df["ff"] = df.ff - (1.0 - dreb_scale) * df.c_dreb
        df["ff_m1"] = df.ff_m1 - (1.0 - dreb_scale) * df.c_dreb_m1
        df["c_dreb"] = dreb_scale * df.c_dreb
    agg = {c: "sum" for c in [*CHANNEL_COLS, "ff", "ff_m1", "off_poss", "def_poss"]}
    s = df.groupby("person_id").agg(agg)
    last_team = df.sort_values("game_date").groupby("person_id").team_id.last()
    s["team_id"] = last_team
    s["ff_per100"] = 100 * s.ff / ((s.off_poss + s.def_poss) / 2)

    gs = pd.read_parquet(GMSC)
    gs = gs[(gs.season == season) & (gs.season_type == "regular_season") & (gs.status == "played")]
    g = gs.groupby("person_id").agg(
        gp=("game_score", "size"), minutes=("minutes", "sum"), gmsc=("game_score", "sum")
    )
    s = s.join(g, how="left")
    s["gmsc_rank"] = s.gmsc.rank(ascending=False, method="min")
    s["ff_rank"] = s.ff.rank(ascending=False, method="min")
    s["ff_m1_rank"] = s.ff_m1.rank(ascending=False, method="min")

    ro = pd.read_parquet(ROSTERS / f"{season}_regular_season.parquet")
    ro = ro.drop_duplicates("person_id").set_index("person_id")
    s["player"] = (ro.first_name.astype(str) + " " + ro.family_name.astype(str)).reindex(s.index)
    box = pd.read_parquet(
        PROC / season / "regular_season.parquet", columns=["team_id", "team_abbr"]
    )
    abbr = box.drop_duplicates("team_id").set_index("team_id").team_abbr
    s["team"] = s.team_id.map(abbr)
    s = s.sort_values("ff", ascending=False)

    cols = [
        "ff_rank",
        "player",
        "team",
        "gp",
        "minutes",
        "ff",
        "ff_per100",
        *CHANNEL_COLS,
        "gmsc",
        "gmsc_rank",
        "ff_m1_rank",
    ]
    out = s[cols].reset_index()
    FEAT.mkdir(parents=True, exist_ok=True)
    tag = "" if dreb_scale == 1.0 else f"_dreb{dreb_scale:g}"
    out.to_csv(FEAT / f"ff_composite_{season}{tag}.csv", index=False)

    q = s[s.minutes >= 500]
    r_all = np.corrcoef(s.ff.fillna(0), s.gmsc.fillna(0))[0, 1]
    r_q = np.corrcoef(q.ff, q.gmsc)[0, 1]
    rho_q = q.ff.rank().corr(q.gmsc.rank())
    top = set(s.head(n).index)
    top_m1 = set(s.sort_values("ff_m1", ascending=False).head(n).index)
    top_gs = set(s.sort_values("gmsc", ascending=False).head(n).index)
    print(f"player-seasons {len(s)}; with 500+ min {len(q)}")
    print(f"corr(ff, gmsc) all {r_all:.3f}; 500+ min pearson {r_q:.3f} spearman {rho_q:.3f}")
    print(
        f"top-{n} overlap: accounting vs M1 weights {len(top & top_m1)}; ff vs gmsc {len(top & top_gs)}"
    )
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.head(n).round(1).to_string(index=False))
    return out


def passthrough(season: str, min_chances: int = 400) -> None:
    """Do individual rebound rates add up at the team level?

    The rebound channels pay a player for every board he collects. That is
    only right if his boards are boards the team would not otherwise get.
    Test, leak-free: rates from the PRIOR season, outcomes from this one.

    1. Individual persistence b_ind: player's rate this season on his
       prior-season rate (chance-weighted). Absorbs measurement noise and
       role drift, the attenuation any prior-rate predictor carries.
    2. Lineup pass-through b_team: for every rebound event, P(defense
       collects) on the sum of the five defenders' prior DREB rates, with
       the five offensive players' prior OREB rates as a control (and the
       mirror for OREB).
    3. Ratio b_team / b_ind: share of a player's individual edge that
       survives to the team. 1.0 means boards are additive; well below 1
       means collectors take boards from teammates.
    Events kept only when all ten players have min_chances prior chances.
    """
    import json

    prior = _prior(season)
    old = (
        pd.read_parquet(CHANNELS / f"{prior}.parquet")
        .groupby("person_id")[["oreb", "oreb_chances", "dreb", "dreb_chances"]]
        .sum()
    )
    cur = (
        pd.read_parquet(CHANNELS / f"{season}.parquet")
        .groupby("person_id")[["oreb", "oreb_chances", "dreb", "dreb_chances"]]
        .sum()
    )
    o_rate = (old.oreb / old.oreb_chances).where(old.oreb_chances >= min_chances)
    d_rate = (old.dreb / old.dreb_chances).where(old.dreb_chances >= min_chances)

    def wls_slope(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
        a = np.column_stack([x, np.ones(len(x))]) * np.sqrt(w.to_numpy())[:, None]
        b = y.to_numpy() * np.sqrt(w.to_numpy())
        return float(np.linalg.lstsq(a, b, rcond=None)[0][0])

    b_ind = {}
    for kind, rate in [("oreb", o_rate), ("dreb", d_rate)]:
        ch = f"{kind}_chances"
        j = pd.DataFrame({"x": rate, "y": cur[kind] / cur[ch], "w": cur[ch]}).dropna()
        j = j[j.w >= min_chances]
        b_ind[kind] = wls_slope(j.x, j.y, j.w)
        print(f"  individual persistence {kind}: b_ind {b_ind[kind]:.3f} (players {len(j)})")

    o_map, d_map = o_rate.dropna().to_dict(), d_rate.dropna().to_dict()
    home = _home_ids_local(season)
    pos = pd.read_parquet(
        REPO / "data" / "player_state" / "possessions" / f"{season}_regular_season.parquet",
        columns=["game_id", "off_team_id", "home_lineup", "away_lineup", "events"],
    )
    rows = []
    for r in pos.itertuples(index=False):
        h = home.get(str(r.game_id))
        if h is None or r.home_lineup is None or r.away_lineup is None:
            continue
        off5 = r.home_lineup if r.off_team_id == h else r.away_lineup
        def5 = r.away_lineup if r.off_team_id == h else r.home_lineup
        so = [o_map.get(int(p)) for p in off5]
        sd = [d_map.get(int(p)) for p in def5]
        if any(v is None for v in so + sd):
            continue
        for e in json.loads(r.events):
            if e["t"] != "reb" or e.get("team_reb"):
                continue
            rows.append((sum(so), sum(sd), int(bool(e.get("off")))))
    ev = pd.DataFrame(rows, columns=["s_off", "s_def", "off_won"])
    a = np.column_stack([ev.s_off, ev.s_def, np.ones(len(ev))])
    y = ev.off_won.to_numpy(dtype=float)
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    resid = y - a @ coef
    cov = np.linalg.inv(a.T @ a) * resid.var()
    se_o, se_d = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    b_team_oreb, b_team_dreb = coef[0], -coef[1]
    print(f"  rebound events kept: {len(ev):,}")
    print(
        f"  team pass-through OREB: b_team {b_team_oreb:.3f} (se {se_o:.3f})  "
        f"ratio {b_team_oreb / b_ind['oreb']:.2f}"
    )
    print(
        f"  team pass-through DREB: b_team {b_team_dreb:.3f} (se {se_d:.3f})  "
        f"ratio {b_team_dreb / b_ind['dreb']:.2f}"
    )


def _home_ids_local(season: str) -> dict[str, int]:
    df = pd.read_parquet(
        PROC / season / "regular_season.parquet", columns=["game_id", "team_id", "is_home"]
    )
    h = df[df.is_home].drop_duplicates("game_id")
    return dict(zip(h.game_id.astype(str), h.team_id.astype(int), strict=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["weights", "build", "validate", "leaders", "passthrough"])
    ap.add_argument("season")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dreb-scale", type=float, default=1.0)
    args = ap.parse_args()
    if args.cmd == "weights":
        print(
            "leak-free (seasons before target):",
            fit_weights([s for s in _seasons() if s < args.season]),
        )
        print("all seasons (Table 4 reproduction):", fit_weights(_seasons()))
    elif args.cmd == "build":
        build(args.season)
    elif args.cmd == "validate":
        validate(args.season)
    elif args.cmd == "passthrough":
        passthrough(args.season)
    else:
        leaders(args.season, args.n, args.dreb_scale)


if __name__ == "__main__":
    main()
