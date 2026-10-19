"""Player-level projections for one target season, fitted only on earlier seasons.

Two small regressions, each refit for every target season:

value model    v_S = single-season RAPM in S (possession weighted), on
               end-of-prior-season as-of RAPM, a missing-history flag, a
               team-change flag, and, when the bio and draft pulls exist,
               age, age squared, and log draft pick for rookies and
               players within three years of their draft.

minutes model  mpg_S = minutes per scheduled game in S, on last season's
               and the season before's minutes per game, games played
               share, projected value, rookie flag and draft slot, team
               change, and two within-roster ranks (projected value, last
               season's minutes) that let a player's role depend on who
               else is on the roster.

Team rating = sum over the opening roster of projected value times
projected minutes share, scaled so the shares sum to five players on the
floor. Configuration flags switch each piece back to its naive version so
every step's contribution can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

YOUNG_YEARS = 3
MISSED_SEASON_GP = 0.25
YOUNG_AGE = 23.0


@dataclass(frozen=True)
class Config:
    value: str = "model"
    minutes: str = "model"
    use_age: bool = True
    use_draft: bool = True
    use_out_g1: bool = False
    team_age: bool = False
    healthy_minutes: bool = True
    value_terms: str = "base"
    min_train_seasons: int = 3


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])
    if w is None:
        return np.linalg.lstsq(x1, y, rcond=None)[0]
    sw = np.sqrt(w)
    return np.linalg.lstsq(x1 * sw[:, None], y * sw, rcond=None)[0]


def _predict(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return b[0] + x @ b[1:]


def add_design(f: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    f = f.copy()
    f["miss_v"] = f.v_p.isna().astype(float)
    f["v_p0"] = f.v_p.fillna(0.0)
    f["nt"] = f.new_team.astype(float)
    f["rk"] = f.rookie.astype(float)
    have_age = bool(cfg.use_age and f.age.notna().mean() > 0.5)
    have_draft = bool(cfg.use_draft and f.drafted.mean() > 0.3)
    if have_age:
        f["age_c"] = f.age_c.fillna(f.age_c.median())
        f["age_c2"] = f.age_c**2
    if have_draft:
        ysd = f.years_since_draft
        young = ysd.between(1, YOUNG_YEARS) & ~f.rookie
        f["rk_lp"] = f.rk * f.log_pick
        f["young_lp"] = young.astype(float) * f.log_pick
        f["young"] = young.astype(float)
    lost = (f.gp_frac_p < MISSED_SEASON_GP) & (f.mpg_p2 > f.mpg_p)
    f["missed_last"] = lost.astype(float)
    f["mpg_healthy"] = np.where(lost, f.mpg_p2, f.mpg_p)
    f["out_g1"] = f.out_g1.fillna(0.0) if "out_g1" in f.columns else 0.0
    f["out_g1_mpg"] = f.out_g1 * f.mpg_p
    f.attrs["use_out_g1"] = bool(cfg.use_out_g1)
    f.attrs["use_healthy"] = bool(cfg.healthy_minutes)
    f.attrs["value_terms"] = cfg.value_terms
    f["v_neg"] = np.minimum(f.v_p0, 0.0)
    yng = (f.age <= YOUNG_AGE).astype(float) * (1.0 - f.miss_v)
    f["yng"] = yng
    f["yng_v"] = yng * f.v_p0
    f["yng_lp"] = yng * f.log_pick
    f["yng_age"] = yng * (YOUNG_AGE - f.age.fillna(YOUNG_AGE))
    f.attrs["have_age"] = have_age
    f.attrs["have_draft"] = have_draft
    return f


def value_columns(f: pd.DataFrame) -> list[str]:
    cols = ["v_p0", "miss_v", "nt"]
    if f.attrs.get("have_age"):
        cols += ["age_c", "age_c2"]
    if f.attrs.get("have_draft"):
        cols += ["rk_lp", "young_lp", "young"]
    terms = str(f.attrs.get("value_terms", "base"))
    if "kink" in terms:
        cols += ["v_neg"]
    if "young" in terms and f.attrs.get("have_age"):
        cols += ["yng", "yng_v", "yng_age"]
        if f.attrs.get("have_draft"):
            cols += ["yng_lp"]
    return cols


def minutes_columns(f: pd.DataFrame) -> list[str]:
    cols = ["mpg_p", "mpg_p2", "gp_frac_p", "v_hat", "rk", "nt", "v_rank", "mpg_rank"]
    if f.attrs.get("use_healthy"):
        cols += ["mpg_healthy", "missed_last"]
    if f.attrs.get("use_out_g1"):
        cols += ["out_g1", "out_g1_mpg"]
    if f.attrs.get("have_age"):
        cols += ["age_c", "age_c2"]
    if f.attrs.get("have_draft"):
        cols += ["rk_lp"]
    return cols


def _ranks(f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby(["season", "team_id"])
    f["v_rank"] = g.v_hat.rank(ascending=False)
    use_healthy = bool(f.attrs.get("use_healthy")) and "mpg_healthy" in f
    f["mpg_rank"] = (g.mpg_healthy if use_healthy else g.mpg_p).rank(ascending=False)
    return f


def project_season(f: pd.DataFrame, season: str, cfg: Config) -> pd.DataFrame:
    """Projected value and minutes for every rostered player in `season`."""
    train = f[f.season < season]
    test = f[f.season == season].copy()

    if cfg.value == "model":
        vc = value_columns(f)
        t = train[train.v_S.notna() & (train.poss_S > 0)]
        b = _wls(t[vc].to_numpy(float), t.v_S.to_numpy(float), t.poss_S.to_numpy(float))
        test["v_hat"] = _predict(b, test[vc].to_numpy(float))
        tr = train.copy()
        tr["v_hat"] = _predict(b, tr[vc].to_numpy(float))
    else:
        rook = train[train.miss_v == 1]
        fill = (
            float(np.average(rook.v_S.fillna(0), weights=rook.poss_S.fillna(0) + 1e-9))
            if len(rook)
            else -0.6
        )
        test["v_hat"] = test.v_p.fillna(fill)
        tr = train.copy()
        tr["v_hat"] = tr.v_p.fillna(fill)

    if cfg.minutes == "model":
        tr = _ranks(tr)
        test = _ranks(test)
        mc = minutes_columns(f)
        bm = _wls(tr[mc].to_numpy(float), tr.mpg_S.to_numpy(float))
        test["mpg_hat"] = np.clip(_predict(bm, test[mc].to_numpy(float)), 0.0, 40.0)
    else:
        rook_mpg = float(train.loc[train.rk == 1, "mpg_S"].mean()) if len(train) else 9.0
        test["mpg_hat"] = np.where(test.rk == 1, rook_mpg, test.mpg_p)

    tot = test.groupby("team_id").mpg_hat.transform("sum")
    test["share5"] = test.mpg_hat / tot * 5.0
    test["contrib"] = test.share5 * test.v_hat
    return test


def apply_availability(proj: pd.DataFrame, games_out: pd.Series, sched: int = 82) -> pd.DataFrame:
    """Scale projected minutes by expected availability and re-spread the shares.

    `games_out` maps person_id to games expected missed. A player out half
    the season keeps half his projected minutes; the minutes he loses go to
    teammates through the same rescale to five players on the floor.
    """
    proj = proj.copy()
    out = proj.person_id.map(games_out).fillna(0.0).clip(0, sched)
    proj["avail"] = 1.0 - out / sched
    proj["mpg_hat"] = proj.mpg_hat * proj.avail
    tot = proj.groupby(["season", "team_id"]).mpg_hat.transform("sum")
    proj["share5"] = proj.mpg_hat / tot * 5.0
    proj["contrib"] = proj.share5 * proj.v_hat
    return proj


def team_ratings(proj: pd.DataFrame) -> pd.DataFrame:
    g = proj.groupby(["season", "team_id"])
    out = g.agg(rating=("contrib", "sum"), n_roster=("person_id", "size")).reset_index()
    key = ["season", "team_id"]
    rook = proj.assign(rs=proj.share5 * proj.rk).groupby(key).rs.sum() / 5
    out["rookie_share"] = out.set_index(key).index.map(rook)
    if "age" in proj.columns and proj.age.notna().any():
        age = (
            proj.assign(wa=proj.share5 * proj.age.fillna(proj.age.median())).groupby(key).wa.sum()
            / 5
        )
        out["team_age"] = out.set_index(key).index.map(age)
    else:
        out["team_age"] = np.nan
    return out
