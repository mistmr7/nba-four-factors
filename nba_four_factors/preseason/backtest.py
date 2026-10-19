"""Walk-forward backtest and live picks for preseason win totals.

Pass 1 (player level): for every season S from the first with three
training seasons, fit the value and minutes models on seasons before S and
sum the opening roster into an out-of-sample team rating.

Pass 2 (team level): for every test season S, fit on out-of-sample ratings
from seasons before S
    margin per game  = a + b * rating
    win pct          = c + d * margin
    blend win pct    = e + f * vegas pct + g * model pct
and predict S. A team's pick is over when the blended win total is above
the line, under when below. Pushes on the actual result are dropped.

Pre-registered split: design choices are judged on the dev window
(2005-06 through 2015-16) only, the holdout window (2016-17 through
2025-26) is reported once per configuration and never used to choose.

Run from repo root:
    python -m nba_four_factors.preseason.backtest
    python -m nba_four_factors.preseason.backtest --prior zero --value raw --minutes naive
    python -m nba_four_factors.preseason.backtest --live 2026_27
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from nba_four_factors.preseason import frame, history
from nba_four_factors.preseason import inputs as io
from nba_four_factors.preseason.project import (
    Config,
    add_design,
    apply_availability,
    project_season,
    team_ratings,
)

FIRST_ROSTER_SEASON = "1998_99"
DEV = ("2005_06", "2015_16")
HOLDOUT = ("2016_17", "2025_26")
MIN_CAL_SEASONS = 4
EDGES = (0.0, 2.0, 3.0, 5.0)


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(x1, y, rcond=None)[0]


def team_outcomes(seasons: list[str]) -> pd.DataFrame:
    rows = []
    for s in seasons:
        g = io.load_games(s)
        t = g.groupby("team_id").agg(margin=("margin", "mean"), ng=("margin", "size"))
        t["wins"] = g.assign(w=g.margin > 0).groupby("team_id").w.sum()
        t["season"] = s
        rows.append(t.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["wpct"] = out.wins / out.ng
    return out


def build_design(prior: str, cfg: Config, live: str | None = None) -> pd.DataFrame:
    seasons = [s for s in io.processed_seasons() if s >= FIRST_ROSTER_SEASON]
    ro = history.all_opening_rosters(seasons)
    if live:
        lr = io.load_live_rosters(live)
        if lr.empty:
            raise SystemExit(
                f"no live rosters under data/raw/preseason/rosters/{live}; "
                "run scripts/pull_preseason_inputs.py --only rosters first"
            )
        lr = lr.rename(columns={"age": "age_live"})[["season", "team_id", "person_id", "age_live"]]
        ro = pd.concat([ro, lr], ignore_index=True)
    f = frame.build_frame(ro, prior=prior)
    return add_design(f, cfg)


def oos_ratings(
    f: pd.DataFrame, cfg: Config, availability: dict[str, pd.Series] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(f.season.unique())
    ratings, players = [], []
    for i, s in enumerate(seasons):
        if i < cfg.min_train_seasons:
            continue
        p = project_season(f, s, cfg)
        if availability and s in availability:
            p = apply_availability(p, availability[s])
        players.append(p)
        ratings.append(team_ratings(p))
    return pd.concat(ratings, ignore_index=True), pd.concat(players, ignore_index=True)


def calibrate(r: pd.DataFrame, team_age: bool = False) -> pd.DataFrame:
    """Pass 2: team-level walk-forward calibration and Vegas blend.

    Ratings are centered within each season first. The league's margins sum
    to zero every year, so only a team's rating relative to the rest of the
    league can matter; without centering, any season-to-season drift in the
    average roster rating (more highly drafted rookies, a different as-of
    window) shows up as every team projecting a win or two too high.
    With team_age, minutes-weighted roster age (also centered) enters the
    margin regression beside the rating.
    """
    r = r.copy()
    r["rating_raw"] = r.rating
    r["rating"] = r.rating - r.groupby("season").rating.transform("mean")
    feats = ["rating"]
    if team_age:
        r["team_age_c"] = r.team_age - r.groupby("season").team_age.transform("mean")
        r["team_age_c"] = r.team_age_c.fillna(0.0)
        feats.append("team_age_c")
    seasons = sorted(r.season.unique())
    out = []
    for i, s in enumerate(seasons):
        if i < MIN_CAL_SEASONS:
            continue
        tr = r[r.season.isin(seasons[:i]) & r.margin.notna()]
        te = r[r.season == s].copy()
        b = _ols(tr[feats].to_numpy(), tr.margin.to_numpy())
        bw = _ols(tr[["margin"]].to_numpy(), tr.wpct.to_numpy())
        te["m_hat"] = b[0] + te[feats].to_numpy() @ b[1:]
        te["w_model"] = bw[0] + bw[1] * te.m_hat
        tv = tr.dropna(subset=["vpct"])
        tv_w = bw[0] + bw[1] * (b[0] + tv[feats].to_numpy() @ b[1:])
        bb = _ols(np.column_stack([tv.vpct, tv_w]), tv.wpct.to_numpy())
        te["w_blend"] = bb[0] + bb[1] * te.vpct + bb[2] * te.w_model
        te["b_rating"] = b[1]
        if team_age:
            te["b_team_age"] = b[2]
        te["blend_vegas_wt"] = bb[1]
        te["blend_model_wt"] = bb[2]
        out.append(te)
    return pd.concat(out, ignore_index=True)


def attach_vegas(r: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    v = io.load_vegas()
    r = r.merge(outcomes, on=["season", "team_id"], how="left")
    r = r.merge(
        v[["season", "team_id", "tricode", "win_total"]], on=["season", "team_id"], how="left"
    )
    r["sched"] = r.season.map(io.schedule_length)
    r["vpct"] = r.win_total / r.sched
    return r


def score(df: pd.DataFrame, window: tuple[str, str]) -> dict[str, float]:
    d = df[(df.season >= window[0]) & (df.season <= window[1])].dropna(subset=["vpct", "wpct"])
    out: dict[str, float] = {"n": float(len(d))}
    actual = d.wpct * 82
    out["vegas_mae"] = float(np.mean(np.abs(d.vpct * 82 - actual)))
    for k in ["w_model", "w_blend"]:
        out[f"{k}_mae"] = float(np.mean(np.abs(d[k] * 82 - actual)))
    for k in ["w_model", "w_blend"]:
        edge = (d[k] - d.vpct) * d.sched
        result = (d.wpct - d.vpct) * d.sched
        valid = result.abs() > 1e-9
        for e in EDGES:
            sel = valid & (edge.abs() > e)
            hits = (np.sign(edge) == np.sign(result))[sel]
            out[f"{k}_hit_{e:g}"] = float(hits.mean()) if len(hits) else float("nan")
            out[f"{k}_n_{e:g}"] = float(sel.sum())
    per = d[(d.wpct - d.vpct).abs() > 1e-9]
    per = per.assign(hit=np.sign(per.w_blend - per.vpct) == np.sign(per.wpct - per.vpct))
    by_season = per.groupby("season").hit.mean()
    out["blend_seasons_over_50"] = float((by_season > 0.5).mean())
    return out


def run(prior: str, cfg: Config) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    f = build_design(prior, cfg)
    print(
        f"inputs: age used={f.attrs['have_age']} draft used={f.attrs['have_draft']} "
        f"opening-night out flag used={f.attrs['use_out_g1']}"
    )
    ratings, _players = oos_ratings(f, cfg)
    outcomes = team_outcomes(sorted(ratings.season.unique()))
    r = attach_vegas(ratings, outcomes)
    cal = calibrate(r, team_age=cfg.team_age)
    return cal, {"dev": score(cal, DEV), "holdout": score(cal, HOLDOUT)}


def live_availability(season: str) -> tuple[pd.Series, pd.DataFrame]:
    """Expected games missed for the live season, matched to the live roster.

    Match is on accent-folded name and team, so two players sharing a name
    on different teams cannot be confused.
    """
    inj = io.load_live_injuries(season)
    names = io.load_live_roster_names(season)
    if inj.empty or names.empty:
        return pd.Series(dtype=float), inj
    key = names.assign(k=names.player.map(io.name_key))[["k", "tricode", "person_id"]]
    inj = inj.assign(k=inj.player.map(io.name_key)).merge(key, on=["k", "tricode"], how="left")
    matched = inj.dropna(subset=["person_id"])
    games = matched.groupby("person_id").games_out.max()
    games.index = games.index.astype(int)
    return games, inj


def live(prior: str, cfg: Config, season: str) -> pd.DataFrame:
    f = build_design(prior, cfg, live=season)
    games_out, inj = live_availability(season)
    if not inj.empty:
        missing = inj[inj.person_id.isna()].player.tolist()
        print(f"availability: {int(inj.person_id.notna().sum())} of {len(inj)} injury rows matched")
        if missing:
            print("  not on a live roster: " + ", ".join(missing))
    ratings, players = oos_ratings(f, cfg, {season: games_out})
    outcomes = team_outcomes([s for s in sorted(ratings.season.unique()) if s != season])
    r = attach_vegas(ratings, outcomes)
    cal = calibrate(r, team_age=cfg.team_age)
    picks = cal[cal.season == season].copy()
    picks["proj_wins_model"] = picks.w_model * picks.sched
    picks["proj_wins_blend"] = picks.w_blend * picks.sched
    picks["edge"] = picks.proj_wins_blend - picks.win_total
    picks["pick"] = np.where(picks.edge > 0, "OVER", "UNDER")
    io.OUT.mkdir(parents=True, exist_ok=True)
    tag = "_team_age" if cfg.team_age else ""
    players[players.season == season].to_parquet(io.OUT / f"live_players_{season}{tag}.parquet")
    picks.to_csv(io.OUT / f"live_picks_{season}{tag}.csv", index=False)
    return picks


def _print(name: str, s: dict[str, float]) -> None:
    print(
        f"  {name:8s} n={int(s['n']):4d}  MAE vegas {s['vegas_mae']:.2f}  model {s['w_model_mae']:.2f}"
        f"  blend {s['w_blend_mae']:.2f}"
    )
    for k in ["w_model", "w_blend"]:
        parts = [f">{e:g}: {s[f'{k}_hit_{e:g}']:.3f} ({int(s[f'{k}_n_{e:g}'])})" for e in EDGES]
        print(f"           {k:8s} O/U hit  " + "  ".join(parts))
    print(f"           seasons with blend hit rate above 50%: {s['blend_seasons_over_50']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["theta", "zero"], default="theta")
    ap.add_argument("--value", choices=["model", "raw"], default="model")
    ap.add_argument("--minutes", choices=["model", "naive"], default="model")
    ap.add_argument("--no-age", action="store_true")
    ap.add_argument("--no-draft", action="store_true")
    ap.add_argument("--out-g1", action="store_true")
    ap.add_argument("--team-age", action="store_true")
    ap.add_argument("--no-healthy-minutes", action="store_true")
    ap.add_argument(
        "--value-terms", choices=["base", "kink", "young", "kink+young"], default="base"
    )
    ap.add_argument("--live", default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    cfg = replace(
        Config(),
        value=args.value,
        minutes=args.minutes,
        use_age=not args.no_age,
        use_draft=not args.no_draft,
        use_out_g1=args.out_g1,
        team_age=args.team_age,
        healthy_minutes=not args.no_healthy_minutes,
        value_terms=args.value_terms,
    )
    if args.live:
        picks = live(args.prior, cfg, args.live)
        cols = ["tricode", "win_total", "proj_wins_model", "proj_wins_blend", "edge", "pick"]
        print(picks.sort_values("edge")[cols].round(1).to_string(index=False))
        return
    cal, scores = run(args.prior, cfg)
    print(f"prior={args.prior} value={args.value} minutes={args.minutes}")
    for name, s in scores.items():
        _print(name, s)
    if args.save:
        io.OUT.mkdir(parents=True, exist_ok=True)
        tag = f"{args.prior}_{args.value}_{args.minutes}" + ("_team_age" if cfg.team_age else "")
        cal.to_parquet(io.OUT / f"backtest_{tag}.parquet")


if __name__ == "__main__":
    main()
