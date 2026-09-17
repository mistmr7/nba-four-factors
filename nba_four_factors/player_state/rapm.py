"""Offense/defense regularized adjusted plus-minus from the stint table.

Every stint contributes two observations, one per offensive team:

    points per 100 possessions = intercept
                                 + home_court * [offense is the home team]
                                 + sum(offense coefficients of the five attackers)
                                 + sum(defense coefficients of the five defenders)

Rows are weighted by the stint's possessions (Oliver's estimate, the same
count the four-factors tables use). Player coefficients are shrunk toward a
prior with ridge penalty lambda; intercept and home court are never
penalized. With a zero prior this is ordinary RAPM. With theta_pred from the
scalar Kalman filter as the prior it is the box-prior family (RPM, LEBRON),
and the two are run side by side so the comparison is a result, not a choice.

Sign convention: ORAPM is the offense coefficient (positive good). DRAPM is
the negated defense coefficient so positive is good on both ends.
RAPM = ORAPM + DRAPM, in points per 100 possessions.

As-of-date fits: one fit per weekly cutoff on every stint strictly before
the cutoff, so a rating dated d is usable for any game on or after d with
no leakage. Older stints can be down-weighted with a half-life in days.

Outputs, under data/player_state/:
    rapm_cv.csv                     lambda cross-validation curve
    rapm_season.parquet             one fit per season, all seasons stacked
    rapm_asof_<prior>.parquet       weekly as-of ratings, prior in {zero, theta}

Run from repo root:
    python -m nba_four_factors.player_state.rapm cv --seasons 2023_24
    python -m nba_four_factors.player_state.rapm fit --lam 3000
    python -m nba_four_factors.player_state.rapm asof --lam 3000 --prior zero
    python -m nba_four_factors.player_state.rapm asof --lam 3000 --prior theta --prior-scale 0.35
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .stints import AWAY_COLS, HOME_COLS, available_seasons, load_stints

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "player_state"
THETA = OUT_DIR / "player_form_predictive_wf.parquet"
CV_PATH = OUT_DIR / "rapm_cv.csv"
SEASON_PATH = OUT_DIR / "rapm_season.parquet"

DEFAULT_LAMBDA = 3000.0
DEFAULT_LAMBDAS = (500.0, 1000.0, 2000.0, 3000.0, 5000.0, 8000.0)
WARMUP_DAYS = 14
STEP_DAYS = 7
MIN_STINTS = 500

PriorFn = Callable[[pd.Timestamp], tuple[pd.Series | None, pd.Series | None]]


# ---------------------------------------------------------------------------
# Design matrix and solver
# ---------------------------------------------------------------------------


def aggregate_lineup_pairs(stints: pd.DataFrame, weight_col: str = "weight") -> pd.DataFrame:
    """Collapse stints with identical ten-player lineups into one row.

    Weighted least squares on rows with identical predictors equals a single
    row carrying the summed weight and the weight-averaged target, so this
    loses nothing and shrinks the design matrix several-fold. weight_col
    scales possessions and points alike (time decay); it defaults to 1.
    """
    df = stints.copy()
    if weight_col not in df.columns:
        df[weight_col] = 1.0
    for c in ("home_poss", "away_poss", "home_pts", "away_pts"):
        df[c] = df[c] * df[weight_col]
    keys = HOME_COLS + AWAY_COLS
    return df.groupby(keys, as_index=False)[
        ["home_poss", "away_poss", "home_pts", "away_pts", "seconds"]
    ].sum()


def build_design(stints: pd.DataFrame):
    """(X, y, w, players) with columns [intercept, home_court, off_<p>..., def_<p>...]."""
    players = np.unique(stints[HOME_COLS + AWAY_COLS].to_numpy().ravel())
    idx = {pid: i for i, pid in enumerate(players)}
    n_p = len(players)
    off0, def0 = 2, 2 + n_p

    H = stints[HOME_COLS].to_numpy()
    A = stints[AWAY_COLS].to_numpy()
    hp = stints["home_poss"].to_numpy(dtype=float)
    ap = stints["away_poss"].to_numpy(dtype=float)
    hpts = stints["home_pts"].to_numpy(dtype=float)
    apts = stints["away_pts"].to_numpy(dtype=float)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    y: list[float] = []
    w: list[float] = []
    r = 0
    for i in range(len(stints)):
        for off_ids, def_ids, pts, poss, home_court in (
            (H[i], A[i], hpts[i], hp[i], 1.0),
            (A[i], H[i], apts[i], ap[i], 0.0),
        ):
            if poss <= 0:
                continue
            rows.extend([r] * 12)
            cols.extend([0, 1])
            vals.extend([1.0, home_court])
            cols.extend(off0 + idx[p] for p in off_ids)
            cols.extend(def0 + idx[p] for p in def_ids)
            vals.extend([1.0] * 10)
            y.append(100.0 * pts / poss)
            w.append(poss)
            r += 1
    X = sp.csr_matrix((vals, (rows, cols)), shape=(r, 2 + 2 * n_p))
    return X, np.asarray(y), np.asarray(w), players


def ridge_solve(
    X, y, w, lam: float, prior: np.ndarray | None = None, unpenalized: int = 2
) -> np.ndarray:
    """Weighted ridge toward a prior: (X'WX + lam P) b = X'W y + lam P prior."""
    p = X.shape[1]
    if prior is None:
        prior = np.zeros(p)
    Xw = X.multiply(w[:, None]).tocsr()
    A = (X.T @ Xw).toarray()
    rhs = np.asarray(X.T @ (w * y)).ravel()
    pen = np.ones(p)
    pen[:unpenalized] = 0.0
    A[np.diag_indices(p)] += lam * pen
    rhs = rhs + lam * pen * prior
    return np.linalg.solve(A, rhs)


def prior_vector(
    players: np.ndarray, prior_off: pd.Series | None, prior_def: pd.Series | None
) -> np.ndarray:
    n_p = len(players)
    prior = np.zeros(2 + 2 * n_p)
    if prior_off is not None:
        prior[2 : 2 + n_p] = prior_off.reindex(players).fillna(0.0).to_numpy()
    if prior_def is not None:
        prior[2 + n_p :] = -prior_def.reindex(players).fillna(0.0).to_numpy()
    return prior


@dataclass
class RapmFit:
    ratings: pd.DataFrame
    intercept: float
    home_court: float
    lam: float
    n_rows: int
    n_players: int


def fit_rapm(
    stints: pd.DataFrame,
    lam: float,
    prior_off: pd.Series | None = None,
    prior_def: pd.Series | None = None,
    aggregate: bool = True,
) -> RapmFit:
    """Fit O/D RAPM. Priors are Series indexed by person_id in points per 100, positive good."""
    df = aggregate_lineup_pairs(stints) if aggregate else stints
    X, y, w, players = build_design(df)
    n_p = len(players)
    b = ridge_solve(X, y, w, lam, prior_vector(players, prior_off, prior_def))
    off_poss = np.asarray(X[:, 2 : 2 + n_p].T @ w).ravel()
    def_poss = np.asarray(X[:, 2 + n_p :].T @ w).ravel()
    ratings = pd.DataFrame(
        {
            "person_id": players,
            "orapm": b[2 : 2 + n_p],
            "drapm": -b[2 + n_p :],
            "off_poss": off_poss,
            "def_poss": def_poss,
        }
    )
    ratings["rapm"] = ratings["orapm"] + ratings["drapm"]
    return RapmFit(ratings, float(b[0]), float(b[1]), lam, X.shape[0], n_p)


def cv_lambda(
    stints: pd.DataFrame, lambdas=DEFAULT_LAMBDAS, k: int = 5, seed: int = 7
) -> pd.DataFrame:
    """K-fold CV over lambda, folds split by game so a stint never predicts itself."""
    df = stints[(stints.home_poss > 0) & (stints.away_poss > 0)].reset_index(drop=True)
    X, y, w, _ = build_design(df)
    groups = np.repeat(df["game_id"].to_numpy(), 2)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    fold_of = dict(zip(uniq, rng.integers(0, k, len(uniq)), strict=True))
    folds = np.array([fold_of[g] for g in groups])
    out = []
    for lam in lambdas:
        sse, sw = 0.0, 0.0
        for f in range(k):
            tr, te = folds != f, folds == f
            b = ridge_solve(X[tr], y[tr], w[tr], lam)
            pred = X[te] @ b
            sse += float(np.sum(w[te] * (y[te] - pred) ** 2))
            sw += float(np.sum(w[te]))
        out.append({"lambda": lam, "weighted_mse": sse / sw})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# As-of-date schedule
# ---------------------------------------------------------------------------


def weekly_cutoffs(
    game_dates: pd.Series, warmup_days: int = WARMUP_DAYS, step_days: int = STEP_DAYS
) -> list[pd.Timestamp]:
    d = pd.to_datetime(game_dates)
    first, last = d.min().normalize(), d.max().normalize()
    return list(
        pd.date_range(
            first + pd.Timedelta(days=warmup_days),
            last + pd.Timedelta(days=step_days),
            freq=f"{step_days}D",
        )
    )


def decay_weights(
    game_dates: pd.Series, cutoff: pd.Timestamp, halflife_days: float | None
) -> np.ndarray:
    if halflife_days is None:
        return np.ones(len(game_dates))
    age = (cutoff - pd.to_datetime(game_dates)).dt.days.to_numpy().astype(float)
    return np.power(0.5, age / halflife_days)


def fit_asof(
    stints: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    lam: float,
    halflife_days: float | None = None,
    window_days: int | None = None,
    prior_fn: PriorFn | None = None,
    min_stints: int = MIN_STINTS,
) -> pd.DataFrame:
    """One fit per cutoff on stints dated strictly before it; results stacked with asof_date."""
    st = stints.copy()
    st["game_date"] = pd.to_datetime(st["game_date"])
    out = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        sub = st[st["game_date"] < cutoff]
        if window_days is not None:
            sub = sub[sub["game_date"] >= cutoff - pd.Timedelta(days=window_days)]
        if len(sub) < min_stints:
            continue
        sub = sub.assign(weight=decay_weights(sub["game_date"], cutoff, halflife_days))
        prior_off, prior_def = prior_fn(cutoff) if prior_fn else (None, None)
        fit = fit_rapm(sub, lam, prior_off=prior_off, prior_def=prior_def)
        out.append(
            fit.ratings.assign(
                asof_date=cutoff,
                intercept=fit.intercept,
                home_court=fit.home_court,
                n_stints=len(sub),
            )
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def ratings_for_date(asof: pd.DataFrame, game_date: pd.Timestamp) -> pd.DataFrame:
    """The latest as-of ratings dated on or before a game date."""
    cuts = asof["asof_date"].drop_duplicates().sort_values()
    eligible = cuts[cuts <= pd.Timestamp(game_date)]
    if eligible.empty:
        return asof.iloc[0:0]
    return asof[asof["asof_date"] == eligible.iloc[-1]]


# ---------------------------------------------------------------------------
# Kalman theta_pred as a prior
# ---------------------------------------------------------------------------


def load_theta() -> pd.DataFrame:
    """Walk-forward theta_pred rows (per-36 Game Score level) with dates."""
    t = pd.read_parquet(
        THETA, columns=["person_id", "game_date", "minutes", "theta_pred", "status"]
    )
    t = t[t["status"] == "played"].copy()
    t["game_date"] = pd.to_datetime(t["game_date"])
    return t.sort_values("game_date")


def theta_asof(theta: pd.DataFrame, cutoff: pd.Timestamp, lookback_days: int = 400) -> pd.Series:
    """Each player's latest theta_pred before the cutoff, centered on the minutes-weighted mean."""
    sub = theta[
        (theta["game_date"] < cutoff)
        & (theta["game_date"] >= cutoff - pd.Timedelta(days=lookback_days))
    ]
    last = sub.groupby("person_id").tail(1).set_index("person_id")
    if last.empty:
        return pd.Series(dtype=float)
    mu = np.average(last["theta_pred"], weights=np.maximum(last["minutes"], 1.0))
    return last["theta_pred"] - mu


def make_theta_prior(scale: float, split: float = 0.5) -> PriorFn:
    """Prior in points per 100: scale * centered theta, split between offense and defense."""
    theta = load_theta()

    def prior_fn(cutoff: pd.Timestamp):
        centered = theta_asof(theta, cutoff) * scale
        return centered * split, centered * (1.0 - split)

    return prior_fn


def estimate_prior_scale(season_ratings: pd.DataFrame, min_poss: float = 2000.0) -> float:
    """Least squares slope of season RAPM on season-end centered theta, pooled over seasons.

    Use ratings from seasons before the ones the prior will be applied to, so
    the scale is a training-set quantity like every other parameter in the lane.
    """
    theta = load_theta()
    xs, ys = [], []
    for _season, r in season_ratings.groupby("season"):
        r = r[r["off_poss"] >= min_poss]
        end = r["season_end"].iloc[0] if "season_end" in r.columns else None
        if end is None:
            continue
        t = theta_asof(theta, pd.Timestamp(end) + pd.Timedelta(days=1))
        j = r.set_index("person_id")["rapm"].to_frame().join(t.rename("theta"), how="inner")
        xs.append(j["theta"].to_numpy())
        ys.append(j["rapm"].to_numpy())
    x = np.concatenate(xs)
    yv = np.concatenate(ys)
    x = x - x.mean()
    return float(np.dot(x, yv - yv.mean()) / np.dot(x, x))


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def fit_seasons(seasons: list[str], lam: float) -> pd.DataFrame:
    out = []
    for season in seasons:
        st = load_stints([season])
        fit = fit_rapm(st, lam)
        r = fit.ratings.assign(
            season=season,
            season_end=pd.to_datetime(st["game_date"]).max(),
            intercept=fit.intercept,
            home_court=fit.home_court,
            lam=lam,
        )
        out.append(r)
        print(
            f"{season} players {fit.n_players} rows {fit.n_rows} intercept {fit.intercept:.2f} hca {fit.home_court:.2f}"
        )
    return pd.concat(out, ignore_index=True)


def asof_seasons(
    seasons: list[str],
    lam: float,
    prior: str,
    prior_scale: float,
    halflife_days: float | None,
    window_days: int | None,
    trailing_seasons: int,
) -> pd.DataFrame:
    """Weekly as-of fits per season, each on that season plus the trailing seasons before it."""
    all_seasons = available_seasons("regular_season")
    prior_fn = make_theta_prior(prior_scale) if prior == "theta" else None
    out = []
    for season in seasons:
        i = all_seasons.index(season)
        window = all_seasons[max(0, i - trailing_seasons) : i + 1]
        st = load_stints(window)
        st["game_date"] = pd.to_datetime(st["game_date"])
        this = st[st["season"] == season]
        cuts = weekly_cutoffs(this["game_date"])
        t0 = time.time()
        asof = fit_asof(
            st, cuts, lam, halflife_days=halflife_days, window_days=window_days, prior_fn=prior_fn
        )
        asof["season"] = season
        asof["prior"] = prior
        asof["lam"] = lam
        out.append(asof)
        print(
            f"{season} {asof.asof_date.nunique()} cutoffs, {len(asof):,} rows, {time.time() - t0:.0f}s"
        )
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("cv")
    s.add_argument("--seasons", nargs="*", default=["2023_24"])
    s.add_argument("--lambdas", default=",".join(str(int(x)) for x in DEFAULT_LAMBDAS))

    s = sub.add_parser("fit")
    s.add_argument("--seasons", nargs="*", default=None)
    s.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)

    s = sub.add_parser("asof")
    s.add_argument("--seasons", nargs="*", default=None)
    s.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    s.add_argument("--prior", choices=["zero", "theta"], default="zero")
    s.add_argument(
        "--prior-scale",
        type=float,
        default=0.0,
        help="points per 100 per unit of centered theta_pred",
    )
    s.add_argument("--halflife-days", type=float, default=365.0)
    s.add_argument("--window-days", type=int, default=None)
    s.add_argument("--trailing-seasons", type=int, default=2)

    s = sub.add_parser("prior-scale")
    s.add_argument("--through", default="2017_18", help="last season used to estimate the scale")

    a = ap.parse_args()
    seasons = getattr(a, "seasons", None) or available_seasons("regular_season")

    if a.cmd == "cv":
        st = load_stints(seasons)
        cv = cv_lambda(st, lambdas=[float(x) for x in a.lambdas.split(",")])
        cv.to_csv(CV_PATH, index=False)
        print(cv.to_string(index=False))
        print(f"best lambda {cv.sort_values('weighted_mse')['lambda'].iloc[0]:.0f}")
    elif a.cmd == "fit":
        r = fit_seasons(seasons, a.lam)
        r.to_parquet(SEASON_PATH, index=False)
        print(f"{len(r):,} rows -> {SEASON_PATH}")
    elif a.cmd == "asof":
        if a.prior == "theta" and a.prior_scale <= 0:
            raise SystemExit("--prior theta needs --prior-scale > 0 (see the prior-scale command)")
        r = asof_seasons(
            seasons,
            a.lam,
            a.prior,
            a.prior_scale,
            a.halflife_days,
            a.window_days,
            a.trailing_seasons,
        )
        path = OUT_DIR / f"rapm_asof_{a.prior}.parquet"
        r.to_parquet(path, index=False)
        print(f"{len(r):,} rows -> {path}")
    elif a.cmd == "prior-scale":
        r = pd.read_parquet(SEASON_PATH)
        r = r[r["season"] <= a.through]
        print(
            f"prior scale through {a.through}: {estimate_prior_scale(r):.4f} points per 100 per theta unit"
        )


if __name__ == "__main__":
    main()
