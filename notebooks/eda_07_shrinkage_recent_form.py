"""
EDA 07: Recent form and reliability-weighted shrinkage.

Purpose
This notebook verifies the methodological choices behind the recent-form
feature and its empirical-Bayes shrinkage before any of it is committed to
the methods section as a number. Each section tests one claim and prints a
readout so you can see whether reality matches the design.

Claims under test
C1. Standardizing once at the factor level yields differentials with mean
    near 0 and standard deviation near 1.
C2. The composite NET built from standardized differentials sits on a
    sensible, interpretable scale.
C3. The trailing rolling mean is causal. It excludes the current game and
    leaks no future information.
C4. Within-season team performance is autocorrelated, so the effective
    sample size of a W-game window is below W. EDA 06 found roughly 7 for
    W equal to 15.
C5. The variance decomposition yields a positive signal variance, so the
    reliability weight is well defined.
C6. Per-team reliability weights span a meaningful range and rise with team
    consistency.
C7. Shrinkage pulls noisy form toward zero while preserving magnitude for
    consistent teams.
C8. Shrunk form predicts out of sample at least as well as raw form, and a
    best window is identifiable.

Methodological notes
Standardization happens once, at the factor-differential level. The rolling
mean inherits that standardization and is not re-standardized. The shrinkage
weight is a variance ratio, not a z-score: w equals signal variance over
signal plus noise variance. Signal variance is a league-wide quantity. Noise
variance is per team, from that team's own rolling spread, corrected for the
effective sample size. In the predictive sweep the variance components and
the effective sample size are estimated on training folds only, which keeps
the window and shrinkage selection leakage-free.

Run order
Set SMOKE_TEST to True first and run top to bottom. The synthetic generator
builds team-season series with a known autocorrelation and a known
between-team signal, and the sections confirm the estimators recover those
known values. Once the smoke test passes, set SMOKE_TEST to False and point
DATA_PATH at the processed four-factor differentials.
"""

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thesis_style import apply_thesis_style

apply_thesis_style()

# %%
# ----------------------------------------------------------------------
# CONFIG
# Adjust these to match your processed layer, then run top to bottom.
# ----------------------------------------------------------------------

# Toggle the synthetic smoke test. Run with True first to validate the
# estimators, then set to False to run on real data.
SMOKE_TEST = False

# Path to the processed per-team-per-game four-factor differentials.
# Expected one row per team per game, two rows per contest.
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "four_factor_diffs.parquet"

# Where figures are written. Gitignored, regenerated locally.
FIG_DIR = Path(__file__).resolve().parent / "figures" / "eda_07"

# Column names in the processed file. Rename the right-hand strings only.
COL_SEASON = "season"
COL_GAME_DATE = "game_date"
COL_TEAM = "team_id"
COL_EFG = "eFG_diff"
COL_TOV = "TOV_diff"
COL_OREB = "OREB_diff"
COL_FTR = "FTR_diff"

# Outcome column used as the prediction target in the window sweep.
# This should be the team's realized net performance in the game.
# If you have an actual point margin, point this at it. Otherwise the
# composite NET of the current game is used as the target.
COL_MARGIN = "margin"

# Whether COL_MARGIN exists in the file. If False, the sweep targets the
# current-game composite NET instead.
HAVE_MARGIN = False

# The four differentials are assumed pre-signed so that a positive value is
# good for the team. eFG_diff is team minus opponent. TOV_diff is opponent
# minus own. OREB_diff is team minus opponent. FTR_diff is team minus
# opponent. Confirm this matches your construction.

# Per-era standardization applies to FTR only. Era cut is the season start
# year. EDA 06 found FTR variance differs pre and post 2012.
FTR_ERA_CUT_YEAR = 2012

# Composite weights applied to the standardized differentials.
# These are placeholders. The refit weights from the regression baseline
# replace them later. The form and shrinkage machinery is robust to the
# exact weights, so any reasonable set is fine for verification.
COMPOSITE_WEIGHTS = {
    COL_EFG: 0.40,
    COL_TOV: 0.25,
    COL_OREB: 0.20,
    COL_FTR: 0.15,
}

# Candidate rolling windows for the sweep.
WINDOW_CANDIDATES = [5, 10, 15, 20, 30, 40]

# The primary window used for the descriptive sections 3 through 7.
PRIMARY_WINDOW = 15

# Minimum games required before a rolling feature is emitted. Below this the
# rolling estimate is too thin to trust and the row is dropped from feature
# use.
MIN_PERIODS = 5

FACTOR_COLS = [COL_EFG, COL_TOV, COL_OREB, COL_FTR]

FIG_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ----------------------------------------------------------------------
# Helpers: season parsing and effective sample size
# ----------------------------------------------------------------------


def season_start_year(season: str) -> int:
    """Return the starting calendar year of a season string like 2015-16."""
    return int(str(season)[:4])


def effective_sample_size_ar1(series: np.ndarray, window: int) -> float:
    """
    Effective sample size of the mean of `window` consecutive observations
    under an AR(1) approximation, using the lag-1 autocorrelation of `series`.

    For independent data this returns `window`. For positively autocorrelated
    data it returns less than `window`, which is the quantity the shrinkage
    noise term must divide by.
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return float(window)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return float(window)
    rho = np.sum(x[1:] * x[:-1]) / denom
    rho = float(np.clip(rho, -0.999, 0.999))
    ks = np.arange(1, window)
    inflation = 1.0 + 2.0 * np.sum((1.0 - ks / window) * (rho**ks))
    if inflation <= 0:
        return float(window)
    return float(window / inflation)


def effective_sample_size_from_shrink(sigma_pergame: float, sigma_rolling: float) -> float:
    """
    Effective sample size implied by how much the rolling mean shrinks the
    per-game spread. This is the EDA 06 method and serves as a cross-check on
    the autocorrelation estimate.
    """
    if sigma_rolling <= 0:
        return float("nan")
    return float((sigma_pergame / sigma_rolling) ** 2)


def center_within_team_season(frame: pd.DataFrame) -> pd.Series:
    """
    Subtract each team-season's own mean from the composite NET. The result is
    the noise wiggle with the team's fixed strength removed, which is what the
    n_eff and signal-variance estimators assume they are measuring. This is an
    offline, full-season operation used only for league-constant estimation,
    never for a live feature.
    """
    grp = frame.groupby([COL_TEAM, COL_SEASON])["composite_net"]
    return frame["composite_net"] - grp.transform("mean")


def noise_variance_first_diff(frame: pd.DataFrame, rho: float) -> float:
    """
    Estimate the per-game noise variance from first differences within each
    team-season. Differencing consecutive games cancels the team's fixed
    strength without estimating a sample mean, which avoids the bias that
    season-mean centering introduces on short series. For an AR(1) noise
    process the variance of consecutive differences equals 2 * noise_var *
    (1 - rho), so noise_var is recovered by inverting that relationship.
    """
    diffs = frame.groupby([COL_TEAM, COL_SEASON])["composite_net"].diff().dropna()
    diff_var = float(diffs.var(ddof=1))
    denom = 2.0 * (1.0 - rho)
    if denom <= 0:
        return float("nan")
    return diff_var / denom


def lag1_autocorr(series: np.ndarray) -> float:
    """Lag-1 autocorrelation of a series, used to invert the first-difference
    noise estimate."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return float("nan")
    return float(np.clip(np.sum(x[1:] * x[:-1]) / denom, -0.999, 0.999))


# %%
# ----------------------------------------------------------------------
# Synthetic data generator for the smoke test
# ----------------------------------------------------------------------


def make_synthetic(
    n_teams: int = 30,
    n_seasons: int = 10,
    games_per_season: int = 82,
    rho: float = 0.6,
    signal_sd: float = 0.5,
    noise_sd: float = 0.9,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Build a synthetic per-team-per-game frame with known structure.

    Each team-season has a true latent strength drawn with standard deviation
    `signal_sd`. The observed per-game composite is that strength plus an
    AR(1) noise process with lag-1 autocorrelation `rho` and innovation scale
    set so the marginal noise standard deviation is `noise_sd`. The four
    factor columns are filled with the same observed value split across the
    composite weights, which is enough to exercise the pipeline. Known truth:
    the between-team signal variance is signal_sd squared, and the noise
    autocorrelation is rho.
    """
    rng = np.random.default_rng(seed)
    rows = []
    innovation_sd = noise_sd * np.sqrt(1.0 - rho**2)
    weight_sum = sum(COMPOSITE_WEIGHTS.values())
    for s in range(n_seasons):
        season = f"{2000 + s}-{str(2000 + s + 1)[2:]}"
        for t in range(n_teams):
            strength = rng.normal(0.0, signal_sd)
            noise = np.empty(games_per_season)
            noise[0] = rng.normal(0.0, noise_sd)
            for g in range(1, games_per_season):
                noise[g] = rho * noise[g - 1] + rng.normal(0.0, innovation_sd)
            observed = strength + noise
            for g in range(games_per_season):
                row = {
                    COL_SEASON: season,
                    COL_GAME_DATE: pd.Timestamp("2000-10-01") + pd.Timedelta(days=g + s * 400),
                    COL_TEAM: f"T{t:02d}",
                }
                for col in FACTOR_COLS:
                    row[col] = observed[g] * COMPOSITE_WEIGHTS[col] / weight_sum
                row[COL_MARGIN] = observed[g]
                rows.append(row)
    df = pd.DataFrame(rows)
    return df


# %%
# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------

if SMOKE_TEST:
    print("Running on SYNTHETIC data. Known rho = 0.6, signal_sd = 0.5, noise_sd = 0.9.")
    df = make_synthetic()
    HAVE_MARGIN = True
else:
    print(f"Loading real data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

df = df.sort_values([COL_TEAM, COL_SEASON, COL_GAME_DATE]).reset_index(drop=True)
print(f"Rows: {len(df):,}  Teams: {df[COL_TEAM].nunique()}  Seasons: {df[COL_SEASON].nunique()}")

# %%
# ----------------------------------------------------------------------
# Section 1: Standardize the four differentials once, at the factor level
# Tests C1.
# ----------------------------------------------------------------------


def standardize_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each differential using cross-sectional spread. Pooled for eFG,
    TOV, OREB. Per-era for FTR. Mean is centered by construction of the
    differentials, so this is primarily a spread rescale.
    """
    out = frame.copy()
    for col in [COL_EFG, COL_TOV, COL_OREB]:
        mu = out[col].mean()
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = (out[col] - mu) / sd
    era = out[COL_SEASON].map(season_start_year) >= FTR_ERA_CUT_YEAR
    out["_era_post"] = era
    z_ftr = pd.Series(index=out.index, dtype=float)
    for _is_post, grp in out.groupby("_era_post"):
        mu = grp[COL_FTR].mean()
        sd = grp[COL_FTR].std(ddof=0)
        z_ftr.loc[grp.index] = (grp[COL_FTR] - mu) / sd
    out[f"{COL_FTR}_z"] = z_ftr
    out = out.drop(columns="_era_post")
    return out


df = standardize_factors(df)

print("Standardized factor checks (target mean ~0, sd ~1):")
for col in FACTOR_COLS:
    z = df[f"{col}_z"]
    print(f"  {col}_z: mean={z.mean():+.4f}  sd={z.std(ddof=0):.4f}")

# %%
# ----------------------------------------------------------------------
# Section 2: Build the composite NET from the standardized differentials
# Tests C2.
# ----------------------------------------------------------------------


def build_composite(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Composite NET is the weighted sum of the standardized differentials. The
    differentials are pre-signed so positive is good, so all weights are
    positive and a higher NET means a better single-game performance.
    """
    out = frame.copy()
    net = np.zeros(len(out))
    for col, w in COMPOSITE_WEIGHTS.items():
        net = net + w * out[f"{col}_z"].to_numpy()
    out["composite_net"] = net
    return out


df = build_composite(df)

net = df["composite_net"]
print("Composite NET scale:")
print(
    f"  mean={net.mean():+.4f}  sd={net.std(ddof=0):.4f}  min={net.min():+.3f}  max={net.max():+.3f}"
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(net, bins=60, color="#4C78A8", alpha=0.85)
ax.set_title("Composite NET distribution (standardized differentials)")
ax.set_xlabel("Composite NET")
ax.set_ylabel("Count of team-games")
fig.tight_layout()
fig.savefig(FIG_DIR / "2_composite_net_distribution.png", dpi=130)
plt.close(fig)

# %%
# ----------------------------------------------------------------------
# Section 3: Trailing rolling mean of the composite, causal
# Tests C3.
# ----------------------------------------------------------------------


def add_rolling_form(frame: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    """
    Trailing rolling mean of composite NET per team within a season. The
    shift by one excludes the current game, which is what makes the feature
    causal. Rolling spread is also computed for the shrinkage noise term.
    """
    out = frame.copy()
    grp = out.groupby([COL_TEAM, COL_SEASON])["composite_net"]
    shifted = grp.shift(1)
    roll = shifted.groupby([out[COL_TEAM], out[COL_SEASON]])
    out[f"form_mean_{window}"] = roll.transform(
        lambda s: s.rolling(window, min_periods=min_periods).mean()
    )
    out[f"form_var_{window}"] = roll.transform(
        lambda s: s.rolling(window, min_periods=min_periods).var(ddof=1)
    )
    return out


df = add_rolling_form(df, PRIMARY_WINDOW, MIN_PERIODS)

form_col = f"form_mean_{PRIMARY_WINDOW}"
leak_check = df.groupby([COL_TEAM, COL_SEASON]).head(1)[form_col].notna().sum()
print("Causality check:")
print(f"  Rows that are the first game of a team-season with a non-null form: {leak_check}")
print("  This must be 0. A non-zero value means the current game leaked into its own feature.")

valid_form = df[form_col].dropna()
print(
    f"  Rolling form (window {PRIMARY_WINDOW}): mean={valid_form.mean():+.4f}  sd={valid_form.std(ddof=0):.4f}"
)

# %%
# ----------------------------------------------------------------------
# Section 4: Effective sample size from autocorrelation
# Tests C4.
# ----------------------------------------------------------------------

df["composite_net_centered"] = center_within_team_season(df)

per_team_neff = []
for (_team, _season), grp in df.groupby([COL_TEAM, COL_SEASON]):
    series = grp["composite_net_centered"].to_numpy()
    if np.sum(~np.isnan(series)) >= PRIMARY_WINDOW:
        per_team_neff.append(effective_sample_size_ar1(series, PRIMARY_WINDOW))

neff_ar1 = float(np.nanmean(per_team_neff)) if per_team_neff else float("nan")
per_team_rho = []
for (_team, _season), grp in df.groupby([COL_TEAM, COL_SEASON]):
    series = grp["composite_net_centered"].to_numpy()
    if np.sum(~np.isnan(series)) >= PRIMARY_WINDOW:
        r = lag1_autocorr(series)
        if not np.isnan(r):
            per_team_rho.append(r)

rho_hat = float(np.nanmean(per_team_rho)) if per_team_rho else float("nan")
print(f"  pooled lag-1 autocorrelation (rho): {rho_hat:+.3f}  (synthetic truth is 0.6)")
sigma_pergame = df["composite_net_centered"].std(ddof=0)
sigma_rolling = (
    df.groupby([COL_TEAM, COL_SEASON])["composite_net_centered"]
    .transform(lambda s: s.rolling(PRIMARY_WINDOW, min_periods=MIN_PERIODS).mean())
    .std(ddof=0)
)
neff_shrink = effective_sample_size_from_shrink(sigma_pergame, sigma_rolling)

print(f"Effective sample size for a {PRIMARY_WINDOW}-game window:")
print(f"  AR(1) autocorrelation method: n_eff ~ {neff_ar1:.2f}")
print(f"  Sigma-shrink method (EDA 06 style): n_eff ~ {neff_shrink:.2f}")
print(f"  Naive independent assumption would be {PRIMARY_WINDOW}.")
if SMOKE_TEST:
    print(
        "  Smoke test: with rho = 0.6 the two methods should roughly agree and sit below the window."
    )

# %%
# ----------------------------------------------------------------------
# Section 5: Variance decomposition
# Tests C5.
# ----------------------------------------------------------------------


def variance_components(frame: pd.DataFrame, window: int, n_eff: float, rho: float) -> dict:
    """
    Observed variance is the cross-sectional spread of the rolling form across
    all team-games. Per-game noise variance is estimated by first differences
    to avoid the season-mean centering bias. Noise variance of the rolling
    mean is that per-game noise divided by the effective sample size. Signal
    variance is the observed spread net of that noise.
    """
    form = frame[f"form_mean_{window}"]
    observed_var = float(form.var(ddof=0))
    pergame_noise = noise_variance_first_diff(frame, rho)
    noise_of_mean = pergame_noise / n_eff
    signal_var = observed_var - noise_of_mean
    return {
        "observed_var": observed_var,
        "noise_var": noise_of_mean,
        "signal_var": signal_var,
        "pergame_noise": pergame_noise,
        "global_w": signal_var / observed_var if observed_var > 0 else float("nan"),
    }


vc = variance_components(df, PRIMARY_WINDOW, neff_ar1, rho_hat)
print("Variance decomposition (primary window):")
print(f"  observed_var = {vc['observed_var']:.5f}")
print(f"  noise_var    = {vc['noise_var']:.5f}")
print(f"  signal_var   = {vc['signal_var']:.5f}")
print(f"  global w     = {vc['global_w']:.3f}")
print(f"  pergame_noise = {vc['pergame_noise']:.5f}")
if vc["signal_var"] <= 0:
    print(
        "  WARNING: signal variance is not positive. The weight is undefined. Inspect n_eff and inputs."
    )
if SMOKE_TEST:
    print("  Smoke test: signal_var should be close to signal_sd squared, which is 0.25.")

fig, ax = plt.subplots(figsize=(7, 3))
ax.barh([0], [max(vc["signal_var"], 0)], color="#1D9E75", label="signal")
ax.barh(
    [0], [max(vc["noise_var"], 0)], left=[max(vc["signal_var"], 0)], color="#888780", label="noise"
)
ax.set_yticks([])
ax.set_xlabel("Variance of rolling form")
ax.set_title("Variance decomposition: observed = signal + noise")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "5_variance_decomposition.png", dpi=130)
plt.close(fig)

# %%
# ----------------------------------------------------------------------
# Section 6: Per-team reliability weights
# Tests C6.
# ----------------------------------------------------------------------


def add_reliability_weight(
    frame: pd.DataFrame, window: int, signal_var: float, n_eff: float
) -> pd.DataFrame:
    """
    Per team-game reliability weight. Signal variance is the global league
    quantity. Noise variance is this team's own rolling variance divided by
    the effective sample size. The weight rises toward 1 for consistent teams
    and falls toward 0 for erratic teams.
    """
    out = frame.copy()
    noise_var = out[f"form_var_{window}"] / n_eff
    out[f"w_{window}"] = signal_var / (signal_var + noise_var)
    return out


df = add_reliability_weight(df, PRIMARY_WINDOW, vc["signal_var"], neff_ar1)

w_col = f"w_{PRIMARY_WINDOW}"
w_valid = df[w_col].dropna()
print("Reliability weight distribution:")
print(f"  min={w_valid.min():.3f}  median={w_valid.median():.3f}  max={w_valid.max():.3f}")
print(f"  fraction below 0.5: {(w_valid < 0.5).mean():.3f}")

team_sd = np.sqrt(df[f"form_var_{PRIMARY_WINDOW}"])
corr_w_sd = pd.Series(df[w_col]).corr(team_sd)
print(f"  correlation of w with rolling SD (should be strongly negative): {corr_w_sd:+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(w_valid, bins=50, color="#EF9F27", alpha=0.85)
axes[0].set_title("Reliability weight distribution")
axes[0].set_xlabel("w")
axes[0].set_ylabel("Count of team-games")
sample = (
    df[[w_col, f"form_var_{PRIMARY_WINDOW}"]]
    .dropna()
    .sample(min(4000, df[[w_col]].dropna().shape[0]), random_state=1)
)
axes[1].scatter(
    np.sqrt(sample[f"form_var_{PRIMARY_WINDOW}"]), sample[w_col], s=4, alpha=0.3, color="#888780"
)
axes[1].set_title("Weight falls as recent form gets noisier")
axes[1].set_xlabel("Rolling standard deviation of form")
axes[1].set_ylabel("w")
fig.tight_layout()
fig.savefig(FIG_DIR / "6_reliability_weights.png", dpi=130)
plt.close(fig)

# %%
# ----------------------------------------------------------------------
# Section 7: Shrunk form versus raw form
# Tests C7.
# ----------------------------------------------------------------------

df[f"shrunk_form_{PRIMARY_WINDOW}"] = df[w_col] * df[form_col]

raw = df[form_col].dropna()
shr = df[f"shrunk_form_{PRIMARY_WINDOW}"].dropna()
print("Raw versus shrunk form:")
print(f"  raw    sd = {raw.std(ddof=0):.4f}")
print(f"  shrunk sd = {shr.std(ddof=0):.4f}  (should be smaller, form pulled toward zero)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(raw, bins=60, color="#4C78A8", alpha=0.5, label="raw rolling form")
ax.hist(shr, bins=60, color="#534AB7", alpha=0.5, label="shrunk form")
ax.set_title("Shrinkage pulls noisy form toward zero")
ax.set_xlabel("Form value")
ax.set_ylabel("Count of team-games")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "7_raw_vs_shrunk_form.png", dpi=130)
plt.close(fig)

# %%
# ----------------------------------------------------------------------
# Section 8: Predictive window sweep, raw versus shrunk
# Tests C8.
# Variance components and effective sample size are estimated on training
# seasons only inside each fold, which keeps window and shrinkage selection
# leakage-free.
# ----------------------------------------------------------------------


def fit_predict_form(train: pd.DataFrame, test: pd.DataFrame, feature: str, target: str) -> float:
    """
    Fit a simple one-feature linear regression of target on feature using
    training rows, predict the test rows, and return the test mean absolute
    error. A one-feature fit isolates the predictive content of the form
    feature itself.
    """
    tr = train[[feature, target]].dropna()
    te = test[[feature, target]].dropna()
    if len(tr) < 50 or len(te) < 10:
        return float("nan")
    x = tr[feature].to_numpy()
    y = tr[target].to_numpy()
    b = np.polyfit(x, y, 1)
    pred = np.polyval(b, te[feature].to_numpy())
    return float(np.mean(np.abs(te[target].to_numpy() - pred)))


def sweep_window(base: pd.DataFrame, window: int) -> dict:
    """
    Expanding-window season splits. For each test season, estimate the
    variance components and effective sample size on prior seasons only,
    build raw and shrunk form, fit on prior seasons, and score the test
    season. Returns mean out-of-sample MAE for raw and shrunk features.
    """
    work = add_rolling_form(base, window, MIN_PERIODS)
    target = COL_MARGIN if HAVE_MARGIN else "composite_net"
    seasons = sorted(work[COL_SEASON].unique(), key=season_start_year)
    raw_errs = []
    shr_errs = []
    for i in range(2, len(seasons)):
        train_seasons = set(seasons[:i])
        test_season = seasons[i]
        train = work[work[COL_SEASON].isin(train_seasons)]
        test = work[work[COL_SEASON] == test_season]

        neff_fold = []
        for (_t, _s), g in train.groupby([COL_TEAM, COL_SEASON]):
            series = g["composite_net"].to_numpy()
            if np.sum(~np.isnan(series)) >= window:
                neff_fold.append(effective_sample_size_ar1(series, window))
        nf = float(np.nanmean(neff_fold)) if neff_fold else float(window)

        form = train[f"form_mean_{window}"]
        rvar = train[f"form_var_{window}"]
        observed_var = float(form.var(ddof=0))
        noise_var = float((rvar / nf).mean())
        signal_var = max(observed_var - noise_var, 1e-9)

        train = train.copy()
        test = test.copy()
        for frame in (train, test):
            nvar = frame[f"form_var_{window}"] / nf
            frame[f"w_{window}"] = signal_var / (signal_var + nvar)
            frame[f"shrunk_form_{window}"] = frame[f"w_{window}"] * frame[f"form_mean_{window}"]

        raw_errs.append(fit_predict_form(train, test, f"form_mean_{window}", target))
        shr_errs.append(fit_predict_form(train, test, f"shrunk_form_{window}", target))

    return {
        "window": window,
        "raw_mae": float(np.nanmean(raw_errs)),
        "shrunk_mae": float(np.nanmean(shr_errs)),
    }


base_df = build_composite(
    standardize_factors(
        df[
            [COL_SEASON, COL_GAME_DATE, COL_TEAM, *FACTOR_COLS]
            + ([COL_MARGIN] if HAVE_MARGIN else [])
        ].copy()
    )
)

results = [sweep_window(base_df, w) for w in WINDOW_CANDIDATES]
res_df = pd.DataFrame(results)
res_df["shrink_improvement"] = res_df["raw_mae"] - res_df["shrunk_mae"]

print("Window sweep (lower MAE is better):")
print(res_df.to_string(index=False))

best_raw = res_df.loc[res_df["raw_mae"].idxmin(), "window"]
best_shrunk = res_df.loc[res_df["shrunk_mae"].idxmin(), "window"]
print(f"  Best window by raw form:    {best_raw}")
print(f"  Best window by shrunk form: {best_shrunk}")
print("  Positive shrink_improvement means shrinkage reduced out-of-sample error.")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(res_df["window"], res_df["raw_mae"], marker="o", color="#4C78A8", label="raw form")
ax.plot(res_df["window"], res_df["shrunk_mae"], marker="o", color="#534AB7", label="shrunk form")
ax.set_title("Out-of-sample error by window: raw versus shrunk form")
ax.set_xlabel("Window size (games)")
ax.set_ylabel("Mean absolute error")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "8_window_sweep.png", dpi=130)
plt.close(fig)

# %%
print("Done. Figures written to", FIG_DIR)
print("If the smoke test passed, set SMOKE_TEST = False and rerun on real data.")


# %%
# ----------------------------------------------------------------------
# Section 7b: Direct reliability via regression to the mean
# The slope of current performance on trailing form equals the global
# reliability weight. It uses no autocorrelation parameter and no effective
# sample size, so it does not inherit the short-panel AR(1) bias. The target
# must be composite_net, not margin, so predictor and target share units and
# the slope is a dimensionless reliability in [0, 1].
# ----------------------------------------------------------------------


def direct_global_reliability(frame: pd.DataFrame, window: int) -> dict:
    """
    Regress current-game composite NET on the trailing form mean. The trailing
    window excludes the current game, so the current game's noise is
    independent of the predictor, and the slope estimates signal variance over
    observed variance, which is the global reliability weight.
    """
    cols = frame[[f"form_mean_{window}", "composite_net"]].dropna()
    x = cols[f"form_mean_{window}"].to_numpy()
    y = cols["composite_net"].to_numpy()
    if x.size < 100:
        return {"slope": float("nan"), "n": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "n": int(x.size)}


direct = direct_global_reliability(df, PRIMARY_WINDOW)
signal_var_direct = direct["slope"] * vc["observed_var"]

print("Direct reliability via regression to the mean:")
print(f"  direct global reliability (slope): {direct['slope']:.3f}")
print(f"  decomposition global w:            {vc['global_w']:.3f}")
print(f"  implied signal variance (direct):  {signal_var_direct:.5f}")
print(f"  decomposition signal variance:     {vc['signal_var']:.5f}")
if SMOKE_TEST:
    print("  Smoke test: direct slope should land near 0.53, implied signal near 0.236.")
    print("  If decomposition w sits well above the slope, the direct estimate is less biased.")

df[f"w_direct_{PRIMARY_WINDOW}"] = signal_var_direct / (
    signal_var_direct + df[f"form_var_{PRIMARY_WINDOW}"] / neff_ar1
)
df[f"shrunk_form_direct_{PRIMARY_WINDOW}"] = df[f"w_direct_{PRIMARY_WINDOW}"] * df[form_col]

w_dir = df[f"w_direct_{PRIMARY_WINDOW}"].dropna()
corr_dir = pd.Series(df[f"w_direct_{PRIMARY_WINDOW}"]).corr(
    np.sqrt(df[f"form_var_{PRIMARY_WINDOW}"])
)
print("Recalibrated weight (direct level, decomposition shape):")
print(f"  median={w_dir.median():.3f}  (decomposition median was {df[w_col].median():.3f})")
print(f"  correlation with rolling SD (shape preserved): {corr_dir:+.3f}")
