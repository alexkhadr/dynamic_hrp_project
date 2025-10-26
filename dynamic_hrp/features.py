# This module builds regime-identification features for Hidden Markov Models (HMM)
# from weekly returns and signals. It includes volatility, correlations, momentum dispersion,
# skew/kurtosis, and optionally VIX-related features, with standardized versions (expanding z-scores).

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# -------------------------------------------------------
# Helper: Annualization factor for volatility scaling
# -------------------------------------------------------
def _ann_factor(freq: str = "W") -> float:
    # Convert frequency to upper case and return sqrt of periods per year
    if freq.upper().startswith("D"): return np.sqrt(252)   # daily → √252
    if freq.upper().startswith("W"): return np.sqrt(52)    # weekly → √52
    if freq.upper().startswith("M"): return np.sqrt(12)    # monthly → √12
    return np.sqrt(52)                                     # default: weekly

# -------------------------------------------------------
# Helper: Average off-diagonal correlation among assets
# -------------------------------------------------------
def _pairwise_avg_offdiag_corr(ret_window: pd.DataFrame) -> float:
    # Drop all-NaN columns to avoid invalid correlations
    X = ret_window.dropna(how="all", axis=1)
    if X.shape[1] < 2: return np.nan   # Need at least two assets
    C = X.corr()                       # Compute correlation matrix
    if C.shape[0] < 2: return np.nan
    n = C.shape[0]
    # Extract all off-diagonal elements (i.e., pairwise correlations)
    offdiag = C.values[~np.eye(n, dtype=bool)]
    # Return mean of off-diagonal correlations
    return float(np.nanmean(offdiag))

# -------------------------------------------------------
# Helper: Rolling apply using custom function on DataFrame windows
# -------------------------------------------------------
def _rolling_apply_dates(df: pd.DataFrame, window: int, func) -> pd.Series:
    idx = df.index
    out = pd.Series(index=idx, dtype=float)
    # Iterate through all rolling windows and apply user-defined function
    for i in range(window - 1, len(idx)):
        w = df.iloc[i - window + 1 : i + 1]   # subset of size = window
        out.iloc[i] = func(w)
    return out

# -------------------------------------------------------
# Helper: Expanding z-score normalization (no look-ahead)
# -------------------------------------------------------
def _expanding_zscore(df: pd.DataFrame, min_periods: int = 20) -> pd.DataFrame:
    # Compute expanding mean and std with at least `min_periods` of history
    # Shift(1) ensures we use only past information (no future leakage)
    mu = df.expanding(min_periods=min_periods).mean().shift(1)
    sd = df.expanding(min_periods=min_periods).std(ddof=1).shift(1)
    # Standardize each feature as (x - mean) / std
    return (df - mu) / sd

# -------------------------------------------------------
# Core: Build raw HMM features
# -------------------------------------------------------
def build_hmm_features(
    weekly_returns: pd.DataFrame,         # weekly returns for each asset
    weekly_signals: pd.DataFrame,         # standardized signals for each asset
    vix_daily: pd.Series | pd.DataFrame | None = None,  # optional daily VIX
    window_weeks: int = 26,               # rolling window size (half-year)
    freq: str = "W",                      # sampling frequency
) -> pd.DataFrame:
    """
    Build regime features for each week t using a rolling window of weekly returns/signals.

    Outputs columns:
      - vol_mean, vol_p75            : mean and 75th percentile of rolling volatility
      - corr_mean_offdiag            : mean pairwise correlation among assets
      - mom_dispersion               : cross-sectional std of momentum signals
      - ew_skew, ew_kurt             : rolling skewness and kurtosis of mean portfolio returns
      - vix_level, dvix (optional)   : current VIX and its weekly change
    """
    # Copy data to avoid modifying inputs
    R = weekly_returns.copy()
    S = weekly_signals.reindex(R.index).copy()

    # Compute annualized volatility features
    ann = _ann_factor(freq)
    vol_panel = R.rolling(window_weeks).std(ddof=1) * ann   # rolling vol per asset
    vol_mean = vol_panel.mean(axis=1)                       # mean vol across assets
    vol_p75  = vol_panel.quantile(0.75, axis=1)             # 75th percentile vol (upper tail)

    # Compute mean pairwise correlation across assets
    corr_mean_offdiag = _rolling_apply_dates(R, window_weeks, _pairwise_avg_offdiag_corr)

    # Momentum dispersion: how diverse signals are across assets at each time
    mom_dispersion = S.std(axis=1)

    # Optional: include VIX level and weekly change
    vix_level = dvix = None
    if vix_daily is not None:
        # Handle VIX input as either DataFrame (column 'VIX') or Series
        if isinstance(vix_daily, pd.DataFrame):
            vix_series = vix_daily["VIX"] if "VIX" in vix_daily.columns else vix_daily.iloc[:, 0]
        else:
            vix_series = vix_daily

        # Ensure numeric conversion and resample to weekly frequency
        vix_series = pd.to_numeric(vix_series, errors="coerce")
        vix_w = vix_series.resample("W-FRI").last().reindex(R.index)
        vix_level = vix_w
        dvix = vix_w.diff(1)  # 1-week change in VIX

    # Equal-weighted (EW) portfolio returns: mean return across assets
    ew_ret = R.mean(axis=1)

    # Rolling higher-moment features (distribution shape)
    ew_skew = ew_ret.rolling(window_weeks).apply(lambda x: skew(x, bias=False), raw=False)
    ew_kurt = ew_ret.rolling(window_weeks).apply(lambda x: kurtosis(x, fisher=True, bias=False), raw=False)

    # Combine all features into a single DataFrame
    feats = pd.DataFrame({
        "vol_mean": vol_mean,
        "vol_p75": vol_p75,
        "corr_mean_offdiag": corr_mean_offdiag,
        "mom_dispersion": mom_dispersion,
        "ew_skew": ew_skew,
        "ew_kurt": ew_kurt,
    }, index=R.index)

    # Add optional VIX features if available
    if vix_level is not None: feats["vix_level"] = vix_level
    if dvix is not None: feats["dvix"] = dvix

    # Drop rows that are entirely NaN (early periods)
    return feats.dropna(how="all")

# -------------------------------------------------------
# Core: Standardize features (expanding z-scores)
# -------------------------------------------------------
def standardize_features_expanding(
    features: pd.DataFrame,
    min_periods: int = 26
) -> pd.DataFrame:
    """Expanding z-scores with a 1-step lag (no look-ahead)."""
    return _expanding_zscore(features, min_periods=min_periods)

# -------------------------------------------------------
# Wrapper: Build + standardize features together
# -------------------------------------------------------
def build_and_standardize_hmm_features(
    weekly_returns: pd.DataFrame,
    weekly_signals: pd.DataFrame,
    vix_daily: pd.Series | pd.DataFrame | None = None,
    window_weeks: int = 26,
    freq: str = "W",
    min_periods_std: int = 26
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline to:
      1) Build raw rolling HMM features from returns/signals (+ optional VIX).
      2) Standardize them using expanding z-scores (lagged for no look-ahead).
    Returns:
      (features_raw, features_std)
    """
    raw = build_hmm_features(weekly_returns, weekly_signals, vix_daily, window_weeks, freq)
    std = standardize_features_expanding(raw, min_periods=min_periods_std)
    return raw, std