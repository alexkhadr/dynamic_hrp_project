from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def _ann_factor(freq: str = "W") -> float:
    if freq.upper().startswith("D"): return np.sqrt(252)
    if freq.upper().startswith("W"): return np.sqrt(52)
    if freq.upper().startswith("M"): return np.sqrt(12)
    return np.sqrt(52)

def _pairwise_avg_offdiag_corr(ret_window: pd.DataFrame) -> float:
    X = ret_window.dropna(how="all", axis=1)
    if X.shape[1] < 2: return np.nan
    C = X.corr()
    if C.shape[0] < 2: return np.nan
    n = C.shape[0]
    offdiag = C.values[~np.eye(n, dtype=bool)]
    return float(np.nanmean(offdiag))

def _rolling_apply_dates(df: pd.DataFrame, window: int, func) -> pd.Series:
    idx = df.index
    out = pd.Series(index=idx, dtype=float)
    for i in range(window - 1, len(idx)):
        w = df.iloc[i - window + 1 : i + 1]
        out.iloc[i] = func(w)
    return out

def _expanding_zscore(df: pd.DataFrame, min_periods: int = 20) -> pd.DataFrame:
    mu = df.expanding(min_periods=min_periods).mean().shift(1)
    sd = df.expanding(min_periods=min_periods).std(ddof=1).shift(1)
    return (df - mu) / sd

def build_hmm_features(
    weekly_returns: pd.DataFrame,
    weekly_signals: pd.DataFrame,
    vix_daily: pd.Series | pd.DataFrame | None = None,
    window_weeks: int = 26,
    freq: str = "W",
) -> pd.DataFrame:
    """
    Build regime features for each week t using a rolling window of weekly returns/signals.

    Outputs columns:
      - vol_mean, vol_p75
      - corr_mean_offdiag
      - mom_dispersion
      - vix_level (optional), dvix (optional)
      - ew_skew, ew_kurt
    """
    R = weekly_returns.copy()
    S = weekly_signals.reindex(R.index).copy()

    ann = _ann_factor(freq)
    vol_panel = R.rolling(window_weeks).std(ddof=1) * ann
    vol_mean = vol_panel.mean(axis=1)
    vol_p75  = vol_panel.quantile(0.75, axis=1)

    corr_mean_offdiag = _rolling_apply_dates(R, window_weeks, _pairwise_avg_offdiag_corr)
    mom_dispersion = S.std(axis=1)

    vix_level = dvix = None
    if vix_daily is not None:
        if isinstance(vix_daily, pd.DataFrame):
            vix_series = vix_daily["VIX"] if "VIX" in vix_daily.columns else vix_daily.iloc[:, 0]
        else:
            vix_series = vix_daily
        vix_series = pd.to_numeric(vix_series, errors="coerce")
        vix_w = vix_series.resample("W-FRI").last().reindex(R.index)
        vix_level = vix_w
        dvix = vix_w.diff(1)

    ew_ret = R.mean(axis=1)
    ew_skew = ew_ret.rolling(window_weeks).apply(lambda x: skew(x, bias=False), raw=False)
    ew_kurt = ew_ret.rolling(window_weeks).apply(lambda x: kurtosis(x, fisher=True, bias=False), raw=False)

    feats = pd.DataFrame({
        "vol_mean": vol_mean,
        "vol_p75": vol_p75,
        "corr_mean_offdiag": corr_mean_offdiag,
        "mom_dispersion": mom_dispersion,
        "ew_skew": ew_skew,
        "ew_kurt": ew_kurt,
    }, index=R.index)

    if vix_level is not None: feats["vix_level"] = vix_level
    if dvix is not None: feats["dvix"] = dvix

    return feats.dropna(how="all")

def standardize_features_expanding(
    features: pd.DataFrame,
    min_periods: int = 26
) -> pd.DataFrame:
    """Expanding z-scores with a 1-step lag (no look-ahead)."""
    return _expanding_zscore(features, min_periods=min_periods)

def build_and_standardize_hmm_features(
    weekly_returns: pd.DataFrame,
    weekly_signals: pd.DataFrame,
    vix_daily: pd.Series | pd.DataFrame | None = None,
    window_weeks: int = 26,
    freq: str = "W",
    min_periods_std: int = 26
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features_raw, features_std)."""
    raw = build_hmm_features(weekly_returns, weekly_signals, vix_daily, window_weeks, freq)
    std = standardize_features_expanding(raw, min_periods=min_periods_std)
    return raw, std
