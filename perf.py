# -------------------------------------------------------------
# Performance Statistics Utility
# -------------------------------------------------------------
# Computes key portfolio performance metrics from a PnL (return) series,
# including higher moments (Skew, Kurtosis) needed for DSR.
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
# Import required statistical functions
from scipy.stats import skew, kurtosis 

def perf_stats(pnl: pd.Series, freq: str = "W") -> pd.Series:
    """
    Compute key performance statistics from a PnL (return) series.

    Parameters
    ----------
    pnl : pd.Series
        Series of periodic returns (weekly by default).
    freq : str, default "W"
        Frequency of returns — "W" for weekly, "D" for daily.
        Determines annualization factor for Sharpe and CAGR.

    Returns
    -------
    pd.Series
        Performance metrics including: Mean, Vol, Sharpe, CAGR, MaxDD, 
        Skew, and Kurtosis.
    """
    # Drop missing returns
    r = pnl.dropna()
    N_obs = len(r)

    # --- Basic summary statistics ---
    mean = r.mean()         # average periodic return
    vol = r.std()           # standard deviation of returns

    # Annualization factor: 52 weeks/year or 252 days/year
    ann_factor_sq = 52 if freq == "W" else 252
    ann_factor = np.sqrt(ann_factor_sq)

    # Annualized Sharpe ratio (assuming zero risk-free rate)
    sharpe = (mean / vol) * ann_factor if vol != 0 else np.nan
    
    # Non-normality moments (used for DSR)
    # Note: bias=False for unbiased estimators; fisher=True for excess kurtosis
    skewness = skew(r.values, bias=False) if N_obs >= 8 else np.nan 
    kurt = kurtosis(r.values, fisher=True, bias=False) if N_obs >= 8 else np.nan

    # Compound Annual Growth Rate (CAGR)
    cagr = (1 + r).prod()**(ann_factor_sq/N_obs) - 1 if N_obs > 0 else 0.0

    # --- Drawdown analysis ---
    cum = (1 + r).cumprod()      # cumulative growth of $1
    dd = 1 - cum / cum.cummax()  # drawdown series (as fraction)
    maxdd = dd.max() if not dd.empty else 0.0  # maximum drawdown

    # Return all metrics as a labeled Series
    return pd.Series({
        "Mean": mean,
        "Vol": vol,
        "Sharpe": sharpe,
        "CAGR": cagr,
        "MaxDD": maxdd,
        "Skew": skewness,
        "Kurtosis": kurt,
        "N_obs": N_obs # Number of observations
    })