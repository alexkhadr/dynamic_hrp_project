# -------------------------------------------------------------
# Performance Statistics Utility
# -------------------------------------------------------------
# Computes key portfolio performance metrics from a PnL (return) series:
#   • Mean weekly/daily return
#   • Volatility
#   • Sharpe ratio (annualized)
#   • CAGR (compound annual growth rate)
#   • Max Drawdown (worst peak-to-trough loss)
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd

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
        Performance metrics including:
          - Mean   : average periodic return
          - Vol    : standard deviation of returns
          - Sharpe : annualized Sharpe ratio
          - CAGR   : compound annual growth rate
          - MaxDD  : maximum drawdown
    """
    # Drop missing returns
    r = pnl.dropna()

    # --- Basic summary statistics ---
    mean = r.mean()       # average periodic return
    vol = r.std()         # standard deviation of returns

    # Annualization factor: 52 weeks/year or 252 days/year
    ann_factor = np.sqrt(52) if freq == "W" else np.sqrt(252)

    # Annualized Sharpe ratio (assuming zero risk-free rate)
    sharpe = (mean / vol) * ann_factor if vol != 0 else np.nan

    # Compound Annual Growth Rate (CAGR)
    # (1 + total return)^(annual periods / total obs) - 1
    cagr = (1 + r).prod()**(52/len(r)) - 1 if len(r) > 0 else 0.0

    # --- Drawdown analysis ---
    cum = (1 + r).cumprod()          # cumulative growth of $1
    dd = 1 - cum / cum.cummax()      # drawdown series (as fraction)
    maxdd = dd.max() if not dd.empty else 0.0  # maximum drawdown

    # Return all metrics as a labeled Series
    return pd.Series({
        "Mean": mean,
        "Vol": vol,
        "Sharpe": sharpe,
        "CAGR": cagr,
        "MaxDD": maxdd
    })

