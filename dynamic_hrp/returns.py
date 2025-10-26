# -------------------------------------------------------------
# Return Calculations and Weekly Alignment Utilities
# -------------------------------------------------------------
# This module provides small helper functions for:
#   • Cleaning and validating price data
#   • Computing daily and weekly log returns
#   • Generating weekly end-of-week (e.g., Friday) prices
#   • Shifting returns to account for delayed execution
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# Helper: Sanitize price data
# -------------------------------------------------------------
def _sanitize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure price data is numeric and nonnegative.
      - Converts all entries to numeric (invalid → NaN)
      - Replaces nonpositive values (<=0) with NaN
    This helps prevent invalid log-return computations.
    """
    p = prices.apply(pd.to_numeric, errors="coerce").copy()
    p[p <= 0] = np.nan
    return p

# -------------------------------------------------------------
# Weekly price aggregation
# -------------------------------------------------------------
def weekly_last_from_daily(prices_daily: pd.DataFrame, week_day: str = "FRI") -> pd.DataFrame:
    """
    Convert daily price data to weekly prices by taking the last
    available observation each week (default: Friday).
    Uses pandas resample with rule 'W-{week_day}'.
    """
    return prices_daily.resample(f"W-{week_day}").last().dropna(how="all")

# -------------------------------------------------------------
# Daily log returns
# -------------------------------------------------------------
def daily_log_returns(prices_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns:
        r_t = ln(P_t / P_{t-1})
    where P_t are sanitized (numeric and >0) prices.
    """
    p = _sanitize_prices(prices_daily)
    r = np.log(p / p.shift(1))
    return r.dropna(how="all")

# -------------------------------------------------------------
# Weekly log returns derived from daily prices
# -------------------------------------------------------------
def weekly_log_returns_from_daily(prices_daily: pd.DataFrame, week_day: str = "FRI"):
    """
    Compute weekly prices and log returns from daily prices.
      1) Resample to weekly (e.g., W-FRI)
      2) Take log differences between consecutive weeks
    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame)
        (weekly_prices, weekly_log_returns)
    """
    p = _sanitize_prices(prices_daily)
    p_w = weekly_last_from_daily(p, week_day=week_day)
    r_w = np.log(p_w / p_w.shift(1))
    return p_w.dropna(how="all"), r_w.dropna(how="all")

# -------------------------------------------------------------
# Execution delay adjustment
# -------------------------------------------------------------
def apply_weekly_execution_delay(
    weekly_weights_dates: pd.Index,
    weekly_returns: pd.DataFrame,
    delay_weeks: int = 1,
) -> pd.DataFrame:
    """
    Apply an execution delay to weekly returns.

    This ensures that portfolio weights chosen at week t are
    applied to returns starting at week t + delay_weeks.

    Example:
        delay_weeks = 1 → next-week execution
        delay_weeks = 0 → immediate execution

    Parameters
    ----------
    weekly_weights_dates : pd.Index
        Index of dates on which portfolio weights exist.
    weekly_returns : pd.DataFrame
        Weekly return matrix aligned on weekly dates.
    delay_weeks : int
        Number of weeks to shift returns upward.

    Returns
    -------
    pd.DataFrame
        Shifted weekly returns aligned to weight dates.
    """
    if delay_weeks <= 0:
        return weekly_returns.reindex(weekly_weights_dates)
    r = weekly_returns.shift(-delay_weeks)
    return r.reindex(weekly_weights_dates)
