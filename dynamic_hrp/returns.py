from __future__ import annotations
import numpy as np
import pandas as pd

def _sanitize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.apply(pd.to_numeric, errors="coerce").copy()
    p[p <= 0] = np.nan
    return p

def weekly_last_from_daily(prices_daily: pd.DataFrame, week_day: str = "FRI") -> pd.DataFrame:
    return prices_daily.resample(f"W-{week_day}").last().dropna(how="all")

def daily_log_returns(prices_daily: pd.DataFrame) -> pd.DataFrame:
    p = _sanitize_prices(prices_daily)
    r = np.log(p / p.shift(1))
    return r.dropna(how="all")

def weekly_log_returns_from_daily(prices_daily: pd.DataFrame, week_day: str = "FRI"):
    p = _sanitize_prices(prices_daily)
    p_w = weekly_last_from_daily(p, week_day=week_day)
    r_w = np.log(p_w / p_w.shift(1))
    return p_w.dropna(how="all"), r_w.dropna(how="all")

def apply_weekly_execution_delay(
    weekly_weights_dates: pd.Index,
    weekly_returns: pd.DataFrame,
    delay_weeks: int = 1,
) -> pd.DataFrame:
    """
    Shift weekly returns up by delay_weeks so weights chosen at t earn from t+delay.
    """
    if delay_weeks <= 0:
        return weekly_returns.reindex(weekly_weights_dates)
    r = weekly_returns.shift(-delay_weeks)
    return r.reindex(weekly_weights_dates)
