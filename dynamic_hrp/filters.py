# dynamic_hrp/filters.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Union

def apply_cusum_filter(
    series: pd.Series,
    threshold: float,
    start_date: Optional[Union[str, pd.Timestamp]] = None
) -> pd.DatetimeIndex:
    """
    Applies the CUSUM filter to a series (typically daily returns) to sample events
    where the cumulative price change (deviation) exceeds a threshold.

    Parameters
    ----------
    series : pd.Series
        The input series (e.g., daily Equal-Weighted log returns).
    threshold : float
        The minimum deviation magnitude (e.g., 0.005 for 0.5% change)
        required to flag an event.
    start_date : Optional[Union[str, pd.Timestamp]], optional
        Date to start the filtering process.

    Returns
    -------
    pd.DatetimeIndex
        An index of dates where CUSUM events occurred.
    """
    if start_date:
        series = series.loc[series.index >= start_date]

    t_events = []
    h = threshold
    s_pos = 0  # Cumulative sum for positive deviations
    s_neg = 0  # Cumulative sum for negative deviations

    # Drop NaNs and ensure index is sorted
    s = series.dropna().sort_index()
    if s.empty:
        return pd.DatetimeIndex([])

    # Iterate through the returns
    for t_idx, r_t in s.items():
        # Update positive cumulative sum
        s_pos = max(0, s_pos + r_t)
        
        # Update negative cumulative sum (r_t is subtracted if negative)
        s_neg = min(0, s_neg + r_t) 

        # Check for positive event (upward structural shift)
        if s_pos > h:
            t_events.append(t_idx)
            s_pos = 0  # Reset
            s_neg = 0  # Reset
        # Check for negative event (downward structural shift)
        elif s_neg < -h:
            t_events.append(t_idx)
            s_pos = 0  # Reset
            s_neg = 0  # Reset

    return pd.DatetimeIndex(t_events)