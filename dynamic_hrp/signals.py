# -------------------------------------------------------------
# Time-Series Momentum (TSMOM) Signal Generators
# -------------------------------------------------------------
# Implements two common TSMOM signal construction methods:
#   1) Moving-average crossover style (price vs. MA)
#   2) Standardized past return style (return / volatility)
# Also includes an EMA-based smoother for noisy binary signals.
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# 1. Moving-Average Based TSMOM Signal
# -------------------------------------------------------------
def tsmom_signal_ma(prices_weekly: pd.DataFrame, lookbacks: list[int] = [13, 26, 52]):
    """
    Generate time-series momentum (TSMOM) signals using the sign of (price - moving average)
    across multiple lookback windows.

    Logic:
        For each lookback L:
          signal_t = sign(Price_t - MA_L_t)

    This produces a +1 when the price is above its moving average (uptrend),
    -1 when below (downtrend), and 0 when equal or undefined.

    Parameters
    ----------
    prices_weekly : pd.DataFrame
        Weekly price data (assets as columns).
    lookbacks : list[int]
        List of lookback lengths (in weeks) for moving averages.

    Returns
    -------
    tuple(pd.DataFrame, dict)
        (combined_signal, {L: signal_df_per_L})
        combined_signal = average of all signals across lookbacks.
    """
    # Coerce all entries to numeric and copy
    px = prices_weekly.apply(pd.to_numeric, errors="coerce")
    signals_dict = {}

    for L in lookbacks:
        # Rolling moving average (min_periods=L/2 allows shorter warm-up)
        ma = px.rolling(L, min_periods=L//2).mean()
        # Binary signal: +1 if price > MA, -1 if price < MA
        s = np.sign(px - ma)
        s.name = f"MA_{L}"
        signals_dict[L] = s

    # Stack along hierarchical index (lookback, date), then average over lookbacks
    signals = pd.concat(signals_dict.values(), axis=0, keys=signals_dict.keys())
    signals = signals.groupby(level=1).mean()
    signals.index = px.index  # ensure index matches original prices

    return signals, signals_dict


# -------------------------------------------------------------
# 2. Return/Volatility Based TSMOM Signal
# -------------------------------------------------------------
def tsmom_signal_return(prices_weekly: pd.DataFrame, lookbacks: list[int] = [13, 26, 52]):
    """
    Generate TSMOM signals using standardized past returns.

    Logic:
        For each lookback L:
            momentum = (P_t / P_{t-L}) - 1
            vol      = rolling_std(log_returns, L)
            zscore   = momentum / vol
            signal_t = sign(zscore)

    This method identifies trend direction by comparing past cumulative returns
    relative to their volatility.

    Parameters
    ----------
    prices_weekly : pd.DataFrame
        Weekly price data.
    lookbacks : list[int]
        Lookback periods (in weeks) for momentum and volatility.

    Returns
    -------
    tuple(pd.DataFrame, dict)
        (combined_signal, {L: signal_df_per_L})
        combined_signal = mean of individual signals across lookbacks.
    """
    # Compute log returns (used for volatility estimation)
    r = np.log(prices_weekly / prices_weekly.shift(1))
    signals_dict = {}

    for L in lookbacks:
        # Momentum as cumulative % return over L weeks
        mom = prices_weekly / prices_weekly.shift(L) - 1
        # Volatility proxy: rolling standard deviation of log returns
        vol = r.rolling(L).std()
        # Z-score = normalized momentum by volatility
        zscore = mom / vol
        s = np.sign(zscore)   # sign of z-score = directional signal
        s.name = f"RET_{L}"
        signals_dict[L] = s

    # Combine signals by averaging across lookbacks
    signals = pd.concat(signals_dict.values(), axis=0, keys=signals_dict.keys())
    signals = signals.groupby(level=1).mean()
    signals.index = prices_weekly.index

    return signals, signals_dict


# -------------------------------------------------------------
# 3. Exponential Moving Average (EMA) Smoothing
# -------------------------------------------------------------
def smooth_signals(signals: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
    """
    Apply exponential moving average (EMA) smoothing to discrete +/-1 signals.

    Useful for:
      • Reducing frequent sign flips in noisy signals
      • Creating continuous transition weights between -1 and +1

    Parameters
    ----------
    signals : pd.DataFrame
        Discrete or noisy signal DataFrame (same structure as prices).
    alpha : float
        EMA smoothing parameter (0 < alpha ≤ 1).
        Smaller alpha → smoother signal.

    Returns
    -------
    pd.DataFrame
        Smoothed signals (continuous values between -1 and +1).
    """
    return signals.ewm(alpha=alpha).mean()
