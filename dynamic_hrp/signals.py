from __future__ import annotations
import numpy as np
import pandas as pd

def tsmom_signal_ma(prices_weekly: pd.DataFrame, lookbacks: list[int] = [13, 26, 52]):
    """TSMOM via sign(price - moving average) across multiple lookbacks; returns (combined, dict)."""
    px = prices_weekly.apply(pd.to_numeric, errors="coerce")
    signals_dict = {}
    for L in lookbacks:
        ma = px.rolling(L, min_periods=L//2).mean()
        s = np.sign(px - ma)
        s.name = f"MA_{L}"
        signals_dict[L] = s
    signals = pd.concat(signals_dict.values(), axis=0, keys=signals_dict.keys())
    signals = signals.groupby(level=1).mean()
    signals.index = px.index
    return signals, signals_dict

def tsmom_signal_return(prices_weekly: pd.DataFrame, lookbacks: list[int] = [13, 26, 52]):
    """TSMOM via standardized past return sign across lookbacks; returns (combined, dict)."""
    r = np.log(prices_weekly / prices_weekly.shift(1))
    signals_dict = {}
    for L in lookbacks:
        mom = prices_weekly / prices_weekly.shift(L) - 1
        vol = r.rolling(L).std()
        zscore = mom / vol
        s = np.sign(zscore)
        s.name = f"RET_{L}"
        signals_dict[L] = s
    signals = pd.concat(signals_dict.values(), axis=0, keys=signals_dict.keys())
    signals = signals.groupby(level=1).mean()
    signals.index = prices_weekly.index
    return signals, signals_dict

def smooth_signals(signals: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
    """EMA-smooth discrete +/-1 signals."""
    return signals.ewm(alpha=alpha).mean()
