from __future__ import annotations
import numpy as np
import pandas as pd

def perf_stats(pnl: pd.Series, freq: str = "W") -> pd.Series:
    r = pnl.dropna()
    mean = r.mean()
    vol = r.std()
    sharpe = (mean / vol) * (np.sqrt(52) if freq == "W" else np.sqrt(252))
    cagr = (1 + r).prod()**(52/len(r)) - 1 if len(r) > 0 else 0.0
    cum = (1 + r).cumprod()
    dd = 1 - cum / cum.cummax()
    maxdd = dd.max() if not dd.empty else 0.0
    return pd.Series({"Mean": mean, "Vol": vol, "Sharpe": sharpe, "CAGR": cagr, "MaxDD": maxdd})
