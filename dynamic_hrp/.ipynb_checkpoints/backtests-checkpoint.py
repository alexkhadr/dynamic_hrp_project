from __future__ import annotations
import numpy as np
import pandas as pd
from .hmm_wf import fit_hmm_walkforward
from .hrp import hrp_variance, hrp_cvar

def align_and_trim_to_full_rows(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Inner-join all inputs on their common index, then trim from the first
    row that has no NaNs across the concatenated matrix. Returns trimmed
    DataFrames in the same order as provided.
    """
    from functools import reduce
    common_idx = reduce(lambda a, b: a.intersection(b), (d.index for d in dfs))
    aligned = [d.loc[common_idx].sort_index() for d in dfs]

    big = pd.concat(aligned, axis=1)
    mask = big.notna().all(axis=1)
    if not mask.any():
        raise ValueError("No date has complete data across inputs.")
    start = mask.idxmax()
    return [d.loc[start:] for d in aligned]

def backtest_dynamic_hrp(
    features_std: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    lookback_weeks: int = 52,
    n_components: int = 3,
    rebalance_freq: int = 1,
    min_train_weeks: int = 104,
):
    """
    Walk-forward Dynamic HRP with regime switching (index-safe).
    """
    common_idx = weekly_returns.index.intersection(features_std.index)
    if len(common_idx) == 0:
        raise ValueError("No overlap between features_std and weekly_returns indices.")
    weekly_returns = weekly_returns.loc[common_idx].sort_index()
    features_std   = features_std.loc[common_idx].sort_index()

    hmm_out = fit_hmm_walkforward(
        features_std=features_std,
        n_components=n_components,
        covariance_type="diag",
        refit_every_weeks=4,
        min_train_weeks=min_train_weeks,
        min_state_duration=2,
        random_state=42,
    )
    regimes = hmm_out["state_label"].reindex(common_idx)

    # choose a safe start index
    first_regime_pos = regimes.first_valid_index()
    if first_regime_pos is None:
        raise ValueError("No valid regime labels.")
    pos_regime_start = regimes.index.get_indexer([first_regime_pos])[0]
    t0 = max(lookback_weeks, pos_regime_start + 1)
    if t0 >= len(weekly_returns):
        raise ValueError(f"Not enough data: t0={t0}, len={len(weekly_returns)}.")

    weights = pd.DataFrame(index=weekly_returns.index, columns=weekly_returns.columns, dtype=float)
    pnl = pd.Series(index=weekly_returns.index, dtype=float)

    for t in range(t0, len(weekly_returns)):
        if (t - t0) % rebalance_freq != 0:
            continue

        hist_returns   = weekly_returns.iloc[t - lookback_weeks : t]
        current_regime = regimes.iloc[t - 1]

        if current_regime == "Crisis":
            w = hrp_cvar(hist_returns)
        elif current_regime == "Trending":
            w = hrp_variance(hist_returns)
        else:
            w = 0.5 * hrp_variance(hist_returns) + 0.5 * hrp_cvar(hist_returns)

        if np.nansum(np.abs(w.values)) > 0:
            w = w / np.nansum(np.abs(w.values))
        weights.iloc[t] = w

        if t + 1 < len(weekly_returns):
            pnl.iloc[t + 1] = float(np.nansum(w.values * weekly_returns.iloc[t + 1].fillna(0).values))

    weights = weights.ffill()
    pnl = pnl.fillna(0.0)
    return {
        "regimes": regimes,
        "weights": weights,
        "pnl": pnl,
        "cum_pnl": pnl.cumsum(),
    }

def backtest_equal_weight(weekly_returns: pd.DataFrame, rebalance_freq: int = 1) -> dict:
    idx = weekly_returns.index
    cols = weekly_returns.columns
    N = len(cols)
    w = pd.Series(1.0 / N, index=cols)
    weights = pd.DataFrame(index=idx, columns=cols, dtype=float)
    pnl = pd.Series(index=idx, dtype=float)
    for t in range(len(idx)):
        if t % rebalance_freq == 0:
            weights.iloc[t] = w
        else:
            weights.iloc[t] = weights.iloc[t-1]
        if t+1 < len(idx):
            pnl.iloc[t+1] = float(np.nansum(weights.iloc[t].values * weekly_returns.iloc[t+1].fillna(0).values))
    return {"weights": weights.ffill(), "pnl": pnl.fillna(0.0), "cum_pnl": pnl.fillna(0.0).cumsum()}

def backtest_static_hrp_var(weekly_returns: pd.DataFrame, lookback_weeks: int = 52) -> dict:
    """Estimate one HRP (variance) weight vector using the first lookback window; hold it fixed."""
    if len(weekly_returns) <= lookback_weeks:
        raise ValueError("Not enough data to form static HRP baseline.")
    w0 = hrp_variance(weekly_returns.iloc[:lookback_weeks])
    w0 = w0 / np.nansum(np.abs(w0.values))
    idx = weekly_returns.index
    weights = pd.DataFrame([w0.values]*len(idx), index=idx, columns=weekly_returns.columns)
    pnl = (weekly_returns.shift(-1) * weights).sum(axis=1)  # trade next week
    return {"weights": weights, "pnl": pnl.fillna(0.0), "cum_pnl": pnl.fillna(0.0).cumsum()}
