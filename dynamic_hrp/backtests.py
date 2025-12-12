# This module implements a small backtesting toolkit for regime-aware and baseline portfolios.
# It includes:
#   1) A utility to align multiple DataFrames on a common date index and trim to fully-observed rows.
#   2) A walk-forward Dynamic HRP backtest that switches risk models by HMM-inferred market regime.
#   3) Two baselines: Equal-Weight and Static HRP (variance-based).

from __future__ import annotations
import numpy as np
import pandas as pd
from .hmm_wf import fit_hmm_walkforward        # walk-forward Hidden Markov Model labeling of regimes
from .hrp import hrp_variance, hrp_cvar        # Hierarchical Risk Parity allocators (variance, CVaR)
from .supervised_regime_predict import supervised_regime_prediction

def align_and_trim_to_full_rows(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Inner-join all inputs on their common index, then trim from the first
    row that has no NaNs across the concatenated matrix. Returns trimmed
    DataFrames in the same order as provided.
    """
    from functools import reduce
    # Find the intersection of all indexes (dates common to every DataFrame).
    common_idx = reduce(lambda a, b: a.intersection(b), (d.index for d in dfs))

    # Reindex each DataFrame to the common index and sort chronologically.
    aligned = [d.loc[common_idx].sort_index() for d in dfs]

    # Concatenate side-by-side to check where every column (across all inputs) is fully non-NA.
    big = pd.concat(aligned, axis=1)

    # Boolean mask: True on rows with NO NaNs across the entire concatenated frame.
    mask = big.notna().all(axis=1)

    # If no row is fully observed, we cannot start a clean backtest.
    if not mask.any():
        raise ValueError("No date has complete data across inputs.")

    # First fully-observed timestamp (start of clean history).
    start = mask.idxmax()

    # Return each original DF trimmed from that start date onward (preserving input order).
    return [d.loc[start:] for d in aligned]

def backtest_dynamic_hrp(
    features_std: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    lookback_weeks: int = 52,     # rolling window length for risk estimation
    n_components: int = 3,        # number of HMM latent states (regimes)
    rebalance_freq: int = 1,      # rebalance every 'rebalance_freq' weeks
    min_train_weeks: int = 104,   # minimum history for HMM to begin labeling
):
    """
    Walk-forward Dynamic HRP with regime switching (index-safe).

    Logic:
      • Align features and returns by shared dates.
      • Run walk-forward HMM on standardized features to label regimes over time.
      • For each rebalance date t, use the last 'lookback_weeks' of returns and the
        current regime to pick an allocator:
            - "Crisis"    -> HRP by CVaR (tail-risk aware)
            - "Trending"  -> HRP by variance (trend / lower-tail focus less emphasized)
            - otherwise   -> blend 50/50 variance and CVaR HRP
      • Normalize weights to sum of absolute values = 1 (allows long-only or long/short depending on allocators).
      • Apply weights at t to next week's realized returns (t+1) to compute PnL.
    """
    # Align both inputs to the shared date range; preserve chronological order.
    common_idx = weekly_returns.index.intersection(features_std.index)
    if len(common_idx) == 0:
        raise ValueError("No overlap between features_std and weekly_returns indices.")
    weekly_returns = weekly_returns.loc[common_idx].sort_index()
    features_std   = features_std.loc[common_idx].sort_index()

    # Walk-forward HMM regime labeling on the (standardized) feature space.
    hmm_out = fit_hmm_walkforward(
        features_std=features_std,
        n_components=n_components,
        covariance_type="diag",     # diagonal covariance to reduce overfitting
        refit_every_weeks=4,        # periodically refit to allow time variation
        min_train_weeks=min_train_weeks,
        min_state_duration=2,       # simple persistence guardrail
        random_state=42,
    )
    # Align regime labels to our working date index.
    regimes = hmm_out["state_label"].reindex(common_idx)

    # Identify the first valid regime label and choose a safe start index t0:
    # must have at least 'lookback_weeks' of return history and at least 1 labeled regime before t.
    first_regime_pos = regimes.first_valid_index()
    if first_regime_pos is None:
        raise ValueError("No valid regime labels.")
    pos_regime_start = regimes.index.get_indexer([first_regime_pos])[0]
    t0 = max(lookback_weeks, pos_regime_start + 1)
    if t0 >= len(weekly_returns):
        raise ValueError(f"Not enough data: t0={t0}, len={len(weekly_returns)}.")

    # Preallocate weight and PnL containers over the full index.
    weights = pd.DataFrame(index=weekly_returns.index, columns=weekly_returns.columns, dtype=float)
    pnl = pd.Series(index=weekly_returns.index, dtype=float)

    # Walk through time and rebalance on schedule.
    for t in range(t0, len(weekly_returns)):
        # Only compute new weights when it's a rebalance week; otherwise carry forward.
        if (t - t0) % rebalance_freq != 0:
            continue

        # Historical window of returns for risk estimation.
        hist_returns   = weekly_returns.iloc[t - lookback_weeks : t]
        # Use the most recent regime label (information up to t-1).
        current_regime = regimes.iloc[t - 1]

        # Choose allocator based on regime; blend for "Other/Neutral" states.
        if current_regime == "Crisis":
            w = hrp_cvar(hist_returns)
        elif current_regime == "Trending":
            w = hrp_variance(hist_returns)
        else:
            w = 0.5 * hrp_variance(hist_returns) + 0.5 * hrp_cvar(hist_returns)

        # Normalize to L1 = 1 (sum of absolute weights), guarding against all-NaN or zero vector.
        if np.nansum(np.abs(w.values)) > 0:
            w = w / np.nansum(np.abs(w.values))

        # Record weights at time t (others will be ffilled later).
        weights.iloc[t] = w

        # Realized next-period PnL: apply weights decided at t to returns at t+1 (simple one-period slippage).
        if t + 1 < len(weekly_returns):
            pnl.iloc[t + 1] = float(np.nansum(w.values * weekly_returns.iloc[t + 1].fillna(0).values))

    # Forward-fill weights to cover non-rebalance weeks; fill missing PnL with zeros.
    weights = weights.ffill()
    pnl = pnl.fillna(0.0)

    # Return a small result bundle: regimes, weights, weekly PnL, and cumulative PnL.
    return {
        "regimes": regimes,
        "weights": weights,
        "pnl": pnl,
        "cum_pnl": pnl.cumsum(),
    }

def backtest_equal_weight(weekly_returns: pd.DataFrame, rebalance_freq: int = 1) -> dict:
    """
    Simple baseline: equally-weighted portfolio across all assets.
    Rebalance every 'rebalance_freq' weeks; otherwise hold weights constant.
    PnL at t+1 uses the weights chosen at t.
    """
    idx = weekly_returns.index
    cols = weekly_returns.columns
    N = len(cols)

    # Fixed equal-weight vector (sum to 1).
    w = pd.Series(1.0 / N, index=cols)

    # Preallocate containers.
    weights = pd.DataFrame(index=idx, columns=cols, dtype=float)
    pnl = pd.Series(index=idx, dtype=float)

    for t in range(len(idx)):
        # Set or carry weights depending on rebalance schedule.
        if t % rebalance_freq == 0:
            weights.iloc[t] = w
        else:
            weights.iloc[t] = weights.iloc[t-1]

        # Realized PnL at t+1 using weights set at t.
        if t+1 < len(idx):
            pnl.iloc[t+1] = float(np.nansum(weights.iloc[t].values * weekly_returns.iloc[t+1].fillna(0).values))

    # Fill any gaps from initial periods and accumulate PnL over time.
    return {"weights": weights.ffill(), "pnl": pnl.fillna(0.0), "cum_pnl": pnl.fillna(0.0).cumsum()}

def backtest_static_hrp_var(weekly_returns: pd.DataFrame, lookback_weeks: int = 52) -> dict:
    """
    Static HRP (variance-based) baseline:
      • Estimate one set of HRP-variance weights using ONLY the first 'lookback_weeks'.
      • Hold those weights constant for the entire sample.
      • Compute next-period PnL by applying fixed weights to weekly returns (shifted by one week).
    """
    # Need at least one full lookback window to estimate the static weights.
    if len(weekly_returns) <= lookback_weeks:
        raise ValueError("Not enough data to form static HRP baseline.")

    # Fit HRP (variance) on the initial lookback window.
    w0 = hrp_variance(weekly_returns.iloc[:lookback_weeks])

    # Normalize to L1 = 1 to keep scale consistent.
    w0 = w0 / np.nansum(np.abs(w0.values))

    # Broadcast the fixed weight vector across all dates.
    idx = weekly_returns.index
    weights = pd.DataFrame([w0.values]*len(idx), index=idx, columns=weekly_returns.columns)

    # Use next week's returns to compute realized PnL (simple one-period lag).
    pnl = (weekly_returns.shift(-1) * weights).sum(axis=1)  # trade next week

    # Return weights, weekly PnL, and cumulative PnL (fill NaNs arising from the shift).
    return {"weights": weights, "pnl": pnl.fillna(0.0), "cum_pnl": pnl.fillna(0.0).cumsum()}



# --- 4. Supervised HRP Backtest (Uses XGBoost Regime) ---
def backtest_supervised_hrp(
    features_std: pd.DataFrame,
    ret_weekly_trim: pd.DataFrame, 
    prices_daily_for_cusum: pd.DataFrame, 
    lookback_weeks: int = 52,
    rebalance_freq: int = 1,
    min_train_weeks: int = 104,
    cusum_threshold: float = 0.005 
) -> dict:
    """
    Backtest HRP with regime switching driven by Supervised Model (XGBoost)
    using CUSUM-filtered labels.
    """
    # 1. Generate Supervised Regime Labels
    supervised_regimes, final_model = supervised_regime_prediction(
        # Pass features_std from the input
        features_std=features_std, 
        # Pass prices_daily_for_cusum from the input, but name it 'prices_daily'
        prices_daily=prices_daily_for_cusum, 
        # Pass other parameters
        min_train_weeks=min_train_weeks,
        cusum_threshold=cusum_threshold
        # Add refit_every_weeks and test_window_weeks if you included them
    )
    
    # Align and trim data 
    df_aligned = pd.concat(
        [ret_weekly_trim, supervised_regimes.rename("regime")], axis=1
    ).dropna(subset=["regime"])
    
    weekly_returns_trim = df_aligned.drop(columns=["regime"])
    regimes = df_aligned["regime"]
    
    t0 = lookback_weeks + 1 
    if t0 >= len(weekly_returns_trim):
        # Fall back gracefully if not enough history
        # print("Not enough data to run full supervised HRP backtest.")
        return {"regimes": pd.Series(dtype=str), "weights": pd.DataFrame(), "pnl": pd.Series(), "cum_pnl": pd.Series()}

    weights = pd.DataFrame(index=weekly_returns_trim.index, columns=weekly_returns_trim.columns, dtype=float)
    pnl = pd.Series(index=weekly_returns_trim.index, dtype=float)

    from .hrp import hrp_variance, hrp_cvar 

    # Walk-forward portfolio construction
    for t in range(t0, len(weekly_returns_trim)):
        if (t - t0) % rebalance_freq != 0:
            continue

        hist_returns = weekly_returns_trim.iloc[t - lookback_weeks : t]
        current_regime = regimes.iloc[t] 

        if current_regime == "Crisis":
            w = hrp_cvar(hist_returns)
        elif current_regime == "Trending":
            w = hrp_variance(hist_returns)
        else:
            w = 0.5 * hrp_variance(hist_returns) + 0.5 * hrp_cvar(hist_returns)

        abs_sum = np.nansum(np.abs(w.values))
        if abs_sum > 0:
            w = w / abs_sum

        weights.iloc[t] = w

        if t + 1 < len(weekly_returns_trim):
            pnl.iloc[t + 1] = float(np.nansum(w.values * weekly_returns_trim.iloc[t + 1].fillna(0).values))

    weights = weights.ffill()
    pnl = pnl.fillna(0.0)

    return {
        "regimes": regimes,
        "weights": weights,
        "pnl": pnl,
        "cum_pnl": pnl.cumsum(),
    }