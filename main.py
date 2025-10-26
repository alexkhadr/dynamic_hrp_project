"""
Main runner for the Dynamic HRP pipeline.

High-level workflow:
  1) Load raw CSVs
  2) Split OHLCV blocks for energy/metals and build master price panels
  3) Define investable universe & features (daily/weekly)
  4) Compute weekly returns (W-FRI) and enforce a 1-week execution delay
  5) Build trend-following signals (TSMOM)
  6) Construct HMM regime features (vol, corr, skew, etc.)
  7) Run walk-forward HMM to classify market regimes
  8) Backtest Dynamic HRP vs Equal-Weight and Static HRP benchmarks
  9) Evaluate performance and visualize results
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")  # suppress warnings for clean console output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Import project modules ---
from dynamic_hrp.io_data import load_raw_frames, split_ohlcv_blocks, build_price_panels
from dynamic_hrp.universe import define_investable_universe
from dynamic_hrp.returns import weekly_log_returns_from_daily, apply_weekly_execution_delay
from dynamic_hrp.signals import tsmom_signal_ma, smooth_signals
from dynamic_hrp.features import build_and_standardize_hmm_features
from dynamic_hrp.backtests import (
    align_and_trim_to_full_rows,
    backtest_dynamic_hrp,
    backtest_equal_weight,
    backtest_static_hrp_var,
)
from dynamic_hrp.graphs import plot_statewise_strategy_hists, plot_hist_by_regime
from perf import perf_stats  # local performance stats module

# =============================================================
# -------------------------- MAIN FUNCTION --------------------
# =============================================================
def run(convert_rx1_to_usd: bool = False, show_plots: bool = True):
    """
    Execute the full Dynamic HRP pipeline.

    Parameters
    ----------
    convert_rx1_to_usd : bool, default False
        Whether to convert the RX1 (Bund future) prices into USD terms.
    show_plots : bool, default True
        Whether to display cumulative return and histogram plots.
    """

    # ---------------------------------------------------------
    # 1) Load raw data from all input CSVs
    # ---------------------------------------------------------
    raw = load_raw_frames()  # dictionary of raw dataframes

    # ---------------------------------------------------------
    # 2) Split the wide energy/metals OHLCV file into individual tickers
    # ---------------------------------------------------------
    energy_blocks = split_ohlcv_blocks(raw["Energy_Comdty"])
    # Retrieve Gold (GC1) and Crude Oil (CL1) dataframes (may vary depending on file labels)
    GC1 = energy_blocks.get("GC1 Comdty", pd.DataFrame())
    CL1 = energy_blocks.get("CL1 Comdty", pd.DataFrame())

    # ---------------------------------------------------------
    # 3) Build master price panels (daily & weekly)
    # ---------------------------------------------------------
    prices_daily, prices_weekly, sub = build_price_panels(
        Spot_FX=raw["Spot_FX"],
        TY1_Comdty=raw["TY1_Comdty"],
        RX1_Comdty=raw["RX1_Comdty"],
        Equity_Futures=raw["Equity_Futures"],
        Volatility_Index=raw["Volatility_Index"],
        GC1=GC1,
        CL1=CL1,
        convert_rx1_to_usd=convert_rx1_to_usd,
    )

    # ---------------------------------------------------------
    # 4) Define investable universe and feature panels
    # ---------------------------------------------------------
    # Tradables: ES1, TY1, EU1 (or RX1 fallback), CL1
    # Feature: VIX (volatility index)
    univ_d, univ_w, feats_d, mapping = define_investable_universe(prices_daily, prices_weekly)

    # If weekly prices were not provided, construct them from daily data
    if univ_w is None or univ_w.empty:
        from dynamic_hrp.returns import weekly_last_from_daily
        univ_w = weekly_last_from_daily(univ_d, week_day="FRI")

    # ---------------------------------------------------------
    # 5) Compute weekly returns and apply 1-week execution delay
    # ---------------------------------------------------------
    p_w, ret_weekly = weekly_log_returns_from_daily(univ_d, week_day="FRI")

    ret_weekly_exec = apply_weekly_execution_delay(
        weekly_weights_dates=p_w.index,
        weekly_returns=ret_weekly,
        delay_weeks=1,  # ensures signals at t are executed at t+1
    )

    # ---------------------------------------------------------
    # 6) Build time-series momentum (TSMOM) signals
    # ---------------------------------------------------------
    signals_ma, _ = tsmom_signal_ma(univ_w, lookbacks=[13, 26, 52])
    # Apply exponential smoothing to reduce noise and sign flips
    signals_ma_smooth = smooth_signals(signals_ma)

    # ---------------------------------------------------------
    # 7) Build Hidden Markov Model (HMM) features
    # ---------------------------------------------------------
    # Regime features include volatility, correlations, skew, kurtosis, etc.
    # Standardized via expanding z-scores to avoid look-ahead bias.
    features_raw, features_std = build_and_standardize_hmm_features(
        weekly_returns=ret_weekly,
        weekly_signals=signals_ma,      # use unsmoothed signals for feature construction
        vix_daily=feats_d.get("VIX"),   # optional volatility input
        window_weeks=26,
        freq="W",
        min_periods_std=26,
    )

    # ---------------------------------------------------------
    # 8) Align all data to full overlapping rows and run backtests
    # ---------------------------------------------------------
    features_std_trim, ret_weekly_trim, signals_trim = align_and_trim_to_full_rows(
        features_std, ret_weekly, signals_ma_smooth
    )

    # --- Run backtests ---
    bt_dyn = backtest_dynamic_hrp(features_std_trim, ret_weekly_trim)  # regime-switching HRP
    bt_eq  = backtest_equal_weight(ret_weekly_trim)                    # equal-weight benchmark
    bt_hrp = backtest_static_hrp_var(ret_weekly_trim)                  # static HRP baseline

    # ---------------------------------------------------------
    # 9) Evaluate performance (weekly statistics)
    # ---------------------------------------------------------
    print("\nPerformance (weekly):")
    print("Dynamic HRP:\n", perf_stats(bt_dyn["pnl"]))
    print("Equal Weight:\n", perf_stats(bt_eq["pnl"]))
    print("Static HRP (Var):\n", perf_stats(bt_hrp["pnl"]))

    # ---------------------------------------------------------
    # 10) Plot regime and strategy histograms
    # ---------------------------------------------------------
    plot_hist_by_regime(bt_dyn["pnl"], bt_dyn["regimes"])
    plot_statewise_strategy_hists(bt_dyn, bt_eq, bt_hrp, bins=40)

    # ---------------------------------------------------------
    # 11) Plot cumulative PnL comparison
    # ---------------------------------------------------------
    if show_plots:
        pd.concat({
            "Dynamic HRP": bt_dyn["cum_pnl"],
            "Equal Weight": bt_eq["cum_pnl"],
            "Static HRP (Var)": bt_hrp["cum_pnl"],
        }, axis=1).plot(
            figsize=(10, 5),
            title="Strategy vs Benchmarks (Cumulative Return)"
        )
        plt.tight_layout()
        plt.show()


# =============================================================
# --------------------------- ENTRY POINT ---------------------
# =============================================================
if __name__ == "__main__":
    # Run full pipeline (default: RX1 not converted to USD, plots enabled)
    run(convert_rx1_to_usd=False, show_plots=True)
