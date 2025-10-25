"""
Main runner for the Dynamic HRP pipeline.

Steps:
  1) Load raw CSVs
  2) Split OHLCV blocks for energy/metals and build master price panels
  3) Define investable universe & features (daily/weekly)
  4) Compute weekly returns (W-FRI) and enforce a 1-week execution delay (optional)
  5) Build trend signals (TSMOM) and regime features (+ standardized, no look-ahead)
  6) Walk-forward HMM to label regimes
  7) Backtest Dynamic HRP vs Equal-Weight and Static HRP baselines
  8) Print performance stats and plot cumulative PnL
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from perf import perf_stats

def run(convert_rx1_to_usd: bool = False, show_plots: bool = True):
    # 1) Load raw data
    raw = load_raw_frames()

    # 2) Build OHLCV blocks for energy/metals (AT-03)
    energy_blocks = split_ohlcv_blocks(raw["Energy_Comdty"])
    # Keys here depend on exact label in your CSV after cleaning — adjust if different
    GC1 = energy_blocks.get("GC1 Comdty", pd.DataFrame())
    CL1 = energy_blocks.get("CL1 Comdty", pd.DataFrame())

    # 3) Master price panels
    prices_daily, prices_weekly, sub = build_price_panels(
        Spot_FX=raw["Spot_FX"],
        TY1_Comdty=raw["TY1_Comdty"],
        RX1_Comdty=raw["RX1_Comdty"],
        Equity_Futures=raw["Equity_Futures"],
        Volatility_Index=raw["Volatility_Index"],
        GC1=GC1,
        CL1=CL1,
        convert_rx1_to_usd=convert_rx1_to_usd
    )

    # 4) Investable universe (map to ES1, TY1, EU1/RX1, CL1) + feature panel (VIX)
    univ_d, univ_w, feats_d, mapping = define_investable_universe(prices_daily, prices_weekly)
    if univ_w is None or univ_w.empty:
        # fallback: build weekly prices from daily for tradables
        from dynamic_hrp.returns import weekly_last_from_daily
        univ_w = weekly_last_from_daily(univ_d, week_day="FRI")

    # 5) Weekly returns and (optional) execution delay
    p_w, ret_weekly = weekly_log_returns_from_daily(univ_d, week_day="FRI")
    ret_weekly_exec = apply_weekly_execution_delay(
        weekly_weights_dates=p_w.index,
        weekly_returns=ret_weekly,
        delay_weeks=1
    )

    print("Weekly returns shape:", ret_weekly.shape)
    print("Weekly returns (execution-delayed) shape:", ret_weekly_exec.shape)

    # 6) Trend signals
    signals_ma, _ = tsmom_signal_ma(univ_w, lookbacks=[13, 26, 52])
    signals_ma_smooth = smooth_signals(signals_ma)

    # 7) HMM features (raw + standardized)
    features_raw, features_std = build_and_standardize_hmm_features(
        weekly_returns=ret_weekly,
        weekly_signals=signals_ma,      # use combined TSMOM signals (unsmoothed) for features
        vix_daily=feats_d.get("VIX"),   # optional; pass None to skip
        window_weeks=26,
        freq="W",
        min_periods_std=26
    )

    # 8) Align & backtest
    features_std_trim, ret_weekly_trim, signals_trim = align_and_trim_to_full_rows(
        features_std, ret_weekly, signals_ma_smooth
    )

    bt_dyn = backtest_dynamic_hrp(features_std_trim, ret_weekly_trim)
    bt_eq  = backtest_equal_weight(ret_weekly_trim)
    bt_hrp = backtest_static_hrp_var(ret_weekly_trim)

    print("\nPerformance (weekly):")
    print("Dynamic HRP:\n", perf_stats(bt_dyn["pnl"]))
    print("Equal Weight:\n", perf_stats(bt_eq["pnl"]))
    print("Static HRP (Var):\n", perf_stats(bt_hrp["pnl"]))

    # 9) Plot cumulative PnL
    if show_plots:
        pd.concat({
            "Dynamic HRP": bt_dyn["cum_pnl"],
            "Equal Weight": bt_eq["cum_pnl"],
            "Static HRP (Var)": bt_hrp["cum_pnl"],
        }, axis=1).plot(figsize=(10,5), title="Strategy vs Benchmarks (Cumulative Return)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run(convert_rx1_to_usd=False, show_plots=True)
