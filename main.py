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
from dynamic_hrp.returns import weekly_log_returns_from_daily, apply_weekly_execution_delay, weekly_last_from_daily, daily_log_returns
from dynamic_hrp.signals import tsmom_signal_ma, smooth_signals
from dynamic_hrp.features import build_and_standardize_hmm_features
from dynamic_hrp.backtests import (
    align_and_trim_to_full_rows,
    backtest_dynamic_hrp,
    backtest_equal_weight,
    backtest_static_hrp_var,
    backtest_supervised_hrp
)
from dynamic_hrp.frac_diff import fractionally_differentiated_log_price
from dynamic_hrp.graphs import plot_statewise_strategy_hists, plot_hist_by_regime, plot_cumulative_pnl
from perf import perf_stats  # local performance stats module
from dynamic_hrp.utils_io import save_table_markdown
from dynamic_hrp.dsr_metrics import get_deflated_sharpe_ratio
from dynamic_hrp.cluster_analysis import cluster_feature_importance
from dynamic_hrp.supervised_regime_predict import _get_target_labels_cusum, supervised_regime_prediction, REGIME_MAP
from xgboost import XGBClassifier # <-- ADD THIS


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
    # 1) Load raw data
    # ---------------------------------------------------------
    raw = load_raw_frames()

    # ---------------------------------------------------------
    # 2) Split energy/metals OHLCV blocks
    # ---------------------------------------------------------
    energy_blocks = split_ohlcv_blocks(raw["Energy_Comdty"])
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
    # 4) Define investable universe and features
    # ---------------------------------------------------------
    univ_d, univ_w, feats_d, mapping = define_investable_universe(prices_daily, prices_weekly)

    if univ_w is None or univ_w.empty:
        # Fallback if weekly prices weren't generated in step 3
        # (needs `from dynamic_hrp.returns import weekly_last_from_daily` at top level)
        univ_w = weekly_last_from_daily(univ_d, week_day="FRI")

    # ---------------------------------------------------------
    # 5) Compute weekly returns with 1-week execution delay
    # NOTE: `ret_weekly` here is the standard d=1 log return, used for PnL calculation
    # ---------------------------------------------------------
    p_w, ret_weekly = weekly_log_returns_from_daily(univ_d, week_day="FRI")

    ret_weekly_exec = apply_weekly_execution_delay(
        weekly_weights_dates=p_w.index,
        weekly_returns=ret_weekly,
        delay_weeks=1,
    )

    # ---------------------------------------------------------
    # 6) Build TSMOM signals and smooth them
    # ---------------------------------------------------------
    signals_ma, _ = tsmom_signal_ma(univ_w, lookbacks=[13, 26, 52])
    signals_ma_smooth = smooth_signals(signals_ma)

    # =========================================================
    # 6b) NEW: Compute Fractionally Differentiated Series (for HMM features only)
    # =========================================================
    print("\nStarting Fractional Differentiation (Searching for Optimal d)...")
    # Optimal 'd' is found, and the log price series is differenced using that 'd'
    frac_diff_d = fractionally_differentiated_log_price(univ_d, p_value=0.05)
    
    # Resample the daily fractionally differenced series to weekly frequency (W-FRI)
    frac_diff_w = weekly_last_from_daily(frac_diff_d, week_day="FRI") 
    print("Fractional Differentiation Complete.")
    
    # ---------------------------------------------------------
    # 7) Build HMM regime features
    # KEY CHANGE: Use the fractionally differenced series (frac_diff_w)
    # ---------------------------------------------------------
    features_raw, features_std = build_and_standardize_hmm_features(
        weekly_returns=frac_diff_w,
        weekly_signals=signals_ma,
        vix_daily=feats_d.get("VIX"),
        window_weeks=26,
        freq="W",
        min_periods_std=26,
    )

    # ---------------------------------------------------------
    # --- Step 8) Align all data and run backtests ---
    # ---------------------------------------------------------

    features_std_trim, ret_weekly_trim, signals_trim = align_and_trim_to_full_rows(
        features_std, ret_weekly, signals_ma_smooth
    )

    # --- Backtest Runs ---
    
    # HMM-based HRP
    bt_dyn = backtest_dynamic_hrp(features_std_trim, ret_weekly_trim)
    
    # Supervised-based Dynamic HRP
    print("Starting Supervised HRP Backtest")
    bt_sup = backtest_supervised_hrp(features_std = features_std_trim,
                                     ret_weekly_trim = ret_weekly_trim,
                                     prices_daily_for_cusum = univ_d,
                                     cusum_threshold = 0.005
                                    )
    
    # Baselines
    bt_eq = backtest_equal_weight(ret_weekly_trim)
    bt_hrp = backtest_static_hrp_var(ret_weekly_trim)

    
    # ---------------------------------------------------------
    # 8b) Feature Analysis: Clustered Importance (for Supervised Model)
    # ---------------------------------------------------------

    # NOTE: To do this right, we need the final model and the data it was trained on.
    # The simplest method is to retrain the model once on the entire trimmed dataset
    # to get a single, final model object for analysis.
    
    # We need to extract the training features and targets:
    # from dynamic_hrp.regime_predict import _get_target_labels_cusum, supervised_regime_prediction
    
    y_labels = _get_target_labels_cusum(univ_d, weekly_index=features_std_trim.index, threshold=0.005)
    
    # Align features (X) and CUSUM-derived labels (Y) for the full period
    df_analysis = pd.concat([features_std_trim, y_labels.rename("target")], axis=1).dropna(subset=["target"])
    X_full = df_analysis.drop(columns=["target"])
    Y_map = {v: k for k, v in REGIME_MAP.items()}
    Y_full = df_analysis["target"].map(Y_map)
    
    # Retrain final XGBoost model for analysis (best practice for feature importance)
    final_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                                n_estimators=50, max_depth=3, random_state=42, 
                                n_jobs=-1, verbosity=0)
    final_model.fit(X_full, Y_full)

    # Run Clustered Feature Importance
    print("\nStarting Clustered Feature Importance Analysis...")
    feat_imp_df = cluster_feature_importance(
        X_train=X_full,
        y_train=Y_full,
        model=final_model,
        corr_threshold=0.7 # Tighter clustering for better separation
    )
    
    save_table_markdown(feat_imp_df, out_dir="Tables", name="clustered_feature_importance")

    # ---------------------------------------------------------
    # 9) Evaluate performance (weekly statistics)
    # ---------------------------------------------------------
    perf_dyn = perf_stats(bt_dyn["pnl"])
    perf_sup = perf_stats(bt_sup["pnl"]) # New entry
    perf_eq  = perf_stats(bt_eq["pnl"])
    perf_hrp = perf_stats(bt_hrp["pnl"])

    perf_table = pd.DataFrame({
        "Dynamic HRP (HMM)": perf_dyn,
        "Dynamic HRP (Supervised)": perf_sup, # New column
        "Equal Weight": perf_eq,
        "Static HRP (Var)": perf_hrp,
    }).T
    
    # =========================================================
    # 9a) NEW: Compute Deflated Sharpe Ratio (DSR)
    # =========================================================
    # N_trials: Effective number of strategies being compared (Dyn HRP, EW, Static HRP)
    n_trials = 3 
    # Annualization factor for weekly data is 52
    annualization_factor = 52 
    # N_obs is the number of periodic observations, which is the same for all strategies.
    n_obs_periodic = perf_table["N_obs"].iloc[0] 
    # N_obs for DSR must be in *annualized* terms
    n_obs_annualized = n_obs_periodic / annualization_factor

    dsr_values = {}
    for name, row in perf_table.iterrows():
        # Compute DSR using the Sharpe, Skew, and Kurtosis from perf_stats
        dsr = get_deflated_sharpe_ratio(
            sr=row["Sharpe"],
            n_trials=n_trials,
            n_obs=n_obs_annualized,
            skew=row["Skew"],
            kurt=row["Kurtosis"],
        )
        dsr_values[name] = dsr

    # Add DSR to the performance table and remove the temporary N_obs
    perf_table["DSR"] = pd.Series(dsr_values)
    perf_table = perf_table.drop(columns=["N_obs"])


    save_table_markdown(perf_table, out_dir="Tables", name="performance_weekly")

    # ---------------------------------------------------------
    # 9b) Statistical test: Does Dynamic HRP outperform Static HRP?
    # ---------------------------------------------------------
    from scipy import stats
    import numpy as np

    r_dyn = bt_dyn["pnl"].dropna()
    r_static = bt_hrp["pnl"].dropna()

    aligned = pd.concat([r_dyn, r_static], axis=1).dropna()
    r_dyn = aligned.iloc[:, 0]
    r_static = aligned.iloc[:, 1]

    # Tests
    t_stat, p_val_t = stats.ttest_rel(r_dyn, r_static, alternative='greater')
    stat_w, p_val_w = stats.wilcoxon(r_dyn, r_static, alternative='greater')

    n_boot = 10000
    boot_diff = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(len(r_dyn), len(r_dyn), replace=True)
        boot_diff[i] = r_dyn.iloc[sample].mean() - r_static.iloc[sample].mean()
    p_val_boot = np.mean(boot_diff <= 0)

    tests_table = pd.DataFrame(
        {
            "Statistic": [t_stat, stat_w, np.nan],
            "P-value": [p_val_t, p_val_w, p_val_boot],
            "Alt. Hypothesis": ["mean_dyn > mean_static"] * 3,
        },
        index=["Paired t-test", "Wilcoxon", "Bootstrap"]
    )

    save_table_markdown(tests_table, out_dir="Tables", name="stat_tests_dynamic_vs_static")

    # ---------------------------------------------------------
    # 10) Plot regime and strategy histograms
    # ---------------------------------------------------------
    plot_hist_by_regime(
        bt_dyn["pnl"], bt_dyn["regimes"],
        save_path="Figures/hist_by_regime.png",
        show=show_plots,
        verbose=False,
        save_tables_to="Tables/per_regime_weekly_return_stats.md",
        save_table_markdown_fn=save_table_markdown,
    )

    plot_statewise_strategy_hists(
        bt_dyn, bt_eq, bt_hrp, bins=40,
        save_path="Figures/hist_by_strategy_and_regime.png",
        show=show_plots,
        verbose=False,
        save_tables_to="Tables/per_state_strategy_summary.md",
        save_table_markdown_fn=save_table_markdown,
    )


    # ---------------------------------------------------------
    # 11) Plot cumulative PnL comparison
    # ---------------------------------------------------------
    plot_cumulative_pnl(
        bt_dyn, bt_eq, bt_hrp,
        save_path="Figures/cumulative_pnl.png", show=show_plots
    )

# =============================================================
# --------------------------- ENTRY POINT ---------------------
# =============================================================
if __name__ == "__main__":
    # Run full pipeline (default: RX1 not converted to USD, plots enabled)
    run(convert_rx1_to_usd=False, show_plots=False)
