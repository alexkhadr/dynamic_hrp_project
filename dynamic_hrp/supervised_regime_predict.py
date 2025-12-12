# dynamic_hrp/regime_predict.py (CUSUM-Filtered, Walk-Forward XGBoost)
from __future__ import annotations
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from dynamic_hrp.returns import daily_log_returns # Required for CUSUM base series
from dynamic_hrp.filters import apply_cusum_filter # The CUSUM event sampler
from dynamic_hrp.frac_diff import adf_test_series 
from typing import Literal

# --- 1. Constants and Mapping ---
REGIME_MAP = {
    0: "Trending",
    1: "Neutral",
    2: "Crisis",
}

# --- 2. CUSUM-Based Target Labeling ---
def _get_target_labels_cusum(
    prices_daily: pd.DataFrame,
    weekly_index: pd.Index,
    threshold: float = 0.005, # CUSUM threshold (e.g., 0.5% daily deviation)
) -> pd.Series:
    """
    Defines the supervised target variable (Y) by sampling a daily, 
    Equal-Weighted (EW) return/volatility series only at CUSUM event dates.
    
    The final labels are then mapped and forward-filled to the required weekly index.
    """
    # 1. Calculate Daily EW Returns
    # Note: daily_log_returns is a function imported from your returns.py
    daily_ret = daily_log_returns(prices_daily)
    # Mean (Equal-Weighted) daily return series
    ew_daily_ret = daily_ret.mean(axis=1).dropna() 

    # 2. Apply CUSUM Filter to identify event dates 
    event_dates = apply_cusum_filter(ew_daily_ret, threshold=threshold)
    # print(f"CUSUM Filter generated {len(event_dates)} event dates.")
    
    # 3. Use CUSUM events to sample the daily series
    # We sample the *future* performance/volatility relative to the event date.
    
    # Target 1: Future EW return (for profitability)
    future_ew_ret = ew_daily_ret.shift(-1).loc[event_dates]
    
    # Target 2: Future EW volatility (proxy for risk)
    daily_ew_vol = daily_ret.std(axis=1)
    future_ew_vol = daily_ew_vol.shift(-1).loc[event_dates]

    # Combine sampled data, dropping events near the end/start where future data is missing
    df_events = pd.DataFrame({
        "future_ret": future_ew_ret,
        "future_vol": future_ew_vol
    }).dropna()

    if df_events.empty:
        return pd.Series(index=weekly_index, dtype=str)

    # 4. Labeling Logic 
    ret_threshold_low = df_events["future_ret"].quantile(0.33)
    ret_threshold_high = df_events["future_ret"].quantile(0.66)
    vol_low_threshold = df_events["future_vol"].quantile(0.33)
    vol_high_threshold = df_events["future_vol"].quantile(0.66)
    
    labels = pd.Series(index=df_events.index, dtype=int)
    
    # Label 0: Trending/Calm (High return, Low volatility)
    labels[
        (df_events["future_ret"] > ret_threshold_high) & (df_events["future_vol"] < vol_low_threshold)
    ] = 0
    
    # Label 2: Crisis/Turbulent (Low return, High volatility)
    labels[
        (df_events["future_ret"] < ret_threshold_low) & (df_events["future_vol"] > vol_high_threshold)
    ] = 2
    
    # Label 1: Neutral (Fills all other conditions)
    labels.fillna(1, inplace=True)

    # 5. Resample Labels to Weekly Frequency
    # Forward-fill daily series: the regime holds until the next CUSUM event
    full_labels_daily = labels.reindex(prices_daily.index, method='ffill')
    
    # Sample the daily FFILL-ed regime label only on the required weekly dates
    return full_labels_daily.reindex(weekly_index, method='ffill') \
                            .map(REGIME_MAP).rename("supervised_regime").dropna()


# --- 3. Walk-Forward XGBoost Prediction ---
def supervised_regime_prediction(
    features_std: pd.DataFrame,
    prices_daily: pd.DataFrame,
    min_train_weeks: int = 104, # Minimum history to fit the model (e.g., 2 years)
    refit_every_weeks: int = 4,  # How often to retrain the model (e.g., monthly)
    test_window_weeks: int = 4,  # How far out to predict before refitting
    cusum_threshold: float = 0.005
) -> pd.Series:
    """
    Walk-forward XGBoost classification to predict the market regime, 
    with target labels sampled via CUSUM filter.
    """
    
    # 1. Prepare Features (X) and Target (Y)
    y_labels = _get_target_labels_cusum(
        prices_daily, 
        weekly_index=features_std.index,
        threshold=cusum_threshold
    )
    
    # Align features (X) and CUSUM-derived labels (Y)
    df = pd.concat([features_std, y_labels], axis=1).dropna(subset=["supervised_regime"])
    
    X = df.drop(columns=["supervised_regime"]).values
    Y_map = {v: k for k, v in REGIME_MAP.items()} 
    Y = df["supervised_regime"].map(Y_map).values
    dates = df.index
    
    if len(dates) <= min_train_weeks:
        return pd.Series(dtype=str)

    # 2. Walk-Forward Loop Setup
    predicted_regimes = pd.Series(index=dates, dtype="object")
    
    # XGBoost setup 
    base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                               n_estimators=50, max_depth=3, random_state=42, 
                               n_jobs=-1, verbosity=0)

    start_ix = min_train_weeks
    model = base_model # Initial model placeholder
    
    # Perform a single fit on initial window to get a starting model instance
    model.fit(X[:start_ix], Y[:start_ix])

    for t_idx in range(start_ix, len(dates), test_window_weeks):
        
        X_train = X[:t_idx]
        Y_train = Y[:t_idx]
        
        X_test_block = X[t_idx : t_idx + test_window_weeks]
        test_dates_block = dates[t_idx : t_idx + test_window_weeks]

        # Conditional Model Retraining
        if (t_idx - start_ix) % refit_every_weeks == 0:
            # print(f"Refitting XGBoost at date: {dates[t_idx].date()}")
            model = base_model 
            try:
                model.fit(X_train, Y_train)
            except ValueError:
                continue
        
        # Prediction
        if X_test_block.size > 0:
            y_pred_int = model.predict(X_test_block)
            
            predicted_labels = pd.Series(y_pred_int, index=test_dates_block).map(REGIME_MAP)
            predicted_regimes.loc[test_dates_block] = predicted_labels
            
    # 3. Final Cleanup and Initial Period Prediction
    X_initial = X[:start_ix]
    initial_dates = dates[:start_ix]
    
    if X_initial.size > 0:
        # Use the final trained model to predict regimes for the initial training period
        initial_pred_int = model.predict(X_initial)
        initial_labels = pd.Series(initial_pred_int, index=initial_dates).map(REGIME_MAP)
        predicted_regimes.loc[initial_dates] = initial_labels
        
    predicted_regimes = predicted_regimes.ffill().fillna("Neutral")
    
    return predicted_regimes.reindex(features_std.index).ffill(), model
