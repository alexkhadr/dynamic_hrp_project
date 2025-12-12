# dynamic_hrp/frac_diff.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller

# =============================================================
# 1. Fractional Differentiation Kernel
# =============================================================

def get_weights_fixed(d: float, max_len: int) -> np.ndarray:
    """
    Computes the fixed-width weighting kernel for fractional differentiation.
    
    Weights are defined by: w_k = -w_{k-1} * (d - k + 1) / k, where w_0 = 1.
    """
    weights = np.zeros(max_len)
    weights[0] = 1.0
    for k in range(1, max_len):
        weights[k] = -weights[k-1] * (d - k + 1) / k
    return weights

# =============================================================
# 2. Fixed-Width Fractional Difference
# =============================================================

def fractional_difference_fixed(series: pd.Series, d: float, window_size: int = 100) -> pd.Series:
    """
    Applies fractional differentiation of order 'd' using a fixed-width window.
    
    The resulting series will have NaNs until the window is fully observed.
    """
    # 1. Compute weights
    weights = get_weights_fixed(d, window_size)
    
    # 2. Apply convolution (rolling sum)
    f_diff_series = series.apply(np.log).to_frame(name="log_price")
    
    # Use EWM for fractional differencing approximation: sum_{k=0}^{T-1} w_k * p_{t-k}
    # This is achieved by convolution, but a direct approach is clearer here:
    diff_values = []
    
    for t in range(window_size - 1, len(f_diff_series)):
        # Get the lookback window
        window = f_diff_series["log_price"].iloc[t - window_size + 1 : t + 1].values
        
        # Apply dot product: sum(weights * prices)
        # Note: Weights must be reversed to match chronological order of window
        diff_value = np.dot(weights[::-1], window)
        diff_values.append(diff_value)
        
    # 3. Create resulting series, retaining original index and NaN offset
    diff_series = pd.Series(diff_values, index=f_diff_series.index[window_size - 1 :])
    
    # Scale to approximate returns behavior (optional but often helpful)
    return diff_series.diff().dropna() # Take the first difference of the fractional output for returns-like stationarity

# =============================================================
# 3. Optimal d Search & ADF Test
# =============================================================

def adf_test_series(series: pd.Series, p_value: float = 0.05) -> bool:
    """Run ADF test and return True if stationary (p-value < threshold)."""
    # Drop leading NaNs which may corrupt the test
    series = series.dropna() 
    if len(series) < 20: # ADF needs sufficient observations
        return False
    
    try:
        # Perform ADF test
        result = adfuller(series, autolag='AIC')
        p = result[1]
        return p < p_value
    except Exception:
        return False

def fractionally_differentiated_log_price(
    prices_daily: pd.DataFrame, 
    p_value: float = 0.05,
    d_range: np.ndarray | None = None,
    window_size: int = 100
) -> pd.DataFrame:
    """
    Finds the minimum fractional order 'd' that makes the log price series
    stationary (p-value < threshold) and returns the differentiated series.
    """
    if d_range is None:
        # Search range from high memory (d=0.0) to full diff (d=1.0)
        d_range = np.linspace(0.01, 1.0, 20) 
        
    log_price_series = prices_daily.apply(np.log)
    
    # Dictionary to hold the differentiated series for each asset
    frac_diff_results = {}
    
    for asset in log_price_series.columns:
        print(f"Searching for optimal 'd' for {asset}...")
        best_d = 1.0
        
        for d in d_range:
            # Differentiate the asset's log price series
            f_diff_series = fractional_difference_fixed(
                prices_daily[asset], d=d, window_size=window_size
            )
            
            # Check for stationarity
            if adf_test_series(f_diff_series, p_value=p_value):
                # Found the minimum d that passes the test
                best_d = d
                print(f" -> Found optimal d={best_d:.2f} for {asset}")
                break
                
        # Apply the final optimal d (or default to 1.0 if none passed)
        final_series = fractional_difference_fixed(
            prices_daily[asset], d=best_d, window_size=window_size
        )
        frac_diff_results[asset] = final_series
        
    # Combine results into a DataFrame
    return pd.DataFrame(frac_diff_results)