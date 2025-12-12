# dynamic_hrp/dsr_metrics.py
# -------------------------------------------------------------
# Probabilistic and Deflated Sharpe Ratio (DSR) calculations.
# Based on Lopez de Prado's DSR methodology.
# -------------------------------------------------------------
from __future__ import annotations
import numpy as np
from scipy.stats import norm

# -------------------------------------------------------------
# 1. Probabilistic Sharpe Ratio (PSR)
# -------------------------------------------------------------
def get_probabilistic_sharpe_ratio(
    sr: float, 
    sr_null: float, 
    n_obs: int, 
    skew: float, 
    kurt: float
) -> float:
    """
    Calculates the Probabilistic Sharpe Ratio (PSR).
    
    PSR = Probability that the true SR is greater than a benchmark SR_null (usually 0).
    
    Parameters:
    - sr: Observed Annualized Sharpe Ratio.
    - sr_null: Null Hypothesis Sharpe Ratio (e.g., 0).
    - n_obs: Number of (annualized) observations used to compute SR.
    - skew: Skewness of periodic returns.
    - kurt: Excess kurtosis of periodic returns.
    
    Returns:
    - PSR (probability between 0 and 1).
    """
    if n_obs < 10: return np.nan
        
    # Standard deviation of the Sharpe Ratio estimate (Z_SR)
    z_sr = np.sqrt(n_obs - 1) * (sr - sr_null) / np.sqrt(
        1 - skew * sr + (kurt / 4) * sr**2
    )
    
    # PSR is the cumulative distribution function (CDF) of the Z_SR
    return norm.cdf(z_sr)

# -------------------------------------------------------------
# 2. Deflated Sharpe Ratio (DSR) - Required Helpers
# -------------------------------------------------------------

def get_benchmark_sharpe_ratio(n_trials: int, t_obs: int) -> float:
    """
    Calculates the expected maximum Sharpe Ratio (SR*) under the null hypothesis
    of zero mean return (SR_null=0), adjusted for multiple trials.
    
    Based on the assumption of independent, normally distributed returns.
    
    Parameters:
    - n_trials: Effective number of independent backtest trials performed.
    - t_obs: Number of *annualized* observations used (e.g., 52 weeks * years).
    
    Returns:
    - Expected Maximum Sharpe Ratio (SR*).
    """
    if n_trials <= 0 or t_obs <= 0:
        return np.nan
    
    # Expected maximum of N standard normal variables
    z_max = norm.ppf(1 - 1/n_trials)
    
    # Expected Maximum Sharpe Ratio (SR*)
    sr_max = z_max / np.sqrt(t_obs)
    
    return sr_max

# -------------------------------------------------------------
# 3. Deflated Sharpe Ratio (DSR)
# -------------------------------------------------------------
def get_deflated_sharpe_ratio(
    sr: float, 
    n_trials: int, 
    n_obs: int, 
    skew: float, 
    kurt: float
) -> float:
    """
    Calculates the Deflated Sharpe Ratio (DSR).
    
    DSR is the probability that the true Sharpe Ratio is greater than the
    expected maximum Sharpe Ratio (SR*) achieved from N_trials.
    
    Parameters:
    - sr: Observed Annualized Sharpe Ratio.
    - n_trials: Effective number of independent backtest trials performed.
    - n_obs: Number of *annualized* observations used to compute SR.
    - skew: Skewness of periodic returns.
    - kurt: Excess kurtosis of periodic returns.
    
    Returns:
    - DSR (probability between 0 and 1).
    """
    if n_obs < 10: return np.nan
        
    # 1. Calculate the expected maximum Sharpe Ratio (SR*) under the null
    sr_star = get_benchmark_sharpe_ratio(n_trials, n_obs)
    
    # 2. Calculate the DSR (which is PSR with SR_null = SR*)
    dsr = get_probabilistic_sharpe_ratio(sr, sr_star, n_obs, skew, kurt)
    
    return dsr