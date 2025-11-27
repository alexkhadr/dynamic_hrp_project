# -------------------------------------------------------------
# Hierarchical Risk Parity (HRP) Portfolio Construction Module
# -------------------------------------------------------------
# Implements both variance-based and CVaR-based HRP weighting schemes
# (following López de Prado, 2016). Uses hierarchical clustering to
# allocate capital inversely proportional to cluster risk.
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from .denoising import denoise_cov


# Try importing Ledoit-Wolf shrinkage covariance estimator (optional)
try:
    from sklearn.covariance import LedoitWolf
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# -------------------------------------------------------------
# Covariance Estimation
# -------------------------------------------------------------
def cov_shrink(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute covariance matrix using Ledoit–Wolf shrinkage
    (fallback to simple sample covariance if sklearn is unavailable).
    """
    X = returns.values
    # Drop rows containing NaNs or infs
    X = X[np.isfinite(X).all(1)]
    if X.shape[0] < 2:
        # Not enough data → return small diagonal matrix to avoid singularity
        return pd.DataFrame(np.eye(returns.shape[1]) * 1e-8, index=returns.columns, columns=returns.columns)

    # Ledoit–Wolf shrinkage if available
    if _HAVE_SK:
        lw = LedoitWolf().fit(X)
        cov = lw.covariance_
    else:
        cov = np.cov(X, rowvar=False)

    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


# -------------------------------------------------------------
# Correlation and Distance Matrices
# -------------------------------------------------------------
def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix from covariance matrix."""
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1e-12  # Avoid division by zero
    C = cov.values / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(C, index=cov.index, columns=cov.columns)


def dist_from_corr(corr: pd.DataFrame) -> pd.DataFrame:
    """
    López de Prado-style distance metric derived from correlation:
      d(i,j) = sqrt(0.5 * (1 - corr(i,j)))
    Ensures that similar (highly correlated) assets are closer in clustering space.
    """
    return np.sqrt(0.5 * (1 - corr.clip(-1, 1)))


# -------------------------------------------------------------
# Seriation (cluster ordering)
# -------------------------------------------------------------
def seriation_order(link_mat: np.ndarray, n: int) -> list[int]:
    """
    Extract hierarchical ordering (leaf order) from linkage matrix.
    The order determines how assets are recursively grouped and allocated.
    """
    order = [int(link_mat[-1, 0]), int(link_mat[-1, 1])]
    while any(i >= n for i in order):
        new_order = []
        for i in order:
            if i < n:
                new_order.append(i)
            else:
                # Replace cluster node with its two children
                a = int(link_mat[i - n, 0])
                b = int(link_mat[i - n, 1])
                new_order.extend([a, b])
        new_order, order = [], new_order
        order = new_order if new_order else order
    return order


# -------------------------------------------------------------
# Cluster-level risk and allocation utilities
# -------------------------------------------------------------
def ivp_w(cov_sub: pd.DataFrame) -> np.ndarray:
    """
    Compute inverse-variance portfolio (IVP) weights for a given covariance submatrix.
    Used as intra-cluster weighting scheme.
    """
    iv = 1.0 / np.diag(cov_sub.values)
    iv[~np.isfinite(iv)] = 0.0
    s = iv.sum()
    if s <= 0:
        return np.ones(len(iv)) / len(iv)
    return iv / s


def cluster_var(cov: pd.DataFrame, idx: list[int]) -> float:
    """
    Compute variance (risk) of a cluster identified by asset indices.
    The cluster variance = wᵀ Σ w, where w are inverse-variance weights.
    """
    cov_sub = cov.iloc[idx, idx]
    w = ivp_w(cov_sub)
    return float(w @ cov_sub.values @ w)


def cluster_cvar(returns: pd.DataFrame, cov: pd.DataFrame, idx: list[int], alpha: float = 0.95) -> float:
    """
    Compute Conditional Value-at-Risk (CVaR) for a cluster of assets.

    Steps:
      - Build cluster return series as weighted sum of asset returns.
      - Estimate VaR(α) and take mean of tail losses beyond that threshold.
      - Used as an alternative risk measure to variance.
    """
    if len(idx) == 1:
        # Single-asset cluster
        series = returns.iloc[:, idx[0]].dropna()
        if series.empty:
            return 0.0
        losses = -series.values
        var = np.quantile(losses, alpha) if len(losses) > 1 else np.mean(losses)
        tail = losses[losses >= var]
        return float(np.mean(tail)) if tail.size > 0 else float(var)

    # Multi-asset cluster
    R = returns.iloc[:, idx].dropna(how="any")
    if R.shape[0] < 2:
        return 0.0
    cov_sub = cov.iloc[idx, idx]
    w = ivp_w(cov_sub)
    losses = -(R.values @ w)  # portfolio losses
    var = np.quantile(losses, alpha) if len(losses) > 1 else np.mean(losses)
    tail = losses[losses >= var]
    return float(np.mean(tail)) if tail.size > 0 else float(var)


# -------------------------------------------------------------
# Recursive bisection allocation (core HRP algorithm)
# -------------------------------------------------------------
def _hrp_recursive(order: list[int], cov: pd.DataFrame, risk_func) -> np.ndarray:
    """
    Hierarchical recursive risk allocation:
      - Splits cluster tree recursively into left/right halves.
      - Allocates weights between sub-clusters inversely to their risk.
      - Repeats until reaching individual assets.
    """
    n = len(order)
    w = np.ones(n)
    clusters = [order]

    while clusters:
        new_clusters = []
        for c in clusters:
            if len(c) <= 1:
                continue
            split = len(c) // 2
            left, right = c[:split], c[split:]

            # Compute cluster-level risks
            risk_l = risk_func(left)
            risk_r = risk_func(right)
            denom = (risk_l + risk_r)

            # Allocate inversely to risk: higher-risk cluster gets smaller weight
            alpha = 0.5 if denom <= 0 else 1.0 - (risk_l / denom)
            w[[order.index(i) for i in left]]  *= alpha
            w[[order.index(i) for i in right]] *= (1.0 - alpha)

            # Add sub-clusters to process further
            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        clusters = new_clusters

    # Normalize final weights
    return w / w.sum()


# -------------------------------------------------------------
# HRP with Variance Risk Measure
# -------------------------------------------------------------
def hrp_variance(returns_window: pd.DataFrame) -> pd.Series:
    """
    Compute Hierarchical Risk Parity (HRP) portfolio weights using variance as risk metric.
    """
    cols = list(returns_window.columns)

    # --- RMT Denoising Parameters ---
    T = len(returns_window.dropna(how='all'))
    N = len(cols)
    q = T / N
    
    # Step 1: Estimate covariance
    cov = cov_shrink(returns_window)
    
    # NEW STEP 1b: Denoise the covariance matrix
    cov_denoised = denoise_cov(cov, q=q, bandwidth=0.01) # Use the computed q=T/N ratio

    # Step 2: Convert denoised covariance to correlation
    corr = corr_from_cov(cov_denoised) # <-- Use DENOISED COV

    # Step 3: Convert correlation to distance matrix (for clustering)
    dist = dist_from_corr(corr)
    condensed = squareform(dist.values, checks=False)

    # Step 4: Perform hierarchical clustering (single linkage)
    Z = linkage(condensed, method="single")

    # Step 5: Extract leaf ordering (asset sequence)
    order = seriation_order(Z, n=len(cols))

    # Step 6: Recursively allocate weights based on cluster variance
    risk = lambda idx: cluster_var(cov_denoised, idx) # <-- Use DENOISED COV
    w_ord = _hrp_recursive(order, cov_denoised, risk) # <-- Use DENOISED COV

    # Step 7: Map ordered weights back to asset labels
    w = pd.Series(0.0, index=cols)
    for pos, col_idx in enumerate(order):
        w.iloc[col_idx] = w_ord[pos]
    return w

# -------------------------------------------------------------
# HRP with CVaR Risk Measure
# -------------------------------------------------------------
def hrp_cvar(returns_window: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    """
    Compute Hierarchical Risk Parity (HRP) portfolio weights using CVaR as risk metric.
    """
    cols = list(returns_window.columns)

    # --- RMT Denoising Parameters ---
    T = len(returns_window.dropna(how='all'))
    N = len(cols)
    q = T / N

    # Step 1: Estimate covariance
    cov = cov_shrink(returns_window)

    # NEW STEP 1b: Denoise the covariance matrix
    cov_denoised = denoise_cov(cov, q=q, bandwidth=0.01) # Use the computed q=T/N ratio

    # Step 2: Convert denoised covariance to correlation
    corr = corr_from_cov(cov_denoised) # <-- Use DENOISED COV

    # Step 3: Convert correlation to López de Prado distance
    dist = dist_from_corr(corr)
    condensed = squareform(dist.values, checks=False)

    # Step 4: Hierarchical clustering
    Z = linkage(condensed, method="single")

    # Step 5: Asset ordering and recursive allocation using CVaR
    order = seriation_order(Z, n=len(cols))
    
    # NOTE: CVaR still needs the returns_window for tail calculation, but the
    # recursive HRP allocation should use the denoised matrix for risk measurement 
    # (specifically, the `cluster_cvar` function internally uses `ivp_w`, which 
    # uses the covariance submatrix for weighting).
    risk = lambda idx: cluster_cvar(returns_window, cov_denoised, idx, alpha=alpha) # <-- Use DENOISED COV
    w_ord = _hrp_recursive(order, cov_denoised, risk) # <-- Use DENOISED COV

    # Step 6: Re-map weights to asset names
    w = pd.Series(0.0, index=cols)
    for pos, col_idx in enumerate(order):
        w.iloc[col_idx] = w_ord[pos]
    return w