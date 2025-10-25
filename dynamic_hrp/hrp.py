from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

try:
    from sklearn.covariance import LedoitWolf
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

def cov_shrink(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit–Wolf shrinkage (fallback to sample covariance)."""
    X = returns.values
    X = X[np.isfinite(X).all(1)]
    if X.shape[0] < 2:
        return pd.DataFrame(np.eye(returns.shape[1])*1e-8, index=returns.columns, columns=returns.columns)
    if _HAVE_SK:
        lw = LedoitWolf().fit(X)
        cov = lw.covariance_
    else:
        cov = np.cov(X, rowvar=False)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1e-12
    C = cov.values / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(C, index=cov.index, columns=cov.columns)

def dist_from_corr(corr: pd.DataFrame) -> pd.DataFrame:
    # López de Prado distance
    return np.sqrt(0.5 * (1 - corr.clip(-1, 1)))

def seriation_order(link_mat: np.ndarray, n: int) -> list[int]:
    order = [int(link_mat[-1, 0]), int(link_mat[-1, 1])]
    while any(i >= n for i in order):
        new_order = []
        for i in order:
            if i < n:
                new_order.append(i)
            else:
                a = int(link_mat[i - n, 0]); b = int(link_mat[i - n, 1])
                new_order.extend([a, b])
        new_order, order = [], new_order
        order = new_order if new_order else order
    return order

def ivp_w(cov_sub: pd.DataFrame) -> np.ndarray:
    iv = 1.0 / np.diag(cov_sub.values)
    iv[~np.isfinite(iv)] = 0.0
    s = iv.sum()
    if s <= 0: return np.ones(len(iv)) / len(iv)
    return iv / s

def cluster_var(cov: pd.DataFrame, idx: list[int]) -> float:
    cov_sub = cov.iloc[idx, idx]
    w = ivp_w(cov_sub)
    return float(w @ cov_sub.values @ w)

def cluster_cvar(returns: pd.DataFrame, cov: pd.DataFrame, idx: list[int], alpha: float = 0.95) -> float:
    if len(idx) == 1:
        series = returns.iloc[:, idx[0]].dropna()
        if series.empty: return 0.0
        losses = -series.values
        var = np.quantile(losses, alpha) if len(losses) > 1 else np.mean(losses)
        tail = losses[losses >= var]
        return float(np.mean(tail)) if tail.size > 0 else float(var)

    R = returns.iloc[:, idx].dropna(how="any")
    if R.shape[0] < 2: return 0.0
    cov_sub = cov.iloc[idx, idx]
    w = ivp_w(cov_sub)
    losses = -(R.values @ w)
    var = np.quantile(losses, alpha) if len(losses) > 1 else np.mean(losses)
    tail = losses[losses >= var]
    return float(np.mean(tail)) if tail.size > 0 else float(var)

def _hrp_recursive(order: list[int], cov: pd.DataFrame, risk_func) -> np.ndarray:
    n = len(order); w = np.ones(n); clusters = [order]
    while clusters:
        new_clusters = []
        for c in clusters:
            if len(c) <= 1: continue
            split = len(c)//2; left, right = c[:split], c[split:]
            risk_l = risk_func(left); risk_r = risk_func(right)
            denom = (risk_l + risk_r)
            alpha = 0.5 if denom <= 0 else 1.0 - (risk_l/denom)
            w[[order.index(i) for i in left]]  *= alpha
            w[[order.index(i) for i in right]] *= (1.0 - alpha)
            if len(left)>1:  new_clusters.append(left)
            if len(right)>1: new_clusters.append(right)
        clusters = new_clusters
    return w / w.sum()

def hrp_variance(returns_window: pd.DataFrame) -> pd.Series:
    cols = list(returns_window.columns)
    cov = cov_shrink(returns_window)
    corr = corr_from_cov(cov)
    dist = dist_from_corr(corr)
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="single")
    order = seriation_order(Z, n=len(cols))
    risk = lambda idx: cluster_var(cov, idx)
    w_ord = _hrp_recursive(order, cov, risk)
    w = pd.Series(0.0, index=cols)
    for pos, col_idx in enumerate(order): w.iloc[col_idx] = w_ord[pos]
    return w

def hrp_cvar(returns_window: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    cols = list(returns_window.columns)
    cov = cov_shrink(returns_window)
    corr = corr_from_cov(cov)
    dist = dist_from_corr(corr)
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="single")
    order = seriation_order(Z, n=len(cols))
    risk = lambda idx: cluster_cvar(returns_window, cov, idx, alpha=alpha)
    w_ord = _hrp_recursive(order, cov, risk)
    w = pd.Series(0.0, index=cols)
    for pos, col_idx in enumerate(order): w.iloc[col_idx] = w_ord[pos]
    return w
