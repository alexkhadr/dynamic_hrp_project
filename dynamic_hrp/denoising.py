# dynamic_hrp/denoising.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import eigh

def get_marchenko_pastur_max(q: float, bandwidth: float = 0.01) -> float:
    """
    Calculates the theoretical maximum eigenvalue (lambda_max) for a purely random
    matrix (noise) based on the Marchenko-Pastur distribution.
    q = T / N (time steps / features)
    """
    # If T/N is very small (q < 1), the max eigenvalue is calculated differently
    if q < 1.0:
        q_inv = 1.0 / q
        lambda_max = (1.0 + np.sqrt(q_inv))**2
    else:
        lambda_max = (1.0 + np.sqrt(1.0/q))**2
        
    # The original formula assumes the returns are standardized (mean=0, variance=1).
    # Since we are working with the covariance matrix, we use the formula based on
    # unit variance (sigma^2 = 1).
    
    # Apply a small bandwidth adjustment for practical use
    return lambda_max + bandwidth

def denoise_cov(cov: pd.DataFrame, q: float, bandwidth: float = 0.01) -> pd.DataFrame:
    """
    Denoises a covariance matrix using Random Matrix Theory (RMT).
    
    Parameters:
    - cov: Empirical covariance matrix (DataFrame).
    - q: Ratio T/N (number of observations / number of assets).
    - bandwidth: Small adjustment for the Marchenko-Pastur upper bound.
    
    Returns:
    - Denoised covariance matrix (DataFrame).
    """
    # 1. Decompose the matrix
    # eigh is used for symmetric matrices
    eigen_values, eigen_vectors = eigh(cov.values)
    
    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # 2. Find the noise threshold (lambda_max)
    lambda_max = get_marchenko_pastur_max(q, bandwidth)
    
    # 3. Separate signal and noise eigenvalues
    # Noise eigenvalues are those below the Marchenko-Pastur max
    noise_mask = eigen_values < lambda_max
    
    # 4. Denoise: Replace all noise eigenvalues with their average
    noise_eigenvalues = eigen_values[noise_mask]
    avg_noise_eigenvalue = np.mean(noise_eigenvalues) if noise_eigenvalues.size > 0 else 0.0
    
    # Replace noise components with the average
    denoised_eigen_values = eigen_values.copy()
    denoised_eigen_values[noise_mask] = avg_noise_eigenvalue
    
    # 5. Reconstitute the denoised covariance matrix
    # D = diag(denoised_eigen_values)
    D = np.diag(denoised_eigen_values)
    # Reconstituted Matrix = V @ D @ V.T
    denoised_cov_matrix = eigen_vectors @ D @ eigen_vectors.T
    
    return pd.DataFrame(denoised_cov_matrix, index=cov.index, columns=cov.columns)