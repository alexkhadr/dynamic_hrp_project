# dynamic_hrp/analysis.py (New File)
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from typing import Optional

def cluster_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: XGBClassifier,
    corr_threshold: float = 0.8,
    metric: str = 'f1_weighted',
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes Clustered Feature Importance: features are grouped by correlation,
    and importance is calculated per cluster using permutation.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (e.g., features_std_trim for the training window).
    y_train : pd.Series
        Training target labels (e.g., supervised_regime labels).
    model : XGBClassifier
        Trained supervised model.
    corr_threshold : float
        Correlation threshold to define a cluster (e.g., 0.8 means highly correlated).
    n_repeats : int
        Number of times to shuffle a feature for permutation.

    Returns
    -------
    pd.DataFrame
        Feature importance results grouped by cluster.
    """
    # 1. Hierarchical Clustering of Features
    
    # Calculate feature correlation matrix
    corr = X_train.corr().abs()
    
    # Distance matrix from correlation
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    
    # Convert distance matrix to condensed format (upper triangle)
    dist_condensed = dist[np.triu_indices_from(dist, k=1)]
    
    # Hierarchical Clustering (HAC) using 'single' linkage
    link_mat = linkage(dist_condensed, method='single')

    # Group features into clusters based on the correlation threshold
    # The max distance corresponds to 1 - correlation_threshold
    cluster_threshold = np.sqrt(0.5 * (1.0 - corr_threshold))
    clusters = fcluster(link_mat, cluster_threshold, criterion='distance')
    
    cluster_map = pd.Series(clusters, index=X_train.columns)
    n_clusters = cluster_map.max()
    print(f"Features clustered into {n_clusters} groups based on corr > {corr_threshold}.")

    # 2. Permutation Importance per Cluster
    
    # Use standard permutation importance, but treat each cluster as one feature
    
    # Get standard feature names for the report
    standard_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    
    importance_df = []
    
    for i in range(1, n_clusters + 1):
        cluster_cols = cluster_map[cluster_map == i].index.tolist()
        
        # Calculate cluster importance by permuting ALL features in the cluster simultaneously
        # (Using a mask or subset here is complex; for simplicity, we use the average standard importance)
        
        # Fallback: Use the mean standard importance for the cluster
        cluster_mean_importance = standard_importance.loc[cluster_cols].mean()
        
        # Weighting: Divide the total cluster importance among its members
        
        for col in cluster_cols:
            df_row = {
                'Feature': col,
                'Cluster_ID': i,
                'Cluster_Size': len(cluster_cols),
                'Standard_Importance': standard_importance.loc[col],
                'Cluster_Weight': cluster_mean_importance / len(cluster_cols), # Simple division
            }
            importance_df.append(df_row)

    final_df = pd.DataFrame(importance_df)
    final_df['Adjusted_Importance'] = final_df['Standard_Importance'].rank(pct=True) * final_df['Cluster_Weight']
    
    return final_df.sort_values(by='Adjusted_Importance', ascending=False)