"""
clustering.py — Customer clustering utilities.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Cluster label mapping (sorted by total spend)
CLUSTER_NAMES = {
    0: "💎 Premium",
    1: "🔄 Frequent",
    2: "💸 Occasional",
    3: "🟢 New / Low",
}


def assign_cluster_names(labels: np.ndarray,
                         cust_df: pd.DataFrame) -> dict:
    """
    Assign meaningful names to clusters based on average spend.
    Returns a mapping {cluster_id: name_str}.
    """
    cust_df = cust_df.copy()
    cust_df["Cluster"] = labels
    avg_spend = cust_df.groupby("Cluster")["TotalSpend"].mean().sort_values(ascending=False)
    names_ordered = ["💎 Premium", "🔄 Frequent", "💸 Occasional", "🟢 New / Low"]
    mapping = {}
    for i, cluster_id in enumerate(avg_spend.index):
        mapping[cluster_id] = names_ordered[min(i, len(names_ordered)-1)]
    return mapping


def pca_project(X_scaled: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project scaled feature matrix to 2D using PCA."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X_scaled)


def get_cluster_profiles(cust_df: pd.DataFrame,
                         labels: np.ndarray,
                         cluster_names: dict) -> pd.DataFrame:
    """Return per-cluster statistics summary."""
    df = cust_df.copy()
    df["Cluster"] = labels
    df["ClusterName"] = df["Cluster"].map(cluster_names)

    profile = (df.groupby("ClusterName")
                 .agg(
                     CustomerCount=("CustomerID", "count"),
                     AvgSpend=("TotalSpend",        "mean"),
                     AvgOrders=("OrderCount",        "mean"),
                     AvgRating=("AvgRating",         "mean"),
                     AvgDelivery=("AvgDeliveryTime", "mean"),
                 )
                 .reset_index()
                 .round(2))
    return profile
