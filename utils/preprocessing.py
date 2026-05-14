"""
preprocessing.py — Feature engineering & encoding helpers
Extracts clean ML-ready features from the raw DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ── Categorical columns used in classification ────────────────────────────────
CAT_COLS   = ["City", "Cuisine", "PaymentMode", "Gender"]
TARGET_COL = "DeliveryStatus"

# Ordered label map for delivery status
STATUS_MAP = {"On Time": 0, "Late": 1, "Cancelled": 2}
STATUS_INV = {v: k for k, v in STATUS_MAP.items()}


def encode_categoricals(df: pd.DataFrame):
    """
    Label-encode categorical columns.
    Returns: (encoded_df, encoders_dict)
    """
    encoders = {}
    df_enc = df.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        df_enc[col + "_enc"] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


def encode_new_row(row: dict, encoders: dict) -> dict:
    """Encode a single input row using fitted encoders."""
    enc_row = {}
    for col, le in encoders.items():
        val = row.get(col, "")
        if val in le.classes_:
            enc_row[col + "_enc"] = le.transform([val])[0]
        else:
            enc_row[col + "_enc"] = 0   # fallback to first class
    return enc_row


def build_classification_features(df: pd.DataFrame, encoders: dict):
    """
    Build feature matrix X and target y for delivery-status classification.
    """
    df_enc, _ = encode_categoricals(df) if not encoders else (df.copy(), encoders)

    # Apply existing encoders if provided
    if encoders:
        for col, le in encoders.items():
            df_enc[col + "_enc"] = le.transform(df_enc[col].astype(str))

    feature_cols = [
        "Age", "Rating", "ItemsPerOrder",
        "City_enc", "Cuisine_enc", "PaymentMode_enc", "Gender_enc",
    ]
    df_enc = df_enc.dropna(subset=feature_cols + [TARGET_COL])
    X = df_enc[feature_cols].values
    y = df_enc[TARGET_COL].map(STATUS_MAP).values
    return X, y, feature_cols


def build_ann_features(df: pd.DataFrame, encoders: dict):
    """
    Build feature matrix and target for ANN delivery-time regression.
    """
    df_enc = df.copy()
    for col, le in encoders.items():
        vals = df_enc[col].astype(str)
        known = vals.isin(le.classes_)
        df_enc[col + "_enc"] = 0
        df_enc.loc[known, col + "_enc"] = le.transform(vals[known])

    feature_cols = [
        "Age", "Rating", "ItemsPerOrder",
        "City_enc", "Cuisine_enc", "PaymentMode_enc",
    ]
    df_enc = df_enc.dropna(subset=feature_cols + ["DeliveryTime"])
    X = df_enc[feature_cols].values
    y = df_enc["DeliveryTime"].values
    return X, y, feature_cols


def build_clustering_features(df: pd.DataFrame):
    """
    Build features for customer clustering:
    total spend, order count, avg rating, avg delivery time.
    """
    customer_df = (
        df.groupby("CustomerID")
        .agg(
            TotalSpend=("Revenue", "sum"),
            OrderCount=("OrderID", "count"),
            AvgRating=("Rating", "mean"),
            AvgDeliveryTime=("DeliveryTime", "mean"),
            UniqueCuisines=("Cuisine", "nunique"),
        )
        .reset_index()
    )
    customer_df.fillna(0, inplace=True)
    feature_cols = ["TotalSpend", "OrderCount", "AvgRating",
                    "AvgDeliveryTime", "UniqueCuisines"]
    return customer_df, feature_cols


def get_train_test(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def get_scaler(X_train: np.ndarray):
    """Fit and return a StandardScaler."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
