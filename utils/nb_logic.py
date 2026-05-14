"""
nb_logic.py — Single source of truth for all ML logic.
Mirrors the EXACT code from Part1–Part7 notebooks:
  same features, same random_state=42, same splits, same params.

All heavy functions are cached with st.cache_data / st.cache_resource
so they train once and reload instantly on every page.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Dataset path — works locally AND on Streamlit Cloud ──────────────────────
# Priority: data/ subfolder (cloud/repo) → parent dirs (local dev)
def _find_csv() -> Path:
    csv_name = "Final_New_dataset.csv"
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / csv_name,   # FoodDeliveryAI/data/ (cloud)
        Path(__file__).resolve().parent.parent / csv_name,              # FoodDeliveryAI/ (fallback)
        Path(__file__).resolve().parent.parent.parent / csv_name,       # StreamLit/ (local dev)
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {csv_name}. Checked: {[str(c) for c in candidates]}")

DATA_PATH = _find_csv()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_preprocess() -> pd.DataFrame:
    """
    Mirrors Part1.ipynb preprocessing exactly.
    Returns the fully processed DataFrame (30,000 × 22+).
    """
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Parse dates
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], dayfirst=True, errors="coerce")

    # Rename Amount → TotalOrderValue (mirrors Part6/Part7)
    if "Amount" in df.columns and "TotalOrderValue" not in df.columns:
        df.rename(columns={"Amount": "TotalOrderValue"}, inplace=True)

    # Numeric coercion
    for col in ["Age", "Rating", "TotalOrderValue", "DeliveryTime", "ItemsPerOrder"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived date features (Part1 Cell 25 style)
    df["order_month"]     = df["OrderDate"].dt.month
    df["order_dayofweek"] = df["OrderDate"].dt.dayofweek
    df["order_day"]       = df["OrderDate"].dt.day
    df["YearMonth"]       = df["OrderDate"].dt.to_period("M")

    # Derived features (Part1 Cell 25)
    order_freq = (df.groupby("CustomerID")["OrderID"]
                    .count().reset_index()
                    .rename(columns={"OrderID": "OrderFrequency"}))
    df = df.merge(order_freq, on="CustomerID", how="left")

    avg_dt = (df.groupby("CustomerID")["DeliveryTime"]
                .mean().round(2).reset_index()
                .rename(columns={"DeliveryTime": "AvgDeliveryTime"}))
    df = df.merge(avg_dt, on="CustomerID", how="left")

    df["TotalOrderValue"] = df["TotalOrderValue"].fillna(0)
    df["IsLate"]      = (df["DeliveryStatus"] == "Late").astype(int)
    df["IsOnTime"]    = (df["DeliveryStatus"] == "On Time").astype(int)
    df["IsCancelled"] = (df["DeliveryStatus"] == "Cancelled").astype(int)

    df.dropna(subset=["OrderID", "CustomerID", "DeliveryStatus"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Classification Models
# Mirrors Part3.ipynb exactly: same FEATURES, same cat_cols encoding,
# same scaler, same train_test_split(test_size=0.2, random_state=42, stratify=y)
# ═══════════════════════════════════════════════════════════════════════════════

# Exact features from Part3 notebook
FEATURES_CLS = ["Age", "City", "Rating", "Cuisine", "ItemsPerOrder",
                "PaymentMode", "TotalOrderValue", "DeliveryTime",
                "order_dayofweek", "order_month", "Gender"]
TARGET_CLS   = "DeliveryStatus"
CAT_COLS_CLS = ["City", "Cuisine", "PaymentMode", "Gender"]
NUM_COLS_CLS = ["Age", "Rating", "ItemsPerOrder", "TotalOrderValue",
                "DeliveryTime", "order_dayofweek", "order_month"]


@st.cache_resource(show_spinner=False)
def train_classifiers():
    """
    Trains LR, DT, RF, SVM exactly as in Part3.ipynb.
    Returns (trained_dict, results_dict, le_target, encoders_dict, scaler, X_test, y_test)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)

    df = load_and_preprocess()
    data = df[FEATURES_CLS + [TARGET_CLS]].dropna().copy()

    # Encode categoricals (mirrors Part3 Cell 11)
    encoders = {}
    for col in CAT_COLS_CLS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    data[TARGET_CLS] = le_target.fit_transform(data[TARGET_CLS].astype(str))
    TARGET_CLASSES = list(le_target.classes_)

    X = data[FEATURES_CLS].copy()
    y = data[TARGET_CLS].values

    # Scale numerics (mirrors Part3)
    scaler = StandardScaler()
    X[NUM_COLS_CLS] = scaler.fit_transform(X[NUM_COLS_CLS])

    # Split — exact Part3 params
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Exact Part3 models & hyperparams
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM":                 SVC(probability=True, random_state=42),
    }

    trained = {}
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        results[name] = {
            "Accuracy":  round(accuracy_score(y_te, y_pred) * 100, 2),
            "Precision": round(precision_score(y_te, y_pred, average="weighted", zero_division=0) * 100, 2),
            "Recall":    round(recall_score(y_te, y_pred, average="weighted", zero_division=0) * 100, 2),
            "F1 Score":  round(f1_score(y_te, y_pred, average="weighted", zero_division=0) * 100, 2),
        }
        trained[name] = model

    return trained, results, le_target, encoders, scaler, X_te, y_te, TARGET_CLASSES


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Association Rules (mirrors Part2 + Part6 Cell 7)
# min_support=0.01, min_confidence=0.10, lift>=1.0
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def compute_association_rules():
    """
    Mirrors Part2.ipynb + Part6 association rules exactly.
    Returns (rules_df, freq_items_df, item_freq_dict)
    """
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules

    df = load_and_preprocess()

    # Build transactions (mirrors Part6 Cell 7)
    def build_transactions(df):
        records = []
        for _, row in df.drop_duplicates("OrderID").iterrows():
            raw   = str(row["ItemName"])
            items = [i.strip() for i in raw.split(",") if i.strip()]
            items = list(dict.fromkeys(items))   # remove dupes, keep order
            records.append({"OrderID": row["OrderID"], "Items": items})
        tx_df = pd.DataFrame(records)
        return tx_df["Items"].tolist(), tx_df

    all_txns, tx_df = build_transactions(df)
    multi_txns = [t for t in all_txns if len(t) >= 2]

    # Item frequency
    from collections import Counter
    flat = [item for txn in all_txns for item in txn]
    item_freq = dict(Counter(flat).most_common())

    # One-hot encode
    te = TransactionEncoder()
    te_arr = te.fit_transform(multi_txns)
    basket = pd.DataFrame(te_arr, columns=te.columns_)

    # Apriori — exact Part2 params
    freq_items = apriori(basket, min_support=0.01, use_colnames=True, max_len=3)
    freq_items["size"] = freq_items["itemsets"].apply(len)
    freq_items["itemsets_str"] = freq_items["itemsets"].apply(
        lambda x: ", ".join(sorted(x))
    )

    if freq_items.empty:
        return pd.DataFrame(), freq_items, item_freq

    try:
        rules = association_rules(
            freq_items, metric="confidence", min_threshold=0.10,
            num_itemsets=len(freq_items)
        )
    except TypeError:
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.10)

    rules = rules[rules["lift"] >= 0.5].copy()   # Part7 uses 0.5 — only 14 items, lift naturally < 1
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules["rule"] = rules["antecedents_str"] + " → " + rules["consequents_str"]
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    return rules, freq_items, item_freq, tx_df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Customer Clustering
# Mirrors Part4.ipynb exactly:
# Features: OrderFrequency, TotalSpending, AvgDeliveryTime, Cuisine_enc
# KMeans(n_clusters=3, random_state=42, n_init=10)
# Remap by avg spending: 0=High-Value, 1=Regular, 2=Occasional
# ═══════════════════════════════════════════════════════════════════════════════

CLUST_FEATS   = ["OrderFrequency", "TotalSpending", "AvgDeliveryTime", "Cuisine_enc"]
CLUSTER_LABELS = {0: "💰 High-Value", 1: "🔁 Regular", 2: "😐 Occasional"}
CLUSTER_COLORS = {0: "#FF6B6B",       1: "#4ECDC4",   2: "#A78BFA"}


@st.cache_resource(show_spinner=False)
def train_clustering():
    """
    Mirrors Part4.ipynb clustering exactly.
    Returns (cust_df, km_model, scaler, le_cuisine)
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering

    df = load_and_preprocess()

    # Aggregate per customer (mirrors Part4 Cell 11)
    preferred_cuisine = (df.groupby("CustomerID")["Cuisine"]
                           .agg(lambda x: x.value_counts().idxmax())
                           .reset_index()
                           .rename(columns={"Cuisine": "PreferredCuisine"}))

    avg_rating = (df.groupby("CustomerID")["Rating"]
                    .mean().reset_index()
                    .rename(columns={"Rating": "AvgRating"}))

    cust = (df.groupby("CustomerID")
              .agg(OrderFrequency  = ("OrderID",          "count"),
                   TotalSpending   = ("TotalOrderValue",  "sum"),
                   AvgDeliveryTime = ("DeliveryTime",     "mean"),
                   LateOrders      = ("DeliveryStatus",   lambda x: (x == "Late").sum()))
              .reset_index())

    cust = cust.merge(preferred_cuisine, on="CustomerID")
    cust = cust.merge(avg_rating, on="CustomerID")
    cust["LateRate"] = (cust["LateOrders"] / cust["OrderFrequency"] * 100).round(2)

    le_cuisine = LabelEncoder()
    cust["Cuisine_enc"] = le_cuisine.fit_transform(cust["PreferredCuisine"])

    # Scale & cluster (mirrors Part4 exactly)
    sc_cust = StandardScaler()
    X = sc_cust.fit_transform(cust[CLUST_FEATS])

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cust["Cluster_raw"] = km.fit_predict(X)

    # Remap by spending (mirrors Part4)
    spend_order = (cust.groupby("Cluster_raw")["TotalSpending"]
                       .mean().sort_values(ascending=False).index.tolist())
    remap = {spend_order[0]: 0, spend_order[1]: 1, spend_order[2]: 2}
    cust["Cluster"] = cust["Cluster_raw"].map(remap)
    cust["Segment"] = cust["Cluster"].map(CLUSTER_LABELS)

    # Agglomerative (mirrors Part4 Cell 8)
    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    cust["Hierarchical_raw"] = agg.fit_predict(X)
    # Align hierarchical labels to KMeans semantic labels
    cust["Hierarchical_Cluster"] = cust["Hierarchical_raw"].map(
        lambda hc: cust[cust["Hierarchical_raw"] == hc]["Cluster"].mode()[0]
    )

    return cust, km, sc_cust, le_cuisine


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — ANN Delivery Time Prediction
# Mirrors Part5.ipynb exactly:
# Features: ItemsPerOrder, Rating, City, Cuisine, PaymentMode
# ANN: Dense(64,relu) → Dense(32,relu) → Dense(16,relu) → Dense(1,linear)
# optimizer=adam, loss=mse, EarlyStopping(patience=5)
# split: test_size=0.2, random_state=42 (NO stratify — regression)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURES_ANN = ["ItemsPerOrder", "Rating", "City", "Cuisine", "PaymentMode"]
TARGET_ANN   = "DeliveryTime"
CAT_ANN      = ["City", "Cuisine", "PaymentMode"]
NUM_ANN      = ["ItemsPerOrder", "Rating"]


@st.cache_resource(show_spinner=False)
def train_ann():
    """
    Trains ANN exactly as Part5.ipynb.
    Returns (ann_model, history_dict, encoders, scaler, X_test, y_test, mae, rmse, r2)
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    tf.random.set_seed(42)
    np.random.seed(42)

    df = load_and_preprocess()
    data = df[FEATURES_ANN + [TARGET_ANN]].copy().dropna()

    # Encode (mirrors Part5 Cell 5)
    encoders = {}
    for col in CAT_ANN:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    X = data[FEATURES_ANN].values
    y = data[TARGET_ANN].values

    # Split (mirrors Part5 Cell 6 — no stratify for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale only NUM_ANN (mirrors Part5 Cell 7)
    NUM_IDX = [FEATURES_ANN.index(c) for c in NUM_ANN]
    scaler = StandardScaler()
    X_train[:, NUM_IDX] = scaler.fit_transform(X_train[:, NUM_IDX])
    X_test[:, NUM_IDX]  = scaler.transform(X_test[:, NUM_IDX])

    # ANN architecture (mirrors Part5 Cell 8 exactly)
    ann = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1,  activation="linear"),
    ], name="ANN_DeliveryTime")
    ann.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train (mirrors Part5 Cell 9)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = ann.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred = ann.predict(X_test, verbose=0).flatten()
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return ann, history.history, encoders, scaler, X_test, y_test, y_pred, mae, rmse, r2


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — Business Insights
# Restaurant scoring: Rating*0.40 + OnTimeRate*0.40 + (NormVolume*5)*0.20
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def compute_restaurant_insights():
    """Mirrors Part6 Cell 15 exactly."""
    df = load_and_preprocess()

    restaurant_perf = df.groupby("RestaurantID").agg(
        OrderVolume    = ("OrderID",          "count"),
        AvgRating      = ("Rating",           "mean"),
        AvgDeliveryMin = ("DeliveryTime",     "mean"),
        AvgOrderValue  = ("TotalOrderValue",  "mean"),
        TotalRevenue   = ("TotalOrderValue",  "sum"),
        OnTimeRate     = ("DeliveryStatus",   lambda x: (x == "On Time").mean() * 100),
        LateRate       = ("DeliveryStatus",   lambda x: (x == "Late").mean() * 100),
    ).round(2)

    restaurant_perf["Score"] = (
        restaurant_perf["AvgRating"]   * 0.40 +
        restaurant_perf["OnTimeRate"]  * 0.40 +
        (restaurant_perf["OrderVolume"] /
         restaurant_perf["OrderVolume"].max() * 5) * 0.20
    ).round(3)

    top_restaurants = (restaurant_perf
                       .sort_values("Score", ascending=False)
                       .head(10)
                       .reset_index())
    return restaurant_perf, top_restaurants


@st.cache_data(show_spinner=False)
def compute_delivery_bottlenecks():
    """Mirrors Part6 Cell 19 — delivery bottleneck analysis."""
    df = load_and_preprocess()

    # By city
    city_stats = df.groupby("City").agg(
        AvgDelivery = ("DeliveryTime", "mean"),
        LateRate    = ("IsLate",       "mean"),
        Orders      = ("OrderID",      "count"),
    ).round(2).reset_index()
    city_stats["LateRate"] *= 100

    # By cuisine
    cuisine_stats = df.groupby("Cuisine").agg(
        AvgDelivery = ("DeliveryTime", "mean"),
        LateRate    = ("IsLate",       "mean"),
        Orders      = ("OrderID",      "count"),
    ).round(2).reset_index()
    cuisine_stats["LateRate"] *= 100

    # By day of week
    dow_stats = df.groupby("order_dayofweek").agg(
        AvgDelivery = ("DeliveryTime", "mean"),
        LateRate    = ("IsLate",       "mean"),
        Orders      = ("OrderID",      "count"),
    ).round(2).reset_index()
    dow_stats["LateRate"] *= 100
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_stats["Day"] = dow_stats["order_dayofweek"].map(lambda d: day_names[d])

    return city_stats, cuisine_stats, dow_stats


@st.cache_data(show_spinner=False)
def compute_customer_segment_insights():
    """Mirrors Part6 Cell 17 — customer segment KPIs."""
    cust, _, _, _ = train_clustering()
    seg_summary = cust.groupby("Segment").agg(
        Customers    = ("CustomerID",     "count"),
        Avg_Orders   = ("OrderFrequency", "mean"),
        Avg_Spending = ("TotalSpending",  "mean"),
        Avg_Delivery = ("AvgDeliveryTime","mean"),
        Avg_Rating   = ("AvgRating",      "mean"),
        Late_Rate    = ("LateRate",       "mean"),
    ).round(2).reset_index()
    return seg_summary


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — Regression Model (RF Regressor for delivery time)
# Mirrors Part6 Cell 13 exactly
# ═══════════════════════════════════════════════════════════════════════════════

FEATURES_REG = ["ItemsPerOrder", "Rating", "City", "Cuisine", "PaymentMode"]
TARGET_REG   = "DeliveryTime"
CAT_REG      = ["City", "Cuisine", "PaymentMode"]
NUM_REG_IDX  = [0, 1]   # ItemsPerOrder, Rating


@st.cache_resource(show_spinner=False)
def train_delivery_regressor():
    """Mirrors Part6 Cell 13 RF Regressor exactly."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = load_and_preprocess()
    data_reg = df[FEATURES_REG + [TARGET_REG]].copy().dropna()

    le_reg = {}
    for col in CAT_REG:
        le = LabelEncoder()
        data_reg[col] = le.fit_transform(data_reg[col].astype(str))
        le_reg[col] = le

    X_reg = data_reg[FEATURES_REG].values
    y_reg = data_reg[TARGET_REG].values

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    sc_reg = StandardScaler()
    X_tr_r[:, NUM_REG_IDX] = sc_reg.fit_transform(X_tr_r[:, NUM_REG_IDX])
    X_te_r[:, NUM_REG_IDX] = sc_reg.transform(X_te_r[:, NUM_REG_IDX])

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_tr_r, y_tr_r)
    y_pred_r = rf_reg.predict(X_te_r)

    mae  = mean_absolute_error(y_te_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_te_r, y_pred_r))
    r2   = r2_score(y_te_r, y_pred_r)

    feat_imp = pd.Series(rf_reg.feature_importances_, index=FEATURES_REG).sort_values(ascending=False)

    return rf_reg, le_reg, sc_reg, X_te_r, y_te_r, y_pred_r, mae, rmse, r2, feat_imp
