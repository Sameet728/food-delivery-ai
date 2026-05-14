"""
model_trainer.py — Train all ML models once and persist to disk.

On first run: trains Logistic Regression, Decision Tree, Random Forest, SVM,
KMeans, ANN, and mines association rules from the dataset.
On subsequent runs: loads pre-trained artifacts from models/ directory.

All models use st.cache_resource so they are loaded once per server session.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CLF_PATH     = os.path.join(MODELS_DIR, "delivery_classifier.pkl")
CLUSTER_PATH = os.path.join(MODELS_DIR, "clustering_model.pkl")
ANN_PATH     = os.path.join(MODELS_DIR, "ann_model.keras")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
ENC_PATH     = os.path.join(MODELS_DIR, "encoders.pkl")
RULES_PATH   = os.path.join(MODELS_DIR, "association_rules.pkl")
ANN_HIST_PATH= os.path.join(MODELS_DIR, "ann_history.pkl")


# ── Classification ────────────────────────────────────────────────────────────
def _train_classifiers(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM":                 SVC(probability=True, kernel="rbf", random_state=42),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        Xtr = X_tr_s if name in ("Logistic Regression", "SVM") else X_train
        Xte = X_te_s if name in ("Logistic Regression", "SVM") else X_test
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }
        trained[name] = model

    return trained, results, scaler


# ── Clustering ────────────────────────────────────────────────────────────────
def _train_clustering(customer_features: np.ndarray, k: int = 4):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_s = scaler.fit_transform(customer_features)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_s)

    agg = AgglomerativeClustering(n_clusters=k)
    agg.fit(X_s)

    return km, agg, scaler


# ── ANN ───────────────────────────────────────────────────────────────────────
def _train_ann(X_train, y_train, X_test, y_test):
    try:
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(42)
        np.random.seed(42)

        # Normalise targets
        y_mean, y_std = y_train.mean(), y_train.std()
        y_tr_n = (y_train - y_mean) / y_std
        y_te_n = (y_test  - y_mean) / y_std

        model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64,  activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32,  activation="relu"),
            keras.layers.Dense(1),
        ])

        model.compile(optimizer=keras.optimizers.Adam(0.001),
                      loss="mse", metrics=["mae"])

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        history = model.fit(
            X_train, y_tr_n,
            validation_data=(X_test, y_te_n),
            epochs=60, batch_size=64,
            callbacks=[early_stop],
            verbose=0,
        )
        return model, history.history, y_mean, y_std
    except Exception as e:
        return None, {}, 0, 1


# ── Association Rules ─────────────────────────────────────────────────────────
def _mine_association_rules(df: pd.DataFrame):
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

    # Build basket: each order → set of items ordered
    def parse_items(s):
        if isinstance(s, str):
            return [i.strip() for i in s.split(",")]
        return []

    df["ItemList"] = df["ItemName"].apply(parse_items)

    # One-hot encode
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_arr = te.fit_transform(df["ItemList"].tolist())
    basket = pd.DataFrame(te_arr, columns=te.columns_)

    # Frequent itemsets (lower support for richer rules)
    freq_items = fpgrowth(basket, min_support=0.05, use_colnames=True)
    freq_items["length"] = freq_items["itemsets"].apply(len)

    rules = association_rules(freq_items, metric="lift", min_threshold=1.0,
                              num_itemsets=len(freq_items))
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return freq_items, rules


# ── Master trainer ─────────────────────────────────────────────────────────────
def train_and_save_all(df: pd.DataFrame, progress_callback=None):
    """
    Train all models from scratch and save to disk.
    Call this only when model files are missing.
    """
    from utils.preprocessing import (
        encode_categoricals, build_classification_features,
        build_clustering_features, build_ann_features,
        get_train_test, get_scaler,
    )

    def _cb(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    _cb("🔧 Encoding features...", 0.05)
    df_enc, encoders = encode_categoricals(df)
    joblib.dump(encoders, ENC_PATH)

    # ── Classification ────────────────────────────────────────────────────────
    _cb("🌲 Training classifiers...", 0.15)
    X, y, _ = build_classification_features(df, encoders)
    X_tr, X_te, y_tr, y_te = get_train_test(X, y)
    clf_models, clf_results, clf_scaler = _train_classifiers(X_tr, y_tr, X_te, y_te)
    joblib.dump({"models": clf_models, "results": clf_results,
                 "scaler": clf_scaler, "X_test": X_te, "y_test": y_te},
                CLF_PATH)

    # ── Scaler (shared for ANN) ───────────────────────────────────────────────
    _cb("⚖️  Fitting scaler...", 0.35)
    X_ann, y_ann, _ = build_ann_features(df, encoders)
    from sklearn.preprocessing import StandardScaler
    ann_scaler = StandardScaler()
    ann_scaler.fit(X_ann)
    joblib.dump(ann_scaler, SCALER_PATH)

    X_ann_s = ann_scaler.transform(X_ann)
    X_ann_tr, X_ann_te, y_ann_tr, y_ann_te = get_train_test(X_ann_s, y_ann)

    # ── ANN ───────────────────────────────────────────────────────────────────
    _cb("🧠 Training ANN...", 0.45)
    ann_model, ann_hist, y_mean, y_std = _train_ann(
        X_ann_tr, y_ann_tr, X_ann_te, y_ann_te
    )
    if ann_model is not None:
        ann_model.save(ANN_PATH)
        joblib.dump({"history": ann_hist, "y_mean": y_mean, "y_std": y_std},
                    ANN_HIST_PATH)

    # ── Clustering ────────────────────────────────────────────────────────────
    _cb("🔵 Training clustering...", 0.70)
    cust_df, feat_cols = build_clustering_features(df)
    km, agg, cl_scaler = _train_clustering(cust_df[feat_cols].values)
    joblib.dump({"kmeans": km, "agglomerative": agg,
                 "scaler": cl_scaler, "customer_df": cust_df,
                 "feature_cols": feat_cols},
                CLUSTER_PATH)

    # ── Association rules ─────────────────────────────────────────────────────
    _cb("🔗 Mining association rules...", 0.85)
    freq_items, rules = _mine_association_rules(df)
    joblib.dump({"freq_items": freq_items, "rules": rules}, RULES_PATH)

    _cb("✅ All models trained!", 1.0)
    return True


# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_classifiers():
    if not os.path.exists(CLF_PATH):
        return None
    return joblib.load(CLF_PATH)


@st.cache_resource(show_spinner=False)
def load_clustering():
    if not os.path.exists(CLUSTER_PATH):
        return None
    return joblib.load(CLUSTER_PATH)


@st.cache_resource(show_spinner=False)
def load_ann():
    if not os.path.exists(ANN_PATH):
        return None, None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(ANN_PATH)
        hist  = joblib.load(ANN_HIST_PATH) if os.path.exists(ANN_HIST_PATH) else {}
        return model, hist
    except Exception:
        return None, None


@st.cache_resource(show_spinner=False)
def load_encoders():
    if not os.path.exists(ENC_PATH):
        return None
    return joblib.load(ENC_PATH)


@st.cache_resource(show_spinner=False)
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        return None
    return joblib.load(SCALER_PATH)


@st.cache_resource(show_spinner=False)
def load_rules():
    if not os.path.exists(RULES_PATH):
        return None, None
    data = joblib.load(RULES_PATH)
    return data["freq_items"], data["rules"]


def models_exist() -> bool:
    """Return True only if all required model files are present."""
    return all(os.path.exists(p) for p in [
        CLF_PATH, CLUSTER_PATH, SCALER_PATH, ENC_PATH, RULES_PATH
    ])
