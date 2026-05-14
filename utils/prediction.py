"""
prediction.py — Inference wrappers for classification and ANN models.
"""

import numpy as np
import pandas as pd
from utils.preprocessing import STATUS_INV, CAT_COLS


def predict_delivery_status(models_data: dict, encoders: dict,
                            input_dict: dict) -> dict:
    """
    Run all classifiers on a single input row.
    Returns dict: {model_name: {label, proba}}
    """
    from utils.preprocessing import encode_new_row
    enc = encode_new_row(input_dict, encoders)

    feature_values = np.array([[
        float(input_dict.get("Age", 30)),
        float(input_dict.get("Rating", 3.5)),
        float(input_dict.get("ItemsPerOrder", 2)),
        enc.get("City_enc", 0),
        enc.get("Cuisine_enc", 0),
        enc.get("PaymentMode_enc", 0),
        enc.get("Gender_enc", 0),
    ]])

    results = {}
    scaler  = models_data.get("scaler")
    X_s     = scaler.transform(feature_values) if scaler else feature_values

    for name, model in models_data["models"].items():
        X_input = X_s if name in ("Logistic Regression", "SVM") else feature_values
        try:
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None
            results[name] = {
                "label": STATUS_INV.get(pred, str(pred)),
                "proba": proba.tolist() if proba is not None else None,
            }
        except Exception as e:
            results[name] = {"label": "Error", "proba": None}
    return results


def predict_delivery_time(ann_model, scaler, encoders: dict,
                          input_dict: dict,
                          y_mean: float, y_std: float) -> float:
    """
    Predict delivery time in minutes using the ANN model.
    """
    from utils.preprocessing import encode_new_row
    enc = encode_new_row(input_dict, encoders)

    feature_values = np.array([[
        float(input_dict.get("Age", 30)),
        float(input_dict.get("Rating", 3.5)),
        float(input_dict.get("ItemsPerOrder", 2)),
        enc.get("City_enc", 0),
        enc.get("Cuisine_enc", 0),
        enc.get("PaymentMode_enc", 0),
    ]])

    X_s = scaler.transform(feature_values)
    pred_norm = ann_model.predict(X_s, verbose=0)[0][0]
    pred_minutes = float(pred_norm * y_std + y_mean)
    return max(10.0, round(pred_minutes, 1))


def get_confusion_matrix(models_data: dict) -> dict:
    """Return confusion matrices for all models using the held-out test set."""
    from sklearn.metrics import confusion_matrix
    X_test = models_data.get("X_test")
    y_test = models_data.get("y_test")
    scaler = models_data.get("scaler")
    X_s    = scaler.transform(X_test) if scaler else X_test

    cms = {}
    for name, model in models_data["models"].items():
        X_input = X_s if name in ("Logistic Regression", "SVM") else X_test
        try:
            y_pred = model.predict(X_input)
            cm = confusion_matrix(y_test, y_pred)
            cms[name] = cm
        except Exception:
            cms[name] = np.zeros((3,3), dtype=int)
    return cms
