"""
data_loader.py — Cached dataset loading and preprocessing
Loads Final_New_dataset.csv and returns a clean, enriched DataFrame.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

# ── Path configuration — works locally AND on Streamlit Cloud ────────────────
def _find_csv():
    csv_name = "Final_New_dataset.csv"
    base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # FoodDeliveryAI/
    candidates = [
        os.path.join(base, "data",  csv_name),   # FoodDeliveryAI/data/ (cloud)
        os.path.join(base,          csv_name),    # FoodDeliveryAI/ (fallback)
        os.path.dirname(base) + os.sep + csv_name,  # StreamLit/ (local dev)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {csv_name}. Checked: {candidates}")

DATA_PATH = _find_csv()


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load and clean the food delivery dataset.
    Returns a fully preprocessed DataFrame with derived columns.
    """
    df = pd.read_csv(DATA_PATH)

    # ── Column cleanup ────────────────────────────────────────────────────────
    df.columns = df.columns.str.strip()

    # ── Parse dates ──────────────────────────────────────────────────────────
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], dayfirst=True, errors="coerce")

    # ── Numeric coercion ─────────────────────────────────────────────────────
    for col in ["Age", "Rating", "Amount", "DeliveryTime", "ItemsPerOrder"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Derived date features ────────────────────────────────────────────────
    df["Month"]      = df["OrderDate"].dt.month
    df["MonthName"]  = df["OrderDate"].dt.strftime("%b")
    df["Quarter"]    = df["OrderDate"].dt.quarter
    df["DayOfWeek"]  = df["OrderDate"].dt.dayofweek          # 0 = Mon
    df["DayName"]    = df["OrderDate"].dt.strftime("%a")
    df["IsWeekend"]  = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Year"]       = df["OrderDate"].dt.year
    df["Week"]       = df["OrderDate"].dt.isocalendar().week.astype(int)

    # ── Revenue proxy ────────────────────────────────────────────────────────
    df["Revenue"] = df["Amount"].fillna(0)

    # ── Delivery flag ─────────────────────────────────────────────────────────
    df["IsLate"]       = (df["DeliveryStatus"] == "Late").astype(int)
    df["IsCancelled"]  = (df["DeliveryStatus"] == "Cancelled").astype(int)
    df["IsOnTime"]     = (df["DeliveryStatus"] == "On Time").astype(int)

    # ── Drop rows with critical nulls ─────────────────────────────────────────
    df.dropna(subset=["OrderID", "CustomerID", "DeliveryStatus"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_kpis(df: pd.DataFrame) -> dict:
    """Compute top-level KPIs from the dataframe."""
    total_orders     = len(df)
    total_customers  = df["CustomerID"].nunique()
    total_restaurants= df["RestaurantID"].nunique()
    avg_delivery     = df["DeliveryTime"].mean()
    total_revenue    = df["Revenue"].sum()
    late_pct         = (df["IsLate"].sum() / total_orders * 100)
    top_cuisine      = df["Cuisine"].value_counts().idxmax()
    best_rest        = (df.groupby("RestaurantID")["Rating"]
                         .mean().idxmax())

    return {
        "Total Orders":       f"{total_orders:,}",
        "Total Customers":    f"{total_customers:,}",
        "Total Restaurants":  f"{total_restaurants:,}",
        "Avg Delivery Time":  f"{avg_delivery:.1f} min",
        "Total Revenue":      f"₹{total_revenue:,.0f}",
        "Late Deliveries %":  f"{late_pct:.1f}%",
        "Top Cuisine":        top_cuisine,
        "Best Restaurant":    f"R{best_rest}",
    }
