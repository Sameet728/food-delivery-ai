"""
recommendation.py — Association-rule based food recommendation engine.
"""

import pandas as pd
import numpy as np
from typing import Optional


def get_recommendations(rules: pd.DataFrame, item: str,
                        top_n: int = 5) -> pd.DataFrame:
    """
    Given a food item, return top-N recommended companion items
    based on association rules (sorted by lift).
    """
    if rules is None or rules.empty:
        return pd.DataFrame()

    # Find rules where the item is in antecedents
    mask = rules["antecedents"].apply(lambda x: item in x)
    matching = rules[mask].copy()

    if matching.empty:
        # Fallback: search consequents
        mask2 = rules["consequents"].apply(lambda x: item in x)
        matching = rules[mask2].copy()

    if matching.empty:
        return pd.DataFrame()

    # Explode consequents to individual items
    rows = []
    for _, row in matching.iterrows():
        for cons_item in row["consequents"]:
            if cons_item != item:
                rows.append({
                    "Item":       cons_item,
                    "Support":    round(row["support"],    4),
                    "Confidence": round(row["confidence"], 4),
                    "Lift":       round(row["lift"],       4),
                })

    if not rows:
        return pd.DataFrame()

    result = (pd.DataFrame(rows)
                .sort_values("Lift", ascending=False)
                .drop_duplicates("Item")
                .head(top_n)
                .reset_index(drop=True))
    return result


def get_popular_combinations(freq_items: pd.DataFrame,
                             min_length: int = 2,
                             top_n: int = 10) -> pd.DataFrame:
    """Return most frequent itemsets with ≥ min_length items."""
    if freq_items is None or freq_items.empty:
        return pd.DataFrame()

    multi = freq_items[freq_items["length"] >= min_length].copy()
    multi["Items"] = multi["itemsets"].apply(lambda x: ", ".join(sorted(x)))
    return (multi[["Items", "support", "length"]]
            .sort_values("support", ascending=False)
            .head(top_n)
            .reset_index(drop=True))


def recommend_restaurants(df: pd.DataFrame, cuisine: str,
                          max_budget: int, min_rating: float,
                          top_n: int = 8) -> pd.DataFrame:
    """
    Recommend restaurants based on cuisine preference, budget, and rating.
    Returns a DataFrame with restaurant details.
    """
    fdf = df.copy()
    if cuisine and cuisine != "All":
        fdf = fdf[fdf["Cuisine"] == cuisine]
    if min_rating > 0:
        fdf = fdf[fdf["Rating"] >= min_rating]
    if max_budget > 0:
        fdf = fdf[fdf["Revenue"] <= max_budget]

    if fdf.empty:
        return pd.DataFrame()

    rest_df = (fdf.groupby("RestaurantID")
                  .agg(
                      AvgRating=("Rating",   "mean"),
                      TotalOrders=("OrderID", "count"),
                      AvgRevenue=("Revenue",  "mean"),
                      Cuisine=("Cuisine",    lambda x: x.mode()[0]),
                      OnTimePct=("IsOnTime",  "mean"),
                  )
                  .reset_index())

    rest_df["Score"] = (0.5 * rest_df["AvgRating"] / 5 +
                        0.3 * rest_df["OnTimePct"] +
                        0.2 * (rest_df["TotalOrders"] /
                               rest_df["TotalOrders"].max()))

    result = (rest_df.sort_values("Score", ascending=False)
                     .head(top_n)
                     .reset_index(drop=True))
    result["RestaurantLabel"] = "Restaurant " + result["RestaurantID"].astype(str)
    result["AvgRating"]  = result["AvgRating"].round(2)
    result["AvgRevenue"] = result["AvgRevenue"].round(0)
    result["OnTimePct"]  = (result["OnTimePct"] * 100).round(1)
    return result


def recommend_items_by_budget(df: pd.DataFrame, budget: int,
                              cuisine: Optional[str] = None,
                              top_n: int = 10) -> pd.DataFrame:
    """
    Recommend individual food items within budget, optionally filtered by cuisine.
    """
    fdf = df.copy()
    if cuisine and cuisine != "All":
        fdf = fdf[fdf["Cuisine"] == cuisine]

    # Parse item names & prices
    rows = []
    for _, row in fdf.iterrows():
        if not isinstance(row["ItemName"], str):
            continue
        items = [i.strip() for i in row["ItemName"].split(",")]
        for item in items:
            rows.append({"Item": item, "Cuisine": row["Cuisine"],
                         "Rating": row["Rating"]})

    if not rows:
        return pd.DataFrame()

    item_df = pd.DataFrame(rows)
    summary = (item_df.groupby("Item")
                      .agg(Count=("Item","count"),
                           AvgRating=("Rating","mean"))
                      .reset_index()
                      .sort_values("Count", ascending=False)
                      .head(top_n)
                      .reset_index(drop=True))
    summary["AvgRating"] = summary["AvgRating"].round(2)
    return summary
