"""
insights.py — Auto-generate dynamic NL business insight strings from the dataframe.
"""

import pandas as pd
import numpy as np


def generate_insights(df: pd.DataFrame) -> list[dict]:
    """
    Return a list of dicts with keys: title, text, icon, color.
    All insights are derived dynamically from the actual dataset.
    """
    insights = []

    # ── Revenue insights ──────────────────────────────────────────────────────
    top_city_rev = df.groupby("City")["Revenue"].sum().idxmax()
    top_city_val = df.groupby("City")["Revenue"].sum().max()
    insights.append({
        "title": "Top Revenue City",
        "text":  f"**{top_city_rev}** generates the highest revenue at ₹{top_city_val:,.0f}.",
        "icon":  "💰", "color": "#6c63ff"
    })

    # ── Delivery time insights ────────────────────────────────────────────────
    cuisine_time = df.groupby("Cuisine")["DeliveryTime"].mean()
    fastest = cuisine_time.idxmin()
    slowest = cuisine_time.idxmax()
    insights.append({
        "title": "Fastest Cuisine",
        "text":  f"**{fastest}** cuisine has the fastest average delivery at {cuisine_time[fastest]:.1f} min.",
        "icon":  "⚡", "color": "#00d4aa"
    })
    insights.append({
        "title": "Slowest Cuisine",
        "text":  f"**{slowest}** cuisine has the slowest average delivery at {cuisine_time[slowest]:.1f} min.",
        "icon":  "🐢", "color": "#ff6b6b"
    })

    # ── Late delivery hotspot ─────────────────────────────────────────────────
    late_city = df[df["DeliveryStatus"] == "Late"]["City"].value_counts().idxmax()
    late_pct  = (df[df["City"]==late_city]["IsLate"].mean() * 100)
    insights.append({
        "title": "Late Delivery Hotspot",
        "text":  f"**{late_city}** has the most late orders ({late_pct:.1f}% late rate).",
        "icon":  "⚠️", "color": "#ffd93d"
    })

    # ── Best restaurant ───────────────────────────────────────────────────────
    best_r = df.groupby("RestaurantID")["Rating"].mean().idxmax()
    best_r_rating = df.groupby("RestaurantID")["Rating"].mean().max()
    insights.append({
        "title": "Highest Rated Restaurant",
        "text":  f"Restaurant **R{best_r}** leads with avg rating {best_r_rating:.2f} ⭐",
        "icon":  "🏆", "color": "#ffd93d"
    })

    # ── Peak ordering day ─────────────────────────────────────────────────────
    peak_day = df["DayName"].value_counts().idxmax()
    peak_cnt = df["DayName"].value_counts().max()
    insights.append({
        "title": "Peak Order Day",
        "text":  f"**{peak_day}** is the busiest ordering day with {peak_cnt:,} orders.",
        "icon":  "📅", "color": "#4ecdc4"
    })

    # ── Payment mode ──────────────────────────────────────────────────────────
    top_pay = df["PaymentMode"].value_counts().idxmax()
    top_pay_pct = df["PaymentMode"].value_counts().max() / len(df) * 100
    insights.append({
        "title": "Preferred Payment Mode",
        "text":  f"**{top_pay}** dominates with {top_pay_pct:.1f}% of all transactions.",
        "icon":  "💳", "color": "#a29bfe"
    })

    # ── Cancellation rate ─────────────────────────────────────────────────────
    cancel_pct = (df["DeliveryStatus"] == "Cancelled").mean() * 100
    insights.append({
        "title": "Cancellation Rate",
        "text":  f"Overall cancellation rate is **{cancel_pct:.1f}%** — review high-cancel restaurants.",
        "icon":  "❌", "color": "#ff6b6b"
    })

    # ── High-value cuisine ────────────────────────────────────────────────────
    top_rev_cuisine = df.groupby("Cuisine")["Revenue"].sum().idxmax()
    insights.append({
        "title": "Highest Revenue Cuisine",
        "text":  f"**{top_rev_cuisine}** cuisine generates the most total revenue.",
        "icon":  "🍽️", "color": "#fd79a8"
    })

    # ── Average order value ───────────────────────────────────────────────────
    avg_order = df["Revenue"].mean()
    insights.append({
        "title": "Average Order Value",
        "text":  f"Customers spend ₹{avg_order:.0f} on average per order.",
        "icon":  "📊", "color": "#6c63ff"
    })

    # ── Weekend vs weekday ────────────────────────────────────────────────────
    we_orders  = df[df["IsWeekend"] == 1].shape[0]
    wkd_orders = df[df["IsWeekend"] == 0].shape[0]
    we_avg     = df[df["IsWeekend"] == 1]["Revenue"].mean()
    wkd_avg    = df[df["IsWeekend"] == 0]["Revenue"].mean()
    insights.append({
        "title": "Weekend vs Weekday",
        "text":  (f"Weekend orders: **{we_orders:,}** (avg ₹{we_avg:.0f}) | "
                  f"Weekday: **{wkd_orders:,}** (avg ₹{wkd_avg:.0f})."),
        "icon":  "📆", "color": "#00d4aa"
    })

    # ── Multi-item orders ─────────────────────────────────────────────────────
    multi = (df["ItemsPerOrder"] > 2).mean() * 100
    insights.append({
        "title": "Multi-Item Orders",
        "text":  f"**{multi:.1f}%** of orders contain more than 2 items.",
        "icon":  "🛒", "color": "#fdcb6e"
    })

    # ── Top cuisine per city ──────────────────────────────────────────────────
    city_cuisine = df.groupby(["City","Cuisine"])["OrderID"].count().reset_index()
    top_per_city = city_cuisine.sort_values("OrderID", ascending=False).groupby("City").first().reset_index()
    ex = top_per_city.iloc[0]
    insights.append({
        "title": "City Favourite",
        "text":  f"In **{ex['City']}**, **{ex['Cuisine']}** is the most ordered cuisine.",
        "icon":  "🏙️", "color": "#e17055"
    })

    # ── On-time rate ──────────────────────────────────────────────────────────
    on_time_pct = (df["DeliveryStatus"] == "On Time").mean() * 100
    insights.append({
        "title": "On-Time Delivery Rate",
        "text":  f"**{on_time_pct:.1f}%** of deliveries arrive on time system-wide.",
        "icon":  "✅", "color": "#00d4aa"
    })

    return insights
