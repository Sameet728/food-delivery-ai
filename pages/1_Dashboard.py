"""
1_Dashboard.py — Executive KPI Dashboard with full sidebar filters.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils.helper import inject_css, page_header, sidebar_filters, download_csv
from utils.data_loader import load_data, get_kpis
import utils.visualization as viz

st.set_page_config(page_title="Dashboard | FoodDelivery AI",
                   page_icon="🏠", layout="wide")
inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Dashboard")
    st.markdown("Executive overview of all key metrics.")
    st.markdown("---")

df_raw = load_data()
df     = sidebar_filters(df_raw)

page_header("Executive Dashboard",
            "Real-time analytics across orders, revenue, and delivery performance.",
            "🏠")

if df.empty:
    st.warning("No data matches the current filters. Try adjusting the sidebar filters.")
    st.stop()

# ── KPI Row ────────────────────────────────────────────────────────────────────
kpis_raw = get_kpis(df)
kpi_items = [
    ("📦 Orders",        kpis_raw["Total Orders"]),
    ("👥 Customers",     kpis_raw["Total Customers"]),
    ("🏪 Restaurants",   kpis_raw["Total Restaurants"]),
    ("⏱️ Avg Delivery",  kpis_raw["Avg Delivery Time"]),
    ("💰 Revenue",       kpis_raw["Total Revenue"]),
    ("⚠️ Late %",        kpis_raw["Late Deliveries %"]),
    ("🍽️ Top Cuisine",   kpis_raw["Top Cuisine"]),
    ("⭐ Best Rest.",    kpis_raw["Best Restaurant"]),
]

cols = st.columns(8)
for col, (label, value) in zip(cols, kpi_items):
    col.metric(label, value)

st.markdown("<br>", unsafe_allow_html=True)


# ── Row 1: Revenue trend + Orders by city ─────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(viz.revenue_trend_chart(df), width='stretch')
with col2:
    st.plotly_chart(viz.orders_by_city_chart(df), width='stretch')

# ── Row 2: Cuisine donut + Delivery status + Payment mode ────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(viz.cuisine_donut(df), width='stretch')
with col2:
    st.plotly_chart(viz.delivery_status_chart(df), width='stretch')
with col3:
    st.plotly_chart(viz.payment_mode_chart(df), width='stretch')

# ── Row 3: Top restaurants + City heatmap ────────────────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    st.plotly_chart(viz.top_restaurants_chart(df), width='stretch')
with col2:
    st.plotly_chart(viz.city_cuisine_heatmap(df), width='stretch')

# ── Monthly trend table ───────────────────────────────────────────────────────
import pandas as pd
st.markdown("### 📅 Monthly Order Summary")
monthly = (df.groupby(["Year","MonthName"])
             .agg(Orders=("OrderID","count"),
                  Revenue=("Revenue","sum"),
                  AvgDeliveryTime=("DeliveryTime","mean"),
                  LateOrders=("IsLate","sum"))
             .reset_index()
             .sort_values(["Year","MonthName"]))
monthly["Revenue"] = monthly["Revenue"].map("₹{:,.0f}".format)
monthly["AvgDeliveryTime"] = monthly["AvgDeliveryTime"].map("{:.1f} min".format)
st.dataframe(monthly, width='stretch', hide_index=True)

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("---")
download_csv(df, "dashboard_data.csv", "📥 Download Filtered Data")
