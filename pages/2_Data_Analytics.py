"""
2_Data_Analytics.py — EDA & Data Analytics
Mirrors Part1.ipynb preprocessing exactly via nb_logic.py.
No notebook execution required — loads live from CSV.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Data Analytics | Food Delivery AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0c29,#302b63,#24243e);}
[data-testid="stSidebar"] *{color:#e0e0e0!important;}
.hero{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);border-radius:18px;
      padding:2rem 2.5rem;margin-bottom:1.5rem;border:1px solid rgba(100,180,255,.15);}
.hero h1{color:#e2e8f0;font-size:2rem;margin:0 0 .4rem;}
.hero p{color:#94a3b8;margin:0;}
.kpi{background:rgba(255,255,255,.05);border-radius:14px;padding:1.2rem 1.5rem;
     border:1px solid rgba(255,255,255,.08);text-align:center;}
.kpi .val{color:#e2e8f0;font-size:1.8rem;font-weight:700;}
.kpi .lbl{color:#64748b;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📊 Data Analytics & EDA</h1>
  <p>Exploratory analysis mirroring Part1.ipynb — derived features, distributions, and encoding. Live from CSV.</p>
</div>
""", unsafe_allow_html=True)

from utils.nb_logic import load_and_preprocess

with st.spinner("📊 Loading and preprocessing data…"):
    df = load_and_preprocess()

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (f"{len(df):,}",                   "Total Rows"),
    (f"{df['CustomerID'].nunique():,}", "Unique Customers"),
    (f"{df['RestaurantID'].nunique():,}","Unique Restaurants"),
    (f"{df.isnull().sum().sum():,}",    "Missing Values"),
    (f"{len(df.columns):,}",            "Columns"),
]
for col, (val, lbl) in zip([c1,c2,c3,c4,c5], kpis):
    with col:
        st.markdown(f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Dataset", "📦 Orders", "🚚 Delivery", "🍕 Cuisine & Payment", "⚙️ Features"])

# ── Tab 1: Dataset overview (Part1 summary) ───────────────────────────────────
with tab1:
    with st.expander("🔍 Preview Dataset (first 100 rows)", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)

    with st.expander("📈 Column Info & Null Counts"):
        info_df = pd.DataFrame({
            "Column":   df.columns,
            "Dtype":    df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null":     df.isnull().sum().values,
            "Unique":   [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True)

    with st.expander("📊 Descriptive Statistics"):
        st.dataframe(df.describe().round(2), use_container_width=True)

# ── Tab 2: Order distributions (mirrors Part1 Cell 24) ───────────────────────
with tab2:
    st.markdown("**Order Frequency Distribution** (mirrors Part1 derived features)")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(df, x="OrderFrequency", nbins=30, title="Customer Order Frequency",
                           color_discrete_sequence=["#1abc9c"])
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df, x="TotalOrderValue", nbins=30, title="Order Amount Distribution (₹)",
                           color_discrete_sequence=["#f39c12"])
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(df, x="AvgDeliveryTime", nbins=30, title="Avg Delivery Time per Customer (min)",
                        color_discrete_sequence=["#e74c3c"])
    fig2.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                       yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
    st.plotly_chart(fig2, use_container_width=True)

    # Monthly order trend (Part1 style)
    monthly = df.groupby("YearMonth").agg(Orders=("OrderID","count")).reset_index()
    monthly["YearMonth_dt"] = monthly["YearMonth"].dt.to_timestamp()
    monthly = monthly.sort_values("YearMonth_dt")
    fig3 = px.line(monthly, x="YearMonth_dt", y="Orders", title="Monthly Order Volume",
                   markers=True, color_discrete_sequence=["#45B7D1"])
    fig3.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                       yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: Delivery analysis (mirrors Part1 Cell 24) ─────────────────────────
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        status_counts = df["DeliveryStatus"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(status_counts, names="Status", values="Count",
                     title="Delivery Status Distribution",
                     color="Status",
                     color_discrete_map={"On Time":"#2ecc71","Late":"#e74c3c","Cancelled":"#f39c12"},
                     hole=0.4)
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df, x="DeliveryTime", nbins=30, title="Delivery Time Distribution (min)",
                           color_discrete_sequence=["#9b59b6"])
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig, use_container_width=True)

    # Age distribution
    fig_age = px.histogram(df.drop_duplicates("CustomerID"), x="Age", nbins=20,
                            title="Customer Age Distribution",
                            color_discrete_sequence=["#3498db"])
    fig_age.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
    st.plotly_chart(fig_age, use_container_width=True)

# ── Tab 4: Cuisine & Payment analysis (mirrors Part2 Cell 13) ────────────────
with tab4:
    cuisine_stats = df.groupby("Cuisine").agg(
        Avg_Amount  = ("TotalOrderValue", "mean"),
        Avg_Items   = ("ItemsPerOrder",   "mean"),
        Order_Count = ("OrderID",         "count"),
    ).round(2).reset_index()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Avg Order Amount by Cuisine", "Avg Items per Order by Cuisine"])
    palette = px.colors.qualitative.Plotly
    fig.add_trace(go.Bar(x=cuisine_stats["Cuisine"], y=cuisine_stats["Avg_Amount"],
                         marker_color=palette[:len(cuisine_stats)], name="Avg Amount"), row=1, col=1)
    fig.add_trace(go.Bar(x=cuisine_stats["Cuisine"], y=cuisine_stats["Avg_Items"],
                         marker_color=palette[3:3+len(cuisine_stats)], name="Avg Items"), row=1, col=2)
    fig.update_layout(height=420, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
    fig.update_xaxes(tickangle=20, gridcolor="rgba(255,255,255,.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        pay = df["PaymentMode"].value_counts().reset_index()
        pay.columns = ["Mode","Count"]
        fig_pay = px.bar(pay, x="Mode", y="Count", title="Payment Mode Distribution",
                         color="Mode", color_discrete_sequence=palette)
        fig_pay.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                               showlegend=False, yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig_pay, use_container_width=True)

    with c2:
        city_orders = df.groupby("City")["OrderID"].count().reset_index()
        city_orders.columns = ["City","Orders"]
        fig_city = px.bar(city_orders.sort_values("Orders", ascending=True),
                          x="Orders", y="City", orientation="h",
                          title="Orders by City", color="Orders",
                          color_continuous_scale="Blues")
        fig_city.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                                showlegend=False, xaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig_city, use_container_width=True)

# ── Tab 5: Derived features (mirrors Part1 Cell 25) ───────────────────────────
with tab5:
    st.markdown("**Derived Features from Part1.ipynb** — OrderFrequency, AvgDeliveryTime")
    c1, c2, c3 = st.columns(3)
    with c1:
        c1.metric("Avg Order Frequency", f"{df['OrderFrequency'].mean():.1f}")
    with c2:
        c2.metric("Avg Delivery Time", f"{df['AvgDeliveryTime'].mean():.1f} min")
    with c3:
        c3.metric("Avg Order Value", f"₹{df['TotalOrderValue'].mean():.1f}")

    # Label encoding (mirrors Part1 Cell 28)
    st.markdown("**Label Encoding — Categorical Variables** (Part1.ipynb Cell 28)")
    cat_cols = ["Cuisine", "City", "PaymentMode", "Gender"]
    from sklearn.preprocessing import LabelEncoder
    rows = []
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            classes = sorted(df[col].dropna().unique())
            encoded = le.fit_transform(classes)
            for orig, enc in zip(classes, encoded):
                rows.append({"Column": col, "Original Value": orig, "Encoded": int(enc)})
    enc_df = pd.DataFrame(rows)
    st.dataframe(enc_df, use_container_width=True, height=350)

    # Gender & Rating distributions
    c1, c2 = st.columns(2)
    with c1:
        gen = df["Gender"].value_counts().reset_index()
        gen.columns = ["Gender","Count"]
        fig_g = px.pie(gen, names="Gender", values="Count", title="Gender Distribution",
                       color_discrete_sequence=["#6366f1","#f472b6"], hole=0.4)
        fig_g.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
        st.plotly_chart(fig_g, use_container_width=True)
    with c2:
        fig_r = px.histogram(df, x="Rating", nbins=20, title="Restaurant Rating Distribution",
                             color_discrete_sequence=["#f59e0b"])
        fig_r.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font=dict(color="#e2e8f0"), xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                             yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig_r, use_container_width=True)
