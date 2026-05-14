"""
7_Business_Insights.py — mirrors Part6.ipynb exactly.
Restaurant scoring: Rating*0.40 + OnTimeRate*0.40 + NormVolume*5*0.20
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

st.set_page_config(page_title="Business Insights | Food Delivery AI", page_icon="💡", layout="wide")

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
.insight-badge{display:inline-block;background:linear-gradient(90deg,#6366f1,#8b5cf6);
               color:#fff;border-radius:8px;padding:.3rem .8rem;font-size:.8rem;
               font-weight:600;margin-bottom:.8rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>💡 Business Insights</h1>
  <p>Restaurant scoring, customer segments & delivery bottlenecks — mirrors Part6.ipynb scoring formula exactly.</p>
</div>
""", unsafe_allow_html=True)

from utils.nb_logic import (compute_restaurant_insights, compute_delivery_bottlenecks,
                             compute_customer_segment_insights, load_and_preprocess)

with st.spinner("💡 Computing business insights…"):
    restaurant_perf, top_restaurants = compute_restaurant_insights()
    city_stats, cuisine_stats, dow_stats = compute_delivery_bottlenecks()
    seg_summary = compute_customer_segment_insights()
    df = load_and_preprocess()

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Restaurants", "👥 Customer Segments", "🚚 Delivery Bottlenecks", "📈 Order Trends"])

# ── Tab 1: Top Restaurants (mirrors Part6 Cell 15) ───────────────────────────
with tab1:
    st.markdown('<div class="insight-badge">Insight 1 — Top Performing Restaurants</div>', unsafe_allow_html=True)
    st.caption("Score = Rating×0.40 + OnTimeRate×0.40 + (NormVolume×5)×0.20 — exact Part6 formula")

    PALETTE = px.colors.qualitative.Plotly
    rest_ids = top_restaurants["RestaurantID"].astype(str).tolist()

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Overall Score", "Avg Rating & On-Time Rate", "Revenue (₹)"])

    fig.add_trace(go.Bar(y=rest_ids[::-1], x=top_restaurants["Score"][::-1],
                         orientation="h", marker_color=PALETTE[0], name="Score"),
                  row=1, col=1)
    fig.add_trace(go.Bar(y=rest_ids[::-1], x=top_restaurants["AvgRating"][::-1],
                         orientation="h", marker_color=PALETTE[1], name="Avg Rating", opacity=0.9),
                  row=1, col=2)
    fig.add_trace(go.Bar(y=rest_ids[::-1], x=top_restaurants["OnTimeRate"][::-1],
                         orientation="h", marker_color=PALETTE[2], name="OnTime%", opacity=0.9),
                  row=1, col=2)
    fig.add_trace(go.Bar(y=rest_ids[::-1], x=top_restaurants["TotalRevenue"][::-1],
                         orientation="h", marker_color=PALETTE[3], name="Revenue"),
                  row=1, col=3)

    fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e2e8f0"), barmode="overlay",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Top 10 Restaurant Table"):
        st.dataframe(top_restaurants[["RestaurantID","OrderVolume","AvgRating",
                                      "AvgDeliveryMin","OnTimeRate","LateRate",
                                      "TotalRevenue","Score"]].style.format({
            "AvgRating":"{:.2f}","AvgDeliveryMin":"{:.1f}",
            "OnTimeRate":"{:.1f}%","LateRate":"{:.1f}%",
            "TotalRevenue":"₹{:,.0f}","Score":"{:.3f}",
        }), use_container_width=True)

# ── Tab 2: Customer Segments (mirrors Part6 Cell 17) ────────────────────────
with tab2:
    st.markdown('<div class="insight-badge">Insight 2 — Customer Segment Behaviour</div>', unsafe_allow_html=True)

    color_map = {"💰 High-Value": "#FF6B6B", "🔁 Regular": "#4ECDC4", "😐 Occasional": "#A78BFA"}

    fig_bar = px.bar(seg_summary, x="Segment",
                     y=["Avg_Orders","Avg_Spending","Avg_Delivery","Avg_Rating"],
                     barmode="group", title="Segment KPIs Comparison",
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    fig_bar.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(seg_summary.set_index("Segment").style.format("{:.2f}"), use_container_width=True)

# ── Tab 3: Delivery Bottlenecks (mirrors Part6 Cell 19) ─────────────────────
with tab3:
    st.markdown('<div class="insight-badge">Insight 3 — Delivery Bottlenecks</div>', unsafe_allow_html=True)

    sub1, sub2 = st.columns(2)
    with sub1:
        fig_city = px.bar(city_stats.sort_values("LateRate", ascending=False),
                          x="City", y="LateRate", color="LateRate",
                          color_continuous_scale="Reds",
                          title="Late Delivery Rate by City (%)")
        fig_city.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                               showlegend=False,
                               xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                               yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig_city, use_container_width=True)

    with sub2:
        fig_cuis = px.bar(cuisine_stats.sort_values("AvgDelivery", ascending=False),
                          x="Cuisine", y="AvgDelivery", color="AvgDelivery",
                          color_continuous_scale="Oranges",
                          title="Avg Delivery Time by Cuisine (min)")
        fig_cuis.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                               showlegend=False,
                               xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                               yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig_cuis, use_container_width=True)

    fig_dow = px.bar(dow_stats, x="Day", y=["AvgDelivery","LateRate"],
                     barmode="group", title="Delivery Metrics by Day of Week",
                     color_discrete_map={"AvgDelivery":"#45B7D1","LateRate":"#FF6B6B"})
    fig_dow.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
    st.plotly_chart(fig_dow, use_container_width=True)

# ── Tab 4: Order Trends (mirrors Part7 Cell 13) ──────────────────────────────
with tab4:
    st.markdown('<div class="insight-badge">Order Trends Over Time</div>', unsafe_allow_html=True)

    monthly = df.groupby("YearMonth").agg(
        Orders      = ("OrderID",       "count"),
        AvgDelivery = ("DeliveryTime",  "mean"),
        LateRate    = ("IsLate",        lambda x: x.mean() * 100),
    ).reset_index()
    monthly["YearMonth_dt"] = monthly["YearMonth"].dt.to_timestamp()
    monthly = monthly.sort_values("YearMonth_dt")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Monthly Order Volume",
                                        "Avg Delivery Time (min)",
                                        "Late Delivery Rate (%)"])
    x = monthly["YearMonth_dt"]

    fig.add_trace(go.Scatter(x=x, y=monthly["Orders"], fill="tozeroy",
                             fillcolor="rgba(69,183,209,.2)", line=dict(color="#45B7D1", width=2.5),
                             mode="lines+markers", name="Orders"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=monthly["AvgDelivery"], fill="tozeroy",
                             fillcolor="rgba(255,160,122,.2)", line=dict(color="#FFA07A", width=2.5),
                             mode="lines+markers", name="Avg Delivery"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=monthly["LateRate"], fill="tozeroy",
                             fillcolor="rgba(255,107,107,.2)", line=dict(color="#FF6B6B", width=2.5),
                             mode="lines+markers", name="Late Rate"), row=3, col=1)

    fig.update_layout(height=660, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e2e8f0"), showlegend=False)
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)")
    st.plotly_chart(fig, use_container_width=True)
