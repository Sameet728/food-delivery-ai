"""
app.py — Food Delivery AI Analytics Platform
Landing page — Option B architecture (all models cache via nb_logic.py).
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Food Delivery AI Analytics",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0a0b14;color:#e2e8f0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0c29,#302b63,#24243e);}
[data-testid="stSidebar"] *{color:#e0e0e0!important;}
[data-testid="metric-container"]{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
    border-radius:14px;padding:16px;}
[data-testid="metric-container"] label{color:#64748b!important;font-size:11px!important;}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-size:22px!important;font-weight:700!important;}
.stTabs [data-baseweb="tab-list"]{background:#111827;border-radius:10px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:7px;color:#94a3b8;
    font-weight:500;padding:8px 20px;transition:all .2s;}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
    background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff!important;
    box-shadow:0 2px 12px rgba(99,102,241,.4);}
.stButton>button{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;
    border:none;border-radius:10px;font-weight:600;padding:10px 24px;transition:all .2s;
    box-shadow:0 4px 12px rgba(99,102,241,.3);}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(99,102,241,.5);}
hr{border-color:rgba(99,102,241,.2)!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#0a0b14;}
::-webkit-scrollbar-thumb{background:#6366f1;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

from utils.data_loader import load_data

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 16px;">
        <div style="font-size:48px;filter:drop-shadow(0 0 20px rgba(99,102,241,.6));">🍔</div>
        <div style="font-size:20px;font-weight:800;background:linear-gradient(135deg,#6366f1,#06b6d4);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:8px;">
            FoodDelivery AI
        </div>
        <div style="font-size:11px;color:#475569;margin-top:4px;letter-spacing:.1em;">
            ANALYTICS PLATFORM
        </div>
    </div>
    <hr style="border-color:rgba(99,102,241,.25);">
    """, unsafe_allow_html=True)
    st.markdown("### 📌 Navigate")
    st.markdown("""
    - 🏠 **Dashboard** — KPIs & Overview
    - 📊 **Data Analytics** — EDA (Part1)
    - 🔗 **Association Rules** — Apriori (Part2)
    - 🌲 **Delivery Prediction** — 4 Models (Part3)
    - 👥 **Customer Segmentation** — KMeans (Part4)
    - 🧠 **ANN Prediction** — Deep Learning (Part5)
    - 💡 **Business Insights** — Scoring (Part6)
    - 🎯 **Recommendation** — Rule Engine (Part7)
    - ℹ️ **About Project** — Team & Docs
    """)
    st.markdown("<hr style='border-color:rgba(99,102,241,.2);'>", unsafe_allow_html=True)
    st.caption("© 2024 Food Delivery AI Analytics")

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:48px 0 32px;">
    <div style="font-size:72px;filter:drop-shadow(0 0 30px rgba(99,102,241,.7));margin-bottom:16px;">🍔</div>
    <h1 style="font-size:48px;font-weight:800;margin:0;
        background:linear-gradient(135deg,#6366f1,#06b6d4,#10b981);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Food Delivery AI Analytics
    </h1>
    <p style="color:#64748b;font-size:18px;max-width:620px;margin:16px auto 0;line-height:1.7;">
        AI-Powered Analytics Platform &mdash; mirroring your Part1&ndash;Part7 Jupyter notebooks
        with exact same models, parameters &amp; results.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Feature Cards ─────────────────────────────────────────────────────────────
features = [
    ("#6366f1", "📊", "Executive Dashboard",
     "Real-time KPIs, revenue trends, city analytics with interactive Plotly charts"),
    ("#06b6d4", "🤖", "ML Prediction Engine",
     "4 classifiers (LR, DT, RF, SVM) + ANN — exact same params as Part3 & Part5 notebooks"),
    ("#10b981", "🎯", "Smart Recommendation",
     "Apriori live rules (182 patterns) powering personalised food recommendations"),
    ("#f59e0b", "👥", "Customer Segmentation",
     "KMeans k=3 + Agglomerative — mirrors Part4 clustering with PCA visualisation"),
    ("#ef4444", "💡", "Business Insights",
     "Restaurant scoring formula: Rating×0.4 + OnTime×0.4 + Volume×0.2 (Part6)"),
    ("#8b5cf6", "🔗", "Association Mining",
     "Apriori: 105 itemsets → 182 rules from 14 menu items across 30K orders (Part2)"),
]

row1 = st.columns(3)
row2 = st.columns(3)
for col, (color, icon, title, desc) in zip(row1 + row2, features):
    with col:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}10,rgba(15,17,35,0.8));
             border:1px solid {color}30;border-radius:16px;padding:24px;
             margin-bottom:12px;transition:transform .2s;
             box-shadow:0 4px 24px {color}15;">
            <div style="font-size:32px;margin-bottom:10px;">{icon}</div>
            <div style="color:#e2e8f0;font-weight:700;font-size:15px;margin-bottom:8px;">{title}</div>
            <div style="color:#64748b;font-size:12px;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── System Status ─────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### ⚙️ System Status — Option B (Mirror Notebook Logic)")
    st.markdown("""
    <div style="background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.25);
         border-radius:14px;padding:20px;margin-top:8px;">
        <div style="color:#10b981;font-weight:700;font-size:16px;margin-bottom:10px;">
            ✅ All systems ready — no manual training required
        </div>
        <div style="color:#94a3b8;font-size:13px;line-height:2;">
            All models train automatically on first page visit via <code style="background:#1e293b;
            padding:2px 6px;border-radius:4px;color:#a5b4fc;">@st.cache_resource</code> in
            <code style="background:#1e293b;padding:2px 6px;border-radius:4px;color:#a5b4fc;">utils/nb_logic.py</code>
            &nbsp;— trained once, cached for the entire session.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px;">
        <div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);border-radius:12px;padding:14px;">
            <div style="color:#a5b4fc;font-weight:600;font-size:13px;">🌲 Classification (Part3)</div>
            <div style="color:#64748b;font-size:11px;margin-top:4px;">LR · DT · RF · SVM &nbsp;|&nbsp; random_state=42 · test_size=0.2</div>
        </div>
        <div style="background:rgba(6,182,212,.08);border:1px solid rgba(6,182,212,.2);border-radius:12px;padding:14px;">
            <div style="color:#67e8f9;font-weight:600;font-size:13px;">🔗 Association Rules (Part2)</div>
            <div style="color:#64748b;font-size:11px;margin-top:4px;">Apriori · support=0.01 · confidence=0.10</div>
        </div>
        <div style="background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);border-radius:12px;padding:14px;">
            <div style="color:#6ee7b7;font-weight:600;font-size:13px;">👥 Clustering (Part4)</div>
            <div style="color:#64748b;font-size:11px;margin-top:4px;">KMeans k=3 · n_init=10 · Agglomerative Ward</div>
        </div>
        <div style="background:rgba(139,92,246,.08);border:1px solid rgba(139,92,246,.2);border-radius:12px;padding:14px;">
            <div style="color:#c4b5fd;font-weight:600;font-size:13px;">🧠 ANN Regressor (Part5)</div>
            <div style="color:#64748b;font-size:11px;margin-top:4px;">Dense(64→32→16→1) · Adam · EarlyStopping(patience=5)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(6,182,212,.08));
         border:1px solid rgba(16,185,129,.3);border-radius:16px;
         padding:28px;text-align:center;margin-top:8px;">
        <div style="font-size:40px;margin-bottom:8px;">✅</div>
        <div style="color:#10b981;font-weight:700;font-size:16px;">Models Ready</div>
        <div style="color:#475569;font-size:12px;margin-top:6px;">Auto-cached on first visit</div>
        <hr style="border-color:rgba(16,185,129,.2);margin:14px 0;">
        <div style="color:#94a3b8;font-size:11px;">Navigate any page →<br>models load automatically</div>
    </div>
    """, unsafe_allow_html=True)

# ── Dataset at a Glance ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Dataset at a Glance")

with st.spinner("Loading dataset…"):
    df = load_data()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Orders",  f"{len(df):,}")
c2.metric("Customers",     f"{df['CustomerID'].nunique():,}")
c3.metric("Restaurants",   f"{df['RestaurantID'].nunique():,}")
c4.metric("Cities",        f"{df['City'].nunique():,}")
c5.metric("Cuisines",      f"{df['Cuisine'].nunique():,}")
c6.metric("Date Range",
          f"{df['OrderDate'].min().strftime('%b %Y')}–{df['OrderDate'].max().strftime('%b %Y')}")

# ── Delivery Status KPI Tiles ─────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
on_time  = round((df["DeliveryStatus"] == "On Time").mean() * 100, 1)
late_pct = round((df["DeliveryStatus"] == "Late").mean() * 100, 1)
canc_pct = round((df["DeliveryStatus"] == "Cancelled").mean() * 100, 1)
avg_val  = round(df["Revenue"].mean(), 1)

c1, c2, c3, c4 = st.columns(4)
for col, color, icon, val, lbl in zip(
    [c1, c2, c3, c4],
    ["#10b981", "#ef4444", "#f59e0b", "#6366f1"],
    ["✅", "⏰", "❌", "₹"],
    [f"{on_time}%", f"{late_pct}%", f"{canc_pct}%", f"{avg_val}"],
    ["On Time Deliveries", "Late Deliveries", "Cancelled Orders", "Avg Order Value"],
):
    with col:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}12,rgba(10,11,20,.8));
             border:1px solid {color}30;border-radius:14px;padding:20px;text-align:center;
             box-shadow:0 4px 20px {color}10;">
            <div style="font-size:24px;margin-bottom:6px;">{icon}</div>
            <div style="color:{color};font-size:28px;font-weight:800;">{val}</div>
            <div style="color:#64748b;font-size:11px;margin-top:4px;text-transform:uppercase;
                 letter-spacing:.08em;">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:48px 0 24px;color:#374151;font-size:12px;">
    <div style="font-size:28px;margin-bottom:8px;">🍔</div>
    Food Delivery Analytics &amp; Intelligent Recommendation System<br>
    <span style="color:#6366f1;">Streamlit</span> &nbsp;·&nbsp;
    <span style="color:#06b6d4;">Scikit-learn</span> &nbsp;·&nbsp;
    <span style="color:#10b981;">TensorFlow</span> &nbsp;·&nbsp;
    <span style="color:#f59e0b;">Plotly</span> &nbsp;·&nbsp;
    <span style="color:#8b5cf6;">MLxtend</span>
</div>
""", unsafe_allow_html=True)
