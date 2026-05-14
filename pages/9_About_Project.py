"""
9_About_Project.py — Project overview, architecture, ML models, workflow, team.
Premium dark UI — Option B architecture.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from utils.data_loader import load_data

st.set_page_config(page_title="About Project | Food Delivery AI", page_icon="ℹ️", layout="wide")

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
hr{border-color:rgba(99,102,241,.2)!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#0a0b14;}
::-webkit-scrollbar-thumb{background:#6366f1;border-radius:3px;}
.obj-card{background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.18);
    border-radius:12px;padding:14px 18px;margin-bottom:8px;}
.obj-item{color:#e2e8f0;font-size:13px;line-height:2.0;}
.step-card{background:#111827;border-left:4px solid #6366f1;border-radius:0 12px 12px 0;
    padding:16px 20px;margin-bottom:12px;}
.step-title{color:#e2e8f0;font-weight:700;font-size:14px;margin-bottom:4px;}
.step-desc{color:#94a3b8;font-size:12px;line-height:1.7;}
.team-card{background:linear-gradient(135deg,rgba(99,102,241,.1),rgba(15,17,35,.8));
    border:1px solid rgba(99,102,241,.25);border-radius:16px;padding:28px;
    text-align:center;margin-bottom:16px;}
.badge{display:inline-block;border-radius:20px;padding:3px 12px;
    font-size:11px;font-weight:600;margin:3px 2px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ℹ️ About Project")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,rgba(99,102,241,.12),rgba(6,182,212,.06),rgba(10,11,20,.9));
     border:1px solid rgba(99,102,241,.25);border-radius:20px;padding:40px;
     margin-bottom:28px;text-align:center;">
    <div style="font-size:52px;margin-bottom:12px;
         filter:drop-shadow(0 0 24px rgba(99,102,241,.6));">🍔</div>
    <h1 style="font-size:30px;font-weight:800;margin:0 0 12px;
        background:linear-gradient(135deg,#6366f1,#06b6d4,#10b981);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Food Delivery Analytics &<br>Intelligent Recommendation System
    </h1>
    <p style="color:#94a3b8;font-size:15px;max-width:700px;margin:0 auto;line-height:1.8;">
        An end-to-end AI-powered analytics platform integrating machine learning,
        deep learning, and market basket analysis — deployed as a professional
        Streamlit dashboard that mirrors your Part1–Part7 Jupyter notebook results exactly.
    </p>
</div>
""", unsafe_allow_html=True)

df = load_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview", "🏗️ Architecture", "🤖 ML Models", "🔬 Workflow", "👥 Team"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Project Objectives")
        objectives = [
            "Integrate and preprocess food delivery data from multiple sources",
            "Perform exploratory data analysis to identify trends (Part1)",
            "Implement market basket analysis using Apriori algorithm (Part2)",
            "Build multi-class delivery status classifiers: LR, DT, RF, SVM (Part3)",
            "Develop customer segmentation with KMeans & Hierarchical (Part4)",
            "Train ANN model for delivery time regression (Part5)",
            "Generate automated business intelligence & restaurant scoring (Part6)",
            "Build intelligent recommendation engine from association rules (Part7)",
            "Deploy as a professional Streamlit dashboard — Option B architecture",
        ]
        items_html = "".join(f'<li style="color:#e2e8f0;font-size:13px;line-height:2.2;list-style:none;padding:0;">✅ {o}</li>' for o in objectives)
        st.markdown(f'<div class="obj-card"><ul style="margin:0;padding:0;">{items_html}</ul></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 📊 Dataset Summary")
        dataset_info = {
            "Attribute": ["Total Records","Features","Date Range","Cities",
                          "Cuisines","Restaurants","Unique Customers",
                          "Menu Items","Delivery Statuses","Payment Modes"],
            "Value": [
                f"{len(df):,}",
                f"{len(df.columns)}",
                f"{df['OrderDate'].min().strftime('%b %Y')} – {df['OrderDate'].max().strftime('%b %Y')}",
                f"{df['City'].nunique()}",
                f"{df['Cuisine'].nunique()}",
                f"{df['RestaurantID'].nunique()}",
                f"{df['CustomerID'].nunique():,}",
                "14 unique items",
                "On Time · Late · Cancelled",
                "Cash · Card · UPI",
            ]
        }
        st.dataframe(pd.DataFrame(dataset_info), use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background:#111827;border:1px solid rgba(6,182,212,.2);
             border-radius:12px;padding:16px;font-size:12px;color:#94a3b8;margin-top:12px;">
            <b style="color:#67e8f9;">Orders</b> ←→ <b style="color:#a5b4fc;">Customers</b><br>
            <b style="color:#67e8f9;">Orders</b> ←→ <b style="color:#f87171;">Restaurants</b><br>
            <b style="color:#67e8f9;">Orders</b> ←→ <b style="color:#fcd34d;">Items</b> ←→ <b style="color:#6ee7b7;">Menu</b><br>
            <b style="color:#67e8f9;">Orders</b> ←→ <b style="color:#c4b5fd;">Payments</b> ←→ <b style="color:#f9a8d4;">Delivery</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Technology Stack")
    tech = [
        ("Frontend / UI",         "#6366f1", "Streamlit · Custom CSS · Inter Font · Dark Theme"),
        ("Data Processing",       "#06b6d4", "Pandas · NumPy · Scikit-learn Preprocessing"),
        ("Machine Learning",      "#10b981", "Scikit-learn (LR, DT, RF, SVM, KMeans, Agglomerative)"),
        ("Deep Learning",         "#8b5cf6", "TensorFlow / Keras — ANN Dense(64→32→16→1)"),
        ("Visualisation",         "#f59e0b", "Plotly Express · Plotly Graph Objects · NetworkX"),
        ("Association Mining",    "#ef4444", "MLxtend — Apriori · TransactionEncoder"),
        ("Caching / Performance", "#06b6d4", "@st.cache_data + @st.cache_resource (nb_logic.py)"),
        ("Architecture",          "#10b981", "Option B — Mirror Notebook Logic (Zero Retrain)"),
    ]
    cols = st.columns(4)
    for i, (cat, color, tech_str) in enumerate(tech):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color}0e,rgba(17,24,39,.9));
                 border:1px solid {color}28;border-radius:12px;padding:14px;margin-bottom:10px;min-height:90px;">
                <div style="color:{color};font-weight:700;font-size:12px;margin-bottom:6px;">{cat}</div>
                <div style="color:#94a3b8;font-size:11px;line-height:1.6;">{tech_str}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Architecture
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏗️ Option B Architecture — Mirror Notebook Logic")
    st.markdown("""
    <div style="background:#0d1117;border:1px solid rgba(99,102,241,.25);
         border-radius:16px;padding:28px;font-family:monospace;font-size:12px;
         color:#e2e8f0;line-height:1.9;overflow-x:auto;">
    <pre style="color:#e2e8f0;background:transparent;margin:0;white-space:pre-wrap;">
┌─────────────────────────────────────────────────────────────────────┐
│         FOOD DELIVERY AI — Option B (Mirror Notebook Logic)         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📁 Final_New_dataset.csv  (30,000 rows × 19 cols)                  │
│       │                                                              │
│       ▼                                                              │
│  🔧 utils/nb_logic.py  ──  Single source of truth                   │
│       │    (mirrors Part1–Part7 notebooks: same params & seeds)      │
│       │                                                              │
│       ├─ load_and_preprocess()        @st.cache_data                │
│       │    Amount → TotalOrderValue · date features · OrderFreq      │
│       │                                                              │
│       ├─ train_classifiers()          @st.cache_resource            │
│       │    LR · DT · RF · SVM  (random_state=42, test_size=0.2)     │
│       │    stratify=y · StandardScaler · LabelEncoder  ── Part3     │
│       │                                                              │
│       ├─ compute_association_rules()  @st.cache_data                │
│       │    Apriori(support=0.01, confidence=0.10, lift≥0.5)         │
│       │    105 itemsets → 182 rules  ── Part2                       │
│       │                                                              │
│       ├─ train_clustering()           @st.cache_resource            │
│       │    KMeans(k=3, n_init=10) + Agglomerative(Ward)             │
│       │    Remap by avg spending  ── Part4                          │
│       │                                                              │
│       ├─ train_ann()                  @st.cache_resource            │
│       │    Dense(64→32→16→1) Adam MSE EarlyStopping(patience=5)    │
│       │    ── Part5                                                  │
│       │                                                              │
│       ├─ compute_restaurant_insights()  @st.cache_data              │
│       │    Score = Rating×0.40 + OnTimeRate×0.40 + Volume×0.20     │
│       │    ── Part6                                                  │
│       │                                                              │
│       └─ train_delivery_regressor()   @st.cache_resource            │
│            RF Regressor for delivery time feature importance         │
│                                                                      │
│  🌐 Streamlit Pages (all call nb_logic, zero notebook execution)    │
│       app.py           →  Landing · system status · KPIs            │
│       1_Dashboard      →  Executive KPIs · revenue trends           │
│       2_Data_Analytics →  EDA charts (Part1)                        │
│       3_Association_Rules → Apriori/rules/recommender (Part2)       │
│       4_Delivery_Prediction → 4 classifiers + predictor (Part3)     │
│       5_Customer_Segmentation → KMeans+PCA+predictor (Part4)        │
│       6_ANN_Prediction → Loss curve + actual vs pred (Part5)        │
│       7_Business_Insights → Restaurant scoring + trends (Part6)     │
│       8_Recommendation_System → Live engine + network (Part7)       │
│       9_About          →  This page                                 │
└─────────────────────────────────────────────────────────────────────┘
    </pre></div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚡ Performance Strategy")
    perf = [
        ("@st.cache_data",     "#6366f1", "load_and_preprocess, compute_association_rules, restaurant insights — re-run only when CSV changes"),
        ("@st.cache_resource", "#06b6d4", "train_classifiers, train_clustering, train_ann — loaded once per session, shared across all pages"),
        ("Zero Retrain",       "#10b981", "No pkl files · No notebook execution · No model_trainer.py · Pure Python in nb_logic.py"),
        ("Exact Params",       "#f59e0b", "Same random_state=42 · same splits · same features as your Part1–Part7 notebooks"),
    ]
    cols = st.columns(2)
    for i, (title, color, desc) in enumerate(perf):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color}0e,rgba(17,24,39,.9));
                 border:1px solid {color}28;border-radius:12px;padding:16px;margin-bottom:10px;">
                <div style="color:{color};font-weight:700;font-size:13px;margin-bottom:6px;">⚡ {title}</div>
                <div style="color:#94a3b8;font-size:12px;line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML Models
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 ML Algorithms — Exact Notebook Parameters")

    models_data = [
        ("🌲 Logistic Regression",        "Classification", "Part3", "Delivery status (On Time/Late/Cancelled)", "max_iter=1000, random_state=42"),
        ("🌳 Decision Tree (CART)",        "Classification", "Part3", "Delivery status (interpretable)",          "random_state=42"),
        ("🌲 Random Forest",               "Classification", "Part3", "Delivery status (best classifier)",        "n_estimators=100, random_state=42, n_jobs=-1"),
        ("⚡ SVM",                         "Classification", "Part3", "Delivery status (kernel trick)",           "SVC(probability=True, random_state=42)"),
        ("🔵 K-Means Clustering",          "Clustering",     "Part4", "Customer segmentation (3 segments)",       "n_clusters=3, random_state=42, n_init=10"),
        ("🌿 Agglomerative Clustering",    "Clustering",     "Part4", "Hierarchical customer grouping",           "n_clusters=3, linkage='ward'"),
        ("🧠 ANN — Dense Network",         "Regression",     "Part5", "Delivery time estimation (minutes)",       "Dense(64→32→16→1), Adam, MSE, EarlyStopping(5)"),
        ("🔗 Apriori Algorithm",           "Assoc. Mining",  "Part2", "Food item co-purchase patterns",           "support=0.01, confidence=0.10, lift≥0.5"),
        ("📉 RF Regressor (Surrogate)",    "Regression",     "Part6", "Feature importance for delivery time",     "n_estimators=100, random_state=42"),
        ("📐 PCA",                         "Dim. Reduction", "Part4", "Cluster visualisation (2D scatter)",       "n_components=2, random_state=42"),
    ]

    cat_colors = {
        "Classification": "#6366f1", "Clustering": "#06b6d4",
        "Regression": "#10b981",     "Assoc. Mining": "#f59e0b",
        "Dim. Reduction": "#8b5cf6",
    }

    for algo, cat, part, purpose, params in models_data:
        color = cat_colors.get(cat, "#94a3b8")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}08,rgba(17,24,39,.9));
             border:1px solid {color}25;border-radius:12px;padding:14px 18px;margin-bottom:8px;
             display:flex;gap:16px;align-items:flex-start;">
            <div style="min-width:200px;">
                <div style="color:#e2e8f0;font-weight:700;font-size:14px;">{algo}</div>
                <div>
                    <span class="badge" style="background:{color}18;color:{color};border:1px solid {color}30;">{cat}</span>
                    <span class="badge" style="background:#1e293b;color:#94a3b8;border:1px solid #334155;">📓 {part}</span>
                </div>
            </div>
            <div style="flex:1;">
                <div style="color:#94a3b8;font-size:12px;margin-bottom:4px;">{purpose}</div>
                <div style="color:#475569;font-size:11px;font-family:monospace;background:#0d1117;
                     border-radius:6px;padding:4px 8px;display:inline-block;">{params}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Key Results (from your dataset)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("On Time Deliveries",  f"{round((df['DeliveryStatus']=='On Time').mean()*100,1)}%")
    c2.metric("Late Deliveries",     f"{round((df['DeliveryStatus']=='Late').mean()*100,1)}%")
    c3.metric("Avg Delivery Time",   f"{round(df['DeliveryTime'].mean(),1)} min")
    c4.metric("Association Rules",   "182 rules")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Workflow
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔬 Project Workflow — Part1 to Part7")
    steps = [
        ("1️⃣", "Part1 — Data Preprocessing",
         "Loaded 30,000 orders · Parsed dates (dayfirst=True) · Renamed Amount→TotalOrderValue · "
         "Derived OrderFrequency, AvgDeliveryTime · LabelEncoding + StandardScaler · IsLate/IsOnTime flags"),
        ("2️⃣", "Part2 — Association Rule Mining (Apriori)",
         "Built per-order transactions from ItemName (comma-split) · TransactionEncoder · "
         "Apriori(support=0.01) → 105 itemsets · association_rules(confidence=0.10, lift≥0.5) → 182 rules"),
        ("3️⃣", "Part3 — Delivery Status Classification",
         "4 models: LR · DT · RF · SVM · Features: Age, City, Rating, Cuisine, ItemsPerOrder, "
         "PaymentMode, TotalOrderValue, DeliveryTime, dayofweek, month, Gender · "
         "Split: test_size=0.2, random_state=42, stratify=y"),
        ("4️⃣", "Part4 — Customer Segmentation",
         "Aggregate per-customer: OrderFrequency, TotalSpending, AvgDeliveryTime, Cuisine_enc · "
         "KMeans(k=3, n_init=10, random_state=42) · Remap clusters by avg spending · Agglomerative(Ward)"),
        ("5️⃣", "Part5 — ANN Delivery Time Prediction",
         "Features: ItemsPerOrder, Rating, City, Cuisine, PaymentMode · "
         "Dense(64→32→16→1) · Adam · MSE · EarlyStopping(patience=5) · test_size=0.2, random_state=42"),
        ("6️⃣", "Part6 — Business Intelligence & Insights",
         "Restaurant scoring: Rating×0.40 + OnTimeRate×0.40 + NormVolume×5×0.20 · "
         "Delivery bottleneck analysis by city/cuisine/day-of-week · Customer segment KPIs"),
        ("7️⃣", "Part7 — Visualisations & Recommendation Engine",
         "Association rule network graph (NetworkX) · Interactive food recommender · "
         "Monthly order trends · Geographic heatmaps · Executive summary dashboard"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="font-size:26px;min-width:34px;">{icon}</div>
                <div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Team
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 👥 Project Team")

    team = [
        {"name": "Team Member 1", "role": "ML Engineer & Data Scientist",
         "contribution": "Association Rules · Classification Models · ANN", "emoji": "👨‍💻", "color": "#6366f1"},
        {"name": "Team Member 2", "role": "Data Analyst & Visualisation",
         "contribution": "EDA · Dashboard · Business Insights", "emoji": "📊", "color": "#06b6d4"},
        {"name": "Team Member 3", "role": "Streamlit Developer & UI/UX",
         "contribution": "App Architecture · Dark UI · Deployment", "emoji": "🚀", "color": "#10b981"},
    ]

    cols = st.columns(3)
    for i, m in enumerate(team):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="team-card">
                <div style="font-size:52px;margin-bottom:10px;
                     filter:drop-shadow(0 0 12px {m['color']}80);">{m['emoji']}</div>
                <div style="color:#e2e8f0;font-size:17px;font-weight:700;margin-bottom:4px;">{m['name']}</div>
                <div style="color:{m['color']};font-size:13px;font-weight:600;margin-bottom:10px;">{m['role']}</div>
                <div style="color:#94a3b8;font-size:12px;">{m['contribution']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Project footer badge
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,.1),rgba(6,182,212,.06));
         border:1px solid rgba(99,102,241,.25);border-radius:16px;padding:28px;
         text-align:center;margin-top:8px;">
        <div style="color:#a5b4fc;font-size:18px;font-weight:700;margin-bottom:8px;">🎓 Academic Project</div>
        <div style="color:#94a3b8;font-size:13px;line-height:1.8;">
            Food Delivery Analytics and Intelligent Recommendation System<br>
            Final Year Project &nbsp;·&nbsp; 2024
        </div>
        <div style="margin-top:16px;">
            <span class="badge" style="background:#6366f118;color:#a5b4fc;border:1px solid #6366f130;">Python 3.13</span>
            <span class="badge" style="background:#06b6d418;color:#67e8f9;border:1px solid #06b6d430;">Streamlit</span>
            <span class="badge" style="background:#ef444418;color:#fca5a5;border:1px solid #ef444430;">TensorFlow</span>
            <span class="badge" style="background:#f59e0b18;color:#fcd34d;border:1px solid #f59e0b30;">Scikit-learn</span>
            <span class="badge" style="background:#10b98118;color:#6ee7b7;border:1px solid #10b98130;">MLxtend</span>
            <span class="badge" style="background:#8b5cf618;color:#c4b5fd;border:1px solid #8b5cf630;">Plotly</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
