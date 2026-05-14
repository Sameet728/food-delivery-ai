"""
5_Customer_Segmentation.py — mirrors Part4.ipynb exactly.
KMeans(n_clusters=3, random_state=42, n_init=10)
Features: OrderFrequency, TotalSpending, AvgDeliveryTime, Cuisine_enc
Remap by avg spending: 0=High-Value, 1=Regular, 2=Occasional
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
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Customer Segmentation | Food Delivery AI", page_icon="👥", layout="wide")

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
.seg-card{border-radius:14px;padding:1.4rem 1.6rem;border:1px solid;margin-bottom:.5rem;}
.seg-title{font-size:1.2rem;font-weight:700;}
.seg-stat{font-size:.85rem;margin-top:.3rem;opacity:.8;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>👥 Customer Segmentation</h1>
  <p>KMeans k=3 + Agglomerative Clustering — mirrors Part4.ipynb (random_state=42, n_init=10)</p>
</div>
""", unsafe_allow_html=True)

from utils.nb_logic import (train_clustering, CLUSTER_LABELS, CLUSTER_COLORS,
                             CLUST_FEATS, compute_customer_segment_insights)

with st.spinner("🔵 Running clustering (first load only)…"):
    cust, km, sc_cust, le_cuisine = train_clustering()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{len(cust):,}")
c2.metric("Segments", "3")
agree = (cust["Cluster"] == cust["Hierarchical_Cluster"]).mean() * 100
c3.metric("KMeans↔Hierarch. Agreement", f"{agree:.1f}%")
c4.metric("High-Value Customers",
          f"{(cust['Cluster']==0).sum():,}")

st.markdown("")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Cluster Scatter", "🌡 Feature Heatmap", "📋 Summary Table", "🔍 Predict Segment"])

# ── Tab 1: PCA scatter (mirrors Part4 Cell 9) ─────────────────────────────────
with tab1:
    X_raw = sc_cust.transform(cust[CLUST_FEATS])
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X_raw)
    cust["PCA1"] = X2[:, 0]
    cust["PCA2"] = X2[:, 1]
    cust["Segment"] = cust["Cluster"].map(CLUSTER_LABELS)

    color_map = {v: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()}

    fig1 = px.scatter(
        cust, x="PCA1", y="PCA2", color="Segment",
        color_discrete_map=color_map, opacity=0.7,
        title=f"K-Means Clusters (PCA 2D)  |  Var explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%  PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
        labels={"PCA1":"Principal Component 1","PCA2":"Principal Component 2"},
    )
    fig1.update_traces(marker=dict(size=4))
    fig1.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                       legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig1, use_container_width=True)

    # Spending vs Frequency scatter (mirrors Part4 Cell 9 ax2)
    fig2 = px.scatter(
        cust, x="OrderFrequency", y="TotalSpending", color="Segment",
        color_discrete_map=color_map, opacity=0.7,
        title="Total Spending vs Order Frequency",
        labels={"OrderFrequency":"Order Frequency","TotalSpending":"Total Spending (₹)"},
    )
    fig2.update_traces(marker=dict(size=4))
    fig2.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                       legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Feature heatmap (mirrors Part4 Cell 11) ───────────────────────────
with tab2:
    heat = cust.groupby("Cluster").agg(
        Frequency  = ("OrderFrequency",  "mean"),
        Spending   = ("TotalSpending",   "mean"),
        DelivTime  = ("AvgDeliveryTime", "mean"),
        Rating     = ("AvgRating",       "mean"),
    ).rename(index=CLUSTER_LABELS)

    heat_norm = (heat - heat.min()) / (heat.max() - heat.min())

    fig = go.Figure(go.Heatmap(
        z=heat_norm.values,
        x=heat_norm.columns.tolist(),
        y=heat_norm.index.tolist(),
        text=heat.round(1).values,
        texttemplate="%{text}",
        colorscale="RdYlGn", showscale=True,
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Feature Summary Heatmap (normalised, raw values annotated)",
        height=380, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Violin plots per segment
    melt = cust.melt(id_vars="Segment", value_vars=["OrderFrequency","TotalSpending","AvgDeliveryTime","AvgRating"])
    fig_v = px.violin(melt, x="Segment", y="value", color="Segment",
                      facet_col="variable", color_discrete_map=color_map,
                      title="Distribution per Segment",
                      labels={"value":"Value","variable":""})
    fig_v.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                        showlegend=False)
    st.plotly_chart(fig_v, use_container_width=True)

# ── Tab 3: Summary table ──────────────────────────────────────────────────────
with tab3:
    seg_df = compute_customer_segment_insights()
    st.markdown("**Cluster Summary**")
    st.dataframe(seg_df.set_index("Segment").style.format("{:.2f}"), use_container_width=True)

    # Distribution pie
    dist = cust["Segment"].value_counts().reset_index()
    dist.columns = ["Segment", "Count"]
    fig_p = px.pie(dist, names="Segment", values="Count",
                   title="Customer Distribution by Segment",
                   color="Segment", color_discrete_map=color_map)
    fig_p.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e2e8f0"),
                        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_p, use_container_width=True)

    st.download_button(
        "⬇️ Download Clustered Customers CSV",
        cust[["CustomerID","OrderFrequency","TotalSpending","AvgDeliveryTime",
              "PreferredCuisine","AvgRating","Cluster","Segment","Hierarchical_Cluster"]
             ].to_csv(index=False).encode("utf-8"),
        "clustered_customers.csv", "text/csv"
    )

# ── Tab 4: Segment predictor (mirrors Part4 Cell 16 widget) ──────────────────
with tab4:
    st.markdown("### 🔍 Predict Customer Segment")
    st.caption("Mirrors Part4.ipynb prediction widget — uses same KMeans model & scaler")

    cuisines = sorted(cust["PreferredCuisine"].unique())
    c1, c2 = st.columns(2)
    with c1:
        freq    = st.slider("Order Frequency", 1, 50, 10)
        spend   = st.slider("Total Spending (₹)", 100, 20000, 3000, 100)
    with c2:
        deliv   = st.slider("Avg Delivery Time (min)", 10, 90, 40)
        cuisine = st.selectbox("Preferred Cuisine", cuisines)

    if st.button("🔍 Predict Segment", type="primary", use_container_width=True):
        try:
            cuisine_enc = le_cuisine.transform([cuisine])[0]
            raw = np.array([[freq, spend, deliv, cuisine_enc]])
            scaled = sc_cust.transform(raw)
            pred_raw = km.predict(scaled)[0]

            # Apply same remap
            spend_order = (cust.groupby("Cluster_raw")["TotalSpending"]
                               .mean().sort_values(ascending=False).index.tolist())
            remap = {spend_order[0]: 0, spend_order[1]: 1, spend_order[2]: 2}
            pred = remap.get(pred_raw, pred_raw)

            label  = CLUSTER_LABELS[pred]
            color  = CLUSTER_COLORS[pred]
            msgs   = {
                0: "VIP customer! Offer premium deals and loyalty rewards.",
                1: "Loyal regular! Provide combo deals and discounts.",
                2: "Occasional user — send re-engagement offers.",
            }
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.05);border-radius:16px;
                        padding:1.5rem 2rem;border:2px solid {color};text-align:center;margin-top:1rem">
              <div style="font-size:2.5rem;margin-bottom:.5rem">{label}</div>
              <div style="color:{color};font-size:1.1rem;font-weight:600">{msgs[pred]}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
