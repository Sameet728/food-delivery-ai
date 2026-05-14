"""
8_Recommendation_System.py — Smart Recommendations (Option B)
Uses association rules computed live from nb_logic.py (mirrors Part2.ipynb exactly).
No notebook execution required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="Recommendations | Food Delivery AI", page_icon="🎯", layout="wide")

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
.rec-card{background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);
          border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:.8rem;}
.rec-item{color:#a5b4fc;font-size:1.1rem;font-weight:600;}
.rec-stats{color:#64748b;font-size:.82rem;margin-top:.3rem;}
.kpi{background:rgba(255,255,255,.05);border-radius:14px;padding:1.2rem 1.5rem;
     border:1px solid rgba(255,255,255,.08);text-align:center;}
.kpi .val{color:#e2e8f0;font-size:1.8rem;font-weight:700;}
.kpi .lbl{color:#64748b;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎯 Intelligent Recommendation System</h1>
  <p>Powered by Apriori association rules mined live from your dataset — mirrors Part2.ipynb exactly.
     Recommends items based on real co-purchase patterns.</p>
</div>
""", unsafe_allow_html=True)

# ── Load rules from nb_logic ──────────────────────────────────────────────────
from utils.nb_logic import compute_association_rules, load_and_preprocess

with st.spinner("⛏ Mining association rules (first load only)…"):
    rules, freq_items, item_freq, tx_df = compute_association_rules()

df = load_and_preprocess()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    [f"{df['OrderID'].nunique():,}", f"{len(item_freq):,}",
     f"{len(rules):,}", f"{rules['lift'].max():.3f}" if len(rules) > 0 else "—"],
    ["Total Orders", "Unique Items", "Rules Mined", "Max Lift"],
):
    with col:
        st.markdown(f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True)

st.markdown("")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommender", "🕸 Rule Network", "📊 Rules Analysis", "📋 All Rules"])

# ── Tab 1: Live Recommender (mirrors Part2 Cell 12 widget) ───────────────────
with tab1:
    st.markdown("### 🎯 Get Personalised Recommendations")
    st.caption("Select items you ordered → get what others also ordered (based on Apriori rules)")

    if len(rules) == 0:
        st.error("No association rules found. Check dataset.")
        st.stop()

    all_items_list = sorted(item_freq.keys())
    col_l, col_r = st.columns([2, 1])
    with col_l:
        selected_item = st.selectbox("🛒 Item ordered:", all_items_list)
    with col_r:
        min_lift_val = float(rules["lift"].min())
        max_lift_val = float(rules["lift"].max())
        min_lift = st.slider(
            "Min Lift",
            min_value=round(min_lift_val, 2),
            max_value=round(max_lift_val, 2),
            value=round(min_lift_val, 2),
            step=0.01,
        )

    if selected_item:
        # Match by antecedents_str (single item rules)
        mask = (
            rules["antecedents_str"].str.contains(selected_item, case=False, na=False)
            & (rules["lift"] >= min_lift)
        )
        recs = rules[mask].sort_values("confidence", ascending=False).head(8)

        if len(recs) == 0:
            st.info(f"No recommendations found for **{selected_item}**. Try lowering Min Lift.")
        else:
            st.markdown(f"**Customers who ordered `{selected_item}` also ordered:**")
            for i, (_, r) in enumerate(recs.iterrows(), 1):
                st.markdown(f"""
                <div class="rec-card">
                  <div class="rec-item">#{i} &nbsp; {r['consequents_str']}</div>
                  <div class="rec-stats">
                    Lift: <b>{r['lift']:.3f}</b> &nbsp;|&nbsp;
                    Confidence: <b>{r['confidence']:.3f}</b> &nbsp;|&nbsp;
                    Support: <b>{r['support']:.4f}</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Item Frequency bar ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Item Popularity (from your dataset)**")
    top_items = pd.DataFrame(list(item_freq.items()), columns=["Item", "Count"]).sort_values("Count", ascending=True)
    fig = px.bar(top_items, x="Count", y="Item", orientation="h",
                 color="Count", color_continuous_scale="Purples",
                 title="Item Order Frequency")
    fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e2e8f0"), showlegend=False,
                      xaxis=dict(gridcolor="rgba(255,255,255,.08)"))
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Interactive Association Network (mirrors Part7 Cell 11) ────────────
with tab2:
    st.markdown("### 🕸 Association Rule Network")
    st.caption("Top 20 rules by lift — visualised as a directed network graph (mirrors Part7.ipynb Cell 11)")

    if len(rules) > 0:
        top_net = rules.sort_values("lift", ascending=False).head(20)

        G = nx.DiGraph()
        for _, row in top_net.iterrows():
            G.add_edge(row["antecedents_str"], row["consequents_str"],
                       weight=row["lift"], conf=row["confidence"])

        pos = nx.spring_layout(G, seed=42, k=2)

        # Edges
        edge_x, edge_y = [], []
        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                line=dict(width=1.5, color="rgba(165,180,252,.4)"),
                                hoverinfo="none")

        # Nodes
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = list(G.nodes())
        node_degree = [G.degree(n) for n in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=[12 + d * 4 for d in node_degree],
                        color=node_degree, colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Degree", thickness=12)),
            text=node_text, textposition="top center",
            textfont=dict(color="#e2e8f0", size=11),
            hoverinfo="text",
        )

        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            title="Top 20 Association Rules — Network Graph",
            showlegend=False, height=560,
            paper_bgcolor="rgba(15,17,23,1)", plot_bgcolor="rgba(15,17,23,1)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.info("No rules to display.")

# ── Tab 3: Support / Confidence / Lift charts (mirrors Part2 Cell 11) ─────────
with tab3:
    st.markdown("### 📊 Rules Analysis (mirrors Part2.ipynb Cell 11)")

    if len(rules) > 0:
        col1, col2 = st.columns(2)

        with col1:
            top_lift = rules.nlargest(15, "lift")
            fig = px.bar(top_lift, x="lift", y="rule", orientation="h",
                         color="lift", color_continuous_scale="Blues",
                         title="Top 15 Rules by Lift")
            fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                              showlegend=False, yaxis=dict(categoryorder="total ascending"),
                              xaxis=dict(gridcolor="rgba(255,255,255,.08)"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top_conf = rules.nlargest(15, "confidence")
            fig2 = px.bar(top_conf, x="confidence", y="rule", orientation="h",
                          color="confidence", color_continuous_scale="Greens",
                          title="Top 15 Rules by Confidence")
            fig2.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                               showlegend=False, yaxis=dict(categoryorder="total ascending"),
                               xaxis=dict(gridcolor="rgba(255,255,255,.08)"))
            st.plotly_chart(fig2, use_container_width=True)

        # Support vs Confidence scatter
        fig3 = px.scatter(rules, x="support", y="confidence", color="lift",
                          size="lift", hover_data=["rule"],
                          color_continuous_scale="Plasma",
                          title="Support vs Confidence (colour = Lift)",
                          labels={"support":"Support","confidence":"Confidence","lift":"Lift"})
        fig3.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                           xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: Full rules table ───────────────────────────────────────────────────
with tab4:
    st.markdown(f"**{len(rules):,} Association Rules** (Apriori, min_support=0.01, min_confidence=0.10, lift≥0.5)")

    search = st.text_input("🔍 Search rules (e.g. 'Biryani'):")
    display_df = rules[["rule", "support", "confidence", "lift"]].copy()
    if search:
        display_df = display_df[display_df["rule"].str.contains(search, case=False, na=False)]

    st.dataframe(
        display_df.sort_values("lift", ascending=False).style.format({
            "support": "{:.4f}", "confidence": "{:.3f}", "lift": "{:.3f}"
        }),
        use_container_width=True, height=480,
    )

    st.download_button(
        "⬇️ Download Rules CSV",
        rules[["rule", "support", "confidence", "lift",
               "antecedents_str", "consequents_str"]].to_csv(index=False).encode("utf-8"),
        "association_rules.csv", "text/csv",
    )
