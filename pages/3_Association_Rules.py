"""
3_Association_Rules.py — mirrors Part2.ipynb exactly.
Same Apriori params: min_support=0.01, min_confidence=0.10, lift>=1.0
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Association Rules | Food Delivery AI", page_icon="🔗", layout="wide")

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
.kpi .lbl{color:#64748b;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;}
.rec-card{background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);
          border-radius:14px;padding:1rem 1.4rem;margin-bottom:.6rem;}
.rec-item{color:#a5b4fc;font-size:1.05rem;font-weight:600;}
.rec-stats{color:#64748b;font-size:.82rem;margin-top:.2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔗 Association Rule Mining</h1>
  <p>Apriori algorithm — min_support=0.01, min_confidence=0.10, lift≥0.5 &nbsp;·&nbsp; Mirrors Part2.ipynb exactly.</p>
</div>
""", unsafe_allow_html=True)

# ── Load rules ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.nb_logic import compute_association_rules, load_and_preprocess

with st.spinner("⛏ Mining association rules (first load only)…"):
    result = compute_association_rules()
    rules, freq_items, item_freq, tx_df = result

df = load_and_preprocess()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_orders    = df["OrderID"].nunique()
unique_items    = len(item_freq)
total_rules     = len(rules)
max_lift        = rules["lift"].max() if len(rules) else 0

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    [f"{total_orders:,}", f"{unique_items:,}", f"{total_rules:,}", f"{max_lift:.2f}"],
    ["Total Transactions", "Unique Items", "Rules Mined", "Max Lift"],
):
    with col:
        st.markdown(f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Top Rules", "📈 Support vs Confidence", "🍕 Item Frequency", "🎯 Recommender"])

# ── Tab 1: Top Rules bar chart ────────────────────────────────────────────────
with tab1:
    if len(rules) == 0:
        st.info("No rules found with current thresholds.")
    else:
        top_n = st.slider("Show top N rules by Lift", 10, min(50, len(rules)), 20)
        top = rules.head(top_n).copy()

        fig = make_subplots(rows=1, cols=3, subplot_titles=["Top Rules by Lift", "Top Rules by Confidence", "Top Rules by Support"])

        top_lift = rules.nlargest(15, "lift")
        top_conf = rules.nlargest(15, "confidence")
        top_sup  = rules.nlargest(15, "support")

        palette = px.colors.qualitative.Plotly

        fig.add_trace(go.Bar(y=top_lift["rule"], x=top_lift["lift"],
                             orientation="h", marker_color=palette[0],
                             name="Lift"), row=1, col=1)
        fig.add_trace(go.Bar(y=top_conf["rule"], x=top_conf["confidence"],
                             orientation="h", marker_color=palette[1],
                             name="Confidence"), row=1, col=2)
        fig.add_trace(go.Bar(y=top_sup["rule"], x=top_sup["support"],
                             orientation="h", marker_color=palette[2],
                             name="Support"), row=1, col=3)

        fig.update_layout(height=520, showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e2e8f0"))
        fig.update_xaxes(gridcolor="rgba(255,255,255,.08)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**All Rules Table**")
        disp = rules[["rule","support","confidence","lift"]].copy()
        disp.columns = ["Rule","Support","Confidence","Lift"]
        st.dataframe(disp.style.format({"Support":"{:.4f}","Confidence":"{:.3f}","Lift":"{:.3f}"}),
                     use_container_width=True, height=350)

# ── Tab 2: Scatter (Support vs Confidence, coloured by Lift) ─────────────────
with tab2:
    if len(rules) > 0:
        fig = px.scatter(
            rules, x="support", y="confidence", color="lift",
            size="lift", hover_data=["rule","lift"],
            color_continuous_scale="Viridis",
            labels={"support":"Support","confidence":"Confidence","lift":"Lift"},
            title="Support vs Confidence (colour = Lift)",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e2e8f0"), height=480)
        fig.update_traces(marker=dict(opacity=0.75, line=dict(width=0)))
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Item frequency ─────────────────────────────────────────────────────
with tab3:
    top_items = pd.DataFrame(list(item_freq.items()), columns=["Item","Count"]).head(25)
    fig = px.bar(top_items, x="Count", y="Item", orientation="h",
                 color="Count", color_continuous_scale="Blues",
                 title="Top 25 Most Ordered Items")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e2e8f0"), height=560, showlegend=False,
                      yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

    # Cuisine analysis (mirrors Part2 Cell 13)
    st.markdown("---")
    st.markdown("### 🍕 Cuisine Analysis")
    cuisine_stats = df.groupby("Cuisine").agg(
        Avg_Amount  = ("TotalOrderValue", "mean"),
        Avg_Items   = ("ItemsPerOrder",   "mean"),
        Order_Count = ("OrderID",         "count"),
    ).round(2).reset_index()

    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=["Avg Order Amount by Cuisine", "Avg Items per Order by Cuisine"])
    palette6 = px.colors.qualitative.Plotly
    fig2.add_trace(go.Bar(x=cuisine_stats["Cuisine"], y=cuisine_stats["Avg_Amount"],
                          marker_color=palette6[:len(cuisine_stats)], name="Avg Amount"),
                   row=1, col=1)
    fig2.add_trace(go.Bar(x=cuisine_stats["Cuisine"], y=cuisine_stats["Avg_Items"],
                          marker_color=palette6[3:3+len(cuisine_stats)], name="Avg Items"),
                   row=1, col=2)
    fig2.update_layout(height=420, showlegend=False,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#e2e8f0"))
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 4: Recommender (mirrors Part2 Cell 12 widget) ────────────────────────
with tab4:
    st.markdown("### 🎯 Food Recommendation Engine")
    st.caption("Select items ordered — get recommendations based on association rules (same logic as Part2 notebook widget)")

    if len(rules) > 0 and "antecedents_str" in rules.columns:
        all_items = sorted(rules["antecedents_str"].unique())
        selected = st.selectbox("🛒 Item ordered:", all_items)
        min_lift_r = st.slider("Min Lift", 0.5, float(max(rules["lift"].max(), 0.9)), 0.5, 0.05, key="rec_lift")

        if selected:
            mask = (rules["antecedents_str"] == selected) & (rules["lift"] >= min_lift_r)
            recs = rules[mask].sort_values("lift", ascending=False).head(8)

            if len(recs) == 0:
                st.info("No recommendations found. Try lowering Min Lift.")
            else:
                st.markdown(f"**Customers who ordered `{selected}` also bought:**")
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
    else:
        st.info("No rules available for recommendations.")
