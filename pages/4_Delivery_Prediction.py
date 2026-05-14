"""
4_Delivery_Prediction.py — mirrors Part3.ipynb exactly.
LR, DT, RF, SVM with same features, same split, same random_state=42.
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
from sklearn.metrics import confusion_matrix, classification_report, f1_score

st.set_page_config(page_title="Classification | Food Delivery AI", page_icon="🌲", layout="wide")

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
.metric-box{background:rgba(255,255,255,.05);border-radius:14px;padding:1.2rem 1.4rem;
            border:1px solid rgba(255,255,255,.08);text-align:center;}
.metric-box .val{font-size:1.8rem;font-weight:700;color:#e2e8f0;}
.metric-box .lbl{font-size:.78rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;}
.model-card{background:rgba(255,255,255,.04);border-radius:14px;padding:1rem 1.4rem;
            border:1px solid rgba(255,255,255,.08);margin-bottom:.8rem;}
.model-name{color:#a5b4fc;font-weight:600;font-size:1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🌲 Delivery Status Classification</h1>
  <p>LR · Decision Tree · Random Forest · SVM — mirrors Part3.ipynb (random_state=42, test_size=0.2, stratify=y)</p>
</div>
""", unsafe_allow_html=True)

# ── Train models ──────────────────────────────────────────────────────────────
from utils.nb_logic import train_classifiers, FEATURES_CLS

MODEL_COLORS = {
    "Logistic Regression": "#3498db",
    "Decision Tree":       "#9b59b6",
    "Random Forest":       "#2ecc71",
    "SVM":                 "#e67e22",
}

with st.spinner("🌲 Training classifiers (first load only)…"):
    trained, results, le_target, encoders, scaler, X_te, y_te, TARGET_CLASSES = train_classifiers()

# ── Model selector ────────────────────────────────────────────────────────────
selected_model = st.selectbox("Select model to inspect:", list(results.keys()))
res = results[selected_model]

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, metric in zip([c1, c2, c3, c4], ["Accuracy", "Precision", "Recall", "F1 Score"]):
    with col:
        st.markdown(f'<div class="metric-box"><div class="val">{res[metric]:.1f}%</div><div class="lbl">{metric}</div></div>', unsafe_allow_html=True)

st.markdown("")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🔥 Confusion Matrix", "📈 Per-Class F1", "🔮 Predict"])

# ── Tab 1: Model comparison bar chart ────────────────────────────────────────
with tab1:
    df_res = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

    fig = go.Figure()
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors  = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"]

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric, x=df_res["Model"], y=df_res[metric],
            marker_color=color, opacity=0.9,
        ))

    fig.update_layout(
        barmode="group", height=420, title="Model Performance Comparison",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.dataframe(df_res.set_index("Model").style.format("{:.2f}%"), use_container_width=True)

# ── Tab 2: Confusion matrix ───────────────────────────────────────────────────
with tab2:
    model = trained[selected_model]
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig = go.Figure(go.Heatmap(
        z=cm, x=TARGET_CLASSES, y=TARGET_CLASSES,
        text=cm, texttemplate="%{text}",
        colorscale="Blues", showscale=True,
        hoverongaps=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {selected_model}",
        xaxis_title="Predicted", yaxis_title="Actual",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Classification Report"):
        report = classification_report(
            le_target.inverse_transform(y_te),
            le_target.inverse_transform(y_pred),
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ── Tab 3: Per-class F1 (mirrors Part3 radar/bar) ────────────────────────────
with tab3:
    per_class = {}
    for name, m in trained.items():
        yp = m.predict(X_te)
        per_class[name] = dict(zip(TARGET_CLASSES,
                                   f1_score(y_te, yp, average=None, zero_division=0) * 100))

    df_pc = pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "Model"})
    df_pc_melt = df_pc.melt(id_vars="Model", var_name="Class", value_name="F1 Score")

    fig = px.bar(df_pc_melt, x="Class", y="F1 Score", color="Model",
                 barmode="group", title="Per-Class F1 Score — All Models",
                 color_discrete_map=MODEL_COLORS)
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                      yaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance for RF
    rf = trained["Random Forest"]
    if hasattr(rf, "feature_importances_"):
        fi = pd.Series(rf.feature_importances_, index=FEATURES_CLS).sort_values(ascending=True)
        fig_fi = px.bar(fi.reset_index(), x=0, y="index",
                        orientation="h", title="Random Forest Feature Importance",
                        color=0, color_continuous_scale="Viridis",
                        labels={"index": "Feature", 0: "Importance"})
        fig_fi.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                             showlegend=False,
                             xaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig_fi, use_container_width=True)

# ── Tab 4: Live prediction widget (mirrors Part3 Cell 32) ────────────────────
with tab4:
    st.markdown("### 🔮 Predict Delivery Status")
    st.caption("Mirrors the Part3.ipynb interactive widget exactly.")

    col1, col2 = st.columns(2)
    with col1:
        age         = st.slider("Customer Age", 18, 60, 30)
        rating      = st.slider("Restaurant Rating", 2.5, 5.0, 3.5, 0.1)
        order_val   = st.slider("Order Value (₹)", 50, 850, 300, 10)
        delivery_t  = st.slider("Delivery Time (min)", 15, 70, 40)
        items       = st.selectbox("Items per Order", [1, 2, 3, 4], index=1)
    with col2:
        city        = st.selectbox("City", ["Ahmedabad","Bangalore","Chennai","Delhi","Hyderabad","Kolkata","Mumbai","Pune"])
        cuisine     = st.selectbox("Cuisine", ["Chinese","Continental","Fast Food","Italian","North Indian","South Indian"])
        payment     = st.selectbox("Payment Mode", ["Card","Cash","UPI"])
        gender      = st.selectbox("Gender", ["Female","Male"])
        order_date  = st.text_input("Order Date (DD-MM-YYYY)", "15-06-2025")
        model_sel   = st.selectbox("Model", list(trained.keys()))

    if st.button("🔮 Predict", type="primary", use_container_width=True):
        try:
            dt = pd.to_datetime(order_date, dayfirst=True)
            row = {
                "Age": age, "City": city, "Rating": rating,
                "Cuisine": cuisine, "ItemsPerOrder": items,
                "PaymentMode": payment, "TotalOrderValue": order_val,
                "DeliveryTime": delivery_t,
                "order_dayofweek": dt.dayofweek,
                "order_month": dt.month, "Gender": gender,
            }
            inp = pd.DataFrame([row])
            from utils.nb_logic import CAT_COLS_CLS, NUM_COLS_CLS, FEATURES_CLS
            for col in CAT_COLS_CLS:
                inp[col] = encoders[col].transform([str(inp[col].values[0])])
            inp[NUM_COLS_CLS] = scaler.transform(inp[NUM_COLS_CLS])
            pred = trained[model_sel].predict(inp[FEATURES_CLS])
            status = le_target.inverse_transform(pred)[0]
            emoji = {"On Time": "✅", "Late": "⏰", "Cancelled": "❌"}.get(status, "❓")
            color = {"On Time": "#27ae60", "Late": "#c0392b", "Cancelled": "#e67e22"}.get(status, "#fff")
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.05);border-radius:16px;padding:1.5rem 2rem;
                        border:2px solid {color};text-align:center;margin-top:1rem">
              <div style="font-size:3rem">{emoji}</div>
              <div style="color:{color};font-size:1.5rem;font-weight:700">{status}</div>
              <div style="color:#94a3b8;font-size:.9rem;margin-top:.3rem">
                Model: {model_sel}
              </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
