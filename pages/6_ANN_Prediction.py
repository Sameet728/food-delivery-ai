"""
6_ANN_Prediction.py — mirrors Part5.ipynb exactly.
ANN: Dense(64)→Dense(32)→Dense(16)→Dense(1)
Features: ItemsPerOrder, Rating, City, Cuisine, PaymentMode
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

st.set_page_config(page_title="ANN Prediction | Food Delivery AI", page_icon="🧠", layout="wide")

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
.kpi{background:rgba(255,255,255,.05);border-radius:14px;padding:1.2rem 1.4rem;
     border:1px solid rgba(255,255,255,.08);text-align:center;}
.kpi .val{color:#e2e8f0;font-size:1.8rem;font-weight:700;}
.kpi .lbl{color:#64748b;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧠 ANN Delivery Time Prediction</h1>
  <p>Dense(64)→Dense(32)→Dense(16)→Dense(1) · Adam · MSE · EarlyStopping(patience=5) · mirrors Part5.ipynb exactly</p>
</div>
""", unsafe_allow_html=True)

from utils.nb_logic import train_ann, FEATURES_ANN, CAT_ANN, NUM_ANN

with st.spinner("🧠 Training ANN (first load only)…"):
    ann, history, encoders, scaler, X_test, y_test, y_pred, mae, rmse, r2 = train_ann()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="kpi"><div class="val">{mae:.2f} min</div><div class="lbl">MAE</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi"><div class="val">{rmse:.2f} min</div><div class="lbl">RMSE</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi"><div class="val">{r2:.4f}</div><div class="lbl">R² Score</div></div>', unsafe_allow_html=True)

st.markdown("")

tab1, tab2, tab3, tab4 = st.tabs(["📉 Loss Curve", "🎯 Actual vs Predicted", "📊 Feature Importance", "🔮 Predict"])

# ── Tab 1: Loss Curve (mirrors Part5 Cell 12) ─────────────────────────────────
with tab1:
    epochs = list(range(1, len(history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["loss"], name="Training Loss",
                             line=dict(color="#3b82f6", width=2.5), mode="lines"))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Validation Loss",
                                 line=dict(color="#f59e0b", width=2.5, dash="dash"), mode="lines"))
    fig.update_layout(
        title="Training & Validation Loss (MSE) — Part5.ipynb",
        xaxis_title="Epoch", yaxis_title="Loss (MSE)",
        height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if "mae" in history:
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(x=epochs, y=history["mae"], name="Training MAE",
                                     line=dict(color="#10b981", width=2), mode="lines"))
        if "val_mae" in history:
            fig_mae.add_trace(go.Scatter(x=epochs, y=history["val_mae"], name="Val MAE",
                                         line=dict(color="#f87171", width=2, dash="dash"), mode="lines"))
        fig_mae.update_layout(title="MAE per Epoch", xaxis_title="Epoch", yaxis_title="MAE",
                              height=350, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                              legend=dict(bgcolor="rgba(0,0,0,0)"),
                              yaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                              xaxis=dict(gridcolor="rgba(255,255,255,.08)"))
        st.plotly_chart(fig_mae, use_container_width=True)

# ── Tab 2: Actual vs Predicted scatter (mirrors Part5 Cell 13) ───────────────
with tab2:
    sample_size = min(1000, len(y_test))
    idx = np.random.RandomState(42).choice(len(y_test), sample_size, replace=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test[idx], y=y_pred[idx], mode="markers",
        marker=dict(color="#6366f1", size=4, opacity=0.6),
        name="Predictions",
    ))
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color="#f59e0b", dash="dash", width=2),
                             name="Perfect Fit"))
    fig.update_layout(
        title="Actual vs Predicted Delivery Time — Part5.ipynb",
        xaxis_title="Actual (min)", yaxis_title="Predicted (min)",
        height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.08)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    residuals = y_pred - y_test
    fig_r = px.histogram(residuals[idx], nbins=50, title="Residuals Distribution",
                         color_discrete_sequence=["#8b5cf6"])
    fig_r.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                        xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,.08)"))
    st.plotly_chart(fig_r, use_container_width=True)

# ── Tab 3: Feature importance via RF surrogate (mirrors Part5 Cell 14) ────────
with tab3:
    from utils.nb_logic import train_delivery_regressor
    rf_reg, le_reg, sc_reg, X_te_r, y_te_r, y_pred_r, mae_r, rmse_r, r2_r, feat_imp = train_delivery_regressor()

    fig = px.bar(
        feat_imp.reset_index(),
        x=0, y="index", orientation="h",
        color=0, color_continuous_scale="Viridis",
        title="Feature Importance — Random Forest Regressor (Part6 surrogate)",
        labels={"index": "Feature", 0: "Importance"},
    )
    fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                      showlegend=False,
                      xaxis=dict(gridcolor="rgba(255,255,255,.08)"))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("RF MAE",  f"{mae_r:.2f} min")
    c2.metric("RF RMSE", f"{rmse_r:.2f} min")
    c3.metric("RF R²",   f"{r2_r:.4f}")

# ── Tab 4: Live ANN predictor (mirrors Part5 widget) ─────────────────────────
with tab4:
    st.markdown("### 🔮 Predict Delivery Time")
    st.caption("Uses the exact trained ANN — same architecture and weights as Part5.ipynb")

    c1, c2 = st.columns(2)
    with c1:
        items   = st.slider("Items per Order", 1, 4, 2)
        rating  = st.slider("Restaurant Rating", 1.0, 5.0, 3.5, 0.1)
    with c2:
        city    = st.selectbox("City", sorted(encoders["City"].classes_))
        cuisine = st.selectbox("Cuisine", sorted(encoders["Cuisine"].classes_))
        payment = st.selectbox("Payment Mode", sorted(encoders["PaymentMode"].classes_))

    if st.button("⚡ Predict Delivery Time", type="primary", use_container_width=True):
        try:
            NUM_IDX = [FEATURES_ANN.index(c) for c in NUM_ANN]
            city_enc    = encoders["City"].transform([city])[0]
            cuisine_enc = encoders["Cuisine"].transform([cuisine])[0]
            payment_enc = encoders["PaymentMode"].transform([payment])[0]

            row = np.array([[items, rating, city_enc, cuisine_enc, payment_enc]], dtype=float)
            row[:, NUM_IDX] = scaler.transform(row[:, NUM_IDX])
            pred_time = ann.predict(row)[0]

            st.markdown(f"""
            <div style="background:rgba(255,255,255,.05);border-radius:16px;
                        padding:1.5rem 2rem;border:2px solid #6366f1;text-align:center;margin-top:1rem">
              <div style="font-size:3rem">⏱️</div>
              <div style="color:#a5b4fc;font-size:2rem;font-weight:700">{pred_time:.1f} minutes</div>
              <div style="color:#94a3b8;font-size:.9rem;margin-top:.3rem">Predicted delivery time (ANN)</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    with st.expander("ℹ️ ANN Architecture (Part5.ipynb)"):
        st.code("""
Sequential([
    Dense(64, activation='relu', input_shape=(5,)),   # ItemsPerOrder, Rating, City, Cuisine, PaymentMode
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1,  activation='linear'),   # Delivery time in minutes
])
compile(optimizer='adam', loss='mse', metrics=['mae'])
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
train_test_split(test_size=0.2, random_state=42)  # No stratify (regression)
        """, language="python")
