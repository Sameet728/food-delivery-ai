"""
visualization.py — Reusable Plotly chart factory functions.
All charts follow the dark dashboard theme.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Theme palette ─────────────────────────────────────────────────────────────
DARK_BG      = "#0f1117"
CARD_BG      = "#1a1d2e"
ACCENT1      = "#6c63ff"   # purple
ACCENT2      = "#00d4aa"   # teal
ACCENT3      = "#ff6b6b"   # red
ACCENT4      = "#ffd93d"   # yellow
ACCENT5      = "#4ecdc4"   # cyan
PALETTE      = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
                "#a29bfe", "#fd79a8", "#fdcb6e", "#00b894", "#e17055"]

LAYOUT_BASE = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="Inter, sans-serif", color="#e0e0e0", size=12),
    margin=dict(l=30, r=30, t=50, b=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0")),
)

AXIS_STYLE = dict(
    gridcolor="rgba(255,255,255,0.05)",
    tickcolor="#555",
    linecolor="rgba(255,255,255,0.1)",
    title_font=dict(color="#aaa"),
    tickfont=dict(color="#aaa"),
)


def _apply_base(fig, title=""):
    """Apply consistent dark theme to any figure."""
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text=title, font=dict(size=16, color="#ffffff"), x=0.01
    ))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ── KPI Card HTML ─────────────────────────────────────────────────────────────
def kpi_card_html(title: str, value: str, icon: str,
                  color: str = ACCENT1, delta: str = "") -> str:
    delta_html = f'<div style="color:#00d4aa;font-size:12px;margin-top:4px;">{delta}</div>' if delta else ""
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {CARD_BG});
        border: 1px solid {color}44;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px {color}22;
        transition: transform 0.2s ease;
    ">
        <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
        <div style="color:#aaa;font-size:12px;font-weight:500;letter-spacing:0.5px;text-transform:uppercase;">{title}</div>
        <div style="color:#fff;font-size:24px;font-weight:700;margin-top:4px;">{value}</div>
        {delta_html}
    </div>
    """


def kpi_row(kpis: list[dict]):
    """Render a row of KPI cards. Each dict has title, value, icon, color."""
    cols_html = "".join(
        f'<div style="flex:1;min-width:140px;">{kpi_card_html(**kpi)}</div>'
        for kpi in kpis
    )
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px;">{cols_html}</div>'


# ── Revenue Trend ─────────────────────────────────────────────────────────────
def revenue_trend_chart(df: pd.DataFrame) -> go.Figure:
    monthly = (df.groupby(["Year", "Month"])["Revenue"].sum()
                 .reset_index()
                 .sort_values(["Year", "Month"]))
    monthly["Period"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["Period"], y=monthly["Revenue"],
        mode="lines+markers",
        line=dict(color=ACCENT1, width=2.5),
        marker=dict(size=6, color=ACCENT1),
        fill="tozeroy",
        fillcolor=f"rgba(108,99,255,0.12)",
        name="Revenue",
    ))
    return _apply_base(fig, "📈 Monthly Revenue Trend")


# ── Orders by City ────────────────────────────────────────────────────────────
def orders_by_city_chart(df: pd.DataFrame) -> go.Figure:
    city_data = df["City"].value_counts().reset_index()
    city_data.columns = ["City", "Orders"]
    fig = px.bar(city_data, x="City", y="Orders",
                 color="Orders", color_continuous_scale=["#6c63ff","#00d4aa"])
    fig.update_coloraxes(showscale=False)
    return _apply_base(fig, "🏙️ Orders by City")


# ── Cuisine Distribution ──────────────────────────────────────────────────────
def cuisine_donut(df: pd.DataFrame) -> go.Figure:
    cuisine_data = df["Cuisine"].value_counts().reset_index()
    cuisine_data.columns = ["Cuisine", "Count"]
    fig = go.Figure(go.Pie(
        labels=cuisine_data["Cuisine"], values=cuisine_data["Count"],
        hole=0.55, marker=dict(colors=PALETTE),
        textinfo="percent+label",
        textfont=dict(color="#fff"),
    ))
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text="🍽️ Orders by Cuisine", font=dict(size=16, color="#fff"), x=0.01
    ))
    return fig


# ── Delivery Status Donut ─────────────────────────────────────────────────────
def delivery_status_chart(df: pd.DataFrame) -> go.Figure:
    status_data = df["DeliveryStatus"].value_counts().reset_index()
    status_data.columns = ["Status", "Count"]
    colors = {"On Time": ACCENT2, "Late": ACCENT3, "Cancelled": ACCENT4}
    fig = go.Figure(go.Pie(
        labels=status_data["Status"], values=status_data["Count"],
        hole=0.55,
        marker=dict(colors=[colors.get(s, ACCENT1) for s in status_data["Status"]]),
        textinfo="percent+label",
        textfont=dict(color="#fff"),
    ))
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text="📦 Delivery Status", font=dict(size=16, color="#fff"), x=0.01
    ))
    return fig


# ── Payment Mode Bar ──────────────────────────────────────────────────────────
def payment_mode_chart(df: pd.DataFrame) -> go.Figure:
    pay_data = df["PaymentMode"].value_counts().reset_index()
    pay_data.columns = ["Mode", "Count"]
    fig = px.bar(pay_data, x="Mode", y="Count",
                 color="Mode", color_discrete_sequence=PALETTE)
    fig.update_layout(showlegend=False)
    return _apply_base(fig, "💳 Payment Mode Distribution")


# ── Top Restaurants ───────────────────────────────────────────────────────────
def top_restaurants_chart(df: pd.DataFrame, n: int = 10) -> go.Figure:
    rest_data = (df.groupby("RestaurantID")
                   .agg(Orders=("OrderID","count"), AvgRating=("Rating","mean"),
                        Revenue=("Revenue","sum"))
                   .reset_index()
                   .sort_values("Revenue", ascending=False)
                   .head(n))
    rest_data["Label"] = "R" + rest_data["RestaurantID"].astype(str)
    fig = px.bar(rest_data, x="Label", y="Revenue",
                 color="AvgRating", color_continuous_scale=["#ff6b6b","#ffd93d","#00d4aa"],
                 hover_data={"Orders": True, "AvgRating": ":.2f"})
    return _apply_base(fig, "🏆 Top Restaurants by Revenue")


# ── Correlation Heatmap ───────────────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = ["Age", "Rating", "Amount", "DeliveryTime", "ItemsPerOrder"]
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=12, color="#fff"),
    ))
    return _apply_base(fig, "🔥 Correlation Heatmap")


# ── Feature Distribution ──────────────────────────────────────────────────────
def feature_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=[ACCENT1])
    fig.update_traces(marker_line_width=0)
    return _apply_base(fig, f"📊 Distribution of {col}")


# ── Confusion Matrix ─────────────────────────────────────────────────────────
def confusion_matrix_chart(cm: np.ndarray, labels: list[str], title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,"#1a1d2e"],[1,ACCENT1]],
        text=cm, texttemplate="%{text}",
        textfont=dict(size=14, color="#fff"),
    ))
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text=title, font=dict(size=14, color="#fff"), x=0.01
    ))
    fig.update_xaxes(title="Predicted", **AXIS_STYLE)
    fig.update_yaxes(title="Actual", **AXIS_STYLE)
    return fig


# ── PCA Cluster Scatter ────────────────────────────────────────────────────────
def cluster_scatter(X_pca: np.ndarray, labels: np.ndarray,
                    cluster_names: dict) -> go.Figure:
    df_plot = pd.DataFrame({
        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
        "Cluster": [cluster_names.get(l, f"Cluster {l}") for l in labels],
    })
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster",
                     color_discrete_sequence=PALETTE[:4],
                     opacity=0.75)
    return _apply_base(fig, "🔵 Customer Clusters (PCA)")


# ── ANN Training History ──────────────────────────────────────────────────────
def ann_history_chart(history: dict) -> go.Figure:
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Training Loss (MSE)", "Training MAE"))
    fig.add_trace(go.Scatter(x=epochs, y=history.get("loss", []),
                             name="Train Loss", line=dict(color=ACCENT1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history.get("val_loss", []),
                             name="Val Loss",   line=dict(color=ACCENT3, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history.get("mae", []),
                             name="Train MAE",  line=dict(color=ACCENT2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history.get("val_mae", []),
                             name="Val MAE",    line=dict(color=ACCENT4, dash="dash")), row=1, col=2)
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text="🧠 ANN Training History", font=dict(size=16, color="#fff"), x=0.01
    ))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ── Association Rule Network ──────────────────────────────────────────────────
def association_network(rules: pd.DataFrame, max_rules: int = 30) -> go.Figure:
    import networkx as nx

    top = rules.head(max_rules)
    G = nx.DiGraph()
    for _, row in top.iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        G.add_edge(ant, con, weight=row["lift"])

    pos = nx.spring_layout(G, seed=42, k=1.2)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1, color="rgba(108,99,255,0.4)"),
                             hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=20, color=ACCENT1, line=dict(width=1, color=ACCENT2)),
        text=list(G.nodes()), textposition="top center",
        textfont=dict(size=10, color="#fff"),
        hoverinfo="text",
    ))
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="🔗 Item Association Network",
                                 font=dict(size=16, color="#fff"), x=0.01),
                      showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ── Spending vs Frequency Scatter ─────────────────────────────────────────────
def spending_frequency_scatter(cust_df: pd.DataFrame, labels: np.ndarray,
                               cluster_names: dict) -> go.Figure:
    df_plot = cust_df.copy()
    df_plot["Cluster"] = [cluster_names.get(l, f"C{l}") for l in labels]
    fig = px.scatter(df_plot, x="OrderCount", y="TotalSpend",
                     color="Cluster", size="AvgRating",
                     color_discrete_sequence=PALETTE[:4],
                     hover_data=["CustomerID", "AvgDeliveryTime"],
                     opacity=0.8)
    return _apply_base(fig, "💰 Spending vs Order Frequency")


# ── City Heatmap ──────────────────────────────────────────────────────────────
def city_cuisine_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(index="City", columns="Cuisine",
                           values="OrderID", aggfunc="count", fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Viridis",
        texttemplate="%{z}",
        textfont=dict(size=10),
    ))
    return _apply_base(fig, "🗺️ City × Cuisine Order Volume")


# ── Revenue by Cuisine ────────────────────────────────────────────────────────
def revenue_by_cuisine(df: pd.DataFrame) -> go.Figure:
    data = (df.groupby("Cuisine")["Revenue"].sum()
              .reset_index().sort_values("Revenue", ascending=True))
    fig = go.Figure(go.Bar(
        x=data["Revenue"], y=data["Cuisine"],
        orientation="h",
        marker=dict(color=data["Revenue"], colorscale="viridis"),
    ))
    return _apply_base(fig, "💵 Revenue by Cuisine")
