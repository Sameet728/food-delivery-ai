"""
helper.py — UI helpers: CSS injection, download buttons, section headers.
"""

import io
import base64
import pandas as pd
import streamlit as st


# ── Master dark CSS ────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: #0f1117;
    color: #e0e0e0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13152a 0%, #0f1117 100%);
    border-right: 1px solid rgba(108,99,255,0.2);
}
[data-testid="stSidebar"] * { color: #ccc !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #aaa !important; }

/* ── Headers ── */
h1 { background: linear-gradient(135deg,#6c63ff,#00d4aa);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent;
     font-weight:700 !important; }
h2, h3 { color: #e0e0e0 !important; font-weight: 600 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1d2e, #13152a);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 20px rgba(108,99,255,0.1);
}
[data-testid="metric-container"] label { color: #aaa !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #fff !important; font-size: 24px !important; font-weight: 700 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1d2e;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #aaa;
    font-weight: 500;
    padding: 8px 20px;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6c63ff, #8b83ff);
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(108,99,255,0.4);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #5a52d5);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(108,99,255,0.3);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(108,99,255,0.5);
}

/* ── Inputs ── */
.stTextInput > div > input,
.stNumberInput > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #1a1d2e !important;
    border: 1px solid rgba(108,99,255,0.3) !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
}

/* ── DataFrames ── */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(108,99,255,0.2);
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #1a1d2e !important;
    border-radius: 8px !important;
    border: 1px solid rgba(108,99,255,0.2) !important;
    color: #e0e0e0 !important;
    font-weight: 600 !important;
}

/* ── Success / Info / Warning ── */
.stSuccess { background: rgba(0,212,170,0.15) !important; border: 1px solid #00d4aa44; }
.stInfo    { background: rgba(108,99,255,0.15) !important; border: 1px solid #6c63ff44; }
.stWarning { background: rgba(255,211,61,0.15) !important; border: 1px solid #ffd93d44; }
.stError   { background: rgba(255,107,107,0.15) !important; border: 1px solid #ff6b6b44; }

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6c63ff, #00d4aa) !important;
}

/* ── Divider ── */
hr { border-color: rgba(108,99,255,0.2) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #6c63ff; border-radius: 3px; }

/* ── Cards ── */
.insight-card {
    background: linear-gradient(135deg, #1a1d2e, #13152a);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.rec-card {
    background: linear-gradient(135deg, #1a1d2e, #13152a);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: transform 0.2s;
}
.rec-card:hover { transform: translateY(-4px); }
</style>
"""


def inject_css():
    """Inject the global dark theme CSS."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a styled page header."""
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <h1 style="margin-bottom:4px;">{icon} {title}</h1>
        {'<p style="color:#aaa;font-size:14px;margin:0;">'+subtitle+'</p>' if subtitle else ''}
    </div>
    <hr style="border-color:rgba(108,99,255,0.2);margin-bottom:24px;">
    """, unsafe_allow_html=True)


def section_header(title: str, icon: str = ""):
    st.markdown(f"""
    <h3 style="color:#e0e0e0;font-weight:600;margin-top:24px;margin-bottom:12px;">
        {icon} {title}
    </h3>
    """, unsafe_allow_html=True)


def render_kpi_cards(kpis: list[dict]):
    """Render KPI cards using st.metric styled grid."""
    from utils.visualization import kpi_card_html
    n = len(kpis)
    cols = st.columns(n)
    icons = ["📦","👥","🏪","⏱️","💰","⚠️","🍽️","⭐"]
    colors = ["#6c63ff","#00d4aa","#ff6b6b","#ffd93d",
              "#4ecdc4","#a29bfe","#fd79a8","#fdcb6e"]
    for i, (col, kpi) in enumerate(zip(cols, kpis)):
        with col:
            icon  = kpi.get("icon",  icons[i % len(icons)])
            color = kpi.get("color", colors[i % len(colors)])
            st.markdown(kpi_card_html(kpi["title"], kpi["value"], icon, color),
                        unsafe_allow_html=True)


def download_csv(df: pd.DataFrame, filename: str = "data.csv", label: str = "📥 Download CSV"):
    """Render a styled CSV download button."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv,
                       file_name=filename, mime="text/csv")


def download_chart_html(fig, filename: str = "chart.html"):
    """Render a download button for an interactive Plotly chart HTML."""
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    st.download_button("📊 Download Chart (HTML)", data=html_bytes,
                       file_name=filename, mime="text/html")


def badge(text: str, color: str = "#6c63ff") -> str:
    return (f'<span style="background:{color}22;color:{color};'
            f'border:1px solid {color}44;border-radius:20px;'
            f'padding:3px 10px;font-size:11px;font-weight:600;">{text}</span>')


def model_not_ready_warning():
    st.warning("⚙️ Models not yet trained. Go to **🏠 Dashboard** and click **Train Models** to get started.")


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return filtered dataframe."""
    st.sidebar.markdown("### 🔍 Filters")

    cities     = ["All"] + sorted(df["City"].dropna().unique().tolist())
    cuisines   = ["All"] + sorted(df["Cuisine"].dropna().unique().tolist())
    statuses   = ["All"] + sorted(df["DeliveryStatus"].dropna().unique().tolist())
    payments   = ["All"] + sorted(df["PaymentMode"].dropna().unique().tolist())

    sel_city    = st.sidebar.selectbox("🏙️ City",     cities)
    sel_cuisine = st.sidebar.selectbox("🍽️ Cuisine",  cuisines)
    sel_status  = st.sidebar.selectbox("📦 Status",   statuses)
    sel_payment = st.sidebar.selectbox("💳 Payment",  payments)

    min_date = df["OrderDate"].min().date()
    max_date = df["OrderDate"].max().date()
    date_range = st.sidebar.date_input("📅 Date Range",
                                       value=(min_date, max_date),
                                       min_value=min_date, max_value=max_date)

    fdf = df.copy()
    if sel_city    != "All": fdf = fdf[fdf["City"]           == sel_city]
    if sel_cuisine != "All": fdf = fdf[fdf["Cuisine"]        == sel_cuisine]
    if sel_status  != "All": fdf = fdf[fdf["DeliveryStatus"] == sel_status]
    if sel_payment != "All": fdf = fdf[fdf["PaymentMode"]    == sel_payment]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        fdf = fdf[(fdf["OrderDate"] >= start) & (fdf["OrderDate"] <= end)]

    st.sidebar.markdown(f"**{len(fdf):,}** rows after filter")
    return fdf
