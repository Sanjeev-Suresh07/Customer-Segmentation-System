import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "cleaned_customer_behavior.csv"
NEW_CUSTOMERS_FILE = DATA_DIR / "new_customers.csv"

SCALER_FILE = MODEL_DIR / "scaler.pkl"
ENCODER_FILE = MODEL_DIR / "encoder.pkl"
MODEL_FILE = MODEL_DIR / "kmeans_model.pkl"

DATA_DIR.mkdir(exist_ok=True)

# =========================================================
# COLORS  —  EarthGuard-inspired eco-dark palette
# =========================================================
BG          = "#0b191e"
SURFACE     = "#0f2027"
CARD_BG     = "rgba(255,255,255,0.04)"
GREEN       = "#86d028"          # brand accent
GREEN_DARK  = "#5a9216"
CREAM       = "#f0f4f8"
SAGE        = "#b2cfc4"
PEACH       = "#d9917a"
WHEAT       = "#c9b87c"
TEXT        = "#e8f1ec"
MUTED       = "rgba(255,255,255,0.55)"

SEGMENT_NAMES = {
    0: "Bronze Regular Customers",
    1: "Gold High Value Customers",
    2: "Silver Regular Customers",
    3: "Gold Premium Customers",
    4: "Silver Inactive Customers",
    5: "Bronze Discount Customers",
    6: "Silver Moderate Customers"
}

SEGMENT_COLORS = [
    "#86d028",   # brand green
    "#d9917a",   # peach
    "#c9b87c",   # wheat
    "#5a9216",   # dark green
    "#4aa3a2",   # teal
    "#8b6fae",   # violet
    "#3a8fb5",   # blue
]

STRATEGIES = {
    "Bronze Regular Customers":
        "Encourage repeat purchases using loyalty rewards and personalized recommendations.",
    "Gold High Value Customers":
        "Provide exclusive benefits, premium support, and early access to new products.",
    "Silver Regular Customers":
        "Use targeted offers and bundles to increase purchase frequency.",
    "Gold Premium Customers":
        "Strengthen loyalty through VIP rewards and premium experiences.",
    "Silver Inactive Customers":
        "Re-engage customers with comeback offers and relevant recommendations.",
    "Bronze Discount Customers":
        "Use targeted discounts and limited-time offers to encourage purchases.",
    "Silver Moderate Customers":
        "Encourage repeat purchases using bundles, personalized recommendations, and loyalty incentives."
}

# =========================================================
# CSS  —  EarthGuard premium design system
# =========================================================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Grotesk:wght@400;500;600;700&family=Instrument+Serif:ital@1&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── App shell ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
    background: #0b191e !important;
    color: #e8f1ec !important;
}

/* Atmospheric multi-layer gradient — mirrors the EarthGuard video overlay */
.stApp {
    background-image:
        radial-gradient(ellipse at 18% 12%, rgba(134,208,40,0.08) 0, transparent 40%),
        radial-gradient(ellipse at 85% 8%,  rgba(74,163,162,0.06) 0, transparent 30%),
        radial-gradient(ellipse at 60% 90%, rgba(134,208,40,0.10) 0, transparent 38%),
        linear-gradient(160deg, #0b191e 0%, #0f2027 50%, #0b191e 100%) !important;
    background-attachment: fixed !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0b191e; }
::-webkit-scrollbar-thumb { background: rgba(134,208,40,0.35); border-radius: 3px; }

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: #f0f4f8 !important;
}
p, span, div, label {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Block container ── */
.block-container {
    max-width: 1480px !important;
    padding-top: 3.8rem !important;
    padding-bottom: 3.5rem !important;
}

/* ── HERO HEADER ── */
.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    color: #86d028 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    text-transform: uppercase;
    margin-bottom: 10px !important;
}
.eyebrow::before {
    content: '';
    display: inline-block;
    width: 28px;
    height: 2px;
    background: #86d028;
    border-radius: 2px;
    flex-shrink: 0;
}

.hero-title {
    font-family: 'Syne', sans-serif !important;
    color: #f0f4f8 !important;
    font-size: 2.9rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    letter-spacing: -0.5px !important;
    text-shadow: 0 4px 24px rgba(0,0,0,0.55) !important;
    margin: 0 0 8px 0 !important;
}

.hero-italic {
    font-family: 'Instrument Serif', serif !important;
    font-style: italic !important;
    color: #86d028 !important;
    font-size: 3.1rem !important;
    text-shadow: 0 0 28px rgba(134,208,40,0.4) !important;
    display: block;
    margin-top: 2px;
}

.hero-subtitle {
    color: rgba(255,255,255,0.62) !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    max-width: 540px;
    margin-bottom: 28px !important;
}

/* ── Section headings ── */
.section-title {
    font-family: 'Syne', sans-serif !important;
    color: #f0f4f8 !important;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    margin-top: 24px;
    margin-bottom: 3px;
    letter-spacing: -0.2px;
}

.section-subtitle {
    color: rgba(255,255,255,0.50) !important;
    font-size: 0.875rem !important;
    margin-bottom: 16px;
}

/* ── NAVIGATION — glass pill ── */
div[role="radiogroup"] {
    background: rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 50px !important;
    padding: 6px 8px !important;
    gap: 4px !important;
    width: fit-content !important;
}

div[role="radiogroup"] label {
    background: transparent !important;
    border-radius: 50px !important;
    padding: 8px 20px !important;
    transition: all 0.25s ease !important;
}

div[role="radiogroup"] label p,
div[role="radiogroup"] label span,
div[role="radiogroup"] label div {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.65) !important;
    opacity: 1 !important;
}

div[role="radiogroup"] label:has(input:checked) {
    background: #86d028 !important;
    box-shadow: 0 0 18px rgba(134,208,40,0.38) !important;
}
div[role="radiogroup"] label:has(input:checked) p,
div[role="radiogroup"] label:has(input:checked) span,
div[role="radiogroup"] label:has(input:checked) div {
    color: #0b191e !important;
}

div[role="radiogroup"] label:hover:not(:has(input:checked)) {
    background: rgba(134,208,40,0.12) !important;
}
div[role="radiogroup"] label:hover:not(:has(input:checked)) p,
div[role="radiogroup"] label:hover:not(:has(input:checked)) span,
div[role="radiogroup"] label:hover:not(:has(input:checked)) div {
    color: #86d028 !important;
}

/* ── METRIC CARDS — glass morphism ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 18px !important;
    padding: 24px 22px !important;
    min-height: 120px !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px) !important;
    border-color: rgba(134,208,40,0.40) !important;
    box-shadow:
        0 12px 32px rgba(0,0,0,0.35),
        0 0 0 1px rgba(134,208,40,0.25),
        0 0 24px rgba(134,208,40,0.15) !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #86d028 !important;
    opacity: 1 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #f0f4f8 !important;
    opacity: 1 !important;
}

[data-testid="stMetric"] * { opacity: 1 !important; }

/* ── GLASS CARDS ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 14px;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease, background 0.25s ease;
}
.glass-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(134,208,40,0.35);
    transform: translateY(-4px);
    box-shadow: 0 14px 36px rgba(0,0,0,0.30), 0 0 24px rgba(134,208,40,0.10);
}
.glass-card h3 {
    font-family: 'Syne', sans-serif !important;
    color: #f0f4f8 !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
}
.glass-card p {
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    margin: 0 !important;
}
.glass-card .card-count {
    font-family: 'Syne', sans-serif !important;
    color: #86d028 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
    display: block;
}
.glass-card .card-tag {
    display: inline-block;
    background: rgba(134,208,40,0.12);
    border: 1px solid rgba(134,208,40,0.25);
    color: #86d028 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 50px;
    padding: 3px 12px;
    margin-top: 12px;
}

/* Prediction result card */
.prediction-card {
    background: linear-gradient(135deg, rgba(15,48,62,0.92), rgba(10,30,40,0.92)) !important;
    border: 1px solid rgba(134,208,40,0.30) !important;
    border-radius: 18px !important;
    padding: 28px !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 0 40px rgba(134,208,40,0.08) !important;
}
.prediction-card .pred-label {
    color: #86d028 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
    display: block;
}
.prediction-card h2 {
    font-family: 'Syne', sans-serif !important;
    color: #f0f4f8 !important;
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    margin: 0 0 8px 0 !important;
    text-shadow: 0 0 20px rgba(134,208,40,0.20) !important;
}
.prediction-card .pred-cluster {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.88rem !important;
}
.prediction-card .pred-strategy {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
    margin-top: 12px !important;
    padding-top: 12px !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── INPUTS ── */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.65) !important;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #f0f4f8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: rgba(134,208,40,0.55) !important;
    box-shadow: 0 0 0 3px rgba(134,208,40,0.12) !important;
}

[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: rgba(134,208,40,0.15) !important;
    border: 1px solid rgba(134,208,40,0.30) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
    color: #86d028 !important;
}

/* ── BUTTONS ── */
.stButton button,
.stDownloadButton button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    background: rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 50px !important;
    color: #f0f4f8 !important;
    padding: 10px 24px !important;
    transition: all 0.25s ease !important;
}
.stButton button:hover,
.stDownloadButton button:hover {
    background: rgba(134,208,40,0.14) !important;
    border-color: rgba(134,208,40,0.55) !important;
    color: #86d028 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 0 20px rgba(134,208,40,0.18) !important;
}
/* Primary form submit */
.stButton button[kind="primaryFormSubmit"],
div[data-testid="stForm"] .stButton button {
    background: #86d028 !important;
    color: #0b191e !important;
    border-color: #86d028 !important;
    box-shadow: 0 4px 18px rgba(134,208,40,0.30) !important;
}
div[data-testid="stForm"] .stButton button:hover {
    background: #76b821 !important;
    border-color: #76b821 !important;
    color: #0b191e !important;
    box-shadow: 0 6px 26px rgba(134,208,40,0.45) !important;
}

/* ── DATAFRAME / TABLE ── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease !important;
}
[data-testid="stDataFrame"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 26px rgba(0,0,0,0.22);
}

/* ── CHARTS ── */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
}
[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-3px);
    border-color: rgba(134,208,40,0.22) !important;
    box-shadow: 0 10px 28px rgba(0,0,0,0.22), 0 0 20px rgba(134,208,40,0.06);
}

/* ── CAPTION ── */
[data-testid="stCaptionContainer"] {
    color: rgba(255,255,255,0.40) !important;
    font-size: 0.78rem !important;
}

/* ── ALERTS / INFO ── */
.stAlert {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
    color: #e8f1ec !important;
}

/* ── DIVIDER ── */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 28px 0 !important;
}

/* ── CHECKBOX ── */
.stCheckbox label {
    color: rgba(255,255,255,0.70) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── OVERVIEW stat chip ── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(134,208,40,0.10);
    border: 1px solid rgba(134,208,40,0.22);
    border-radius: 50px;
    padding: 5px 14px 5px 10px;
    color: #86d028 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    width: fit-content;
    margin-bottom: 18px;
}
.stat-chip::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #86d028;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(134,208,40,0.6);
    animation: pulse-dot 2.5s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.6; transform: scale(0.75); }
}

/* Hide Streamlit branding */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()


# =========================================================
# LOAD SAVED MODELS
# =========================================================
@st.cache_resource
def load_models():
    required_files = [SCALER_FILE, ENCODER_FILE, MODEL_FILE]
    missing_files = [str(p) for p in required_files if not p.exists()]
    if missing_files:
        st.error("Missing model files: " + ", ".join(missing_files))
        return None, None, None
    try:
        scaler  = joblib.load(SCALER_FILE)
        encoder = joblib.load(ENCODER_FILE)
        model   = joblib.load(MODEL_FILE)
        return scaler, encoder, model
    except Exception as error:
        st.error(f"Could not load saved models: {error}")
        return None, None, None


df = load_data()
scaler, encoder, kmeans_model = load_models()

if df.empty:
    st.error(
        "Dataset not found. Ensure data/cleaned_customer_behavior.csv exists."
    )
    st.stop()


# =========================================================
# FEATURE COLUMNS
# =========================================================
NUMERIC_FEATURES     = ["Age", "Total Spend"]
CATEGORICAL_FEATURES = ["Membership Type", "Discount Applied"]


def prepare_customer_features(customer_df):
    numeric_scaled      = scaler.transform(customer_df[NUMERIC_FEATURES].copy())
    categorical_encoded = encoder.transform(customer_df[CATEGORICAL_FEATURES].copy())
    return np.concatenate([numeric_scaled, categorical_encoded], axis=1)


def predict_customer(customer_df):
    features     = prepare_customer_features(customer_df)
    cluster_id   = int(kmeans_model.predict(features)[0])
    segment_name = SEGMENT_NAMES.get(cluster_id, f"Segment {cluster_id}")
    return cluster_id, segment_name


# =========================================================
# FIND COLUMNS
# =========================================================
def find_column(names):
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    for col in df.columns:
        for name in names:
            if name.lower() in str(col).lower():
                return col
    return None


segment_col    = find_column(["Segment Name", "Segment", "Cluster Name", "Cluster", "Customer Segment"])
spend_col      = find_column(["Total Spend", "Total_Spend", "Spend"])
age_col        = find_column(["Age", "Customer Age"])
membership_col = find_column(["Membership Type", "Membership_Type"])
discount_col   = find_column(["Discount Applied", "Discount_Applied"])

if segment_col:
    if pd.api.types.is_numeric_dtype(df[segment_col]):
        df["Dashboard Segment"] = df[segment_col].map(
            lambda x: SEGMENT_NAMES.get(int(x), f"Segment {x}") if pd.notna(x) else "Unknown"
        )
    else:
        df["Dashboard Segment"] = df[segment_col].astype(str)
else:
    df["Dashboard Segment"] = "All Customers"


# =========================================================
# HELPERS
# =========================================================
def section(title, subtitle=""):
    subtitle_html = (
        f'<div class="section-subtitle">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div class="section-title">{title}</div>{subtitle_html}',
        unsafe_allow_html=True
    )


def style_chart(fig, height=370):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.75)", family="Space Grotesk"),
        margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(
            font=dict(color="rgba(255,255,255,0.70)"),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(color="rgba(255,255,255,0.60)", gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        yaxis=dict(color="rgba(255,255,255,0.60)", gridcolor="rgba(255,255,255,0.07)", zeroline=False),
    )
    return fig


def show_chart(fig, height=370):
    if fig is None or not getattr(fig, "data", None):
        st.info("There is not enough valid data to display this chart.")
        return
    style_chart(fig, height)
    st.plotly_chart(fig, use_container_width=True)


def safe_numeric(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


# =========================================================
# HERO HEADER
# =========================================================
st.markdown(
    '<div class="eyebrow">CUSTOMER INSIGHTS PLATFORM</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-title">Customer Segmentation'
    '<span class="hero-italic">Intelligence.</span>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-subtitle">'
    'Understand customer behaviour and discover meaningful groups '
    'for personalized, high-impact marketing.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# NAVIGATION  —  glass pill bar
# =========================================================
pages = [
    "Overview",
    "Segment Explorer",
    "Segment Summary",
    "Customer Data",
    "Customer Prediction",
]

page = st.radio(
    "Navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed"
)

st.write("")


# =========================================================
# ██████  OVERVIEW
# =========================================================
if page == "Overview":

    section(
        "Project Overview",
        "A high-level look at customer groups and overall spending patterns."
    )

    average_spend = (
        pd.to_numeric(df[spend_col], errors="coerce").mean()
        if spend_col else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL CUSTOMERS",   f"{len(df):,}")
    c2.metric("CUSTOMER SEGMENTS", df["Dashboard Segment"].nunique())
    c3.metric(
        "AVERAGE SPEND",
        f"{average_spend:,.2f}" if pd.notna(average_spend) else "N/A"
    )
    c4.metric("CLUSTERING METHOD", "K-Means")

    counts = (
        df["Dashboard Segment"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )

    left, right = st.columns([1.25, 1])

    if len(counts) == 1:
        only_segment = str(counts.iloc[0]["Segment"])
        only_count   = int(counts.iloc[0]["Customers"])
        total_count  = max(1, int(counts["Customers"].sum()))

        with left:
            section("Customer distribution", "Current dataset coverage.")
            st.markdown(f"""
            <div class="glass-card" style="min-height:240px;display:flex;flex-direction:column;justify-content:center;gap:8px;">
                <div class="stat-chip">Segment found</div>
                <div style="font-family:'Syne',sans-serif;color:#f0f4f8;font-size:1.55rem;font-weight:800;">{only_segment}</div>
                <div style="font-family:'Syne',sans-serif;color:#86d028;font-size:2.8rem;font-weight:800;line-height:1;">{only_count:,}</div>
                <div style="color:rgba(255,255,255,0.50);font-size:0.88rem;">customers in the loaded dataset</div>
            </div>""", unsafe_allow_html=True)
            st.caption("A segment comparison will appear here when the dataset contains multiple segments.")

        with right:
            section("Segment share", "Current segment coverage.")
            st.markdown(f"""
            <div class="glass-card" style="min-height:240px;display:flex;flex-direction:column;justify-content:center;gap:8px;">
                <div class="stat-chip">Customer base share</div>
                <div style="font-family:'Syne',sans-serif;color:#86d028;font-size:3.2rem;font-weight:800;line-height:1;text-shadow:0 0 28px rgba(134,208,40,0.35);">100%</div>
                <div style="font-family:'Syne',sans-serif;color:#f0f4f8;font-size:1.1rem;font-weight:800;">{only_segment}</div>
                <div style="color:rgba(255,255,255,0.50);font-size:0.88rem;">{only_count:,} of {total_count:,} customers</div>
            </div>""", unsafe_allow_html=True)
            st.caption("The loaded data has one segment — share would only repeat 100%.")
    else:
        with left:
            section("Customer distribution", "Compare the number of customers across segments.")
            fig = px.bar(
                counts.sort_values("Customers"),
                x="Customers", y="Segment", orientation="h",
                color="Segment", text="Customers",
                color_discrete_sequence=SEGMENT_COLORS
            )
            fig.update_traces(
                textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.80)", size=11),
                marker_line_color="rgba(0,0,0,0)", marker_line_width=0,
                marker_opacity=0.88
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Number of customers",
                yaxis_title="",
                yaxis=dict(categoryorder="total ascending")
            )
            show_chart(fig, max(330, 70 * len(counts) + 100))

        with right:
            section("Segment share", "Each segment's percentage of the customer base.")
            share = counts.copy()
            share["Share"] = share["Customers"] / max(1, share["Customers"].sum()) * 100
            ordered = share.sort_values("Share")
            fig = px.bar(
                ordered, x="Share", y="Segment", orientation="h",
                color="Segment",
                text=ordered["Share"].map(lambda v: f"{v:.1f}%"),
                color_discrete_sequence=SEGMENT_COLORS
            )
            fig.update_traces(
                textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.80)", size=11),
                cliponaxis=False,
                marker_opacity=0.88
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Share of customers (%)",
                yaxis_title="",
                xaxis=dict(range=[0, max(105, float(share["Share"].max()) * 1.2)])
            )
            show_chart(fig, max(330, 70 * len(counts) + 100))

    if spend_col:
        section("Customer spending", "The spread and typical spending level for each segment.")
        temp = df[["Dashboard Segment", spend_col]].copy()
        temp[spend_col] = pd.to_numeric(temp[spend_col], errors="coerce")
        temp = temp.dropna(subset=[spend_col])
        if not temp.empty:
            if temp["Dashboard Segment"].nunique() > 1:
                fig = px.violin(
                    temp, x="Dashboard Segment", y=spend_col,
                    color="Dashboard Segment", box=True, points="all",
                    color_discrete_sequence=SEGMENT_COLORS
                )
                fig.update_traces(
                    meanline_visible=True, jitter=0.25, pointpos=0,
                    marker=dict(size=3, opacity=0.30)
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_title="Customer segment",
                    yaxis_title="Total spend"
                )
                show_chart(fig, 460)
            else:
                fig = px.histogram(
                    temp, x=spend_col, nbins=18, marginal="box",
                    color_discrete_sequence=["#86d028"]
                )
                fig.update_traces(marker_line_color="rgba(0,0,0,0)", marker_opacity=0.80)
                fig.update_layout(
                    xaxis_title="Total spend",
                    yaxis_title="Number of customers",
                    showlegend=False
                )
                show_chart(fig, 440)
                st.caption(
                    "Only one segment is present — showing the spending distribution within that segment."
                )
        else:
            st.info("No valid spending values are available for this chart.")


# =========================================================
# ██████  SEGMENT EXPLORER
# =========================================================
elif page == "Segment Explorer":

    section(
        "Segment Explorer",
        "Select a segment to view its customer profile and behaviour patterns."
    )

    segments = sorted(df["Dashboard Segment"].dropna().unique())
    selected = st.selectbox("Choose a customer segment", segments)
    segment_df = df[df["Dashboard Segment"] == selected].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("CUSTOMERS", len(segment_df))

    if spend_col:
        avg_spend = pd.to_numeric(segment_df[spend_col], errors="coerce").mean()
        c2.metric("AVERAGE SPEND", f"{avg_spend:,.2f}" if pd.notna(avg_spend) else "N/A")
    else:
        c2.metric("AVERAGE SPEND", "N/A")

    if age_col:
        avg_age = pd.to_numeric(segment_df[age_col], errors="coerce").mean()
        c3.metric("AVERAGE AGE", f"{avg_age:.1f}" if pd.notna(avg_age) else "N/A")
    else:
        c3.metric("AVERAGE AGE", "N/A")

    left, right = st.columns(2)

    with left:
        section("Segment profile", "Summary statistics for this customer group.")
        profile_cols = [c for c in [age_col, spend_col, membership_col, discount_col] if c]
        if profile_cols:
            st.dataframe(
                segment_df[profile_cols].describe(include="all").T,
                use_container_width=True
            )

    with right:
        section("Membership distribution", "Membership types within this segment.")
        if membership_col:
            membership_counts = (
                segment_df[membership_col]
                .value_counts()
                .reset_index()
            )
            membership_counts.columns = ["Membership", "Customers"]
            if not membership_counts.empty:
                fig = px.pie(
                    membership_counts,
                    names="Membership", values="Customers",
                    hole=0.55,
                    color_discrete_sequence=SEGMENT_COLORS
                )
                fig.update_traces(
                    textinfo="percent+label",
                    textfont=dict(color="#0b191e", size=12),
                    marker_line_color="rgba(11,25,30,0.8)",
                    marker_line_width=2,
                    pull=[0.04] * len(membership_counts)
                )
                show_chart(fig, 340)
            else:
                st.info("No membership values available for this segment.")
        else:
            st.info("Membership column not found.")

    section("Customers in this segment", "Individual records belonging to the selected group.")
    st.dataframe(
        segment_df.drop(columns=["Dashboard Segment"], errors="ignore"),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ██████  SEGMENT SUMMARY
# =========================================================
elif page == "Segment Summary":

    section(
        "Segment Summary",
        "Compare customer groups and their key characteristics at a glance."
    )

    summary = (
        df.groupby("Dashboard Segment")
        .size()
        .reset_index(name="Customers")
        .rename(columns={"Dashboard Segment": "Customer Segment"})
    )

    if spend_col:
        temp = df.copy()
        temp["_spend"] = pd.to_numeric(temp[spend_col], errors="coerce")
        spend_summary = (
            temp.groupby("Dashboard Segment")["_spend"]
            .mean()
            .reset_index(name="Average Spend")
            .rename(columns={"Dashboard Segment": "Customer Segment"})
        )
        summary = summary.merge(spend_summary, on="Customer Segment")

    if age_col:
        temp = df.copy()
        temp["_age"] = pd.to_numeric(temp[age_col], errors="coerce")
        age_summary = (
            temp.groupby("Dashboard Segment")["_age"]
            .mean()
            .reset_index(name="Average Age")
            .rename(columns={"Dashboard Segment": "Customer Segment"})
        )
        summary = summary.merge(age_summary, on="Customer Segment")

    st.dataframe(summary, use_container_width=True, hide_index=True)

    section(
        "Personalized Marketing Strategies",
        "Suggested approaches tailored to each customer segment."
    )

    for _, row in summary.iterrows():
        name     = row["Customer Segment"]
        strategy = STRATEGIES.get(
            name,
            "Use personalized offers and relevant recommendations to improve engagement."
        )
        st.markdown(f"""
        <div class="glass-card">
            <span class="card-count">{int(row['Customers']):,} customers</span>
            <h3>{name}</h3>
            <p>{strategy}</p>
            <span class="card-tag">View strategy →</span>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# ██████  CUSTOMER DATA
# =========================================================
elif page == "Customer Data":

    section(
        "Customer Database",
        "Search, filter, and export customer records."
    )

    search = st.text_input(
        "Search customer records",
        placeholder="Search by customer ID, name, or any value…"
    )

    filtered_df = df.copy()

    if search:
        mask = filtered_df.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        filtered_df = filtered_df[mask]

    segments          = sorted(df["Dashboard Segment"].dropna().unique())
    selected_segments = st.multiselect(
        "Filter by customer segment",
        segments,
        default=segments
    )
    filtered_df = filtered_df[filtered_df["Dashboard Segment"].isin(selected_segments)]

    c1, c2 = st.columns(2)
    c1.metric("MATCHING CUSTOMERS", len(filtered_df))
    c2.metric("TOTAL RECORDS",      len(df))

    st.dataframe(
        filtered_df.drop(columns=["Dashboard Segment"], errors="ignore"),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇  Download filtered data",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_data.csv",
        mime="text/csv"
    )


# =========================================================
# ██████  CUSTOMER PREDICTION
# =========================================================
elif page == "Customer Prediction":

    section(
        "Add & Predict a Customer",
        "Enter customer details to predict their segment and save the record."
    )

    if scaler is None or encoder is None or kmeans_model is None:
        st.error(
            "Saved model files could not be loaded. "
            "Check the error above and verify your files in the models/ folder."
        )
        st.stop()

    with st.form("customer_prediction_form"):
        st.markdown(
            '<div class="section-subtitle">Fill in the customer details below.</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            customer_name = st.text_input("Customer name")
            age           = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
            total_spend   = st.number_input("Total Spend", min_value=0.0, value=500.0, step=50.0)

        with col2:
            membership = st.selectbox("Membership Type", list(encoder.categories_[0]))
            discount   = st.selectbox("Discount Applied", list(encoder.categories_[1]))

        submitted = st.form_submit_button(
            "Predict Customer Segment",
            use_container_width=True
        )

    if submitted:
        customer_input = pd.DataFrame([{
            "Age":             age,
            "Total Spend":     total_spend,
            "Membership Type": membership,
            "Discount Applied": discount,
        }])
        try:
            cluster_id, segment_name = predict_customer(customer_input)
            st.session_state["latest_prediction"] = {
                "Customer Name":    customer_name.strip() or "New Customer",
                "Age":              age,
                "Total Spend":      total_spend,
                "Membership Type":  membership,
                "Discount Applied": discount,
                "Predicted Cluster": cluster_id,
                "Predicted Segment": segment_name,
            }
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if "latest_prediction" in st.session_state:
        result   = st.session_state["latest_prediction"]
        strategy = STRATEGIES.get(result["Predicted Segment"], "")

        st.markdown(f"""
        <div class="prediction-card">
            <span class="pred-label">Prediction result</span>
            <h2>{result['Predicted Segment']}</h2>
            <div class="pred-cluster">Predicted cluster ID: {result['Predicted Cluster']}</div>
            <div class="pred-strategy">{strategy}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Save this customer", use_container_width=True):
            saved_record = result.copy()
            saved_record["Saved At"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            saved_record["Record ID"] = str(uuid4())
            new_row = pd.DataFrame([saved_record])

            if NEW_CUSTOMERS_FILE.exists():
                existing = pd.read_csv(NEW_CUSTOMERS_FILE)
                if "Record ID" not in existing.columns:
                    existing["Record ID"] = [str(uuid4()) for _ in range(len(existing))]
                updated = pd.concat([existing, new_row], ignore_index=True, sort=False)
            else:
                updated = new_row

            updated.to_csv(NEW_CUSTOMERS_FILE, index=False)
            st.success(f"Customer saved to {NEW_CUSTOMERS_FILE.name}.")

    st.write("")

    section(
        "Previously Added Customers",
        "New customer records saved through this page."
    )

    if NEW_CUSTOMERS_FILE.exists():
        saved_customers = pd.read_csv(NEW_CUSTOMERS_FILE)
        if not saved_customers.empty:
            if "Record ID" not in saved_customers.columns:
                saved_customers["Record ID"] = [str(uuid4()) for _ in range(len(saved_customers))]
                saved_customers.to_csv(NEW_CUSTOMERS_FILE, index=False)

            st.dataframe(
                saved_customers.drop(columns=["Record ID"], errors="ignore"),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("**Select customers to delete**")
            selected_ids = []
            for i, row in saved_customers.iterrows():
                label = (
                    f"{row.get('Customer Name', 'Customer')}  —  "
                    f"{row.get('Predicted Segment', 'Unknown')}  —  "
                    f"{row.get('Saved At', 'No date')}  (record {i + 1})"
                )
                if st.checkbox(label, key=f"delete_customer_{row['Record ID']}"):
                    selected_ids.append(str(row["Record ID"]))

            confirm_delete = st.checkbox(
                "I confirm that I want to permanently delete the selected customer(s).",
                key="confirm_saved_customer_delete"
            )

            if st.button(
                "Delete selected customer(s)",
                type="secondary",
                disabled=(not selected_ids or not confirm_delete),
                use_container_width=True
            ):
                latest = pd.read_csv(NEW_CUSTOMERS_FILE)
                if "Record ID" not in latest.columns:
                    st.error("Record identifiers are missing. No data was deleted.")
                else:
                    remaining     = latest[~latest["Record ID"].astype(str).isin(selected_ids)]
                    deleted_count = len(latest) - len(remaining)
                    if deleted_count == 0:
                        st.warning("Selected record(s) were not found; no data was deleted.")
                    else:
                        remaining.to_csv(NEW_CUSTOMERS_FILE, index=False)
                        st.success(f"Deleted {deleted_count} customer record(s).")
                        st.rerun()

            st.download_button(
                "⬇  Download saved customers",
                data=saved_customers.drop(columns=["Record ID"], errors="ignore")
                                   .to_csv(index=False).encode("utf-8"),
                file_name="new_customers.csv",
                mime="text/csv"
            )
        else:
            st.info("No new customers have been saved yet.")
    else:
        st.info("No new customers have been saved yet.")


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<hr>
<p style="
    text-align:center;
    color:rgba(255,255,255,0.30);
    font-family:'Space Grotesk',sans-serif;
    font-size:0.72rem;
    font-weight:600;
    letter-spacing:3px;
    text-transform:uppercase;
    margin:0;
">
    Customer Segmentation Intelligence &nbsp;·&nbsp; K-Means Clustering
</p>
""", unsafe_allow_html=True)