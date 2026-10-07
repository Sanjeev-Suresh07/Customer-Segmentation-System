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
# COLORS
# =========================================================
BG = "#304840"
CREAM = "#F7F4EA"
WHITE = "#FFFFFF"
SAGE = "#DCE8D8"
GREEN = "#52796F"
PEACH = "#E8C9B8"
WHEAT = "#E8DFAF"
TEXT = "#263A34"

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
    "#52796F",
    "#E8C9B8",
    "#E8DFAF",
    "#8A9A9A",
    "#D18A76",
    "#B98261",
    "#6F9388"
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
# CSS
# =========================================================
st.markdown("""
<style>
:root {
    --forest: #304840;
    --cream: #F7F4EA;
    --sage: #DCE8D8;
    --green: #52796F;
    --peach: #E8C9B8;
    --wheat: #E8DFAF;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
    background: var(--forest) !important;
}

/* Soft botanical texture without affecting content contrast */
.stApp {
    background-image:
        radial-gradient(ellipse at 8% 8%, rgba(111,147,136,0.16) 0, rgba(111,147,136,0.05) 19%, transparent 38%),
        radial-gradient(ellipse at 92% 22%, rgba(220,232,216,0.10) 0, transparent 28%),
        radial-gradient(ellipse at 75% 92%, rgba(82,121,111,0.20) 0, transparent 34%),
        linear-gradient(135deg, #304840 0%, #2B4139 55%, #304840 100%) !important;
    background-attachment: fixed !important;
}

.block-container {
    max-width: 1500px;
    padding-top: 3.5rem !important;
    padding-bottom: 3rem;
}

/* Header */
.eyebrow {
    color: var(--sage) !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 3px !important;
    line-height: 1.8 !important;
    padding-top: 8px !important;
    margin-top: 8px !important;
    margin-bottom: 14px !important;
    overflow: visible !important;
}

.hero-title {
    color: var(--cream) !important;
    font-size: 2.65rem !important;
    font-weight: 800 !important;
    line-height: 1.25 !important;
    margin: 0 0 10px 0 !important;
    padding: 0 !important;
}

.hero-subtitle {
    color: var(--sage) !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    margin-bottom: 26px !important;
}

/* General text */
h1, h2, h3, h4, p {
    color: var(--cream);
}

.section-title {
    color: var(--cream) !important;
    font-size: 1.35rem;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 4px;
}

.section-subtitle {
    color: var(--sage) !important;
    font-size: 0.9rem;
    margin-bottom: 18px;
}

/* Navigation */
div[role="radiogroup"] {
    background: var(--cream) !important;
    padding: 8px !important;
    border-radius: 14px !important;
    gap: 8px !important;
    width: fit-content;
}

div[role="radiogroup"] label {
    color: var(--forest) !important;
    background: transparent !important;
    border-radius: 10px !important;
    padding: 9px 16px !important;
    opacity: 1 !important;
}

div[role="radiogroup"] label p,
div[role="radiogroup"] label span,
div[role="radiogroup"] label div {
    color: var(--forest) !important;
    opacity: 1 !important;
    font-weight: 650 !important;
}

div[role="radiogroup"] label:has(input:checked) {
    background: var(--sage) !important;
}

div[role="radiogroup"] label {
    transition: box-shadow 0.2s ease, transform 0.2s ease, background 0.2s ease !important;
}
div[role="radiogroup"] label:hover {
    background: #EAF1E7 !important;
    box-shadow: 0 0 0 1px rgba(232,201,184,0.18), 0 0 16px rgba(232,137,91,0.18) !important;
    transform: translateY(-1px);
}

/* Hover effects */
[data-testid="stMetric"],
.hover-card,
[data-testid="stDataFrame"],
div[data-testid="stPlotlyChart"] {
    transition:
        transform 0.22s ease,
        box-shadow 0.22s ease,
        border-color 0.22s ease !important;
}

[data-testid="stMetric"]:hover,
.hover-card:hover {
    transform: translateY(-6px);
    box-shadow:
        0 12px 28px rgba(0,0,0,0.24),
        0 0 0 1px rgba(232,137,91,0.32),
        0 0 24px rgba(232,137,91,0.28) !important;
    border-color: rgba(232,137,91,0.62) !important;
}

div[data-testid="stPlotlyChart"]:hover,
[data-testid="stDataFrame"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.18);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--cream) !important;
    border: 1px solid rgba(220,232,216,0.45) !important;
    border-radius: 17px !important;
    padding: 22px !important;
    min-height: 125px;
}

[data-testid="stMetric"] * {
    color: var(--forest) !important;
    opacity: 1 !important;
}

[data-testid="stMetricLabel"] {
    font-weight: 750 !important;
}

/* Custom cards */
.hover-card {
    background: var(--cream);
    border: 1px solid rgba(220,232,216,0.4);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 14px;
}

.hover-card h3,
.hover-card p {
    color: var(--forest) !important;
}

/* Translucent dark-blue cards for the one-segment overview state */
.overview-summary-card {
    background: linear-gradient(135deg, rgba(20, 48, 73, 0.88), rgba(31, 65, 91, 0.78)) !important;
    border: 1px solid rgba(164, 198, 219, 0.38) !important;
    backdrop-filter: blur(9px);
    -webkit-backdrop-filter: blur(9px);
}
.overview-summary-card,
.overview-summary-card div {
    color: #F7F4EA !important;
}
.prediction-card {
    background: linear-gradient(135deg, rgba(24,54,78,0.86), rgba(20,43,65,0.86)) !important;
    border: 1px solid rgba(150,190,215,0.32) !important;
    backdrop-filter: blur(8px);
}
.prediction-card h2, .prediction-card p { color: #F4FAFF !important; }

/* Inputs */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label {
    color: var(--cream) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
    color: var(--forest) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] input {
    color: var(--forest) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] div {
    color: var(--forest);
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: var(--cream) !important;
    color: var(--forest) !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background: #1E302B !important;
    color: var(--cream) !important;
    border: 1px solid rgba(220,232,216,0.28) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #263D35 !important;
    color: var(--cream) !important;
    transform: translateY(-1px);
    box-shadow: 0 0 0 1px rgba(232,201,184,0.14), 0 0 11px rgba(232,137,91,0.16);
}

/* Tables */
[data-testid="stDataFrame"] {
    background: var(--cream) !important;
    border-radius: 12px;
    padding: 8px;
}

[data-testid="stCaptionContainer"] {
    color: var(--sage) !important;
}

/* Selectable tags / chips */
[data-baseweb="tag"] {
    background: #DCE8D8 !important;
    border: 1px solid #8EAD9B !important;
    transition: box-shadow 0.2s ease, filter 0.2s ease !important;
}
[data-baseweb="tag"]:hover {
    box-shadow: 0 0 14px rgba(232,137,91,0.18) !important;
    filter: brightness(1.04);
}

[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(220,232,216,0.18);
    border-radius: 14px;
    padding: 5px;
}


hr {
    border-color: rgba(220,232,216,0.3) !important;
}

footer {
    visibility: hidden;
}
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
# LOAD SAVED MODELS - JOBLIB FIX
# =========================================================
@st.cache_resource
def load_models():
    required_files = [SCALER_FILE, ENCODER_FILE, MODEL_FILE]

    missing_files = [
        str(path) for path in required_files if not path.exists()
    ]

    if missing_files:
        st.error("Missing model files: " + ", ".join(missing_files))
        return None, None, None

    try:
        scaler = joblib.load(SCALER_FILE)
        encoder = joblib.load(ENCODER_FILE)
        model = joblib.load(MODEL_FILE)

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
NUMERIC_FEATURES = ["Age", "Total Spend"]
CATEGORICAL_FEATURES = ["Membership Type", "Discount Applied"]


def prepare_customer_features(customer_df):
    numeric_data = customer_df[NUMERIC_FEATURES].copy()
    categorical_data = customer_df[CATEGORICAL_FEATURES].copy()

    numeric_scaled = scaler.transform(numeric_data)
    categorical_encoded = encoder.transform(categorical_data)

    return np.concatenate(
        [numeric_scaled, categorical_encoded],
        axis=1
    )


def predict_customer(customer_df):
    features = prepare_customer_features(customer_df)
    cluster_id = int(kmeans_model.predict(features)[0])
    segment_name = SEGMENT_NAMES.get(cluster_id, f"Segment {cluster_id}")

    return cluster_id, segment_name


# =========================================================
# FIND COLUMNS
# =========================================================
def find_column(names):
    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    for col in df.columns:
        for name in names:
            if name.lower() in str(col).lower():
                return col

    return None


segment_col = find_column([
    "Segment Name", "Segment", "Cluster Name",
    "Cluster", "Customer Segment"
])

spend_col = find_column(["Total Spend", "Total_Spend", "Spend"])
age_col = find_column(["Age", "Customer Age"])
membership_col = find_column(["Membership Type", "Membership_Type"])
discount_col = find_column(["Discount Applied", "Discount_Applied"])


if segment_col:
    if pd.api.types.is_numeric_dtype(df[segment_col]):
        df["Dashboard Segment"] = df[segment_col].map(
            lambda x: SEGMENT_NAMES.get(int(x), f"Segment {x}")
            if pd.notna(x) else "Unknown"
        )
    else:
        df["Dashboard Segment"] = df[segment_col].astype(str)
else:
    df["Dashboard Segment"] = "All Customers"


# =========================================================
# HELPERS
# =========================================================
def section(title, subtitle):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def style_chart(fig, height=370):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CREAM, family="Arial"),
        margin=dict(l=20, r=20, t=25, b=25),
        legend=dict(font=dict(color=CREAM)),
        xaxis=dict(
            color=CREAM,
            gridcolor="rgba(220,232,216,0.15)"
        ),
        yaxis=dict(
            color=CREAM,
            gridcolor="rgba(220,232,216,0.15)"
        )
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
# HEADER
# =========================================================
st.markdown(
    '<div class="eyebrow">CUSTOMER INSIGHTS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Customer Segmentation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Understand customer behaviour and discover meaningful groups '
    'for personalized marketing.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# NAVIGATION
# =========================================================
pages = [
    "Overview",
    "Segment Explorer",
    "Segment Summary",
    "Customer Data",
    "Customer Prediction"
]

page = st.radio(
    "Navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed"
)

st.write("")


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":

    section(
        "Project overview",
        "A quick view of customer groups and overall spending."
    )

    average_spend = (
        pd.to_numeric(df[spend_col], errors="coerce").mean()
        if spend_col else None
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("TOTAL CUSTOMERS", f"{len(df):,}")
    c2.metric("CUSTOMER SEGMENTS", df["Dashboard Segment"].nunique())

    c3.metric(
        "AVERAGE SPEND",
        f"{average_spend:,.2f}"
        if pd.notna(average_spend) else "N/A"
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
        only_count = int(counts.iloc[0]["Customers"])
        total_count = max(1, int(counts["Customers"].sum()))
        with left:
            section("Customer distribution", "Current dataset coverage.")
            st.markdown(f"""
            <div class="hover-card overview-summary-card" style="min-height:250px;display:flex;flex-direction:column;justify-content:center;">
                <div style="color:#52796F;font-size:0.82rem;font-weight:800;letter-spacing:1.5px;">SEGMENT FOUND</div>
                <div style="color:#263A34;font-size:1.65rem;font-weight:800;margin:10px 0;">{only_segment}</div>
                <div style="color:#263A34;font-size:2.6rem;font-weight:800;">{only_count:,}</div>
                <div style="color:#52665E;">customers in the loaded dataset</div>
            </div>""", unsafe_allow_html=True)
            st.caption("A segment comparison will appear here when the dataset contains multiple segments.")
        with right:
            section("Segment share", "Current segment coverage.")
            st.markdown(f"""
            <div class="hover-card overview-summary-card" style="min-height:250px;display:flex;flex-direction:column;justify-content:center;">
                <div style="color:#52796F;font-size:0.82rem;font-weight:800;letter-spacing:1.5px;">CUSTOMER BASE SHARE</div>
                <div style="color:#263A34;font-size:3rem;font-weight:800;margin:12px 0;">100%</div>
                <div style="color:#263A34;font-size:1.1rem;font-weight:700;">{only_segment}</div>
                <div style="color:#52665E;">{only_count:,} of {total_count:,} customers</div>
            </div>""", unsafe_allow_html=True)
            st.caption("The loaded data currently has one segment, so a share chart would only repeat 100%.")
    else:
        with left:
            section("Customer distribution", "Compare the number of customers across segments.")
            fig = px.bar(counts.sort_values("Customers"), x="Customers", y="Segment", orientation="h",
                         color="Segment", text="Customers", color_discrete_sequence=SEGMENT_COLORS)
            fig.update_traces(textposition="outside", textfont=dict(color=CREAM, size=12),
                              marker_line_color=BG, marker_line_width=1)
            fig.update_layout(showlegend=False, xaxis_title="Number of customers", yaxis_title="",
                              yaxis=dict(categoryorder="total ascending"))
            show_chart(fig, max(330, 70 * len(counts) + 100))
        with right:
            section("Segment share", "Each segment's percentage of the customer base.")
            share = counts.copy()
            share["Share"] = share["Customers"] / max(1, share["Customers"].sum()) * 100
            ordered = share.sort_values("Share")
            fig = px.bar(ordered, x="Share", y="Segment", orientation="h", color="Segment",
                         text=ordered["Share"].map(lambda v: f"{v:.1f}%"), color_discrete_sequence=SEGMENT_COLORS)
            fig.update_traces(textposition="outside", textfont=dict(color=CREAM, size=11), cliponaxis=False)
            fig.update_layout(showlegend=False, xaxis_title="Share of customers (%)", yaxis_title="",
                              xaxis=dict(range=[0, max(105, float(share["Share"].max()) * 1.2)]))
            show_chart(fig, max(330, 70 * len(counts) + 100))

    if spend_col:
        section("Customer spending", "See the spread and typical spending level for each segment.")
        temp = df[["Dashboard Segment", spend_col]].copy()
        temp[spend_col] = pd.to_numeric(temp[spend_col], errors="coerce")
        temp = temp.dropna(subset=[spend_col])
        if not temp.empty:
            if temp["Dashboard Segment"].nunique() > 1:
                fig = px.violin(temp, x="Dashboard Segment", y=spend_col, color="Dashboard Segment",
                                box=True, points="all", color_discrete_sequence=SEGMENT_COLORS)
                fig.update_traces(meanline_visible=True, jitter=0.25, pointpos=0, marker=dict(size=3, opacity=0.35))
                fig.update_layout(showlegend=False, xaxis_title="Customer segment", yaxis_title="Total spend")
                show_chart(fig, 440)
            else:
                fig = px.histogram(temp, x=spend_col, nbins=18, marginal="box",
                                   color_discrete_sequence=["#E8C9B8"])
                fig.update_traces(marker_line_color=BG, marker_line_width=1)
                fig.update_layout(xaxis_title="Total spend", yaxis_title="Number of customers", showlegend=False)
                show_chart(fig, 440)
                st.caption("Only one segment is present, so this chart shows the spending distribution within that segment instead.")
        else:
            st.info("No valid spending values are available for this chart.")


# =========================================================
# SEGMENT EXPLORER
# =========================================================
elif page == "Segment Explorer":

    section(
        "Explore customer segments",
        "Select a segment to view its customer profile and behaviour."
    )

    segments = sorted(
        df["Dashboard Segment"].dropna().unique()
    )

    selected = st.selectbox(
        "Choose a customer segment",
        segments
    )

    segment_df = df[
        df["Dashboard Segment"] == selected
    ].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric("CUSTOMERS", len(segment_df))

    if spend_col:
        avg_spend = pd.to_numeric(
            segment_df[spend_col],
            errors="coerce"
        ).mean()

        c2.metric(
            "AVERAGE SPEND",
            f"{avg_spend:,.2f}"
            if pd.notna(avg_spend) else "N/A"
        )
    else:
        c2.metric("AVERAGE SPEND", "N/A")

    if age_col:
        avg_age = pd.to_numeric(
            segment_df[age_col],
            errors="coerce"
        ).mean()

        c3.metric(
            "AVERAGE AGE",
            f"{avg_age:.1f}"
            if pd.notna(avg_age) else "N/A"
        )
    else:
        c3.metric("AVERAGE AGE", "N/A")

    left, right = st.columns(2)

    with left:
        section(
            "Segment profile",
            "Summary of this customer group."
        )

        profile_cols = [
            col for col in [
                age_col,
                spend_col,
                membership_col,
                discount_col
            ]
            if col is not None
        ]

        if profile_cols:
            st.dataframe(
                segment_df[profile_cols].describe(include="all").T,
                use_container_width=True
            )

    with right:
        section(
            "Membership distribution",
            "Membership types in this segment."
        )

        if membership_col:
            membership_counts = (
                segment_df[membership_col]
                .value_counts()
                .reset_index()
            )

            membership_counts.columns = [
                "Membership",
                "Customers"
            ]

            if not membership_counts.empty:
                fig = px.pie(
                    membership_counts,
                names="Membership",
                values="Customers",
                hole=0.5,
                    color_discrete_sequence=[GREEN, PEACH, WHEAT, "#8A9A9A"]
                )
                fig.update_traces(textinfo="percent+label", textfont=dict(color=BG), marker_line_color=BG, marker_line_width=2)
                show_chart(fig, 330)
            else:
                st.info("No membership values are available for this segment.")
        else:
            st.info("Membership column not found.")

    section(
        "Customers in this segment",
        "Records belonging to the selected group."
    )

    st.dataframe(
        segment_df.drop(
            columns=["Dashboard Segment"],
            errors="ignore"
        ),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SEGMENT SUMMARY
# =========================================================
elif page == "Segment Summary":

    section(
        "Segment summary",
        "Compare customer groups and their key characteristics."
    )

    summary = (
        df.groupby("Dashboard Segment")
        .size()
        .reset_index(name="Customers")
        .rename(columns={
            "Dashboard Segment": "Customer Segment"
        })
    )

    if spend_col:
        temp = df.copy()
        temp["_spend"] = pd.to_numeric(
            temp[spend_col],
            errors="coerce"
        )

        spend_summary = (
            temp.groupby("Dashboard Segment")["_spend"]
            .mean()
            .reset_index(name="Average Spend")
            .rename(columns={
                "Dashboard Segment": "Customer Segment"
            })
        )

        summary = summary.merge(
            spend_summary,
            on="Customer Segment"
        )

    if age_col:
        temp = df.copy()
        temp["_age"] = pd.to_numeric(
            temp[age_col],
            errors="coerce"
        )

        age_summary = (
            temp.groupby("Dashboard Segment")["_age"]
            .mean()
            .reset_index(name="Average Age")
            .rename(columns={
                "Dashboard Segment": "Customer Segment"
            })
        )

        summary = summary.merge(
            age_summary,
            on="Customer Segment"
        )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    section(
        "Personalized marketing strategies",
        "Suggested approaches based on each customer segment."
    )

    for _, row in summary.iterrows():
        name = row["Customer Segment"]

        st.markdown(
            f"""
            <div class="hover-card">
                <h3>{name}</h3>
                <p><b>{row['Customers']} customers</b></p>
                <p>{STRATEGIES.get(name, 'Use personalized offers and relevant recommendations to improve engagement.')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# CUSTOMER DATA
# =========================================================
elif page == "Customer Data":

    section(
        "Customer database",
        "Search, filter, and view customer records."
    )

    search = st.text_input(
        "Search customer records",
        placeholder="Search by customer ID or any value..."
    )

    filtered_df = df.copy()

    if search:
        mask = filtered_df.astype(str).apply(
            lambda col: col.str.contains(
                search,
                case=False,
                na=False
            )
        ).any(axis=1)

        filtered_df = filtered_df[mask]

    segments = sorted(
        df["Dashboard Segment"].dropna().unique()
    )

    selected_segments = st.multiselect(
        "Filter by customer segment",
        segments,
        default=segments
    )

    filtered_df = filtered_df[
        filtered_df["Dashboard Segment"].isin(selected_segments)
    ]

    c1, c2 = st.columns(2)

    c1.metric("MATCHING CUSTOMERS", len(filtered_df))
    c2.metric("TOTAL RECORDS", len(df))

    st.dataframe(
        filtered_df.drop(
            columns=["Dashboard Segment"],
            errors="ignore"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered customer data",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_data.csv",
        mime="text/csv"
    )


# =========================================================
# CUSTOMER PREDICTION
# =========================================================
elif page == "Customer Prediction":

    section(
        "Add and predict a customer",
        "Enter customer details to predict their segment and save the record."
    )

    if scaler is None or encoder is None or kmeans_model is None:
        st.error(
            "Saved model files could not be loaded. "
            "Check the error above and your files in the models folder."
        )
        st.stop()

    with st.form("customer_prediction_form"):

        st.markdown(
            '<div class="section-subtitle">'
            'Enter the customer information below.'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            customer_name = st.text_input("Customer name")

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=100,
                value=25,
                step=1
            )

            total_spend = st.number_input(
                "Total Spend",
                min_value=0.0,
                value=500.0,
                step=50.0
            )

        with col2:
            membership = st.selectbox(
                "Membership Type",
                list(encoder.categories_[0])
            )

            discount = st.selectbox(
                "Discount Applied",
                list(encoder.categories_[1])
            )

        submitted = st.form_submit_button(
            "Predict Customer Segment",
            use_container_width=True
        )

    if submitted:
        customer_input = pd.DataFrame([{
            "Age": age,
            "Total Spend": total_spend,
            "Membership Type": membership,
            "Discount Applied": discount
        }])

        try:
            cluster_id, segment_name = predict_customer(
                customer_input
            )

            st.session_state["latest_prediction"] = {
                "Customer Name": customer_name.strip() or "New Customer",
                "Age": age,
                "Total Spend": total_spend,
                "Membership Type": membership,
                "Discount Applied": discount,
                "Predicted Cluster": cluster_id,
                "Predicted Segment": segment_name
            }

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if "latest_prediction" in st.session_state:
        result = st.session_state["latest_prediction"]

        st.markdown(
            f"""
            <div class="hover-card prediction-card">
                <p style="color:#C9E4F2;font-weight:800;letter-spacing:2px;">
                    PREDICTION RESULT
                </p>
                <h2 style="color:#F4FAFF;">{result['Predicted Segment']}</h2>
                <p style="color:#E2F0F8;">
                    Predicted cluster: {result['Predicted Cluster']}
                </p>
                <p style="color:#E2F0F8;">
                    {STRATEGIES.get(result['Predicted Segment'], '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Save this customer",
            use_container_width=True
        ):
            saved_record = result.copy()

            saved_record["Saved At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            saved_record["Record ID"] = str(uuid4())
            new_row = pd.DataFrame([saved_record])

            if NEW_CUSTOMERS_FILE.exists():
                existing = pd.read_csv(NEW_CUSTOMERS_FILE)
                if "Record ID" not in existing.columns:
                    existing["Record ID"] = [str(uuid4()) for _ in range(len(existing))]
                updated = pd.concat([existing, new_row], ignore_index=True, sort=False)
            else:
                updated = new_row

            updated.to_csv(
                NEW_CUSTOMERS_FILE,
                index=False
            )

            st.success(
                f"Customer saved successfully to {NEW_CUSTOMERS_FILE.name}."
            )

    st.write("")

    section(
        "Previously added customers",
        "New customer records saved through this page."
    )

    if NEW_CUSTOMERS_FILE.exists():
        saved_customers = pd.read_csv(NEW_CUSTOMERS_FILE)
        if not saved_customers.empty:
            if "Record ID" not in saved_customers.columns:
                saved_customers["Record ID"] = [str(uuid4()) for _ in range(len(saved_customers))]
                saved_customers.to_csv(NEW_CUSTOMERS_FILE, index=False)

            st.dataframe(saved_customers.drop(columns=["Record ID"], errors="ignore"), use_container_width=True, hide_index=True)

            st.markdown("**Select the customer you want to delete**")
            selected_ids = []
            for i, row in saved_customers.iterrows():
                label = (f"{row.get('Customer Name', 'Customer')} — "
                         f"{row.get('Predicted Segment', 'Unknown')} — "
                         f"{row.get('Saved At', 'No date')} (record {i + 1})")
                if st.checkbox(label, key=f"delete_customer_{row['Record ID']}"):
                    selected_ids.append(str(row["Record ID"]))

            confirm_delete = st.checkbox(
                "I confirm that I want to permanently delete the selected customer(s).",
                key="confirm_saved_customer_delete"
            )
            if st.button("Delete selected customer(s)", type="secondary",
                         disabled=(not selected_ids or not confirm_delete),
                         use_container_width=True):
                latest = pd.read_csv(NEW_CUSTOMERS_FILE)
                if "Record ID" not in latest.columns:
                    st.error("Record identifiers are missing. No data was deleted.")
                else:
                    remaining = latest[~latest["Record ID"].astype(str).isin(selected_ids)]
                    deleted_count = len(latest) - len(remaining)
                    if deleted_count == 0:
                        st.warning("Selected record(s) were not found; no data was deleted.")
                    else:
                        remaining.to_csv(NEW_CUSTOMERS_FILE, index=False)
                        st.success(f"Deleted {deleted_count} selected customer record(s).")
                        st.rerun()
        else:
            st.info("No new customers have been saved yet.")

        st.download_button(
            "Download saved customers",
            data=saved_customers.drop(columns=["Record ID"], errors="ignore").to_csv(index=False).encode("utf-8"),
            file_name="new_customers.csv",
            mime="text/csv"
        )
    else:
        st.info("No new customers have been saved yet.")


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<hr>
<p style="text-align:center;color:#DCE8D8;font-size:0.8rem;">
    Customer Segmentation Dashboard
</p>
""", unsafe_allow_html=True)