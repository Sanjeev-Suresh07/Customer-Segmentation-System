import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from sklearn.metrics import silhouette_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation & Behavior Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

NUMERIC_FEATURES = [
    "Age",
    "Total Spend"
]

CATEGORICAL_FEATURES = [
    "Membership Type",
    "Discount Applied"
]

N_CLUSTERS = 7


# ============================================================
# SEGMENT NAMES
# ============================================================

SEGMENT_NAMES = {
    0: "Bronze Regular Customers",
    1: "Gold High Value Customers",
    2: "Silver Regular Customers",
    3: "Gold Premium Customers",
    4: "Silver Inactive Customers",
    5: "Bronze Discount Customers",
    6: "Silver Moderate Customers"
}


# ============================================================
# MARKETING STRATEGIES
# ============================================================

MARKETING_STRATEGIES = {

    "Bronze Regular Customers":
        "Use personalized recommendations and loyalty rewards to encourage higher spending.",

    "Gold High Value Customers":
        "Focus on retention, exclusive offers, loyalty benefits, and personalized promotions.",

    "Silver Regular Customers":
        "Encourage repeat purchases through targeted recommendations and moderate promotions.",

    "Gold Premium Customers":
        "Provide premium products, VIP benefits, exclusive offers, and priority rewards.",

    "Silver Inactive Customers":
        "Use re-engagement campaigns, reminders, and limited-time offers to encourage return purchases.",

    "Bronze Discount Customers":
        "Use targeted discounts, bundles, and personalized offers to increase purchase frequency.",

    "Silver Moderate Customers":
        "Encourage repeat purchases using bundles, personalized recommendations, and loyalty incentives."
}


# ============================================================
# SEGMENT COLORS
# ============================================================

SEGMENT_COLORS = {

    "Bronze Regular Customers": "#6366F1",

    "Gold High Value Customers": "#10B981",

    "Silver Regular Customers": "#8B5CF6",

    "Gold Premium Customers": "#F59E0B",

    "Silver Inactive Customers": "#EF4444",

    "Bronze Discount Customers": "#EC4899",

    "Silver Moderate Customers": "#3B82F6"
}


# ============================================================
# LOAD DATA + SAVED MODEL
# ============================================================

@st.cache_data
def load_data_and_model():

    # --------------------------------------------------------
    # Load cleaned customer dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        "data/cleaned_customer_behavior.csv"
    )


    # --------------------------------------------------------
    # Load saved preprocessing objects and K-Means model
    # --------------------------------------------------------

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    encoder = joblib.load(
        "models/encoder.pkl"
    )

    kmeans = joblib.load(
        "models/kmeans_model.pkl"
    )


    # --------------------------------------------------------
    # Transform numerical features
    # --------------------------------------------------------

    X_numeric = scaler.transform(
        df[NUMERIC_FEATURES]
    )


    # --------------------------------------------------------
    # Transform categorical features
    # --------------------------------------------------------

    X_categorical = encoder.transform(
        df[CATEGORICAL_FEATURES]
    )


    # --------------------------------------------------------
    # Combine transformed features
    # --------------------------------------------------------

    X_final = pd.concat(
        [
            pd.DataFrame(X_numeric),
            pd.DataFrame(X_categorical)
        ],
        axis=1
    ).values


    # --------------------------------------------------------
    # Generate predictions using saved K-Means model
    # --------------------------------------------------------

    df["Cluster"] = kmeans.predict(
        X_final
    )


    # --------------------------------------------------------
    # Assign segment names
    # --------------------------------------------------------

    df["Segment"] = df["Cluster"].map(
        SEGMENT_NAMES
    )


    # --------------------------------------------------------
    # Assign marketing strategies
    # --------------------------------------------------------

    df["Marketing Strategy"] = df[
        "Segment"
    ].map(
        MARKETING_STRATEGIES
    )


    # --------------------------------------------------------
    # Calculate Silhouette Score
    # --------------------------------------------------------

    silhouette = silhouette_score(
        X_final,
        df["Cluster"]
    )


    return df, silhouette


# Load final application data
df, FINAL_SILHOUETTE = load_data_and_model()

FINAL_CLUSTERS = df["Cluster"].nunique()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {

        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(16,185,129,0.10),
                transparent 30%
            ),

            radial-gradient(
                circle at 10% 10%,
                rgba(99,102,241,0.10),
                transparent 30%
            ),

            #0B0F17;
    }


    .main {
        padding-top: 1rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #111827 0%,
                #0B111B 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.08);
    }


    section[data-testid="stSidebar"] .block-container {

        padding-top: 2rem;
    }


    .sidebar-title {

        font-size: 1.05rem;

        font-weight: 700;

        color: #F8FAFC;

        margin-bottom: 0.8rem;
    }


    .sidebar-description {

        color: #AAB4C5;

        font-size: 0.86rem;

        line-height: 1.65;
    }


    .sidebar-divider {

        border-top:
            1px solid rgba(255,255,255,0.12);

        margin: 1.5rem 0;
    }


    .model-card {

        background:
            linear-gradient(
                135deg,
                rgba(16,185,129,0.10),
                rgba(15,23,42,0.55)
            );

        border:
            1px solid rgba(16,185,129,0.35);

        border-radius: 14px;

        padding: 1rem;

        box-shadow:
            0 8px 30px rgba(0,0,0,0.18);
    }


    .model-line {

        color: #CBD5E1;

        font-size: 0.82rem;

        margin: 0.42rem 0;
    }


    .model-value {

        color: #F8FAFC;

        font-weight: 600;
    }


    .ready-badge {

        display: inline-block;

        margin-top: 0.65rem;

        padding: 0.35rem 0.7rem;

        border-radius: 999px;

        color: #6EE7B7;

        border:
            1px solid rgba(16,185,129,0.45);

        background:
            rgba(16,185,129,0.08);

        font-size: 0.75rem;

        font-weight: 600;
    }


    .pipeline {

        color: #AAB4C5;

        font-size: 0.82rem;

        line-height: 1.75;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {

        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,0.10),
                rgba(16,185,129,0.08)
            );

        border:
            1px solid rgba(255,255,255,0.07);

        border-radius: 18px;

        padding: 1.4rem 1.5rem;

        margin-bottom: 1.25rem;

        box-shadow:
            0 12px 40px rgba(0,0,0,0.16);
    }


    .hero h1 {

        color: #F8FAFC;

        margin: 0 0 0.55rem 0;

        font-size: 2rem;

        font-weight: 800;
    }


    .hero p {

        color: #AAB4C5;

        margin: 0;

        font-size: 0.94rem;

        line-height: 1.7;
    }


    /* ========================================================
       KPI CARDS
       ======================================================== */

    .metric-card {

        background:
            linear-gradient(
                145deg,
                rgba(30,41,59,0.78),
                rgba(15,23,42,0.88)
            );

        border:
            1px solid rgba(148,163,184,0.18);

        border-radius: 16px;

        padding: 1.15rem;

        min-height: 125px;

        box-shadow:
            0 8px 28px rgba(0,0,0,0.18);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }


    .metric-card:hover {

        transform:
            translateY(-3px);

        border-color:
            rgba(16,185,129,0.45);

        box-shadow:
            0 12px 35px rgba(16,185,129,0.10);
    }


    .metric-label {

        color: #94A3B8;

        font-size: 0.78rem;

        margin-bottom: 0.55rem;
    }


    .metric-value {

        color: #F8FAFC;

        font-size: 1.65rem;

        font-weight: 750;
    }


    .metric-accent {

        width: 45px;

        height: 4px;

        border-radius: 10px;

        margin-top: 0.8rem;

        background:
            linear-gradient(
                90deg,
                #10B981,
                #6366F1
            );
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-title {

        color: #F8FAFC;

        font-size: 1.25rem;

        font-weight: 750;

        margin-top: 1.2rem;

        margin-bottom: 0.35rem;
    }


    .section-description {

        color: #94A3B8;

        font-size: 0.86rem;

        margin-bottom: 0.9rem;
    }


    /* ========================================================
       CHART TITLES
       ======================================================== */

    .chart-title {

        color: #F8FAFC;

        font-size: 0.92rem;

        font-weight: 700;

        margin:
            0.15rem 0
            0.3rem 0.35rem;
    }


    .chart-subtitle {

        color: #64748B;

        font-size: 0.75rem;

        margin:
            0 0
            0.35rem 0.35rem;
    }


    /* ========================================================
       SEGMENT CARD
       ======================================================== */

    .segment-card {

        background:
            linear-gradient(
                145deg,
                rgba(30,41,59,0.75),
                rgba(15,23,42,0.9)
            );

        border:
            1px solid rgba(148,163,184,0.16);

        border-radius: 16px;

        padding: 1.1rem;

        margin-top: 0.5rem;

        box-shadow:
            0 10px 30px rgba(0,0,0,0.18);
    }


    .segment-name {

        color: #F8FAFC;

        font-size: 1.05rem;

        font-weight: 750;

        margin-bottom: 0.5rem;
    }


    .segment-text {

        color: #AAB4C5;

        font-size: 0.84rem;

        line-height: 1.6;
    }


    /* ========================================================
       INFO CARD
       ======================================================== */

    .info-card {

        background:
            linear-gradient(
                135deg,
                rgba(16,185,129,0.08),
                rgba(15,23,42,0.80)
            );

        border-left:
            3px solid #10B981;

        border-radius: 12px;

        padding: 1rem 1.1rem;

        margin: 1rem 0;

        box-shadow:
            0 8px 25px rgba(16,185,129,0.06);
    }


    .info-title {

        color: #6EE7B7;

        font-weight: 700;

        margin-bottom: 0.35rem;
    }


    .info-text {

        color: #CBD5E1;

        line-height: 1.6;

        font-size: 0.86rem;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {

        text-align: center;

        color: #64748B;

        font-size: 0.75rem;

        padding:
            1.5rem 0
            0.5rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">'
        '📊 Customer Analytics'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-description">'
        'Analyze customer purchasing behavior, identify meaningful '
        'customer segments, and discover targeted marketing '
        'opportunities.'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-title">'
        'Model Information'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="model-card">'

        '<div class="model-line">'
        'Algorithm: '
        '<span class="model-value">'
        'K-Means Clustering'
        '</span>'
        '</div>'

        '<div class="model-line">'
        'Clusters: '
        f'<span class="model-value">'
        f'{FINAL_CLUSTERS}'
        '</span>'
        '</div>'

        '<div class="model-line">'
        'Numerical Features: '
        '<span class="model-value">'
        '2'
        '</span>'
        '</div>'

        '<div class="model-line">'
        'Categorical Features: '
        '<span class="model-value">'
        '2'
        '</span>'
        '</div>'

        '<div class="model-line">'
        'Scaling: '
        '<span class="model-value">'
        'MinMaxScaler'
        '</span>'
        '</div>'

        '<div class="model-line">'
        'Encoding: '
        '<span class="model-value">'
        'OneHotEncoder'
        '</span>'
        '</div>'

        '<div class="model-line">'
        'Silhouette Score: '
        f'<span class="model-value">'
        f'{FINAL_SILHOUETTE:.3f}'
        '</span>'
        '</div>'

        '<span class="ready-badge">'
        '● Model Ready'
        '</span>'

        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # INTERACTIVE FILTERS
    # ========================================================

    st.markdown(
        '<div class="sidebar-title">'
        'Dashboard Filters'
        '</div>',
        unsafe_allow_html=True
    )


    membership_options = sorted(
        df[
            "Membership Type"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    selected_memberships = st.multiselect(
        "Membership Type",

        membership_options,

        default=membership_options
    )


    discount_options = sorted(
        df[
            "Discount Applied"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    selected_discount = st.multiselect(
        "Discount Applied",

        discount_options,

        default=discount_options,

        format_func=lambda x:
            "Yes" if x else "No"
    )


    filtered_df = df[
        df[
            "Membership Type"
        ].isin(
            selected_memberships
        )
        &
        df[
            "Discount Applied"
        ].isin(
            selected_discount
        )
    ].copy()


    st.caption(
        f"Showing {len(filtered_df)} "
        f"of {len(df)} customers"
    )


    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-title">'
        'Project Pipeline'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="pipeline">'
        'Dataset → Cleaning → EDA → '
        'Feature Selection → Scaling → Encoding → '
        'K-Means → Customer Segments → Marketing Insights'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'

    '<h1>'
    '📊 Customer Segmentation & Behavior Analysis'
    '</h1>'

    '<p>'
    'Transform customer behavior into actionable insights. '
    'Explore purchasing patterns, understand customer segments, '
    'and identify targeted marketing opportunities using '
    'data-driven K-Means clustering.'
    '</p>'

    '</div>',

    unsafe_allow_html=True
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(
    filtered_df
)


segment_count = filtered_df[
    "Segment"
].nunique()


average_spend = (

    filtered_df[
        "Total Spend"
    ].mean()

    if len(filtered_df) > 0

    else 0
)


average_rating = (

    filtered_df[
        "Average Rating"
    ].mean()

    if len(filtered_df) > 0

    else 0
)


high_value_count = len(
    filtered_df[
        filtered_df[
            "Segment"
        ]
        ==
        "Gold High Value Customers"
    ]
)


# ============================================================
# KPI CARDS
# ============================================================

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)


with kpi1:

    st.markdown(
        '<div class="metric-card">'

        '<div class="metric-label">'
        'Total Customers'
        '</div>'

        f'<div class="metric-value">'
        f'{total_customers}'
        f'</div>'

        '<div class="metric-accent"></div>'

        '</div>',

        unsafe_allow_html=True
    )


with kpi2:

    st.markdown(
        '<div class="metric-card">'

        '<div class="metric-label">'
        'Customer Segments'
        '</div>'

        f'<div class="metric-value">'
        f'{segment_count}'
        f'</div>'

        '<div class="metric-accent"></div>'

        '</div>',

        unsafe_allow_html=True
    )


with kpi3:

    st.markdown(
        '<div class="metric-card">'

        '<div class="metric-label">'
        'Average Spend'
        '</div>'

        f'<div class="metric-value">'
        f'₹{average_spend:,.2f}'
        f'</div>'

        '<div class="metric-accent"></div>'

        '</div>',

        unsafe_allow_html=True
    )


with kpi4:

    st.markdown(
        '<div class="metric-card">'

        '<div class="metric-label">'
        'Average Rating'
        '</div>'

        f'<div class="metric-value">'
        f'{average_rating:.2f}'
        f'</div>'

        '<div class="metric-accent"></div>'

        '</div>',

        unsafe_allow_html=True
    )


with kpi5:

    st.markdown(
        '<div class="metric-card">'

        '<div class="metric-label">'
        'High Value Customers'
        '</div>'

        f'<div class="metric-value">'
        f'{high_value_count}'
        f'</div>'

        '<div class="metric-accent"></div>'

        '</div>',

        unsafe_allow_html=True
    )


# ============================================================
# OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Customer Segmentation Overview'
    '</div>',

    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-description">'
    'The final K-Means model groups customers using demographic, '
    'spending, membership, and discount characteristics. '
    'Use the interactive filters in the sidebar to explore '
    'different customer populations.'
    '</div>',

    unsafe_allow_html=True
)


# ============================================================
# CHART 1 — CUSTOMER DISTRIBUTION
# ============================================================

segment_counts = (
    filtered_df[
        "Segment"
    ]
    .value_counts()
    .reset_index()
)


segment_counts.columns = [
    "Segment",
    "Customers"
]


segment_counts = segment_counts.sort_values(
    "Customers",
    ascending=True
)


fig_distribution = px.bar(

    segment_counts,

    x="Customers",

    y="Segment",

    orientation="h",

    text="Customers",

    color="Segment",

    color_discrete_map=SEGMENT_COLORS
)


fig_distribution.update_traces(

    textposition="outside",

    hovertemplate=(
        "<b>%{y}</b><br>"
        "Customers: %{x}"
        "<extra></extra>"
    )
)


fig_distribution.update_layout(

    height=390,

    margin=dict(
        l=5,
        r=30,
        t=5,
        b=5
    ),

    showlegend=False,

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="#CBD5E1"
    ),

    xaxis=dict(
        title="Customers",

        gridcolor=
        "rgba(148,163,184,0.15)"
    ),

    yaxis=dict(
        title="",

        categoryorder=
        "total ascending"
    )
)


# ============================================================
# CHART 2 — SPENDING VS ITEMS
# ============================================================

fig_scatter = px.scatter(

    filtered_df,

    x="Total Spend",

    y="Items Purchased",

    color="Segment",

    color_discrete_map=SEGMENT_COLORS,

    hover_data=[
        "Customer ID",
        "Age",
        "Average Rating",
        "Days Since Last Purchase",
        "Membership Type",
        "Discount Applied"
    ]
)


fig_scatter.update_traces(

    marker=dict(
        size=9,
        opacity=0.82
    )
)


fig_scatter.update_layout(

    height=390,

    margin=dict(
        l=5,
        r=5,
        t=5,
        b=5
    ),

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="#CBD5E1"
    ),

    legend=dict(

        title="Customer Segment",

        font=dict(
            size=10
        )
    ),

    xaxis=dict(

        title="Total Spend",

        gridcolor=
        "rgba(148,163,184,0.15)"
    ),

    yaxis=dict(

        title="Items Purchased",

        gridcolor=
        "rgba(148,163,184,0.15)"
    )
)


# ============================================================
# CHART CONTAINERS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    with st.container(border=True):

        st.markdown(
            '<div class="chart-title">'
            'Customer Distribution by Segment'
            '</div>'

            '<div class="chart-subtitle">'
            'Interactive segment population overview'
            '</div>',

            unsafe_allow_html=True
        )


        st.plotly_chart(

            fig_distribution,

            use_container_width=True,

            config={
                "displayModeBar": "hover",
                "displaylogo": False
            }
        )


with col2:

    with st.container(border=True):

        st.markdown(
            '<div class="chart-title">'
            'Spending vs Items Purchased'
            '</div>'

            '<div class="chart-subtitle">'
            'Hover over customers for detailed behavior'
            '</div>',

            unsafe_allow_html=True
        )


        st.plotly_chart(

            fig_scatter,

            use_container_width=True,

            config={
                "displayModeBar": "hover",
                "displaylogo": False
            }
        )


# ============================================================
# INTERACTIVE TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔎 Segment Explorer",
        "📋 Segment Summary",
        "👥 Customer Data"
    ]
)


# ============================================================
# TAB 1 — SEGMENT EXPLORER
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        'Explore a Customer Segment'
        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-description">'
        'Select a segment to examine its purchasing behavior, '
        'customer size, and recommended marketing approach.'
        '</div>',

        unsafe_allow_html=True
    )


    available_segments = sorted(
        filtered_df[
            "Segment"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    if available_segments:

        selected_segment = st.selectbox(

            "Select Customer Segment",

            available_segments
        )


        segment_df = filtered_df[
            filtered_df[
                "Segment"
            ]
            ==
            selected_segment
        ]


        segment_customers = len(
            segment_df
        )


        segment_spend = (
            segment_df[
                "Total Spend"
            ].mean()
        )


        segment_items = (
            segment_df[
                "Items Purchased"
            ].mean()
        )


        segment_days = (
            segment_df[
                "Days Since Last Purchase"
            ].mean()
        )


        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )


        with metric1:

            st.metric(
                "Customers",
                segment_customers
            )


        with metric2:

            st.metric(
                "Average Spend",
                f"₹{segment_spend:,.2f}"
            )


        with metric3:

            st.metric(
                "Average Items",
                f"{segment_items:.2f}"
            )


        with metric4:

            st.metric(
                "Days Since Purchase",
                f"{segment_days:.2f}"
            )


        strategy = (
            segment_df[
                "Marketing Strategy"
            ].iloc[0]
        )


        st.markdown(
            '<div class="segment-card">'

            f'<div class="segment-name">'
            f'{selected_segment}'
            f'</div>'

            '<div class="segment-text">'

            f'<b>Marketing Strategy:</b> '
            f'{strategy}'

            '</div>'

            '</div>',

            unsafe_allow_html=True
        )


    else:

        st.warning(
            "No customer segments match the "
            "current sidebar filters."
        )


# ============================================================
# TAB 2 — SEGMENT SUMMARY
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">'
        'Segment Summary'
        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-description">'
        'Compare the main behavioral characteristics '
        'of each customer segment.'
        '</div>',

        unsafe_allow_html=True
    )


    summary = filtered_df.groupby(
        "Segment"
    ).agg(

        Customers=(
            "Customer ID",
            "count"
        ),

        Avg_Spend=(
            "Total Spend",
            "mean"
        ),

        Avg_Items=(
            "Items Purchased",
            "mean"
        ),

        Avg_Rating=(
            "Average Rating",
            "mean"
        ),

        Avg_Days=(
            "Days Since Last Purchase",
            "mean"
        )

    ).reset_index()


    summary.columns = [

        "Customer Segment",

        "Customers",

        "Average Spend",

        "Average Items",

        "Average Rating",

        "Days Since Purchase"
    ]


    summary[
        [
            "Average Spend",
            "Average Items",
            "Average Rating",
            "Days Since Purchase"
        ]
    ] = summary[
        [
            "Average Spend",
            "Average Items",
            "Average Rating",
            "Days Since Purchase"
        ]
    ].round(2)


    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# TAB 3 — CUSTOMER DATA
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">'
        'Customer Data'
        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-description">'
        'Explore the customer-level records produced '
        'by the final segmentation model.'
        '</div>',

        unsafe_allow_html=True
    )


    search_customer = st.text_input(

        "Search Customer ID",

        placeholder="Enter Customer ID..."
    )


    display_df = filtered_df.copy()


    if search_customer:

        display_df = display_df[
            display_df[
                "Customer ID"
            ]
            .astype(str)
            .str.contains(
                search_customer,
                case=False,
                na=False
            )
        ]


    st.dataframe(

        display_df,

        use_container_width=True,

        hide_index=True
    )


    csv_data = display_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(

        label="⬇ Download Filtered Customer Data",

        data=csv_data,

        file_name=
        "customer_segments_filtered.csv",

        mime="text/csv"
    )


# ============================================================
# FINAL MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="info-card">'

    '<div class="info-title">'
    'Final Model Performance'
    '</div>'

    '<div class="info-text">'

    'The final K-Means customer segmentation model uses '
    '<b>Age</b>, <b>Total Spend</b>, '
    '<b>Membership Type</b>, and '
    '<b>Discount Applied</b>. Numerical features are '
    'processed using MinMaxScaler and categorical features '
    'using OneHotEncoder. The model uses '

    f'<b>{FINAL_CLUSTERS}</b> clusters and achieved a '
    f'Silhouette Score of '
    f'<b>{FINAL_SILHOUETTE:.6f}</b>.'

    '</div>'

    '</div>',

    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Customer Segmentation & Behavior Analysis • '
    'K-Means Clustering • '
    'Data-Driven Marketing Insights'
    '</div>',

    unsafe_allow_html=True
)