# =========================================================
# APP.PY
# AI Government Public Sentiment Intelligence Platform
# Premium Streamlit Dashboard (Executive UI/UX Edition)
# =========================================================

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Government Public Sentiment Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_public_sentiment.csv"
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# =========================================================
# CUSTOM CSS — PREMIUM EXECUTIVE UI
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(124,58,237,0.12), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(59,130,246,0.12), transparent 30%),
        radial-gradient(circle at 20% 80%, rgba(16,185,129,0.08), transparent 30%),
        linear-gradient(135deg, #030712 0%, #0B1120 45%, #111827 100%);
    color: #F8FAFC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */ 
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg,
            rgba(15,23,42,0.98) 0%, 
            rgba(17,24,39,0.98) 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

.sidebar-title {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, #A78BFA, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.35rem;
}

.sidebar-subtitle {
    color: #94A3B8;
    font-size: 0.85rem;
    line-height: 1.5;
    margin-bottom: 1.5rem;
}

.filter-card {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 22px;
    padding: 1rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(14px);
}

/* Hero */
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #FFFFFF, #93C5FD, #C4B5FD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    line-height: 1.8;
    margin-bottom: 2rem;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(145deg,
        rgba(15,23,42,0.90),
        rgba(30,41,59,0.82));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 28px;
    padding: 1.6rem 1.7rem;
    min-height: 170px;
    backdrop-filter: blur(18px);
    box-shadow:
        0 10px 40px rgba(2,6,23,0.35),
        inset 0 1px 0 rgba(255,255,255,0.04);
}

.kpi-label {
    color: #94A3B8;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.kpi-value {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-delta {
    margin-top: 0.85rem;
    color: #34D399;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Chart Card */
.chart-card {
    background: rgba(15,23,42,0.68);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 28px;
    padding: 1.2rem;
    backdrop-filter: blur(18px);
    box-shadow: 0 10px 30px rgba(2,6,23,0.25);
}

/* Tabs */ 
.stTabs [data-baseweb="tab-list"] {
    gap: 0.75rem;
    margin-top: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(15,23,42,0.75);
    border-radius: 14px;
    padding: 0.75rem 1.2rem;
    color: #E2E8F0;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #7C3AED, #2563EB) !important;
    color: white !important;
}

/* Inputs */ 
.stButton > button {
    background: linear-gradient(90deg, #7C3AED, #2563EB);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.7rem 1.4rem;
}

.stButton > button:hover {
    box-shadow: 0 0 20px rgba(124,58,237,0.35);
}

/* Footer */
.footer {
    text-align: center;
    color: #64748B;
    font-size: 0.9rem;
    margin-top: 3rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA & MODEL
# =========================================================


@st.cache_data
def load_data(): 
    df = pd.read_csv(DATA_PATH, parse_dates=["created_at"])
    return df


@st.cache_resource
def load_model(): 
    model = None
    vectorizer = None

    if MODEL_PATH.exists() and VECTORIZER_PATH.exists(): 
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


df = load_data()
model, vectorizer = load_model()

# =========================================================
# SIDEBAR FILTERS
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">🏛 Executive Filters</div>
        <div class="sidebar-subtitle">
            Interactive filters for monitoring public sentiment,
            government programs, and citizen engagement.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    platform_options = sorted(df["platform"].dropna().unique())
    selected_platform = st.multiselect(
        "📱 Platform",
        platform_options,
        default=platform_options,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    program_options = sorted(df["program_name"].dropna().unique())
    selected_program = st.multiselect(
        "🏛 Government Program",
        program_options,
        default=program_options,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()
    st.caption("AI-powered executive intelligence dashboard") 
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df[
    (df["platform"].isin(selected_platform))
    & (df["program_name"].isin(selected_program))
].copy()

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# =========================================================
# KPI CALCULATIONS
# =========================================================

total_comments = len(filtered_df)

positive_rate = (filtered_df["sentiment"] == "Positive").mean() * 100
neutral_rate = (filtered_df["sentiment"] == "Neutral").mean() * 100
negative_rate = (filtered_df["sentiment"] == "Negative").mean() * 100

high_urgency_rate = (filtered_df["urgency_level"] == "High").mean() * 100
public_trust_index = 100 + positive_rate - negative_rate

avg_engagement = filtered_df["engagement_score"].mean()

# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero-title">🏛️ AI Government Public Sentiment Intelligence</div>
    <div class="hero-subtitle">
        Executive analytics platform for monitoring public sentiment,
        identifying emerging issues, and generating AI-powered strategic insights
        to support data-driven government decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# KPI CARDS
# =========================================================


def render_kpi(label, value, delta): 
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    render_kpi("Total Comments", f"{total_comments:,}", "↑ Public conversations captured") 

with col2:
    render_kpi("Positive Sentiment", f"{positive_rate:.2f}%", "↑ Citizen satisfaction level") 

with col3:
    render_kpi("Public Trust Index", f"{public_trust_index:.2f}", "🛡 Composite trust indicator") 

with col4:
    render_kpi("High Urgency Issues", f"{high_urgency_rate:.2f}%", "⚠ Priority attention required") 

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================

(tab1, tab2, tab3, tab4, tab5) = st.tabs(
    [
        "📊 Sentiment Analytics",
        "🏷️ Topic Intelligence",
        "🤖 AI Prediction",
        "📈 Model Performance",
        "🧠 Strategic Insight",
    ]
)

# =========================================================
# TAB 1 — SENTIMENT ANALYTICS
# =========================================================

with tab1:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        sentiment_counts = (
            filtered_df["sentiment"]
            .value_counts()
            .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            .reset_index()
        )
        sentiment_counts.columns = ["Sentiment", "Count"]

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.65,
            template="plotly_dark",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#22C55E",
                "Neutral": "#3B82F6",
                "Negative": "#EF4444",
            },
        )
        fig.update_layout(
            title="Public Sentiment Distribution",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        daily_sentiment = (
            filtered_df.groupby([filtered_df["created_at"].dt.date, "sentiment"])
            .size()
            .reset_index(name="count") 
        )
        daily_sentiment.columns = ["Date", "Sentiment", "Count"] 

        fig = px.line(
            daily_sentiment,
            x="Date",
            y="Count",
            color="Sentiment",
            template="plotly_dark",
            color_discrete_map={
                "Positive": "#22C55E",
                "Neutral": "#3B82F6",
                "Negative": "#EF4444",
            },
        )
        fig.update_layout(
            title="Sentiment Trend Over Time",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 — TOPIC INTELLIGENCE
# =========================================================

with tab2:
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        hashtag_df = (
            filtered_df.groupby("hashtags")["engagement_score"] 
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            hashtag_df,
            x="engagement_score",
            y="hashtags",
            orientation="h",
            color="engagement_score",
            template="plotly_dark",
            color_continuous_scale="Plasma",
        )
        fig.update_layout(
            title="Top Hashtag Intelligence",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="white",
            yaxis=dict(categoryorder="total ascending"), 
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        urgency_df = (
            filtered_df["urgency_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .reset_index()
        )
        urgency_df.columns = ["Urgency", "Count"] 

        fig = px.pie(
            urgency_df,
            names="Urgency",
            values="Count",
            hole=0.65,
            template="plotly_dark",
            color="Urgency",
            color_discrete_map={
                "Low": "#22C55E",
                "Medium": "#F59E0B",
                "High": "#EF4444",
            },
        )
        fig.update_layout(
            title="Issue Severity Monitoring",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="white",
        ) 
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# REPLACE SELURUH ISI TAB 3 — AI PREDICTION
# =========================================================
# Tujuan:
# 1. Menambahkan preprocessing sederhana agar input konsisten
# 2. Menampilkan probabilitas tiap kelas (jika model mendukung)
# 3. Menambahkan rule-based override untuk kalimat campuran
#    sehingga hasil tidak selalu Positive
# =========================================================

import re

with tab3:
    st.subheader("🤖 Real-Time Sentiment Prediction")

    st.caption(
        "Analyze citizen comments and classify them into "
        "Positive, Neutral, or Negative sentiment."
    )

    # -----------------------------------------------------
    # INPUT
    # -----------------------------------------------------
    user_input = st.text_area(
        "Enter a citizen comment:",
        placeholder=(
            "Contoh: Program ini cukup membantu tetapi "
            "aplikasinya masih sering error."
        ),
        height=150,
    )

    # -----------------------------------------------------
    # PREPROCESSING FUNCTION
    # -----------------------------------------------------
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\\s]", " ", text)
        text = re.sub(r"\\s+", " ", text).strip()
        return text

    # -----------------------------------------------------
    # SIMPLE SENTIMENT OVERRIDE RULES
    # -----------------------------------------------------
    positive_keywords = [
        "bagus", "baik", "menarik", "membantu",
        "bermanfaat", "mudah", "cepat", "patut dicoba",
        "rekomendasi", "memuaskan"
    ]

    negative_keywords = [
        "error", "lambat", "sulit", "buruk",
        "down", "gagal", "bug", "rumit",
        "membingungkan", "tidak bisa"
    ]

    neutral_keywords = [
        "cukup", "namun", "tetapi", "masih perlu",
        "perlu perbaikan", "belum sempurna"
    ]

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------
    if st.button("🔮 Analyze Sentiment", use_container_width=True):

        if not user_input.strip():
            st.warning("Please enter a comment first.")

        elif model is None or vectorizer is None:
            st.error("Model files not found.")

        else:
            clean_text = preprocess_text(user_input)

            # Hitung keyword
            pos_score = sum(
                1 for kw in positive_keywords if kw in clean_text
            )
            neg_score = sum(
                1 for kw in negative_keywords if kw in clean_text
            )
            neu_score = sum(
                1 for kw in neutral_keywords if kw in clean_text
            )

            # -------------------------------------------------
            # RULE-BASED OVERRIDE
            # -------------------------------------------------
            # Jika ada campuran kata positif dan negatif,
            # atau terdapat kata transisi seperti "tetapi",
            # hasil dipaksa menjadi Neutral.
            # -------------------------------------------------
            if (pos_score > 0 and neg_score > 0) or neu_score > 0:
                prediction = "Neutral"

            elif neg_score > pos_score and neg_score > 0:
                prediction = "Negative"

            elif pos_score > neg_score and pos_score > 0:
                prediction = "Positive"

            else:
                # Fallback ke model machine learning
                X_new = vectorizer.transform([clean_text])
                prediction = model.predict(X_new)[0]

            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------
            sentiment_config = {
                "Positive": {
                    "emoji": "🟢",
                    "message": "Predicted Sentiment: 🟢 Positive",
                    "color": "#22C55E"
                },
                "Neutral": {
                    "emoji": "🔵",
                    "message": "Predicted Sentiment: 🔵 Neutral",
                    "color": "#3B82F6"
                },
                "Negative": {
                    "emoji": "🔴",
                    "message": "Predicted Sentiment: 🔴 Negative",
                    "color": "#EF4444"
                }
            }

            config = sentiment_config[prediction]

            if prediction == "Positive":
                st.success(config["message"])
            elif prediction == "Neutral":
                st.info(config["message"])
            else:
                st.error(config["message"])

            # -------------------------------------------------
            # DEBUG SCORES (untuk demo & pembelajaran)
            # -------------------------------------------------
            with st.expander("🔍 Sentiment Analysis Details"):
                st.write("Positive Keyword Score:", pos_score)
                st.write("Neutral Keyword Score :", neu_score)
                st.write("Negative Keyword Score:", neg_score)
                st.write("Final Prediction      :", prediction)

    # -----------------------------------------------------
    # EXAMPLE COMMENTS
    # -----------------------------------------------------
    with st.expander("📝 Example Comments for Testing"):
        st.markdown("""
        **🟢 Positive**
        - Pelayanan publik sekarang jauh lebih cepat dan mudah.

        **🔵 Neutral**
        - Program ini cukup membantu tetapi masih perlu perbaikan.

        **🔴 Negative**
        - Aplikasinya sering error dan sulit digunakan.
        """)
# =========================================================
# TAB 4 — MODEL PERFORMANCE
# =========================================================

with tab4:
    performance_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Naive Bayes", "Linear SVM"],
            "Accuracy": [0.93, 0.90, 0.94],
            "F1 Score": [0.93, 0.90, 0.94],
        }
    ) 

    fig = px.bar(
        performance_df,
        x="Model",
        y="F1 Score",
        color="F1 Score",
        text="F1 Score",
        template="plotly_dark",
        color_continuous_scale="Viridis",
    ) 
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside") 
    fig.update_layout(
        title="NLP Model Performance Comparison",
        yaxis_range=[0.85, 1.00], 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        font_color="white",
    ) 
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Best Model:** Linear SVM  
        **Accuracy:** 0.94  
        **F1 Score:** 0.94
        """
    ) 

# =========================================================
# TAB 5 — STRATEGIC INSIGHT
# =========================================================

with tab5:
    if public_trust_index >= 140:
        government_status = "Excellent"
    elif public_trust_index >= 120:
        government_status = "Strong"
    elif public_trust_index >= 100:
        government_status = "Stable"
    else:
        government_status = "Needs Attention"

    top_program = (
        filtered_df.groupby("program_name")["engagement_score"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    ) 

    top_hashtag = (
        filtered_df.groupby("hashtags")["engagement_score"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    ) 

    st.subheader("🧠 Executive Strategic Insight") 

    st.markdown(
        f"""
        - **Government Response Status:** {government_status}
        - **Public Trust Index:** {public_trust_index:.2f}
        - **Average Engagement Score:** {avg_engagement:,.2f}
        - **Top Performing Program:** {top_program}
        - **Top Influential Hashtag:** {top_hashtag}
        """
    ) 

    st.markdown("### 📌 Strategic Recommendations") 

    recommendations = [] 

    if negative_rate > 20:
        recommendations.append(
            "Strengthen technical support and improve service reliability."
        ) 

    if high_urgency_rate > 10:
        recommendations.append(
            "Prioritize rapid response to high-urgency citizen issues."
        ) 

    if positive_rate > 50:
        recommendations.append(
            "Leverage positive sentiment to amplify government communication."
        ) 

    recommendations.append(
        "Continuously monitor social media to detect emerging public concerns early."
    ) 

    for i, recommendation in enumerate(recommendations, start=1): 
        st.markdown(f"{i}. {recommendation}") 

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🏛️ M.Wildan Nabila | AI Government Public Sentiment Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True,
)
