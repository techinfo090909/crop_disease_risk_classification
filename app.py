import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crop Disease Risk Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# REFERENCE DATA — field guide + model comparison
# (from the project's own EDA / model-comparison run)
# ============================================================

FIELD_GUIDE = {
    "Crop": "Crop species being monitored. Different crops carry different baseline susceptibility to a given disease.",
    "Disease": "Target disease being assessed. Each disease has its own favourable environmental triggers.",
    "Growth Stage": "Current growth phase of the crop. Plants are often most vulnerable at specific stages (e.g. flowering).",
    "Day of Year": "Calendar day (1–365). Captures seasonal effects on disease pressure.",
    "Average Temperature": "Mean daily temperature (°C). Most fungal/bacterial pathogens have an optimal temperature range for growth.",
    "Average Humidity": "Mean relative humidity (%). High humidity prolongs leaf wetness and favours spore germination.",
    "Rainfall": "Total rainfall (mm). Rain can splash-disperse pathogens and raises ambient humidity.",
    "Leaf Wetness": "Hours the leaf surface stayed wet. A key driver of fungal infection — most pathogens need a minimum wetness duration to infect.",
    "Wind Speed": "Average wind speed (km/h). Affects spore dispersal range and how quickly the canopy dries.",
    "Soil Moisture": "Soil moisture (%). Influences plant stress levels and root-borne disease risk.",
    "Prior Regional Incidence": "Disease incidence (%) already reported in the region — a strong proxy for local inoculum pressure.",
}

MODEL_COMPARISON = pd.DataFrame({
    "Model": ["KNN", "Decision Tree", "Logistic Regression"],
    "Accuracy": [0.6915, 0.6725, 0.6860],
    "Precision": [0.6909, 0.6767, 0.6962],
    "Recall": [0.6735, 0.6540, 0.6415],
    "F1 Score": [0.6810, 0.6631, 0.6599],
})
BEST_MODEL_NAME = "KNN"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #f0f7f3 0%, #f5f7f6 45%, #eef2f0 100%);
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ---------------- SIDEBAR ---------------- */

    section[data-testid="stSidebar"] { background: #12372A; }
    section[data-testid="stSidebar"] * { color: #e7f2ec !important; }

    .sidebar-title {
        font-family: 'Poppins', sans-serif;
        font-size: 20px; font-weight: 700;
        color: #ffffff !important; margin-bottom: 2px;
    }
    .sidebar-sub { font-size: 12.5px; color: #a9c9ba !important; margin-bottom: 18px; }
    .sidebar-chip {
        display: inline-block; background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.18); border-radius: 8px;
        padding: 3px 10px; font-size: 11.5px; font-weight: 600;
        margin: 3px 4px 3px 0; color: #d9eee4 !important;
    }
    .sidebar-feature {
        font-size: 13px; padding: 5px 0;
        border-bottom: 1px solid rgba(255,255,255,0.08); color: #d9eee4 !important;
    }
    .best-model-badge {
        background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.25);
        border-radius: 12px; padding: 10px 14px; margin-top: 6px;
    }

    /* ---------------- HERO ---------------- */

    .hero {
        background: linear-gradient(135deg, #12372A 0%, #1F5C43 55%, #2C7A56 100%);
        padding: 34px 40px; border-radius: 22px; margin-bottom: 26px;
        box-shadow: 0 14px 34px rgba(18, 55, 42, 0.22);
    }
    .hero-badge {
        display: inline-block; background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.3); color: #eafff4;
        font-size: 11.5px; font-weight: 700; letter-spacing: 0.8px;
        text-transform: uppercase; padding: 5px 12px; border-radius: 20px; margin-bottom: 12px;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif; color: white;
        font-size: 36px; font-weight: 800; margin: 0; letter-spacing: -0.6px;
    }
    .hero-subtitle {
        color: #d9eee4; font-size: 15.5px; margin-top: 8px;
        margin-bottom: 0; max-width: 650px; line-height: 1.5;
    }

    /* ---------------- TABS ---------------- */

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: white; padding: 6px; border-radius: 14px;
        border: 1px solid #e1e8e4; box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px; border-radius: 10px; padding: 0 18px;
        font-weight: 650; color: #4a5952;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1F5C43 0%, #12372A 100%) !important;
        color: white !important;
    }

    /* ---------------- SECTION HEADERS ---------------- */

    .section-title {
        font-family: 'Poppins', sans-serif; color: #12372A;
        font-size: 21px; font-weight: 700; margin-top: 6px; margin-bottom: 4px;
    }
    .section-description { color: #66736d; font-size: 13.5px; margin-bottom: 16px; }
    .group-label {
        color: #1F5C43; font-size: 12.5px; font-weight: 700;
        letter-spacing: 0.6px; text-transform: uppercase; margin: 4px 0 10px 0; opacity: 0.85;
    }

    /* ---------------- CARDS ---------------- */

    .input-card {
        background: white; padding: 26px 26px 10px 26px; border-radius: 18px;
        border: 1px solid #e1e8e4; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04); margin-bottom: 22px;
    }
    .guide-card {
        background: white; border: 1px solid #e1e8e4; border-left: 5px solid #1F5C43;
        border-radius: 14px; padding: 16px 20px; margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.03);
    }
    .guide-card b { color: #12372A; font-size: 14.5px; }
    .guide-card p { color: #5c6b63; font-size: 13.5px; margin: 4px 0 0 0; line-height: 1.5; }

    .prediction-card {
        border-radius: 22px; padding: 32px; text-align: center;
        margin-top: 22px; margin-bottom: 10px; color: white;
        box-shadow: 0 16px 34px rgba(0,0,0,0.14);
    }
    .risk-low { background: linear-gradient(135deg, #1F7A4D 0%, #2FA968 100%); }
    .risk-medium { background: linear-gradient(135deg, #B8860B 0%, #E0A526 100%); }
    .risk-high { background: linear-gradient(135deg, #9E2A2B 0%, #C63838 100%); }

    .prediction-label {
        color: rgba(255,255,255,0.85); font-size: 12.5px; font-weight: 700;
        letter-spacing: 1.4px; text-transform: uppercase;
    }
    .prediction-risk {
        font-family: 'Poppins', sans-serif; color: white; font-size: 44px;
        font-weight: 800; margin-top: 6px; margin-bottom: 6px;
    }
    .prediction-description {
        color: rgba(255,255,255,0.92); font-size: 14px; max-width: 560px;
        margin: 0 auto; line-height: 1.5;
    }

    /* ---------------- METRIC CARDS ---------------- */

    .metric-card {
        background: white; border: 1px solid #e1e8e4; border-left: 5px solid #1F5C43;
        border-radius: 14px; padding: 18px 20px; box-shadow: 0 5px 16px rgba(0, 0, 0, 0.04);
    }
    .metric-title { color: #66736d; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.7px; }
    .metric-value { color: #12372A; font-family: 'Poppins', sans-serif; font-size: 26px; font-weight: 800; margin-top: 6px; }

    /* ---------------- PROBABILITY BARS ---------------- */

    .prob-row { margin-bottom: 14px; }
    .prob-row-label { display: flex; justify-content: space-between; font-size: 13.5px; font-weight: 650; color: #33443c; margin-bottom: 5px; }
    .prob-track { background: #e7ece9; border-radius: 8px; height: 12px; width: 100%; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 8px; transition: width 0.6s ease; }
    .fill-low   { background: linear-gradient(90deg, #1F7A4D, #2FA968); }
    .fill-medium{ background: linear-gradient(90deg, #B8860B, #E0A526); }
    .fill-high  { background: linear-gradient(90deg, #9E2A2B, #C63838); }

    .summary-card {
        background: white; border-radius: 18px; padding: 22px;
        border: 1px solid #e1e8e4; box-shadow: 0 5px 18px rgba(0, 0, 0, 0.04);
    }

    .best-badge {
        display: inline-block; background: #eafff4; color: #12372A;
        border: 1px solid #bfe6d2; border-radius: 20px; padding: 4px 12px;
        font-size: 12px; font-weight: 700;
    }

    /* ---------------- BUTTONS ---------------- */

    div.stFormSubmitButton > button {
        width: 100%; height: 54px; border-radius: 12px; border: none;
        background: linear-gradient(135deg, #1F5C43 0%, #12372A 100%);
        color: white; font-size: 16.5px; font-weight: 700; letter-spacing: 0.2px;
        transition: 0.2s ease; box-shadow: 0 8px 20px rgba(31, 92, 67, 0.28);
    }
    div.stFormSubmitButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 10px 24px rgba(31, 92, 67, 0.36);
        background: linear-gradient(135deg, #12372A 0%, #0b241a 100%); color: white; border: none;
    }
    div.stFormSubmitButton > button[kind="secondary"] {
        background: white; color: #33443c; border: 1.5px solid #c9d6cf;
        box-shadow: none; font-weight: 650;
    }
    div.stFormSubmitButton > button[kind="secondary"]:hover {
        border-color: #1F5C43; color: #1F5C43; background: #f3f8f5;
    }

    /* ---------------- INPUT LABELS ---------------- */

    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #33443c !important; font-weight: 600 !important; font-size: 13.5px !important;
    }

    /* ---------------- FOOTER ---------------- */

    .footer {
        text-align: center; color: #7a8781; font-size: 12px;
        margin-top: 35px; padding-top: 18px; border-top: 1px solid #dfe6e2;
    }
    .footer-stack {
        display: inline-block; background: #eef4f0; border-radius: 8px;
        padding: 3px 10px; margin: 0 3px; font-weight: 600; color: #4a5952;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = Path("models/crop_disease_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Model could not be loaded.")
    st.code(str(e))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">🌱 Crop Disease Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">ML Classification Prototype</div>', unsafe_allow_html=True)

    st.markdown(
        '<span class="sidebar-chip">Agriculture</span>'
        '<span class="sidebar-chip">Classification</span>'
        '<span class="sidebar-chip">Free &amp; Open-Source</span>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="best-model-badge">
            <div style="font-size:11px; letter-spacing:0.6px; text-transform:uppercase; color:#a9c9ba;">Deployed Model</div>
            <div style="font-size:17px; font-weight:800; color:white; font-family:'Poppins',sans-serif;">{BEST_MODEL_NAME}</div>
            <div style="font-size:12px; color:#cfe6da;">Selected via macro F1-score comparison</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title" style="font-size:15px;">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px; line-height:1.55; color:#cfe6da;">'
        'Predicts crop disease risk (Low / Medium / High) from crop, disease and '
        'environmental readings using a trained classification model.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11.5px; color:#8fae9d;">Educational prototype — '
        'not a substitute for professional agronomic advice.</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-badge">🔬 Machine Learning Prototype</div>
    <div class="hero-title">🌱 Crop Disease Risk Analyzer</div>
    <div class="hero-subtitle">
        Enter crop, disease and environmental readings to classify disease risk
        into Low, Medium or High — with a full probability and confidence breakdown.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab_analyze, tab_guide, tab_insights = st.tabs(
    ["🔍  Risk Analyzer", "📖  Field Guide", "📊  Model Insights"]
)


# ============================================================
# TAB 1 — RISK ANALYZER
# ============================================================

with tab_analyze:

    st.markdown(
        '<div class="section-title">🌾 Crop &amp; Environmental Information</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-description">'
        'Fill in the details below — hover the <b>ⓘ</b> next to any field for what it means — '
        'then click <b>Analyze Disease Risk</b>.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    with st.form(key="risk_form"):

        st.markdown('<div class="group-label">🌿 Crop Details</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            crop = st.selectbox(
                "Crop", ["Cotton", "Rice", "Potato", "Wheat", "Grape"],
                help=FIELD_GUIDE["Crop"]
            )
        with col2:
            disease = st.selectbox(
                "Disease",
                ["Bacterial Blight", "Blast", "Late Blight", "Stripe Rust", "Powdery Mildew"],
                help=FIELD_GUIDE["Disease"]
            )
        with col3:
            growth_stage = st.selectbox(
                "Growth Stage", ["Seedling", "Vegetative", "Flowering", "Maturity"],
                help=FIELD_GUIDE["Growth Stage"]
            )

        st.markdown('<div class="group-label">🌤️ Weather Conditions</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            day_of_year = st.number_input(
                "Day of Year", min_value=1, max_value=365, value=180, step=1,
                help=FIELD_GUIDE["Day of Year"]
            )
        with col2:
            avg_temp_c = st.number_input(
                "Average Temperature (°C)", min_value=-20.0, max_value=60.0, value=28.0, step=0.1,
                help=FIELD_GUIDE["Average Temperature"]
            )
        with col3:
            avg_humidity_pct = st.number_input(
                "Average Humidity (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1,
                help=FIELD_GUIDE["Average Humidity"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            rainfall_mm = st.number_input(
                "Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0,
                help=FIELD_GUIDE["Rainfall"]
            )
        with col2:
            leaf_wetness_hours = st.number_input(
                "Leaf Wetness (hours)", min_value=0.0, max_value=24.0, value=8.0, step=0.5,
                help=FIELD_GUIDE["Leaf Wetness"]
            )
        with col3:
            wind_speed_kmph = st.number_input(
                "Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=8.0, step=0.5,
                help=FIELD_GUIDE["Wind Speed"]
            )

        st.markdown('<div class="group-label">🌱 Field &amp; Historical Data</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            soil_moisture_pct = st.number_input(
                "Soil Moisture (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5,
                help=FIELD_GUIDE["Soil Moisture"]
            )
        with col2:
            prior_regional_incidence_pct = st.number_input(
                "Prior Regional Incidence (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5,
                help=FIELD_GUIDE["Prior Regional Incidence"]
            )
        with col3:
            st.write("")

        st.markdown("<br>", unsafe_allow_html=True)

        fcol1, fcol2 = st.columns([3, 1])
        with fcol1:
            predict_button = st.form_submit_button("🔍  Analyze Disease Risk", use_container_width=True)
        with fcol2:
            reset_button = st.form_submit_button("↺  Reset", use_container_width=True, type="secondary")

    st.markdown('</div>', unsafe_allow_html=True)

    if reset_button:
        st.rerun()

    if predict_button:

        input_data = pd.DataFrame({
            "crop": [crop],
            "disease": [disease],
            "growth_stage": [growth_stage],
            "day_of_year": [day_of_year],
            "avg_temp_c": [avg_temp_c],
            "avg_humidity_pct": [avg_humidity_pct],
            "rainfall_mm": [rainfall_mm],
            "leaf_wetness_hours": [leaf_wetness_hours],
            "wind_speed_kmph": [wind_speed_kmph],
            "soil_moisture_pct": [soil_moisture_pct],
            "prior_regional_incidence_pct": [prior_regional_incidence_pct]
        })

        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            probability_dict = {
                str(cls): float(prob) for cls, prob in zip(classes, probabilities)
            }

            risk_descriptions = {
                "Low": "Conditions are largely unfavourable for disease development. Routine monitoring is sufficient.",
                "Medium": "Some conditions favour disease development. Increased monitoring is advisable.",
                "High": "Conditions strongly favour disease development. Preventive action is recommended."
            }
            risk_css_class = {
                "Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"
            }.get(str(prediction), "risk-medium")

            low_prob = probability_dict.get("Low", 0) * 100
            medium_prob = probability_dict.get("Medium", 0) * 100
            high_prob = probability_dict.get("High", 0) * 100

            # composite 0-100 weighted risk score, used only for the gauge visual
            risk_score = (low_prob * 0) + (medium_prob * 0.5) + (high_prob * 1.0)

            st.markdown('<div class="section-title">📊 Risk Assessment Result</div>', unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1.1, 1])

            with res_col1:
                st.markdown(f"""
                <div class="prediction-card {risk_css_class}">
                    <div class="prediction-label">Predicted Disease Risk</div>
                    <div class="prediction-risk">{str(prediction).upper()} RISK</div>
                    <div class="prediction-description">
                        {risk_descriptions.get(str(prediction), "Risk category predicted by the trained machine learning model.")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    number={"suffix": "", "font": {"size": 40, "family": "Poppins"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#66736d"},
                        "bar": {"color": "#12372A", "thickness": 0.28},
                        "bgcolor": "white",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 33], "color": "#bdeccf"},
                            {"range": [33, 66], "color": "#f6dfa0"},
                            {"range": [66, 100], "color": "#f3b3b3"},
                        ],
                    },
                    title={"text": "Composite Risk Score", "font": {"size": 14, "color": "#66736d"}}
                ))
                gauge.update_layout(
                    height=230, margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"}
                )
                st.plotly_chart(gauge, use_container_width=True)

            st.markdown('<div class="section-title">📈 Risk Probability</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🟢 Low Risk</div>
                    <div class="metric-value">{low_prob:.2f}%</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🟡 Medium Risk</div>
                    <div class="metric-value">{medium_prob:.2f}%</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🔴 High Risk</div>
                    <div class="metric-value">{high_prob:.2f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-row-label"><span>Low Risk</span><span>{low_prob:.2f}%</span></div>
                <div class="prob-track"><div class="prob-fill fill-low" style="width:{low_prob:.2f}%;"></div></div>
            </div>
            <div class="prob-row">
                <div class="prob-row-label"><span>Medium Risk</span><span>{medium_prob:.2f}%</span></div>
                <div class="prob-track"><div class="prob-fill fill-medium" style="width:{medium_prob:.2f}%;"></div></div>
            </div>
            <div class="prob-row">
                <div class="prob-row-label"><span>High Risk</span><span>{high_prob:.2f}%</span></div>
                <div class="prob-track"><div class="prob-fill fill-high" style="width:{high_prob:.2f}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 Prediction Input Summary</div>', unsafe_allow_html=True)
            st.markdown('<div class="summary-card">', unsafe_allow_html=True)

            summary = pd.DataFrame({
                "Parameter": [
                    "Crop", "Disease", "Growth Stage", "Day of Year",
                    "Average Temperature", "Average Humidity", "Rainfall",
                    "Leaf Wetness", "Wind Speed", "Soil Moisture",
                    "Prior Regional Incidence"
                ],
                "Value": [
                    crop, disease, growth_stage, day_of_year,
                    f"{avg_temp_c:.1f} °C", f"{avg_humidity_pct:.1f} %",
                    f"{rainfall_mm:.1f} mm", f"{leaf_wetness_hours:.1f} hours",
                    f"{wind_speed_kmph:.1f} km/h", f"{soil_moisture_pct:.1f} %",
                    f"{prior_regional_incidence_pct:.1f} %"
                ]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                "ℹ️ This application is an educational machine-learning prototype. "
                "The prediction should not be treated as a professional agricultural diagnosis."
            )

        except Exception as e:
            st.error("⚠️ Prediction failed.")
            st.write("Please check that the saved model was trained with the same input features used by this application.")
            st.code(str(e))


# ============================================================
# TAB 2 — FIELD GUIDE
# ============================================================

with tab_guide:
    st.markdown('<div class="section-title">📖 Field Guide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">What each input means and why it matters for disease-risk prediction.</div>',
        unsafe_allow_html=True
    )

    guide_col1, guide_col2 = st.columns(2)
    items = list(FIELD_GUIDE.items())
    half = (len(items) + 1) // 2

    with guide_col1:
        for name, desc in items[:half]:
            st.markdown(f'<div class="guide-card"><b>{name}</b><p>{desc}</p></div>', unsafe_allow_html=True)

    with guide_col2:
        for name, desc in items[half:]:
            st.markdown(f'<div class="guide-card"><b>{name}</b><p>{desc}</p></div>', unsafe_allow_html=True)


# ============================================================
# TAB 3 — MODEL INSIGHTS
# ============================================================

with tab_insights:
    st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">Three foundational classifiers were trained and compared on the held-out test set.</div>',
        unsafe_allow_html=True
    )

    st.markdown(f'<span class="best-badge">✓ Deployed model: {BEST_MODEL_NAME} — selected by highest macro F1-score</span>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    display_df = MODEL_COMPARISON.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = (display_df[col] * 100).map(lambda v: f"{v:.2f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    bar = go.Figure()
    for metric, color in zip(
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        ["#1F5C43", "#2FA968", "#E0A526", "#12372A"]
    ):
        bar.add_trace(go.Bar(
            name=metric,
            x=MODEL_COMPARISON["Model"],
            y=MODEL_COMPARISON[metric] * 100,
            marker_color=color
        ))
    bar.update_layout(
        barmode="group", height=340,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        legend={"orientation": "h", "y": -0.2},
        yaxis={"title": "Score (%)", "range": [0, 100]}
    )
    st.plotly_chart(bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 How It Works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="guide-card">'
        '<p>Crop, disease and growth-stage fields are one-hot encoded; numeric weather and field readings are scaled. '
        'The processed features are passed to the deployed classifier, which outputs a probability for each risk '
        'category (Low / Medium / High). The category with the highest probability is shown as the prediction, '
        'and all three probabilities are displayed for transparency.</p>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Crop Disease Risk Classification • Machine Learning Prototype
    <br>
    <span class="footer-stack">Python</span>
    <span class="footer-stack">Scikit-learn</span>
    <span class="footer-stack">Streamlit</span>
</div>
""", unsafe_allow_html=True)