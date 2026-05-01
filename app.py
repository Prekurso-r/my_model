import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'figure.facecolor': '#0f1923',
    'axes.facecolor': '#0f1923',
    'text.color': '#e0f2f1',
    'axes.labelcolor': '#e0f2f1',
    'xtick.color': '#e0f2f1',
    'ytick.color': '#e0f2f1',
})

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetIQ · Risk Assessment",
    page_icon="🩸",
    layout="centered"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0b1622;
    --surface:   #111f2e;
    --surface2:  #162538;
    --border:    rgba(0,212,170,0.15);
    --accent:    #00d4aa;
    --accent2:   #0af0c8;
    --accent-dim:#00a882;
    --danger:    #ff4f6e;
    --warn:      #ffb830;
    --text:      #d6eae6;
    --muted:     #7a9ea8;
    --radius:    14px;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2rem 4rem !important; max-width: 780px; }

/* ── Animated background dots ── */
body::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        radial-gradient(circle at 15% 20%, rgba(0,212,170,0.06) 0%, transparent 50%),
        radial-gradient(circle at 85% 75%, rgba(10,240,200,0.05) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 1.5rem 2rem;
    position: relative;
}
.hero-logo {
    display: inline-flex;
    align-items: center; justify-content: center;
    width: 70px; height: 70px;
    border-radius: 20px;
    background: linear-gradient(135deg, var(--accent), #008f74);
    font-size: 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 0 40px rgba(0,212,170,0.4);
    animation: pulse-glow 3s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 30px rgba(0,212,170,0.3); }
    50%      { box-shadow: 0 0 55px rgba(0,212,170,0.6); }
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem; font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--accent2) 0%, #b8fff0 60%, var(--text) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem;
}
.hero p {
    color: var(--muted); font-size: 1rem; font-weight: 300;
    max-width: 480px; margin: 0 auto; line-height: 1.65;
}

/* ── Section headers ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--accent);
    margin: 0 0 0.8rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem; font-weight: 700;
    color: var(--text); margin: 0 0 1.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
    transition: border-color .25s;
}
.card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.5;
}
.card:hover { border-color: rgba(0,212,170,0.35); }

/* ── Divider ── */
.my-divider {
    border: none; border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Streamlit inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stSelectbox label, .stNumberInput label,
div[data-testid="stWidgetLabel"] > p {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em;
}
input[type="number"] {
    background: transparent !important;
    color: var(--text) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #008f74) !important;
    color: #001a14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    transition: all .25s !important;
    box-shadow: 0 4px 24px rgba(0,212,170,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,212,170,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result cards ── */
.result-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 1rem; margin: 1.2rem 0;
}
.result-metric {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.result-metric .label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem;
}
.result-metric .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
}
.result-metric .value.accent { color: var(--accent); }
.result-metric .value.warn    { color: var(--warn); }
.result-metric .value.danger  { color: var(--danger); }

/* ── Risk badge ── */
.risk-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.6rem 1.2rem; border-radius: 100px;
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem; font-weight: 700; letter-spacing: 0.04em;
    margin: 1rem 0;
}
.risk-badge.low    { background: rgba(0,212,170,0.12); color: var(--accent); border: 1px solid rgba(0,212,170,0.3); }
.risk-badge.medium { background: rgba(255,184,48,0.12); color: var(--warn);   border: 1px solid rgba(255,184,48,0.3); }
.risk-badge.high   { background: rgba(255,79,110,0.12); color: var(--danger); border: 1px solid rgba(255,79,110,0.3); }

/* ── Progress bar ── */
.prog-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 100px; height: 10px; overflow: hidden;
    margin: 0.6rem 0 1.4rem;
}
.prog-bar {
    height: 100%; border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1);
}
.prog-bar.low    { background: linear-gradient(90deg, #00a882, var(--accent)); }
.prog-bar.medium { background: linear-gradient(90deg, #e08000, var(--warn)); }
.prog-bar.high   { background: linear-gradient(90deg, #c0002a, var(--danger)); }

/* ── Interpretation box ── */
.interp-box {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    font-size: 0.92rem; line-height: 1.65;
    color: var(--text); margin-top: 1rem;
}
.interp-box.warn   { border-color: var(--warn); }
.interp-box.danger { border-color: var(--danger); }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-size: 0.84rem !important;
}
details[open] .streamlit-expanderHeader { border-radius: 10px 10px 0 0 !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Caption ── */
.footer-note {
    text-align: center; color: var(--muted);
    font-size: 0.76rem; margin-top: 3rem; letter-spacing: 0.02em;
}

/* ── Tooltip help text ── */
.stTooltipIcon { color: var(--accent) !important; }

/* ── Streamlit metric ── */
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-logo">🩸</div>
    <h1>Diabetes Risk Prediction AI</h1>
    <p>AI-powered diabetes risk assessment using XGBoost & explainable SHAP analysis. Developed by Kirubel Muluken
    Fill in the patient profile below to receive an instant prediction.</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load("diabetes_xgboost_shap.joblib")

expected_cols = [
    'age', 'hypertension', 'heart_disease', 'bmi',
    'HbA1c_level', 'blood_glucose_level',
    'gender_Male', 'gender_Other',
    'smoking_history_current', 'smoking_history_ever',
    'smoking_history_former', 'smoking_history_never',
    'smoking_history_not current'
]

# ── Section 1 – Patient profile ───────────────────────────────────────────────
st.markdown('<p class="section-label">Step 1</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">🧑‍⚕️ Patient Profile</p>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age    = st.number_input("Age (years)", 1, 120, 30)
    bmi    = st.number_input("Body Mass Index (BMI)", 10.0, 80.0, 25.0, step=0.1)
with col2:
    hypertension  = st.selectbox("Hypertension",  [0, 1], format_func=lambda x: "Yes" if x else "No")
    heart_disease = st.selectbox("Heart Disease",  [0, 1], format_func=lambda x: "Yes" if x else "No")
    smoking       = st.selectbox(
        "Smoking History",
        ["never", "current", "not current", "former", "ever"],
        help="Select the patient's current or most recent smoking status"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2 – Clinical values ───────────────────────────────────────────────
st.markdown('<p class="section-label">Step 2</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">🧪 Clinical Measurements</p>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    hba1c = st.number_input(
        "HbA1c Level (%)", 3.0, 15.0, 5.5, step=0.1,
        help="Reflects average blood glucose over the past 2–3 months. Normal < 5.7%"
    )
with col4:
    glucose = st.number_input(
        "Blood Glucose Level (mg/dL)", 50.0, 300.0, 100.0, step=1.0,
        help="Fasting blood glucose. Normal < 100 mg/dL"
    )

# Live reference mini-guide
st.markdown(f"""
<div style="display:flex; gap:0.8rem; margin-top:1rem; flex-wrap:wrap;">
    <div style="flex:1; min-width:140px; background:rgba(0,212,170,0.08);
         border:1px solid rgba(0,212,170,0.2); border-radius:10px; padding:0.7rem 1rem;">
        <div style="font-size:0.7rem; letter-spacing:.1em; text-transform:uppercase;
             color:var(--muted); margin-bottom:0.25rem;">HbA1c entered</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
             color:{'#ff4f6e' if hba1c>=6.5 else '#ffb830' if hba1c>=5.7 else '#00d4aa'};">
             {hba1c:.1f}%</div>
        <div style="font-size:0.72rem; color:var(--muted); margin-top:0.15rem;">
             {'Diabetic range' if hba1c>=6.5 else 'Pre-diabetic range' if hba1c>=5.7 else 'Normal range'}</div>
    </div>
    <div style="flex:1; min-width:140px; background:rgba(0,212,170,0.08);
         border:1px solid rgba(0,212,170,0.2); border-radius:10px; padding:0.7rem 1rem;">
        <div style="font-size:0.7rem; letter-spacing:.1em; text-transform:uppercase;
             color:var(--muted); margin-bottom:0.25rem;">Glucose entered</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
             color:{'#ff4f6e' if glucose>=200 else '#ffb830' if glucose>=100 else '#00d4aa'};">
             {glucose:.0f} mg/dL</div>
        <div style="font-size:0.72rem; color:var(--muted); margin-top:0.15rem;">
             {'High glucose' if glucose>=200 else 'Elevated glucose' if glucose>=100 else 'Normal glucose'}</div>
    </div>
    <div style="flex:1; min-width:140px; background:rgba(0,212,170,0.08);
         border:1px solid rgba(0,212,170,0.2); border-radius:10px; padding:0.7rem 1rem;">
        <div style="font-size:0.7rem; letter-spacing:.1em; text-transform:uppercase;
             color:var(--muted); margin-bottom:0.25rem;">BMI entered</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
             color:{'#ff4f6e' if bmi>=30 else '#ffb830' if bmi>=25 else '#00d4aa'};">
             {bmi:.1f}</div>
        <div style="font-size:0.72rem; color:var(--muted); margin-top:0.15rem;">
             {'Obese' if bmi>=30 else 'Overweight' if bmi>=25 else 'Normal weight'}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Build input df ────────────────────────────────────────────────────────────
input_dict = {
    "age": age, "hypertension": hypertension,
    "heart_disease": heart_disease, "bmi": bmi,
    "HbA1c_level": hba1c, "blood_glucose_level": glucose,
    "gender_Male":  1 if gender == "Male"  else 0,
    "gender_Other": 1 if gender == "Other" else 0,
    "smoking_history_current":     1 if smoking == "current"     else 0,
    "smoking_history_ever":        1 if smoking == "ever"        else 0,
    "smoking_history_former":      1 if smoking == "former"      else 0,
    "smoking_history_never":       1 if smoking == "never"       else 0,
    "smoking_history_not current": 1 if smoking == "not current" else 0,
}
input_df = pd.DataFrame([input_dict]).reindex(columns=expected_cols, fill_value=0)

# ── Predict button ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Step 3</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">🔍 Run Assessment</p>', unsafe_allow_html=True)

predict = st.button("⚡ Analyse Diabetes Risk", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if predict:
    probability = model.predict_proba(input_df)[0][1]
    pct         = probability * 100

    if pct < 30:
        risk_key   = "low"
        risk_label = "Low Risk"
        badge_icon = "✅"
        interp = ("The model estimates a <strong>low likelihood of diabetes</strong>. "
                  "Key markers are within acceptable ranges. Routine monitoring is still recommended.")
    elif pct < 60:
        risk_key   = "medium"
        risk_label = "Moderate Risk"
        badge_icon = "⚠️"
        interp = ("The model identifies <strong>moderate diabetes risk</strong>. "
                  "One or more clinical markers are elevated. Consider lifestyle review and follow-up testing.")
    else:
        risk_key   = "high"
        risk_label = "High Risk"
        badge_icon = "🚨"
        interp = ("The model detects a <strong>high likelihood of diabetes</strong>. "
                  "Multiple risk factors are present. Immediate clinical evaluation is strongly advised.")

    st.markdown(f"""
    <div class="card" style="margin-top:0.5rem;">
        <p class="section-label">Assessment Result</p>
        <div class="result-grid">
            <div class="result-metric">
                <div class="label">Risk Probability</div>
                <div class="value {'accent' if risk_key=='low' else 'warn' if risk_key=='medium' else 'danger'}">
                    {pct:.1f}%
                </div>
            </div>
            <div class="result-metric">
                <div class="label">Risk Category</div>
                <div class="value {'accent' if risk_key=='low' else 'warn' if risk_key=='medium' else 'danger'}">
                    {risk_label}
                </div>
            </div>
        </div>

        <div class="prog-wrap">
            <div class="prog-bar {risk_key}" style="width:{min(pct,100):.1f}%;"></div>
        </div>

        <span class="risk-badge {risk_key}">{badge_icon} {risk_label}</span>

        <div class="interp-box {'warn' if risk_key=='medium' else 'danger' if risk_key=='high' else ''}">
            {interp}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:2rem;">Explainability</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔬 Why did the model decide this?</p>', unsafe_allow_html=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1923')
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        ),
        show=False
    )
    ax = plt.gca()
    ax.set_facecolor('#0f1923')
    plt.gcf().set_facecolor('#0f1923')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a4a')
    ax.tick_params(colors='#7a9ea8')
    ax.xaxis.label.set_color('#7a9ea8')

    st.pyplot(fig)
    plt.close(fig)

    # ── Raw features expander ─────────────────────────────────────────────────
    with st.expander("🗂️ View raw model input features"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    ⚠️ DiabetIQ is intended for educational and clinical decision-support use only —
    it does not replace professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)
