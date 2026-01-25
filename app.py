import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ============================
# Page config
# ============================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩸",
    layout="centered"
)

# ============================
# Load XGBoost model (NO scaler)
# ============================
model = joblib.load("diabetes_xgboost_shap.joblib")

# ============================
# Header
# ============================
st.title("🩸 Diabetes Risk Prediction")
st.markdown(
    """
    This application estimates **diabetes risk** using an  
    **XGBoost machine learning model** with **explainable AI (SHAP)**.
    """
)

st.divider()

# ============================
# Expected model columns
# ============================
expected_cols = [
    'age',
    'hypertension',
    'heart_disease',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level',
    'gender_Male',
    'gender_Other',
    'smoking_history_current',
    'smoking_history_ever',
    'smoking_history_former',
    'smoking_history_never',
    'smoking_history_not current'
]

# ============================
# Patient Information
# ============================
st.subheader("🧑‍⚕️ Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age (years)", 1, 120, 30)
    bmi = st.number_input("Body Mass Index (BMI)", 10.0, 80.0, 25.0)

with col2:
    hypertension = st.selectbox("Hypertension", [0, 1], help="0 = No, 1 = Yes")
    heart_disease = st.selectbox(
        "Heart Disease", [0, 1], help="0 = No, 1 = Yes")
    smoking = st.selectbox(
        "Smoking History",
        ["never", "current", "not current", "former", "ever"]
    )

st.divider()

# ============================
# Clinical Measurements
# ============================
st.subheader("🧪 Clinical Measurements")

col3, col4 = st.columns(2)

with col3:
    hba1c = st.number_input(
        "HbA1c Level (%)",
        3.0, 15.0, 5.5,
        help="Average blood glucose over the last 2–3 months"
    )

with col4:
    glucose = st.number_input(
        "Blood Glucose Level (mg/dL)",
        50.0, 300.0, 100.0
    )

st.divider()

# ============================
# Input construction
# ============================
input_dict = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose,
    "gender_Male": 1 if gender == "Male" else 0,
    "gender_Other": 1 if gender == "Other" else 0,
    "smoking_history_current": 1 if smoking == "current" else 0,
    "smoking_history_ever": 1 if smoking == "ever" else 0,
    "smoking_history_former": 1 if smoking == "former" else 0,
    "smoking_history_never": 1 if smoking == "never" else 0,
    "smoking_history_not current": 1 if smoking == "not current" else 0,
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=expected_cols, fill_value=0)

# ============================
# Prediction
# ============================
st.subheader("📊 Prediction")

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= 0.25)

    st.divider()
    st.subheader("🧠 Model Assessment")

    # ----------------------------
    # Risk interpretation bands
    # ----------------------------
    if probability < 0.30:
        risk_level = "Low Risk"
        interpretation = "The model estimates a **low likelihood of diabetes**."
    elif probability < 0.60:
        risk_level = "Moderate Risk"
        interpretation = "The model identifies **moderate diabetes risk**."
    else:
        risk_level = "High Risk"
        interpretation = "The model detects a **high likelihood of diabetes**."

    colA, colB = st.columns(2)

    with colA:
        st.metric("Diabetes Risk Probability", f"{probability:.2%}")

    with colB:
        st.metric("Risk Category", risk_level)

    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ Model Prediction: Diabetes Likely")
    else:
        st.success("✅ Model Prediction: Diabetes Unlikely")

    st.write(interpretation)

    # ============================
    # SHAP Explainability
    # ============================
    st.subheader("🔍 Why did the model make this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        ),
        show=False
    )
    st.pyplot(fig)

    with st.expander("🔎 View Model Input Features"):
        st.dataframe(input_df, use_container_width=True)

st.caption(
    "⚠️ This tool is intended for educational and decision-support use only and does not replace medical advice."
)
