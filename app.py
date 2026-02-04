import gradio as gr
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ============================
# Load model
# ============================
model = joblib.load("diabetes_xgboost_shap.joblib")

# Expected columns
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
# Prediction function
# ============================


def predict_diabetes(
    gender, age, bmi, hypertension, heart_disease, smoking,
    hba1c, glucose, show_shap=False
):
    # ----------------------------
    # Prepare input DataFrame
    # ----------------------------
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

    # ----------------------------
    # Make prediction
    # ----------------------------
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= 0.25)

    # ----------------------------
    # Risk interpretation
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

    prediction_text = "Diabetes Likely ⚠️" if prediction == 1 else "Diabetes Unlikely ✅"

    # ----------------------------
    # SHAP plot
    # ----------------------------
    def shap_plot_to_numpy(fig):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

    shap_fig = None
    if show_shap:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            ),
            show=False
        )
        shap_img = shap_plot_to_numpy(fig)
        return shap_img

    return probability, risk_level, prediction_text, interpretation, shap_fig, input_df


# ============================
# Gradio Interface
# ============================
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Dropdown(["Male", "Female", "Other"], label="Gender"),
        gr.Number(label="Age", value=30),
        gr.Number(label="BMI", value=25.0),
        gr.Dropdown([0, 1], label="Hypertension", type="index"),
        gr.Dropdown([0, 1], label="Heart Disease", type="index"),
        gr.Dropdown(["never", "current", "not current",
                    "former", "ever"], label="Smoking History"),
        gr.Number(label="HbA1c Level (%)", value=5.5),
        gr.Number(label="Blood Glucose Level (mg/dL)", value=100.0),
        gr.Checkbox(label="Show SHAP Explainability", value=False)
    ],
    outputs=[
        # probability
        gr.Label(num_top_classes=1, label="Diabetes Risk Probability"),
        # Low/Moderate/High Risk
        gr.Textbox(label="Risk Category"),
        # Diabetes Likely/Unlikely
        gr.Textbox(label="Prediction"),
        # Text interpretation
        gr.Textbox(label="Interpretation"),
        # SHAP plot
        gr.Image(type="numpy", label="SHAP Waterfall Plot"),
        # input DataFrame
        gr.Dataframe(
    headers=[
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender_Male",
        "gender_Other",
        "smoking_history_current",
        "smoking_history_ever",
        "smoking_history_former",
        "smoking_history_never",
        "smoking_history_not current",
    ],
    label="Input Features",
)

    ],
    title="🩸 Diabetes Risk Prediction",
    description="Predicts diabetes risk using an XGBoost model with SHAP explainability.",
    live=False
)

# ============================
# Launch app
# ============================
iface.launch(ssr_mode=False)


