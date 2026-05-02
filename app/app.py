import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os

# Load artifacts
@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "heart_model.pkl"))
    preprocessor = joblib.load(os.path.join(base_dir, "preprocessor.pkl"))
    with open(os.path.join(base_dir, "feature_names.pkl"), "rb") as f:
        feature_names = joblib.load(f)
    with open(os.path.join(base_dir, "sample_patient.json"), "r") as f:
        sample_patient = json.load(f)
    return model, preprocessor, feature_names, sample_patient

model, preprocessor, feature_names, sample_patient = load_resources()

st.title("Assignment #4")
st.markdown("""
This dashboard provides a decision-support tool for assessing the likelihood of heart disease in patients based on 13 clinical features. 
""")

st.header("Patient Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (20-80)", min_value=20, max_value=100, value=int(sample_patient.get("age", 50)))
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", options=[0, 1], index=int(sample_patient.get("sex", 1)))
    cp = st.selectbox("Chest Pain Type (1-4)", options=[1, 2, 3, 4], index=int(sample_patient.get("cp", 3))-1)
    trestbps = st.number_input("Resting BP (80-220 mmHg)", min_value=50, max_value=250, value=int(sample_patient.get("trestbps", 120)))
    chol = st.number_input("Cholesterol (100-600 mg/dl)", min_value=100, max_value=600, value=int(sample_patient.get("chol", 200)))

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 (0 = False, 1 = True)", options=[0, 1], index=int(sample_patient.get("fbs", 0)))
    restecg = st.selectbox("Resting ECG (0-2)", options=[0, 1, 2], index=int(sample_patient.get("restecg", 0)))
    thalach = st.number_input("Max Heart Rate (70-210)", min_value=50, max_value=250, value=int(sample_patient.get("thalach", 150)))
    exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", options=[0, 1], index=int(sample_patient.get("exang", 0)))

with col3:
    oldpeak = st.number_input("ST Depression (0.0 - 6.0)", min_value=0.0, max_value=10.0, value=float(sample_patient.get("oldpeak", 1.0)), step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (1-3)", options=[1, 2, 3], index=int(sample_patient.get("slope", 2))-1)
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3], index=int(sample_patient.get("ca", 0)))
    thal = st.selectbox("Thalassemia (3=Normal, 6=Fixed, 7=Reversable)", options=[3, 6, 7], index=[3, 6, 7].index(int(sample_patient.get("thal", 3))))

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    processed_input = preprocessor.transform(input_data)
    
    y_pred = model.predict(processed_input)[0]
    y_proba = model.predict_proba(processed_input)[0]
    confidence = y_proba[1] if y_pred == 1 else y_proba[0]

    st.header("Results Panel")
    
    if y_pred == 1:
        st.error(f"⚠️ Predicted Class: Disease Present")
    else:
        st.success(f"✅ Predicted Class: No Disease")

    st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")

    # Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_input)
    # Binary classification SHAP returns list for RF in older versions, 3D array in newer versions
    if isinstance(shap_values, list):
        pat_shap = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        pat_shap = shap_values[0, :, 1]
    else:
        pat_shap = shap_values[0]

    top_idx = np.argsort(np.abs(pat_shap))[-3:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_vals = [pat_shap[i] for i in top_idx]

    st.subheader("Top 3 Driving Features")
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['red' if v > 0 else 'blue' for v in top_vals]
    ax.barh(top_features, top_vals, color=colors)
    ax.set_xlabel("SHAP Value (Impact on prediction)")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Nurse-readable Explanation")
    
    dir_1 = "increased" if top_vals[0] > 0 else "decreased"
    dir_2 = "increased" if top_vals[1] > 0 else "decreased"
    
    explanation = f"The model's prediction was most strongly driven by the patient's {top_features[0]} (which {dir_1} the risk score) and {top_features[1]} (which {dir_2} the risk score). "
    explanation += f"These signals suggest the patient {'may need closer cardiac review' if y_pred == 1 else 'appears stable based on these specific markers'}, but the output is decision support rather than a diagnosis."
    
    st.info(explanation)
