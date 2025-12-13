import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load("diabetes_xgboost.pkl")

st.title("🩺 Diabetes Prediction App")

# ------------ USER INPUT ---------------
gender = st.selectbox("Gender", ["female", "male", "other"])
age = st.number_input("Age", min_value=1, max_value=120)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])

sm = st.selectbox("Smoking History", 
                  ["never", "former", "current", "not current", "ever"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0)
glucose = st.number_input("Blood Glucose Level", min_value=50.0, max_value=400.0)


# ------------ MATCH TRAINING ENCODING ---------------
# Label Encoding for gender (binary)
gender_map = {"female": 0, "male": 1, "other": 2}
gender_encoded = gender_map[gender]

# One-Hot Encoding for smoking_history (drop_first=True logic)
smoking_categories = ["current", "ever", "former", "never", "not current"]
smoking_encoded = [1 if sm == cat else 0 for cat in smoking_categories]

# ------------ CREATE INPUT ROW ---------------
data = {
    "gender": gender_encoded,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose
}

# Convert to DataFrame
input_df = pd.DataFrame([data])

# Add One-Hot columns
for i, cat in enumerate(smoking_categories):
    input_df[f"smoking_history_{cat}"] = smoking_encoded[i]

# ------------ SCALING ---------------
scaler = StandardScaler()
scaled_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

input_df[scaled_cols] = scaler.fit_transform(input_df[scaled_cols])

# ------------ PREDICTION ---------------
if st.button("Predict Diabetes"):
    
    prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if prob >= 0.5 else 0

    if prediction == 1:
        st.error(f"⚠️ Diabetes Positive (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Diabetes Negative (Probability: {prob:.2f})")

