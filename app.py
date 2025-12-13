import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("diabetes_xgboost.pkl")

st.title("🩺 Diabetes Prediction App")

# ------------ USER INPUT ---------------
gender = st.selectbox("Gender", ["female", "male", "other"])
age = st.number_input("Age", min_value=1, max_value=120)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])

sm = st.selectbox("Smoking History",
                  ["never", "former", "current", "not current", "ever", "no info"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0)
glucose = st.number_input("Blood Glucose Level", min_value=50.0, max_value=400.0)

# ------------ MATCH TRAINING ENCODING ---------------

# Gender encoding → creates: gender_male, gender_other (female is baseline)
gender_male = 1 if gender == "male" else 0
gender_other = 1 if gender == "other" else 0

# Smoking history One-Hot (matching your training columns)
smoking_history_ever = 1 if sm == "ever" else 0
smoking_history_former = 1 if sm == "former" else 0
smoking_history_never = 1 if sm == "never" else 0
smoking_history_no_info = 1 if sm == "no info" else 0
smoking_history_not_current = 1 if sm == "not current" else 0

# ------------ CREATE INPUT ROW IN EXACT MODEL ORDER ---------------

input_df = pd.DataFrame([[
    age,
    hypertension,
    heart,
    bmi,
    hba1c,
    glucose,
    gender_male,
    gender_other,
    smoking_history_ever,
    smoking_history_former,
    smoking_history_never,
    smoking_history_no_info,
    smoking_history_not_current
]], columns=[
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'gender_male', 'gender_other',
    'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
    'smoking_history_no info', 'smoking_history_not current'
])

# ------------ SCALE NUMERICAL FEATURES ---------------
scaler = StandardScaler()
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
input_df[num_cols] = scaler.fit_transform(input_df[num_cols])

# ------------ PREDICTION ---------------
if st.button("Predict Diabetes"):

    prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if prob >= 0.50 else 0

    if prediction == 1:
        st.error(f"⚠️ Diabetes Positive (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Diabetes Negative (Probability: {prob:.2f})")
