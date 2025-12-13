import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_xgboost.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction App")

st.write("Enter your details below to check diabetes risk.")

# Gender
gender = st.selectbox("Gender", ["male", "female", "other"])
gender = 1 if gender == "male" else 0

# Age
age = st.number_input("Age", min_value=1, max_value=120)

# Hypertension
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])

# Heart Disease
heart = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])

# Smoking History
sm = st.selectbox("Smoking History",
                  ["never", "former", "current", "not current", "ever", "no info"])

# Manual encoding
if sm == "never":
    sm = 0
elif sm == "former":
    sm = 1
elif sm == "current":
    sm = 2
elif sm == "not current":
    sm = 3
elif sm == "ever":
    sm = 4
else:
    sm = 5

# BMI
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)

# HbA1c
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0)

# Blood Glucose
glucose = st.number_input("Blood Glucose Level", min_value=50.0, max_value=400.0)

# Predict Button
if st.button("Predict Diabetes"):

    new_data = np.array([[gender, age, hypertension, heart, sm, bmi, hba1c, glucose]])
    
    prob = model.predict_proba(new_data)[0][1]
    threshold = 0.50  # Change this if needed

    pred = 1 if prob >= threshold else 0

    st.subheader("🔍 Prediction Result:")
    
    if pred == 1:
        st.error(f"⚠️ Diabetes Positive (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Diabetes Negative (Probability: {prob:.2f})")
