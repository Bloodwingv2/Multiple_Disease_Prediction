import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load models and scalers safely
models_loaded = False
try:
    model_diabetes = joblib.load("Models/diabetes_model.pkl")
    model_heart = joblib.load("Models/heart_model.pkl")
    model_parkinsons = joblib.load("Models/parkinsons_model.pkl")

    scaler_diabetes = joblib.load("Models/diabetes_scaler.pkl")
    scaler_heart = joblib.load("Models/heart_scaler.pkl")
    scaler_parkinsons = joblib.load("Models/parkinsons_scaler.pkl")

    models_loaded = True
except FileNotFoundError as e:
    st.error(f"Error loading models or scalers: {e}")
except joblib.externals.joblib.exceptions as e:
    st.error(f"Joblib error: {e}")

# Streamlit UI
st.title("🩺 Multiple Disease Prediction")
st.sidebar.title("🔍 Select a Disease")

disease_option = st.sidebar.selectbox(
    "Choose a Disease:",
    ["Diabetes", "Heart Disease", "Parkinson's"]
)

if models_loaded:
    if disease_option == "Diabetes":
        st.subheader("Diabetes Prediction")
        
        # User Inputs
        user_input = {
            "Pregnancies": st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1),
            "Glucose": st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300),
            "BloodPressure": st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=180),
            "SkinThickness": st.number_input("Skin Thickness (mm)", min_value=0, max_value=100),
            "Insulin": st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900),
            "BMI": st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0),
            "DiabetesPedigreeFunction": st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5),
            "Age": st.number_input("Age", min_value=1, max_value=120)
        }
        
        if st.button("🔮 Predict Diabetes"):
            input_df = pd.DataFrame([list(user_input.values())], columns=user_input.keys())
            try:
                input_scaled = scaler_diabetes.transform(input_df)
                prediction = model_diabetes.predict(input_scaled)
                st.success("✅ Diabetic" if prediction[0] == 1 else "❌ Non-Diabetic")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    elif disease_option == "Heart Disease":
        st.subheader("Heart Disease Prediction")
        
        # User Inputs
        user_input = {
            "Age": st.number_input("Age", min_value=1, max_value=120),
            "Sex": st.selectbox("Sex (0: Female, 1: Male)", [0, 1]),
            "ChestPainType": st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3),
            "RestingBP": st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200),
            "Cholesterol": st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600),
            "FastingBS": st.selectbox("Fasting Blood Sugar > 120 mg/dL (0: No, 1: Yes)", [0, 1]),
            "RestingECG": st.number_input("Resting ECG (0-2)", min_value=0, max_value=2),
            "MaxHR": st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220),
            "ExerciseAngina": st.selectbox("Exercise Induced Angina (0: No, 1: Yes)", [0, 1]),
            "Oldpeak": st.number_input("ST Depression (0.0 - 5.0)", min_value=0.0, max_value=5.0),
            "ST_Slope": st.number_input("Slope of ST Segment (0-2)", min_value=0, max_value=2),
            "MajorVessels": st.number_input("Major Vessels Colored (0-4)", min_value=0, max_value=4),
            "Thal": st.number_input("Thalassemia (0-3)", min_value=0, max_value=3)
        }
        
        if st.button("🔮 Predict Heart Disease"):
            input_df = pd.DataFrame([list(user_input.values())], columns=user_input.keys())
            try:
                input_scaled = scaler_heart.transform(input_df)
                prediction = model_heart.predict(input_scaled)
                st.success("✅ Heart Disease Detected" if prediction[0] == 1 else "❌ No Heart Disease")
            except Exception as e:
                st.error(f