import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load models and scalers
try:
    model_diabetes = joblib.load("Models/diabetes_model.pkl")
    model_heart = joblib.load("Models/heart_model.pkl")
    model_parkinsons = joblib.load("Models/parkinsons_model.pkl")

    scaler_diabetes = joblib.load("Models/diabetes_scaler.pkl")
    scaler_heart = joblib.load("Models/heart_scaler.pkl")
    scaler_parkinsons = joblib.load("Models/parkinsons_scaler.pkl")

    models_loaded = True
except:
    st.error("Error loading models or scalers. Make sure they are trained and saved properly.")
    models_loaded = False

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
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100)
        insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
        age = st.number_input("Age", min_value=1, max_value=120)

        if st.button("🔮 Predict Diabetes"):
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                      columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

            input_scaled = scaler_diabetes.transform(input_data)  # Apply scaling
            prediction = model_diabetes.predict(input_scaled)
            st.success("✅ Diabetic" if prediction[0] == 1 else "❌ Non-Diabetic")

    elif disease_option == "Heart Disease":
        st.subheader("Heart Disease Prediction")
        
        # User Inputs
        age = st.number_input("Age", min_value=1, max_value=120)
        sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
        cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (0: No, 1: Yes)", [0, 1])
        restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220)
        exang = st.selectbox("Exercise Induced Angina (0: No, 1: Yes)", [0, 1])
        oldpeak = st.number_input("ST Depression (0.0 - 5.0)", min_value=0.0, max_value=5.0)
        slope = st.number_input("Slope of ST Segment (0-2)", min_value=0, max_value=2)
        ca = st.number_input("Major Vessels Colored (0-4)", min_value=0, max_value=4)
        thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3)

        if st.button("🔮 Predict Heart Disease"):
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                      columns=["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG",
                                               "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", "MajorVessels", "Thal"])

            input_scaled = scaler_heart.transform(input_data)  # Apply scaling
            prediction = model_heart.predict(input_scaled)
            st.success("✅ Heart Disease Detected" if prediction[0] == 1 else "❌ No Heart Disease")

    elif disease_option == "Parkinson's":
        st.subheader("Parkinson's Disease Prediction")
        
        # User Inputs
        fo = st.number_input("MDVP:Fo(Hz)", min_value=50.0, max_value=300.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", min_value=50.0, max_value=600.0)
        flo = st.number_input("MDVP:Flo(Hz)", min_value=50.0, max_value=300.0)
        jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=0.1)
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=0.1)
        nhr = st.number_input("NHR", min_value=0.0, max_value=1.0)
        hnr = st.number_input("HNR", min_value=0.0, max_value=40.0)
        rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0)
        dfa = st.number_input("DFA", min_value=0.0, max_value=1.0)
        spread1 = st.number_input("Spread1", min_value=-10.0, max_value=0.0)
        spread2 = st.number_input("Spread2", min_value=0.0, max_value=1.0)
        d2 = st.number_input("D2", min_value=0.0, max_value=3.0)
        ppe = st.number_input("PPE", min_value=0.0, max_value=1.0)

        if st.button("🔮 Predict Parkinson's"):
            input_data = pd.DataFrame([[fo, fhi, flo, jitter, shimmer, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]],
                                      columns=["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer",
                                               "NHR", "HNR", "RPDE", "DFA", "Spread1", "Spread2", "D2", "PPE"])

            input_scaled = scaler_parkinsons.transform(input_data)  # Apply scaling
            prediction = model_parkinsons.predict(input_scaled)
            st.success("✅ Parkinson's Detected" if prediction[0] == 1 else "❌ No Parkinson's")
