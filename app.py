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

healthy_man = {
    "pregnancies": 1,
    "glucose": 85,
    "blood_pressure": 70,
    "skin_thickness": 20,
    "insulin": 80,
    "bmi": 24.0,
    "dpf": 0.5,
    "age": 30
}

diabetic_man = {
    "pregnancies": 5,
    "glucose": 150,
    "blood_pressure": 85,
    "skin_thickness": 35,
    "insulin": 200,
    "bmi": 32.0,
    "dpf": 1.2,
    "age": 50
}

healthy_heart = {
    "age": 30,            # Age - Generally younger individuals are at lower risk
    "sex": 1,             # Sex - Male (can vary depending on your dataset)
    "cp": 0,              # Chest Pain Type - 0 (No typical angina)
    "trestbps": 120,      # Resting Blood Pressure - Around 120 mm Hg (normal range)
    "chol": 180,          # Cholesterol - Around 180 mg/dL (healthy range)
    "fbs": 0,             # Fasting Blood Sugar > 120 mg/dL - 0 (Healthy fasting blood sugar)
    "restecg": 0,         # Resting ECG - 0 (Normal ECG)
    "thalach": 170,       # Max Heart Rate Achieved - High capacity for physical effort
    "exang": 0,           # Exercise Induced Angina - 0 (No angina during exercise)
    "oldpeak": 0.0,       # ST Depression - 0.0 (No depression)
    "slope": 2,           # Slope of ST Segment - 2 (Upsloping, considered healthier)
    "ca": 0,              # Major Vessels Colored - 0 (No major vessel blockages)
    "thal": 2             # Thalassemia - 2 (Normal blood flow in stress test)
}

unhealthy_heart = {
    "age": 65,            # Older age - Higher risk group
    "sex": 1,             # Sex - Male (slightly higher risk statistically)
    "cp": 2,              # Chest Pain Type - 2 (Atypical angina or discomfort)
    "trestbps": 150,      # Resting Blood Pressure - Elevated (Hypertension Stage 1)
    "chol": 280,          # Cholesterol - High cholesterol level
    "fbs": 1,             # Fasting Blood Sugar > 120 mg/dL - Elevated
    "restecg": 1,         # Resting ECG - Possible signs of abnormal heart function
    "thalach": 120,       # Max Heart Rate Achieved - Lower than average due to poor heart function
    "exang": 1,           # Exercise Induced Angina - Chest pain triggered by exertion
    "oldpeak": 2.5,       # ST Depression - Moderate elevation indicating ischemia
    "slope": 1,           # Slope of ST Segment - Flat slope (common in heart disease)
    "ca": 2,              # Major Vessels Colored - 2 (Indicating partial blockages)
    "thal": 1             # Thalassemia - 1 (Fixed defect, common in heart conditions)
}

# Initialize session state for inputs
if "inputs" not in st.session_state:
    st.session_state.inputs = healthy_man.copy()

if "inputs2" not in st.session_state:
    st.session_state.inputs2 = healthy_heart.copy()

if models_loaded:
    if disease_option == "Diabetes":
        st.subheader("Diabetes Prediction")

        # Button to autofill default values
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Fill Healthy Values"):
                st.session_state.inputs = healthy_man.copy()
                st.rerun()  # Refresh UI

        with col2:
            if st.button("⚠️ Fill Diabetic Values"):
                st.session_state.inputs = diabetic_man.copy()
                st.rerun()  # Refresh UI

        
                # Input fields with default values
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1,
                                    value=st.session_state.inputs["pregnancies"])
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200,
                                value=st.session_state.inputs["glucose"])
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150,
                                        value=st.session_state.inputs["blood_pressure"])
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100,
                                        value=st.session_state.inputs["skin_thickness"])
        insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900,
                                value=st.session_state.inputs["insulin"])
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0,
                            value=st.session_state.inputs["bmi"])
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5,
                            value=st.session_state.inputs["dpf"])
        age = st.number_input("Age", min_value=1, max_value=120,
                            value=st.session_state.inputs["age"])

        if st.button("🔮 Predict Diabetes"):
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                      columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

            input_scaled = scaler_diabetes.transform(input_data)  # Apply scaling
            prediction = model_diabetes.predict(input_scaled)
            st.success("✅ Diabetic" if prediction[0] == 1 else "❌ Non-Diabetic")

    elif disease_option == "Heart Disease":
        st.subheader("Heart Disease Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Fill Healthy Values"):
                st.session_state.inputs2 = healthy_heart.copy()
                st.rerun()  # Refresh UI

        with col2:
            if st.button("⚠️ Fill Heart-Risk Values"):
                st.session_state.inputs2 = unhealthy_heart.copy()
                st.rerun()  # Refresh UI

        # User Inputs for Heart Disease
        age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.inputs2["age"])
        sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1], index=[0, 1].index(st.session_state.inputs2["sex"]))
        cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=st.session_state.inputs2["cp"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=st.session_state.inputs2["trestbps"])
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=st.session_state.inputs2["chol"])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (0: No, 1: Yes)", [0, 1], index=[0, 1].index(st.session_state.inputs2["fbs"]))
        restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=st.session_state.inputs2["restecg"])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=st.session_state.inputs2["thalach"])
        exang = st.selectbox("Exercise Induced Angina (0: No, 1: Yes)", [0, 1], index=[0, 1].index(st.session_state.inputs2["exang"]))
        oldpeak = st.number_input("ST Depression (0.0 - 5.0)", min_value=0.0, max_value=5.0, value=st.session_state.inputs2["oldpeak"])
        slope = st.number_input("Slope of ST Segment (0-2)", min_value=0, max_value=2, value=st.session_state.inputs2["slope"])
        ca = st.number_input("Major Vessels Colored (0-4)", min_value=0, max_value=4, value=st.session_state.inputs2["ca"])
        thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, value=st.session_state.inputs2["thal"])

        if st.button("🔮 Predict Heart Disease"):
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                      columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

            input_scaled = scaler_heart.transform(input_data)  # Apply scaling
            prediction = model_heart.predict(input_scaled)
            st.success("✅ Heart Disease Detected" if prediction[0] == 1 else "❌ No Heart Disease")

    elif disease_option == "Parkinson's":
        st.subheader("Parkinson's Disease Prediction")

        # User Inputs for Parkinson's
        fo = st.number_input("MDVP:Fo(Hz)", min_value=50.0, max_value=300.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", min_value=50.0, max_value=600.0)
        flo = st.number_input("MDVP:Flo(Hz)", min_value=50.0, max_value=300.0)
        jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=0.1)
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.1)
        rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=1.0)
        ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=1.0)
        ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=1.0)
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=0.1)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=0.1)
        apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=1.0)
        apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=1.0)
        apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=1.0)
        dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=1.0)
        nhr = st.number_input("NHR", min_value=0.0, max_value=1.0)
        hnr = st.number_input("HNR", min_value=0.0, max_value=40.0)
        rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0)
        dfa = st.number_input("DFA", min_value=0.0, max_value=1.0)
        spread1 = st.number_input("spread1", min_value=-10.0, max_value=0.0)
        spread2 = st.number_input("spread2", min_value=0.0, max_value=1.0)
        d2 = st.number_input("D2", min_value=0.0, max_value=3.0)
        ppe = st.number_input("PPE", min_value=0.0, max_value=1.0)

        if st.button("🔮 Predict Parkinson's"):
            # Create DataFrame for user input
            input_data = pd.DataFrame([[fo, fhi, flo, jitter, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]],
                                      columns=["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", 
                                               "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", 
                                               "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", 
                                               "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"])

            # Apply scaling
            input_scaled = scaler_parkinsons.transform(input_data)

            # Make the prediction
            prediction = model_parkinsons.predict(input_scaled)

            # Show prediction result
            st.success("✅ Parkinson's Detected" if prediction[0] == 1 else "❌ No Parkinson's")
