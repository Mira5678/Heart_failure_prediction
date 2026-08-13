import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    #user inputs
    age = st.number_input("Age(years)", min_value=0, max_value=100)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting Blodd Pressure (mm Hg)", min_value=0, max_value=300)
    cholesteral = st.number_input("Cholesteral (mm/dl)", min_value=0, max_value=1000)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", min_value=0, max_value=10)
    st_slope = st.selectbox("Slope of Peak Exercise ST segment", ["Up", "Flat", "Down"])

    # Keep the original values used during model training. The saved models are
    # pipelines that include OneHotEncoder and expect the raw labels such as M/F,
    # Y/N, and the actual category names, not integer encodings.
    sex_value = "M" if sex == "Male" else "F"
    fasting_bs_value = 0 if fasting_bs == "<= 120 mg/dl" else 1
    exercise_angina_value = "Y" if exercise_angina == "Yes" else "N"

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_value],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesteral],
        "FastingBS": [fasting_bs_value],
        "RestingECG": [resting_ECG],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina_value],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    algonames = ["Logistic Regression", "SVM", "Gradient Boosting"]
    filenames = ["logistic_regression.pkl", "svm.pkl", "gbc.pkl"]

    #make predictions for each model based on user inputs
    def predict_heart_disease(data):
        predictions = []
        for filename in filenames:
            with open(filename, 'rb') as model_file:
                model = pickle.load(model_file)
            prediction = model.predict(data)[0]
            predictions.append(prediction)
        return predictions

    #create a submit button to make predictions
    if st.button("Submit"):
        st.subheader("Results")
        st.markdown("-------------------")

        result = predict_heart_disease(input_data)

        for i, model_name in enumerate(algonames):
            st.subheader(model_name)
            if result[i] == 0:
                st.write("No heart disease detected")
            else:
                st.write("Heart disease detected")

