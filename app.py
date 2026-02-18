import streamlit as st
import pickle
import numpy as np
import os

# load model
model = os.path.join(os.path.dirname(__file__), "heart_model.pkl")

st.title("Heart Disease Prediction App")

st.write("Enter patient information:")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0, 1])
cp = st.number_input("Chest Pain Type", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 2])
thalach = st.number_input("Max Heart Rate", 60, 220)

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")