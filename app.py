import streamlit as st
import pickle
import numpy as np
import pandas as pd

# load model
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction App")

# ======================
# Manual Input
# ======================
st.header("Manual Input")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0, 1])
cp = st.number_input("Chest Pain Type", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 2])
thalach = st.number_input("Max Heart Rate", 60, 220)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("CA", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict Manual"):

    input_data = np.array([[
    age, sex, cp, trestbps, chol,
    fbs, restecg, thalach,
    exang, oldpeak, slope, ca, thal
]])


    prediction = model.predict(input_data)

    st.write("Prediction result:", prediction)
    st.write("Prediction value:", prediction[0])

    # optional meaning
    if prediction[0] == 1:
        st.write("Meaning: Heart Disease")
    else:
        st.write("Meaning: No Heart Disease")


st.write("Expected features:", model.n_features_in_)
# st.write("Actual features:", input_data.shape[1])

# ======================
# CSV Upload
# ======================
st.header("CSV Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Input Data:")
    st.dataframe(df)

    input_data = df[[
        'age','sex','cp','trestbps','chol',
        'fbs','restecg','thalach',
        'exang','oldpeak','slope','ca','thal'
    ]].values

    prediction = model.predict(input_data)

    st.write("Prediction array:")
    st.write(prediction)

    df["Prediction"] = prediction

    st.write("Result:")
    st.dataframe(df)
