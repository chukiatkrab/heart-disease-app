import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# =========================
# LOAD MODEL (FIXED)
# =========================
model_path = os.path.join(os.path.dirname(__file__), "heart_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# =========================
# TITLE
# =========================
st.title("Heart Disease Prediction App")

# =========================
# PART 1: Manual input
# =========================
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

if st.button("Predict"):

    input_data = np.array([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")

# # =========================
# # PART 2: CSV Upload
# # =========================
# st.header("Predict from CSV File")

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file is not None:

#     df = pd.read_csv(uploaded_file)

#     st.write("CSV Preview:")
#     st.dataframe(df)

#     input_data = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']].values

#     prediction = model.predict(input_data)

#     df['Prediction'] = prediction

#     st.write("Prediction Result:")
#     st.dataframe(df)

#     csv = df.to_csv(index=False).encode('utf-8')

#     st.download_button(
#         label="Download Result CSV",
#         data=csv,
#         file_name="heart_prediction_result.csv",
#         mime="text/csv"
#     )