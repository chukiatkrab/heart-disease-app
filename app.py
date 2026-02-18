import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ======================
# Load model
# ======================
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction App")

# debug info
# st.write("Model expects:", model.n_features_in_, "features")
# if hasattr(model, "feature_names_in_"):
#     st.write("Feature names:", list(model.feature_names_in_))


# ======================
# Manual Input (8 features only)
# ======================
st.header("Manual Input")

age = st.number_input("Age", 20, 100, value=50)
sex = st.selectbox("Sex (0=female, 1=male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, value=120)
chol = st.number_input("Cholesterol", 100, 600, value=200)
fbs = st.selectbox("Fasting Blood Sugar >120 (0=no, 1=yes)", [0, 1])
restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220, value=150)


if st.button("Predict Manual"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach
    ]])

    prediction = model.predict(input_data)

    # st.write("Prediction result:", prediction)
    # st.write("Prediction value:", prediction[0])

    if prediction[0] == 1 :
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")

    st.write("Prediction value:", prediction[0])
# ======================
# CSV Upload (8 features only)
# ======================
st.header("CSV Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Input Data:")
    st.dataframe(df)

    # select only required 8 features
    required_columns = [
        'age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach'
    ]

    input_data = df[required_columns].values

    prediction = model.predict(input_data)

    df["Prediction"] = prediction

    st.write("Prediction Result:")
    st.dataframe(df)

    st.write("Prediction array:", prediction)
