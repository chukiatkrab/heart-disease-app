import streamlit as st
import pickle
import pandas as pd

# ======================
# Load model
# ======================
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction App")

# show model features (debug)
st.write("Model expects features:")
st.write(list(model.feature_names_in_))

# ======================
# Manual Input (CORRECT FEATURES)
# ======================
st.header("Manual Input")

chol = st.number_input("Cholesterol", 100, 600, 200)
age = st.number_input("Age", 20, 100, 50)
slope = st.selectbox("Slope (0–2)", [0,1,2])
cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
thal = st.selectbox("Thal (0–3)", [0,1,2,3])
ca = st.selectbox("CA (0–4)", [0,1,2,3,4])
sex = st.selectbox("Sex (0=female, 1=male)", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "chol": chol,
        "age": age,
        "slope": slope,
        "cp": cp,
        "thal": thal,
        "ca": ca,
        "sex": sex,
        "oldpeak": oldpeak
    }])

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.write("Prediction:", prediction[0])
    st.write("Probability:", probability)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")

# ======================
# CSV Upload
# ======================
st.header("CSV Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(df)

    required_columns = [
        "chol",
        "age",
        "slope",
        "cp",
        "thal",
        "ca",
        "sex",
        "oldpeak"
    ]

    input_df = df[required_columns]

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    df["Prediction"] = prediction

    st.write("Prediction Result:")
    st.dataframe(df)

    st.write("Prediction array:", prediction)
    st.write("Prediction probability:", probability)
