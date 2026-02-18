import streamlit as st
import pickle
import pandas as pd

# ======================
# Load model
# ======================
with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction App")

# แสดง info model (debug)
st.write("Model loaded successfully")
st.write("Model expects features:", list(model.feature_names_in_))

st.write("Model features:", list(model.feature_names_in_))

# ======================
# Manual Input Section
# ======================
st.header("Manual Input")

age = st.number_input("Age", 20, 100, value=50)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain", [0,1,2,3])
chol = st.number_input("Cholesterol", 100, 600, value=200)
ca = st.selectbox("Number of major vessels (ca)", [0,1,2,3])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, value=1.0)
slope = st.selectbox("Slope", [0,1,2])
thal = st.selectbox("Thal", [0,1,2,3])


if st.button("Predict Manual"):

    # ใช้ DataFrame (สำคัญมาก)
    input_df = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'cp': cp,
    'chol': chol,
    'ca': ca,
    'oldpeak': oldpeak,
    'slope': slope,
    'thal': thal
}])


    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.write("Raw prediction:", prediction[0])
    st.write("Probability:", probability)

    if prediction[0] == 1:
        st.success("Heart Disease Detected")
    else:
        st.success("No Heart Disease")


# ======================
# CSV Upload Section
# ======================
st.header("CSV Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Input Data:")
    st.dataframe(df)

    required_columns = [
    'age',
    'sex',
    'cp',
    'chol',
    'ca',
    'oldpeak',
    'slope',
    'thal'
]


    # ตรวจสอบ columns
    if all(col in df.columns for col in required_columns):

        input_df = df[required_columns]

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        df["Prediction"] = prediction

        st.write("Prediction Result:")
        st.dataframe(df)

        st.write("Prediction array:", prediction)
        st.write("Prediction probabilities:", probability)

    else:
        st.error("CSV file must contain required columns:")
        st.write(required_columns)
