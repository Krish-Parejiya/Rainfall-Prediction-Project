import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("rainfall.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
feature_names = model_data["feature_names"]

st.set_page_config(page_title="Rainfall Predictor", layout="wide")

st.title("🌧 Rainfall Prediction using Machine Learning")
st.write("Enter weather parameters to predict rainfall.")

st.sidebar.header("Input Weather Parameters")

pressure = st.sidebar.slider("Pressure (hPa)", 980.0, 1040.0, 1010.0)
dewpoint = st.sidebar.slider("Dew Point (°C)", -10.0, 35.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 80)
cloud = st.sidebar.slider("Cloud Cover (%)", 0, 100, 50)
sunshine = st.sidebar.slider("Sunshine (hours)", 0.0, 12.0, 5.0)
winddirection = st.sidebar.slider("Wind Direction (°)", 0, 360, 180)
windspeed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 60.0, 10.0)

input_data = [[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]]
input_df = pd.DataFrame(input_data, columns=feature_names)

if st.button("Predict Rainfall"):

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("🌧 Rainfall Expected")
    else:
        st.success("☀ No Rainfall Expected")

    st.write("Rainfall Probability:", round(probability[0][1]*100,2), "%")

    st.subheader("Input Weather Data")

    st.dataframe(input_df)