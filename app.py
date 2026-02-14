import streamlit as st
import joblib
import numpy as np
import os

MODEL_FILE = "weather_model.pkl"

st.set_page_config(page_title="Tomorrow Weather Predictor", page_icon="🌦")

# Modern Gradient Background
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.main {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
}
.result-card {
    background:black;
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🌦 Tomorrow Weather Predictor")

temperature = st.number_input("🌡 Current Temperature (°C)", value=30.0)
humidity = st.number_input("💧 Humidity (%)", value=70.0)
pressure = st.number_input("🌬 Pressure (hPa)", value=1010.0)

if st.button("🚀 Predict Tomorrow Weather"):

    if not os.path.exists(MODEL_FILE):
        st.error("Model not found! Run model_train.py first.")
    else:
        temp_model, rain_model = joblib.load(MODEL_FILE)

        features = np.array([[temperature, humidity, pressure]])

        tomorrow_temp = temp_model.predict(features)[0]
        rain_prediction = rain_model.predict(features)[0]

        weather_status = "🌧 Rainy" if rain_prediction == 1 else "☀️ Not Rainy"

        st.markdown(f"""
        <div class="result-card">
            <h2>📊 Tomorrow Forecast</h2>
            <h3>🌡 Temperature: {tomorrow_temp:.1f} °C</h3>
            <h3>{weather_status}</h3>
        </div>
        """, unsafe_allow_html=True)
