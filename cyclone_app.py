import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load(r"D:\For papers\Third paper\python\best_pipeline.pkl")

st.set_page_config(page_title="Standard tangential-inlet Lapple Cyclone Efficiency Predictor", layout="centered")
st.title("🌪️Standard tangential-inlet Lapple Cyclone Efficiency Estimator")
st.markdown("Predict cyclone separation efficiency for spherical particles based on physical properties.")

# --- Input fields ---
phi = st.number_input("1️⃣ Enter particle shape Φ (sphericity): ** Range `0.5 ≤ Φ ≤ 1`", min_value=0.5, max_value=1.0, value=0.9)
rho_s = st.number_input("2️⃣ Enter particle density ρₛ (kg/m³):** Range `700 ≤ ρₛ ≤ 3320`", min_value=700.0, max_value=3320.0, value=1500.0)
d_h = st.number_input("3️⃣ Enter cyclone hydraulic diameter dₕ (mm)", min_value=1.0, max_value=500.0, value=50.0)
d_p = st.number_input("4️⃣ Enter particle diameter dₚ (μm) :** Range `0.1 ≤ dₚ ≤ 10`", min_value=0.1, max_value=10.0, value=1.0)
alpha_p = st.number_input("5️⃣ Enter volume fraction αₚ:** Range `1e-6 ≤ αₚ ≤ 1e-4", min_value=1e-6, max_value=1e-3, format="%.6f", value=1e-5)

# --- Transform inputs to model features ---
X = phi  # x
R = rho_s / 1.225  # density ratio
H = d_p / (d_h * 1000)  # dimensionless particle size
alpha = alpha_p  # direct mapping

# Apply log transforms
log_H = np.log1p(H)
log_R = np.log1p(R)
log_alpha = np.log1p(alpha)

# Construct model input
input_df = pd.DataFrame([[X, log_H, log_R, log_alpha]], columns=['x', 'log_H', 'log_R', 'log_alpha'])

# Predict
y_pred = model.predict(input_df)[0]
y_pred = np.expm1(y_pred)
y_pred = np.clip(y_pred, 0, 1)

# --- Output ---
st.subheader("✅ Prediction Result")
st.success(f"Your cyclone efficiency is equal to **{y_pred:.5f}** for your best spherical particle.")

st.caption("Model output is bounded to [0, 1] to reflect physical limits.")
