import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "best_pipeline.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Standard-tangential-inlet-Lapple-Cyclone-Efficiency-Estimator", layout="centered")
st.title("🌪️ Standard-tangential-inlet-Lapple-Cyclone-Efficiency-Estimator-Based-on-Shape-Factor")

# Preface/instructions
st.markdown(":bulb: **Instructions**")
Enter your particle characteristics to estimate the separation efficiency of a cyclone.

All calculations where estimated based on cyclone main diameter **Dc = 200 mm**
# --- Inputs ---
st.markdown("**1️⃣ Φ (sphericity):** Range `0.5 ≤ Φ ≤ 1`")
phi = st.number_input("Particle Shape Φ", min_value=0.5, max_value=1.0, value=0.9)

st.markdown("**2️⃣ ρₛ (particle density kg/m³):** Range `700 ≤ ρₛ ≤ 3320`")
rho_s = st.number_input("Particle Density ρₛ", min_value=700.0, max_value=3320.0, value=1500.0)

st.markdown("**3️⃣ dₚ (particle diameter μm):** Range `0.1 ≤ dₚ ≤ 10`")
d_p = st.number_input("Particle Diameter dₚ", min_value=0.1, max_value=10.0, value=1.0)

st.markdown("**4️⃣ αₚ (volume fraction):** Range `1e-6 ≤ αₚ ≤ 1e-3`")
alpha_p = st.number_input("Volume Fraction αₚ", min_value=1e-6, max_value=1e-3, format="%.6f", value=1e-5)

# --- Transform inputs ---
X = phi
R = rho_s / 1.225  # Dimensionless density ratio
d_h = 200.0  # Fixed hydraulic diameter in mm
H = d_p / (d_h * 1000)  # Convert mm → m, then dimensionless size
alpha = alpha_p

log_H = np.log1p(H)
log_R = np.log1p(R)
log_alpha = np.log1p(alpha)

X_input = pd.DataFrame([[X, log_H, log_R, log_alpha]], columns=['x', 'log_H', 'log_R', 'log_alpha'])

# --- Predict ---
y_pred = model.predict(X_input)[0]
y_pred = np.expm1(y_pred)
y_pred = np.clip(y_pred, 0, 1)

st.subheader("📈 Predicted Result")
st.success(f"Your cyclone efficiency is **{y_pred:.5f}** for this spherical particle.")
st.caption("Predicted efficiency is dimensionless and bounded between 0 and 1.")
