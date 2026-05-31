import streamlit as st
import sys
import os

# Add project root to path
ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(0, ROOT)

from src.predict import predict_engine_condition

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="centered"
)

# =========================
# Sidebar
# =========================

st.sidebar.title("⚙️ About")

st.sidebar.info(
    """
    Predictive Maintenance System

    This application predicts whether an engine is:

    ✅ Healthy

    ❌ Faulty

    using operational engine parameters.
    """
)

# =========================
# Main Title
# =========================

st.title("⚙️ Predictive Maintenance System")

st.markdown(
    "Enter the engine parameters below and click **Predict Engine Condition**."
)

# =========================
# Input Fields
# =========================

rpm = st.number_input(
    "Engine RPM",
    min_value=0.0,
    value=1500.0
)

lub_pressure = st.number_input(
    "Lub Oil Pressure",
    min_value=0.0,
    value=3.5
)

fuel_pressure = st.number_input(
    "Fuel Pressure",
    min_value=0.0,
    value=4.0
)

coolant_pressure = st.number_input(
    "Coolant Pressure",
    min_value=0.0,
    value=2.5
)

lub_temp = st.number_input(
    "Lub Oil Temperature",
    min_value=0.0,
    value=80.0
)

coolant_temp = st.number_input(
    "Coolant Temperature",
    min_value=0.0,
    value=90.0
)

# =========================
# Prediction Button
# =========================

if st.button("🔍 Predict Engine Condition"):

    sample = {
        "Engine_RPM": rpm,
        "Lub_Oil_Pressure": lub_pressure,
        "Fuel_Pressure": fuel_pressure,
        "Coolant_Pressure": coolant_pressure,
        "Lub_Oil_Temperature": lub_temp,
        "Coolant_Temperature": coolant_temp
    }

    result = predict_engine_condition(sample)

    st.divider()

    if result == 0:
        st.success("✅ Prediction: Healthy Engine")
    else:
        st.error("❌ Prediction: Faulty Engine")

# =========================
# Footer
# =========================

st.markdown("---")
st.caption(
    "Built with Python, Scikit-Learn, Streamlit & Hugging Face"
)
