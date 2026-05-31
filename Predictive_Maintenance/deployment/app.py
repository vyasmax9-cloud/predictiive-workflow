import streamlit as st

from Predictive_Maintenance.src.predict import (
    predict_engine_condition
)

st.title(
    "Predictive Maintenance System"
)

st.subheader(
    "Enter Engine Parameters"
)

rpm = st.number_input(
    "Engine RPM",
    min_value=0.0
)

lub_pressure = st.number_input(
    "Lub Oil Pressure",
    min_value=0.0
)

fuel_pressure = st.number_input(
    "Fuel Pressure",
    min_value=0.0
)

coolant_pressure = st.number_input(
    "Coolant Pressure",
    min_value=0.0
)

lub_temp = st.number_input(
    "Lub Oil Temperature",
    min_value=0.0
)

coolant_temp = st.number_input(
    "Coolant Temperature",
    min_value=0.0
)

if st.button(
    "Predict Engine Condition"
):

    sample = {

        "Engine_RPM": rpm,

        "Lub_Oil_Pressure": lub_pressure,

        "Fuel_Pressure": fuel_pressure,

        "Coolant_Pressure": coolant_pressure,

        "Lub_Oil_Temperature": lub_temp,

        "Coolant_Temperature": coolant_temp

    }

    result = predict_engine_condition(
        sample
    )

    if result == 0:

        st.success(
            "Prediction: Healthy Engine"
        )

    else:

        st.error(
            "Prediction: Faulty Engine"
        )
