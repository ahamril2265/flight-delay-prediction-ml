import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

from shap_utils import get_shap_explanation

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(
    page_title="Flight Delay Predictor",
    layout="wide"
)

# --------------------------------------------------
# Load Model (Cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")

model = load_model()

# --------------------------------------------------
# Prediction Helper
# --------------------------------------------------
def predict_delay(input_df):
    proba = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= 0.5)
    return prediction, proba

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("✈️ Flight Delay Prediction System")
st.markdown(
    """
    Predict whether a flight is **likely to be delayed before departure**  
    using a machine-learning model trained on real US airline data.
    """
)

with st.form("flight_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.text_input("Airline Code (e.g. AA, DL, UA)", "AA")
        origin = st.text_input("Origin Airport", "JFK")
        destination = st.text_input("Destination Airport", "LAX")

    with col2:
        dep_hour = st.slider("Scheduled Departure Hour", 0, 23, 10)
        day_of_week = st.selectbox(
            "Day of Week",
            [1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x - 1]
        )

    with col3:
        distance = st.number_input("Flight Distance (miles)", 100, 5000, 1000)
        is_peak = st.checkbox("Peak Hour")
        is_weekend = st.checkbox("Weekend")

    submitted = st.form_submit_button("Predict Delay")

# --------------------------------------------------
# Prediction + Explanation
# --------------------------------------------------
if submitted:
    route = f"{origin}_{destination}"

    input_df = pd.DataFrame([{
        "AIRLINE": airline,
        "ORIGIN_AIRPORT": origin,
        "DESTINATION_AIRPORT": destination,
        "ROUTE": route,
        "DAY_OF_WEEK": day_of_week,
        "DEP_HOUR": dep_hour,
        "IS_PEAK_HOUR": int(is_peak),
        "IS_WEEKEND": int(is_weekend),
        "DISTANCE": distance
    }])

    prediction, probability = predict_delay(input_df)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Likely Delayed — Probability: {probability:.2%}")
    else:
        st.success(f"✅ Likely On-Time — Delay Probability: {probability:.2%}")

    # --------------------------------------------------
    # SHAP Explanation
    # --------------------------------------------------
    st.subheader("Why this prediction?")

    try:
        shap_values, expected_value = get_shap_explanation(model, input_df)

        fig, ax = plt.subplots(figsize=(10, 3))
        shap.force_plot(
            expected_value[1],
            shap_values[1][0],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation could not be generated for this input.")
        st.text(str(e))

