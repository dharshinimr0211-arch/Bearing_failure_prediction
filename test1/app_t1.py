import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Current folder
current_folder = Path(__file__).parent

# Load model
model = joblib.load(
    current_folder / "bearing_failure_model_t1.pkl"
)

# Load dataset
df = pd.read_csv(
    current_folder / "bearing3_features_labeled.csv"
)

# Title
st.title("Bearing Condition Monitor")

# Select sample
sample_number = st.slider(
    "Select Sample Number",
    0,
    len(df)-1,
    len(df)-1
)

sample = df.iloc[[sample_number]]

# Features only
X_sample = sample[
    [
        "RMS",
        "Peak",
        "STD",
        "Kurtosis",
        "Skewness"
    ]
]

# Prediction
prediction = model.predict(X_sample)

# Probabilities
probability = model.predict_proba(X_sample)

healthy_prob = probability[0][0] * 100
failure_prob = probability[0][1] * 100

# Display probabilities
st.metric("Healthy Probability", f"{healthy_prob:.2f}%")
st.metric("Failure Probability", f"{failure_prob:.2f}%")

# Progress bar
st.progress(int(failure_prob))

# Status message
if failure_prob < 50:
    st.success("Machine Status: HEALTHY")

elif failure_prob < 80:
    st.warning(
        "Machine Status: WARNING\n\n"
        "Maintenance recommended.\n"
    )

else:
    st.error(
        "CRITICAL WARNING: Bearing may fail within the next 36 hours.\n\n"
        "Immediate inspection recommended."
    )