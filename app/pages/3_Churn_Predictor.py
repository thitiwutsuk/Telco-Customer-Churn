import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data import CATEGORICAL_COLS, NUMERIC_COLS, load_clean_data  # noqa: E402
from src.model import get_production_pipeline  # noqa: E402

st.set_page_config(page_title="Churn Predictor — Telco Churn", page_icon="🔮", layout="centered")

df = load_clean_data()
pipeline = get_production_pipeline(df)

st.title("Churn Predictor")
st.caption(
    "Enter a customer's details to estimate their churn probability, using a Logistic Regression "
    "model trained on the full dataset (the model selected in Step 16/18)."
)

with st.form("customer_form"):
    st.subheader("Demographics")
    c1, c2, c3, c4 = st.columns(4)
    gender = c1.selectbox("Gender", sorted(df["gender"].unique()))
    senior = c2.selectbox("Senior citizen", sorted(df["SeniorCitizen"].unique()))
    partner = c3.selectbox("Has partner", sorted(df["Partner"].unique()))
    dependents = c4.selectbox("Has dependents", sorted(df["Dependents"].unique()))

    st.subheader("Account")
    c1, c2, c3 = st.columns(3)
    contract = c1.selectbox("Contract", sorted(df["Contract"].unique()))
    paperless = c2.selectbox("Paperless billing", sorted(df["PaperlessBilling"].unique()))
    payment = c3.selectbox("Payment method", sorted(df["PaymentMethod"].unique()))

    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure (months)", int(df["tenure"].min()), int(df["tenure"].max()), 12)
    monthly = c2.slider("Monthly charges", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()), 70.0)
    total = c3.number_input("Total charges", min_value=0.0, value=round(monthly * tenure, 2))

    st.subheader("Services")
    c1, c2, c3 = st.columns(3)
    phone = c1.selectbox("Phone service", sorted(df["PhoneService"].unique()))
    multiple_lines = c2.selectbox("Multiple lines", sorted(df["MultipleLines"].unique()))
    internet = c3.selectbox("Internet service", sorted(df["InternetService"].unique()))

    c1, c2, c3 = st.columns(3)
    online_security = c1.selectbox("Online security", sorted(df["OnlineSecurity"].unique()))
    online_backup = c2.selectbox("Online backup", sorted(df["OnlineBackup"].unique()))
    device_protection = c3.selectbox("Device protection", sorted(df["DeviceProtection"].unique()))

    c1, c2 = st.columns(2)
    tech_support = c1.selectbox("Tech support", sorted(df["TechSupport"].unique()))
    streaming_tv = c2.selectbox("Streaming TV", sorted(df["StreamingTV"].unique()))
    streaming_movies = st.selectbox("Streaming movies", sorted(df["StreamingMovies"].unique()))

    submitted = st.form_submit_button("Predict churn probability", use_container_width=True)

if submitted:
    input_row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "PhoneService": phone, "MultipleLines": multiple_lines, "InternetService": internet,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
    }])[CATEGORICAL_COLS + NUMERIC_COLS]

    probability = pipeline.predict_proba(input_row)[0, 1]

    st.divider()
    st.subheader("Result")

    if probability >= 0.5:
        st.error(f"**High risk** — estimated {probability:.1%} probability of churning")
    elif probability >= 0.3:
        st.warning(f"**Medium risk** — estimated {probability:.1%} probability of churning")
    else:
        st.success(f"**Low risk** — estimated {probability:.1%} probability of churning")

    st.progress(min(probability, 1.0))
    st.caption(
        "Risk tiers are illustrative thresholds (30% / 50%), not derived from a cost-benefit "
        "analysis of retention spend vs. churn cost — see Step 10 for the rule-based segment "
        "definition used in the EDA phase instead."
    )
