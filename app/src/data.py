"""Shared data loading for the Streamlit app.

Mirrors the cleaning already done in `notebooks/telco_churn_eda.ipynb` (Steps 1-2) and the column
groups defined in `notebooks/telco_churn_modeling.ipynb` (Step 12) — this module just loads the
already-cleaned CSV those notebooks produce, it does not repeat the cleaning itself.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "telco_churn_clean.csv"

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

RANDOM_STATE = 42


@st.cache_data(show_spinner="Loading dataset...")
def load_clean_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            f"Cleaned dataset not found at `{DATA_PATH}`. Run `telco_churn_eda.ipynb` end-to-end "
            "at least once to generate it before launching this app."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH, index_col="customerID")
    df["Churn_numeric"] = (df["Churn"] == "Yes").astype(int)
    return df
