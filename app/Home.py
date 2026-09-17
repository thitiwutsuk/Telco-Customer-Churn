import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from src.data import load_clean_data  # noqa: E402

st.set_page_config(page_title="Telco Customer Churn", page_icon="📉", layout="wide")

df = load_clean_data()
n_customers = len(df)
churn_rate = df["Churn_numeric"].mean()

st.title("Telco Customer Churn — Problem Statement")

st.markdown(
    """
A telecom company is losing customers (**churn**) at a meaningful rate, but doesn't know **who**
is likely to leave or **why**. Acquiring a new customer costs far more than retaining an existing
one — without knowing which customers are at risk, retention spend is a guess instead of a
targeted decision.

**Project goal:** use the company's own customer records to (1) find which customer attributes are
statistically associated with churn, and (2) build a model that scores each customer's churn
probability, so retention efforts can be targeted at the customers most likely to leave.
"""
)

col1, col2, col3 = st.columns(3)
col1.metric("Customers analyzed", f"{n_customers:,}")
col2.metric("Base churn rate", f"{churn_rate:.1%}")
col3.metric("At risk (highest-risk segment)", "73.5%", help="Month-to-month + Fiber optic + tenure < 12 months + no security/support add-ons")

st.divider()

st.subheader("Key Results")
r1, r2, r3 = st.columns(3)
with r1:
    st.markdown("**Best model: Logistic Regression**")
    st.markdown("73.8% recall, 0.827 ROC-AUC on the churn class — beats Random Forest (65.5% recall) "
                "on the metric that matters most: catching customers who will actually churn.")
with r2:
    st.markdown("**Confirmed churn drivers**")
    st.markdown("`Contract` type, `OnlineSecurity`/`TechSupport` subscription, and `MonthlyCharges` — "
                "agreed on by statistical tests (Chi-square / Mann-Whitney) and both ML models.")
with r3:
    st.markdown("**Actionable segment found**")
    st.markdown("10.5% of customers carry a 73.5% churn rate and account for 29% of all churn — "
                "a small, high-leverage group for retention outreach.")

st.divider()

st.subheader("Dataset")
st.markdown(
    """
- **Source:** IBM Sample Data Sets (also distributed on Kaggle as *Telco Customer Churn*)
- **Size:** 7,043 customers × 21 columns (7,032 × 20 after cleaning)
- **Target:** `Churn` (Yes/No) — whether the customer left the service
- **Features:** demographics, account/billing information, and subscribed services
"""
)

st.info("Use the sidebar to explore the EDA findings, compare model results, or try the live churn predictor.")
