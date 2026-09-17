import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency, mannwhitneyu

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data import CATEGORICAL_COLS, NUMERIC_COLS, load_clean_data  # noqa: E402

st.set_page_config(page_title="EDA — Telco Churn", page_icon="📊", layout="wide")
sns.set_style("whitegrid")

df = load_clean_data()

st.title("Exploratory Data Analysis")
st.caption("Mirrors Steps 3-10 of `notebooks/telco_churn_eda.ipynb` — pick a feature below to explore it interactively.")

st.subheader("Churn rate by category")
cat_choice = st.selectbox("Categorical feature", CATEGORICAL_COLS, index=CATEGORICAL_COLS.index("Contract"))

rate_by_cat = (
    df.groupby(cat_choice, observed=True)["Churn_numeric"].mean().sort_values(ascending=False) * 100
)
count_by_cat = df.groupby(cat_choice, observed=True).size()

contingency = pd.crosstab(df[cat_choice], df["Churn"])
chi2, p_value, _, _ = chi2_contingency(contingency)

col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=rate_by_cat.index, y=rate_by_cat.values, ax=ax, color="#4C72B0")
    ax.axhline(df["Churn_numeric"].mean() * 100, linestyle="--", color="gray", label="Base rate (26.6%)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(cat_choice)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)
with col2:
    st.markdown(f"**Chi-square test:** p = `{p_value:.4g}`")
    st.markdown("Statistically significant (p < 0.05)" if p_value < 0.05 else "Not statistically significant")
    st.dataframe(
        pd.DataFrame({"Churn rate": rate_by_cat.round(1).astype(str) + "%", "n": count_by_cat[rate_by_cat.index]})
    )

st.divider()

st.subheader("Numeric features vs. churn")
num_choice = st.selectbox("Numeric feature", NUMERIC_COLS, index=NUMERIC_COLS.index("tenure"))

churned = df.loc[df["Churn"] == "Yes", num_choice]
retained = df.loc[df["Churn"] == "No", num_choice]
u_stat, mw_p = mannwhitneyu(churned, retained)

col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y=num_choice, order=["No", "Yes"], ax=ax, palette=["#55A868", "#C44E52"])
    st.pyplot(fig)
with col2:
    st.markdown(f"**Mann-Whitney U test:** p = `{mw_p:.4g}`")
    st.markdown("Statistically significant (p < 0.05)" if mw_p < 0.05 else "Not statistically significant")
    st.metric("Median — churned", f"{churned.median():,.2f}")
    st.metric("Median — retained", f"{retained.median():,.2f}")

st.divider()

st.subheader("Correlation between numeric features")
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(df[NUMERIC_COLS + ["Churn_numeric"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
st.pyplot(fig)
st.caption(
    "`tenure` and `TotalCharges` correlate strongly (0.83) since `TotalCharges` accumulates over "
    "`tenure` — a multicollinearity note carried into the ML modeling phase (Step 7)."
)

st.divider()

st.subheader("Retention curve — churn rate by tenure")
tenure_bins = pd.cut(df["tenure"], bins=range(0, 79, 6), right=False)
retention = df.groupby(tenure_bins, observed=True)["Churn_numeric"].mean() * 100
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot([str(i.left) + "-" + str(i.right - 1) for i in retention.index], retention.values, marker="o", color="#C44E52")
ax.set_ylabel("Churn rate (%)")
ax.set_xlabel("Tenure (months)")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)
st.caption("Churn peaks in the first months and declines steadily — months 1-6 are the critical retention window (Step 9).")

st.divider()

st.subheader("Highest-risk segment (Step 10)")
risk_mask = (
    (df["Contract"] == "Month-to-month")
    & (df["InternetService"] == "Fiber optic")
    & (df["tenure"] < 12)
    & (df["OnlineSecurity"] == "No")
    & (df["TechSupport"] == "No")
)
risk_segment = df[risk_mask]
c1, c2, c3 = st.columns(3)
c1.metric("Segment size", f"{len(risk_segment):,} ({len(risk_segment) / len(df):.1%} of customers)")
c2.metric("Segment churn rate", f"{risk_segment['Churn_numeric'].mean():.1%}")
c3.metric("Share of all churn", f"{risk_segment['Churn_numeric'].sum() / df['Churn_numeric'].sum():.1%}")
st.caption("Defined as: month-to-month contract + Fiber optic + tenure < 12 months + no OnlineSecurity + no TechSupport.")
