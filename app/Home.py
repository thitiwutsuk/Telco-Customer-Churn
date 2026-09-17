import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency, mannwhitneyu

sys.path.append(str(Path(__file__).resolve().parent))
from src.data import CATEGORICAL_COLS, NUMERIC_COLS, load_clean_data  # noqa: E402
from src.model import CV_SCORING, get_evaluation_results, get_production_pipeline, get_robustness_results  # noqa: E402

st.set_page_config(page_title="Telco Customer Churn", page_icon="📉", layout="wide")
sns.set_style("whitegrid")

df = load_clean_data()

st.title("Telco Customer Churn")

tab_problem, tab_eda, tab_models, tab_predict = st.tabs(
    ["Problem Statement", "Exploratory Data Analysis", "Model Results", "Churn Predictor"]
)

# ---------------------------------------------------------------------------
# Tab 1: Problem Statement
# ---------------------------------------------------------------------------
with tab_problem:
    n_customers = len(df)
    churn_rate = df["Churn_numeric"].mean()

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
    col3.metric(
        "At risk (highest-risk segment)", "73.5%",
        help="Month-to-month + Fiber optic + tenure < 12 months + no security/support add-ons",
    )

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

    st.info("Use the tabs above to explore the EDA findings, compare model results, or try the live churn predictor.")

# ---------------------------------------------------------------------------
# Tab 2: Exploratory Data Analysis
# ---------------------------------------------------------------------------
with tab_eda:
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
        fig, ax = plt.subplots(figsize=(5, 3))
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
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(data=df, x="Churn", y=num_choice, order=["No", "Yes"], hue="Churn",
                    palette=["#55A868", "#C44E52"], legend=False, ax=ax)
        st.pyplot(fig)
    with col2:
        st.markdown(f"**Mann-Whitney U test:** p = `{mw_p:.4g}`")
        st.markdown("Statistically significant (p < 0.05)" if mw_p < 0.05 else "Not statistically significant")
        st.metric("Median — churned", f"{churned.median():,.2f}")
        st.metric("Median — retained", f"{retained.median():,.2f}")

    st.divider()

    st.subheader("Correlation between numeric features")
    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(df[NUMERIC_COLS + ["Churn_numeric"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
        st.pyplot(fig)
    st.caption("`tenure`↔`TotalCharges` correlate strongly (0.83) — a multicollinearity note for the ML phase.")

    st.divider()

    st.subheader("Retention curve — churn rate by tenure")
    tenure_bins = pd.cut(df["tenure"], bins=range(0, 79, 6), right=False)
    retention = df.groupby(tenure_bins, observed=True)["Churn_numeric"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([str(i.left) + "-" + str(i.right - 1) for i in retention.index], retention.values, marker="o", color="#C44E52")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("Tenure (months)")
    plt.xticks(rotation=45, ha="right")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(fig)
    st.caption("Churn peaks early — months 1-6 are the critical retention window (Step 9).")

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

# ---------------------------------------------------------------------------
# Tab 3: Model Results
# ---------------------------------------------------------------------------
with tab_models:
    results = get_evaluation_results(df)

    st.caption(
        f"Logistic Regression vs. Random Forest, evaluated on a held-out test set "
        f"({results['n_test']:,} customers, {results['n_churn_test']:,} of whom actually churned) — "
        "mirrors Step 16 of `notebooks/telco_churn_modeling.ipynb`."
    )

    st.subheader("Metric comparison")
    st.dataframe(results["metrics_df"].style.format("{:.3f}").highlight_max(axis=0, color="#d4edda"))
    st.caption("Logistic Regression wins on Recall/ROC-AUC (catches more churners) — the recommended model.")

    st.divider()

    st.subheader("Confusion matrices")
    cols = st.columns(2)
    for col, (name, cm) in zip(cols, results["confusion_matrices"].items()):
        with col:
            fig, ax = plt.subplots(figsize=(3, 2.6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, annot_kws={"size": 8},
                        xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.tick_params(labelsize=7)
            st.pyplot(fig)

    st.divider()

    st.subheader("ROC curve")
    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(4, 3.3))
        for name, color in [("Logistic Regression", "#4C72B0"), ("Random Forest", "#55A868")]:
            fpr, tpr, auc_score = results["roc_data"][name]
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})", color=color, linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.legend(fontsize=7)
        st.pyplot(fig)

    st.divider()

    with st.expander("Robustness checks — 5-fold Cross-Validation & Learning Curve (Steps 16.4-16.5)"):
        st.caption("Confirms the 80/20 split result above wasn't a fluke.")
        robustness = get_robustness_results(df)

        st.markdown("**5-fold Stratified Cross-Validation**")
        cv_summary = []
        for name, scores in robustness["cv_results"].items():
            row = {"Model": name}
            for metric in CV_SCORING:
                vals = scores[f"test_{metric}"]
                row[f"{metric} (mean)"] = vals.mean()
                row[f"{metric} (std)"] = vals.std()
            cv_summary.append(row)
        st.dataframe(pd.DataFrame(cv_summary).set_index("Model").round(3))
        st.caption("Std stays small — the Logistic Regression Recall advantage holds up across folds.")

        st.markdown("**Learning Curve (Recall)**")
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
        for ax, (name, color) in zip(axes, [("Logistic Regression", "#4C72B0"), ("Random Forest", "#55A868")]):
            sizes, train_scores, val_scores = robustness["learning_curve_results"][name]
            ax.plot(sizes, train_scores.mean(axis=1), "o--", color=color, alpha=0.5, label="Train", markersize=3)
            ax.plot(sizes, val_scores.mean(axis=1), "o-", color=color, label="Validation", markersize=3)
            ax.axvline(x=len(df) * 0.8, linestyle=":", color="gray")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Training samples", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)
        axes[0].set_ylabel("Recall (churn)", fontsize=8)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(fig)
        st.caption("Logistic Regression plateaus early; Random Forest's gap is overfitting, not a data-size issue.")

    st.divider()

    st.subheader("Feature importance (Step 17)")
    top_n = st.slider("Show top N features", 5, 20, 10)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Logistic Regression — coefficients**")
        top_coef = results["coef_df"].head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(4, top_n * 0.28 + 0.6))
        colors = ["#C44E52" if c > 0 else "#4C72B0" for c in top_coef["coefficient"]]
        ax.barh(top_coef["feature"], top_coef["coefficient"], color=colors)
        ax.set_xlabel("Coefficient", fontsize=8)
        ax.tick_params(labelsize=7)
        st.pyplot(fig)
    with col2:
        st.markdown("**Random Forest — feature importances**")
        top_imp = results["importance_df"].head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(4, top_n * 0.28 + 0.6))
        ax.barh(top_imp["feature"], top_imp["importance"], color="#55A868")
        ax.set_xlabel("Importance", fontsize=8)
        ax.tick_params(labelsize=7)
        st.pyplot(fig)

    st.caption(
        "`Contract`, `OnlineSecurity`/`TechSupport`, `MonthlyCharges` confirmed as drivers by both models. "
        "Fiber optic's negative coefficient is a multicollinearity artifact (Step 17.3), not a real effect."
    )

# ---------------------------------------------------------------------------
# Tab 4: Churn Predictor
# ---------------------------------------------------------------------------
with tab_predict:
    pipeline = get_production_pipeline(df)

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
