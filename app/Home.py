import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency, mannwhitneyu

sys.path.append(str(Path(__file__).resolve().parent))
from src.data import CATEGORICAL_COLS, NUMERIC_COLS, load_clean_data, load_raw_data  # noqa: E402
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
    n_churned = int(df["Churn_numeric"].sum())
    churn_rate = df["Churn_numeric"].mean()

    st.subheader("The Problem")
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
    col2.metric("Customers who churned", f"{n_churned:,}")
    col3.metric("Base churn rate", f"{churn_rate:.1%}", help="Roughly 1 in 4 customers left the service.")

    st.divider()

    st.subheader("The Data")
    st.markdown(
        """
- **Source:** IBM Sample Data Sets (also distributed on Kaggle as *Telco Customer Churn*)
- **Size:** 7,043 customers × 21 columns (7,032 × 20 after cleaning)
- **Target:** `Churn` (Yes/No) — whether the customer left the service
- **Features:** demographics, account/billing information, and subscribed services
"""
    )

    st.markdown("**Raw data sample (first 10 rows)**")
    st.dataframe(load_raw_data().head(10), hide_index=True)
    st.caption("As loaded from `data/raw/` before any cleaning — e.g. `SeniorCitizen` is still 0/1 and "
               "`TotalCharges` is still stored as text.")

    st.divider()

    st.subheader("The Approach")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown("**Clean**")
        st.markdown("Convert `TotalCharges` to numeric; drop 11 blank rows (all brand-new customers, tenure = 0).")
    with a2:
        st.markdown("**Explore**")
        st.markdown("Test every feature against churn (Chi-square / Mann-Whitney) and define a high-risk segment.")
    with a3:
        st.markdown("**Model**")
        st.markdown("Train Logistic Regression and Random Forest on SMOTE-balanced data; validate with 5-fold CV.")
    with a4:
        st.markdown("**Predict**")
        st.markdown("Score any customer's churn probability with the selected model.")

    st.divider()

    st.subheader("Key Results")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**Why they churn — confirmed drivers**")
        st.markdown("`Contract` type, `OnlineSecurity`/`TechSupport` subscription, and `MonthlyCharges` — "
                    "agreed on by statistical tests (Chi-square / Mann-Whitney) and both ML models.")
    with r2:
        st.markdown("**Who churns — highest-risk segment**")
        st.markdown("Month-to-month + Fiber optic + tenure < 12 months + no security/support add-ons: "
                    "10.5% of customers, 73.5% churn rate, 29% of all churn.")
    with r3:
        st.markdown("**Predicting it — Logistic Regression**")
        st.markdown("73.8% recall, 0.827 ROC-AUC on the churn class — beats Random Forest (65.5% recall) "
                    "on the metric that matters most: catching customers who will actually churn.")

    st.info("Next: **Exploratory Data Analysis** for the evidence behind these findings, **Model Results** "
            "for the model comparison, or **Churn Predictor** to score a customer yourself.")

# ---------------------------------------------------------------------------
# Tab 2: Exploratory Data Analysis
# ---------------------------------------------------------------------------
with tab_eda:
    st.caption("Follows Steps 1-10 of `notebooks/telco_churn_eda.ipynb` — each step explains what was done, why, "
               "how to read the result, and what it found.")

    base_rate = df["Churn_numeric"].mean() * 100

    def explain(what, why):
        st.markdown(f"**Approach:** {what}")
        st.markdown(f"**Rationale:** {why}")

    def how_to_read(text):
        st.markdown("**Reading the chart**")
        st.caption(text)

    def finding(text, so_what):
        st.success(f"**Key finding:** {text}\n\n**Business implication:** {so_what}")

    # --- Step 1: Data cleaning ------------------------------------------------
    st.subheader("Step 1 — Load & clean the data")
    explain(
        "Loaded the raw CSV, checked each column's data type and missing values, converted `TotalCharges` from "
        "text to numbers, investigated the rows that failed to convert, and standardized `SeniorCitizen` from 0/1 "
        "to No/Yes.",
        "Every later step depends on the data being correct. A number stored as text can't be averaged or tested, "
        "and deleting rows without knowing why they're broken can quietly bias the results — so the cause is "
        "investigated before anything is removed.",
    )

    raw = load_raw_data()
    raw_total = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    blank_rows = raw[raw_total.isna()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{len(raw):,}")
    c2.metric("Rows with blank TotalCharges", f"{len(blank_rows):,}")
    c3.metric("Rows after cleaning", f"{len(df):,}")
    finding(
        f"`isnull()` reported no missing values, but `TotalCharges` was stored as text because blanks were saved "
        f"as a space. Converting it to numeric exposed {len(blank_rows)} blank rows — all with tenure = "
        f"{', '.join(map(str, blank_rows['tenure'].unique()))}, i.e. brand-new customers who haven't been billed yet.",
        f"The blanks are explained, not errors, and they're only {len(blank_rows) / len(raw):.2%} of the data, so "
        "dropping them is safe. These customers also haven't had a chance to churn yet, so keeping them would add "
        "noise.",
    )

    st.divider()

    # --- Step 2: Target distribution -----------------------------------------
    st.subheader("Step 2 — How many customers churn?")
    explain(
        "Counted how many customers churned (`Churn = Yes`) versus stayed.",
        "This gives the base rate — the company-wide average every group is compared against. A group is only "
        "\"high risk\" if it churns noticeably more than this. It also shows whether the classes are balanced, "
        "which decides how the model must be trained and evaluated.",
    )

    churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"])
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(churn_counts.index, churn_counts.values, color=["#55A868", "#C44E52"])
        for i, v in enumerate(churn_counts.values):
            ax.text(i, v, f"{v:,} ({v / len(df):.1%})", ha="center", va="bottom")
        ax.set_ylabel("Customers")
        ax.set_xlabel("Churn")
        st.pyplot(fig)
    with col2:
        how_to_read("Each bar is the number of customers in that group; the label shows the count and its share of "
                    "all customers.")
    finding(
        f"{base_rate:.1f}% of customers churned ({churn_counts['Yes']:,} of {len(df):,}) — roughly 1 in 4.",
        "The data is imbalanced. A model that always predicts \"No churn\" would already be ~73% accurate while "
        "catching nobody, so accuracy is misleading here. The modeling phase uses SMOTE to balance the training "
        "data and focuses on recall instead.",
    )

    st.divider()

    # --- Step 3: Categorical features vs churn -------------------------------
    st.subheader("Step 3 — Which customer attributes relate to churn?")
    explain(
        "Ran a Chi-square test of independence for each of the 16 categorical features against `Churn`, then "
        "compared the churn rate of every category.",
        "A difference in churn rate between two groups could just be random noise. The Chi-square test checks "
        "whether the difference is large enough to be real: a p-value below 0.05 means there's less than a 5% "
        "chance of seeing a gap this big if the feature had no relationship with churn.",
    )

    chi2_rows = []
    for col in CATEGORICAL_COLS:
        chi2, p, _, _ = chi2_contingency(pd.crosstab(df[col], df["Churn"]))
        chi2_rows.append({"Feature": col, "Chi-square": round(chi2, 1), "p-value": p, "Significant": p < 0.05})
    chi2_df = pd.DataFrame(chi2_rows).sort_values("p-value").reset_index(drop=True)
    not_significant = chi2_df.loc[~chi2_df["Significant"], "Feature"].tolist()

    def top_rate(col):
        rates = df.groupby(col, observed=True)["Churn_numeric"].mean() * 100
        return f"{rates.idxmax()} ({rates.max():.1f}%)"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(chi2_df.style.format({"p-value": "{:.2e}"}), hide_index=True, height=250)
    with col2:
        how_to_read("Features are sorted from strongest to weakest relationship with churn. A higher Chi-square and "
                    "a smaller p-value mean a stronger relationship; `Significant` is True when p < 0.05.")
    finding(
        f"{int(chi2_df['Significant'].sum())} of {len(chi2_df)} features are significantly related to churn; only "
        f"{' and '.join(f'`{c}`' for c in not_significant)} are not. `{chi2_df.loc[0, 'Feature']}` is the strongest. "
        f"Highest-churn categories: {top_rate('Contract')}, {top_rate('InternetService')}, "
        f"{top_rate('PaymentMethod')}.",
        "Churn is driven by how customers are contracted and served, not by who they are — gender makes no "
        "difference. Contract type, internet service and payment method are the levers worth looking at first.",
    )

    cat_choice = st.selectbox("Explore a feature", CATEGORICAL_COLS, index=CATEGORICAL_COLS.index("Contract"))
    rate_by_cat = df.groupby(cat_choice, observed=True)["Churn_numeric"].mean().sort_values(ascending=False) * 100
    count_by_cat = df.groupby(cat_choice, observed=True).size()
    p_value = chi2_df.set_index("Feature").loc[cat_choice, "p-value"]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=rate_by_cat.index, y=rate_by_cat.values, ax=ax, color="#4C72B0")
        ax.axhline(base_rate, linestyle="--", color="gray", label=f"Base rate ({base_rate:.1f}%)")
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel(cat_choice)
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
    with col2:
        how_to_read("Each bar is the share of customers in that category who churned. Bars above the dashed line "
                    "churn more than the company average.")
        st.markdown(f"**Chi-square p-value:** `{p_value:.4g}` — "
                    + ("significant" if p_value < 0.05 else "not significant"))
        st.dataframe(
            pd.DataFrame({"Churn rate": rate_by_cat.round(1).astype(str) + "%", "n": count_by_cat[rate_by_cat.index]})
        )

    st.divider()

    # --- Step 4: Numeric features vs churn -----------------------------------
    st.subheader("Step 4 — Do churners differ on tenure and charges?")
    explain(
        "Compared `tenure`, `MonthlyCharges` and `TotalCharges` between churned and retained customers using "
        "medians and a Mann-Whitney U test.",
        "These features are skewed (many new customers, long tails), so the mean would be pulled by extreme "
        "values. The median and the Mann-Whitney U test don't assume a normal distribution, so they give a fairer "
        "comparison.",
    )

    medians = df.groupby("Churn")[NUMERIC_COLS].median()
    num_choice = st.selectbox("Numeric feature", NUMERIC_COLS, index=NUMERIC_COLS.index("tenure"))
    churned = df.loc[df["Churn"] == "Yes", num_choice]
    retained = df.loc[df["Churn"] == "No", num_choice]
    _, mw_p = mannwhitneyu(churned, retained)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(data=df, x="Churn", y=num_choice, order=["No", "Yes"], hue="Churn",
                    palette=["#55A868", "#C44E52"], legend=False, ax=ax)
        st.pyplot(fig)
    with col2:
        how_to_read("The line inside each box is the median; the box covers the middle 50% of customers. Boxes "
                    "that sit at different heights mean the two groups differ.")
        st.markdown(f"**Mann-Whitney U p-value:** `{mw_p:.4g}` — " + ("significant" if mw_p < 0.05 else "not significant"))
        st.metric("Median — churned", f"{churned.median():,.2f}")
        st.metric("Median — retained", f"{retained.median():,.2f}")

    finding(
        f"All 3 numeric features differ significantly. Churners are much newer (median tenure "
        f"{medians.loc['Yes', 'tenure']:.0f} vs {medians.loc['No', 'tenure']:.0f} months) and pay more per month "
        f"(median {medians.loc['Yes', 'MonthlyCharges']:.2f} vs {medians.loc['No', 'MonthlyCharges']:.2f}).",
        "The typical churner is a newer customer on a more expensive plan. Churners have lower `TotalCharges` only "
        "because they leave earlier — not because they spend less per month.",
    )

    st.divider()

    # --- Step 5: Correlation -------------------------------------------------
    st.subheader("Step 5 — Are the numeric features related to each other?")
    explain(
        "Calculated the correlation between every pair of numeric features, plus churn.",
        "If two features carry the same information, a model can't tell which one really matters, and its "
        "coefficients become unreliable (multicollinearity). This check is done before modeling so the results "
        "can be interpreted correctly later.",
    )

    corr = df[NUMERIC_COLS + ["Churn_numeric"]].corr()
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
        st.pyplot(fig)
    with col2:
        how_to_read("Values range from -1 to 1. Close to 1 (red) means the two rise together, close to -1 (blue) "
                    "means one rises as the other falls, and close to 0 means no linear relationship.")
    finding(
        f"`tenure` and `TotalCharges` are strongly correlated (r = {corr.loc['tenure', 'TotalCharges']:.2f}), since "
        f"total charges ≈ monthly charges × tenure. `tenure` is negatively related to churn "
        f"(r = {corr.loc['tenure', 'Churn_numeric']:.2f}).",
        "Both features are kept, but their model coefficients must be read with care — the effect of tenure gets "
        "split between the two. This explains why `tenure` ranks lower in Logistic Regression than in the EDA.",
    )

    st.divider()

    # --- Step 6: Interaction -------------------------------------------------
    st.subheader("Step 6 — What happens when risk factors combine?")
    explain(
        "Built a pivot table of churn rate by `Contract` × `InternetService`, the two strongest drivers from Step 3.",
        "Step 3 looked at one feature at a time. Real customers have several traits at once, and two risk factors "
        "together can be far worse than either one alone.",
    )

    interaction = df.pivot_table(index="Contract", columns="InternetService", values="Churn_numeric", aggfunc="mean") * 100
    low = interaction.stack().idxmin()
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(interaction, annot=True, fmt=".1f", cmap="Reds", ax=ax)
        st.pyplot(fig)
    with col2:
        how_to_read("Each cell is the churn rate (%) of customers with that contract and internet service. Darker "
                    "red means higher churn.")
    finding(
        f"Month-to-month + Fiber optic churns at {interaction.loc['Month-to-month', 'Fiber optic']:.1f}%, versus "
        f"{interaction.stack().min():.1f}% for {low[0]} + {'no internet' if low[1] == 'No' else low[1]}.",
        "Risk factors stack. Fiber optic customers on longer contracts churn far less, so offering them a longer "
        "contract is a natural retention lever (this is an association, not proof of cause).",
    )

    st.divider()

    # --- Step 7: Retention curve ---------------------------------------------
    st.subheader("Step 7 — When in the customer lifetime does churn happen?")
    explain(
        "Plotted churn rate against tenure, grouped into 6-month bands.",
        "Knowing when customers leave tells the business when to act. Retention offers work best if they arrive "
        "before the peak risk period, not after.",
    )

    tenure_bins = pd.cut(df["tenure"], bins=range(0, 79, 6), right=False)
    retention = df.groupby(tenure_bins, observed=True)["Churn_numeric"].mean() * 100
    by_month = df.groupby("tenure")["Churn_numeric"].mean() * 100
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([f"{i.left}-{i.right - 1}" for i in retention.index], retention.values, marker="o", color="#C44E52")
        ax.axhline(base_rate, linestyle="--", color="gray", label=f"Base rate ({base_rate:.1f}%)")
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel("Tenure (months)")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    with col2:
        how_to_read("Each point is the churn rate of customers in that tenure band. Points above the dashed line "
                    "churn more than the company average.")
    finding(
        f"Churn peaks at month {by_month.idxmax()} ({by_month.max():.1f}%) and falls steadily after that. The first "
        f"band ({retention.index[0].left}-{retention.index[0].right - 1} months) averages {retention.iloc[0]:.1f}% churn.",
        "The first months are the critical retention window. Onboarding and early check-ins matter more than "
        "loyalty rewards for long-term customers, who rarely leave anyway.",
    )

    st.divider()

    # --- Step 8: Risk segment ------------------------------------------------
    st.subheader("Step 8 — Define the highest-risk segment")
    explain(
        "Combined the strongest findings from Steps 3-7 into one rule: month-to-month contract + Fiber optic + "
        "tenure < 12 months + no OnlineSecurity + no TechSupport.",
        "Individual findings are hard to act on. A clearly defined segment gives the business a concrete list of "
        "customers to target, and lets us measure how much of the churn problem it covers.",
    )

    risk_mask = (
        (df["Contract"] == "Month-to-month")
        & (df["InternetService"] == "Fiber optic")
        & (df["tenure"] < 12)
        & (df["OnlineSecurity"] == "No")
        & (df["TechSupport"] == "No")
    )
    risk_segment = df[risk_mask]
    segment_share = len(risk_segment) / len(df)
    segment_rate = risk_segment["Churn_numeric"].mean()
    churn_share = risk_segment["Churn_numeric"].sum() / df["Churn_numeric"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Segment size", f"{len(risk_segment):,} ({segment_share:.1%} of customers)")
    c2.metric("Segment churn rate", f"{segment_rate:.1%}")
    c3.metric("Share of all churn", f"{churn_share:.1%}")
    finding(
        f"Just {segment_share:.1%} of customers churn at {segment_rate:.1%} — about "
        f"{segment_rate * 100 / base_rate:.1f}× the base rate — and account for {churn_share:.1%} of all churn.",
        "Focusing retention effort on about 1 in 10 customers addresses nearly a third of all churn. Bundling "
        "security/support add-ons and offering longer contracts to this group are the natural first actions.",
    )

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
