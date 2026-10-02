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



# Shared layout for every analysis step: Approach / Rationale, a chart with reading notes, then a Key finding box
def explain(what, why):
    st.markdown(f"**Approach:** {what}")
    st.markdown(f"**Rationale:** {why}")


def how_to_read(text):
    st.markdown("**Reading the chart**")
    st.caption(text)


def finding(text, so_what):
    st.success(f"**Key finding:** {text}\n\n**Business implication:** {so_what}")


df = load_clean_data()

st.title("Customer Churn Analysis")

tab_exec, tab_problem, tab_eda, tab_models, tab_predict = st.tabs(
    ["Executive Brief", "Overview", "Exploratory Data Analysis", "Model Results", "Churn Predictor"]
)

# ---------------------------------------------------------------------------
# Tab 0: Executive Brief (big picture first, then drill down to the segment to act on)
# ---------------------------------------------------------------------------
with tab_exec:
    churned = df[df["Churn_numeric"] == 1]
    churn_rate = len(churned) / len(df)
    lost_billing = churned["MonthlyCharges"].sum()
    lost_share = lost_billing / df["MonthlyCharges"].sum()

    def group_stats(mask):
        return int(mask.sum()), df.loc[mask, "Churn_numeric"].mean()

    # --- 1. Big picture -------------------------------------------------------
    st.subheader("1 · Customer Overview")
    st.caption("About 1 in 4 customers cancelled.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(df):,}")
    c2.metric("Stayed", f"{len(df) - len(churned):,}", f"{1 - churn_rate:.1%}", delta_color="off")
    c3.metric("Cancelled", f"{len(churned):,}", f"{churn_rate:.1%}", delta_color="off")
    c4.metric("Monthly billing lost", f"{lost_billing:,.0f}", f"{lost_share:.1%} of total", delta_color="off")

    fig, ax = plt.subplots(figsize=(8, 0.9))
    ax.barh([0], [1 - churn_rate], color="#55A868")
    ax.barh([0], [churn_rate], left=[1 - churn_rate], color="#C44E52")
    ax.text((1 - churn_rate) / 2, 0, f"Stayed  {len(df) - len(churned):,}", ha="center", va="center", color="white")
    ax.text(1 - churn_rate / 2, 0, f"Cancelled  {len(churned):,}", ha="center", va="center", color="white")
    ax.set_xlim(0, 1)
    ax.axis("off")
    col1, _ = st.columns([3, 1])
    with col1:
        st.pyplot(fig)

    st.divider()

    # --- 2. Drivers -----------------------------------------------------------
    st.subheader("2 · Why Customers Cancel")
    st.caption("Share of customers who cancelled in each group.")

    drivers = [
        ("Contract", "Month-to-\nmonth", "1–2 year", df["Contract"] == "Month-to-month", None),
        ("Security / support add-on", "None", "Has one",
         (df["OnlineSecurity"] == "No") & (df["TechSupport"] == "No"), df["InternetService"] != "No"),
        ("Time as customer", "Under\n12 months", "12+ months", df["tenure"] < 12, None),
        ("Payment method", "Electronic\ncheck", "Other", df["PaymentMethod"] == "Electronic check", None),
        ("Internet service", "Fiber optic", "DSL", df["InternetService"] == "Fiber optic", df["InternetService"] != "No"),
        ("Gender", "Male", "Female", df["gender"] == "Male", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.6), sharey=True)
    for ax, (title, risk_name, other_name, risk_mask, scope) in zip(axes.flat, drivers):
        scope = pd.Series(True, index=df.index) if scope is None else scope
        values = [group_stats(scope & risk_mask)[1] * 100, group_stats(scope & ~risk_mask)[1] * 100]
        risk_color = "#BFBFBF" if title == "Gender" else "#C44E52"
        ax.bar([0, 1], values, color=[risk_color, "#D9D9D9"], width=0.6)
        for x, v in enumerate(values):
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([risk_name, other_name], fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 60)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(w_pad=4, h_pad=3)
    _, col, _ = st.columns([1, 4, 1])
    with col:
        st.pyplot(fig)

    st.markdown("**Contract type matters most.** What customers buy and how long they've stayed drive "
                "cancellations — not who they are.")

    st.divider()

    # --- 3. Model evaluation (5-fold cross-validation) -----------------------
    st.subheader("3 · Can We Predict Who Will Leave?")
    st.caption("Each model was tested 5 times on different customer groups. ± shows how much results changed "
               "between tests.")

    plain_metrics = {
        "recall": "Leavers caught",
        "precision": "Alerts that were correct",
        "accuracy": "Overall accuracy",
        "roc_auc": "Ranking quality (0–1)",
    }
    cv_rows = {}
    for name, scores in get_robustness_results(df)["cv_results"].items():
        label = f"{name} (selected)" if name == "Logistic Regression" else name
        cv_rows[label] = {
            plain: (f"{scores[f'test_{key}'].mean():.3f} ± {scores[f'test_{key}'].std():.3f}" if key == "roc_auc"
                    else f"{scores[f'test_{key}'].mean():.1%} ± {scores[f'test_{key}'].std() * 100:.1f}")
            for key, plain in plain_metrics.items()
        }
    st.dataframe(pd.DataFrame(cv_rows).T, width="stretch")

    lr_recall = get_robustness_results(df)["cv_results"]["Logistic Regression"]["test_recall"]
    st.markdown(f"**The selected model catches about {lr_recall.mean():.0%} of customers who leave, "
                "and the result is consistent across all 5 tests.**")

# ---------------------------------------------------------------------------
# Tab 1: Overview (problem statement, executive summary, recommendations)
# ---------------------------------------------------------------------------
with tab_problem:
    n_customers = len(df)
    churned = df[df["Churn_numeric"] == 1]
    n_churned = len(churned)
    churn_rate = n_churned / n_customers
    lost_billing = churned["MonthlyCharges"].sum()
    lost_share = lost_billing / df["MonthlyCharges"].sum()
    early_share = (churned["tenure"] <= 12).mean()
    mtm_share = (churned["Contract"] == "Month-to-month").mean()

    risk_mask = (
        (df["Contract"] == "Month-to-month")
        & (df["InternetService"] == "Fiber optic")
        & (df["tenure"] < 12)
        & (df["OnlineSecurity"] == "No")
        & (df["TechSupport"] == "No")
    )
    segment_share = risk_mask.mean()
    segment_rate = df.loc[risk_mask, "Churn_numeric"].mean()
    segment_churn_share = df.loc[risk_mask, "Churn_numeric"].sum() / n_churned

    lr_metrics = get_evaluation_results(df)["metrics_df"].loc["Logistic Regression"]

    st.caption(f"Internal report · Prepared for the management team · Based on a snapshot of {n_customers:,} "
               "active and former customer accounts")

    # --- Problem statement ---------------------------------------------------
    st.header("Problem Statement")
    st.markdown(
        f"""
**{churn_rate:.1%} of our customers have cancelled their service** — roughly 1 in 4. Together, the customers
we lost were billing **{lost_billing:,.0f} per month**, equal to **{lost_share:.1%} of our total monthly
charges**.

Our retention activity today is not targeted. We have no systematic way to tell which customers are about
to leave or why, so retention offers are either sent too broadly — spending budget on customers who would
have stayed — or reach at-risk customers too late. Since winning a new customer costs considerably more than
keeping an existing one, every avoidable cancellation is a direct loss.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers in scope", f"{n_customers:,}")
    c2.metric("Customers lost", f"{n_churned:,}")
    c3.metric("Churn rate", f"{churn_rate:.1%}")
    c4.metric("Monthly billing lost", f"{lost_billing:,.0f}", f"{lost_share:.1%} of total", delta_color="off")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Questions this report answers**")
        st.markdown(
            """
1. **Who** is leaving?
2. **Why** are they leaving?
3. **When** in the customer lifetime does it happen?
4. **Can we identify** at-risk customers before they leave?
"""
        )
    with col2:
        st.markdown("**Objective**")
        st.markdown(
            "Give the retention team a clear, evidence-based view of churn drivers and a risk score for every "
            "customer, so retention budget goes to the customers most likely to leave."
        )
        st.markdown("**Success criteria**")
        st.markdown(
            "The model must catch the majority of customers who actually leave — missing a churner costs more "
            "than contacting a customer who would have stayed."
        )

    st.divider()

    # --- Executive summary ---------------------------------------------------
    st.header("Executive Summary")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("**Who is leaving**")
        st.markdown(
            f"A clearly defined segment — month-to-month, Fiber optic, under 12 months, no security or support "
            f"add-ons — is only **{segment_share:.1%}** of customers but churns at **{segment_rate:.1%}** and "
            f"accounts for **{segment_churn_share:.1%}** of all churn."
        )
    with s2:
        st.markdown("**Why they leave**")
        st.markdown(
            "Churn is driven by how customers are contracted and served, not by who they are. Contract type, "
            "lack of OnlineSecurity/TechSupport and higher monthly charges are confirmed by both statistical "
            "tests and the models. Gender has no effect."
        )
    with s3:
        st.markdown("**When it happens**")
        st.markdown(
            f"Early. **{early_share:.0%}** of the customers we lost left within their first 12 months, and "
            f"**{mtm_share:.0%}** were on month-to-month contracts. Risk falls sharply after the first year."
        )
    with s4:
        st.markdown("**Can we predict it**")
        st.markdown(
            f"Yes. The selected model catches **{lr_metrics['Recall (churn)']:.1%}** of customers who actually "
            f"leave (ROC-AUC {lr_metrics['ROC-AUC']:.3f}), and the result held up under 5-fold cross-validation."
        )

    st.divider()

    # --- Recommendations -----------------------------------------------------
    st.header("Recommendations")
    st.dataframe(
        pd.DataFrame(
            [
                ["1", "Proactive onboarding programme for the first 6 months",
                 "All new customers", "Churn is highest in the first months of the relationship"],
                ["2", "Incentives to move from month-to-month to 1- or 2-year contracts",
                 "Month-to-month customers", "Contract type is the strongest single driver of churn"],
                ["3", "Bundle OnlineSecurity / TechSupport for new Fiber optic customers",
                 "Fiber optic customers without add-ons", "Customers with these add-ons churn far less"],
                ["4", "Review the Electronic check payment experience",
                 "Electronic check payers", "This payment method has the highest churn rate"],
                ["5", "Use the model's risk score to prioritise retention outreach",
                 "Customers scored as high risk", "Focuses budget where it prevents the most churn"],
            ],
            columns=["Priority", "Action", "Target group", "Evidence"],
        ),
        hide_index=True,
        width="stretch",
    )

    st.divider()

    # --- Data & methodology --------------------------------------------------
    st.header("Data & Methodology")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Data used**")
        st.markdown(
            """
- Customer account snapshot: 7,043 accounts × 21 fields (public IBM *Telco Customer Churn* sample)
- 7,032 accounts after cleaning — 11 brand-new accounts with no billing history removed
- Fields cover demographics, contract and billing, and subscribed services
- Outcome tracked: `Churn` — whether the customer cancelled
"""
        )
    with col2:
        st.markdown("**How the analysis was done**")
        st.markdown(
            """
1. **Clean** — fixed data types and investigated missing values before removing anything
2. **Explore** — tested every field against churn and identified the highest-risk segment
3. **Model** — compared Logistic Regression and Random Forest, validated with 5-fold cross-validation
4. **Score** — deployed the selected model so any customer can be scored
"""
        )

    with st.expander("View a sample of the raw data (first 10 rows)"):
        st.dataframe(load_raw_data().head(10), hide_index=True)
        st.caption("As received, before cleaning — e.g. `SeniorCitizen` is still 0/1 and `TotalCharges` is "
                   "still stored as text.")

    st.markdown("**Limitations**")
    st.caption(
        "Findings show association, not proven cause. The data is a single snapshot, so trends over calendar "
        "time cannot be measured. Model thresholds and the risk segment were set by judgement and should be "
        "refined with retention-cost data before rollout."
    )

    st.info("The following tabs contain the supporting detail: **Exploratory Data Analysis** (evidence behind "
            "each finding), **Model Results** (model selection and validation) and **Churn Predictor** (score an "
            "individual customer).")

# ---------------------------------------------------------------------------
# Tab 2: Exploratory Data Analysis
# ---------------------------------------------------------------------------
with tab_eda:
    st.caption("Follows Steps 1-10 of `notebooks/telco_churn_eda.ipynb` — each step explains what was done, why, "
               "how to read the result, and what it found.")

    base_rate = df["Churn_numeric"].mean() * 100

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
    robustness = get_robustness_results(df)

    st.caption("Follows Steps 12-17 of `notebooks/telco_churn_modeling.ipynb` — each step explains what was done, "
               "why, how to read the result, and what it found.")

    metrics = results["metrics_df"]
    lr, rf = metrics.loc["Logistic Regression"], metrics.loc["Random Forest"]
    n_test, n_churn_test = results["n_test"], results["n_churn_test"]
    n_train = len(df) - n_test
    n_churn_train = int(df["Churn_numeric"].sum()) - n_churn_test

    # --- Step 1: Feature preparation -----------------------------------------
    st.subheader("Step 1 — Prepare the data for modeling")
    explain(
        "Converted `Churn` to 1/0, one-hot encoded the 16 categorical features (`drop_first=True`), split the data "
        "80/20 into training and test sets with stratification, and standardized the 3 numeric features using "
        "statistics from the training set only.",
        "Models only understand numbers, so categories must be turned into 0/1 columns. The test set is locked "
        "away to measure performance on customers the model has never seen. Stratifying keeps the churn rate "
        "identical in both sets, and fitting the scaler on training data only prevents information from the test "
        "set leaking into training.",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model features", f"{len(results['coef_df'])}")
    c2.metric("Training set", f"{n_train:,} customers")
    c3.metric("Test set", f"{n_test:,} customers")
    finding(
        f"The 16 categorical features became {len(results['coef_df']) - len(NUMERIC_COLS)} one-hot columns, giving "
        f"{len(results['coef_df'])} features in total. Both sets keep the same churn rate "
        f"({n_churn_train / n_train:.1%} train, {n_churn_test / n_test:.1%} test).",
        "The test results below reflect how the model would perform on real, new customers with the same "
        "churn mix the business actually sees.",
    )

    st.divider()

    # --- Step 2: SMOTE ------------------------------------------------------
    st.subheader("Step 2 — Balance the training data (SMOTE)")
    explain(
        "Applied SMOTE to the training set only, creating synthetic churn examples until both classes were the "
        "same size. The test set was left untouched.",
        "With only ~1 in 4 customers churning, a model can score well by mostly predicting \"No churn\" and still "
        "miss most churners. Balancing the training data forces the model to learn what churners look like. The "
        "test set stays at the real churn rate so the evaluation stays honest.",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Churners before SMOTE", f"{n_churn_train:,}")
    c2.metric("Non-churners", f"{n_train - n_churn_train:,}")
    c3.metric("Churners after SMOTE", f"{n_train - n_churn_train:,}")
    finding(
        f"The training set went from {n_churn_train:,} churners vs {n_train - n_churn_train:,} non-churners to an "
        f"even {n_train - n_churn_train:,} / {n_train - n_churn_train:,}.",
        "The model now pays equal attention to both outcomes. The synthetic examples are not real customers, "
        "which is a known limitation, so the cross-validation in Step 5 checks the results still hold up.",
    )

    st.divider()

    # --- Step 3: Metric comparison -----------------------------------------
    st.subheader("Step 3 — Compare the two models")
    explain(
        "Trained Logistic Regression and Random Forest on the balanced training data, then scored both on the "
        f"held-out test set ({n_test:,} customers, {n_churn_test:,} of whom actually churned).",
        "Logistic Regression is simple and easy to explain; Random Forest can capture more complex patterns. "
        "Comparing them shows whether the extra complexity pays off. Because missing a churner costs more than a "
        "false alarm, Recall is the main metric rather than Accuracy.",
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(metrics.style.format("{:.3f}").highlight_max(axis=0, color="#d4edda"))
    with col2:
        how_to_read("Green marks the better model on each metric. Recall = share of actual churners the model "
                    "caught. Precision = share of flagged customers who really churned. F1 balances the two. "
                    "ROC-AUC = how well the model ranks churners above non-churners (0.5 = random, 1 = perfect).")
    finding(
        f"Logistic Regression catches more churners (Recall {lr['Recall (churn)']:.1%} vs "
        f"{rf['Recall (churn)']:.1%}) and ranks them better (ROC-AUC {lr['ROC-AUC']:.3f} vs {rf['ROC-AUC']:.3f}). "
        f"Random Forest has higher Accuracy ({rf['Accuracy']:.1%} vs {lr['Accuracy']:.1%}) and Precision.",
        "Logistic Regression is the recommended model. A missed churner is lost revenue, while a false alarm only "
        "costs a retention offer — and the simpler model is also easier to explain to the business.",
    )

    st.divider()

    # --- Step 4: Confusion matrix + ROC ------------------------------------
    st.subheader("Step 4 — Where do the models get it right and wrong?")
    explain(
        "Broke each model's test predictions into a confusion matrix, and plotted the ROC curve across all "
        "possible decision thresholds.",
        "Summary metrics hide the actual counts. The confusion matrix shows exactly how many churners were caught "
        "or missed, and the ROC curve shows whether one model is better regardless of where the cut-off is set.",
    )
    cms = results["confusion_matrices"]
    cols = st.columns(3)
    for col, (name, cm) in zip(cols[:2], cms.items()):
        with col:
            fig, ax = plt.subplots(figsize=(3, 2.6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, annot_kws={"size": 8},
                        xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.tick_params(labelsize=7)
            st.pyplot(fig)
    with cols[2]:
        how_to_read("Rows are what actually happened, columns are what the model predicted. Bottom-right = "
                    "churners caught; bottom-left = churners missed; top-right = false alarms.")

    col1, col2 = st.columns([2, 1])
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
    with col2:
        how_to_read("Each curve shows the trade-off between catching churners (up) and raising false alarms "
                    "(right). A curve closer to the top-left corner is better; the dashed line is random guessing.")

    lr_cm, rf_cm = cms["Logistic Regression"], cms["Random Forest"]
    finding(
        f"Of {n_churn_test:,} actual churners, Logistic Regression caught {lr_cm[1, 1]:,} and missed {lr_cm[1, 0]:,}; "
        f"Random Forest caught {rf_cm[1, 1]:,} and missed {rf_cm[1, 0]:,}. Logistic Regression raised more false "
        f"alarms ({lr_cm[0, 1]:,} vs {rf_cm[0, 1]:,}). Its ROC curve sits slightly above Random Forest's.",
        f"Choosing Logistic Regression saves {lr_cm[1, 1] - rf_cm[1, 1]:,} more at-risk customers in this test set, "
        f"at the cost of {lr_cm[0, 1] - rf_cm[0, 1]:,} extra retention contacts to customers who would have stayed.",
    )

    st.divider()

    # --- Step 5: Cross-validation ------------------------------------------
    st.subheader("Step 5 — Is the result stable? (5-fold cross-validation)")
    explain(
        "Re-ran the full pipeline (scaling → SMOTE → model) with 5-fold stratified cross-validation, so every "
        "customer is used for testing exactly once.",
        "The results above come from a single 80/20 split, which could be lucky or unlucky. Cross-validation "
        "repeats the test 5 times on different splits; a small spread means the result can be trusted.",
    )
    cv_summary = []
    for name, scores in robustness["cv_results"].items():
        row = {"Model": name}
        for metric in CV_SCORING:
            vals = scores[f"test_{metric}"]
            row[f"{metric} (mean)"] = vals.mean()
            row[f"{metric} (std)"] = vals.std()
        cv_summary.append(row)
    cv_df = pd.DataFrame(cv_summary).set_index("Model")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(cv_df.round(3))
    with col2:
        how_to_read("\"mean\" is the average score across the 5 folds; \"std\" is how much it varied between "
                    "folds. A small std means the model performs consistently.")
    cv_lr, cv_rf = cv_df.loc["Logistic Regression"], cv_df.loc["Random Forest"]
    finding(
        f"Logistic Regression averages Recall {cv_lr['recall (mean)']:.3f} ± {cv_lr['recall (std)']:.3f} and "
        f"ROC-AUC {cv_lr['roc_auc (mean)']:.3f} ± {cv_lr['roc_auc (std)']:.3f}, versus Recall "
        f"{cv_rf['recall (mean)']:.3f} ± {cv_rf['recall (std)']:.3f} for Random Forest.",
        "The spread is small and Logistic Regression keeps a wide Recall lead, so the model choice from Step 3 was "
        "not a fluke of one particular split.",
    )

    st.divider()

    # --- Step 6: Learning curve --------------------------------------------
    st.subheader("Step 6 — Would more data help? (learning curve)")
    explain(
        "Trained each model on increasing amounts of data (10% to 100% of the training folds) and compared the "
        "Recall on the training data with the Recall on unseen validation data.",
        "This shows whether the model is limited by the amount of data or by the model itself, and whether it is "
        "overfitting — memorizing the training data instead of learning patterns that generalize.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    gaps = {}
    for ax, (name, color) in zip(axes, [("Logistic Regression", "#4C72B0"), ("Random Forest", "#55A868")]):
        sizes, train_scores, val_scores = robustness["learning_curve_results"][name]
        train_mean, val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)
        gaps[name] = (train_mean[-1], val_mean[-1])
        ax.plot(sizes, train_mean, "o--", color=color, alpha=0.5, label="Train", markersize=3)
        ax.plot(sizes, val_mean, "o-", color=color, label="Validation", markersize=3)
        ax.axvline(x=len(df) * 0.8, linestyle=":", color="gray")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Training samples", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("Recall (churn)", fontsize=8)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        how_to_read("Dashed = Recall on data the model trained on; solid = Recall on unseen data. Lines that "
                    "converge and flatten mean the model has enough data. A large, lasting gap means overfitting.")
    (lr_train, lr_val), (rf_train, rf_val) = gaps["Logistic Regression"], gaps["Random Forest"]
    finding(
        f"Logistic Regression's train and validation Recall converge ({lr_train:.2f} vs {lr_val:.2f}) and flatten "
        f"out. Random Forest scores {rf_train:.2f} on training data but only {rf_val:.2f} on unseen data.",
        "The 80/20 split already provides enough training data, so collecting more rows would not change the "
        "conclusion. Random Forest's weaker Recall comes from overfitting, not from a lack of data.",
    )

    st.divider()

    # --- Step 7: Feature importance ----------------------------------------
    st.subheader("Step 7 — Which features drive the predictions?")
    explain(
        "Ranked features by the size of their Logistic Regression coefficient and by Random Forest's feature "
        "importance, then cross-checked both against the EDA statistics.",
        "A model is only useful to the business if we understand why it flags a customer. Features that the EDA "
        "and both models agree on are the most reliable drivers to act on.",
    )
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
    how_to_read("Left: red bars increase churn risk, blue bars reduce it; longer bars have a bigger effect. "
                "Right: longer bars mean the feature was used more often to split customers, but it doesn't "
                "show direction.")
    coef = results["coef_df"].set_index("feature")["coefficient"]
    finding(
        "Contract, OnlineSecurity/TechSupport and MonthlyCharges rank high in both models and in the EDA, "
        f"confirming them as the main drivers. One result needs care: Fiber optic has a negative coefficient "
        f"({coef['InternetService_Fiber optic']:.2f}) even though Fiber customers churn the most, because its "
        f"effect overlaps with MonthlyCharges ({coef['MonthlyCharges']:+.2f}) — Fiber plans cost more.",
        "Retention actions should focus on the confirmed drivers: moving customers to longer contracts and "
        "bundling security/support add-ons. Individual coefficients should not be read in isolation when "
        "features overlap (multicollinearity).",
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
