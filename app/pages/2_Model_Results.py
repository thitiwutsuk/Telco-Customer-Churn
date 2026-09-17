import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data import load_clean_data  # noqa: E402
from src.model import CV_SCORING, get_evaluation_results, get_robustness_results  # noqa: E402

st.set_page_config(page_title="Model Results — Telco Churn", page_icon="🤖", layout="wide")
sns.set_style("whitegrid")

df = load_clean_data()
results = get_evaluation_results(df)

st.title("Model Results")
st.caption(
    f"Logistic Regression vs. Random Forest, evaluated on a held-out test set "
    f"({results['n_test']:,} customers, {results['n_churn_test']:,} of whom actually churned) — "
    "mirrors Step 16 of `notebooks/telco_churn_modeling.ipynb`."
)

st.subheader("Metric comparison")
st.dataframe(results["metrics_df"].style.format("{:.3f}").highlight_max(axis=0, color="#d4edda"))
st.markdown(
    "**Random Forest** wins on Accuracy/Precision (fewer false alarms). **Logistic Regression** wins "
    "on Recall/ROC-AUC (catches more actual churners) — the costlier error to avoid in a churn problem "
    "is missing a real churner, so **Logistic Regression is the recommended model**."
)

st.divider()

st.subheader("Confusion matrices")
cols = st.columns(2)
for col, (name, cm) in zip(cols, results["confusion_matrices"].items()):
    with col:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

st.divider()

st.subheader("ROC curve")
fig, ax = plt.subplots(figsize=(6, 5))
for name, color in [("Logistic Regression", "#4C72B0"), ("Random Forest", "#55A868")]:
    fpr, tpr, auc_score = results["roc_data"][name]
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=color, linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess (AUC = 0.5)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

st.divider()

with st.expander("Robustness checks — 5-fold Cross-Validation & Learning Curve (Steps 16.4-16.5)"):
    st.markdown(
        "The comparison above uses a single 80/20 split. These checks confirm it wasn't a fluke and "
        "that the training data is sufficient."
    )
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
    st.caption(
        "Std stays small (≤ ~2 points) across folds — the Recall gap between the models is actually "
        "*wider* on average than the single-split numbers above, reinforcing the Logistic Regression pick."
    )

    st.markdown("**Learning Curve (Recall)**")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, color) in zip(axes, [("Logistic Regression", "#4C72B0"), ("Random Forest", "#55A868")]):
        sizes, train_scores, val_scores = robustness["learning_curve_results"][name]
        ax.plot(sizes, train_scores.mean(axis=1), "o--", color=color, alpha=0.5, label="Train")
        ax.plot(sizes, val_scores.mean(axis=1), "o-", color=color, label="Validation")
        ax.axvline(x=len(df) * 0.8, linestyle=":", color="gray")
        ax.set_title(name)
        ax.set_xlabel("Training samples")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Recall (churn)")
    st.pyplot(fig)
    st.caption(
        "Logistic Regression's validation recall plateaus well before the current 80% training size "
        "(dotted line). Random Forest's ~100% train score vs. ~58% validation score at every size is a "
        "model-complexity overfitting signature, not a data-size problem."
    )

st.divider()

st.subheader("Feature importance (Step 17)")
top_n = st.slider("Show top N features", 5, 20, 10)
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Logistic Regression — coefficients**")
    top_coef = results["coef_df"].head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, top_n * 0.35 + 1))
    colors = ["#C44E52" if c > 0 else "#4C72B0" for c in top_coef["coefficient"]]
    ax.barh(top_coef["feature"], top_coef["coefficient"], color=colors)
    ax.set_xlabel("Coefficient (red = increases churn odds)")
    st.pyplot(fig)
with col2:
    st.markdown("**Random Forest — feature importances**")
    top_imp = results["importance_df"].head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, top_n * 0.35 + 1))
    ax.barh(top_imp["feature"], top_imp["importance"], color="#55A868")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

st.caption(
    "`Contract`, `OnlineSecurity`/`TechSupport`, and `MonthlyCharges` are confirmed as churn drivers by "
    "both models and by the EDA statistical tests. `InternetService_Fiber optic` has a *negative* "
    "Logistic Regression coefficient despite having the highest raw churn rate in EDA — a multicollinearity "
    "artifact with `MonthlyCharges` (diagnosed in Step 17.3), not a sign Fiber optic actually protects "
    "against churn."
)
