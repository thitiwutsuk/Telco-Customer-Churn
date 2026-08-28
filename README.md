# Telco Customer Churn — EDA & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.3-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.0-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.12-9B59B6?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-3DA639?style=flat-square&logo=opensourceinitiative&logoColor=white)
![Progress](https://img.shields.io/badge/Progress-Complete%20(18%2F18)-3DA639?style=flat-square)

Exploratory data analysis and predictive modeling on the IBM Telco Customer Churn dataset. Identifies
which customer attributes drive churn, quantifies them with statistical testing, and validates the
findings with two classification models (Logistic Regression, Random Forest).

## Key Results

- **Best model:** Logistic Regression — 73.8% recall / 0.827 ROC-AUC on the churn class, outperforming
  Random Forest (65.5% recall) on the metric that matters most for retention (catching actual churners).
  Confirmed stable via 5-fold cross-validation and a learning curve.
- **Confirmed churn drivers** (agreed on by statistical tests and both models): `Contract` type,
  `OnlineSecurity`/`TechSupport` subscription, `MonthlyCharges`.
- **Highest-risk segment:** month-to-month + Fiber optic + tenure < 12 months + no security/support add-ons
  — 10.5% of customers, 73.5% churn rate, 29% of all churn.

## Project Structure

```
Telco Customer Churn/
├── data/
│   ├── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original dataset, never modified
│   └── processed/telco_churn_clean.csv            # Output of the EDA notebook, input to the modeling notebook
├── notebooks/
│   ├── telco_churn_eda.ipynb                      # Steps 1-11: cleaning + EDA
│   └── telco_churn_modeling.ipynb                 # Steps 12-18: feature engineering + ML
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

- **Source:** IBM Sample Data Sets (also on Kaggle as Telco Customer Churn)
- **Size:** 7,043 rows × 21 columns (7,032 × 20 after cleaning)
- **Target:** `Churn` (Yes/No)
- **Features:** demographics, account info (tenure, contract, billing), and subscribed services
- **Cleaning:** `TotalCharges` coerced from string to numeric; 11 rows with blank values (all
  `tenure == 0`, brand-new customers) dropped; `SeniorCitizen` normalized to Yes/No

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run both notebooks in order from inside `notebooks/` (the modeling notebook depends on the cleaned
CSV the EDA notebook produces):
```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace telco_churn_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace telco_churn_modeling.ipynb
```

## Methodology

18 steps across two notebooks — Steps 1-11 (EDA) in `telco_churn_eda.ipynb`, Steps 12-18 (ML) in
`telco_churn_modeling.ipynb`.

| Step | Focus | Key Result |
|---|---|---|
| 1-2 | Load & clean | Fixed `TotalCharges` dtype; dropped 11 zero-tenure rows |
| 3 | Target distribution | Base churn rate: 26.6% |
| 4 | Univariate analysis | 16 categorical + 3 numeric features surveyed |
| 5 | Categorical vs. churn (Chi-square) | 14/16 features significant (p < 0.05); month-to-month (42.7%), Fiber optic (41.9%), electronic check (45.3%) show highest churn |
| 6 | Numeric vs. churn (Mann-Whitney U) | All 3 features significant; churners have shorter tenure and higher monthly charges |
| 7 | Correlation analysis | `tenure`↔`TotalCharges` r = 0.83 — multicollinearity flagged for the ML phase |
| 8 | Interaction analysis | Month-to-month + Fiber optic = 54.6% churn vs. 0.8% for the lowest-risk combo |
| 9 | Retention curve | Churn peaks at month 1 (62.0%); critical retention window is months 1-6 |
| 10 | Risk segmentation | High-risk segment defined: 10.5% of customers, 73.5% churn rate, 29% of all churn |
| 11 | EDA summary | Consolidated findings + initial business recommendations |
| 12 | Feature engineering | One-hot encoding, stratified 80/20 split, `StandardScaler` fit on train only |
| 13 | Class imbalance (SMOTE) | Training set balanced to 4,130/4,130; test set left at the real 26.6% rate |
| 14-15 | Model training | Logistic Regression + Random Forest, both on the SMOTE-balanced set |
| 16 | Evaluation & robustness | Logistic Regression wins on recall/ROC-AUC; confirmed via 5-fold Stratified CV (16.4) and learning curve (16.5) |
| 17 | Feature importance | Cross-checked coefficients/importances against EDA statistics — 2 conflicts found and diagnosed via multicollinearity |
| 18 | Business summary | Final recommendations + documented limitations |

## Business Recommendations

- Target the 1-6 month retention window with proactive outreach
- Incentivize long-term contracts over month-to-month
- Bundle security/support services for new Fiber optic customers
- Review the Electronic check payment experience
- Use the model's churn probability score to prioritize retention outreach

## Limitations

- Correlation, not causation — findings describe association, not proven cause
- No time dimension in the data (single snapshot)
- SMOTE-balanced training examples are synthetic, not real customers
- No hyperparameter tuning; risk-segment thresholds in Step 10 are hand-picked, not optimized

## Status

- [x] Stage 1 — Data loading & cleaning (Steps 1-2)
- [x] Stage 2 — Exploratory data analysis (Steps 3-11)
- [x] Stage 3 — Feature engineering for ML (Step 12)
- [x] Stage 4 — Class imbalance handling with SMOTE (Step 13)
- [x] Stage 5 — Model training (Steps 14-15)
- [x] Stage 6 — Model evaluation & robustness checks (Step 16, incl. 16.4-16.5)
- [x] Stage 7 — Feature importance & cross-checking (Step 17)
- [x] Stage 8 — Business summary & recommendations (Step 18)

Both notebooks run end-to-end with no errors or warnings.
