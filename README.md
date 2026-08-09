# Telco Customer Churn — EDA & Predictive Modeling

This project analyzes the **IBM Telco Customer Churn** dataset to find which customer factors are
associated with churn (customers leaving the service). It starts with a deep Exploratory Data
Analysis (EDA) — statistical and descriptive — and then builds predictive models (Logistic
Regression and Random Forest) to quantify churn drivers and cross-check the EDA findings.

## Project Structure

```
Telco Customer Churn/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original raw dataset
├── telco_churn_eda.ipynb                      # Main notebook: full EDA + ML modeling workflow
├── requirements.txt                           # Python dependencies (EDA + ML)
├── .venv/                                     # Virtual environment (local only, not committed)
├── .vscode/
│   └── settings.json                          # Hides .venv/ from the VS Code explorer & search
├── .claude/
│   └── settings.json                          # Shared Claude Code permission allowlist
├── .gitignore
└── README.md                                  # This file
```

| File / Folder | Purpose |
|---|---|
| `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original raw dataset |
| `telco_churn_eda.ipynb` | Main notebook containing the full EDA + modeling workflow, organized into steps per the methodology below |
| `requirements.txt` | Python packages required (EDA: pandas/numpy/matplotlib/seaborn/scipy; ML: scikit-learn/imbalanced-learn) |
| `.venv/` | Virtual environment (isolates installed libraries from the system Python; no need to edit/commit) |
| `.vscode/settings.json` | Editor config that hides `.venv/` from the file explorer and search |
| `.claude/settings.json` | Shared Claude Code permission allowlist (reduces repeated approval prompts) |
| `README.md` | This file |

## Dataset

- **Source:** IBM Sample Data Sets (also distributed on Kaggle as Telco Customer Churn)
- **Original size:** 7,043 rows × 21 columns (7,032 rows × 20 columns after cleaning — see Methodology Step 2)
- **Row unit:** one telecom customer per row
- **Target variable:** `Churn` (Yes/No) — whether the customer left the service in the latest period

### Column groups

1. **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
2. **Account information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
3. **Services subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
4. **Target:** `Churn`

### Data quality issues found and how they were handled (full detail in Methodology Steps 1-2)

- `TotalCharges` was read as a string because **11 rows** contained blank (whitespace) values — investigation confirmed all 11 rows have `tenure = 0` (brand-new customers who haven't reached their first billing cycle yet), so these rows were **dropped** since they carry no meaningful churn behavior to compare against other customers.
- `SeniorCitizen` was originally stored as `int` (0/1) and was converted to string `"No"/"Yes"` to match the format of the other flag columns.
- `customerID` was set as the dataframe index, since it's an identifier, not an analytical feature.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open/edit the notebook with Jupyter or VS Code (select the kernel from `.venv`).

Run the full notebook non-interactively (execute + save outputs back into the file):
```bash
jupyter nbconvert --to notebook --execute --inplace telco_churn_eda.ipynb
```

## Methodology

The analysis is organized into 18 steps, each corresponding to one section in `telco_churn_eda.ipynb`.
Steps 1-11 are descriptive EDA; Steps 12-18 build and interpret predictive models. Steps are done in
order, since each step builds understanding or cleans/prepares data that the next step depends on.

### ✅ Step 1: Data Loading & Structural Overview *(done)*
Load the data and inspect its basic structure (`shape`, `dtypes`, `.info()`, `.head()`, `.isnull().sum()`).
Before any analysis, it's essential to understand the "shape" of the data first — each column's data type
(categorical/numeric/identifier) and whether pandas detects any missing values.
**Findings:** the dataset has 7,043×21 rows/columns as expected; `TotalCharges` was read as `object` even
though it should be numeric; `.isnull()` reported 0 missing values even though `TotalCharges` had hidden
blanks (because they were empty strings, not `NaN`).

### ✅ Step 2: Data Cleaning & Type Correction *(done)*
- Converted `TotalCharges` to numeric with `pd.to_numeric(errors='coerce')` so unconvertible values become
  detectable `NaN`s.
- Always verify assumptions before dropping data: confirmed all 11 `NaN` rows have `tenure == 0` before dropping them.
- Normalized `SeniorCitizen` and set `customerID` as the index.
**Principle:** data cleaning must always "verify assumptions before deleting/modifying data" to avoid
silently introducing bias by dropping rows without understanding why they're missing.

### ✅ Step 3: Univariate Analysis — Target Variable *(done)*
Examine the overall Churn proportion (base rate), since every subgroup comparison in later steps will be
benchmarked against this rate.
**Findings:** No = 73.4% (5,163 customers), Yes = 26.6% (1,869 customers) — this is the base rate every
later subgroup churn rate gets compared against. Chart labels are in Thai; `plt.rcParams['font.family']
= 'Tahoma'` was set to fix Thai glyphs not rendering with matplotlib's default font, confirmed working
on re-run.

### ✅ Step 4: Univariate Analysis — All Features *(done)*
Examine the distribution of every variable (categorical: countplot/proportions, numeric:
histogram/boxplot/`describe()`) before relating them to the target.
**Findings:** 16 categorical + 3 numeric features surveyed. `Contract` skews heavily toward Month-to-month
(55.1%), `InternetService`/`PaymentMethod` have 3-4 categories (not just Yes/No), and several service
columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`,
`StreamingMovies`) carry a "No internet service" category (21.6%) distinct from a plain "No" — must be
kept separate when building crosstabs in Step 5. `TotalCharges` is right-skewed (mean 2,283 vs. median
1,397); `tenure` has a bimodal/U-shaped distribution (many very new and many very long-tenured customers);
`MonthlyCharges` is also bimodal (a low cluster around customers with no internet service, a high cluster
around those with internet + add-ons). No IQR outliers found in any of the 3 numeric columns.

### ✅ Step 5: Bivariate Analysis — Categorical Features vs Churn *(done)*
Compare churn *rate* (not raw counts) across categories using crosstabs, and confirm statistical
significance with a **Chi-square test**.
**Findings:** 14 of 16 categorical features are statistically significant (p < 0.05); only `gender` and
`PhoneService` show no relationship with churn. Highest-risk categories, all well above the 26.6% base
rate: `Contract` = Month-to-month (42.7% churn vs. 2.8% for Two year), `InternetService` = Fiber optic
(41.9%), no `OnlineSecurity` (41.8%) or `TechSupport` (41.6%), `PaymentMethod` = Electronic check
(45.3%, highest of all), `PaperlessBilling` = Yes (33.6% vs. 16.4%), and `SeniorCitizen` = Yes (41.7% vs.
23.7%). These high-risk categories cluster together (new, month-to-month, fiber, no add-on services) —
an early signal of the risk segment to be defined in Step 10. Customers with no internet service at all
show a low 7.4% churn rate across every add-on-service column.

### ✅ Step 6: Bivariate Analysis — Numeric Features vs Churn *(done)*
Boxplot/violin plots split by Churn, tested with the **Mann-Whitney U test** (non-parametric, since
normality of the data isn't assumed).
**Findings:** all 3 numeric features are statistically significant (p < 0.05). Median `tenure` is 10
months for churned customers vs. 38 months for retained ones (churners leave early). Median
`MonthlyCharges` is higher for churners (79.65 vs. 64.45) — they pay more per month. Median
`TotalCharges` is lower for churners (703.55 vs. 1,683.60), which mostly reflects their shorter tenure
rather than being a new independent signal. Combined picture: churners tend to be newer customers paying
above-average monthly rates, consistent with the Step 5 finding that Fiber optic (pricier) and
month-to-month contracts drive the highest churn.

### ✅ Step 7: Correlation Analysis *(done)*
Heatmap of numeric variables to check for baseline multicollinearity.
**Findings:** `tenure`↔`TotalCharges` correlate strongly (0.83) and `MonthlyCharges`↔`TotalCharges` also
correlate (0.65) — both expected from how `TotalCharges` accumulates, not new signal. Against `Churn`:
`tenure` is the strongest and most independent numeric predictor (-0.35), `MonthlyCharges` is weakly
positive (+0.19), and `TotalCharges`'s correlation with churn (-0.20) is mostly a reflection of its
strong tie to `tenure` rather than an independent effect — a multicollinearity note to carry into the ML
modeling phase (Steps 12+).

### ⬜ Step 8: Multivariate / Interaction Analysis
Analyze interactions such as Contract × InternetService, Contract × tenure bucket, to identify at-risk
customer "personas."

### ⬜ Step 9: Tenure-based Retention Curve
Churn rate across tenure ranges, to see at which point in the customer lifecycle churn is highest.

### ⬜ Step 10: Customer Risk Segmentation
Synthesize descriptive, rule-based high-risk segments from the results of Steps 5-9.

### ⬜ Step 11: Summary of Key Insights & Recommendations
Summarize the main insights, risk segments, and limitations of the analysis (correlation ≠ causation).

### ⬜ Step 12: Feature Engineering & Encoding for ML
Encode `Churn` to 1/0, one-hot encode categorical features (`drop_first=True` to avoid the dummy
variable trap), split into train/test (`stratify=y`, 80/20, `random_state=42`), and scale numeric
features with `StandardScaler` fit on the training set only (to avoid data leakage).

### ⬜ Step 13: Handle Class Imbalance with SMOTE
Apply SMOTE oversampling to the **training set only** so the model sees enough churn examples to learn
from, while the test set keeps the real-world ~26.6% churn proportion for fair evaluation.

### ⬜ Step 14: Train Model 1 — Logistic Regression
An interpretable baseline model — its coefficients translate directly into churn odds, easy to explain
to a business audience.

### ⬜ Step 15: Train Model 2 — Random Forest
A model that captures non-linear relationships and feature interactions, generally stronger in
practice; used to see whether the added complexity is worth it over the linear baseline.

### ⬜ Step 16: Model Evaluation & Comparison
Evaluate both models on the untouched (non-SMOTE) test set using confusion matrix, precision/recall/F1
(for the churn class specifically), and ROC-AUC — accuracy alone is misleading on imbalanced data.

### ⬜ Step 17: Feature Importance & Interpretation
Extract Logistic Regression coefficients and Random Forest `feature_importances_`, and cross-check them
against the statistical findings from Steps 5-6 (Chi-square/Mann-Whitney) for agreement or conflict.

### ⬜ Step 18: ML Summary & Final Business Recommendations
Tie the ML results together with the EDA findings into one coherent set of business recommendations,
noting limitations (SMOTE uses synthetic data; results are still correlational, not causal).

> The full plan with the rationale behind each step is available at
> `/Users/athens/.claude/plans/insight-data-toasty-blum.md`

## Current Status

Completed through **Step 7**. Steps 8-18 (remaining EDA + full ML modeling phase) are not started yet.
