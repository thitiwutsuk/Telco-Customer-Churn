# Telco Customer Churn — EDA & Predictive Modeling

![Python](https://img.shields.io/badge/python-3.9-blue)
![Pandas](https://img.shields.io/badge/pandas-2.3-150458)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Progress](https://img.shields.io/badge/progress-step%2016%20of%2018-yellow)

This project analyzes the **IBM Telco Customer Churn** dataset to find which customer factors are
associated with churn (customers leaving the service). It starts with a deep Exploratory Data
Analysis (EDA) — statistical and descriptive — and then builds predictive models (Logistic
Regression and Random Forest) to quantify churn drivers and cross-check the EDA findings.

## Project Structure

```
Telco Customer Churn/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original raw dataset (never modified)
│   └── processed/
│       └── telco_churn_clean.csv                  # Cleaned dataset (output of eda notebook, input to modeling notebook)
├── notebooks/
│   ├── telco_churn_eda.ipynb                      # Notebook 1: Steps 1-11 (data cleaning + EDA)
│   └── telco_churn_modeling.ipynb                 # Notebook 2: Steps 12-18 (feature engineering + ML modeling)
├── requirements.txt                               # Python dependencies, version-pinned (EDA + ML)
├── LICENSE                                        # MIT license
├── .venv/                                         # Virtual environment (local only, not committed)
├── .vscode/
│   └── settings.json                              # Hides .venv/ from the VS Code explorer & search
├── .claude/
│   └── settings.json                              # Shared Claude Code permission allowlist
├── .gitignore
└── README.md                                      # This file
```

| File / Folder | Purpose |
|---|---|
| `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original raw dataset — treated as immutable, never edited in place |
| `data/processed/telco_churn_clean.csv` | Cleaned dataset saved at the end of `notebooks/telco_churn_eda.ipynb`, loaded at the start of `notebooks/telco_churn_modeling.ipynb` — avoids duplicating the cleaning code across both notebooks |
| `notebooks/telco_churn_eda.ipynb` | Steps 1-11: data cleaning and exploratory data analysis |
| `notebooks/telco_churn_modeling.ipynb` | Steps 12-18: feature engineering and predictive modeling (continues from the EDA notebook) |
| `requirements.txt` | Python packages required, pinned to the exact tested versions (EDA: pandas/numpy/matplotlib/seaborn/scipy; ML: scikit-learn/imbalanced-learn) |
| `LICENSE` | MIT license |
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

Open/edit the notebooks with Jupyter or VS Code (select the kernel from `.venv`).

Run both notebooks non-interactively, **in order**, from inside `notebooks/` (the modeling notebook loads
`data/processed/telco_churn_clean.csv`, which the EDA notebook produces — it must be run first at least once):
```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace telco_churn_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace telco_churn_modeling.ipynb
```

## Methodology

The analysis is organized into 18 steps, split across two notebooks in `notebooks/`: **Steps 1-11**
(descriptive EDA) are in `telco_churn_eda.ipynb`, and **Steps 12-18** (predictive modeling) are in
`telco_churn_modeling.ipynb`. The EDA notebook saves its cleaned dataframe to
`data/processed/telco_churn_clean.csv` at the end, which the modeling notebook loads at the start — this
avoids duplicating the data-cleaning code across both files. Steps are done in order, since each step
builds understanding or cleans/prepares data that the next step depends on.

**Note on writing style:** Steps 1-7 explain every concept in full each time (aimed at readers with no
programming/stats background). After reviewing that this got repetitive, Steps 8 onward keep the same
beginner-friendly quality for genuinely new content, but no longer re-explain concepts already covered
earlier (e.g., why to use rates instead of raw counts, what a p-value means) — they reference the step
where it was first explained instead. Steps 1-7 were left as-is rather than rewritten.

### ✅ Step 1: Data Loading & Structural Overview *(done)*
Load the data and inspect its basic structure (`shape`, `dtypes`, `.info()`, `.head()`, `.isnull().sum()`).
Before any analysis, it's essential to understand the "shape" of the data first — each column's data type
(categorical/numeric/identifier) and whether pandas detects any missing values.

**Findings:**
- Dataset has 7,043×21 rows/columns, as expected
- `TotalCharges` was read as `object` even though it should be numeric
- `.isnull()` reported 0 missing values even though `TotalCharges` had hidden blanks (they were empty strings, not `NaN`)

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

**Findings:**
- No = 73.4% (5,163 customers), Yes = 26.6% (1,869 customers) — this is the base rate every later subgroup churn rate gets compared against
- Chart labels are in Thai; `plt.rcParams['font.family'] = 'Tahoma'` was set to fix Thai glyphs not rendering with matplotlib's default font, confirmed working on re-run

### ✅ Step 4: Univariate Analysis — All Features *(done)*
Examine the distribution of every variable (categorical: countplot/proportions, numeric:
histogram/boxplot/`describe()`) before relating them to the target.

**Findings:**
- 16 categorical + 3 numeric features surveyed
- `Contract` skews heavily toward Month-to-month (55.1%)
- `InternetService`/`PaymentMethod` have 3-4 categories (not just Yes/No)
- Several service columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) carry a "No internet service" category (21.6%) distinct from a plain "No" — must be kept separate when building crosstabs in Step 5
- `TotalCharges` is right-skewed (mean 2,283 vs. median 1,397)
- `tenure` has a bimodal/U-shaped distribution (many very new and many very long-tenured customers)
- `MonthlyCharges` is also bimodal (a low cluster around customers with no internet service, a high cluster around those with internet + add-ons)
- No IQR outliers found in any of the 3 numeric columns

### ✅ Step 5: Bivariate Analysis — Categorical Features vs Churn *(done)*
Compare churn *rate* (not raw counts) across categories using crosstabs, and confirm statistical
significance with a **Chi-square test**.

**Findings:**
- 14 of 16 categorical features are statistically significant (p < 0.05); only `gender` and `PhoneService` show no relationship with churn
- `Contract` = Month-to-month: 42.7% churn vs. 2.8% for Two year
- `InternetService` = Fiber optic: 41.9% churn
- No `OnlineSecurity`: 41.8% churn; no `TechSupport`: 41.6% churn
- `PaymentMethod` = Electronic check: 45.3% churn (highest of all payment methods)
- `PaperlessBilling` = Yes: 33.6% churn vs. 16.4% for No
- `SeniorCitizen` = Yes: 41.7% churn vs. 23.7% for No
- These high-risk categories cluster together (new, month-to-month, fiber, no add-on services) — an early signal of the risk segment defined in Step 10
- Customers with no internet service at all show a low 7.4% churn rate across every add-on-service column

### ✅ Step 6: Bivariate Analysis — Numeric Features vs Churn *(done)*
Boxplot/violin plots split by Churn, tested with the **Mann-Whitney U test** (non-parametric, since
normality of the data isn't assumed).

**Findings:**
- All 3 numeric features are statistically significant (p < 0.05)
- Median `tenure`: 10 months for churned customers vs. 38 months for retained ones (churners leave early)
- Median `MonthlyCharges`: higher for churners (79.65 vs. 64.45) — they pay more per month
- Median `TotalCharges`: lower for churners (703.55 vs. 1,683.60), mostly reflecting their shorter tenure rather than being a new independent signal
- Combined picture: churners tend to be newer customers paying above-average monthly rates, consistent with the Step 5 finding that Fiber optic (pricier) and month-to-month contracts drive the highest churn

### ✅ Step 7: Correlation Analysis *(done)*
Heatmap of numeric variables to check for baseline multicollinearity.

**Findings:**
- `tenure`↔`TotalCharges` correlate strongly (0.83) — expected from how `TotalCharges` accumulates, not new signal
- `MonthlyCharges`↔`TotalCharges` also correlate (0.65) — same reason, not new signal
- Against `Churn`: `tenure` is the strongest and most independent numeric predictor (-0.35)
- `MonthlyCharges` vs. `Churn`: weakly positive (+0.19)
- `TotalCharges` vs. `Churn` (-0.20) is mostly a reflection of its strong tie to `tenure` rather than an independent effect — a multicollinearity note to carry into the ML modeling phase (Steps 12+)

### ✅ Step 8: Multivariate / Interaction Analysis *(done)*
Analyze interactions such as Contract × InternetService, Contract × tenure bucket, to identify at-risk
customer "personas."

**Findings:**
- Risk compounds when factors combine: Month-to-month + Fiber optic = 54.6% churn vs. Two year + No internet = 0.8% (~70x difference)
- Month-to-month customers in their first 12 months churn at 51.4%, dropping steadily to 22.2% by months 61-72
- Two year customers stay near 0-4% churn regardless of tenure
- Highest-risk profile: month-to-month contract + Fiber optic + under 1 year tenure

### ✅ Step 9: Tenure-based Retention Curve *(done)*
Churn rate across tenure ranges, to see at which point in the customer lifecycle churn is highest.

**Findings:**
- Churn rate peaks at month 1 (62.0%)
- Stays high through months 1-6 (avg ~48.7%)
- Declines steadily to just 8.0% by months 60-72 (1.7% at month 72)
- Months 1-6 are the critical retention window — customers who survive past it are increasingly unlikely to churn

### ✅ Step 10: Customer Risk Segmentation *(done)*
Synthesize descriptive, rule-based high-risk segments from the results of Steps 5-9.

**Findings:**
- Defined segment: Month-to-month + Fiber optic + tenure < 12 months + no OnlineSecurity/TechSupport
- Size: 737 customers (10.5% of all customers)
- Churn rate: 73.5% (2.8x the 26.6% base rate)
- This segment alone accounts for 29.0% of all company-wide churn
- ~59,419 baht/month in at-risk revenue — a small, high-leverage group for retention efforts

### ✅ Step 11: Summary of Key Insights & Recommendations *(done)*
Summarize the main insights, risk segments, and limitations of the analysis (correlation ≠ causation).
Closes out the EDA phase (Steps 1-11) before moving into predictive modeling in Step 12.

**Contents:**
- A consolidated findings table pulling together Steps 3-10
- Business recommendations: target the 1-6 month retention window, incentivize long-term contracts, bundle security/support services for new fiber customers, review the Electronic check payment experience
- Explicit limitations: correlation ≠ causation, no time dimension in the data, hand-picked segment thresholds

*(Steps 12-18 below are in `telco_churn_modeling.ipynb`)*

### ✅ Step 12: Feature Engineering & Encoding for ML *(done)*
Encode `Churn` to 1/0, one-hot encode categorical features (`drop_first=True` to avoid the dummy
variable trap), split into train/test (`stratify=y`, 80/20, `random_state=42`), and scale numeric
features with `StandardScaler` fit on the training set only (to avoid data leakage).

**Result:**
- 16 categorical columns expanded to 27 one-hot columns
- Final feature matrix `X`: 7,032 rows × 30 columns (27 encoded categorical + 3 scaled numeric)
- Train (5,625 rows) and test (1,407 rows) both preserved the 26.6% churn rate via `stratify=y`
- `tenure_bucket` from Step 8 was deliberately excluded from `X` as redundant with `tenure`
- **Bug fix:** `pd.get_dummies()` on pandas 2.x returns `bool` columns rather than 0/1 integers; left mixed
  with the `float64` scaled numeric columns, this silently produces an `object`-dtype array when converted
  for scikit-learn, which caused numerical overflow warnings during model training in Step 14. Fixed by
  chaining `.astype(int)` onto the one-hot encoding step so the whole feature matrix is uniformly numeric.

### ✅ Step 13: Handle Class Imbalance with SMOTE *(done)*
Apply SMOTE oversampling to the **training set only** so the model sees enough churn examples to learn
from, while the test set keeps the real-world ~26.6% churn proportion for fair evaluation.

**Result:**
- Before SMOTE: training set had 4,130 non-churn vs. 1,495 churn examples
- After SMOTE: both classes balanced to 4,130 each (synthetic churn examples generated, not real customers)
- `X_test`/`y_test` were left untouched, still reflecting the real ~26.6% churn rate for fair evaluation in Step 16

### ✅ Step 14: Train Model 1 — Logistic Regression *(done)*
An interpretable baseline model — its coefficients translate directly into churn odds, easy to explain
to a business audience.

**Result:**
- Trained on the SMOTE-balanced training set, predicted on the untouched test set (1,407 customers)
- Predicted 536 customers as churn vs. 374 actual churners — a reasonable sanity check (detailed
  precision/recall/F1/ROC-AUC comparison against Random Forest happens in Step 16)
- Verified the fitted model is numerically sound: finite coefficients (range -5.2 to 7.2), predicted
  probabilities all within a valid [0.001, 0.999] range, converged in ~68 iterations (well under the
  `max_iter=1000` cap) — confirms the Step 12 dtype fix and the RuntimeWarning suppression below are safe

### ✅ Step 15: Train Model 2 — Random Forest *(done)*
A model that captures non-linear relationships and feature interactions, generally stronger in
practice; used to see whether the added complexity is worth it over the linear baseline.

**Result:**
- Trained (`n_estimators=200, random_state=42`) on the same SMOTE-balanced training set as Step 14, predicted on the same untouched test set
- Predicted 437 customers as churn vs. 374 actual churners
- No warnings during training (unlike Logistic Regression) — tree-based models aren't affected by the linear-optimizer numerical issues from Step 14
- Both models' predictions now ready for a rigorous side-by-side comparison in Step 16

### ✅ Step 16: Model Evaluation & Comparison *(done)*
Evaluate both models on the untouched (non-SMOTE) test set using confusion matrix, precision/recall/F1
(for the churn class specifically), and ROC-AUC — accuracy alone is misleading on imbalanced data.

**Result:**

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 74.6% | **77.2%** |
| Precision (churn) | 51.5% | **56.1%** |
| Recall (churn) | **73.8%** | 65.5% |
| F1 (churn) | **0.607** | 0.604 |
| ROC-AUC | **0.827** | 0.816 |

- Random Forest wins on accuracy/precision (fewer false alarms); Logistic Regression wins on recall/ROC-AUC (catches more actual churners)
- For a churn problem, missing an actual churner (false negative) is costlier than a false alarm, so **Logistic Regression is the better choice here** despite being the simpler model
- A concrete case where added model complexity didn't translate into a better outcome for the business objective

### ⬜ Step 17: Feature Importance & Interpretation
Extract Logistic Regression coefficients and Random Forest `feature_importances_`, and cross-check them
against the statistical findings from Steps 5-6 (Chi-square/Mann-Whitney) for agreement or conflict.

### ⬜ Step 18: ML Summary & Final Business Recommendations
Tie the ML results together with the EDA findings into one coherent set of business recommendations,
noting limitations (SMOTE uses synthetic data; results are still correlational, not causal).

> The full plan with the rationale behind each step is available at
> `/Users/athens/.claude/plans/insight-data-toasty-blum.md`

## Current Status

Completed through **Step 16**. Steps 17-18 (remaining ML modeling phase) are not started yet.
