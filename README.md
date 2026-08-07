# Telco Customer Churn — Exploratory Data Analysis

This project performs a deep Exploratory Data Analysis (EDA) on the **IBM Telco Customer Churn**
dataset to find which customer factors are associated with churn (customers leaving the service).
The focus is on statistical and descriptive analysis — **no predictive modeling** is included.

## Project Files

| File | Purpose |
|---|---|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original raw dataset |
| `telco_churn_eda.ipynb` | Main notebook containing the full EDA, organized into steps per the methodology below |
| `requirements.txt` | Python packages required |
| `.venv/` | Virtual environment (isolates installed libraries from the system Python; no need to edit/commit) |
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

The analysis is organized into 11 steps, each corresponding to one section in `telco_churn_eda.ipynb`.
Steps are done in order, since each step builds understanding or cleans data that the next step depends on.

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

### ⬜ Step 3: Univariate Analysis — Target Variable
Examine the overall Churn proportion (base rate), since every subgroup comparison in later steps will be
benchmarked against this rate.

### ⬜ Step 4: Univariate Analysis — All Features
Examine the distribution of every variable (categorical: countplot/proportions, numeric:
histogram/boxplot/`describe()`) before relating them to the target.

### ⬜ Step 5: Bivariate Analysis — Categorical Features vs Churn
Compare churn *rate* (not raw counts) across categories using crosstabs, and confirm statistical
significance with a **Chi-square test**.

### ⬜ Step 6: Bivariate Analysis — Numeric Features vs Churn
Boxplot/violin plots split by Churn, tested with the **Mann-Whitney U test** (non-parametric, since
normality of the data isn't assumed).

### ⬜ Step 7: Correlation Analysis
Heatmap of numeric variables to check for baseline multicollinearity.

### ⬜ Step 8: Multivariate / Interaction Analysis
Analyze interactions such as Contract × InternetService, Contract × tenure bucket, to identify at-risk
customer "personas."

### ⬜ Step 9: Tenure-based Retention Curve
Churn rate across tenure ranges, to see at which point in the customer lifecycle churn is highest.

### ⬜ Step 10: Customer Risk Segmentation
Synthesize descriptive, rule-based high-risk segments from the results of Steps 5-9.

### ⬜ Step 11: Summary of Key Insights & Recommendations
Summarize the main insights, risk segments, and limitations of the analysis (correlation ≠ causation).

> The full plan with the rationale behind each step is available at
> `/Users/athens/.claude/plans/insight-data-toasty-blum.md`

## Current Status

Completed through **Step 2** — the data is clean and ready for Step 3 onward.
