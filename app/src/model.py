"""Model training and evaluation for the Streamlit app.

Reproduces the methodology from `notebooks/telco_churn_modeling.ipynb` (Steps 12-17) so the app's
numbers match the notebook's, computed live and cached so it only runs once per app session.
"""

import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, learning_curve, train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLS, NUMERIC_COLS, RANDOM_STATE

CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _build_encoded_features(df: pd.DataFrame):
    """One-hot encode categorical columns (drop_first, matching Step 12.2) and combine with numeric."""
    X_cat = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True).astype(int)
    X = pd.concat([df[NUMERIC_COLS], X_cat], axis=1)
    y = df["Churn_numeric"]
    return X, y


@st.cache_resource(show_spinner="Training Logistic Regression & Random Forest (Steps 12-16)...")
def get_evaluation_results(_df: pd.DataFrame):
    """Reproduces Steps 12-17: 80/20 split, SMOTE, train both models, evaluate on the held-out test set."""
    X, y = _build_encoded_features(_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test = X_train.copy(), X_test.copy()

    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_train_smote, y_train_smote)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE).fit(X_train_smote, y_train_smote)
    models = {"Logistic Regression": log_reg, "Random Forest": rf}

    metrics_rows, confusion_matrices, roc_data = [], {}, {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (churn)": precision_score(y_test, y_pred),
            "Recall (churn)": recall_score(y_test, y_pred),
            "F1 (churn)": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_proba))

    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")

    coef_df = (
        pd.DataFrame({"feature": X.columns, "coefficient": log_reg.coef_[0]})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
    )
    importance_df = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
    )

    return {
        "metrics_df": metrics_df,
        "confusion_matrices": confusion_matrices,
        "roc_data": roc_data,
        "coef_df": coef_df,
        "importance_df": importance_df,
        "n_test": len(y_test),
        "n_churn_test": int(y_test.sum()),
    }


@st.cache_resource(show_spinner="Running 5-fold cross-validation and learning curve (Steps 16.4-16.5)...")
def get_robustness_results(_df: pd.DataFrame):
    """Reproduces Steps 16.4 (StratifiedKFold CV) and 16.5 (learning curve)."""
    X, y = _build_encoded_features(_df)

    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), NUMERIC_COLS)], remainder="passthrough"
    )
    pipelines = {
        "Logistic Regression": ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {
        name: cross_validate(pipe, X, y, cv=skf, scoring=CV_SCORING)
        for name, pipe in pipelines.items()
    }

    train_sizes_frac = np.linspace(0.1, 1.0, 8)
    learning_curve_results = {}
    for name, pipe in pipelines.items():
        sizes, train_scores, val_scores = learning_curve(
            pipe, X, y, train_sizes=train_sizes_frac, cv=skf, scoring="recall", n_jobs=-1
        )
        learning_curve_results[name] = (sizes, train_scores, val_scores)

    return {"cv_results": cv_results, "learning_curve_results": learning_curve_results}


@st.cache_resource(show_spinner="Training the production model on the full dataset...")
def get_production_pipeline(_df: pd.DataFrame):
    """A single deployable pipeline (raw features in, churn probability out) trained on ALL data.

    Uses OneHotEncoder (handle_unknown='ignore') instead of pandas get_dummies so it can safely
    encode new, single-row input from the predictor page. Chooses Logistic Regression since it was
    the model selected in Step 16/18. Trained on the full dataset since Step 16.5's learning curve
    showed the 80/20 split already had more than enough training data — retraining on 100% squeezes
    out a bit more for the deployed model without a held-out set to worry about anymore.
    """
    X = _df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = _df["Churn_numeric"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    pipeline = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X, y)
    return pipeline
