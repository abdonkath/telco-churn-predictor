"""Train and export the frozen 23-feature XGBoost model for Streamlit.

Run from the repository root:
    python scripts/train_xgboost_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.preprocessing import FEATURE_COLUMNS, preprocess_customers, validate_feature_columns


DATA = ROOT / "telco-data"
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
RANDOM_STATE = 47
FIXED_THRESHOLD = 0.60


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train = pd.read_csv(DATA / "X_train.csv")
    x_test = pd.read_csv(DATA / "X_test.csv")
    y_train = pd.read_csv(DATA / "y_train.csv")["Churn"].map({"No": 0, "Yes": 1})
    y_test = pd.read_csv(DATA / "y_test.csv")["Churn"].map({"No": 0, "Yes": 1})

    validate_feature_columns(x_train.columns)
    validate_feature_columns(x_test.columns)
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Unexpected labels in y_train.csv or y_test.csv")
    return x_train, x_test, y_train.astype(int), y_test.astype(int)


def select_tree_count(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[int, float]:
    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    candidate = XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.05,
        max_depth=4,
        reg_lambda=10.0,
        gamma=0.25,
        scale_pos_weight=3,
        subsample=0.9,
        colsample_bytree=0.5,
        n_estimators=1000,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
    )
    candidate.fit(
        x_fit,
        y_fit,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )
    return int(candidate.best_iteration + 1), float(candidate.best_score)


def rebuild_raw_test_rows() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    indices = raw.index
    _, test_indices = train_test_split(
        indices,
        test_size=0.20,
        stratify=raw["Churn"],
        random_state=RANDOM_STATE,
    )
    return raw.loc[test_indices].reset_index(drop=True)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    x_train, x_test, y_train, y_test = load_split()
    n_estimators, validation_aucpr = select_tree_count(x_train, y_train)

    model = XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.05,
        max_depth=4,
        reg_lambda=10.0,
        gamma=0.25,
        scale_pos_weight=3,
        subsample=0.9,
        colsample_bytree=0.5,
        n_estimators=n_estimators,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
    )
    model.fit(x_train, y_train, verbose=False)

    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= FIXED_THRESHOLD).astype(int)
    matrix = confusion_matrix(y_test, prediction)

    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "average_precision": float(average_precision_score(y_test, probability)),
    }

    model.save_model(MODELS / "xgboost_churn.json")
    (ARTIFACTS / "feature_columns.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8"
    )

    raw_test = rebuild_raw_test_rows()
    rebuilt = preprocess_customers(raw_test)
    if not np.allclose(rebuilt.to_numpy(), x_test.to_numpy()):
        raise RuntimeError("Raw test reconstruction does not match X_test.csv")

    explorer = raw_test.drop(columns=["TotalCharges", "Churn"]).copy()
    explorer["actual_churn"] = np.where(y_test.to_numpy() == 1, "Yes", "No")
    explorer["churn_probability"] = probability
    explorer["predicted_churn"] = np.where(prediction == 1, "Yes", "No")
    explorer["correct_prediction"] = np.where(
        prediction == y_test.to_numpy(), "Correct", "Incorrect"
    )
    explorer.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_.astype(float),
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACTS / "feature_importance.csv", index=False)

    metadata = {
        "model_name": "XGBoost Telco Churn Classifier",
        "model_version": "1.0",
        "target": "Churn",
        "positive_class": "Yes",
        "threshold": FIXED_THRESHOLD,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "n_estimators": n_estimators,
        "validation_average_precision": validation_aucpr,
        "metrics": metrics,
        "confusion_matrix": matrix.astype(int).tolist(),
        "selection_rationale": (
            "XGBoost was frozen as the final team model. The 0.60 decision threshold "
            "keeps churn recall above overall precision so a retention team can identify "
            "more customers who may leave, while still improving precision over the "
            "default class-weighted operating point."
        ),
        "important_note": (
            "This artifact was trained from telco-data/X_train.csv using the exact "
            "23-feature schema produced by feature-engineering.ipynb."
        ),
    }
    (ARTIFACTS / "meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Export complete")
    print(f"Model: {MODELS / 'xgboost_churn.json'}")
    print(f"Threshold: {FIXED_THRESHOLD:.2f}")
    print(f"Trees: {n_estimators}")
    for name, value in metrics.items():
        print(f"{name:>18}: {value:.4f}")
    print("Confusion matrix:", matrix.tolist())


if __name__ == "__main__":
    main()
