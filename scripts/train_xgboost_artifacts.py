"""Rebuild and export the team's final XGBoost model for Streamlit.

This script mirrors ``xgboost_final.ipynb`` and uses the exact Kaggle dataset
``telco-data/Telco_customer_churn.csv`` supplied by the team.

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
import sklearn
import xgboost
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

from src.preprocessing import CATEGORICAL_COLUMNS, prepare_training_frame

DATA = ROOT / "telco-data" / "Telco_customer_churn.csv"
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "xgboost_churn.json"
RANDOM_STATE = 47
DECISION_THRESHOLD = 0.50


def load_and_split() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    raw = pd.read_csv(DATA)
    x, y = prepare_training_frame(raw)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    x_train_encoded = pd.get_dummies(x_train, columns=CATEGORICAL_COLUMNS)
    x_val_encoded = pd.get_dummies(x_val, columns=CATEGORICAL_COLUMNS)
    x_test_encoded = pd.get_dummies(x_test, columns=CATEGORICAL_COLUMNS)

    x_val_encoded = x_val_encoded.reindex(columns=x_train_encoded.columns, fill_value=0)
    x_test_encoded = x_test_encoded.reindex(columns=x_train_encoded.columns, fill_value=0)

    return (
        x_train_encoded,
        x_val_encoded,
        x_test_encoded,
        y_train,
        y_val,
        y_test,
        raw,
    )


def train_model(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> XGBClassifier:
    # Exact hyperparameters from xgboost_final.ipynb.
    model = XGBClassifier(
        seed=47,
        objective="binary:logistic",
        learning_rate=0.05,
        max_depth=4,
        reg_lambda=10.0,
        gamma=0.25,
        scale_pos_weight=3,
        subsample=0.9,
        colsample_bytree=0.5,
        early_stopping_rounds=10,
        eval_metric="aucpr",
    )
    model.fit(x_train, y_train, verbose=False, eval_set=[(x_val, y_val)])
    return model


def build_location_lookup(raw: pd.DataFrame) -> pd.DataFrame:
    lookup = raw[["City", "Zip Code", "Latitude", "Longitude"]].drop_duplicates().copy()
    lookup["label"] = lookup["City"].astype(str) + " — " + lookup["Zip Code"].astype(str)
    lookup.sort_values(["City", "Zip Code"], inplace=True)
    return lookup[["label", "City", "Zip Code", "Latitude", "Longitude"]].reset_index(drop=True)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing team dataset: {DATA}")

    ARTIFACTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    x_train, x_val, x_test, y_train, y_val, y_test, raw = load_and_split()
    model = train_model(x_train, x_val, y_train, y_val)

    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= DECISION_THRESHOLD).astype(int)
    matrix = confusion_matrix(y_test, prediction)

    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "average_precision": float(average_precision_score(y_test, probability)),
    }

    model.save_model(MODEL_PATH)
    feature_columns = x_train.columns.tolist()
    (ARTIFACTS / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )

    # Preserve the exact raw test records by index. train_test_split keeps original indices.
    raw_test = raw.loc[x_test.index].copy()
    explorer = pd.DataFrame(
        {
            "customerID": raw_test["CustomerID"].astype(str).to_numpy(),
            "City": raw_test["City"].astype(str).to_numpy(),
            "ZipCode": raw_test["Zip Code"].to_numpy(),
            "tenure": raw_test["Tenure Months"].to_numpy(),
            "Contract": raw_test["Contract"].astype(str).to_numpy(),
            "InternetService": raw_test["Internet Service"].astype(str).to_numpy(),
            "PaymentMethod": raw_test["Payment Method"].astype(str).to_numpy(),
            "MonthlyCharges": raw_test["Monthly Charges"].to_numpy(),
            "actual_churn": np.where(y_test.to_numpy() == 1, "Yes", "No"),
            "churn_probability": probability,
            "predicted_churn": np.where(prediction == 1, "Yes", "No"),
            "correct_prediction": np.where(
                prediction == y_test.to_numpy(), "Correct", "Incorrect"
            ),
        }
    )
    explorer.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_.astype(float)}
    ).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACTS / "feature_importance.csv", index=False)

    build_location_lookup(raw).to_csv(ARTIFACTS / "location_lookup.csv", index=False)

    metadata = {
        "model_name": "Team Final XGBoost Churn Classifier",
        "model_version": "team-final-reproduction-1.0",
        "source_notebook": "xgboost_final.ipynb",
        "source_dataset": "telco-data/Telco_customer_churn.csv",
        "dataset_rows": int(len(raw)),
        "raw_feature_count": 23,
        "feature_count": int(len(feature_columns)),
        "feature_names": feature_columns,
        "target": "Churn_Value",
        "positive_class": 1,
        "threshold": DECISION_THRESHOLD,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
        "best_iteration": int(model.best_iteration),
        "best_validation_aucpr": float(model.best_score),
        "metrics": metrics,
        "confusion_matrix": matrix.astype(int).tolist(),
        "xgboost_version": xgboost.__version__,
        "scikit_learn_version": sklearn.__version__,
        "selection_rationale": (
            "This deployment reproduces the team's xgboost_final.ipynb workflow and "
            "evaluates it on the same locked 20% holdout split (random_state=47). "
            "Recall is emphasized because missed churners are customers the retention team "
            "would fail to identify."
        ),
        "important_note": (
            "The original fitted in-memory XGBoost object was not committed to the repository. "
            "This saved artifact is a reproducible rebuild from the team's final notebook, "
            "exact Kaggle dataset, split seed, preprocessing, and final hyperparameters."
        ),
        "notebook_recorded_confusion_matrix": [[721, 314], [55, 319]],
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Team XGBoost deployment artifacts exported")
    print(f"Dataset: {DATA}")
    print(f"Model: {MODEL_PATH}")
    print(f"Encoded features: {len(feature_columns)}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Threshold: {DECISION_THRESHOLD:.2f}")
    for name, value in metrics.items():
        print(f"{name:>18}: {value:.4f}")
    print("Confusion matrix:", matrix.tolist())
    if matrix.tolist() != [[721, 314], [55, 319]]:
        print(
            "NOTE: the stored notebook output used a different XGBoost runtime and recorded "
            "[[721, 314], [55, 319]]. Small differences after retraining can occur across "
            "XGBoost versions even with the same data, seed, and hyperparameters."
        )


if __name__ == "__main__":
    main()
