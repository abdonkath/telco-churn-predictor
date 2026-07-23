"""Feature engineering shared by training validation and Streamlit inference.

The final XGBoost model expects the 23-column schema produced by the team's
``feature-engineering.ipynb`` notebook. This module converts human-readable
Telco customer fields into that exact schema.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "MonthlyCharges",
    "InternetService_Fiber optic",
    "InternetService_No",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "tenure_group",
]

_REQUIRED_RAW_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
]


def _yes_no(series: pd.Series, column: str) -> pd.Series:
    mapped = series.map({"Yes": 1, "No": 0, 1: 1, 0: 0, True: 1, False: 0})
    if mapped.isna().any():
        invalid = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected value(s) for {column}: {invalid}")
    return mapped.astype(int)


def _tenure_group(tenure: pd.Series) -> pd.Series:
    values = pd.to_numeric(tenure, errors="coerce")
    if values.isna().any() or (values < 0).any():
        raise ValueError("tenure must contain non-negative numeric values")
    return pd.cut(
        values,
        bins=[-1, 12, 24, 48, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)


def preprocess_customers(raw_customers: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Telco fields into the final 23 XGBoost features.

    Extra columns such as ``customerID``, ``TotalCharges``, and ``Churn`` are
    safely ignored. The output column order is fixed and validated.
    """

    missing = [column for column in _REQUIRED_RAW_COLUMNS if column not in raw_customers]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    raw = raw_customers.copy()
    output = pd.DataFrame(index=raw.index)

    output["gender"] = raw["gender"].map({"Female": 0, "Male": 1})
    if output["gender"].isna().any():
        invalid = sorted(raw.loc[output["gender"].isna(), "gender"].astype(str).unique())
        raise ValueError(f"Unexpected gender value(s): {invalid}")
    output["gender"] = output["gender"].astype(int)

    senior = raw["SeniorCitizen"].map({"Yes": 1, "No": 0, 1: 1, 0: 0, True: 1, False: 0})
    output["SeniorCitizen"] = pd.to_numeric(senior, errors="coerce")
    if output["SeniorCitizen"].isna().any() or not output["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError("SeniorCitizen must be Yes/No or 1/0")
    output["SeniorCitizen"] = output["SeniorCitizen"].astype(int)

    for column in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        output[column] = _yes_no(raw[column], column)

    output["tenure"] = pd.to_numeric(raw["tenure"], errors="coerce")
    if output["tenure"].isna().any() or (output["tenure"] < 0).any():
        raise ValueError("tenure must be a non-negative number")
    output["tenure"] = output["tenure"].astype(int)

    internet_service = raw["InternetService"].astype(str)
    valid_internet = {"DSL", "Fiber optic", "No"}
    invalid_internet = sorted(set(internet_service.unique()) - valid_internet)
    if invalid_internet:
        raise ValueError(f"Unexpected InternetService value(s): {invalid_internet}")

    internet_dependent = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    no_internet = internet_service.eq("No")
    for column in internet_dependent:
        normalized = raw[column].replace({"No internet service": "No"}).copy()
        normalized.loc[no_internet] = "No"
        output[column] = _yes_no(normalized, column)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    output["Contract"] = raw["Contract"].map(contract_map)
    if output["Contract"].isna().any():
        invalid = sorted(raw.loc[output["Contract"].isna(), "Contract"].astype(str).unique())
        raise ValueError(f"Unexpected Contract value(s): {invalid}")
    output["Contract"] = output["Contract"].astype(int)

    output["MonthlyCharges"] = pd.to_numeric(raw["MonthlyCharges"], errors="coerce")
    if output["MonthlyCharges"].isna().any() or (output["MonthlyCharges"] < 0).any():
        raise ValueError("MonthlyCharges must be a non-negative number")

    output["InternetService_Fiber optic"] = internet_service.eq("Fiber optic").astype(int)
    output["InternetService_No"] = internet_service.eq("No").astype(int)

    payment = raw["PaymentMethod"].astype(str)
    valid_payment = {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    }
    invalid_payment = sorted(set(payment.unique()) - valid_payment)
    if invalid_payment:
        raise ValueError(f"Unexpected PaymentMethod value(s): {invalid_payment}")
    output["PaymentMethod_Credit card (automatic)"] = payment.eq(
        "Credit card (automatic)"
    ).astype(int)
    output["PaymentMethod_Electronic check"] = payment.eq("Electronic check").astype(int)
    output["PaymentMethod_Mailed check"] = payment.eq("Mailed check").astype(int)

    multiple_lines = raw["MultipleLines"].astype(str).copy()
    no_phone = output["PhoneService"].eq(0)
    multiple_lines.loc[no_phone] = "No phone service"
    valid_multiple = {"No", "Yes", "No phone service"}
    invalid_multiple = sorted(set(multiple_lines.unique()) - valid_multiple)
    if invalid_multiple:
        raise ValueError(f"Unexpected MultipleLines value(s): {invalid_multiple}")
    output["MultipleLines_No phone service"] = multiple_lines.eq("No phone service").astype(int)
    output["MultipleLines_Yes"] = multiple_lines.eq("Yes").astype(int)

    output["tenure_group"] = _tenure_group(output["tenure"])

    output = output.reindex(columns=FEATURE_COLUMNS)
    if output.isna().any().any():
        bad_columns = output.columns[output.isna().any()].tolist()
        raise ValueError(f"Preprocessing produced missing values in: {bad_columns}")

    return output.reset_index(drop=True)


def validate_feature_columns(columns: Iterable[str]) -> None:
    """Raise a clear error when an artifact uses a different feature schema."""

    received = list(columns)
    if received != FEATURE_COLUMNS:
        raise ValueError(
            "Feature schema mismatch. The model/app must use the exact 23-column "
            "schema from feature-engineering.ipynb."
        )
