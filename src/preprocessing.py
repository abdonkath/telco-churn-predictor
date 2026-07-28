"""Preprocessing for the team's final XGBoost Telco churn model.

This module mirrors the transformations in ``xgboost_final.ipynb``:
- drop leakage / identifier fields during training
- normalize spaces to underscores
- one-hot encode the same categorical columns
- align inference rows to the exact training feature order
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


RAW_MODEL_COLUMNS: list[str] = [
    "City",
    "Zip_Code",
    "Latitude",
    "Longitude",
    "Gender",
    "Senior_Citizen",
    "Partner",
    "Dependents",
    "Tenure_Months",
    "Phone_Service",
    "Multiple_Lines",
    "Internet_Service",
    "Online_Security",
    "Online_Backup",
    "Device_Protection",
    "Tech_Support",
    "Streaming_TV",
    "Streaming_Movies",
    "Contract",
    "Paperless_Billing",
    "Payment_Method",
    "Monthly_Charges",
    "Total_Charges",
]

CATEGORICAL_COLUMNS: list[str] = [
    "City",
    "Gender",
    "Partner",
    "Dependents",
    "Phone_Service",
    "Multiple_Lines",
    "Internet_Service",
    "Online_Security",
    "Device_Protection",
    "Tech_Support",
    "Streaming_TV",
    "Streaming_Movies",
    "Contract",
    "Paperless_Billing",
    "Payment_Method",
    "Senior_Citizen",
    "Online_Backup",
]

NUMERIC_COLUMNS: list[str] = [
    "Zip_Code",
    "Latitude",
    "Longitude",
    "Tenure_Months",
    "Monthly_Charges",
    "Total_Charges",
]


def prepare_training_frame(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce the raw cleanup in the team's final notebook."""

    data = source.copy()
    required = {
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
        "Churn Label",
        "Churn Value",
        "Churn Score",
        "CLTV",
        "Churn Reason",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected column(s): {missing}")

    data.drop(["Churn Label", "Churn Score", "CLTV", "Churn Reason"], axis=1, inplace=True)
    data.drop(["CustomerID", "Count", "Country", "State", "Lat Long"], axis=1, inplace=True)

    data["City"] = data["City"].replace(" ", "_", regex=True)
    data.columns = data.columns.str.replace(" ", "_", regex=False)
    data.loc[data["Total_Charges"] == " ", "Total_Charges"] = "0"
    data["Total_Charges"] = pd.to_numeric(data["Total_Charges"], errors="raise")
    data.replace(" ", "_", regex=True, inplace=True)

    x = data.drop("Churn_Value", axis=1).copy()
    y = pd.to_numeric(data["Churn_Value"], errors="raise").astype(int).copy()
    return x, y


def normalize_app_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize human-readable Streamlit rows to notebook-style raw columns."""

    frame = raw.copy()

    aliases = {
        "Zip Code": "Zip_Code",
        "Senior Citizen": "Senior_Citizen",
        "Tenure Months": "Tenure_Months",
        "Phone Service": "Phone_Service",
        "Multiple Lines": "Multiple_Lines",
        "Internet Service": "Internet_Service",
        "Online Security": "Online_Security",
        "Online Backup": "Online_Backup",
        "Device Protection": "Device_Protection",
        "Tech Support": "Tech_Support",
        "Streaming TV": "Streaming_TV",
        "Streaming Movies": "Streaming_Movies",
        "Paperless Billing": "Paperless_Billing",
        "Payment Method": "Payment_Method",
        "Monthly Charges": "Monthly_Charges",
        "Total Charges": "Total_Charges",
    }
    frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns}, inplace=True)

    missing = [column for column in RAW_MODEL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing customer field(s): {missing}")

    frame = frame[RAW_MODEL_COLUMNS].copy()

    # Match notebook cleanup: every literal space inside string values becomes an underscore.
    string_columns = frame.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        frame[column] = frame[column].astype(str).str.replace(" ", "_", regex=False)

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{column} must contain a valid numeric value")

    if (frame["Tenure_Months"] < 0).any():
        raise ValueError("Tenure_Months cannot be negative")
    if (frame["Monthly_Charges"] < 0).any() or (frame["Total_Charges"] < 0).any():
        raise ValueError("Charge values cannot be negative")

    # Enforce combinations that occur in the source Telco data.
    no_phone = frame["Phone_Service"].eq("No")
    frame.loc[no_phone, "Multiple_Lines"] = "No_phone_service"

    no_internet = frame["Internet_Service"].eq("No")
    internet_dependent = [
        "Online_Security",
        "Online_Backup",
        "Device_Protection",
        "Tech_Support",
        "Streaming_TV",
        "Streaming_Movies",
    ]
    for column in internet_dependent:
        frame.loc[no_internet, column] = "No_internet_service"

    return frame


def encode_customers(raw: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """One-hot encode customer rows and align them to the frozen model schema."""

    normalized = normalize_app_rows(raw)
    encoded = pd.get_dummies(normalized, columns=CATEGORICAL_COLUMNS)
    encoded = encoded.reindex(columns=list(feature_columns), fill_value=0)
    return encoded


def validate_feature_columns(columns: Iterable[str], expected: Iterable[str]) -> None:
    received = list(columns)
    wanted = list(expected)
    if received != wanted:
        raise ValueError(
            f"Feature schema mismatch: received {len(received)} columns, expected {len(wanted)}."
        )
