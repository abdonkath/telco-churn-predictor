"""Streamlit deployment for the team's final Telco churn XGBoost workflow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

from src.preprocessing import encode_customers

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ROOT / "models" / "xgboost_churn.json"

st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 1rem; border-radius: .75rem;}
    .risk-card {padding: 1.25rem; border: 1px solid rgba(128,128,128,.3); border-radius: .8rem; margin-top: .5rem;}
    .small-note {font-size: .9rem; opacity: .8;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    loaded = XGBClassifier()
    loaded.load_model(MODEL_PATH)
    return loaded


@st.cache_data
def load_metadata() -> dict:
    return json.loads((ARTIFACTS / "meta.json").read_text(encoding="utf-8"))


@st.cache_data
def load_feature_columns() -> list[str]:
    return json.loads((ARTIFACTS / "feature_columns.json").read_text(encoding="utf-8"))


@st.cache_data
def load_location_lookup() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "location_lookup.csv")


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "test_predictions.csv")


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "feature_importance.csv")


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def humanize_feature(feature: str) -> str:
    text = feature.replace("_", " ")
    text = text.replace("Fiber optic", "Fiber optic")
    return text


def build_input_row() -> dict:
    st.subheader("Customer profile")
    st.caption("Enter the information currently available for one customer account.")

    locations = load_location_lookup()
    location_label = st.selectbox(
        "Customer location (city / ZIP)",
        locations["label"].tolist(),
        help="ZIP code, latitude, and longitude are filled automatically from the team dataset.",
    )
    location = locations.loc[locations["label"] == location_label].iloc[0]

    demographic_1, demographic_2, demographic_3, demographic_4 = st.columns(4)
    with demographic_1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with demographic_2:
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
    with demographic_3:
        partner = st.selectbox("Partner", ["No", "Yes"])
    with demographic_4:
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    account_1, account_2, account_3, account_4 = st.columns(4)
    with account_1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    with account_2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with account_3:
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with account_4:
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=250.0, value=70.0, step=0.50
        )

    billing_1, billing_2, billing_3, billing_4 = st.columns(4)
    with billing_1:
        total_charges = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=1.0,
            help="Cumulative charges on the customer account.",
        )
    with billing_2:
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
    with billing_3:
        phone = st.selectbox("Phone service", ["Yes", "No"])
    with billing_4:
        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes"],
            help="Handled automatically as 'No phone service' when phone service is No.",
        )

    internet_1, internet_2, internet_3, internet_4 = st.columns(4)
    with internet_1:
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    with internet_2:
        online_security = st.selectbox("Online security", ["No", "Yes"])
    with internet_3:
        online_backup = st.selectbox("Online backup", ["No", "Yes"])
    with internet_4:
        device_protection = st.selectbox("Device protection", ["No", "Yes"])

    service_1, service_2, service_3 = st.columns(3)
    with service_1:
        tech_support = st.selectbox("Tech support", ["No", "Yes"])
    with service_2:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    with service_3:
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])

    return {
        "City": str(location["City"]),
        "Zip_Code": int(location["Zip Code"]),
        "Latitude": float(location["Latitude"]),
        "Longitude": float(location["Longitude"]),
        "Gender": gender,
        "Senior_Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure_Months": int(tenure),
        "Phone_Service": phone,
        "Multiple_Lines": multiple_lines,
        "Internet_Service": internet,
        "Online_Security": online_security,
        "Online_Backup": online_backup,
        "Device_Protection": device_protection,
        "Tech_Support": tech_support,
        "Streaming_TV": streaming_tv,
        "Streaming_Movies": streaming_movies,
        "Contract": contract,
        "Paperless_Billing": paperless,
        "Payment_Method": payment,
        "Monthly_Charges": float(monthly_charges),
        "Total_Charges": float(total_charges),
    }


def render_prediction(
    model: XGBClassifier,
    metadata: dict,
    feature_columns: list[str],
    customer: dict,
) -> None:
    if not st.button("Estimate churn risk", type="primary", use_container_width=True):
        return

    try:
        features = encode_customers(pd.DataFrame([customer]), feature_columns)
        probability = float(model.predict_proba(features)[:, 1][0])
    except (ValueError, TypeError) as error:
        st.error(f"Prediction could not be completed: {error}")
        return

    threshold = float(metadata["threshold"])
    predicted_churn = probability >= threshold

    # The team requested that probability and the derived risk-level card remain hidden
    # to keep the live demo focused on the actionable model decision.
    st.metric("Model decision", "Likely to leave" if predicted_churn else "Likely to stay")

    if predicted_churn:
        recommendation = (
            "Prioritize this customer for a retention review. Consider checking recent "
            "service issues, contract options, support needs, and payment experience."
        )
    else:
        recommendation = (
            "No immediate retention flag was generated. Continue normal service monitoring "
            "and review the account again if customer behavior changes."
        )

    st.markdown(
        f"""
        <div class="risk-card">
          <strong>Recommended business action</strong><br>
          {recommendation}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_test_explorer() -> None:
    st.header("B. Test predictions explorer")
    st.write(
        "Review customers from the locked holdout set. These records were not used to fit the final model."
    )

    data = load_test_predictions()
    filter_1, filter_2, filter_3 = st.columns(3)
    with filter_1:
        correctness = st.selectbox("Prediction result", ["All", "Correct", "Incorrect"])
    with filter_2:
        actual = st.selectbox("Actual churn", ["All", "Yes", "No"])
    with filter_3:
        predicted = st.selectbox("Predicted churn", ["All", "Yes", "No"])

    filtered = data.copy()
    if correctness != "All":
        filtered = filtered[filtered["correct_prediction"] == correctness]
    if actual != "All":
        filtered = filtered[filtered["actual_churn"] == actual]
    if predicted != "All":
        filtered = filtered[filtered["predicted_churn"] == predicted]

    summary_1, summary_2, summary_3 = st.columns(3)
    summary_1.metric("Matching records", f"{len(filtered):,}")
    summary_2.metric(
        "Actual churn rate",
        format_percent((filtered["actual_churn"] == "Yes").mean()) if len(filtered) else "—",
    )
    summary_3.metric(
        "Average predicted risk",
        format_percent(filtered["churn_probability"].mean()) if len(filtered) else "—",
    )

    display_columns = [
        "customerID",
        "City",
        "ZipCode",
        "tenure",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "actual_churn",
        "churn_probability",
        "predicted_churn",
        "correct_prediction",
    ]
    display = filtered[display_columns].copy()
    display["churn_probability"] = display["churn_probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True, height=430)
    st.download_button(
        "Download filtered predictions",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_telco_test_predictions.csv",
        mime="text/csv",
    )


def render_effectiveness(metadata: dict) -> None:
    st.header("C. Model effectiveness")
    st.write(
        f"The metrics below come from the locked {metadata['test_rows']:,}-customer test set "
        f"using the deployed {format_percent(metadata['threshold'])} decision threshold."
    )

    metrics = metadata["metrics"]
    metric_columns = st.columns(5)
    metric_columns[0].metric("Accuracy", format_percent(metrics["accuracy"]))
    metric_columns[1].metric("Precision", format_percent(metrics["precision"]))
    metric_columns[2].metric("Recall", format_percent(metrics["recall"]))
    metric_columns[3].metric("F1 score", f"{metrics['f1']:.3f}")
    metric_columns[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    chart_1, chart_2 = st.columns(2)
    matrix = np.asarray(metadata["confusion_matrix"])
    with chart_1:
        st.subheader("Confusion matrix")
        fig, axis = plt.subplots(figsize=(5.5, 4.2))
        axis.imshow(matrix)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
        axis.set_xticks([0, 1], labels=["Predicted stay", "Predicted churn"])
        axis.set_yticks([0, 1], labels=["Actual stay", "Actual churn"])
        axis.set_xlabel("Model prediction")
        axis.set_ylabel("Actual outcome")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with chart_2:
        st.subheader("Top model features")
        importance = load_feature_importance().head(12).sort_values("importance").copy()
        importance["feature"] = importance["feature"].map(humanize_feature)
        fig, axis = plt.subplots(figsize=(6.5, 4.8))
        axis.barh(importance["feature"], importance["importance"])
        axis.set_xlabel("Relative XGBoost importance")
        axis.set_ylabel("")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.subheader("How to read these results")
    st.write(metadata["selection_rationale"])
    false_positives = int(matrix[0, 1])
    false_negatives = int(matrix[1, 0])
    st.write(
        f"At the deployed threshold, the model correctly identifies {int(matrix[1, 1]):,} churners. "
        f"It produces {false_positives:,} false retention alerts and misses {false_negatives:,} actual churners."
    )
    st.caption(
        "Feature importance describes how strongly the trained model used a feature; it does not prove causation. "
        "Predictions should support, not replace, business judgment."
    )


def main() -> None:
    try:
        model = load_model()
        metadata = load_metadata()
        feature_columns = load_feature_columns()
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        st.error(f"The deployment artifacts could not be loaded: {error}")
        st.stop()

    st.title("Telco Customer Churn Predictor")
    st.write(
        "A business-facing demonstration of the team's final XGBoost workflow for identifying customers who may discontinue service."
    )
    st.caption(
        f"Model: {metadata['model_name']} · {metadata['raw_feature_count']} source features / "
        f"{metadata['feature_count']:,} encoded features · fixed threshold {format_percent(metadata['threshold'])}"
    )

    st.divider()
    st.header("A. Customer churn prediction")
    customer = build_input_row()
    render_prediction(model, metadata, feature_columns, customer)

    st.divider()
    render_test_explorer()

    st.divider()
    render_effectiveness(metadata)


if __name__ == "__main__":
    main()
