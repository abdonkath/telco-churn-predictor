# Streamlit Deployment

The Streamlit app deploys the team's final XGBoost churn workflow from `xgboost_final.ipynb` using the Kaggle `Telco_customer_churn.csv` dataset supplied by the team.

## What is frozen for deployment

The app loads these committed artifacts at runtime:

- `models/xgboost_churn.json` — saved XGBoost model
- `artifacts/feature_columns.json` — exact 1,159-column encoded feature order
- `artifacts/meta.json` — split information, threshold, and evaluation metrics
- `artifacts/test_predictions.csv` — locked test-set explorer data
- `artifacts/feature_importance.csv` — model feature importance
- `artifacts/location_lookup.csv` — city/ZIP/latitude/longitude lookup used by the form

The source dataset used to rebuild the model is:

- `telco-data/Telco_customer_churn.csv`

The training/export script is:

```bash
python scripts/train_xgboost_artifacts.py
```

## Important reproducibility note

The team's notebook contained the final training code and saved output, but it did not commit the fitted in-memory XGBoost object as a model file. The deployment artifact in `models/` is therefore a reproducible rebuild using the exact team dataset, `random_state=47`, preprocessing, train/validation/test split, and final hyperparameters from `xgboost_final.ipynb`.

XGBoost can produce small numerical/tree differences across library versions. The app always reports metrics from the exact saved artifact it is currently serving.

## Run locally

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Streamlit Community Cloud

Deploy the repository and select `app.py` as the entry point. `requirements.txt` contains the runtime dependencies.

## UI decision requested by the team

The customer prediction section intentionally does **not** display the raw churn probability or a derived risk-level label. The probability is still calculated internally so the model can make its binary decision, but the live demo only shows:

- `Model decision` — likely to leave / likely to stay
- `Recommended business action`

The test explorer and model-effectiveness sections remain available for technical review.
