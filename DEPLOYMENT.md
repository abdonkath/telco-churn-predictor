# Streamlit Deployment Guide

This repository contains a deployable Streamlit app for the frozen 23-feature
XGBoost Telco churn model.

## Files used by the app

- `app.py` — Streamlit entry point
- `src/preprocessing.py` — converts business inputs to the exact 23 model features
- `models/xgboost_churn.json` — frozen XGBoost model
- `artifacts/meta.json` — threshold, metrics, feature order, and model metadata
- `artifacts/test_predictions.csv` — locked test-set explorer data
- `artifacts/feature_importance.csv` — chart data
- `requirements.txt` — cloud dependencies

The app loads artifacts. It does not train the model during page startup.

## Run locally

From the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit, normally `http://localhost:8501`.

## Regenerate the model artifacts

Only do this when the team intentionally changes the frozen model:

```bash
python scripts/train_xgboost_artifacts.py
```

After regeneration, run the app and compare a test-explorer record with the
training output before committing the new artifacts.

## Deploy with Streamlit Community Cloud

1. Push the repository to GitHub.
2. Sign in to Streamlit Community Cloud using the GitHub account that can access the repository.
3. Create a new app and select the repository, branch, and `app.py` entry point.
4. Deploy and wait for the dependency installation and app health check.
5. Test the live prediction, test explorer, and model-effectiveness sections.

## Recommended Git commit

```bash
git add app.py src scripts models artifacts requirements.txt .streamlit DEPLOYMENT.md README.md
git commit -m "Add XGBoost Streamlit deployment"
git push
```

## Important consistency rule

`models/xgboost_churn.json`, `artifacts/meta.json`, and
`src/preprocessing.py` must all use the same 23-feature order. Do not replace
only one of these files with an artifact trained from a different dataset.
