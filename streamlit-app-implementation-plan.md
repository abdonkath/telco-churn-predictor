# Streamlit Churn App — Implementation Plan

A beginner-friendly plan for turning this team’s churn model into a deployed Streamlit app.

**Audience:** teammates who have trained models in notebooks but have not built Streamlit apps or deployed anything to the cloud.

**Goal:** after the team picks a final model, ship a public web app where a business user can (1) get a churn prediction from customer inputs, (2) inspect held-out test records (prediction vs reality), and (3) review why the chosen model is effective.

---

## Table of contents

1. [What this plan assumes](#1-what-this-plan-assumes)
2. [Spec sheet](#2-spec-sheet)
3. [Recommended tools](#3-recommended-tools)
4. [Alternative tools](#4-alternative-tools)
5. [Tool comparison at a glance](#5-tool-comparison-at-a-glance)
6. [Recommended architecture](#6-recommended-architecture)
7. [Implementation guide (recommended stack)](#7-implementation-guide-recommended-stack)
8. [Deploying to Streamlit Community Cloud](#8-deploying-to-streamlit-community-cloud)
9. [Testing checklist](#9-testing-checklist)
10. [Phased timeline](#10-phased-timeline)
11. [Risks and beginner pitfalls](#11-risks-and-beginner-pitfalls)
12. [Glossary](#12-glossary)

---

## 1. What this plan assumes

| Assumption | Why it matters |
|---|---|
| The team will **freeze one final model** (e.g. tuned Random Forest + chosen decision threshold) before focusing on the app | The app should load a saved artifact, not retrain on every page load |
| Train/test CSVs already exist under `telco-data/` | The “test record explorer” must only use **test** rows so we never show training examples as “unseen” |
| Features are already engineered (see `X_train.csv` / `X_test.csv`) | The prediction form must collect inputs that can be mapped into those same feature columns |
| This is a **learning / portfolio** project, not a production bank-grade system | Prefer simple, free, easy-to-explain tools over enterprise platforms |
| Deployment target is **Streamlit Community Cloud** | App code, dependencies, and data/model files must live in a **public or connected GitHub repo** |

**Current repo context (as of this plan):**

- Modeling work lives mainly in notebooks (`random-forest-model.ipynb`, `feature-engineering.ipynb`, `eda-telco.ipynb`)
- Feature matrix has **23 columns** after engineering
- Test set size is about **1,409** customers
- There is **not yet** a saved `.joblib` / `.pkl` model file or a Streamlit `app.py` — those are part of this plan

---

## 2. Spec sheet

### 2.1 Product summary

| Field | Spec |
|---|---|
| **Product name (working)** | Telco Churn Predictor |
| **Primary users** | Business owners / retention stakeholders (simulated); also students reviewing the project |
| **Primary job** | Enter customer demographics / account attributes → see predicted churn risk |
| **Secondary jobs** | Explore unseen test records; understand model quality and why this model was chosen |
| **Success criteria** | App runs locally and on Streamlit Community Cloud; three sections work; predictions match notebook results for the same inputs |

### 2.2 App sections (required)

#### Section A — Live prediction (business owner)

| Requirement | Detail |
|---|---|
| **Purpose** | Let a user enter customer information and receive a churn prediction |
| **Inputs** | Form fields for the features the frozen model expects (demographics, services, contract, billing). Prefer **human-readable** labels (e.g. “Contract type: Month-to-month”) even if the model uses encoded integers |
| **Outputs** | Predicted class (`Churn` / `No churn`), churn probability (0–100%), optional short plain-language explanation |
| **Rules** | Use the **same preprocessing + threshold** used when the model was finalized. Do not retrain in the app |
| **Edge cases** | Missing required fields blocked before predict; out-of-range numbers clamped or rejected with a message |

#### Section B — Test-set record explorer (learning / validation)

| Requirement | Detail |
|---|---|
| **Purpose** | Pick a customer who was **held out of training** and compare model prediction to ground truth |
| **Data source** | Only `X_test` + `y_test` (and optional `customerID` if available from the raw CSV join) |
| **UI** | Dropdown or searchable index of test rows; show key features; show predicted label, probability, actual label, and whether they match |
| **Rules** | Never include training rows. Make “test set only” explicit in the UI copy |
| **Nice-to-have** | Filter by correct/incorrect predictions; show confusion-matrix category (TP / FP / TN / FN) |

#### Section C — Model effectiveness (why this model)

| Requirement | Detail |
|---|---|
| **Purpose** | Communicate the internal analysis that justified picking this model |
| **Content** | Metrics (accuracy, precision, recall, F1, ROC-AUC), confusion matrix, ROC and/or precision-recall curves, threshold choice rationale, brief comparison vs alternatives considered (e.g. baseline vs class-weighted RF, logistic regression if evaluated) |
| **Tone** | Business-readable captions under each chart (“What this means for retention”) |
| **Data source** | Precomputed metrics/plots saved as CSV/PNG (or regenerated from test predictions with cached computation). Do not re-run GridSearchCV inside Streamlit |

### 2.3 Non-functional requirements

| Area | Requirement |
|---|---|
| **Performance** | First prediction < ~2 seconds after model load; use caching so the model is not reloaded on every widget change |
| **Reproducibility** | Same Git commit + same artifact → same predictions |
| **Security / privacy** | No real PII beyond the public Telco dataset; no API keys required for core features |
| **Cost** | Free tier only (Streamlit Community Cloud + GitHub) |
| **Maintainability** | Beginner-readable Python; clear folder layout; `requirements.txt` pinned enough to deploy reliably |
| **Documentation** | README section: local run + Cloud deploy steps |

### 2.4 Explicit non-goals (for v1)

- User accounts / login
- Writing predictions to a database
- Retraining the model from the UI
- Serving many concurrent production users
- Real-time streaming data
- Mobile-native apps

### 2.5 Acceptance criteria (definition of done)

- [ ] Final model + threshold saved as a loadable artifact
- [ ] Streamlit app has three navigable sections matching A/B/C above
- [ ] Live form prediction uses the frozen model correctly
- [ ] Test explorer only shows test-set rows and displays actual vs predicted
- [ ] Effectiveness page shows metrics and at least the key comparison narrative
- [ ] App runs locally with `streamlit run`
- [ ] App is deployed on Streamlit Community Cloud from GitHub
- [ ] README documents both local and Cloud workflows

---

## 3. Recommended tools

These are the tools this plan recommends for a beginner team shipping a learning project to Streamlit Community Cloud.

### 3.1 Streamlit (app framework) — **recommended**

| | |
|---|---|
| **What it is** | A Python library that turns scripts into interactive web apps with widgets (sliders, forms, charts) and almost no front-end code |
| **Why recommend it** | Matches the deployment target; uses the same language as your notebooks; very fast path from “model works” to “clickable demo”; team already plans Community Cloud |
| **Justification** | Your users are business stakeholders viewing ML results, not engineers needing a custom React UI. Streamlit is purpose-built for this exact use case |
| **Pros** | Pure Python; quick learning curve; built-in charts/`st.dataframe`; caching (`@st.cache_resource`); free Community Cloud hosting; huge tutorial ecosystem |
| **Cons** | Limited custom UI polish vs React; multi-page apps can get messy without structure; not ideal for complex auth or heavy concurrent traffic; widget re-runs re-execute the script (must learn caching) |

### 3.2 Streamlit Community Cloud (hosting) — **recommended**

| | |
|---|---|
| **What it is** | Streamlit’s free hosting service: connect a GitHub repo, pick `app.py`, and get a public URL |
| **Why recommend it** | Zero server setup; free; designed for Streamlit; good for class/portfolio demos |
| **Justification** | Avoids learning Docker, AWS, or Kubernetes for a learning project. The “ops” work becomes: push to GitHub → click Deploy |
| **Pros** | Free; automatic deploys on git push; HTTPS URL; secrets manager for env vars; no credit card for basic use |
| **Cons** | Resource limits (CPU/RAM/sleep on inactivity); public apps are easy to share but also easy to overload; private repos need a paid Streamlit plan (or make the repo public); cold starts after idle; dependency install can fail if `requirements.txt` is wrong |

### 3.3 GitHub (source control + Cloud connection) — **recommended**

| | |
|---|---|
| **What it is** | Hosted Git repository where your code lives; Streamlit Cloud deploys **from** GitHub |
| **Why recommend it** | Required (practically) for Community Cloud; already the standard for team collaboration |
| **Justification** | Community Cloud pulls your app from a GitHub branch. Without GitHub, the recommended deploy path does not work |
| **Pros** | Free public repos; PR workflow for the team; version history; integrates with Streamlit Cloud |
| **Cons** | Learning Git basics is required; large model/data files need care (Git LFS or keep files small); public repos expose code and data |

### 3.4 Python 3.10+ (runtime) — **recommended**

| | |
|---|---|
| **What it is** | The language/runtime that runs both training notebooks and the Streamlit app |
| **Why recommend it** | Already used in this repo; Streamlit and scikit-learn run on it |
| **Justification** | One language across training and serving reduces context switching for beginners |
| **Pros** | Familiar; rich ML ecosystem; matches notebooks |
| **Cons** | Packaging/environments can confuse beginners (`venv`, dependency conflicts) |

### 3.5 scikit-learn + joblib (model train/save/load) — **recommended**

| | |
|---|---|
| **What it is** | `scikit-learn` trains the model; `joblib` (or `pickle`) serializes the fitted estimator to disk |
| **Why recommend it** | Your Random Forest work is already sklearn-based; joblib is the sklearn-recommended way to persist models |
| **Justification** | Saving `RandomForestClassifier` (and ideally a full preprocessing pipeline + threshold metadata) lets the app load once and predict without notebooks |
| **Pros** | Native fit with your current code; small learning curve (`joblib.dump` / `joblib.load`); works on Community Cloud if versions match |
| **Cons** | Pickled models are **version-sensitive** (sklearn version mismatch can break loading); not a secure format for untrusted files; large forests can be multi‑MB |

### 3.6 pandas (tabular data) — **recommended**

| | |
|---|---|
| **What it is** | Library for reading CSVs and building the single-row feature frame for prediction |
| **Why recommend it** | Already central to this project; Streamlit displays DataFrames well |
| **Justification** | Test explorer and form → model input conversion are DataFrame operations |
| **Pros** | Familiar; CSV I/O; aligns with `telco-data/` |
| **Cons** | Easy to create subtle column-order bugs if form mapping is sloppy |

### 3.7 Matplotlib / Plotly (charts in the app) — **recommended (Plotly preferred in-app)**

| | |
|---|---|
| **What it is** | Charting libraries. Matplotlib is already in notebooks; Plotly gives interactive charts in Streamlit easily |
| **Why recommend Plotly for Section C** | `st.plotly_chart` is interactive (hover tooltips) with little extra code |
| **Justification** | Effectiveness page is about communication; interactive ROC/confusion visuals help non-ML users |
| **Pros (Plotly)** | Interactive; Streamlit-friendly |
| **Cons (Plotly)** | Extra dependency; slightly different API than Matplotlib |
| **Pros (Matplotlib)** | Already used in notebooks; can save static PNGs and display with `st.image` |
| **Cons (Matplotlib)** | Less interactive in-browser unless you regenerate figures carefully |

**Recommendation:** keep Matplotlib for notebook analysis; use **Plotly in Streamlit** for key interactive charts, *or* export PNG metrics plots from notebooks and display them with `st.image` for maximum simplicity.

### 3.8 requirements.txt + venv (dependency management) — **recommended**

| | |
|---|---|
| **What it is** | A text file listing Python packages; a virtual environment isolates them |
| **Why recommend it** | Streamlit Community Cloud installs packages from `requirements.txt` (or `packages.txt` for system deps) |
| **Justification** | Beginners succeed on Cloud when local and Cloud install the **same** package versions |
| **Pros** | Simple; Cloud-native; no new tool to learn beyond pip |
| **Cons** | Easy to forget a package; unpinned versions can break later; weaker than Poetry/uv for complex projects |

### 3.9 Optional but recommended: `streamlit-option-menu` or native `st.navigation` / multipage apps

| | |
|---|---|
| **What it is** | Ways to split the app into pages (Predict / Explore / Effectiveness) |
| **Why recommend native multipage** | Streamlit supports a `pages/` folder or `st.navigation` without extra packages |
| **Justification** | Three clear sections map cleanly to three pages; keeps `app.py` from becoming a 1,000-line monster |
| **Pros** | Cleaner UX; easier team ownership of pages |
| **Cons** | Slightly more file structure to learn |

---

## 4. Alternative tools

Use these only if the team has a specific reason to leave the recommended path.

### 4.1 App frameworks (instead of Streamlit)

#### Gradio

| | |
|---|---|
| **What it is** | Another Python UI library popular for ML demos |
| **Justification for mentioning** | Common Hugging Face demo tool; very fast for “input → prediction” |
| **Pros** | Extremely quick demos; great ML input widgets; Hugging Face Spaces hosting |
| **Cons** | **Not** Streamlit Community Cloud; team already chose Streamlit; multipage “analysis + explorer” apps are often nicer in Streamlit |
| **When to choose** | If you pivot hosting to Hugging Face Spaces |

#### Dash (Plotly)

| | |
|---|---|
| **What it is** | More flexible Python dashboard framework |
| **Pros** | Powerful layouts; strong Plotly integration |
| **Cons** | Steeper learning curve; more boilerplate; Community Cloud is Streamlit-centric |
| **When to choose** | Complex multi-filter BI dashboards |

#### Flask / FastAPI + React (or plain HTML)

| | |
|---|---|
| **What it is** | Separate backend API + custom front end |
| **Pros** | Full control; production-like architecture |
| **Cons** | Much more work; need hosting for API + front end; overkill for this learning project |
| **When to choose** | Course requirement for full-stack, or future production rewrite |

#### Panel / Voila

| | |
|---|---|
| **What it is** | Notebook-oriented dashboard tools |
| **Pros** | Can reuse notebook cells |
| **Cons** | Weaker fit for form-heavy business apps; deploy story is less beginner-friendly than Streamlit Cloud |
| **When to choose** | If the deliverable must stay notebook-shaped |

### 4.2 Hosting (instead of Streamlit Community Cloud)

#### Hugging Face Spaces

| | |
|---|---|
| **Pros** | Free tier; good for Gradio/Streamlit; ML community visibility |
| **Cons** | Extra account/platform; diverges from stated target |
| **When to choose** | Portfolio presence on Hugging Face matters more than Streamlit Cloud |

#### Render / Railway / Fly.io

| | |
|---|---|
| **Pros** | More general web hosting; can run Docker |
| **Cons** | Free tiers change often; more DevOps; must configure start commands yourself |
| **When to choose** | Need a non-Streamlit server or always-on process |

#### AWS / GCP / Azure

| | |
|---|---|
| **Pros** | Industry-standard cloud skills |
| **Cons** | Cost risk; IAM/networking complexity; far beyond v1 scope |
| **When to choose** | Capstone explicitly requires cloud provider experience |

#### Local-only / ngrok demo

| | |
|---|---|
| **Pros** | No deploy friction for a live class presentation |
| **Cons** | Not a durable public URL; laptop must stay on |
| **When to choose** | Temporary presentation before Cloud is ready |

### 4.3 Model persistence / serving (instead of joblib + in-app load)

#### ONNX Runtime

| | |
|---|---|
| **Pros** | Portable across languages/runtimes |
| **Cons** | Extra export step; overkill for sklearn RF learning demo |
| **When to choose** | Need non-Python inference later |

#### MLflow Model Registry

| | |
|---|---|
| **Pros** | Experiment tracking + model versions |
| **Cons** | Heavier setup; needs a tracking server or managed service |
| **When to choose** | Team wants formal experiment tracking beyond notebooks |

#### FastAPI microservice that Streamlit calls

| | |
|---|---|
| **Pros** | Separates UI from model service (cleaner “real world” architecture) |
| **Cons** | Two apps to deploy; more failure modes for beginners |
| **When to choose** | Explicit software-architecture learning goal |

### 4.4 Dependency / environment tools (instead of pip + requirements.txt)

#### Poetry / uv / conda

| | |
|---|---|
| **Pros** | Better lockfiles and reproducibility |
| **Cons** | Streamlit Cloud’s happy path is still primarily `requirements.txt`; extra concepts for beginners |
| **When to choose** | Dependency conflicts become painful; then export a locked `requirements.txt` for Cloud |

### 4.5 Data / artifact storage (instead of Git-tracked files)

#### Git LFS

| | |
|---|---|
| **Pros** | Handles larger binaries in Git |
| **Cons** | Extra setup; Cloud must fetch LFS files correctly |
| **When to choose** | Model file grows large |

#### Cloud object storage (S3, GCS)

| | |
|---|---|
| **Pros** | Keeps repo small |
| **Cons** | Credentials, networking, complexity |
| **When to choose** | Model/data cannot be public in the repo |

**For this project:** keep the fitted model and test CSVs **in the repo** if they stay reasonably small (typical for this Telco RF). That is the simplest Community Cloud path.

---

## 5. Tool comparison at a glance

| Need | Recommended | Strong alternative | Avoid for v1 |
|---|---|---|---|
| App UI | Streamlit | Gradio | Custom React |
| Hosting | Streamlit Community Cloud | Hugging Face Spaces | AWS ECS/EKS |
| Code host | GitHub | — | Deploying from laptop zip only |
| Model save | joblib + sklearn | MLflow | Retrain on every request |
| Tables | pandas | — | Manual CSV parsing |
| Charts | Plotly in app (or PNG from Matplotlib) | Matplotlib only | Heavy BI tools |
| Deps | `requirements.txt` + venv | Poetry/uv (export reqs) | Untracked global installs |

---

## 6. Recommended architecture

### 6.1 Mental model (beginner)

```text
Notebooks (train & evaluate)
        │
        ▼
Save artifacts (model + threshold + metrics)
        │
        ▼
Streamlit app loads artifacts (does not retrain)
        │
        ├── Page 1: form → predict
        ├── Page 2: pick test row → predict vs actual
        └── Page 3: show metrics / charts / narrative
        │
        ▼
Push to GitHub → Streamlit Community Cloud serves public URL
```

### 6.2 Suggested repository layout

```text
netflix-churn-predictor/
├── app/
│   ├── Home.py                 # or app.py at repo root (Cloud entrypoint)
│   ├── pages/
│   │   ├── 1_Predict.py        # Section A
│   │   ├── 2_Test_Explorer.py  # Section B
│   │   └── 3_Model_Effectiveness.py  # Section C
│   └── utils/
│       ├── load_artifacts.py   # cached loaders
│       └── prepare_input.py    # form values → model feature row
├── artifacts/
│   ├── model.joblib            # fitted estimator (or full pipeline)
│   ├── meta.json               # threshold, feature list, metrics summary
│   └── figures/                # optional PNGs for Section C
├── telco-data/
│   ├── X_test.csv
│   ├── y_test.csv
│   └── ...                     # train files used for training only
├── notebooks/                  # optional: move existing notebooks here later
├── scripts/
│   └── export_artifacts.py     # notebook logic → saved files
├── requirements.txt
├── packages.txt                # only if system libs needed (often empty/unused)
├── .gitignore
├── README.md
└── streamlit-app-implementation-plan.md   # this file
```

**Community Cloud note:** you will point Cloud at one entry file, commonly `app/Home.py` or `app.py` at the repo root. Pick one convention and stick to it.

### 6.3 What gets saved in `artifacts/`

| File | Contents |
|---|---|
| `model.joblib` | Fitted sklearn model (ideally a `Pipeline` that includes any encoding/scaling still required at inference) |
| `meta.json` | `feature_names` (exact order), `threshold`, `model_name`, `metrics` (test precision/recall/F1/ROC-AUC), short `selection_rationale` text |
| `test_predictions.csv` (optional) | Precomputed test predictions for fast explorer / charts |
| `figures/*.png` (optional) | Confusion matrix, ROC, comparison charts |

**Critical:** the feature order in `meta.json` must match training. A wrong column order silently breaks predictions.

---

## 7. Implementation guide (recommended stack)

This section is the “how exactly” guide using **Streamlit + joblib + pandas + GitHub + Community Cloud**.

### Phase 0 — Team decision (before coding the app)

1. Freeze the winning model configuration from your comparison work (example: tuned Random Forest, class weight choice, decision threshold).
2. Write 5–10 bullet points in plain English: *why this model won* (metrics + business tradeoff). These bullets become Section C copy.
3. Agree on the human-facing form fields (readable dropdowns) vs raw encoded columns.

### Phase 1 — Export artifacts from the notebook (do this once)

**Goal:** the app never depends on re-running GridSearch.

In a notebook or `scripts/export_artifacts.py`:

1. Load `X_train` / `y_train`, train the **final** model (or re-load the already-fitted object from the notebook session).
2. Save the model:

```python
import joblib
import json
from pathlib import Path

Path("artifacts").mkdir(exist_ok=True)
joblib.dump(final_model, "artifacts/model.joblib")

meta = {
    "model_name": "RandomForestClassifier (final config)",
    "feature_names": list(X_train.columns),
    "threshold": float(best_threshold),  # e.g. 0.54
    "target_labels": {"0": "No churn", "1": "Churn"},
    "metrics_test": {
        "accuracy": ...,
        "precision_churn": ...,
        "recall_churn": ...,
        "f1_churn": ...,
        "roc_auc": ...
    },
    "selection_rationale": [
        "Chosen for best balance of recall and precision under our retention costs.",
        "Outperformed baseline unconstrained forest on validation F1 after tuning.",
        "Threshold raised/lowered from 0.5 to hit the agreed operating point."
    ]
}
Path("artifacts/meta.json").write_text(json.dumps(meta, indent=2))
```

3. Optionally save test predictions:

```python
import pandas as pd
proba = final_model.predict_proba(X_test)[:, 1]
pred = (proba >= meta["threshold"]).astype(int)
pd.DataFrame({
    "y_true": y_test.astype(int).values,
    "y_pred": pred,
    "churn_probability": proba
}).to_csv("artifacts/test_predictions.csv", index=True)
```

4. Confirm reload works in a fresh Python process:

```python
model = joblib.load("artifacts/model.joblib")
assert list(X_test.columns) == meta["feature_names"]
```

5. Commit `artifacts/` (if file size is acceptable) so Cloud can load them.

### Phase 2 — Create the Python environment

```bash
cd /Users/vicch/Git/netflix-churn-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit pandas numpy scikit-learn joblib plotly matplotlib
pip freeze > requirements.txt
```

**Beginner tip:** after the app works, trim `requirements.txt` to packages you actually import (plus their needed versions). Bloated freezes are OK for learning; just keep sklearn versions consistent with the machine that created `model.joblib`.

Add to `.gitignore` if not already present:

```text
.venv/
__pycache__/
.DS_Store
```

Do **not** ignore `artifacts/` if Cloud needs those files from GitHub.

### Phase 3 — Shared helpers

Create `app/utils/load_artifacts.py`:

```python
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]  # repo root
ART = ROOT / "artifacts"
DATA = ROOT / "telco-data"

@st.cache_resource
def load_model():
    return joblib.load(ART / "model.joblib")

@st.cache_data
def load_meta():
    return json.loads((ART / "meta.json").read_text())

@st.cache_data
def load_test_xy():
    X_test = pd.read_csv(DATA / "X_test.csv")
    y_test = pd.read_csv(DATA / "y_test.csv").squeeze("columns")
    # normalize labels to 0/1 if needed
    if y_test.dtype == object:
        y_test = y_test.map({"Yes": 1, "No": 0})
    return X_test, y_test
```

**Why `@st.cache_resource` / `@st.cache_data`?**  
Streamlit re-runs your script whenever a widget changes. Caching prevents reloading the model from disk on every click.

Create `app/utils/prepare_input.py` to map form values → one-row DataFrame with **exact** `feature_names` order.

Because this repo’s `X_*.csv` features are already numeric/encoded, you have two design options:

| Option | Approach | Beginner recommendation |
|---|---|---|
| **A. Expert form** | UI asks for encoded values directly | Faster to code; worse UX |
| **B. Friendly form + mapping** | UI asks “Fiber optic?” etc., then code one-hot / ordinal maps to match training | Better UX; recommended |

Implement Option B carefully using the **same encoding rules** as `feature-engineering.ipynb`. If that feels risky, ship Option A first, then improve labels.

Prediction helper:

```python
import numpy as np
import pandas as pd

def predict_churn(model, meta, feature_row: pd.DataFrame):
    feature_row = feature_row[meta["feature_names"]]  # enforce order
    proba = float(model.predict_proba(feature_row)[0, 1])
    pred = int(proba >= meta["threshold"])
    label = meta["target_labels"][str(pred)]
    return {"prediction": pred, "label": label, "probability": proba}
```

### Phase 4 — Build Section A (Predict page)

File: `app/pages/1_Predict.py`

1. Title + 1–2 sentence explanation: “Enter customer attributes. The model estimates churn risk.”
2. Build form with `st.selectbox`, `st.number_input`, `st.radio` for each human field.
3. On submit (`st.form` + `st.form_submit_button`):
   - convert inputs → feature row
   - call `predict_churn`
   - display:
     - big result: **Likely to churn** / **Likely to stay**
     - probability as `st.metric` or progress bar
     - threshold note: “We flag churn when probability ≥ {threshold}”
4. Add a disclaimer: educational demo on public Telco data; not financial advice.

### Phase 5 — Build Section B (Test explorer)

File: `app/pages/2_Test_Explorer.py`

1. Load `X_test`, `y_test`, model, meta (cached).
2. Let user pick a row index (`st.number_input` or `st.selectbox`).
3. Show the customer’s features (`st.dataframe` of one row transposed for readability).
4. Run prediction for that row.
5. Show a comparison panel:

| Field | Example |
|---|---|
| Predicted | Churn |
| Probability | 0.72 |
| Actual | No churn |
| Outcome | Incorrect (False Positive) |

6. Optional filters: “show only mistakes”, “show only true churners”.
7. Banner text: “These customers were **not** used to train the model.”

### Phase 6 — Build Section C (Effectiveness)

File: `app/pages/3_Model_Effectiveness.py`

1. Load `meta.json` metrics and rationale bullets.
2. Display metric cards with `st.metric`.
3. Charts (pick one path):
   - **Simple path:** `st.image("artifacts/figures/confusion_matrix.png")` etc.
   - **Interactive path:** compute confusion matrix / ROC from `test_predictions.csv` with Plotly.
4. Short comparison table vs rejected approaches (manual markdown table is fine).
5. Explain threshold choice in business language (false negatives vs false positives for retention outreach).

### Phase 7 — Home / navigation

File: `app/Home.py` (or root `app.py`):

- Project intro
- Links/instructions for the three pages
- Dataset citation / project status

Streamlit multipage: files in `pages/` appear in the sidebar automatically when you run the main entrypoint.

### Phase 8 — Run locally

```bash
source .venv/bin/activate
streamlit run app/Home.py
```

Browser opens (usually `http://localhost:8501`). Click through all three sections.

**Sanity check:** pick a known `X_test` row in Section B and confirm probability matches what the notebook produced for that same row (within floating-point tolerance).

### Phase 9 — Clean up for deployment

1. Ensure `requirements.txt` includes at least:

```text
streamlit
pandas
numpy
scikit-learn
joblib
plotly
matplotlib
```

Pin versions once stable, for example:

```text
streamlit==1.39.0
scikit-learn==1.5.2
...
```

2. Confirm artifacts paths work when the working directory is the **repo root** (Cloud runs from repo root).
3. Remove huge unused files from the repo if any (do not upload `venv/` or notebook checkpoints).
4. Update README with Local Run + Deploy sections.

---

## 8. Deploying to Streamlit Community Cloud

### 8.1 What you need before clicking Deploy

| Checklist item | Notes |
|---|---|
| GitHub account | Free is fine |
| Streamlit Community Cloud account | Sign in with GitHub at [https://share.streamlit.io](https://share.streamlit.io) |
| Repo pushed to GitHub | Public repo is simplest on free tier |
| Working entry file | e.g. `app/Home.py` |
| `requirements.txt` at repo root (typical) | Cloud looks here by default |
| Model/data files committed | `artifacts/`, needed `telco-data/` test files |

### 8.2 Push the project to GitHub

If the repo is not on GitHub yet:

1. Create a new GitHub repository (website UI).
2. Locally:

```bash
git add app artifacts requirements.txt README.md telco-data/X_test.csv telco-data/y_test.csv
# add other needed files; avoid venv
git commit -m "Add Streamlit churn app and model artifacts for Community Cloud"
git push -u origin HEAD
```

(Use your team’s branch/PR workflow if you already have one.)

### 8.3 Deploy in the Cloud UI (click-by-click)

1. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select:
   - **Repository:** your churn project
   - **Branch:** `main` (or your deploy branch)
   - **Main file path:** `app/Home.py` (exact path)
4. Click **Deploy**.
5. Wait while Cloud clones the repo and `pip install`s dependencies.
6. When the status is healthy, open the public URL and test all three sections.

### 8.4 After the first deploy

- Every push to the deployed branch can trigger a redeploy.
- If the app fails, open **Manage app** → logs. Common errors:
  - `ModuleNotFoundError` → missing package in `requirements.txt`
  - `FileNotFoundError` → wrong path to `artifacts/` or CSV
  - model load error → sklearn version mismatch between training machine and Cloud
- To fix version mismatch: recreate `model.joblib` in an environment matching Cloud’s installed `scikit-learn`, or pin sklearn in `requirements.txt` to the training version and reboot the app.

### 8.5 Secrets (only if needed)

v1 should need **no secrets**. If you later add an API key, use Streamlit Cloud **Secrets** (TOML) instead of committing keys to GitHub.

### 8.6 What “success” looks like on Cloud

- Public URL loads without errors
- Predict form returns a label + probability
- Test explorer shows actual vs predicted for held-out rows
- Effectiveness page shows metrics/rationale
- A teammate can open the URL on their phone/laptop without installing Python

---

## 9. Testing checklist

### Local

- [ ] `streamlit run` starts without import errors
- [ ] Model loads once (watch for slow reload on every widget — fix with cache)
- [ ] Form rejects incomplete input
- [ ] Prediction changes when inputs change in a sensible direction (spot-check)
- [ ] Test explorer row 0 matches notebook prediction for row 0
- [ ] No training rows appear in explorer
- [ ] Effectiveness metrics match notebook/test evaluation

### Cloud

- [ ] Deploy succeeds
- [ ] Cold start works after idle sleep
- [ ] Paths work on Linux (Cloud) — avoid Windows-only path assumptions
- [ ] App stays within free resource limits (avoid huge grids or re-training)

### Team / demo

- [ ] One-minute demo script: Predict → Explorer → Effectiveness
- [ ] Someone who did not train the model can use the form without help

---

## 10. Phased timeline

Suggested order for a small student team (adjust to your calendar):

| Phase | Work | Rough effort |
|---|---|---|
| 0 | Freeze model + write rationale bullets | 0.5–1 day |
| 1 | Export `model.joblib` + `meta.json` | 0.5 day |
| 2 | Repo layout + requirements + local Streamlit hello world | 0.5 day |
| 3 | Section A predict form | 1–2 days |
| 4 | Section B test explorer | 1 day |
| 5 | Section C effectiveness page | 1 day |
| 6 | Polish copy, disclaimers, README | 0.5 day |
| 7 | GitHub + Community Cloud deploy + bugfix | 0.5–1 day |

**Parallelization tip:** one teammate can draft Section C narrative/figures while another builds the predict form, once artifacts exist.

---

## 11. Risks and beginner pitfalls

| Pitfall | What goes wrong | How to avoid |
|---|---|---|
| Retraining inside Streamlit | Slow app; non-reproducible; Cloud timeouts | Load joblib artifacts only |
| Feature order mismatch | Silent wrong predictions | Save `feature_names` and subset columns explicitly |
| Encoding mismatch | Form “Yes/No” ≠ model’s 0/1 or one-hots | Reuse exact feature-engineering rules; add unit-like asserts |
| Peeking at train in explorer | Invalid “unseen” demo | Only ship `X_test`/`y_test` to that page |
| Unpinned sklearn | Cloud can’t load model | Pin versions; re-export model if needed |
| Committing `.venv` | Huge/slow repo | gitignore the venv |
| Giant notebook as the app | Unmaintainable | Separate `app/` from notebooks |
| No caching | App feels broken/slow | `@st.cache_resource` for model |
| Threshold forgotten | App uses 0.5 while notebook used tuned cutoff | Store threshold in `meta.json` and apply it |

---

## 12. Glossary

| Term | Plain meaning |
|---|---|
| **Streamlit** | Python library that creates a website UI from a script |
| **Community Cloud** | Free host that runs your Streamlit app from GitHub |
| **Artifact** | Saved file produced by training (model, metrics, plots) |
| **joblib** | Tool to save/load Python sklearn models to disk |
| **Threshold** | Probability cutoff; at/above → predict churn |
| **Test set** | Customers held out of training, used to estimate real-world performance |
| **False negative** | Customer who churned but model said “stay” (missed retention opportunity) |
| **False positive** | Customer who stayed but model said “churn” (unnecessary outreach) |
| **Cold start** | Delay when a sleeping free-tier app wakes up |
| **`requirements.txt`** | List of Python packages Cloud installs |

---

## Appendix A — Minimal “hello Streamlit” to build confidence

Before wiring the model, every teammate should successfully run:

```python
# hello_app.py
import streamlit as st

st.title("Hello Churn Team")
name = st.text_input("Your name")
if name:
    st.write(f"Ready to build, {name}!")
```

```bash
streamlit run hello_app.py
```

If this works locally, you are ready for Phases 3–8.

---

## Appendix B — Decision summary

| Decision | Choice | One-line why |
|---|---|---|
| UI framework | Streamlit | Matches team goal + easiest Python ML demo path |
| Hosting | Streamlit Community Cloud | Free, GitHub-native, beginner-friendly |
| Model delivery | joblib artifact + meta.json | No retraining in the app; reproducible |
| Data for explorer | `X_test` / `y_test` only | Honest “unseen customer” demo |
| Charts | Plotly and/or static PNGs | Communicates effectiveness clearly |
| Deps | requirements.txt | What Community Cloud expects |

---

## Appendix C — Mapping to this repository today

| Existing asset | How the app uses it |
|---|---|
| `telco-data/X_test.csv`, `y_test.csv` | Section B explorer (+ optional Section C charts) |
| `telco-data/X_train.csv`, `y_train.csv` | Training/export only — not for explorer |
| `random-forest-model.ipynb` | Source of final model + metrics narrative for Section C |
| `feature-engineering.ipynb` | Source of encoding rules for the predict form mapper |
| `eda-telco.ipynb` | Optional context / charts for effectiveness storytelling |
| `README.md` | Extend with app run + deploy instructions when built |

When the final model is chosen, treat artifact export as the bridge between notebooks and the Streamlit product.

---

*End of plan. Recommended next action: freeze the model, export `artifacts/model.joblib` + `artifacts/meta.json`, then scaffold `app/Home.py` and deploy a stub page to Community Cloud early so hosting issues are discovered before the UI is finished.*
