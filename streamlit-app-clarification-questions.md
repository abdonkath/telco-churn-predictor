# Team Clarification Shortlist — Streamlit Churn App

**Timebox:** ~20–25 minutes  
**Goal:** Lock the decisions that most affect app architecture before implementation.  
**Related:** full plan in `streamlit-app-implementation-plan.md`

---

## Agenda questions

### 1. Freeze one model or allow switching? (~3 min)

- Do we ship **exactly one** frozen model in the app, or a **model switcher** (e.g. Logistic Regression vs Random Forest)?
- If one model: who has **final say**, and by what **primary metric** (F1, recall@precision floor, ROC-AUC, business cost)?
- Do we freeze **before** building the app UI, or scaffold the app in parallel while tuning finishes?

**Decision:** Freeze one model prior to building app UI, F1, recall, accuracy, precision, freezing XGboost

---



### 2. Fixed threshold or user-adjustable? (~2 min)

- Is the decision threshold **fixed** in the saved artifact, or can users **move a slider** and see predictions change?
- If fixed: confirm we store it in something like `meta.json` and use it everywhere (live predict + test explorer).

**Decision: Threshold fixed**

---



### 3. Prediction form: raw business fields vs encoded features? (~3 min)

- Should Section A collect **human-readable Telco fields** (Contract, InternetService, etc.) and engineer features in the app, or collect values that already match `X_*.csv`?
- Are derived fields like `tenure_group` **hidden and computed**, or shown in the form?
- Confirm: form output must match the **same 23-column schema** the frozen model was trained on.

**Decision: Collect Telco fields from app user, tenure_group hidden**

---



### 4. Canonical data + what “test explorer” means (~2 min)

- Are `telco-data/X_test.csv` + `y_test.csv` the **locked** holdout for Section B, or might we regenerate splits later?
- Confirm: explorer shows **test rows only** (never train).
- Nice-to-have now or later: `customerID`, filters for correct/incorrect predictions?

**Decision: Show test rows only, filter for correct/incorrect predidiction**

---



### 5. Audience, tone, and branding (~3 min)

- Primary audience for v1: **business owner demo**, classmates/graders, or recruiters?
- UI tone: **business language** (“likely to leave”) vs **ML language** (p-value, class label)?
- Product framing: **Telco churn** (matches data) vs leaning into the `netflix-churn-predictor` repo name in copy?

**Decision: business owner, business language on app UI, Telco churn**

---



### 6. UI shape for the three sections (~2 min)

- **Multipage** sidebar (Predict / Test explorer / Effectiveness) vs **one scrolling page**?
- Visual bar: Streamlit defaults OK, or light custom theme/logo?
- Desktop-only for demos, or mobile matters?

**Decision:**  One scrolling page, desktop-only, order of sections A, B, then C

---



### 7. Deploy plan still Community Cloud? (~3 min)

- Confirm **Streamlit Community Cloud** + **GitHub** as the path (vs Hugging Face / local-only).
- Can the repo be **public**? Whose GitHub/Streamlit account **owns** the app?
- OK to commit **model artifact + test CSVs** to the repo for Cloud to load?

**Decision: Community Cloud**

---



### 8. v1 scope and definition of done (~3 min)

- If time runs out, rank must-haves: **A live predict** / **B test explorer** / **C effectiveness** (1 = required).
- Done means: **local only**, or **public Cloud URL** required?
- Learning/portfolio bar: “clear and working” vs “feels like a real product”? Any required disclaimer?

**Decision:** 