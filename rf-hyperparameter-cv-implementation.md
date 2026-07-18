# Implementation notes: Random Forest Hyperparameter Tuning (GridSearchCV)

This document records what was implemented in `random-forest-model.ipynb` from `rf-hyperparameter-cv-plan.md` during the chat session that added the F1-maximizing `GridSearchCV` section. It is meant for reviewing **decisions**, **deviations**, and **how the new work sits in the notebook**.

Related docs:

| File | Role |
|---|---|
| `rf-hyperparameter-cv-plan.md` | Spec / acceptance checklist for this work |
| `rf-threshold-cv-plan.md` | Follow-up (threshold CV) — **not** implemented here |
| `random-forest-model.ipynb` | Target notebook |

---

## Goal of this implementation

Add a structured hyperparameter search after the existing overfitting discussion so tree-growth params (`max_depth`, `min_samples_leaf`, `min_samples_split`) are chosen by **mean CV F1 only**, then evaluate the winner once on the held-out test set at threshold 0.5 — without changing the baseline `rf` / `rf_balanced` cells.

---

## Steps actually taken

1. **Clarifying questions (then defaults)**  
   Asked for grid size (full vs leaner) and comparison-table style (dynamic code vs hardcoded markdown). User replied they did not care, so defaults matching the plan’s preferred stance were used (see Decisions below).

2. **Extended existing imports (cell 0)**  
   Added `GridSearchCV`, `StratifiedKFold`, and `f1_score` / `precision_score` / `recall_score`. Did **not** add a second import cell later in the notebook.

3. **Appended a new section after the overfitting markdown (after cell 26)**  
   Left all earlier training, evaluation, comparison, and overfitting cells for `rf` / `rf_balanced` intact.

4. **Configured and fit stratified `GridSearchCV` on train only**  
   - Estimator base: `class_weight="balanced"`, `n_estimators=100`, `random_state=47`, `n_jobs=-1` (same spirit as `rf_balanced`).  
   - Full grid: `max_depth` × `min_samples_leaf` × `min_samples_split` → 60 configs × 5 folds.  
   - Multi-metric `scoring` dict; `refit="f1"`; `return_train_score=True`.  
   - Named winner `rf_tuned = gs.best_estimator_`.

5. **Built a top-10 CV tradeoff table** from `gs.cv_results_` with the columns specified in the plan (params, mean/std F1, precision, recall, accuracy, ROC-AUC, mean train F1, rank).

6. **Mirrored existing evaluation style for `rf_tuned` on test**  
   Accuracy → classification report → ROC-AUC → 1×2 confusion-matrix + ROC plot (same layout as the baseline plot cell).

7. **Added an overfitting check for `rf_tuned`**  
   Train/test accuracy gap, plus train/test **F1** at 0.5 (optional in the plan; included here).

8. **Added a three-way comparison** (`rf` / `rf_balanced` / `rf_tuned`) as a **code-generated** table with a “Change vs `rf_balanced`” column.

9. **Documented selection/decision rules in markdown**  
   Section intro + end-of-section decision-rule cell stating F1-only CV selection, secondary metrics for tradeoffs only, params frozen before test, threshold 0.5 for this section, threshold tuning deferred.

---

## Decisions made (and why)

| Decision | Choice | Why |
|---|---|---|
| Grid size | **Full** 60-combo grid | Plan’s primary recommendation; user had no preference. Targets the overfitting levers named in the notebook (`max_depth` / leaf / split). |
| Starting estimator | Same as `rf_balanced` + fixed `n_estimators=100` | Plan context: balance class weight already helped recall; first pass should regularize growth, not retune tree count. |
| Selection metric | **`refit="f1"` only** | Plan rule: do not blend metrics; F1 is churn-relevant under imbalance (~26.5% churn). |
| Where to place the section | **After** existing overfitting discussion | Plan: keep baseline cells intact; hyperparameter search is the “next step” after diagnosing the ~0.22 gap. |
| Imports | **Extend cell 0** | Avoid duplicate imports mid-notebook; matches how the notebook already centralizes sklearn imports. |
| Comparison table | **Dynamic code cell** | User had no preference. Code stays correct after re-runs and avoids hardcoding numbers before GridSearchCV outputs exist. |
| Train/test F1 in overfitting check | **Included** | Plan listed it as optional but “more aligned with the selection metric”; cheap to add and useful next to accuracy gap. |
| Threshold tuning | **Not implemented** | Explicitly out of scope; belongs in `rf-threshold-cv-plan.md` after freezing `rf_tuned`. |
| Running the search in-chat | **Not run during implementation** | ~300 fits; notebook cells were authored for the user to execute end-to-end. Outputs/tables populate on run. |

---

## Where this implementation followed the plan

- New section titled along the lines of “Hyperparameter tuning (GridSearchCV, maximize F1)”.
- Stratified 5-fold CV on `X_train` / `y_train` only; no test-set use for param selection.
- Full `param_grid` and multi-metric scoring with `refit="f1"`.
- Top CV results table for tradeoff inspection.
- One held-out test evaluation for `rf_tuned` in the same fashion as existing models.
- Overfitting check for the tuned model.
- Three-way comparison including secondary metrics and gap.
- Explicit decision-rule documentation (F1-only selection; threshold 0.5; threshold CV separate).
- Baseline `rf` / `rf_balanced` path left unbroken.

---

## Deviations from the plan (and why)

| Plan text / suggestion | What we did | Why it differed |
|---|---|---|
| Comparison shown as a **markdown** table with filled numeric cells (like the existing `rf` vs `rf_balanced` table) | **Code** cell builds a `DataFrame` of metrics and displays it | User declined to choose style; dynamic table is reproducible after re-fitting and does not require pre-running GridSearchCV to paste numbers. The older two-way markdown comparison (earlier in the notebook) was left as-is. |
| Optional markdown notes **immediately after** the CV top-10 table (rank-2/3 recall tradeoff, unconstrained overfitting signal) | Interpretation notes live **after** the three-way comparison, not as a dedicated post-CV narrative cell | Same content goals (F1-only rule, gap, recall/FN, business preference, don’t silently switch winners) without duplicating long prose before test metrics exist. Unconstrained overfitting signal is still inspectable via `mean_train_f1` vs `mean_test_f1` in the CV table itself. |
| Suggested implementation order listed “decision-rule documentation” as a late step | Brief selection rules also appear in the **section intro** markdown, with a fuller “Decision rule (documented)” cell at the end | Makes the F1-only rule visible before the expensive fit, and still satisfies the plan’s documentation requirement at the close of the section. |
| Leaner grid / `RandomizedSearchCV` called out as alternatives | Not used | User had no preference; stuck to the plan’s primary full-grid path. |
| Acceptance checklist implies runnable verified outputs | Cells authored but **not executed** in this chat | Runtime cost; verification is “run notebook top-to-bottom” by the user/CI later. |

**Non-deviations worth stating:** the plan’s “out of scope” items (threshold CV, searching `n_estimators`, nested CV, custom composite scores, persisting the model) were intentionally **not** added.

---

## How this fits into `random-forest-model.ipynb` structure

High-level notebook flow after this change:

```text
[Setup]
  Imports (incl. GridSearchCV / StratifiedKFold / F1-related metrics)
  Load telco-data train/test CSVs
  Label sanity check

[Baseline path — unchanged]
  Train `rf` (default)
  Predict + evaluate at threshold 0.5
  Train `rf_balanced` (class_weight="balanced")
  Evaluate + compare `rf` vs `rf_balanced` (markdown table)
  Overfitting check for both (~0.22+ gap) + “next step” note

[New: hyperparameter tuning path]
  ## Hyperparameter tuning (GridSearchCV, maximize F1)
  GridSearchCV fit → `rf_tuned`
  CV tradeoff table (top 10 by mean F1)
  ### Evaluating `rf_tuned` … (test @ 0.5)
  ### Overfitting check (`rf_tuned`)
  ### Comparing `rf` vs `rf_balanced` vs `rf_tuned`
  Interpretation notes
  ### Decision rule (documented)
```

### Cell map for the new section (as of this implementation)

| Approx. cell role | Content |
|---|---|
| Section intro | Why F1-only CV; train-only search; test held out |
| `GridSearchCV` setup + `fit` | `param_grid`, `scoring`, `refit="f1"`, `rf_tuned` |
| CV tradeoff table | Top 10 by `rank_test_f1` |
| Test evaluation | Accuracy, report, ROC-AUC, CM + ROC plot |
| Overfitting check | Accuracy gap + F1 gap |
| Three-way comparison | Code table + change vs `rf_balanced` |
| Closing markdown | Interpretation + formal decision rule |

### Structural relationships

- **Depends on earlier cells:** `X_train` / `y_train` / `X_test` / `y_test`, and for the three-way table also `rf`, `rf_balanced`, and their existing test predictions/probabilities.
- **Does not replace** the earlier two-model comparison or overfitting cells; it **extends** the story after the overfitting diagnosis.
- **Preferred future order** (from the plan, not yet in this notebook): freeze `rf_tuned` params → run threshold CV → one final test evaluation of model + threshold.

---

## Acceptance checklist status (implementation chat)

| Checklist item | Status |
|---|---|
| New section without breaking baseline cells | Done |
| Search uses only train via stratified `GridSearchCV` | Done (in code) |
| Scoring includes F1, precision, recall, accuracy, ROC-AUC | Done |
| `refit="f1"` only selects winner | Done |
| Top CV results table | Done |
| `rf_tuned` evaluated once on test | Done (cells ready; run to populate) |
| Overfitting check for `rf_tuned` | Done |
| Comparison includes all three models | Done (dynamic table) |
| F1-only selection documented | Done |
| Test not used to choose hyperparameters | Done (by design) |

---

## Practical notes for re-running / review

- Re-run the notebook from the top (or at least from imports through the new section) before treating printed `best_params_` / metrics as authoritative.
- Expect slower runtime on the GridSearchCV cell (~300 fits with `n_jobs=-1`).
- If the train/test gap shrinks but churn F1 collapses, the plan’s guidance still applies: widen candidates toward milder regularization rather than changing the selection metric mid-notebook.
- Next logical notebook addition (separate plan): threshold CV on frozen `rf_tuned` (`rf-threshold-cv-plan.md`).
