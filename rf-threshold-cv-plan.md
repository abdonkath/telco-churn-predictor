# Plan: Threshold Tuning via Cross-Validation (Option B)

Implement Approach 2 (fine grid search maximizing F1) in `random-forest-model.ipynb` by choosing the decision threshold with stratified CV on the training set, freezing that threshold, then evaluating once on the held-out test set.

---

## Pros and cons of Option B (this plan)

### Pros

- More stable threshold: `best_t` is the cutoff that works best **on average** across folds, not one lucky validation slice.
- Uses the full training set over the course of CV (each training row appears in a validation fold once).
- Lower risk of overfitting the cutoff to a single split or to the test set.
- Better default when the threshold will be treated as part of the final system (model + cutoff).

### Cons

- More code and slower runtime (fit the Random Forest once per fold, e.g. 5× for 5-fold CV).
- Slightly harder to explain at a glance (“mean F1 over folds”) than a single validation table.
- CV estimate can still be a bit optimistic vs true future data — but much better than tuning on test.
- If hyperparameter tuning is added later, nesting gets more complex (inner CV for params/threshold, outer for final evaluation).

---

## What Option A would look like instead (summary)

**Option A — single validation split**

1. Stratified-split current `X_train` / `y_train` into `X_fit` / `X_val` (e.g. 80/20).
2. Fit the Random Forest on `X_fit` only.
3. Run the fine F1 threshold grid on `X_val` probabilities; pick `best_t`.
4. Optionally refit on full `X_train` with the same model settings.
5. Apply frozen `best_t` once on `X_test` and report metrics.

Faster and easier to debug, but `best_t` depends more on that one validation slice. Prefer Option A for a quick first pass; use Option B (this plan) when locking in a more trustworthy operating threshold.

---

## Context in this project

| Item | Detail |
|---|---|
| Notebook | `random-forest-model.ipynb` |
| Features | Pre-engineered CSVs in `telco-data/` (`X_train`, `X_test`, `y_train`, `y_test`) |
| Current models | `rf` (default) and `rf_balanced` (`class_weight="balanced"`) |
| Current cutoff | sklearn default **0.5** via `predict()` |
| Why this matters | Churn is ~26.5% of labels; missing churners is costlier than false alarms |
| Recommended base model for this section | `rf_balanced` (already better churn recall / F1) |

Do **not** search thresholds on `X_test`. Test is for a single final evaluation only.

---

## Implementation plan

### 1. Add a new notebook section after the existing evaluation / overfitting cells

Suggested heading:

```text
## Threshold tuning (cross-validation)
```

Place it after the current overfitting discussion (end of notebook), so the baseline `rf` vs `rf_balanced` comparison at 0.5 remains intact.

### 2. Imports

Add (or extend existing imports):

- `StratifiedKFold` from `sklearn.model_selection`
- `f1_score`, `precision_score`, `recall_score` from `sklearn.metrics` (if not already imported)

`numpy` is already available.

### 3. CV threshold search on training data only

Configuration to use:

| Setting | Value | Reason |
|---|---|---|
| Model | `RandomForestClassifier(n_estimators=100, random_state=47, n_jobs=-1, class_weight="balanced")` | Matches project-preferred imbalance handling |
| Folds | `StratifiedKFold(n_splits=5, shuffle=True, random_state=47)` | Preserves churn rate per fold; reproducible |
| Threshold grid | `np.arange(0.1, 0.9, 0.01)` | Approach 2 fine search |
| Objective | Maximize **mean** F1 across folds for the **Churn** class (positive label = 1) | Aligns with wanting both precision and recall |

Algorithm:

1. For each fold: fit RF on the fold’s train indices; get `predict_proba` on the fold’s val indices.
2. For each threshold `t` in the grid: compute F1 on that fold’s validation predictions.
3. Average F1 across folds for each `t`.
4. Set `best_t = argmax(mean_f1)`.
5. Print `best_t` and its mean CV F1 (optional: also print mean precision/recall at `best_t` for context).

### 4. Optional: small results table / plot

Keep the notebook readable:

- DataFrame of a few thresholds (or the full grid) with mean CV precision, recall, F1.
- Optional line plot: threshold vs mean CV F1, with a vertical line at `best_t`.

This preserves the transparency of Approach 1 while still using Approach 2’s automatic pick.

### 5. Freeze threshold and fit final model on full training set

1. Fit `rf_thresholded` (same hyperparameters as above) on **all** of `X_train` / `y_train`.
2. Compute test probabilities: `test_proba = rf_thresholded.predict_proba(X_test)[:, 1]`.
3. Apply frozen cutoff once:

```python
y_test_hat = (test_proba >= best_t).astype(int)
```

### 6. Evaluate once on the test set (same fashion as existing notebook)

Mirror the evaluation style already established for `rf` and `rf_balanced` so results are easy to compare side by side. Reuse the same helpers and display patterns (`accuracy_score`, `classification_report`, `roc_auc_score`, `ConfusionMatrixDisplay`, `RocCurveDisplay`).

Suggested subsection heading:

```text
### Evaluating `rf_thresholded` @ best_t
```

Run these cells in the **same order** as the existing evaluation block:

1. **Accuracy (baseline)** — `accuracy_score(y_test, y_test_hat)`, same print style as the current accuracy cell.
2. **Classification report** — `classification_report(..., target_names=["No churn", "Churn"])`, so precision / recall / F1 for both classes match the notebook’s existing output format.
3. **ROC-AUC** — `roc_auc_score(y_test, test_proba)`. Note in markdown that ROC-AUC is threshold-independent (ranking quality); it should be nearly identical to `rf_balanced` if the same forest settings are used.
4. **Confusion matrix + ROC curve plot** — same 1×2 layout as the baseline `rf` plot (`ConfusionMatrixDisplay` + `RocCurveDisplay.from_predictions`), with a title that includes `best_t` (e.g. `"Confusion Matrix (threshold={best_t:.2f})"`).

Short markdown note: test metrics at `best_t` are the honest estimate of the **full system** (model + cutoff); CV F1 is only for choosing `best_t`.

### 7. Compare effectiveness vs the other models (same comparison fashion)

Add a comparison section that extends the notebook’s existing  
`### Comparing rf vs rf_balanced (Churn class)` table to a **three-way** comparison on the **same test set**.

Suggested heading:

```text
### Comparing `rf` vs `rf_balanced` vs `rf_thresholded` (Churn class)
```

**Models / decision rules to include**

| Label in table | Model | How labels are produced |
|---|---|---|
| `rf` (baseline) | existing `rf` | `predict()` → threshold **0.5** |
| `rf_balanced` | existing `rf_balanced` | `predict()` → threshold **0.5** |
| `rf_thresholded` | new final RF (`class_weight="balanced"`) | `(proba >= best_t)` with frozen CV threshold |

Reuse already-computed test outputs where possible (`y_pred`, `y_pred_balanced`, `y_pred_proba` / `y_pred_proba_balanced`) so the comparison stays consistent with earlier cells.

**Metrics table (match existing columns + Δ vs `rf_balanced`)**

Build a markdown or `DataFrame` table with the same churn-focused metrics the notebook already uses:

| Metric | `rf` (0.5) | `rf_balanced` (0.5) | `rf_thresholded` (`best_t`) | Change vs `rf_balanced` |
|---|---|---|---|---|
| Precision | … | … | … | … |
| Recall | … | … | … | … |
| F1-score | … | … | … | … |
| False Negatives | … | … | … | … |
| False Positives | … | … | … | … |
| Accuracy | … | … | … | … |
| ROC-AUC | … | … | … | … |

Compute FP / FN from `confusion_matrix(y_test, y_hat).ravel()` the same way the existing comparison derived 210 / 131 FN and 111 / 203 FP.

**Optional code helper** (keeps the notebook DRY and consistent):

```python
def churn_metrics(y_true, y_hat, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    out = {
        "precision": precision_score(y_true, y_hat),
        "recall": recall_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat),
        "accuracy": accuracy_score(y_true, y_hat),
        "fn": fn,
        "fp": fp,
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return out
```

**Side-by-side confusion matrices**

Plot three confusion matrices in one row (or `rf_balanced` vs `rf_thresholded` if space is tight), using the same `ConfusionMatrixDisplay` + `"Blues"` cmap style as the current notebook plots. Goal: visually compare missed churners (FN) and false alarms (FP).

**Interpretation markdown (same tone as existing comparison cell)**

After the table, add a short narrative parallel to the current `rf` vs `rf_balanced` write-up:

- Did F1 improve vs `rf_balanced` @ 0.5? By how much?
- Did recall go up or down? What happened to false negatives vs false positives?
- Restate the business preference: missing a churner is costlier than a false alarm.
- Call out if accuracy moved in the opposite direction of churn F1/recall (expected tradeoff).
- Note that ROC-AUC should barely change if only the threshold changed on a similarly trained forest; large ROC-AUC gaps would imply a different model, not just a different cutoff.

### 8. Overfitting check (same fashion as existing section)

Mirror `### Overfitting check` for the thresholded system. Because accuracy at a non-0.5 threshold is not what `model.score()` uses, compute train/test metrics with the **same frozen `best_t`**:

```python
train_proba = rf_thresholded.predict_proba(X_train)[:, 1]
train_hat = (train_proba >= best_t).astype(int)
test_hat = y_test_hat  # already computed

train_acc_t = accuracy_score(y_train, train_hat)
test_acc_t = accuracy_score(y_test, test_hat)
print(f"Train accuracy (thresholded @ {best_t:.2f}): {train_acc_t:.3f}")
print(f"Test accuracy  (thresholded @ {best_t:.2f}): {test_acc_t:.3f}")
print(f"Gap:                                       {train_acc_t - test_acc_t:.3f}")
```

Optionally also print train vs test **F1 at `best_t`** (more relevant than accuracy for this project). Compare the gap verbally to the existing ~0.22 gaps for `rf` / `rf_balanced`: threshold tuning does **not** fix unconstrained tree overfitting; if the gap remains large, that still points to the hyperparameter follow-ups already noted at the end of the notebook.

### 9. Document the decision rule in the notebook

Add a brief markdown cell stating:

- Threshold was chosen by 5-fold stratified CV maximizing mean F1 on train.
- `best_t` was frozen before touching test labels for selection.
- Default 0.5 remains the reference for earlier cells; this section is the tuned operating point.
- Comparison tables use the same test set and metric definitions as the earlier `rf` vs `rf_balanced` section.

### 10. Out of scope for this plan (follow-ups)

- Hyperparameter tuning (`max_depth`, `min_samples_leaf`, etc.) to address overfitting — separate from threshold selection.
- Tuning threshold to maximize recall subject to a precision floor (business-constrained objective).
- Persisting the model + `best_t` to disk.

---

## Acceptance checklist

- [ ] New section added to `random-forest-model.ipynb` without breaking existing baseline cells
- [ ] Threshold search uses only `X_train` / `y_train` via stratified CV
- [ ] `best_t` is selected by mean CV F1 over `np.arange(0.1, 0.9, 0.01)`
- [ ] Final RF is refit on full train; test evaluated **once** at frozen `best_t`
- [ ] Standalone eval for `rf_thresholded` mirrors existing style: accuracy → classification report → ROC-AUC → CM + ROC plot
- [ ] Three-way comparison table (`rf` / `rf_balanced` / `rf_thresholded`) includes precision, recall, F1, FN, FP (and accuracy / ROC-AUC)
- [ ] Side-by-side confusion matrices included for visual FN/FP comparison
- [ ] Short interpretation markdown in the same style as the existing `rf` vs `rf_balanced` write-up
- [ ] Overfitting check reported for predictions at frozen `best_t` (train vs test)
- [ ] Notebook notes that test was not used to choose the threshold

---

## Suggested implementation order

1. Imports + section heading  
2. CV loop + `best_t`  
3. Optional CV metrics table/plot  
4. Refit on full train + apply frozen `best_t` on test  
5. Standalone eval block (accuracy → report → ROC-AUC → CM/ROC plot)  
6. Three-way comparison table + side-by-side confusion matrices + interpretation markdown  
7. Overfitting check at `best_t`  
8. Decision-rule documentation cell  
