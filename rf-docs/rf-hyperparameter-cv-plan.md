# Plan: Random Forest Hyperparameter Tuning via GridSearchCV

Implement a small structured search over tree-growth hyperparameters in `random-forest-model.ipynb` using stratified `GridSearchCV`. **Select the best configuration by F1 score only**; still track accuracy, precision, recall, ROC-AUC, and the train/test gap so tradeoffs are visible.

This plan addresses the overfitting gap (~0.22+) noted at the end of the notebook. It is separate from threshold tuning (`rf-threshold-cv-plan.md`).

---

## Why F1 is the selection metric (and not a blend)

| Role | Metric(s) | Purpose |
|---|---|---|
| **Selection (primary)** | **F1** (churn / positive class) | Single number `GridSearchCV` maximizes to pick `best_params_` |
| **Reporting (secondary)** | Precision, recall, accuracy, ROC-AUC, FN/FP, train vs test gap | Understand tradeoffs after a winner is chosen |

Do **not** blend F1 + accuracy + ROC-AUC into one composite score for selection. `mean_test_score` in `cv_results_` is simply the **mean CV F1** across folds (sklearn’s “test” = validation fold, not `X_test`).

Accuracy alone is a poor selector here: churn is ~26.5%, so high accuracy can hide weak churn detection. F1 balances precision and recall for the churn class, matching the project’s cost preference (missing a churner is worse than a false alarm, but both still matter).

---

## Strengths and weaknesses of this approach

### Strengths

- **Targets the real problem.** Constraining `max_depth` / `min_samples_leaf` / `min_samples_split` is the right lever for the ~0.22 train/test accuracy gap caused by unconstrained trees.
- **Automated and reproducible.** One `GridSearchCV.fit` evaluates the full small grid; results are driven by `random_state` + fixed folds, not ad hoc guessing.
- **Stable selection via stratified CV.** Mean F1 across folds is less sensitive to one lucky validation slice than a single 80/20 split.
- **Clear winner rule.** Maximizing F1 alone keeps the decision rule simple and churn-relevant; secondary metrics stay available for interpretation.
- **Tradeoffs stay visible.** Tracking precision, recall, accuracy, and ROC-AUC in `cv_results_` shows what the F1 winner costs (e.g. lower recall) without changing who wins.
- **Fits the project style.** Same spirit as the logistic-regression `GridSearchCV` workflow in the README and the F1-first threshold plan.

### Weaknesses

- **Runtime.** A 60-combo × 5-fold grid is ~300 fits — slower than trying one or two manual settings.
- **Grid explosion.** Adding more params or denser value lists (e.g. `n_estimators`, `max_features`) multiplies cost quickly.
- **Discrete candidates only.** The true optimum might sit between grid points (e.g. `min_samples_leaf=12` when only 10 and 20 are tried).
- **CV optimism.** Mean CV F1 can still be a bit higher than true future performance; it must not be confused with the final `X_test` score.
- **F1 is not the full business story.** An F1-best model might not maximize recall; costly false negatives need a human read of the tradeoff table (or a later threshold pass).
- **Composition with threshold tuning.** Doing hyperparameter search and threshold search carelessly can leak test information or overfit the operating point; keep them sequential (or nested).

---

## Other possible approaches (short summary)

These are reasonable alternatives or precursors; this plan still prefers small stratified `GridSearchCV` with `refit="f1"`.

| Approach | What it does | When it helps | Limitation vs this plan |
|---|---|---|---|
| **Manual one-knob trial** | Hand-set e.g. `min_samples_leaf=10` or `max_depth=10`, re-check the overfitting cell | Fast sanity check that regularization works | Easy to miss a better combo; less systematic |
| **Single validation split** | Stratified 80/20 on train; pick params by val F1 once | Quicker debugging / first pass | Winner depends more on one slice; noisier than 5-fold CV |
| **`RandomizedSearchCV`** | Sample `n_iter` combos from the same ranges | Larger spaces or tighter time budget | May miss a strong grid cell; slightly less exhaustive |
| **Leaner grid** | Search only `max_depth` + `min_samples_leaf` (fix split at 2) | Cut fits from ~300 to ~100 | Slightly less coverage of split-related regularization |
| **Tune `n_estimators` only** | Try 100 / 200 / 300 with unconstrained trees | Stabilize predictions after trees are already regularized | Alone does **not** close train≈0.99 overfitting |
| **Threshold tuning only** | Keep current forest; search cutoff for F1 (`rf-threshold-cv-plan.md`) | Improve operating precision/recall without re-fitting trees | Does **not** fix unconstrained tree memorization |
| **Nested CV** | Outer loop estimates generalization; inner loop picks params | Highest rigor for a published estimate | More code and much slower; overkill for this notebook stage |
| **Composite / cost-weighted score** | Optimize a custom blend of F1, recall, accuracy, etc. | Explicit dollar costs for FN vs FP | Harder to explain; not needed if F1 selects and other metrics are reported |

**Recommended stance for this project:** use **this plan** (small F1-maximizing `GridSearchCV`) as the hyperparameter step; optionally start with a **manual one-knob** smoke test; follow with **threshold CV** after params are frozen if a non-0.5 cutoff is desired.

---

## Context in this project

| Item | Detail |
|---|---|
| Notebook | `random-forest-model.ipynb` |
| Features | Pre-engineered CSVs in `telco-data/` (`X_train`, `X_test`, `y_train`, `y_test`) |
| Baseline models | `rf` (default) and `rf_balanced` (`class_weight="balanced"`) |
| Problem to fix | Train accuracy ~0.99 vs test ~0.77 (gap ~0.22+) — unconstrained trees memorizing train |
| Recommended starting estimator | Same as `rf_balanced`: `class_weight="balanced"`, `n_estimators=100`, `random_state=47`, `n_jobs=-1` |
| Related plan | Threshold CV is out of scope here; see `rf-threshold-cv-plan.md` |

Do **not** use `X_test` / `y_test` to choose hyperparameters. Test is for a single final evaluation only.

---

## Implementation plan

### 1. Add a new notebook section after the overfitting discussion

Suggested heading:

```text
## Hyperparameter tuning (GridSearchCV, maximize F1)
```

Place it after the current overfitting markdown (end of notebook), so baseline `rf` / `rf_balanced` cells at threshold 0.5 remain intact.

If both hyperparameter tuning and threshold tuning will eventually live in the notebook, preferred order is:

1. Hyperparameter search (this plan) → freeze growth params  
2. Threshold CV on the tuned model (`rf-threshold-cv-plan.md`) → freeze `best_t`  
3. One final test evaluation of model + threshold  

### 2. Imports

Add (or extend existing imports):

- `GridSearchCV`, `StratifiedKFold` from `sklearn.model_selection`
- `f1_score`, `precision_score`, `recall_score` from `sklearn.metrics` (if not already imported)
- `pandas` as `pd` (if not already available) for `cv_results_` tables

`RandomForestClassifier`, accuracy / ROC / confusion-matrix helpers are already in the notebook.

### 3. Define the search space (small structured grid)

Tune **tree-growth** params that constrain overfitting. Keep `n_estimators=100` fixed in the first pass (more trees mainly stabilize; they do not close the train≈1.0 gap by themselves).

| Hyperparameter | Candidate values | Role |
|---|---|---|
| `max_depth` | `[5, 8, 10, 15, None]` | Global tree complexity cap |
| `min_samples_leaf` | `[1, 5, 10, 20]` | Minimum leaf size (strong regularizer) |
| `min_samples_split` | `[2, 10, 20]` | Minimum samples to split a node |

**Combo count:** 5 × 4 × 3 = **60** configurations × **5** folds = **300** fits (+ 1 final refit). Acceptable for ~5.6k training rows with `n_jobs=-1`.

**Leaner alternative** (if runtime is an issue): drop `min_samples_split` from the grid (fix at `2`) → 20 combos → 100 fits. Or use `RandomizedSearchCV` with `n_iter=20` and the same distributions.

### 4. Configure multi-metric tracking with F1 as the only refit target

Use a **dict of scorers** so CV tracks several metrics, but set `refit="f1"` so only F1 chooses the winner.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "max_depth": [5, 8, 10, 15, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "min_samples_split": [2, 10, 20],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)

scoring = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
}

gs = GridSearchCV(
    estimator=RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=47,
        n_jobs=-1,
    ),
    param_grid=param_grid,
    scoring=scoring,
    refit="f1",              # ONLY this metric selects best_params_
    cv=cv,
    n_jobs=-1,
    return_train_score=True, # enables train vs CV gap inspection
)

gs.fit(X_train, y_train)
```

**Selection rule (explicit):**

- Winner = configuration with highest **mean CV F1** (`gs.best_score_` / `mean_test_f1`).
- Precision, recall, accuracy, and ROC-AUC are logged for every combo but **do not** change who wins.
- `gs.best_estimator_` is the model refit on **full** `X_train` / `y_train` with `best_params_`.

Name the tuned model clearly, e.g. `rf_tuned = gs.best_estimator_`.

### 5. Inspect CV results (tradeoff table)

Build a sorted table from `gs.cv_results_` so the notebook shows *why* F1 won and what was sacrificed.

Suggested columns:

| Column | Meaning |
|---|---|
| `param_max_depth`, `param_min_samples_leaf`, `param_min_samples_split` | Settings |
| `mean_test_f1`, `std_test_f1` | Primary selection metric (+ stability) |
| `mean_test_precision`, `mean_test_recall` | Churn tradeoff |
| `mean_test_accuracy`, `mean_test_roc_auc` | Secondary context |
| `mean_train_f1` | Optional; compare to `mean_test_f1` for CV overfitting signal |
| `rank_test_f1` | Rank by primary metric |

```python
results = pd.DataFrame(gs.cv_results_)
cols = [
    "param_max_depth",
    "param_min_samples_leaf",
    "param_min_samples_split",
    "mean_test_f1",
    "std_test_f1",
    "mean_test_precision",
    "mean_test_recall",
    "mean_test_accuracy",
    "mean_test_roc_auc",
    "mean_train_f1",
    "rank_test_f1",
]
top = results[cols].sort_values("rank_test_f1").head(10)
display(top)

print("Best params:", gs.best_params_)
print(f"Best CV F1: {gs.best_score_:.3f}")
```

Optional markdown notes after the table:

- Did the F1 winner also have strong recall (preferred for costly false negatives)?
- Did a nearby rank-2/3 config have much higher recall with only slightly lower F1? Mention as a business alternative, but do **not** silently switch selection rules mid-notebook.
- Is `mean_train_f1` much higher than `mean_test_f1` for unconstrained rows (`max_depth=None`, `min_samples_leaf=1`)? That confirms the original overfitting diagnosis.

### 6. Evaluate once on the held-out test set (same fashion as existing notebook)

Mirror the evaluation style used for `rf` and `rf_balanced`:

Suggested subsection:

```text
### Evaluating `rf_tuned` (best CV F1 params, threshold 0.5)
```

1. **Accuracy** — `accuracy_score(y_test, y_pred_tuned)`  
2. **Classification report** — `classification_report(..., target_names=["No churn", "Churn"])`  
3. **ROC-AUC** — `roc_auc_score(y_test, y_proba_tuned)`  
4. **Confusion matrix + ROC curve** — same 1×2 layout as baseline plots  

Short note: test metrics are the honest estimate of the tuned forest at the **default 0.5** cutoff. CV F1 was only for choosing hyperparameters.

### 7. Overfitting check (same fashion as existing section)

Replicate `### Overfitting check` for `rf_tuned`:

```python
train_acc_t = rf_tuned.score(X_train, y_train)
test_acc_t = rf_tuned.score(X_test, y_test)
print(f"Train accuracy (tuned): {train_acc_t:.3f}")
print(f"Test accuracy  (tuned): {test_acc_t:.3f}")
print(f"Gap:                    {train_acc_t - test_acc_t:.3f}")
```

Optionally also print train vs test **F1** at 0.5 (more aligned with the selection metric).

**Success criteria (qualitative):**

- Train accuracy no longer ~0.99  
- Gap meaningfully smaller than ~0.22  
- Test churn F1 competitive with or better than `rf_balanced`  
- If gap shrinks but churn F1 collapses, the grid may be too aggressive (too shallow / leaf too large) — widen candidates toward milder regularization

### 8. Compare vs existing models (same comparison fashion)

Extend the notebook’s churn-class comparison table:

```text
### Comparing `rf` vs `rf_balanced` vs `rf_tuned` (Churn class)
```

| Metric | `rf` (0.5) | `rf_balanced` (0.5) | `rf_tuned` (0.5) | Change vs `rf_balanced` |
|---|---|---|---|---|
| Precision | … | … | … | … |
| Recall | … | … | … | … |
| F1-score | … | … | … | … |
| False Negatives | … | … | … | … |
| False Positives | … | … | … | … |
| Accuracy | … | … | … | … |
| ROC-AUC | … | … | … | … |
| Train–test accuracy gap | ~0.225 | ~0.228 | … | … |

Interpretation markdown should state:

- Selection was by **CV F1 only**; other metrics are for tradeoff reading  
- Whether regularization closed the overfitting gap  
- Whether churn recall / FN improved or worsened vs `rf_balanced`  
- Restate business preference: missing a churner is costlier than a false alarm  

### 9. Document the decision rule in the notebook

Add a brief markdown cell stating:

- Hyperparameters were chosen by 5-fold stratified `GridSearchCV` maximizing **mean CV F1**.  
- Precision, recall, accuracy, and ROC-AUC were tracked during CV for tradeoff inspection only.  
- `best_params_` were frozen before evaluating on `X_test`.  
- Default threshold **0.5** remains for this section; threshold tuning (if done) is a separate follow-up on `rf_tuned`.

### 10. Out of scope for this plan (follow-ups)

- Decision-threshold CV maximizing F1 (`rf-threshold-cv-plan.md`) — run **after** freezing `rf_tuned` params.  
- Including `n_estimators` in the grid (optional second pass once growth params are stable).  
- Custom composite / cost-weighted scoring functions.  
- Nested CV (outer test estimation + inner param search) — higher rigor, more complexity.  
- Persisting the tuned model to disk.

---

## Acceptance checklist

- [ ] New section added to `random-forest-model.ipynb` without breaking baseline `rf` / `rf_balanced` cells  
- [ ] Search uses only `X_train` / `y_train` via stratified `GridSearchCV`  
- [ ] `scoring` includes F1, precision, recall, accuracy, ROC-AUC  
- [ ] `refit="f1"` so **only F1** selects `best_params_`  
- [ ] Top CV results table shown for tradeoff inspection  
- [ ] `rf_tuned` evaluated once on test (accuracy → report → ROC-AUC → CM/ROC plot)  
- [ ] Overfitting check reports train/test gap for `rf_tuned`  
- [ ] Comparison table includes `rf` / `rf_balanced` / `rf_tuned` with secondary metrics  
- [ ] Notebook documents F1-only selection + secondary metrics for tradeoffs  
- [ ] Test set was not used to choose hyperparameters  

---

## Suggested implementation order

1. Section heading + imports  
2. `param_grid`, `scoring`, `GridSearchCV` setup + `fit`  
3. Print `best_params_` / `best_score_` + top-10 CV tradeoff table  
4. Standalone test eval for `rf_tuned`  
5. Overfitting check  
6. Three-way comparison table + interpretation markdown  
7. Decision-rule documentation cell  

---

## Relationship to threshold tuning

| Plan | What it chooses | Primary metric |
|---|---|---|
| **This plan** | Tree complexity (`max_depth`, leaf, split) | Mean CV **F1** |
| `rf-threshold-cv-plan.md` | Probability cutoff `best_t` | Mean CV **F1** |

Same primary metric, different lever. Recommended sequence: **hyperparameters first, threshold second**, each frozen before the final test look.
