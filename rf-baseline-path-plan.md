# Plan: Explore the Baseline (Unweighted) Path — Tune + Threshold

Explore an **alternate pipeline** in `random-forest-model.ipynb`: apply the same two levers already used on the class-weighted forest — **growth hyperparameter CV** then **threshold CV** — but start from the **baseline** `RandomForestClassifier` with **no** `class_weight="balanced"`.

This is an exploration / comparison section. It does **not** replace the existing class-weighted path (`rf_balanced` → `rf_tuned` → `rf_thresholded`). Keep both paths in the notebook so tradeoffs are visible side by side.

**Related plans (already implemented on the weighted path):**

- `rf-hyperparameter-cv-plan.md` — GridSearchCV on growth params, `refit="f1"`
- `rf-threshold-cv-plan.md` — stratified CV threshold search (F1 + precision-constrained)

---

## Why explore this path

| Path | Training | Typical behavior @ 0.5 | Typical threshold move |
|---|---|---|---|
| **Class-weighted (current)** | `class_weight="balanced"` | Aggressive: high recall, many FPs | **Raise** cutoff to cut false alarms |
| **Baseline (this plan)** | Default weights | Conservative: lower recall, fewer FPs | **Lower** cutoff to catch more churners |

Both paths use the same knobs (regularize trees, then set an operating point). They approach the FP/FN tradeoff from **opposite sides of 0.5**. Comparing them answers: *Does imbalance handling belong in training (`class_weight`), in the decision rule (threshold), or both?*

**Project facts this plan builds on:**

| Item | Detail |
|---|---|
| Notebook | `random-forest-model.ipynb` |
| Features | `telco-data/` train/test CSVs |
| Baseline `rf` @ 0.5 (approx.) | Precision ~0.58, recall ~0.44, FN high, FP relatively low, train/test gap ~0.23 |
| Weighted tuned path (approx.) | Better recall/ROC, gap ~0.07; FPs high @ 0.5; threshold raised (~0.54–0.58) to cut FPs |
| Churn rate | ~26.5%; missing a churner is costlier than a false alarm |
| Selection metrics (match existing) | Hyperparams: max **mean CV F1**; thresholds: **F1** and **max precision s.t. recall ≥ floor** |

Do **not** use `X_test` / `y_test` to choose hyperparameters or thresholds. Test is for a single final evaluation per frozen system.

---

## Strengths and weaknesses of the baseline path

### Strengths

- **Standard “fit then set policy” recipe.** Train without reweighting, then choose the cutoff from costs / F1 / recall floor — a common convention when you want ranking + an explicit operating point.
- **Often cleaner probabilities @ default training.** Unweighted RF is less biased toward inflating churn scores; calibration can be easier to reason about than with `class_weight="balanced"`.
- **Starts FP-conservative.** Baseline @ 0.5 already has fewer false alarms than weighted models; threshold search mainly **buys recall** rather than repairing an overly aggressive 0.5.
- **Isolates two effects.** Regularization closes the ~0.22 gap; threshold (not class weight) carries most of the imbalance correction — clearer ablation vs the weighted path.
- **Fair comparison design.** Same grid, same CV folds/`random_state`, same threshold grid and recall floor → differences are attributable to `class_weight`, not to unequal search effort.
- **May match weighted operating points after thresholding.** With a low enough cutoff, precision/recall can land near the weighted+raised-threshold system — useful if ROC quality is similar.

### Weaknesses

- **Hyperparams alone won’t fix imbalance.** Constraining `max_depth` / leaf / split reduces overfitting but leaves a majority-biased @0.5 behavior (low recall) unless the threshold moves.
- **May need a much lower threshold.** To hit recall ≥ 0.65–0.70, `best_t` on the baseline path may sit well below 0.5; very low cutoffs can explode FPs.
- **Ranking might be weaker for churn.** If unweighted trees under-emphasize minority splits, ROC-AUC / PR quality can lag the balanced forest — thresholding cannot invent separation that isn’t there.
- **Extra notebook surface area.** Another GridSearch (~300 fits) + threshold CV duplicates runtime and cells; easy to confuse `rf_tuned` (weighted) with `rf_tuned_base` (unweighted).
- **Two “best” systems to explain.** Stakeholders must understand weighted-raise-threshold vs unweighted-lower-threshold; pick one production default deliberately.
- **Same leakage risks as before.** Params and thresholds must be chosen on train CV only; don’t peek at test to decide which path “wins.”

---

## Alternate ways to implement this (short summary)

These are reasonable variants of “baseline path” exploration. **This plan recommends** sequential train-only CV: unweighted `GridSearchCV` (F1) → threshold CV on frozen unweighted params (F1 + precision@recall-floor) → one test eval → compare to the existing weighted path.

| Approach | What it does | Strengths | Weaknesses |
|---|---|---|---|
| **This plan: unweighted GridSearch + threshold CV** | Same structure as weighted path; `class_weight=None` | Apples-to-apples vs current notebook; reproducible; both levers explicit | Runtime (~300 RF fits + threshold folds); more cells |
| **Threshold-only on current `rf`** | Keep unconstrained baseline; only search `best_t` on train CV | Fast; shows how far threshold alone can push recall | Does **not** fix ~0.22 overfitting gap; unfair vs `rf_tuned` |
| **Hyperparams-only on unweighted (stay @ 0.5)** | GridSearch without threshold move | Isolates regularization effect on baseline | Leaves recall weak; incomplete for churn costs |
| **Single val split (params and/or threshold)** | One stratified 80/20 on train instead of 5-fold | Quicker debugging | Noisier winner; less stable than CV |
| **`RandomizedSearchCV` / leaner grid** | Fewer fits on the same unweighted space | Faster exploration | May miss a strong cell |
| **`class_weight` as a searched hyperparameter** | Put `class_weight: [None, "balanced"]` inside one GridSearch | Lets CV pick weighting jointly with depth/leaf | Confounds “baseline path vs weighted path”; harder to narrate; doubles grid |
| **Resampling instead of weights** (e.g. balanced bagging / SMOTE on train folds only) | Change class frequencies rather than weights | Another common imbalance family | Different bias/variance story; more pipeline complexity; out of scope for a thin ablation |
| **Calibrate then threshold** (`CalibratedClassifierCV` on unweighted RF) | Improve probability meaning before cutoff search | Helps if raw RF scores are poorly scaled | Extra step; still need a threshold; more code |
| **Nested CV (outer test estimate)** | Inner: params + threshold; outer: performance | Most rigorous generalization claim | Slow and heavy for this notebook stage |
| **Cost-sensitive / recall-floor-only selection for params** | Choose growth params by recall or custom cost, not F1 | Aligns training selection with business FN cost | Diverges from existing F1-first convention; harder to compare paths |

**Recommended stance:** implement **this plan** as a clearly labeled alternate section; keep the weighted path as the current primary system unless test metrics + business preference clearly favor the baseline path.

---

## Expected behavior (hypothesis to validate)

Use these as checkboxes while reading results — not as guarantees.

1. **Unweighted + tuned @ 0.5:** train/test gap shrinks vs raw `rf`; recall stays relatively low; FPs stay relatively moderate.
2. **Threshold CV:** `best_t` (F1) and especially recall-oriented cutoffs land **below 0.5** (opposite direction from weighted `rf_tuned`).
3. **Precision-constrained pick** (`RECALL_FLOOR`, e.g. 0.65): among feasible low thresholds, max precision still trades FP↔FN; compare FP count to weighted `best_t_prec`.
4. **Head-to-head:** compare ROC-AUC (ranking) and the frozen operating points (F1 and prec@floor) between:
   - weighted: `rf_tuned` + thresholds  
   - unweighted: `rf_tuned_base` + thresholds  

If unweighted ROC-AUC is clearly worse, thresholding may not catch up. If ROC-AUC is similar, operating points may be close after cutoff search.

---

## Implementation plan

### 1. Add a new notebook section (after the weighted threshold section)

Suggested heading:

```text
## Alternate path: unweighted baseline — tune then threshold
```

State explicitly in markdown:

- Same grids / CV / metrics as the weighted path.
- Only intentional training change: **no** `class_weight="balanced"`.
- Existing weighted cells stay intact; this is an ablation / alternate system.

Naming suggestions (avoid clobbering weighted variables):

| Object | Name |
|---|---|
| Grid search | `gs_base` |
| Tuned forest | `rf_tuned_base` |
| F1 threshold | `best_t_base` |
| Precision@floor threshold | `best_t_prec_base` |
| Test preds @ F1 t | `y_test_hat_base` |
| Test preds @ prec t | `y_test_hat_prec_base` |

### 2. Hyperparameter search (mirror weighted GridSearchCV)

Use the **same** `param_grid`, `StratifiedKFold(..., random_state=47)`, and multi-metric `scoring` with `refit="f1"`.

```python
gs_base = GridSearchCV(
    estimator=RandomForestClassifier(
        n_estimators=100,
        # class_weight intentionally omitted (None)
        random_state=47,
        n_jobs=-1,
    ),
    param_grid=param_grid,  # reuse the same dict as weighted search
    scoring=scoring,        # f1, precision, recall, accuracy, roc_auc
    refit="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=47),
    n_jobs=-1,
    return_train_score=True,
)
gs_base.fit(X_train, y_train)
rf_tuned_base = gs_base.best_estimator_
```

Then mirror existing cells:

1. Top-10 CV tradeoff table from `gs_base.cv_results_`.
2. Test eval @ **0.5**: accuracy → classification report → ROC-AUC → CM + ROC plot.
3. Overfitting check (train/test accuracy + F1 gap).
4. Short note: compare gap to raw `rf` (~0.23) and to weighted `rf_tuned` (~0.07).

### 3. Threshold CV on frozen `rf_tuned_base` params

Reuse the weighted threshold procedure:

| Setting | Value |
|---|---|
| Fold models | `RandomForestClassifier(**gs_base.best_params_, n_estimators=100, random_state=47, n_jobs=-1)` — **no** class_weight |
| Folds | `StratifiedKFold(n_splits=5, shuffle=True, random_state=47)` |
| Grid | `np.round(np.arange(0.1, 0.9, 0.01), 2)` |
| Pick A | `best_t_base` = argmax mean CV F1 |
| Pick B | `best_t_prec_base` = max mean CV precision among mean CV recall ≥ `RECALL_FLOOR` (use the **same floor** as weighted, currently `0.65`) |
| Final forest | Reuse `rf_tuned_base` (already fit on full train) |
| Test | Apply frozen cutoffs once; do not re-pick on test |

Include:

- CV table + plot with both vertical lines (and recall floor).
- Standalone test eval for `@ best_t_base` and `@ best_t_prec_base`.
- Overfitting check using the frozen cutoffs (not `model.score()`).

**Interpretation cue:** if both baseline thresholds are `< 0.5` while weighted thresholds were `> 0.5`, call that out — same levers, opposite direction.

### 4. Head-to-head comparison table

Build one churn-class table on the **same** `X_test` / `y_test`, e.g.:

| Column | System |
|---|---|
| `rf (0.5)` | Original baseline |
| `rf_tuned_base (0.5)` | Unweighted + growth tune |
| `base F1 (best_t_base)` | Unweighted + F1 threshold |
| `base prec (best_t_prec_base)` | Unweighted + prec@floor |
| `rf_tuned (0.5)` | Weighted + growth tune (existing) |
| `weighted F1 / prec` | Existing thresholded cutoffs |

Metrics: precision, recall, F1, FN, FP, accuracy, ROC-AUC.

Optional: side-by-side confusion matrices for the two production candidates (e.g. weighted prec vs baseline prec).

### 5. Interpretation markdown (required)

After results, answer explicitly:

- Did unweighted tuning close the overfitting gap?
- Did threshold search move **below** 0.5 as expected?
- At matched recall floors, which path has fewer FPs? Better F1? Better ROC-AUC?
- Is the preferred production system still weighted, or does baseline+threshold win on the business tradeoff?
- Restate: missing churners is costlier than false alarms; recall floor encodes how much catch-rate you will not give up.

### 6. Decision-rule documentation cell

Document:

- Unweighted path params chosen by 5-fold CV **mean F1**; thresholds by mean F1 and by max precision with recall ≥ floor.
- `class_weight` was **not** used on this path.
- All selection on train only; test evaluated once per frozen system.
- Weighted path remains available for comparison; production choice is a documented decision, not an implicit overwrite.

### 7. Out of scope

- Removing or rewriting the existing weighted sections.
- Nested CV.
- Putting `class_weight` inside the same grid as this ablation (that’s a different experiment).
- SMOTE / resampling pipelines.
- Persisting models to disk.

---

## Suggested implementation order

1. Section heading + naming convention markdown  
2. `gs_base` GridSearchCV + `rf_tuned_base`  
3. CV table, test @ 0.5, overfitting check  
4. Threshold CV → `best_t_base`, `best_t_prec_base` + plot  
5. Test evals at both baseline thresholds  
6. Head-to-head comparison vs weighted path + CMs  
7. Interpretation + decision-rule cells  

---

## Acceptance checklist

- [ ] New section added without breaking weighted `rf` / `rf_balanced` / `rf_tuned` / threshold cells  
- [ ] Unweighted `GridSearchCV` uses the same grid/CV/F1-refit pattern; **no** `class_weight`  
- [ ] `rf_tuned_base` evaluated @ 0.5 and overfitting gap reported  
- [ ] Threshold search on train only with F1 pick + precision@`RECALL_FLOOR` pick (same floor as weighted)  
- [ ] Test used once per frozen baseline cutoff  
- [ ] Comparison table includes both paths’ key operating points (at least @0.5 tuned + both thresholds each, or a clear reduced subset)  
- [ ] Markdown states expected opposite threshold direction and records which path is preferred and why  
- [ ] Variable names do not overwrite weighted-path objects (`gs`, `rf_tuned`, `best_t`, …)  

---

## Relationship to existing plans

| Plan | Path | What it chooses |
|---|---|---|
| `rf-hyperparameter-cv-plan.md` | Weighted | Growth params via CV F1 |
| `rf-threshold-cv-plan.md` | Weighted | Cutoffs via CV F1 + prec@floor |
| **This plan** | **Unweighted baseline** | Same two stages, then **compare** to weighted |

**Recommended sequence in the notebook:** keep weighted path as-is → add this alternate section → decide production default from the head-to-head table (not from test-driven re-tuning).
