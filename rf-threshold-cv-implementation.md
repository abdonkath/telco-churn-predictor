# Implementation notes: Threshold Tuning via Cross-Validation

This document records what was implemented in `random-forest-model.ipynb` from `rf-threshold-cv-plan.md` (and related follow-ups) during the chat session that added **decision-threshold CV** on the frozen weighted forest. It is meant for reviewing **decisions**, **deviations**, and **how the new work sits in the notebook**.

Related docs:

| File | Role |
|---|---|
| `rf-threshold-cv-plan.md` | Original spec / acceptance checklist (Option B) |
| `rf-hyperparameter-cv-plan.md` | Prior weighted growth-param CV |
| `rf-hyperparameter-cv-implementation.md` | Notes for the weighted GridSearch section |
| `rf-baseline-path-plan.md` | Later alternate unweighted path (planned in a related chat turn) |
| `rf-baseline-path-implementation.md` | Notes for the unweighted ablation (separate implementation chat) |
| `random-forest-model.ipynb` | Target notebook |

---

## Goal of this implementation

Choose operating thresholds with **5-fold stratified CV on train only**, freeze them, then evaluate once on held-out test — applied to the already-tuned **`rf_tuned`** forest (growth params from GridSearchCV). Primary motivation after hyperparameter tuning: **`rf_tuned` @ 0.5 had strong recall but a large false-positive count**; thresholding is the intended FP lever without undoing regularization.

---

## Steps actually taken

1. **Reviewed results and confirmed the next lever**  
   After `rf_tuned` was available, comparison showed overfitting largely fixed (gap ~0.07) but FPs high @ 0.5 (~279 in an earlier run; later full re-runs differ slightly with different `best_params_`). Agreed the next step was threshold CV on the **frozen tuned** model, not re-opening the growth grid.

2. **Read `rf-threshold-cv-plan.md` and the live notebook end**  
   Mapped plan steps onto cells after the weighted hyperparameter decision-rule section. Confirmed imports (`StratifiedKFold`, F1/precision/recall) were already present from the GridSearch work.

3. **Clarified base model vs plan default**  
   Plan text recommended threshold search on `rf_balanced`. Implementation used **`rf_tuned`** (and fold models built from `gs.best_params_`) because hyperparameter tuning had already completed and was the correct forest to set policy on.

4. **Appended `## Threshold tuning (cross-validation)`**  
   Left all baseline and hyperparameter cells intact. Named the system `rf_thresholded` (aliases the frozen `rf_tuned` forest; only the cutoff changes).

5. **Implemented Option B threshold search on train**  
   - Threshold grid: `np.round(np.arange(0.1, 0.9, 0.01), 2)` (rounding added to avoid `0.539999…` float noise).  
   - Per fold: fit RF with frozen tuned growth params + `class_weight="balanced"`; score validation probabilities across the grid.  
   - Pick **`best_t`**: maximize mean CV F1 (plan primary rule).

6. **Added CV tradeoff table + plot**  
   Mean CV F1 / precision / recall vs threshold, with a vertical line at `best_t` (later extended for the second pick).

7. **Froze cutoff(s), applied once on test**  
   Reused `rf_tuned` as `rf_thresholded` (already fit on full train). Computed `y_test_hat = (proba >= best_t)`. Mirrored existing eval style: accuracy → classification report → ROC-AUC → CM + ROC plot.

8. **Extended comparison beyond the plan’s three-way table**  
   Included `rf` / `rf_balanced` / `rf_tuned` @ 0.5 and thresholded operating points, with **Change vs `rf_tuned`** so the cutoff effect is isolated from growth-param changes. Side-by-side confusion matrices for visual FN/FP comparison.

9. **Overfitting check at frozen threshold(s)**  
   Used predictions at `best_t` (not `model.score()`, which always uses 0.5), plus train/test F1.

10. **Documented decision rules in markdown**  
    Train-only CV selection; test not used to pick cutoffs; 0.5 remains the reference for earlier sections.

11. **Follow-up in the same chat: precision-constrained pick**  
    User asked how to address large FPs; F1-max alone only nudged the cutoff slightly. Added **`best_t_prec`**: maximize mean CV precision among thresholds with mean CV recall ≥ `RECALL_FLOOR`.

12. **Set `RECALL_FLOOR = 0.65` (user choice)**  
    At the initial default `0.70`, `best_t` and `best_t_prec` often coincided (F1 winner already on the floor frontier). User chose `0.65` so the precision pick could diverge and cut more FPs while keeping a stated catch-rate floor.

13. **Verified threshold cells outside the notebook kernel**  
    Dry-ran the new cells with rebuilt state (mapped Yes/No labels; used frozen `best_params_` when skipping a full GridSearch). Confirmed both picks and comparison logic executed cleanly.

---

## Decisions made (and why)

| Decision | Choice | Why |
|---|---|---|
| Base forest for threshold CV | **`rf_tuned`** (not `rf_balanced`) | Hyperparams were already frozen; plan’s “recommended base” was written before that. Tuning threshold on unconstrained balanced trees would ignore the overfitting fix. User direction: keep the “next best step” (threshold on tuned). |
| CV style | **Option B** (5-fold stratified, mean F1) | Matches the written plan’s preferred approach for a trustworthy operating point. |
| Primary pick | **`best_t` = max mean CV F1** | Aligns with hyperparameter selection metric and the plan’s Approach 2. |
| Second pick | **`best_t_prec` = max precision s.t. recall ≥ floor** | Direct response to high FPs; encodes “don’t give up too much recall” via `RECALL_FLOOR`. |
| Default `RECALL_FLOOR` | **`0.65`** (after trying `0.70`) | At 0.70 the two picks often tied; 0.65 makes the FP-reduction pick distinct. User explicitly selected 0.65. |
| Final forest object | **`rf_thresholded = rf_tuned`** (no redundant refit) | Same frozen params already fit on full train; only the decision rule changes. Avoids an extra identical `fit`. |
| Fold models | Refit each fold with `**gs.best_params_` | Prevents leakage from using full-train `rf_tuned` probabilities as if they were out-of-fold. |
| Threshold grid rounding | `np.round(..., 2)` | Clean `best_t` display and stable dict keys; same intent as plan’s `np.arange(0.1, 0.9, 0.01)`. |
| Comparison delta column | **Change vs `rf_tuned`** (not vs `rf_balanced`) | Isolates threshold effect on the regularized forest — the scientifically relevant contrast after hyperparameter tuning. |
| Four-/five-way tables | Include tuned @ 0.5 **and** both cutoffs | Plan’s three-way (`rf` / `rf_balanced` / thresholded) understated the story once `rf_tuned` existed. |
| Placement | **After** hyperparameter decision-rule cell | Matches both plans’ preferred sequence: freeze growth params → then threshold → then test. |

---

## Where this implementation followed the plan

- New section without breaking baseline or hyperparameter cells.
- Stratified 5-fold CV on `X_train` / `y_train` only; test not used to choose `best_t`.
- Fine threshold grid over ~0.1–0.9 step 0.01; maximize mean CV F1 for the primary pick.
- Optional CV metrics table + plot included.
- Final system evaluated once on test at frozen cutoff(s).
- Eval order mirrors existing notebook (accuracy → report → ROC-AUC → CM/ROC).
- Comparison table with precision, recall, F1, FN, FP, accuracy, ROC-AUC.
- Side-by-side confusion matrices.
- Interpretation markdown + decision-rule documentation.
- Overfitting check using the **same frozen threshold** for train and test predictions.

---

## Deviations from the plan (and why)

| Plan text / suggestion | What we did | Why it differed |
|---|---|---|
| Recommended base model: **`rf_balanced`** | Threshold CV on **`rf_tuned`** / `gs.best_params_` | Hyperparameter plan and notebook sequence say: freeze growth params first, then threshold. Using `rf_balanced` would threshold an overfit forest and ignore `rf_tuned`. |
| Comparison: **`rf` vs `rf_balanced` vs `rf_thresholded`** | Also includes **`rf_tuned` @ 0.5** and dual threshold columns; delta vs **`rf_tuned`** | Once tuned exists, the meaningful “before/after cutoff” contrast is vs `rf_tuned` @ 0.5, not only vs `rf_balanced`. |
| Single operating point (`best_t` by F1) | Added **`best_t_prec`** with **`RECALL_FLOOR`** | Plan listed precision-floor / business-constrained objectives as **out of scope**; user later asked specifically to address large FPs. F1-max alone was only a small FP improvement in early dry-runs. |
| Initial floor tried as **0.70** in discussion | User set floor to **0.65** | At 0.70, F1 and prec picks coincided on the frontier; 0.65 was requested so the precision-constrained path is more aggressive on FPs. |
| Plan sketch: refit a new `rf_thresholded` on full train | **`rf_thresholded = rf_tuned`** | Identical hyperparameters and training set; alias avoids a redundant fit and keeps one source of truth for the forest. |
| Comparison helper in plan is a small `churn_metrics(...)` without gap | Used `churn_metrics_at_threshold` (no `model.score` gap in the threshold table); gap handled in a dedicated overfitting cell | Non-0.5 cutoffs make `model.score()` misleading; separating concerns matches the plan’s own overfitting section guidance. |
| CM row: three matrices (or balanced vs thresholded) | Expanded to include tuned + both thresholded cutoffs (later notebook runs show five panels) | Need visual FN/FP for every operating point discussed in the table. |

**Non-deviations worth stating:** nested CV, persisting model+threshold to disk, and searching thresholds on `X_test` were still not done.

---

## How this fits into `random-forest-model.ipynb` structure

High-level flow after threshold work (and later sections added in other chats):

```text
[Setup]
  Imports, load CSVs, label map Yes/No → 0/1

[Baseline — unchanged]
  `rf` @ 0.5
  `rf_balanced` @ 0.5
  Compare + overfitting (~0.22+ gap)

[Weighted hyperparameter path]
  GridSearchCV → `rf_tuned`
  Test @ 0.5, overfitting, 3-way compare, decision rule

[This work: weighted threshold path]
  ## Threshold tuning (cross-validation)
  CV grid → `best_t` (F1), `best_t_prec` (prec @ RECALL_FLOOR)
  Plot + freeze cutoffs on `rf_thresholded` (= `rf_tuned`)
  Test eval @ each cutoff
  Operating-point comparison (vs `rf_tuned`)
  Overfitting @ frozen cutoffs
  Decision rule (documented) — threshold

[Later: unweighted baseline path — separate implementation]
  `gs_base` / `rf_tuned_base` / `best_t_base` / `best_t_prec_base`
  Head-to-head vs weighted path

[Later: model summary]
  Which system for which situation
```

### Cell map for the threshold section (roles)

| Role | Content |
|---|---|
| Section intro | Train-only CV; F1 + precision@floor; applied to `rf_tuned` |
| CV search code | Fold fits with tuned params; `best_t`, `best_t_prec`, `RECALL_FLOOR` |
| CV table / plot | Curves + vertical lines for both picks + recall floor |
| Freeze + apply | `rf_thresholded`, `y_test_hat`, `y_test_hat_prec` |
| Eval @ F1 t | Report + CM/ROC |
| Eval @ prec t | Report + CM/ROC |
| Comparison | Multi-column churn metrics + CMs |
| Closing markdown | Interpretation + formal threshold decision rules |

### Structural relationships

- **Depends on:** mapped labels; `gs.best_params_` / `rf_tuned`; baseline preds for comparison columns; shared metric helpers pattern from the hyperparameter section.
- **Does not replace:** earlier @0.5 evaluations — those remain the reference operating point.
- **Enables:** later unweighted-path ablation to reuse the same `RECALL_FLOOR` and comparison vocabulary.
- **Kernel tip:** with `rf_tuned` / `gs` in memory, re-run from the threshold heading downward after changing `RECALL_FLOOR`; no need to re-run GridSearch unless the kernel restarted.

---

## Results context (for decision analysis)

Exact numbers move when GridSearch winners change across full notebook re-executions. Conceptually, this implementation established:

1. **Thresholding is the FP lever** after regularization — not a substitute for `max_depth` / leaf constraints.
2. **Direction of the cutoff depends on training bias:** weighted/`balanced` forests are often aggressive @ 0.5 → raise threshold; unweighted forests (later ablation) are conservative @ 0.5 → lower threshold. Same API, opposite side of 0.5.
3. **F1-max alone may barely move the cutoff** when CV F1 peaks near 0.5; a **recall floor + max precision** pick is the explicit FP-reduction policy.
4. **Production preference** (documented later in the notebook summary) favored **weighted forest + precision-constrained threshold** when outreach cost matters but recall must stay near the floor — consistent with why `best_t_prec` was added here.

Illustrative live outputs from one full notebook run (weighted path; confirm in your current outputs):

| Operating point | Role |
|---|---|
| `rf_tuned` @ 0.5 | Max recall / fewest FNs; many FPs |
| thresh F1 (`best_t`) | May equal 0.5 if F1 peaks there |
| thresh prec (`best_t_prec`, floor 0.65) | Fewer FPs, recall held near the floor |

---

## Acceptance checklist status (threshold plan)

| Checklist item | Status |
|---|---|
| New section without breaking earlier cells | Done |
| Threshold search on train via stratified CV | Done |
| `best_t` by mean CV F1 on fine grid | Done |
| Final forest + frozen cutoff evaluated once on test | Done |
| Standalone eval mirrors existing style | Done |
| Comparison table with P/R/F1/FN/FP (+ acc/AUC) | Done (extended beyond three-way) |
| Side-by-side confusion matrices | Done |
| Interpretation markdown | Done |
| Overfitting check at frozen threshold | Done |
| Notes that test was not used to choose threshold | Done |
| Plan’s out-of-scope “precision floor” objective | **Intentionally added** as `best_t_prec` (user request) |

---

## Practical notes for re-running / review

- After changing **`RECALL_FLOOR`**, re-run from the threshold CV cell through the comparison/overfitting cells (not the whole notebook if `gs` / `rf_tuned` are warm).
- If F1 and prec picks coincide, the floor is binding at the F1 peak — lower the floor to allow a more FP-conservative cutoff, or accept that F1 already sat on the frontier.
- Do not compare weighted vs unweighted paths until both sections have been run in the **same** kernel session (or via a full top-to-bottom execute) so `best_params_` and thresholds are consistent.
- For a narrative of the **unweighted** ablation and the end-of-notebook chooser, see `rf-baseline-path-implementation.md`.
