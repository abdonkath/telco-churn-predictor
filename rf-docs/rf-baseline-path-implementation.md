# Implementation notes: Unweighted Baseline Path (Tune + Threshold)

This document records what was implemented in `random-forest-model.ipynb` from `rf-baseline-path-plan.md` during the chat session that added the alternate **unweighted** GridSearch + threshold CV path, plus the later model-summary section. It is meant for reviewing **decisions**, **deviations**, and **how the new work sits in the notebook**.

Related docs:

| File | Role |
|---|---|
| `rf-baseline-path-plan.md` | Spec / acceptance checklist for this work |
| `rf-hyperparameter-cv-plan.md` | Weighted growth-param CV (already implemented) |
| `rf-threshold-cv-plan.md` | Weighted threshold CV (already implemented) |
| `rf-hyperparameter-cv-implementation.md` | Prior chat notes for the weighted GridSearch section |
| `random-forest-model.ipynb` | Target notebook |

---

## Goal of this implementation

Explore an **alternate pipeline** that applies the same two levers already used on the class-weighted forest — **growth hyperparameter CV** then **threshold CV** — but starts from a `RandomForestClassifier` with **no** `class_weight="balanced"`. Keep the weighted path intact and compare both on the same held-out test set so tradeoffs are visible side by side.

Secondary follow-up in the same chat: append a **model summary** at the end of the notebook (chooser + per-system strengths/weaknesses) so a reader can pick a production operating point by use case.

---

## Steps actually taken

1. **Read the plan and mapped the existing notebook**  
   Listed cells, extracted the weighted `GridSearchCV` and threshold-CV patterns (`param_grid`, `scoring`, `RECALL_FLOOR`, `tuned_kwargs`, comparison helpers) so the unweighted path could mirror them with `_base` names.

2. **Surfaced plan-vs-notebook inconsistencies and clarifying questions**  
   Documented mismatches (illustrative weighted thresholds in the plan vs live CV values; markdown vs code comparison style elsewhere; optional CMs). Proceeded with plan-aligned defaults where the user had not yet answered production / `RECALL_FLOOR` / README questions.

3. **Appended a new section after the weighted threshold decision-rule cell**  
   Heading: `## Alternate path: unweighted baseline — tune then threshold`. Left all weighted-path objects (`gs`, `rf_tuned`, `best_t`, `best_t_prec`, etc.) untouched.

4. **Unweighted `GridSearchCV` → `rf_tuned_base`**  
   - Reused existing `param_grid` and `scoring` from the weighted section.  
   - Estimator: `n_estimators=100`, `random_state=47`, `n_jobs=-1`, **no** `class_weight`.  
   - Same stratified 5-fold CV (`random_state=47`), `refit="f1"`, `return_train_score=True`.  
   - Named search `gs_base`, winner `rf_tuned_base`.

5. **Mirrored weighted post-search cells for the unweighted forest**  
   Top-10 CV tradeoff table → test eval @ 0.5 → CM + ROC plot → overfitting check (accuracy + F1 gaps), with a note to compare gaps to raw `rf` and weighted `rf_tuned`.

6. **Threshold CV on frozen `rf_tuned_base` params**  
   - Same threshold grid `np.round(np.arange(0.1, 0.9, 0.01), 2)`.  
   - Same `RECALL_FLOOR` variable already defined on the weighted path (0.65).  
   - Fold models via `tuned_kwargs_base` (growth params + fixed knobs; **no** class weight).  
   - Picks: `best_t_base` (max mean CV F1), `best_t_prec_base` (max mean CV precision s.t. recall ≥ floor; ties → higher threshold).  
   - Separate result objects (`cv_threshold_results_base`, etc.) so re-running the baseline section does not clobber weighted threshold tables.

7. **Frozen cutoffs applied once on test**  
   `y_test_hat_base` / `y_test_hat_prec_base`, standalone reports + CM/ROC plots, then overfitting checks using the **frozen cutoffs** (not `model.score()`).

8. **Head-to-head comparison vs weighted path**  
   Code table with columns aligned to the plan (`rf`, `rf_tuned_base` @ 0.5, both baseline thresholds, `rf_tuned` @ 0.5, both weighted thresholds) plus a Δ column for base prec − weighted prec. Optional side-by-side CMs for the two precision@floor candidates only.

9. **Executed the full notebook end-to-end**  
   Used `jupyter nbconvert --execute --inplace` (after a sandbox failure, re-ran with unrestricted permissions and `MPLBACKEND=Agg`). Filled the interpretation markdown from real outputs.

10. **Documented production preference from the head-to-head**  
    Kept weighted path as preferred production system; baseline path retained as ablation.

11. **Later in the same chat: appended a model-summary section**  
    User asked for the chooser (best fit now / cheap outreach / ban class weights / expensive outreach / keep for experiments) plus justifications and per-model strengths/weaknesses at the **end** of the notebook.

---

## Decisions made (and why)

| Decision | Choice | Why |
|---|---|---|
| Placement | **After** weighted threshold section | Plan: ablation / alternate system; do not rewrite or replace weighted cells. |
| Naming | `gs_base`, `rf_tuned_base`, `best_t_base`, `best_t_prec_base`, `y_test_hat_*_base` | Plan naming; avoids overwriting weighted-path objects needed for head-to-head. |
| Grid / CV / scoring | **Reuse** weighted `param_grid`, same StratifiedKFold settings, same multi-metric `scoring`, `refit="f1"` | Fair ablation: differences attributable to `class_weight`, not unequal search effort. |
| `RECALL_FLOOR` | **Reuse** existing variable (0.65) | Plan: same floor as weighted for matched comparison. |
| Fold-model construction | `tuned_kwargs_base` dict pattern (like weighted `tuned_kwargs`) | Structurally parallel to the existing threshold cell; equivalent to the plan’s inline `RandomForestClassifier(**gs_base.best_params_, …)`. |
| Threshold result variable names | `*_base` suffixes throughout | Preserve weighted `cv_threshold_results` / fold dicts if sections are re-run independently. |
| Comparison columns | Plan’s list (omit `rf_balanced` from H2H) | `rf_balanced` already covered upstream; H2H focuses on tuned + thresholded systems. |
| Optional CMs | **Include** 2-panel weighted prec vs base prec only | Plan marked optional; enough to compare production candidates without a 7-panel grid. |
| Production preference | **Weighted + `best_t_prec`** still preferred | At matched recall floor, weighted had fewer FPs and slightly better precision/F1; ROC-AUC essentially tied. Baseline kept as documented alternative. |
| Execute in-chat | **Yes** (full notebook) | Needed real thresholds/metrics to write interpretation (unlike the earlier hyperparameter-only implementation notes, which left execution to the user). |
| Model summary cell | Appended after decision-rule / baseline section | User request: chooser + justifications + per-system overview for future use-case picking. |

---

## Key results recorded from the executed run

Approximate test metrics from the head-to-head (single full-notebook execution in this chat):

| System | Threshold | Precision | Recall | F1 | FN | FP | ROC-AUC |
|---|---|---|---|---|---|---|---|
| `rf` | 0.5 | 0.583 | 0.441 | 0.502 | 209 | 118 | 0.809 |
| `rf_tuned_base` | 0.5 | 0.630 | 0.492 | 0.553 | 190 | 108 | 0.830 |
| base F1 | 0.32 | 0.524 | 0.741 | 0.614 | 97 | 252 | 0.830 |
| base prec | 0.38 | 0.559 | 0.663 | 0.606 | 126 | 196 | 0.830 |
| `rf_tuned` | 0.5 | 0.501 | 0.781 | 0.610 | 82 | 291 | 0.831 |
| weighted F1 | 0.50 | 0.501 | 0.781 | 0.610 | 82 | 291 | 0.831 |
| weighted prec | 0.61 | 0.574 | 0.655 | 0.612 | 129 | 182 | 0.831 |

Other notes from that run:

- Unweighted best params: `max_depth=10`, `min_samples_leaf=5`, `min_samples_split=2` (CV F1 ≈ 0.576).  
- Weighted best params: `max_depth=10`, `min_samples_leaf=20`, `min_samples_split=2`.  
- Unweighted tuned accuracy gap ≈ 0.05 (vs raw `rf` ≈ 0.23) — regularization worked without class weights.  
- Baseline thresholds landed **below** 0.5; weighted F1 threshold landed **on** 0.5; weighted prec raised to **0.61**.

Re-running may shift cutoffs slightly; treat the table as a snapshot of the chat execution, not a hard-coded contract.

---

## Where this implementation followed the plan

- New section after the weighted threshold work; weighted cells left intact.  
- Unweighted `GridSearchCV` with same grid/CV/F1-refit; no `class_weight`.  
- `rf_tuned_base` evaluated @ 0.5 with overfitting gap reported.  
- Threshold search on train only with F1 pick + precision@`RECALL_FLOOR` pick (shared floor).  
- Test used once per frozen baseline cutoff.  
- Head-to-head includes both paths’ key operating points.  
- Markdown states opposite threshold direction and records preferred path.  
- Variable names do not overwrite weighted-path objects.  
- Out-of-scope items skipped: nested CV, putting `class_weight` in the same grid, SMOTE, model persistence, rewriting weighted sections.

---

## Deviations from the plan (and why)

| Plan text / suggestion | What we did | Why it differed |
|---|---|---|
| Plan cites weighted raise ~**0.54–0.58** as typical | Live CV produced weighted `best_t=0.50`, `best_t_prec=0.61` (and an earlier notebook state had ~0.53 / 0.60) | Plan numbers are illustrative hypotheses. Selection always comes from CV; never hard-coded. |
| Plan code snippet builds fold RF inline | Used `tuned_kwargs_base = {…, **gs_base.best_params_}` like the weighted cell | Parallelism with existing notebook style; behaviorally equivalent. |
| Plan comparison table is a conceptual markdown layout | **Code-generated** `DataFrame` (plus Δ column) | Matches how the weighted threshold comparison already works; stays correct after re-runs. |
| Optional side-by-side CMs | Included **two** CMs (weighted prec vs base prec) | Enough for production-candidate comparison; avoided cluttering with every operating point. |
| Plan leaves production choice open after the table | Interpretation **recommends weighted prec** as preferred, baseline as ablation | Metrics supported it (fewer FPs at matched recall floor, tied ROC-AUC). Still documented as a decision, not an overwrite of weighted cells. |
| Plan does not require a closing “model catalog” | Added **Model summary** section at notebook end (user follow-up) | Makes chooser + use cases discoverable without re-reading the whole chat. |
| Earlier related implementation notes often authored cells without executing | **Full notebook executed** in this chat | Interpretation and summary needed real thresholds/metrics. |

**Non-deviations worth stating:** did not remove weighted sections; did not put `class_weight` inside one joint grid; did not use test metrics to choose params or thresholds; did not change `RECALL_FLOOR` independently for the baseline path.

---

## How this fits into `random-forest-model.ipynb` structure

High-level notebook flow after this work:

```text
[Setup]
  Imports, load telco train/test, label sanity check

[Baseline unconstrained path]
  Train `rf` → evaluate @ 0.5
  Train `rf_balanced` → evaluate / compare
  Overfitting check (~0.22+ gap)

[Weighted path — already present]
  ## Hyperparameter tuning … → `rf_tuned` (class_weight="balanced")
  Evaluate @ 0.5, overfitting, three-way comparison
  ## Threshold tuning … → `best_t`, `best_t_prec`
  Test @ both cutoffs, operating-point comparison, decision rule

[New: unweighted alternate path]
  ## Alternate path: unweighted baseline — tune then threshold
  `gs_base` / `rf_tuned_base` (no class_weight)
  CV table, test @ 0.5, overfitting
  Threshold CV → `best_t_base`, `best_t_prec_base`
  Test @ both baseline cutoffs, overfitting
  Head-to-head vs weighted + CMs
  Interpretation + decision rule (unweighted path)

[New: catalog]
  ## Model summary — which system for which situation
  Chooser table with justifications
  Per-system strengths / weaknesses / future use
```

### Structural relationships

- **Depends on earlier cells:** `X_train` / `y_train` / `X_test` / `y_test`, `param_grid`, `scoring`, `RECALL_FLOOR`, weighted predictions (`y_pred_tuned`, `y_test_hat`, …), and `churn_metrics_at_threshold`.  
- **Does not replace** the weighted path; it **extends** the notebook with an ablation after the weighted threshold story is complete.  
- **Narrative arc:** diagnose overfitting → regularize with weights → set weighted operating point → ask whether weights were necessary by repeating tune+threshold without them → summarize which system fits which business situation.

### Approximate cell roles for the new baseline section

| Role | Content |
|---|---|
| Section intro | Same grids/CV; only change is no `class_weight`; naming table |
| `gs_base.fit` | Unweighted GridSearchCV → `rf_tuned_base` |
| CV top-10 | Tradeoff table from `gs_base.cv_results_` |
| Test @ 0.5 | Report + CM/ROC + overfitting gap |
| Threshold CV | `best_t_base` / `best_t_prec_base` + plot |
| Test @ cutoffs | Reports, plots, frozen-cutoff overfitting |
| Head-to-head | Comparison table + two prec@floor CMs |
| Closing markdown | Interpretation (filled from run) + decision rule |
| Model summary (end) | Chooser + per-model catalog |

---

## Acceptance checklist status (implementation chat)

| Checklist item | Status |
|---|---|
| New section without breaking weighted `rf` / `rf_balanced` / `rf_tuned` / threshold cells | Done |
| Unweighted `GridSearchCV` same grid/CV/F1-refit; no `class_weight` | Done |
| `rf_tuned_base` evaluated @ 0.5; overfitting gap reported | Done |
| Threshold search train-only; F1 + precision@`RECALL_FLOOR` (same floor) | Done |
| Test used once per frozen baseline cutoff | Done |
| Comparison table includes both paths’ key operating points | Done |
| Markdown states opposite threshold direction + preferred path | Done |
| Variable names do not overwrite weighted-path objects | Done |
| Model summary / chooser at notebook end | Done (user follow-up; not in original plan checklist) |

---

## Practical notes for re-running / review

- Run the notebook top-to-bottom before treating printed thresholds or the head-to-head as authoritative; cutoffs can shift slightly across full re-executions.  
- Expect two GridSearch passes (~60 configs × 5 folds each) plus two threshold CV loops — slower than the weighted-only notebook.  
- When comparing paths, prefer **precision@floor** columns (shared `RECALL_FLOOR`) over raw @ 0.5 columns; @ 0.5 mixes ranking with an arbitrary default cutoff.  
- If changing `RECALL_FLOOR`, change it once and re-run **both** threshold sections so the ablation stays fair.  
- Production default documented in-notebook: **weighted prec @ `best_t_prec`**; use the closing model summary to pick alternatives by outreach cost or class-weight policy.
