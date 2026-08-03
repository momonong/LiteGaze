# GECO Cross-Subject Generalization Protocol v1

Protocol date: 2026-08-03

Protocol lock: the Git commit that first adds this file and `scripts/evaluate_geco_generalization.py`

Official command: `LEXIGAZE_DEVICE=cpu CUDA_VISIBLE_DEVICES="" uv run python scripts/evaluate_geco_generalization.py`

## Objective

Estimate how well LexiGaze text-derived difficulty features generalize to unseen readers and unseen passages without selecting parameters on the reported test observations.

This experiment does not use a question-answer dataset. The overfitting analogue for GECO is leakage across readers, trials, target reading times, or target-trained feature generators; the controls below address those paths explicitly.

## Data

- Local source: `data/geco/population/`
- Populations: L1 and L2
- Expected participants: 37 (18 L1, 19 L2)
- Expected trials: 5,892 participant-trial pairs
- Per-trial inputs: `layout.csv` and `fixations.csv`
- Duration outcome: positive finite `reading_time`, transformed with `log1p`
- Fixation outcome: `reading_time` is positive and finite

No rows are removed based on the observed outcome magnitude. Rows missing required text features are invalid and cause the run to fail instead of being silently imputed from test data.

The run records a content hash over every used `layout.csv` and `fixations.csv`, plus row, subject, trial, and byte counts.

## Leakage audit

The primary feature set is restricted to signals that do not use GECO reading-time labels:

- `log1p(surprisal_score)`
- `log1p(attention_score)`
- stripped word length
- L2 population indicator

`cognitive_mass` is excluded from the primary model. The current pipeline can generate it with XGBoost/Ridge models trained to predict GECO total reading time, so using it as an ordinary held-out predictor would retain uncertain upstream target leakage. It is reported only as a clearly labelled provenance-risk diagnostic.

The existing single-trial fusion benchmark is also treated as descriptive calibration rather than predictive validation because its simulated gaze dwell is constructed from the same total reading time used as its evaluation target.

## Primary double-holdout analysis

### Split

- Five subject folds, balanced separately within L1 and L2.
- Five trial folds, balanced over unique trial identifiers.
- Ordering is fixed by SHA-256 using seed `20260803`; outcomes are never consulted.
- For each of the 25 subject-fold × trial-fold cells:
  - test rows are the intersection of that subject fold and trial fold;
  - training rows exclude the entire test subject fold and the entire test trial fold;
  - rows sharing only the held-out subject fold or only the held-out trial fold are not used for that model.

Every reported observation is therefore predicted by a model that has seen neither its reader nor its trial.

### Model

- Ridge regression implemented with NumPy.
- Target: `log1p(reading_time)` for fixated words.
- Alpha: fixed at `1.0` before the run.
- Feature means and standard deviations are fitted on the training rows only.
- Intercept is not penalized.
- No hyperparameter search or early stopping is performed.

Prespecified label-free comparators are raw surprisal, raw attention, and word length.

### Primary endpoint

Macro-average of per-participant Spearman correlations between cross-fitted Ridge predictions and observed reading time. Participant is the unit of inference.

Uncertainty and tests:

- 10,000 stratified participant bootstrap samples for a percentile 95% confidence interval.
- 100,000 participant-level sign flips for the two-sided null test against zero.
- 100,000 paired participant-level sign flips for Ridge versus the prespecified surprisal comparator.

Only the Ridge macro Spearman endpoint is primary. Other correlations and error metrics are exploratory and are not used to revise this protocol.

## Secondary known-passage/new-reader analysis

For each participant and word item, compute population priors using all other participants in the same population, trial, and word position:

- fixation-rate prior, evaluated with per-participant ROC AUC and Brier score;
- mean `log1p(reading_time)` among other participants who fixated the item, evaluated with per-participant Spearman correlation and log-time MAE.

The held-out participant's own outcome is subtracted before either prior is calculated. This protocol evaluates a new reader on already observed passages and is reported separately from the new-reader/new-passage primary analysis.

## Stopping and reporting rules

1. Run the full discovered population once with the fixed defaults.
2. Do not change features, folds, alpha, exclusions, or the primary metric after inspecting results.
3. Failed data-integrity checks stop the run and are reported; they are not bypassed by dropping inconvenient subjects or trials.
4. Any future modification is a new protocol version and must report v1 unchanged alongside it.
5. Report negative, null, and provenance-risk results as prominently as positive results.
6. Run CPU-only. No language-model inference or GPU allocation is required because all text features are already cached.
