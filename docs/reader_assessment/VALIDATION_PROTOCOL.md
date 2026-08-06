# Reader Assessment v2 Validation Protocol

## Purpose

Validate the implementation without confusing self-consistent simulation with real measurement validity.

## Stage A — static and metamorphic gates

Required:

- unique passage and question IDs;
- balanced answer-key positions;
- complete construct and provisional-difficulty coverage;
- no answers or explanations in public item payloads;
- fixed layout during ability evidence collection;
- equivalent dwell estimates for equivalent 8 Hz and 16 Hz sample streams;
- tracking confidence changes measurement quality, not raw behaviour;
- unsupported constructs always abstain;
- invalid timestamps and sparse sampling are surfaced as quality limitations;
- signed round results reject answer or passage substitution.

## Stage B — CPU Monte Carlo software validation

The runner evaluates two regimes:

1. `matched_model_assumptions`: responses are generated from the same provisional 3PL assumptions used by the estimator. This is only a mathematical sanity check.
2. `shifted_item_parameters`: the hidden response generator perturbs difficulty, discrimination, and guessing parameters. The estimator never sees the perturbations. This tests robustness to modest item-parameter mismatch.

The runner performs no parameter fitting and never reads question text or answer content when generating correctness. Therefore its output cannot overfit the pilot QA content. It still does **not** constitute item or construct validation.

Run from the repository root:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_reader_assessment_validation `
  --participants 20000 `
  --seed 20260806 `
  --output-dir docs/reader_assessment/experiments `
  --name validation_v2
```

Software gates:

- matched-assumption Spearman correlation ≥ 0.65;
- matched-assumption RMSE < 0.90 logits;
- matched-assumption 95% interval coverage ≥ 0.85;
- shifted-parameter Spearman correlation ≥ 0.55;
- all paths remain between 4 and 6 rounds;
- all static/metamorphic gates pass.

These are engineering regression gates, not claims about human ability measurement.

## Stage C — real-data calibration (not yet run)

Before fitting:

- freeze participant train/development/confirmation groups;
- freeze passage/item train and held-out forms independently;
- freeze exclusions and minimum tracking-quality rules;
- choose an independent external criterion;
- preregister primary endpoints and subgroup/DIF checks.

Required analyses:

- item fit, difficulty, discrimination, guessing, local dependence, and dimensionality;
- person and item separation/reliability;
- posterior predictive checks;
- participant-only, item-only, and joint holdout performance;
- standard error and credible-interval calibration;
- test-retest reliability;
- DIF/fairness by relevant language background, device, and accessibility groups;
- convergent and discriminant validity;
- external confirmation with no threshold retuning.

Only this stage can justify changing `english_proficiency.status` or `general_reading_ability.status` away from `not_estimated`.
