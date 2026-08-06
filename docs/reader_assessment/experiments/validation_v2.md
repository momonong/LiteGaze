# Reader Assessment v2 CPU Validation

- Simulated participants: `40000` (20000 per regime)
- Seed: `20260806`
- Runtime: `290.418 s`
- GPU requested: `False`
- Parameter fitting: `False`
- QA content used for fitting: `False`
- Overall gate: `PASS`
- Operational measurement ready: `False`

## Gates

| Gate | Result |
| :--- | :---: |
| item_bank_static_audit | PASS |
| matched_spearman_at_least_0_65 | PASS |
| matched_rmse_below_0_90 | PASS |
| matched_interval_coverage_at_least_0_85 | PASS |
| shifted_spearman_at_least_0_55 | PASS |
| rounds_within_protocol | PASS |
| gaze_metamorphic_checks | PASS |

## Simulation results

| Regime | Spearman theta | RMSE | 95% interval coverage | Mean rounds |
| :--- | ---: | ---: | ---: | ---: |
| Matched assumptions | 0.8408 | 0.6297 | 0.9365 | 5.9893 |
| Shifted item parameters | 0.8448 | 0.6173 | 0.9401 | 5.9913 |

Max-round fractions were `0.9893` (matched) and `0.9913` (shifted). A high value means the current six-passage pilot bank is not yet an efficient variable-length CAT.

## Interpretation boundary

Simulation validates software behaviour under declared assumptions only. It does not establish item calibration, CEFR linkage, external validity, or fairness.

The shifted regime perturbs difficulty, discrimination, and guessing parameters that the estimator never sees. It is a robustness stress test, not a substitute for real participant/item holdout validation.
