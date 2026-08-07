# Quality-Aware Text/Fusion v2 - Run 001

- Protocol: `quality-aware-text-fusion-v2`
- Protocol commit: `cd1ca19d614e9e7b4a2dfc6e9b5a538100a1ca53`
- Compute: CPU only; no Torch, network, participant data, or QA dataset
- Decision: **`record_failure_without_parameter_changes`**
- Production default changed: **no**

## Outcome

The candidate reduced aggregate MAE from `0.092053` to `0.089963` (`-0.002089`, about `2.27%`), but the frozen decision is **fail** because drift MAE increased from `0.113635` to `0.118118` (`+0.004483`). The candidate therefore remains shadow-only and no v2 parameter was changed after this result.

## Frozen corruption benchmark

| Condition | Rows | Missing | Text MAE | Static MAE | Candidate MAE | Candidate - Static | Mean gaze weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 6144 | 142 | 0.101630 | 0.050922 | 0.046275 | -0.004648 | 0.912636 |
| drift | 6144 | 499 | 0.101920 | 0.113635 | 0.118118 | +0.004483 | 0.582333 |
| jitter | 6144 | 739 | 0.102688 | 0.105492 | 0.095608 | -0.009884 | 0.426092 |
| dropout | 6144 | 3682 | 0.100882 | 0.089287 | 0.088888 | -0.000398 | 0.199160 |
| missing | 6144 | 6144 | 0.100927 | 0.100927 | 0.100927 | +0.000000 | 0.000000 |

## Decision gates

- [x] `occurrence_collision_count_equals_zero`
- [x] `missing_fallback_within_tolerance`
- [x] `mean_gaze_weight_order_met`
- [x] `candidate_aggregate_mae_less_than_static`
- [x] `candidate_clean_within_static_margin`
- [x] `missing_candidate_equals_text_only`
- [ ] `candidate_drift_mae_less_than_static`
- [x] `candidate_jitter_mae_less_than_static`
- [x] `candidate_dropout_mae_less_than_static`

## Interpretation boundary

This run tests deterministic software behavior under a frozen synthetic corruption model. It does not establish benefit on real webcam captures and cannot promote the production default. A later protocol must freeze participant, article, and device/source holdouts before any independent real-capture outcomes are inspected.

## Integrity

- Protocol SHA-256: `ff864584d70d7f623c5459cd2b0fcdd08f99f79d67838eba397f569c2d5a6903`
- Implementation SHA-256: `1eb9c2bd1176f94dac9845068f352ec03788c539145c4fa4d28940360aea7678`
- Occurrence collisions: `0`
- Legacy word-key collisions in the same probe: `91`
- Missing fallback max delta: `0.0`
- Torch imported after run: `False`

## Execution audit

- Valid run-001 GPU observation before: `0%` utilization, `113 MiB` allocated.
- Valid run-001 GPU observation after: `0%` utilization, `113 MiB` allocated.
- Two invalid launcher attempts occurred before run-001: one failed during repository import bootstrap and one failed in adjacent gate-pair construction. Neither wrote a result or report. Both were corrected as runner defects without changing the frozen formula, generator, seed, thresholds, or decision gates.

## Verification

- Offline CPU quality gate: `111/111` tests passed; `0` failures, errors, skips, or unexpected successes.
- Safeguards: network and subprocess probes blocked, credentials cleared, no protected artifact mutation, and Torch not imported.
- Quality-gate GPU observation: `0%` utilization and `112 MiB` allocated both before and after.
