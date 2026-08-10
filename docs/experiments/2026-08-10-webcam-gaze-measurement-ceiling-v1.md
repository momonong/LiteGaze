# Webcam Gaze Measurement Ceiling v1 - Existing-Data Audit

Status: `failed_integrity_gate`; exploratory self-development evidence only. This audit does not promote a model, threshold, gaze quality band, or participant claim.

Machine-readable result: [`results/2026-08-10-webcam-gaze-measurement-ceiling-v1.json`](results/2026-08-10-webcam-gaze-measurement-ceiling-v1.json)

## Integrity and provenance

| Check | Result |
| --- | --- |
| Artifact bindings | passed |
| Server session/manifest capture contract | failed |
| Calibration/evaluation viewport contract | passed |
| Cross-phase camera aspect-ratio integrity | failed |
| Calibration/evaluation target independence | failed |
| Model validation metric consistency | failed |
| Frozen-v2 receipt uncertainty integrity | not_applicable |
| Calibration targets | 13 |
| Evaluation targets | 5 |
| Below-tolerance overlaps | 5 |
| Target-separation tolerance | 0.20 signed = 0.10 viewport fraction |
| Observed minimum target distance | 0.000393 signed = 0.000196 viewport fraction |
| Images, videos, Torch, or GPU opened | no |
| Natural-reading nearest-word index used as truth | no |

Input SHA-256 values:

- `participant_session`: `d420db6a05bfd7685d52da3e13a08a498430b4d8f2c3cba6fb0841e33b23e67e`
- `calibration_session_metadata`: `5ae737030f096c18a16eb8e8e68d5a1772fb1a717183cb430c80e3fe56d78d64`
- `calibration_manifest`: `af61c6d83708292d98a707f6a5fd083bd99db6beae13d369b09ad4d6130bc7d4`
- `model_artifact`: `012dbd0c7c11c8b34536895964f82ed4b2352c0585f2154b6ba0aa548a5ba19c`

Target-distance coordinates use `signed = 2 * viewport_fraction - 1`; the frozen `0.20` signed Euclidean threshold therefore equals `0.10` in `[0, 1]` viewport-fraction coordinates. Distances equal to the threshold are independent; only smaller distances overlap.

Server-side calibration session metadata is authoritative for capture provenance.

- Session capture source: `study-direct-frame`
- Manifest capture sources: `direct-frame`

Cross-phase camera geometry uses the calibration manifest and the participant system-check record. Aspect ratio is a hard integrity boundary; absolute resolution and frame rate are diagnostic warnings only.

- Calibration camera resolutions: `[{"aspect_ratio": 1.7777777777777777, "height_px": 720.0, "width_px": 1280.0}]`
- Calibration actual frame rates (fps): `[30.0]`
- Participant system-check camera: `640x480`; estimated FPS band `15_23`
- Maximum absolute aspect-ratio difference: `0.444444` (hard maximum `0.02`)
- Diagnostic warnings: `absolute_camera_resolution_changed_diagnostic_only, calibration_frame_rate_outside_participant_estimated_band_diagnostic_only`

**Hard provenance failure:** at least one manifest capture field does not match the server-created calibration session. The numeric audit is retained for diagnosis but is ineligible for promotion.

**Hard cross-phase camera-geometry failure:** `cross_phase_camera_aspect_ratio_mismatch`. The numeric audit is retained for diagnosis but cannot support a matched-capture measurement claim.

**Target-independence failure:** calibration and evaluation share the following below-tolerance target region(s): `bottom_left, bottom_right, center, top_left, top_right`. The frozen threshold is `0.20` in signed `[-1, 1]` Euclidean coordinates, equal to `0.10` in `[0, 1]` viewport-fraction coordinates. Metrics remain descriptive and cannot establish target-held-out accuracy.

## Raw fixed-target result

| Phase | Median px | P90 px | Target-macro mean px | Target-macro bias px | Median absolute X px | Median absolute Y px | Coarse nearest-target accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start | 213.30 | 266.07 | 164.56 | 163.01 | 76.76 | 165.64 | 100.00% |
| End | 128.69 | 278.00 | 154.14 | 152.98 | 105.21 | 72.55 | 100.00% |

The five targets are widely separated. High nearest-target accuracy is coarse-region evidence and does not imply line- or word-level resolution.
Target-macro bias is the equal-weight mean magnitude of each target's prediction-centroid bias vector.
P90 uses the participant collection's nearest-rank `ceil(n * p)` definition.

Axis errors preserve direction as well as absolute magnitude:

| Phase | Axis | Signed mean px | Signed median px | Absolute median px | Absolute P90 px |
| --- | --- | ---: | ---: | ---: | ---: |
| Start | X | -53.14 | -69.14 | 76.76 | 161.39 |
| Start | Y | 118.33 | 165.64 | 165.64 | 229.81 |
| End | X | -88.07 | -105.21 | 105.21 | 204.00 |
| End | Y | 77.39 | 72.55 | 72.55 | 241.73 |

Coarse nearest-target confusion matrices (rows are actual targets; columns are predicted targets):

Start:

```json
{
  "bottom_left": {
    "bottom_left": 3,
    "bottom_right": 0,
    "center": 0,
    "top_left": 0,
    "top_right": 0
  },
  "bottom_right": {
    "bottom_left": 0,
    "bottom_right": 3,
    "center": 0,
    "top_left": 0,
    "top_right": 0
  },
  "center": {
    "bottom_left": 0,
    "bottom_right": 0,
    "center": 3,
    "top_left": 0,
    "top_right": 0
  },
  "top_left": {
    "bottom_left": 0,
    "bottom_right": 0,
    "center": 0,
    "top_left": 3,
    "top_right": 0
  },
  "top_right": {
    "bottom_left": 0,
    "bottom_right": 0,
    "center": 0,
    "top_left": 0,
    "top_right": 3
  }
}
```

End:

```json
{
  "bottom_left": {
    "bottom_left": 3,
    "bottom_right": 0,
    "center": 0,
    "top_left": 0,
    "top_right": 0
  },
  "bottom_right": {
    "bottom_left": 0,
    "bottom_right": 3,
    "center": 0,
    "top_left": 0,
    "top_right": 0
  },
  "center": {
    "bottom_left": 0,
    "bottom_right": 0,
    "center": 3,
    "top_left": 0,
    "top_right": 0
  },
  "top_left": {
    "bottom_left": 0,
    "bottom_right": 0,
    "center": 0,
    "top_left": 3,
    "top_right": 0
  },
  "top_right": {
    "bottom_left": 0,
    "bottom_right": 0,
    "center": 0,
    "top_left": 0,
    "top_right": 3
  }
}
```

## Layout-relative resolution

Configured line gap: `27.20 px`; median word width: `40.90 px`.

| Phase | Median in line gaps | P90 in line gaps | Median in word widths | P90 in word widths |
| --- | ---: | ---: | ---: | ---: |
| Start | 7.84 | 9.78 | 5.22 | 6.51 |
| End | 4.73 | 10.22 | 3.15 | 6.80 |

These ratios describe measurement resolution only; the natural-reading trace has no independent line or word truth.

## Target-wise drift

| Target | Centroid drift X px | Centroid drift Y px | Drift magnitude px | Median error change px |
| --- | ---: | ---: | ---: | ---: |
| bottom_left | -20.32 | -0.05 | 20.32 | 21.85 |
| bottom_right | -22.18 | -25.94 | 34.13 | 22.56 |
| center | -70.01 | 36.37 | 78.89 | 58.68 |
| top_left | -30.78 | -120.47 | 124.34 | -47.32 |
| top_right | -31.38 | -94.62 | 99.69 | -106.35 |

A single start-minus-end median can conceal target-specific reversals; the vectors above remain the primary drift description.

## Start-trained temporal correction

Frozen correction: `start_trained_median_translation`. Translation was fit only on start validation and applied once to end validation.

| Metric | Raw end | Corrected end | Corrected - raw |
| --- | ---: | ---: | ---: |
| Median spatial error px | 128.69 | 148.43 | 19.74 |
| P90 spatial error px | 278.00 | 199.41 | -78.59 |

Target-cluster bootstrap (20000 resamples, seed `20260810`) gives a corrected-minus-raw median-error 95% interval of `[-155.90, 144.69] px`; `38.98%` of resamples improve.
The paired resampling unit is `evaluation_target_id` (5 observed clusters; 5 draws per resample). Sampler: `sha256(seed:resample_index:draw_index) modulo target_count`.
The bootstrap interval is descriptive only and does not establish a population-level correction benefit.

This result cannot relabel the session or select a production correction.

## Start-only repeatability proxy (descriptive only)

Claim boundary: `proxy_not_predictive_uncertainty`. The score is computed only from repeated start-validation predictions; target risk is computed only from end-validation target error. The analysis unit is a whole target cluster, not an individual frame or reading sample.

The coverage grid is frozen at `20/40/60/80/100%`; it is not searched, and this result cannot select an abstention threshold, change a quality band, or authorize per-sample abstention.

| Target | Start repeats | Start RMS repeatability px | End samples | End mean error px | End median error px |
| --- | ---: | ---: | ---: | ---: | ---: |
| bottom_left | 3 | 15.18 | 3 | 43.89 | 50.69 |
| bottom_right | 3 | 16.27 | 3 | 108.23 | 108.63 |
| center | 3 | 10.84 | 3 | 282.22 | 278.00 |
| top_left | 3 | 36.45 | 3 | 218.71 | 218.75 |
| top_right | 3 | 42.40 | 3 | 117.65 | 128.69 |

| Requested coverage | Achieved coverage | Retained targets | End target-macro mean error px | End target-macro median error px |
| ---: | ---: | --- | ---: | ---: |
| 20% | 20% | center | 282.22 | 278.00 |
| 40% | 40% | center, bottom_left | 163.05 | 164.35 |
| 60% | 60% | center, bottom_left, bottom_right | 144.78 | 108.63 |
| 80% | 80% | center, bottom_left, bottom_right, top_left | 163.26 | 163.69 |
| 100% | 100% | center, bottom_left, bottom_right, top_left, top_right | 154.14 | 128.69 |

Target-level Spearman association (`spearman_start_proxy_vs_end_target_mean_error`): `-0.100`; a useful low-to-high risk proxy would have a positive association.

At 20% coverage, end target-macro mean error was `282.22 px`, versus `154.14 px` at full coverage (difference `128.08 px`). Recorded conclusion: `available_start_repeatability_proxy_does_not_rank_end_risk`.

This is a preserved negative descriptive result, not predictive uncertainty calibration.

## Receipt-verified held-out uncertainty coverage-risk

Status: `not_evaluable`; model artifact has no uncertainty_v2 bundle.

No fixed coverage-risk curve, threshold, abstention policy, or quality-band change is authorized. The start-only repeatability proxy remains explicitly `proxy_not_predictive_uncertainty`.

## Predictive uncertainty v2 evidence requirements

Status: `required_before_predictive_uncertainty_claim`. A receipt-verified descriptive fixed-target coverage-risk curve is not reconstructable.

Current evidence inventory:

- Model OOF/uncertainty fields: `none`
- Validation uncertainty fields: `none`
- Reconstructable calibration sensor fields: `none`
- Reason: model artifact has no uncertainty_v2 bundle.

A frozen v2 must record one row per outer-fold held-out sample with: `sample_id`, `outer_fold_id`, `outer_holdout_group_id`, `target_id`, `oof_predicted_x_px`, `oof_predicted_y_px`, `oof_residual_x_px`, `oof_residual_y_px`, `oof_spatial_error_px`, `training_only_ood_score`, `training_only_leverage_score`, `training_only_prediction_covariance_px`.

The uncertainty definition must be bound before evaluation with: `uncertainty_definition_id`, `uncertainty_definition_version`, `uncertainty_definition_sha256`, `training_partition_only_fit_proof`, `coverage_grid`, `frozen_abstention_thresholds_or_explicit_none`.

The score, OOD/leverage model, and covariance must be fit using training partitions only. Evaluation requires a new untouched capture, preserves raw and abstained predictions, and cannot use holdout target error to construct the uncertainty score. V1 may not choose a definition or threshold from this descriptive result.

## Model metric contract

| Field | Value |
| --- | ---: |
| Selected calibrator | gaze_polynomial |
| Selected nested outer macro px | 204.58 |
| Selected stage validation px | 204.58 |
| Top-level validation px | 199.18 |
| Hyperparameter CV px | 199.18 |
| Stage hyperparameter CV px | 199.18 |
| Top-level hyperparameter CV px | n/a |
| Metric consistency | failed |

A failed consistency check records the historical M0 artifact bug; it does not rewrite the artifact or substitute the inner CV score for held-out evidence.

## Not evaluable

- Predictive uncertainty calibration: **not evaluable**; model artifact has no uncertainty_v2 bundle.
- Natural-reading line accuracy: **not evaluable**; no independent line-level ground truth exists.
- Natural-reading word accuracy: **not evaluable**; no independent word-level ground truth exists.

## Decision

Preserve the negative and mixed findings. The current data support at most coarse fixed-target development evidence. Any failed integrity check is a hard stop. No quality band, production model, or line/word claim is promoted.
