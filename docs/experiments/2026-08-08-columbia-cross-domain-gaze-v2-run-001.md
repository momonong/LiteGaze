# Columbia Cross-Domain Gaze v2 - Run 001

## Decision

Execution integrity passed, but neither model passed the pre-registered
effectiveness criteria. The negative result is retained without Columbia-driven
changes to label signs, preprocessing, architecture, epochs, seeds, thresholds,
metrics, or populations.

| Frozen model | Macro subject error | Model - zero | 95% subject-bootstrap CI | Subjects beating zero | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Zero-gaze baseline | 12.0455 deg | - | - | - | Reference |
| EyePoseTinyCNN-v1, 3-seed/2-eye ensemble | 14.2841 deg | +2.2386 deg | [+2.0703, +2.4115] deg | 0/56 | Not confirmed |
| Current UniGaze joint production path, uncalibrated | 17.6857 deg | +5.6402 deg | [+5.4423, +5.8572] deg | 0/56 | External baseline not supported |

The candidate median/P90 errors were `14.4942/22.4361` degrees; production
median/P90 errors were `18.4026/30.6227` degrees. Production preprocessing
coverage was nevertheless `5,880/5,880` (`100%`). Coverage therefore passed as
an execution property, not as an accuracy result.

## Data and execution integrity

- Complete public grid: `56` subjects, `5,880` images, `105` images per subject.
- Official eye-corner annotations: `5,865`; fixed MediaPipe fallback: `15/15`.
- Corrupt images, dimension mismatches, duplicate identities, and grid failures:
  all `0`.
- MPIIGaze final-fit source: all `45,000` frozen evaluation rows from `15`
  subjects; source SHA-256
  `cef00ac1806c6d5ea416d71c975f503fd17cb0eac090ff9ad1f39aeb4764ecf0`.
- Candidate seeds: `20260808`, `20260809`, and `20260810`; exactly `11` epochs
  each, with no validation, early stopping, augmentation, AMP, or TF32.
- Model execution: `0.0866` hours; peak process VRAM `0.8281` GiB; peak GPU
  temperature `63.0 C`; peak utilization `100%`.
- Network attempts during training/inference: `0`.
- Production source SHA-256 before and after:
  `ab6ecdd4db6c7ebfbf1a55c51cc123ba487dc4a04f8c37ef4574ef5d60229f1b`.
- All execution gates passed; the production model and default were unchanged.
- A fresh CPU process reproduced all metrics, bootstrap intervals, and decisions
  exactly from ignored row-level evidence without importing Torch.

## Post-hoc coordinate diagnostic

This section was not pre-registered and cannot alter the primary decision. It
exists only to generate a hypothesis for a new experiment.

Ordinary least-squares slopes of prediction on the filename-encoded target,
with Pearson correlation in parentheses, were:

| Model | Pitch slope (r) | Yaw slope (r) |
| --- | ---: | ---: |
| EyePoseTinyCNN-v1 | +0.174 (+0.512) | -0.247 (-0.531) |
| UniGaze joint | +0.795 (+0.886) | -0.971 (-0.959) |

UniGaze therefore retained strong horizontal ordering but in the direction
opposite to the frozen Columbia label mapping. Its mean error was `3.8695`
degrees at horizontal 0 degrees, then approximately `10.5`, `20.0`, and `29.5`
degrees at horizontal magnitudes 5, 10, and 15 degrees. This is consistent with
a coordinate-interface mismatch, not with an absence of horizontal gaze
signal. The TinyCNN result also showed compression and offset, so a sign change
alone would not establish its cross-domain effectiveness.

No sign-flipped score was computed as a replacement result. The same Columbia
outcomes may not select a mapping and then be reused as independent evidence.
The next valid step is a newly frozen left/center/right coordinate-contract
check using fresh LexiGaze captures, followed by a participant/session/device
holdout. Current calibrated modes may learn a raw-angle-to-screen mapping, but
that must be tested directly rather than inferred from this uncalibrated run.

## Reproducibility and claim boundary

- Frozen v2 protocol SHA-256:
  `6f0f03c60365ac5e7d735cecff24064d8c52a6a28c20a83c257dc94319633fee`.
- Frozen implementation SHA-256:
  `94e9677b255090b8c1f8fb67ca729dd48229362fe690ae94d88f93cab5b96b9f`.
- Aggregate result SHA-256:
  `05b3540ab5c6d0f3fad6d5de7a28e0ab9df46e393a40606a0fa45c7d7a33aa15`.
- Ignored row-level evidence SHA-256:
  `602fee2404ebfa26342704ac97d66288ed119147e5b0fc5481a0594695708a99`.

This is one-time external public-dataset engineering evidence. It does not
establish LexiGaze webcam accuracy, authorize a participant pilot, measure
reading or English ability, or justify a production change. Columbia outcomes
may not be used for a tuned rerun presented as independent confirmation.
