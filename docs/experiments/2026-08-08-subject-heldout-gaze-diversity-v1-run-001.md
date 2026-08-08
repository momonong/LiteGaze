# Subject-Heldout Gaze Diversity v1 - Run 001

- Protocol: `subject-heldout-gaze-diversity-v1`
- Protocol commit: `1c66f88c4aa852d3d466527305fd91b3cf70fc9e`
- Data: official MPIIGaze 15-person evaluation subset; 45,000 rows
- Official repeated rows retained: `4`
- Split: 15 outer held-out people, nested validation person, 13 training people
- Production model changed: **no**
- Decision: **`passed`**

## Aggregate result

| Model | Macro subject angular error (deg) |
| --- | ---: |
| Constant training mean | 9.2146 |
| Pose-only ridge | 9.1250 |
| EyePoseTinyCNN-v1 | 6.9450 |
| Shuffled-label sentinel | 9.1952 |

## Held-out subjects

| Subject | Constant | Pose-only | Candidate (3-seed mean) | Sentinel | Candidate - pose |
| --- | ---: | ---: | ---: | ---: | ---: |
| p00 | 8.4356 | 8.2543 | 5.0806 | 8.4361 | -3.1737 |
| p01 | 8.6079 | 8.5084 | 7.3125 | 8.6729 | -1.1959 |
| p02 | 9.0158 | 8.9701 | 5.9379 | 9.0208 | -3.0322 |
| p03 | 9.5967 | 9.5440 | 7.6046 | 9.4954 | -1.9395 |
| p04 | 8.9716 | 8.5678 | 7.4799 | 8.9391 | -1.0879 |
| p05 | 9.7256 | 9.9281 | 6.8365 | 9.7644 | -3.0916 |
| p06 | 10.3409 | 10.2581 | 7.3216 | 10.2807 | -2.9365 |
| p07 | 9.4935 | 9.4616 | 6.8931 | 9.4368 | -2.5685 |
| p08 | 9.6058 | 9.6609 | 7.7826 | 9.5853 | -1.8783 |
| p09 | 8.9639 | 8.9830 | 9.4191 | 8.9638 | +0.4361 |
| p10 | 8.6573 | 8.2309 | 7.3100 | 8.5885 | -0.9209 |
| p11 | 8.4243 | 8.8335 | 5.6215 | 8.4916 | -3.2120 |
| p12 | 8.6124 | 8.6273 | 6.0849 | 8.5931 | -2.5424 |
| p13 | 8.9997 | 8.4028 | 6.7210 | 8.9553 | -1.6818 |
| p14 | 10.7683 | 10.6438 | 6.7686 | 10.7044 | -3.8752 |

## Paired participant inference

- Candidate - pose-only macro difference: `-2.1800` degrees
- Participant-bootstrap 95% CI: `[-2.7186, -1.5944]`
- Held-out subjects improved: `14/15`
- Candidate seed macro SD: `0.1688` degrees

## Decision gates

- [x] `data_audit_passes`
- [x] `subject_overlap_count_equals_zero`
- [x] `candidate_macro_mean_less_than_pose_only`
- [x] `candidate_minus_pose_only_bootstrap_ci_upper_less_than_zero`
- [x] `subjects_candidate_beats_pose_only_at_least`
- [x] `candidate_seed_macro_standard_deviation_within_limit`
- [x] `candidate_worst_subject_mean_within_limit`
- [x] `shuffled_label_macro_worse_than_candidate_by_margin`
- [x] `shuffled_label_does_not_beat_pose_only`
- [x] `gpu_peak_process_memory_within_limit`
- [x] `gpu_temperature_below_limit`
- [x] `wall_time_within_limit`
- [x] `production_model_unchanged`

## Hardware and integrity

- Model execution: `1.318` hours
- Peak process VRAM: `0.559` GiB
- Peak GPU temperature: `70.0` C
- Peak observed utilization: `100.0%`
- Network attempts: `0`
- Protocol SHA-256: `6203b63cb1eb1620bd42d410cc7dd3403ec31ac19a161d91f01ce0b050c49fd9`
- Data SHA-256: `cef00ac1806c6d5ea416d71c975f503fd17cb0eac090ff9ad1f39aeb4764ecf0`
- Implementation SHA-256: `2298f777a56fc38b960175b5eda7959b310c92e0ea0e348b5dc073d7ac3ef86e`
- Production SHA-256 before: `ab6ecdd4db6c7ebfbf1a55c51cc123ba487dc4a04f8c37ef4574ef5d60229f1b`
- Production SHA-256 after: `ab6ecdd4db6c7ebfbf1a55c51cc123ba487dc4a04f8c37ef4574ef5d60229f1b`

## Interpretation boundary

This experiment measures cross-person generalization on the balanced public MPIIGaze eye-image subset. It does not demonstrate improvement on LexiGaze webcam captures, does not compare independently against the production UniGaze joint checkpoint, and cannot change the production default. Any follow-up must use a newly frozen cross-dataset or real-capture protocol.
