# Gaze motion-robustness baseline and implementation record

- Date: 2026-08-05
- Branch: `research/gaze-motion-robustness`
- GPU used in this experiment: no
- Images opened by the coverage audit: no
- External model calls: no

## Goal and frozen hypothesis

Goal: make personalization measurably less sensitive to modest posture and
camera-distance changes without tuning to one question set, participant, or
confirmation run.

Hypothesis frozen before implementation: the immediate bottleneck is the
person-specific calibration protocol and validation split, not insufficient
capacity in the frozen UniGaze encoder. A motion-diverse protocol plus grouped
validation should be attempted before any GPU fine-tuning.

## Historical-data audit

Command:

```powershell
python -X utf8 -m scripts.audit_gaze_motion_coverage
```

The audit reads only `session.json` and `manifest.jsonl`. Its aggregate output
does not expose participant labels, and the source digest makes the snapshot
reproducible. The complete machine-readable snapshot is stored at
[`results/2026-08-05-motion-coverage-baseline.json`](results/2026-08-05-motion-coverage-baseline.json).

| Measure | Baseline |
| --- | ---: |
| Session directories / manifests | 83 / 83 |
| Participant labels (not asserted to be unique people) | 37 |
| Manifest samples | 1,019 |
| Samples with usable head pose | 1,018 |
| Repeat distribution | 1,019 at repeat 0 |
| Labelled motion blocks | 0 |
| Same session-target repeated across conditions | 0 |
| Head-yaw range | -3.306° to 12.974° (16.280° span) |
| Head-pitch range | 13.197° to 33.285° |
| Face-scale proxy range | 0.401 to 0.749 |
| Required capture metadata populated | 0 / 1,019 for every field |
| Source SHA-256 | `54bc65510db9ef14caaddf927b7dc32ba9bd661b5ac7abe3a9d348fa6ac7907c` |

Result: `not_ready`. The historical rows may still be useful for legacy
calibration, but they cannot support a causal claim about robustness to motion.
The aggregate face-scale range also cannot be treated as within-person distance
coverage because sessions and participant labels are mixed.

## Implemented intervention

### Collection protocol

The gaze page now offers `motion_robust`, which repeats all 13 targets in:

1. neutral posture and nominal distance;
2. head left by about 15°;
3. head right by about 15°;
4. 15–20 cm nearer;
5. 15–20 cm farther.

This yields 65 rows with one repeat. The formerly inert four-corner selector
and delay field now also affect collection. Every direct frame and video
timeline row carries protocol, block, burst, posture, distance, lighting,
camera-role, device-class, frame-size, and frame-rate metadata. Raw browser
device IDs are deliberately not stored.

### Frozen readiness gates

A motion-diverse dataset must satisfy all of the following before training:

- at least 50 usable head-pose samples;
- at least five labelled motion blocks;
- at least nine session-target pairs repeated across conditions;
- at least 20° total head-yaw span;
- at least 15° separation between median left/right head yaw;
- `neutral/left/right` and `nominal/near/far` coverage;
- near and far median face scale at least 5% above/below nominal respectively;
- complete required metadata on every sample, including `capture_run_id` and
  `capture_source` so direct and video-derived artifacts cannot be split.
- zero malformed manifest rows.

The web API returns the aggregate audit for one session. Video extraction
preserves the same metadata. Motion-diverse training stops before model loading
when any gate fails.

Motion-diverse feature extraction defaults to CPU. CUDA is used only when an
experiment explicitly sends `allow_cuda=true`; the hidden-CUDA runtime guard
still takes precedence.

### Leakage-resistant training

For compliant data, all samples from one `motion_block_id` are held out
together. Hyperparameters are selected on leave-one-motion-block-out pixel
error. The saved model records both train error and held-out error, and the UI
displays the held-out value.

The training path compares:

| Candidate | Inputs | Promotion rule |
| --- | --- | --- |
| M0 `gaze_polynomial` | UniGaze yaw/pitch, degree 1 or 2 | Baseline |
| M1 `motion_conditioned_ridge_v1` | gaze, head pose, face center/scale, frozen interactions | Must improve held-out error by ≥5 px and ≥5% |

Feature standardization is fitted separately inside every fold, preventing
validation statistics from leaking into training. The winning candidate is
then refitted on all calibration blocks. Cascaded calibration is disabled for
this protocol.

## Deterministic validation

Fourteen focused CPU tests pass. They cover manifest hashing and filtering,
identity-free aggregate output, incomplete historical data, a compliant
five-block dataset, metadata allow-listing, fold disjointness, feature schema,
standardization, and regression fitting.

The duplicate-frame sentinel is intentionally pathological:

| Split | Held-out error |
| --- | ---: |
| Leave one frame out | 0.0025 px |
| Leave one motion block out | 100.0 px |

The example proves that a near-zero frame-level score can coexist with complete
failure on an unseen motion block. It is a validation test, not a measurement
of LexiGaze's real accuracy.

The focused offline worker also passed every safeguard: 14/14 tests, no network
or child-process attempts, no artifact changes, credentials cleared,
`CUDA_VISIBLE_DEVICES=-1`, and PyTorch not imported. The repository-wide local
worker reached 36/39 tests; its three unrelated inspector imports failed because
the checked-in `.venv` targets a missing Python 3.11 installation while the
available fallback is Python 3.12 and cannot load that environment's compiled
`regex` extension. No source regression was observed in those failures. Exact
CI validation on its pinned Python 3.12 environment remains required before
merge.

Validation commands:

```powershell
# Pure metadata and regression tests (CPU only)
python -X utf8 -m unittest -v `
  scripts.test_gaze_calibration_regression `
  scripts.test_gaze_motion_robustness

# Browser syntax and Python compilation
node --check web/static/gaze_page.js
python -X utf8 -m compileall -q core scripts web/routes
```

## Research basis

- [UniGaze (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Qin_UniGaze_Towards_Universal_Gaze_Estimation_via_Large-scale_Pre-Training_WACV_2026_paper.pdf)
  already emphasizes large, curated in-the-wild pretraining, balanced head pose,
  and identity diversity. This supports measuring the small calibration layer
  before fine-tuning the encoder.
- [ETH-XGaze](https://ait.ethz.ch/xgaze) captures more than one million images
  across 18 cameras with extreme head pose and illumination changes, supporting
  explicit pose/camera-domain coverage.
- [Gaze360](https://gaze360.csail.mit.edu/) treats physically unconstrained gaze
  as a temporal and uncertainty-aware problem rather than an independent-frame
  benchmark.
- [Generalizing Eye Tracking with Bayesian Adversarial Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Generalizing_Eye_Tracking_With_Bayesian_Adversarial_Learning_CVPR_2019_paper.html)
  identifies appearance, head-pose, and point-estimation overfit as distinct
  generalization problems.
- [AGG (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Bao_From_Feature_to_Gaze_A_Generalizable_Replacement_of_Linear_Layer_CVPR_2024_paper.html)
  and [UnReGA (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Cai_Source-Free_Adaptive_Gaze_Estimation_by_Uncertainty_Reduction_CVPR_2023_paper.html)
  motivate treating cross-domain adaptation and uncertainty as later, separately
  validated interventions.

## GPU and overfit gates for the next experiment

No GPU run is justified until at least one new session passes the metadata and
physical-separation gates. The first model run should compare M0/M1 only; it
uses the existing frozen encoder and a single recorded session. Encoder
fine-tuning remains blocked until:

1. at least three independently captured motion-diverse sessions exist;
2. one full session is frozen as confirmation data before any tuning;
3. candidate selection uses motion-block folds only;
4. the exact GPU command, wall time, peak VRAM, seed, input digest, and result
   artifact are recorded;
5. the confirmation session is evaluated once after the decision is frozen.

Question-answer datasets are not used by this gaze calibration path. If gaze is
later evaluated inside a reading/QA task, article and question identity must be
group-held-out and must not become calibration features.

## Current conclusion

The project now has a defensible path to train on more diverse data, but it does
not yet have a real protocol-compliant capture. Therefore the honest current
result is infrastructure and methodology improvement, not an accuracy claim.
The next evidence-producing action is to capture one 65-row session and run the
frozen M0/M1 comparison.

## Follow-up

That capture and frozen comparison are now complete. The aggregate result and
single-capture limitations are recorded in
[`2026-08-05-gaze-motion-run-001.md`](2026-08-05-gaze-motion-run-001.md).
