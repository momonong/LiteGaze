# Columbia Cross-Domain Gaze v1 — Experiment Log

## Objective

Use the independently collected Columbia Gaze Data Set as a one-time external
confirmation set for two already-decided LexiGaze gaze paths:

1. the current `unigaze_b16_joint` full-face production baseline; and
2. the `EyePoseTinyCNN-v1` research candidate retained by the frozen
   subject-heldout MPIIGaze experiment.

The experiment is calibration-free and may not change production behavior. It
does not use question-answer outcomes, LexiGaze participant data, reading
outcomes, or demographic attributes.

## Why this is the next experiment

The MPIIGaze subject-heldout run established cross-person evidence within one
dataset, but its own decision boundary explicitly required a genuinely external
dataset or fresh capture before further claims. Columbia provides 5,880 images
from 56 different people, with a complete factorial grid of five head poses,
three vertical gaze directions, and seven horizontal gaze directions. It was
collected independently of MPIIGaze and is absent from UniGaze's official joint
training configuration.

Primary sources:

- [Columbia CEAL dataset page](https://ceal.cs.columbia.edu/columbiagaze/)
- [Columbia CAVE dataset page](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/)
- [Official UniGaze repository](https://github.com/ut-vision/UniGaze)
- [UniGaze WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/papers/Qin_UniGaze_Towards_Universal_Gaze_Estimation_via_Large-scale_Pre-Training_WACV_2026_paper.pdf)

## Frozen design

- Branch: `experiment/columbia-cross-domain-gaze-v1`
- Protocol: `docs/experiments/protocols/2026-08-08-columbia-cross-domain-gaze-v1.json`
- Dataset terms: non-commercial use only; the source archive and annotations
  remain local and must not be committed or redistributed.
- Target convention: `[pitch, yaw] = radians([V, H])`, with positive `V` up
  and positive `H` image-left/subject-left.
- Production path: resize to width 640, then use the unchanged production
  MediaPipe normalization, ImageNet tensor transform, and raw UniGaze angles.
- Research path: train three fixed-seed TinyCNN models on all 45,000 audited
  MPIIGaze rows for exactly 11 epochs, then evaluate official Columbia eye
  corners using a frozen 60x36 affine crop and unit-vector ensemble.
- Baseline: zero gaze `[0, 0]`, defined before outcomes because the complete
  Columbia gaze grid is symmetric.
- Metrics: participant-macro angular error, paired participant bootstrap,
  per-pose/direction diagnostics, worst-subject behavior, and production
  preprocessing coverage.
- Compute: one CUDA device, at most 6 GiB process VRAM, below 82 C, no TF32 or
  AMP for TinyCNN, no network during model execution, and at most eight hours.
- Decision: effectiveness is evaluated separately from execution integrity. A
  scientifically valid negative result is preserved and must not trigger a
  tuned rerun on Columbia.

## Pre-execution record

### 2026-08-08 — protocol freeze

The protocol was written before downloading the Columbia archive or eye-corner
annotations, inspecting any Columbia image, importing the formal model runtime,
or observing a Columbia prediction. Dataset structure and coordinate semantics
were taken from public documentation only. Any result-driven change to model,
sign mapping, crop geometry, resolution, epochs, seeds, threshold, population,
or metric invalidates Columbia as an independent confirmation set.

Protocol commit and SHA-256 are recorded immediately after this entry is
committed.

