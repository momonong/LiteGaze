# Subject-Heldout Gaze Diversity v1 — Experiment Log

## Objective

Test whether a small eye-image plus head-pose model learns gaze information that
generalizes to entirely unseen people. This is a research-only public-dataset
experiment. It does not change the production UniGaze path and does not use any
question-answer outcomes or LexiGaze participant-study data.

## Frozen design

- Branch: `experiment/subject-heldout-gaze-diversity-v1`
- Protocol commit: `1c66f88c4aa852d3d466527305fd91b3cf70fc9e`
- Protocol SHA-256:
  `6203b63cb1eb1620bd42d410cc7dd3403ec31ac19a161d91f01ce0b050c49fd9`
- Dataset: the official 3,000-row-per-person MPIIGaze eye-image evaluation
  subset, 15 people and 45,000 rows total.
- Evaluation: 15-fold leave-one-person-out, with the next person used for
  validation and the other 13 people used for training.
- Candidate: the frozen `EyePoseTinyCNN-v1`, evaluated with three fixed seeds.
- Controls: constant training-mean baseline, pose-only ridge baseline, and a
  within-training-person shuffled-label sentinel.
- Hardware budget: one CUDA GPU, FP32, no TF32 or AMP, at most 6 GiB process
  VRAM, temperature below 82 C, and at most six hours of model execution.

The design follows MPIIGaze's subject-independent evaluation principle and its
official balanced eye-image subset. The production UniGaze joint checkpoint is
excluded because its joint training configuration includes MPIIGaze, so it
would not be an independent comparison on this dataset.

Primary references:

- [MPIIGaze / MPIIFaceGaze dataset and evaluation paper](https://collaborative-ai.org/publications/zhang19_pami.pdf)
- [UniGaze WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/papers/Qin_UniGaze_Towards_Universal_Gaze_Estimation_via_Large-scale_Pre-Training_WACV_2026_paper.pdf)

## Pre-execution record

### 2026-08-08 — protocol freeze

The protocol was committed before importing the model runtime or inspecting any
candidate held-out outcome. Architecture, split schedule, seeds, optimizer,
early stopping, controls, metrics, gates, and resource limits are immutable for
v1. Any outcome-driven change requires a separately frozen v2 protocol.

### 2026-08-08 — legacy path rejected

The old `scripts/eai` training path was not reused because it depends on missing
legacy runtimes, evaluates on random rows instead of held-out people, and
contains a residual connection that adds a tensor to itself. Its checkpoints
and pseudo-labels are explicitly excluded from v1.

### 2026-08-08 — CPU contracts

Thirteen focused tests passed for:

- disjoint nested subject splits;
- exact sample-list row handling;
- gaze-vector and head-pose angle conventions from the dataset documentation;
- training-only pose standardization;
- within-subject sentinel permutation;
- baseline dimensions and numerical behavior;
- deterministic participant bootstrap;
- exact candidate and sentinel seed coverage;
- atomic, integrity-bound fold persistence and resume behavior;
- decision-gate behavior and production-code integrity;
- formal-run network isolation; and
- exclusion of private and question-answer outcomes.

The focused test import was verified not to load PyTorch.

### 2026-08-08 — source-data audit

The full audit passed before model execution:

- 15 subjects;
- 45,000 official rows;
- 1,500 left-eye and 1,500 right-eye rows per subject;
- zero train/validation/test subject overlap across all folds;
- 534 hashed source files; and
- source SHA-256:
  `cef00ac1806c6d5ea416d71c975f503fd17cb0eac090ff9ad1f39aeb4764ecf0`.

The official `p13` sample list contains four repeated file-and-eye references,
giving 44,996 unique identities across 45,000 official rows. The first audit
correctly surfaced this instead of silently proceeding. The final loader keeps
all official rows and reports the four repetitions rather than deduplicating
them: the frozen protocol specifies the exact official 3,000-row list, and
whole-subject assignment means the repetitions cannot cross train, validation,
or test roles.

### 2026-08-08 — repository regression gate

The final pre-run offline quality gate passed all 124 tests in 18 explicit test
targets. Its
safeguards reported no network attempts, no credentials, no PyTorch import, no
GPU access, and no tracked artifact mutation.

### 2026-08-08 — GPU scheduling decision

The CUDA smoke and formal run were deferred when telemetry showed an unrelated
transformer benchmark already occupying the RTX 5090 Laptop GPU, with high
utilization and temperature reaching the protocol boundary. No process was
stopped and LexiGaze did not add a competing workload. Execution resumes only
after a safe idle window.

### 2026-08-08 — synthetic CUDA smoke

The smoke ran only after the competing process completed and three preflight
samples showed a safe window. It passed with:

- exactly 81,570 candidate parameters;
- two complete synthetic training epochs and finite test predictions;
- no real-dataset access and no candidate outcome;
- zero network attempts;
- 0.189 GiB peak process VRAM;
- 65 C peak GPU temperature; and
- 2% peak observed GPU utilization at telemetry checkpoints.

### 2026-08-08 — formal-run scheduling interruption

The first formal attempt started from a clean work directory and completed the
first candidate job. While its second seed was training, another project's
automatic GPU rerun started and shared the same device. The LexiGaze process was
the only process stopped; the other workload was not modified.

The completed job remains integrity-bound to protocol, data, and implementation
hashes. The interrupted second job had reached epoch 18 but had not restored a
checkpoint, evaluated its held-out test subject, or written a job artifact, so
no partial test outcome exists. Resumption must reuse the exact implementation
SHA-256 `2298f777a56fc38b960175b5eda7959b310c92e0ea0e348b5dc073d7ac3ef86e`
and the original frozen parameters. The first completed job consumed 110.2
seconds, 0.559 GiB peak process VRAM, and reached 70 C.

## Formal result

### 2026-08-08 — run 001 completed

All 45 candidate jobs and 15 shuffled-label sentinel jobs completed under the
frozen protocol. Every preregistered gate passed:

- candidate macro subject error: `6.9450` degrees;
- pose-only macro subject error: `9.1250` degrees;
- constant-baseline macro subject error: `9.2146` degrees;
- shuffled-label sentinel macro subject error: `9.1952` degrees;
- candidate minus pose-only: `-2.1800` degrees, a `23.89%` relative reduction;
- participant-bootstrap 95% CI: `[-2.7186, -1.5944]` degrees;
- held-out subjects improved: `14/15`;
- three-seed macro standard deviation: `0.1688` degrees; and
- worst candidate subject error: `9.4191` degrees.

The one non-improved subject was `p09`: the candidate scored `9.4191` degrees
versus `8.9830` for pose-only, a regression of `0.4361` degrees. This exception
is retained as evidence and must not be tuned away on the same subjects.

The shuffled-label sentinel was `2.2503` degrees worse than the candidate and
`0.0702` degrees worse than pose-only. This supports the interpretation that
the candidate used eye-image signal rather than merely exploiting pose or the
split structure.

Completed model jobs consumed `1.318` GPU-hours in aggregate, with `0.559` GiB
peak PyTorch-reserved process VRAM and a `70 C` peak GPU temperature. The
aborted, unscored scheduling attempt contributed approximately five additional
minutes that are not included in the completed-job sum; it produced no
held-out prediction or job artifact. Peak whole-device memory observed during
the completed run was `3571 MiB`. There were zero network attempts, and the
production gaze-source hash remained unchanged.

### 2026-08-08 — independent result audit

An independent post-run calculation loaded all 60 persisted job files instead
of trusting the runner summary. It verified exact candidate and sentinel seed
coverage, equality between job and aggregate records, absence of temporary
files, all four aggregate model metrics, per-seed variability, worst-subject
error, the participant bootstrap interval, resource sums and maxima, finite
JSON values, zero network attempts, and protocol/data/implementation/production
hashes. The independent calculation reproduced the formal decision: `passed`.

### 2026-08-08 — final repository verification

- The isolated offline regression gate passed `124/124` tests across 18
  explicit targets, with network probes blocked, credentials cleared, Torch
  not imported, GPU memory unchanged, and no tracked artifact mutation.
- Ruff lint passed for the gaze-diversity implementation and the updated
  offline gate; all nine experiment Python files matched Ruff formatting.
- Python byte-compilation, `git diff --check`, result finiteness checks, and a
  scan for credential-shaped strings or local absolute paths passed.
- Recomputed implementation and production hashes still exactly matched the
  hashes sealed into the formal result.

## Decision and next boundary

`EyePoseTinyCNN-v1` is retained only as a research baseline. MPIIGaze provides
useful cross-person evidence, but it does not establish improvement for the
LexiGaze webcam pipeline, other devices, new sessions, or production UniGaze.
No production default was changed.

The next gaze experiment must be independently frozen before examining its
outcomes and must use either a genuinely external dataset or fresh participant
capture groups held out by participant, session, and device. The v1 MPIIGaze
subjects must not be used for further architecture or threshold selection.
