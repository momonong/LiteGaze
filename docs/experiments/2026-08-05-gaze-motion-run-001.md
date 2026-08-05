# Gaze motion-shift run 001: preregistration and result

- Date: 2026-08-05
- Branch: `experiment/gaze-motion-shift-robustness`
- Execution device: CPU only
- Question/answer datasets: none
- Evidence class: one real, motion-diverse physical capture

## Frozen objective

Determine whether the existing pose/distance-conditioned calibration layer
(M1) improves over the gaze-only polynomial calibration (M0) when an entire
posture or distance block is unseen during evaluation. This run does not test
cross-person or cross-session generalization.

## Data gate observed before model execution

The new capture contains 65 rows: 13 targets in each of `neutral`, `left`,
`right`, `near`, and `far`. The metadata-only frozen coverage audit returned
`ready` with no issues. Its source SHA-256 is
`87b303655095f360eaa68fc910f12df404345d5c1fad04a2c7e945a59db33d36`.
The complete aggregate audit is stored in
[`results/2026-08-05-motion-run-001-coverage.json`](results/2026-08-05-motion-run-001-coverage.json).

The model was not loaded and no image was opened before freezing the rules
below.

## Frozen candidates and validation

- M0 `gaze_polynomial`: degree 1 or 2; ridge alpha from `1e-4`, `1e-3`,
  `1e-2`, and `0.1`.
- M1 `motion_conditioned_ridge_v1`: the ADR 0003 gaze, head-pose, normalized
  face geometry, face-scale, and interaction schema; ridge alpha from `1e-3`,
  `1e-2`, `0.1`, `1`, and `10`.
- Outer validation holds out one complete motion block. For each outer fold,
  degree and alpha are chosen only by group validation among the other four
  blocks. Statistics for M1 are fitted only on those outer-training rows.
- The primary score is the unweighted mean of the five outer-block mean pixel
  errors. Median and p95 are descriptive only.
- M1 is adopted only if it reduces the primary M0 score by both at least 5 px
  and at least 5%. Equality passes; ties and non-finite results retain M0.
- After the decision, the selected model is fitted on all five blocks. No
  candidate, feature, threshold, or split may be changed after observing this
  run.

## Frozen validity and resource gates

1. The coverage audit must remain `ready` on the exact manifest digest above.
2. Every outer fold must contain one whole block and no row from that block may
   enter inner hyperparameter selection.
3. Model parameters and inference tensors must stay on CPU. The runner records
   `nvidia-smi` before, during, and after execution and fails if its process is
   observed as a GPU compute process.
4. Hugging Face and Transformers are placed in offline mode; cached weights
   are required.
5. No reading article, question, answer, or participant identity is used as an
   input or split variable.
6. This capture is candidate evidence. A separately identified capture remains
   required before any cross-session claim or encoder fine-tuning.

## Commands

```powershell
# Leakage/negative-control contract (pure NumPy)
.\.venv\Scripts\python.exe -X utf8 -m unittest -v `
  scripts.test_gaze_motion_experiment

# Real capture; the local session ID is deliberately not committed
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_gaze_motion_experiment `
  --session-id <local-session-id> `
  --output-model-name motion_run_001_nested_cpu `
  --json-output docs\experiments\results\2026-08-05-motion-run-001.json
```

## Result

Status: `passed`. The machine-readable result is stored at
[`results/2026-08-05-motion-run-001.json`](results/2026-08-05-motion-run-001.json).

| Primary metric | M0 gaze polynomial | M1 motion conditioned |
| --- | ---: | ---: |
| Outer-block macro mean | 330.22 px | **228.91 px** |
| All held-out samples median | 327.15 px | **202.06 px** |
| All held-out samples p95 | 614.83 px | **499.80 px** |

M1 improved the frozen primary score by 101.31 px (30.68%). The required
improvement was 16.51 px, which is the stricter 5% margin for this M0 score and
also exceeds the fixed 5 px floor. The promotion gate therefore passed and the
locally fitted artifact uses `motion_conditioned_ridge_v1`. Its final alpha is
10.0; no alternative was tried after this result.

| Outer block | M0 mean | M1 mean | Improvement |
| --- | ---: | ---: | ---: |
| far | 395.14 px | 390.15 px | 4.98 px |
| left | 342.77 px | 225.74 px | 117.03 px |
| near | 326.02 px | 205.69 px | 120.33 px |
| neutral | 250.92 px | 175.03 px | 75.89 px |
| right | 336.25 px | 147.93 px | 188.32 px |

The run used Python 3.11.9, NumPy 2.4.6, Torch 2.13.0+cu130, eight CPU
threads, cached offline UniGaze weights, and 15.26 seconds of measured runner
time. All 65 normalized images had distinct SHA-256 hashes. Twenty GPU polls
observed 0 MiB, 0% utilization, and no experiment process; Torch CUDA remained
uninitialized. The selected local model is intentionally ignored by Git. The
committed result contains only aggregate metrics and content hashes.

The final isolated quality gate passed 59/59 tests with zero failures, errors,
or skips. It blocked network and subprocess access inside the worker, cleared
provider credentials, detected no artifact changes, did not import Torch, and
recorded 0 MiB / 0% GPU before and after. Its machine-readable record is
[`results/2026-08-05-motion-run-001-quality-gate.json`](results/2026-08-05-motion-run-001-quality-gate.json).

## Decision and limitation

Adopt M1 for this local calibrated model because it cleared the rule fixed
before image/model access. Do not claim general gaze accuracy: absolute error
remains high, and the far block improved by only 4.98 px. This is strong evidence
that head pose and face geometry help within this capture, but only one capture
and one operational participant label were evaluated. A separately collected
run must confirm the decision before cross-session or cross-person claims, and
the far-distance failure is the highest-priority condition for that check.
