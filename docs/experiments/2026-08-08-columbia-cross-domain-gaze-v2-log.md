# Columbia Cross-Domain Gaze v2 — Experiment Log

## Objective

Complete the frozen v1 external evaluation without weakening its model or
effectiveness rules. V2 inherits the entire v1 protocol and changes only the
source-annotation completeness contract: the official README documents 5,865
eye-corner rows and 15 missing rows, so those exact 15 rows receive one fixed
MediaPipe landmark attempt before the inherited affine eye crop.

## Relationship to v1

V1 stopped before extraction, image decoding, model import, checkpoint load,
prediction, or GPU use because its zero-missing-annotation gate contradicted
the official archive README. The machine-readable stop is preserved at
`results/2026-08-08-columbia-cross-domain-gaze-v1-source-preflight.json`.

This v2 correction is based only on source metadata. No Columbia image or model
outcome has been observed. Consequently, v2 can still serve as the one-time
model confirmation, provided it is committed before extraction or decoding.

## Frozen delta

- Inherit v1 protocol commit `26003f6c5904e1cebe5b678ce26deaa0d4cab272`
  and SHA-256
  `1dda7cf10d164e5cdcac0d870c6bfe8e970b9223192c58b026a32ba5cbf84e78`.
- Require exactly 5,865 official annotation rows and exactly 15 missing rows.
- Only the missing rows may use the fallback.
- Resize once to width 1280 and run the pinned MediaPipe FaceLandmarker once.
- Use fixed eye-corner landmark indices already present in production code.
- Map fallback coordinates back to the original 5184x3456 frame and then use
  the unchanged v1 affine crop.
- No retry, alternate resolution, detector, manual label, row dropping,
  imputation, or single-eye substitution is allowed.
- Report official and fallback strata separately; the strata cannot select or
  reweight a model.
- Every model, label sign, crop endpoint, training epoch, seed, metric,
  effectiveness rule, hardware limit, and production boundary remains exactly
  as frozen in v1.

## Pre-execution record

### 2026-08-08 — v2 protocol freeze

The v2 delta was written after reading the official source README but before
archive extraction, Columbia image decoding, model import, checkpoint loading,
prediction, or GPU execution. Protocol commit and SHA-256 are recorded after
this file is committed.

### 2026-08-08 - formal source audit passed

The staged runner verified both downloaded archive hashes, reconstructed the
complete filename-encoded design, and decoded every source JPEG on CPU before
any model runtime was imported.

- Subjects: `56`
- Images: `5,880` (`105` for every subject)
- Subject x head-pose x vertical x horizontal grid failures: `0`
- Duplicate image identities: `0`
- Official eye-corner rows: `5,865`
- Officially missing rows: `15`
- Extra annotation rows: `0`
- Corrupt images: `0`
- Dimension mismatches: `0`
- Image-identity SHA-256:
  `7389732b4a1e27df5882875b2294dff8f93d2de89eceaf4b33351d781e0d6bb0`
- Missing-identity SHA-256:
  `66a55ddb5ce4b478afa1d15da0db96efcebe8ea5c37496f0ce05559ba6098ce8`
- Torch imported during audit: `false`
- GPU utilization before and after the audit: `0%`

The committed aggregate audit is
`results/2026-08-08-columbia-cross-domain-gaze-v2-data-audit.json`. Raw images,
annotations, prepared tensors, checkpoints, and row-level prediction evidence
remain local and excluded from version control.

### Pre-model implementation safeguards

The runner is staged as audit-only, prepare-only, and formal model execution.
Twelve focused CPU tests freeze filename signs, annotation bounds, affine crop
geometry, anatomical-right-eye mirroring, vector fusion, participant-level
metrics, protocol inheritance, aggregate failure codes, and the ignored
prediction-evidence schema. A separate CPU audit can recompute every committed
aggregate and effectiveness decision from that local evidence after the model
run. No Columbia outcome may be used to change these contracts.

## Formal run record

### 2026-08-08 - frozen model execution completed

The one-time confirmation run completed without changing the committed runner.
The formal implementation SHA-256 was
`94e9677b255090b8c1f8fb67ca729dd48229362fe690ae94d88f93cab5b96b9f`.

- Production preprocessing: `5,880/5,880` (`100%` coverage)
- Candidate completeness: `5,880/5,880`; fallback `15/15`
- Zero-gaze macro subject angular error: `12.0455` degrees
- EyePoseTinyCNN-v1 macro subject angular error: `14.2841` degrees
- Candidate minus zero: `+2.2386` degrees; 95% participant-bootstrap CI
  `[+2.0703, +2.4115]`; subjects improved `0/56`
- UniGaze macro subject angular error: `17.6857` degrees
- Production minus zero: `+5.6402` degrees; 95% participant-bootstrap CI
  `[+5.4423, +5.8572]`; subjects improved `0/56`
- Candidate confirmed: `false`
- Production external baseline supported: `false`
- Execution integrity: `passed`

The three candidate fits used the frozen seeds and exactly 11 epochs. Their
state-dict SHA-256 values are:

- `20260808`:
  `4ac9305ec13984aabba1c64ae015a92129c635a92b2b382f6d7291d56a2dd3f4`
- `20260809`:
  `f3ecb93ae771313a282bf6f950eb28b49a44ade85bad571c0659d171941eed1c`
- `20260810`:
  `eba22e41bb8b1046971b5d68f254e6f1108559f3a7f37f7aecbad63b3b6d23e3`

The UniGaze state-dict SHA-256 was
`6a99da8fed8c4fcd0fda85bd14137c68e7c34969a7698106cd6ae157270eb0e6`.
Model execution used `0.0866` hours, reached `0.8281` GiB peak process VRAM
and `63 C`, and made zero network attempts. Production source SHA-256 remained
`ab6ecdd4db6c7ebfbf1a55c51cc123ba487dc4a04f8c37ef4574ef5d60229f1b`
before and after the run.

### Independent aggregate audit

A fresh CPU-only process loaded the ignored prediction evidence, verified its
SHA-256 (`602fee2404ebfa26342704ac97d66288ed119147e5b0fc5481a0594695708a99`),
and exactly recomputed all metrics, bootstrap intervals, and effectiveness
decisions without importing Torch. Every audit check passed.

The committed primary result SHA-256 is
`05b3540ab5c6d0f3fad6d5de7a28e0ab9df46e393a40606a0fa45c7d7a33aa15`.
The result is negative and final under this protocol; Columbia-driven tuning or
a sign-adjusted replacement score is not allowed.

### Post-hoc hypothesis, not a replacement result

Aggregate component regressions found that production pitch retained a slope
of `+0.795` with Pearson `r=+0.886`, while production yaw had a slope of
`-0.971` with `r=-0.959`. This is consistent with a horizontal coordinate
interface mismatch under the frozen Columbia label mapping. The candidate was
more broadly compressed and biased (pitch slope `+0.174`, yaw slope `-0.247`).

These diagnostics were not pre-registered, do not change the failed primary
decisions, and may not select a mapping for a rerun presented as independent.
They motivate a new, separately frozen coordinate-contract check on fresh
LexiGaze left/center/right captures before any model or hardware promotion.
