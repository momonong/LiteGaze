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
