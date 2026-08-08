# Independent Capture Readiness v1 — Engineering Log

## Objective

Prepare an outcome-blind, CPU-only contract for future independent gaze and
fusion validation. The contract must keep participants, sessions, physical
capture runs, article families, and conditionally devices separated across
development, validation, and confirmation roles. Optional laptop-plus-phone
capture must be treated as paired sensor data rather than extra independent
examples.

This branch does not collect real participant data, export a trainable dataset,
train a model, change the canonical pilot from `dry_run_only`, or change a
production default.

## 2026-08-08 — local gap audit

Existing engineering already provides versioned consent, pseudonymous study
IDs, invitation and withdrawal controls, participant-only routes, raw-frame
purging, capture-run provenance, motion-block coverage, and conservative
grouping of legacy direct/video artifacts. The remaining dataset-stage gap is a
machine-checkable plan that binds the following units before outcomes exist:

- participant assignment slot;
- analysis role;
- repeated session slots;
- device and sensor-source slots;
- physical capture-run slots;
- article and near-duplicate family slots; and
- multi-view synchronization, relative-camera calibration, and missing-view
  policy when such a claim is enabled.

The current participant pilot remains intentionally locked by ethics,
governance, external-anchor, contact, retention, encryption, and rehearsal
requirements that software cannot self-approve.

## 2026-08-08 — public-data screen

- Columbia Gaze is a manageable independent candidate: 5,880 images from 56
  people, a direct approximately 2.38 GB non-commercial download, five
  horizontal head poses, and 21 gaze directions. Its high-resolution lab setup
  and different label/preprocessing convention mean it cannot be evaluated
  under the MPIIGaze code path without a separately frozen adapter protocol.
- ETH-XGaze provides broader pose diversity but its smallest face-patch package
  is approximately 130 GB and access requires an institutional request and
  acceptance of additional terms.
- GazeCapture contains approximately 2.5 million mobile-device frames and
  requires registration.
- EYEDIAP uses an access process and cannot be assumed immediately available.

No dataset was downloaded and no model outcome was inspected during this
screen. The next external-data experiment, if pursued, will use its own branch
and protocol.

## Frozen implementation gates

The complete v1 contract and pass/fail rules are frozen in
`protocols/2026-08-08-independent-capture-readiness-v1.json`. Any change to the
schema or gates after implementation begins requires a documented deviation or
a separately frozen v2.

Protocol SHA-256:
`2e1a88b4de8fd6af1c11dbeb5b1eb5b3213c534ecb2142086adc24762fbb3c69`.

## Pending

1. Implement the strict plan schema and aggregate-only audit.
2. Add valid, leakage, unexpected-field, device-holdout, and multi-view tests.
3. Add a synthetic template and researcher documentation.
4. Run focused and repository-wide CPU-only gates.
5. Record whether v1 passed without weakening its checks.
