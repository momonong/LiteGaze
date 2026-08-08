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

## 2026-08-08 — implementation result

V1 passed every frozen engineering gate without changing the protocol. The new
contract provides three intentionally different targets:

- `template` validates schema and isolation but explicitly warns that it is not
  collection authorization;
- `collection` requires a plan frozen before collection; and
- `evidence` counts only bound, non-withdrawn participant/session units and
  requires active device, article, and optional multi-view calibration assets
  to be hashed and authorized.

The exact allow-list rejects unknown or observed-outcome fields. Participant,
session, device, article-family, capture-run, and sensor-source references are
checked for role consistency. Device generalization rejects shared physical
devices. Every multi-view run requires distinct source roles, a timestamp
tolerance, clock strategy, relative-camera calibration slot, calibration hash
at evidence time, and a frozen missing-view policy.

Bindings are isolated digests rather than participant/session IDs. Unbound
slots do not count, while withdrawal requires clearing the digest and
propagating withdrawal to repeated sessions. Audit results contain aggregate
counts and codes only.

## 2026-08-08 — verification result

- Focused suite: `16/16` tests passed.
- Schema type fuzz: `470` mutations, `0` uncaught exceptions. The first pass
  found eight unhashable-role crashes; implementation fixes removed all eight
  without weakening a gate.
- Template target: `template_valid`; its collection target correctly returned
  `not_ready` with `PLAN_NOT_FROZEN`.
- Repository offline gate: `140/140` tests across 19 explicit targets, zero
  failures/errors/skips.
- Safeguards: zero network attempts, provider credentials cleared, process and
  network probes blocked, Torch not imported, `CUDA_VISIBLE_DEVICES=-1`, and no
  tracked artifact mutation.
- Participant audit: `dry_run_ready=true`, `pilot_ready=false`, with all 20
  external/governance activation requirements still visible.
- Ruff, Python byte-compilation, JSON parsing, and `git diff --check` passed.
- Production gaze SHA-256 remained
  `ab6ecdd4db6c7ebfbf1a55c51cc123ba487dc4a04f8c37ef4574ef5d60229f1b`.

Result artifact:
`results/2026-08-08-independent-capture-readiness-v1.json`.

## Decision

Status: `passed` for engineering readiness only. Real participant collection,
dataset export, model effectiveness, and production promotion remain false.
The example keeps its synthetic three-slot minimums and `template_only` status,
so it cannot be mistaken for the final sample-size or split decision.

The next independent model experiment may evaluate Columbia Gaze, but it must
use a new branch and freeze eye-crop, label-convention, coordinate-mapping,
baseline, and resource rules before inspecting a LexiGaze model outcome.
