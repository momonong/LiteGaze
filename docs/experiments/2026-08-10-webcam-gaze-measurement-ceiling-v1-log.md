# Webcam gaze measurement ceiling v1 - implementation log

- Date: 2026-08-10
- Branch: `experiment/webcam-gaze-measurement-ceiling-v1`
- Scope: single-webcam measurement integrity, participant-safe collection,
  geometry/text separation, and a deterministic existing-data audit
- Production promotion: no
- GPU training: no

## Objective and frozen boundaries

The work asks what the current single-webcam pipeline can actually measure
before adding a phone, a second camera, a larger gaze encoder, or a text and
cognitive prior. The following boundaries were frozen before interpreting the
existing session:

- fixed-target evidence must be separated from natural-reading inference;
- calibration and evaluation targets must not overlap within 0.1 viewport
  fraction (0.2 in signed `[-1, 1]` coordinates);
- a complete physical capture group stays together;
- cursor position, nearest-word output, text difficulty, cognitive profile,
  and reading outcomes are not eye-tracking ground truth;
- sensor abstention cannot be rescued by text or cognitive attraction;
- negative results and failed integrity checks remain in the report;
- routine audit and contract validation are CPU-only and offline.

The machine-readable frozen protocol is
[`protocols/2026-08-10-webcam-gaze-measurement-ceiling-v1.json`](protocols/2026-08-10-webcam-gaze-measurement-ceiling-v1.json).

## Read-only findings before implementation

1. Calibration manifest rows recorded 1280x720 at 30 fps while the participant
   system check recorded 640x480 and an estimated `15_23` fps band. The aspect
   ratios differ by `0.444444`, above the frozen `0.02` cross-phase maximum;
   absolute resolution and frame-rate differences remain diagnostic warnings.
2. The linked calibration session said `study-direct-frame`; all manifest rows
   said `direct-frame`. Capture provenance was client-influenced.
3. The five participant validation regions were all below the frozen distance
   from at least one calibration target, so they were not target-held-out.
4. When M0 was retained, the model artifact reported its inner/group
   hyperparameter CV score (`199.18 px`) as the selected held-out score; the
   nested-outer M0 macro was `204.58 px`.
5. Live word mapping divided geometry distance by a cognitive difficulty
   attraction. That made sensor coverage and candidate selection circular.
6. Natural-reading telemetry contained a derived nearest-word index but no
   independent line or word truth and no predictive uncertainty reference.

## Implemented decisions

### Capture and provenance

- Calibration, validation, and reading now share a 1280x720 at 30 fps camera
  intent and a 640-pixel transport width that preserves the camera's actual
  aspect ratio. JPEG quality is 0.8.
- A versioned capture contract records source and transport geometry, resize,
  mirror, facing mode, frame rate, and encoding. Decoded JPEG dimensions must
  match the declaration.
- Aspect, resize, mirror, or facing mismatch abstains with HTTP 409 for a model
  that carries a capture contract. Resolution and frame-rate changes remain
  visible warnings when aspect and geometry semantics are preserved.
- Calibration sample run/source provenance is overwritten from the
  server-created session rather than trusted from each browser request.

### Independent participant geometry gate

- The frozen general-collection v1 JSON and its canonical hash are unchanged,
  preserving already assigned Visit 2 invites.
- A separate additive participant gaze measurement contract records its own
  ID, version, hash, capture rules, and five server-authoritative held-out
  targets selected from the frozen 4x4 grid.
- Training stores every successful screen target used by the fitted personal
  calibration stages. Missing legacy or inherited target provenance fails
  closed.
- The server verifies target ID, signed coordinates, rounded pixel position,
  viewport, per-target count, measurement-contract hash, and distance from the
  actual fitted target set. Client changes or omissions cannot create a pass.
- Start and end validation both require a compatible capture contract and
  passed target independence before any gaze band is eligible. Behavioral word
  review can continue after any gaze downgrade.
- Calibration target labels used for participant model fitting are validated
  and overwritten from the server-frozen point/block assignment rather than
  trusted from browser coordinates.
- Each assessment persists the full canonical additive measurement contract;
  resume and validation use that frozen copy and hash rather than whatever
  contract file happens to be current later.
- The system-check viewport is frozen for the assessment. A resize or
  orientation change aborts validation instead of turning it into ordinary
  failed predictions; a change during reading is persisted as a gaze-integrity
  failure and the behavioral round may still be completed.
- Re-entering an active reading round without a versioned segment-resume
  contract no longer joins two telemetry segments or reports a misleading
  effective sampling rate. The raw segments remain auditable and gaze is
  downgraded to behavioral-only.

### Geometry, text, and cognitive separation

- Production sensor mapping now uses rectangle geometry only. A hit enters the
  measurement buffer only with explicit `geometry_only_v1` mode and
  `sensor_accepted=true`.
- Cognitive attraction is an explicit visualization-only opt-in and cannot
  change raw coordinates, sensor coverage, or abstention.
- A line-first occurrence decoder was added as a shadow-only library. It
  chooses a line, ranks unique occurrence IDs by layout geometry and reading
  order, and emits explicit sensor/line/token abstention reasons. It is covered
  by offline tests but is neither executed nor observed in the production
  mapping path.

### Metric and audit correctness

- Selected nested-outer validation error and hyperparameter CV error are now
  separate model/API fields for both M0 and M1 paths.
- The pure-standard-library audit binds participant session, calibration
  session metadata, all manifest rows, and the model by explicit paths and
  SHA-256 values. It never opens images, videos, Torch, or a network resource.
- It reports target-macro and sample-level spatial errors, directional axes,
  drift vectors, layout-relative resolution, coarse confusion, a start-only
  temporal correction, and a paired target-cluster bootstrap.
- Uncertainty calibration and natural-reading line/word accuracy remain
  explicitly `not_evaluable` rather than inferred from nearest-word telemetry.

### Private exporter gaze-provenance boundary

- The private general-collection exporter now emits schema version 2 and uses
  two explicit validation layers. The current runtime protocol/bank still pass
  the existing whole-design consistency gate; each session's gaze is then
  checked only against that session's frozen additive measurement-contract
  snapshot and hash. The current additive contract is never substituted for a
  missing or older session snapshot.
- Start and end validation samples, summary metrics, sample hashes, validation
  payload hashes, contract provenance, frozen assessment viewport, calibration
  capture compatibility, target independence, final eligibility flags, gaze
  integrity, and telemetry payload/viewport/session bindings are recomputed
  before export. A completed reading session with no telemetry fails gaze
  export closed. Validation errors, telemetry success/rate, head-pose and face
  ranges, drift, continuity, and the final gaze band are recomputed from the
  bound validation summaries, raw telemetry batches, and round timing; stored
  telemetry/final metrics must match, and exported metrics come from the
  recomputation rather than the stored quality object.
- Behavioral session, passage, and word-review rows remain exportable when
  gaze provenance is legacy, unavailable, or invalid. Such gaze is marked
  `excluded`/`unavailable`, recorded in `gaze_excluded_sessions.csv`, and is
  omitted from `gaze_telemetry.csv` and `validation_samples.csv`; it cannot be
  silently mixed with eligible gaze.
- Session rows and eligible gaze rows carry the frozen contract ID, version,
  SHA-256, eligibility, assessment viewport, and pair-comparison status. Visit
  1/Visit 2 gaze is pair-comparable only when both visits are individually
  eligible and have the same frozen contract SHA-256. A legacy sibling does not
  invalidate a sound visit, and two individually sound visits with different
  contract hashes remain usable separately but cannot be paired.
- These hashes detect provenance drift and inconsistent stored payloads; they
  do not turn the existing browser-roundtrip prediction values into
  tamper-resistant evidence. Formal promotion remains forbidden pending the
  server-issued single-use receipt contract described below.

## Existing-data result

The deterministic audit completed but its evidence status is
`failed_integrity_gate`:

- artifact bindings: passed;
- viewport contract: passed;
- cross-phase camera geometry: failed, 1280x720 calibration versus 640x480
  participant system check (`0.444444` aspect-ratio difference, maximum
  `0.02`); resolution and FPS differences are warnings, not hard gates;
- capture provenance: failed;
- target independence: failed, 5 of 5 evaluation targets below tolerance;
- historical model metric consistency: failed;
- start median/P90: `213.30 / 266.07 px`;
- end median/P90: `128.69 / 278.00 px`;
- end median resolution: `4.73` line gaps or `3.15` median word widths;
- coarse five-region accuracy: 100%, which does not establish line or word
  resolution;
- start-fitted translation changed end median by `+19.74 px` (worse) and end
  P90 by `-78.59 px` (better); target-cluster 95% interval
  `[-155.90, 144.69] px`, with 39.21% of resamples improving.

No correction, quality band, threshold, line/word claim, or production model
was promoted. The complete report is
[`2026-08-10-webcam-gaze-measurement-ceiling-v1.md`](2026-08-10-webcam-gaze-measurement-ceiling-v1.md).

## Verification and compute record

- Focused frontend behavior: three independent Node tests passed.
- Changed frontend syntax: Node `--check` passed.
- Pure standard-library/source contract lane: 30 tests passed.
- Dependency-isolated, CPU-only exporter provenance fixtures: eight tests
  passed (eligible export, behavioral-only legacy retention, validation hash
  tamper, missing telemetry, final-quality tamper, pair contract mismatch,
  ineligible-sibling isolation, and shared-worker namespace cleanup).
- Focused measurement-ceiling and frozen-protocol lane: 16 tests passed.
- Changed Python modules: `py_compile` passed.
- Repository patch hygiene: `git diff --check` passed.
- Deterministic current-data output was reproduced twice:
  - JSON SHA-256:
    `f05437b54d92c86574b9db79019e403f03429fd737f904ed1f578446ebbf84a5`
  - Markdown SHA-256:
    `e0f03c1eafd37cc54258e17af68097b8d50f8714a1cc5b8d95de42049947575f`
- Analysis/test commands set `CUDA_VISIBLE_DEVICES=-1`; no work in this
  change imported Torch or launched a model/GPU workload.
- The final GPU snapshot showed the shared RTX 5090 Laptop GPU at 0%
  utilization, 166/24463 MiB allocated, and 53 C, with no Python compute
  process listed. The ambient allocation was not attributed to this CPU-only
  work.

The full Flask/NumPy/OpenCV offline lane was authored and extended but could
not be launched in this sandbox. The isolated worktree has no `.venv`; the
main checkout's launcher points to an external Python 3.11 executable that the
sandbox cannot execute, while system Python 3.14 lacks those project
dependencies. This is recorded as unverified, not as a passing result.

One integrity boundary remains intentionally open: validation predictions and
success flags make a browser-to-server round trip before metrics are stored.
Authentication, fixed targets, model binding, capture checks, and rehearsal
status limit accidental misuse, but this is not tamper-resistant evidence.
Formal participant promotion remains forbidden until a versioned server-side
receipt flow issues single-use predictions bound to session, model, phase,
target, and capture-contract result, then reconstructs metrics without trusting
client-posted prediction values.

## Next evidence step

Run a fresh participant-compatible calibration and the new held-out five-point
start/end validation without resizing the browser after system check. The
result can assign the frozen rehearsal descriptive candidate band and decide
whether gaze remains eligible for exploratory analysis or the session should
remain behavioral-only. Fixed targets and layout ratios still do not establish
natural-reading line or word accuracy. The larger 193-sample motion-ceiling
protocol remains the next dedicated self-development capture when the user is
available.

Before opening formal participant collection, implement the server-issued
prediction receipt contract described above and rerun the complete project
offline lane in the repaired Python 3.11 environment.

Only after a matched-contract run isolates a repeatable posture, distance, or
occlusion failure should a synchronized phone or second-camera v2 be built.
