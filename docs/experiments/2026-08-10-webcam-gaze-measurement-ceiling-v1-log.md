# Webcam gaze measurement ceiling v1 - implementation log

- Date: 2026-08-10
- Branch: `experiment/webcam-gaze-measurement-ceiling-v1`
- Scope: single-webcam measurement integrity, participant-safe collection,
  receipt-bound fixed-target evidence, training-only uncertainty ranking,
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

### Server prediction receipts and model isolation

- Each start/end fixed-target prediction now receives one opaque, high-entropy
  server receipt. Only its SHA-256 and canonical issued-record hash are stored;
  the raw token remains browser-only until one 15-receipt bundle is consumed.
- The server binds every receipt to the session authorization fingerprint,
  assessment, visit/capture-specific model and artifact SHA-256, calibration
  capture, frozen viewport and measurement contract, phase, ordinal, target,
  capture contract, and server prediction. Cross-session, cross-phase, reorder,
  replay, target, model, or artifact mismatch fails closed. Only an exact
  already-consumed bundle is an idempotent retry.
- The validation POST contains only phase and receipt tokens. The server
  reconstructs coordinates, errors, no-face failures, and uncertainty evidence;
  legacy client prediction fields are ignored and explicitly downgrade gaze to
  behavioral-only.
- Current participant clients send the study secret only in the Bearer header,
  not again in JSON bodies. Both prediction aliases and the validation route
  reject non-object JSON with a stable 400 response.
- Personalized artifacts use
  `{participant}_{mode}_{visitN|unpaired}_{sha256(gaze_session_id)[:12]}_general_v1`.
  Visit 2 therefore cannot overwrite Visit 1, and failed cleanup cannot delete
  its sibling. The route also rejects a training response whose reported model
  name differs from the server-computed name.

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
- A separate descriptive repeatability proxy ranks whole target clusters using
  only the RMS dispersion of repeated start predictions, then evaluates the
  frozen 20/40/60/80/100% coverage grid only against end target error. It does
  not search a threshold, change a quality band, or authorize per-sample
  abstention.
- The exact 65-row motion-training path now fails closed unless it has all 13
  targets in each of five complete motion blocks. Each outer block remains
  untouched while its inner training partition selects the model family and
  hyperparameters, fits the pipeline, and fits label-free OOD, leverage, and
  whole-pipeline block-jackknife disagreement state.
- The frozen v2 score is the maximum of training-ECDF component percentiles.
  It persists every outer-fold prediction/residual and partition proof, but the
  score itself accepts no target, residual, text, layout, or cognitive input.
  Coverage-risk uses only the predeclared `[1.0, 0.8, 0.6, 0.4, 0.2]` grid and
  treats the five motion blocks as the effective independent clusters.
- The runtime output is `scored_no_threshold`; threshold, confidence label,
  quality band, error calibrator, conformal guarantee, and production policy
  remain explicitly unselected. The frozen definition SHA-256 is
  `75a24c436e9a89024462268812ecc9be149a1958b3911e5cd71c3974b235a180`;
  the uncertainty protocol payload SHA-256 is
  `a6f622277291ac6484c69606da7197ebfde625f4e1a3e60686ca7241380b42c3`.
- Receipts and the measurement audit validate the same pure-standard-library
  observation schema. The legacy artifact audited below has neither a v2
  bundle nor receipt-bound observations, so its held-out uncertainty curve is
  correctly `not_evaluable`; no OOF evidence is retrofitted from aggregate
  folds. Natural-reading line/word accuracy also remains `not_evaluable`.

### Private exporter gaze-provenance boundary

- The private general-collection exporter now emits schema version 2 and uses
  two explicit validation layers. The current runtime protocol/bank still pass
  the existing whole-design consistency gate; each session's gaze is then
  checked only against that session's frozen additive measurement-contract
  snapshot and hash. The current additive contract is never substituted for a
  missing or older session snapshot.
- Fixed-target start/end validation is reconstructed only from the private
  server-issued receipt registry. The exporter verifies every record and
  bundle hash, consumed phase, session/authorization/assessment/model/artifact/
  capture/viewport/measurement bindings, server-frozen target order, prediction
  outcome, summary sample reconstruction, and validation-payload hash. Raw
  receipt tokens and authorization fingerprints are never exported.
- Receipt uncertainty is revalidated through the shared pure-standard-library
  schema. Ordered observations and their summary hashes are reconstructed from
  `issued.prediction.uncertainty` and bound into the validation payload. Scored
  rows retain the frozen definition SHA, score, component values/percentiles,
  normalized and pixel covariance, and explicit `not_selected`/`null`
  abstention. Legacy unavailable uncertainty leaves fixed-target geometry
  eligible but is marked `not_evaluable`; it cannot become uncertainty evidence
  and no threshold is selected from this export. An explicit no-face receipt is
  instead reported as capture non-coverage: risk inputs remain conditional on
  successful predictions when every success is scored, with no-face count and
  attempted-capture coverage exported separately.
- Natural-reading telemetry remains client-roundtrip evidence with no per-sample
  receipt. `gaze_telemetry.csv` is therefore eligible-only and intentionally
  empty, while diagnostic rows are isolated in
  `reading_telemetry_unverified.csv` with
  `client_roundtrip_unverified`, `prediction_receipt_bound=false`, and formal
  eligibility false. Missing or tampered reading telemetry cannot poison an
  otherwise valid fixed-target validation, but it also cannot become eligible
  gaze evidence.
- Behavioral session, passage, and word-review rows remain exportable when gaze
  provenance is legacy, unavailable, or invalid. Receipt-invalid validation is
  omitted from `validation_samples.csv`; reading telemetry is never mixed into
  that receipt-verified table.
- Session rows distinguish `validation_gaze_export_eligible` from
  `reading_gaze_export_eligible=false`. Visit 1/Visit 2 fixed-target validation
  is pair-comparable only when both visits are individually eligible and use the
  same frozen contract SHA-256. A legacy sibling does not invalidate a sound
  visit, and two sound but different-contract visits remain usable separately.
- The canonical hashes detect provenance drift but are not keyed signatures.
  Formal promotion remains forbidden for this self-development rehearsal, and
  natural-reading gaze remains blocked pending its separate receipt design.
- A read-only current-data dry run found one legacy completed session. It
  retained 6 passage rows, 48 word-review rows, and 708 layout rows; exported 0
  eligible fixed-target validation rows and 0 eligible reading-gaze rows; and
  isolated all 649 reading samples in the unverified diagnostic table. Those
  649 rows are diagnostic-only and are not formal evidence. Runtime uncertainty
  was `not_evaluable`. The dry run parsed JSON/CSV metadata only; it did not
  decode image/video data or use GPU compute.

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
  `[-155.90, 144.69] px`, with 38.98% of resamples improving.
- the start-only target repeatability proxy had Spearman `-0.10` against end
  target mean error. Its lowest-proxy 20% coverage retained `center` and had
  `282.22 px` end target-macro mean error, versus `154.14 px` at 100% coverage
  (`+128.08 px`). With only five target clusters, this is a descriptive
  negative result: `available_start_repeatability_proxy_does_not_rank_end_risk`.

No correction, abstention threshold, quality band, line/word claim, or
production model was promoted. The separate experimental v2 definition is
frozen as `scored_no_threshold`, but this legacy artifact's per-sample
predictive uncertainty remains `not_evaluable`. The complete report is
[`2026-08-10-webcam-gaze-measurement-ceiling-v1.md`](2026-08-10-webcam-gaze-measurement-ceiling-v1.md).

## Capture-readiness preflight and follow-up fixes

After the receipt/uncertainty implementation milestone, a local browser
preflight was used to reduce avoidable failure modes before asking the user for
the remaining physical webcam capture. This preflight did not create a new
measurement result and did not close the matched-contract evidence gate.

The preflight exposed four concrete negative findings:

1. Windows allowed two processes to remain `LISTENING` on the same
   `127.0.0.1:8098`: a service left from the previous day and the current-branch
   service. Requests could therefore reach stale code and reported the old data
   location and raw-frame settings. The researcher must now confirm exactly one
   listener with `netstat -ano | Select-String ':8098\s+.*LISTENING'`; any stale
   exact PID is inspected and stopped explicitly, never by broadly terminating
   Python processes.
2. The participant-safe `GET /api/gaze/health` request initially returned 403
   because the public study allowlist omitted that route. The route is now
   public on the constrained participant surface and returns only `{"ok":true}`
   without a researcher key. The richer backend response remains researcher
   only. `GET /api/study/protocol` is the separate check for activation,
   storage, retention, and self-only scope configuration.
3. The readiness audit still assumed that every optional video scope meant
   full video collection. That stale assumption incorrectly rejected the
   already bounded, explicit, self-only reading-video development scope. The
   audit now validates that exception's exact consent/category/comprehension
   boundaries. It still fails closed for formal pilot readiness whenever any
   optional video scope is present; this change does not authorize external
   participants or formal promotion.
4. The existing Visit 2 plaintext invite could not be recovered. This is an
   intentional one-time-secret property: the registry stores only its hash.
   Starting the server again with `--create-invite-pairs 1` would create a new
   participant/pair assignment rather than recover the existing Visit 2. A
   dedicated command now rotates exactly one lost, unused invite in place,
   invalidates the previous code, preserves pair/form/schedule assignment, and
   prints the replacement once. Used invites remain non-rotatable, and the
   18–72 hour Visit 2 window remains enforced.

Adversarial review then found two additional fail-open risks before delivery.
First, invitation consumption and rotation used only a process-local lock, so
two processes could overwrite each other's registry update. All invitation
registry read-modify-write operations now share an OS-level file lock; both
consume-first and rotate-first process orderings are covered by a standalone
two-process regression. Second, the formal-video boundary existed only in the
readiness CLI while runtime invite/enrollment gates trusted a separate
activation result. The exact optional-scope, full data-category, and full
comprehension-check contracts now live in one core helper used by both runtime
activation and the audit. A self-only video scope or any unreviewed category
therefore prevents formal invite creation and enrollment, even when every
other pilot setting is present.

The participant flow was also tightened around these failures:

- server restarts with an existing registry use `--create-invite-pairs 0`;
- the study page can deliberately clear only the current tab's remembered
  context before another invite/Visit, after warning the researcher to preserve
  the withdrawal code and consent receipt; this action does not withdraw,
  delete, or modify server data;
- calibration no-face and motion/distance quality failures now produce
  actionable guidance while preserving the fail-closed server quality gate;
- low-quality or provenance-invalid gaze messaging distinguishes continuing
  behavioral reading from stopping before reading when usable gaze is required;
- participant access tokens remain in authorization headers and are no longer
  duplicated into browser JSON request bodies;
- the runbook now freezes the external Python 3.11 interpreter, project
  site-packages path, isolated code root, main-repository data root, CPU-only
  `CUDA_VISIBLE_DEVICES=-1`, and offline Hugging Face/Transformers settings.

The remaining step is still a user-assisted physical capture using a single
current-branch listener, stable viewport/camera geometry, all five motion
blocks, and receipt-bound start/end validation. Until that fresh capture exists,
the readiness work is an engineering improvement only: it cannot replace the
legacy `failed_integrity_gate` result or support a line/word accuracy claim.

The final Visit-handoff audit found and closed two additional paired-session
contract gaps before physical capture. The 18–72 hour server gate had been
measured from invite consumption even though the runbook defined it from Visit
1 completion; it now requires exactly one timezone-aware
`general_collection_completed` event and uses that timestamp. The frozen
`same_device_class_and_browser_family` policy had also been descriptive only.
Visit 2 system check now fails closed on either coarse-field mismatch, and the
private exporter independently recomputes the same policy before marking a pair
comparable. Viewport, DPR bucket, camera resolution, and FPS-band differences
remain explicit diagnostics rather than an unreviewed expansion of the frozen
policy.

A standard-library-only `preflight_general_collection_visit` CLI now checks the
dedicated branch/expected commit, capture-critical working files, post-source
server restart, stable single loopback listener, participant-safe HTTP surface,
frozen registry/session digests, completion-anchored window, unused Visit 2,
Visit 1 device reference, and target-linked calibration-image purge. It never
accepts, reads, or prints plaintext invites and never writes study data. Raw
directories belonging to unrelated legacy sessions are warning-only and are
never deleted. Browser-field prefill remains a volatile manual convenience, not
evidence that the researcher still possesses the one-time plaintext code.

## Verification and compute record

- Focused frontend behavior: five independent Node tests passed.
- Changed frontend syntax: Node `--check` passed.
- Training-only uncertainty fixtures: 23 tests passed.
- Single-use prediction receipt fixtures: 9 tests passed.
- Pure-standard-library, CPU-only exporter provenance fixtures: 19 tests passed.
  They cover receipt/bundle/payload hashes; consumed-phase replay; independently
  rehashed session, authorization, assessment, model, artifact, capture,
  viewport, measurement, target-order, and outcome tampering; capture-resolution
  warning semantics; scored and legacy-unavailable uncertainty; invalid
  uncertainty; conditional no-face capture coverage; missing, malformed, and
  tampered client telemetry; pair isolation; and shared-worker namespace
  cleanup.
- Focused measurement-ceiling fixtures: 21 tests passed; the frozen-protocol
  fixtures added 5, for 26 combined tests.
- Participant calibration route fixtures: 5 tests passed, including unique
  Visit/capture names and fail-closed training response binding.
- Participant/general/frontend and cross-process focused lane: 65 tests passed.
  The two true child-process invitation-lock tests are intentionally run
  outside the offline gate, whose safeguard blocks child process creation.
- Paired Visit handoff focused lane: 20 tests passed across both encrypted and
  self-development unencrypted stores, exporter pair-policy recomputation, and
  the read-only preflight. The preflight's 6-test subset also passed under
  Python `-S`, confirming that it has no installed-package dependency.
- Pre-surface offline quality-gate checkpoint: 319 tests passed with 0 failures, 0 errors,
  0 skips, and 0 unexpected successes. Worker time was 23.715 seconds;
  supervisor time was 24.025 seconds and measured wall time was 24.306 seconds.
  `artifact_changes=[]`, credentials were
  cleared, network and child-process probes were blocked, Torch was not
  imported, and `CUDA_VISIBLE_DEVICES=-1` remained in force.
- The persisted quality-gate JSON SHA-256 is
  `ad6d6303f5a5c5b5abbb9c7589498cfe574a614fe9171a9849e46f0f39475e1e`.
- Changed Python modules: `py_compile` passed.
- Repository patch hygiene: `git diff --check` passed.
- Deterministic current-data output was reproduced twice again after the final
  receipt and five-point preflight hardening:
  - JSON SHA-256:
    `24202d8a00d6382fa97d2e95b350525993e357596c8d49921ffdf7944e51c7f4`
  - Markdown SHA-256:
    `d038ffb4992b0180a81a35603c1fff85d483e9dafe568ea97ae1759a2bbe4bde`
- Analysis/test commands set `CUDA_VISIBLE_DEVICES=-1`; no work in this
  change imported Torch or launched a model/GPU workload.
- External GPU snapshots immediately before/after that pre-surface checkpoint
  showed the
  shared RTX 5090 Laptop GPU at 0% utilization and 166/24463 MiB allocated in
  both cases; temperature moved from 52 C to 53 C. The unchanged ambient
  allocation was not attributed to this CPU-only work.
- The persisted machine-readable gate result is
  [`results/2026-08-10-webcam-gaze-measurement-ceiling-v1-quality-gate.json`](results/2026-08-10-webcam-gaze-measurement-ceiling-v1-quality-gate.json);
  its verified SHA-256 is
  `ad6d6303f5a5c5b5abbb9c7589498cfe574a614fe9171a9849e46f0f39475e1e`,
  as recorded above. The removed
  `3ecf59a5...` value was stale and did not match the delivered file.
- The frozen general-collection payload remained unchanged at
  `7c4b25bb306b68bb2a2ee5f34217a67aace0de6778fb3f1ed9b462741a0a26b9`,
  so existing Visit assignments were not invalidated.

## Dedicated 193-row acquisition implementation checkpoint

The previously design-only 193-row protocol now has a dedicated, no-invite,
loopback-only acquisition implementation on `127.0.0.1:8099`. It remains
separate from participant Visit collection on port 8098 and never writes to the
participant-study store. The server owns all 65 calibration and 128 untouched
evaluation rows, target order, model selector, viewport binding, and timing
metadata. Text, cursor, cognitive-profile, reading layout, and client target
labels are rejected from the capture record.

The persistent runner now binds each attempt to the frozen schedule, base-model
bundle, actual inference model echo, decoded frame/capture contract, and a
hash-chained sidecar. Calibration training consumes exactly the 65 frozen rows,
hashes the exact raw/crop/normalized bytes it reads, excludes all 128 evaluation
rows, and verifies the resulting personal-model provenance before purging the
dedicated calibration images. No-face remains a consumed negative attempt; it
is never success-conditioned away.

Crash injection preserved the negative findings that drove the final design:

- encrypted-spool tamper becomes durable `failed_integrity` and requires
  authenticated cleanup;
- partial model output is removed only after a pre-training owned-path intent
  proves that the file belongs to this run;
- a crash after no-face classification cannot re-run preprocessing and change
  the attempt to success;
- a crash after inference but before observation sealing leaves an unsealed
  inference intent, so restart keeps predictor calls at `1 -> 1` and fails
  closed instead of replaying inference;
- wrapper and UniGaze caches are cleared on recovery, and base bundle/model/
  checkpoint identity is checked before and after training.

The only public live-analysis entry point is the canonical
`MeasurementRunner.analyze_verified_run` method, exposed locally as
`POST /api/measurement-ceiling/analysis`. It revalidates the sealed artifact,
attempt sidecar, bound model bytes, training provenance, spool absence, and
calibration-image purge before computing the frozen descriptive analysis. A
persisted file bundle or arbitrary duck-typed object cannot self-promote to
live-runner provenance. The UI keeps `measurement_claim_authorized=false`,
`physical_capture_claim_authorized=false`, and `threshold_selected=false`.

Independent adversarial review ended with P0=0 and P1=0 for starting a local
self-development acquisition rehearsal. The measurement-specific slice passed
120/120 tests. The latest full CPU-only offline gate then passed 475/475 tests
with zero failures, errors, or skips; unittest time was 248.935 seconds, gate
duration 250.383 seconds, and supervisor duration 250.703 seconds.
`artifact_changes=[]`, `network_attempts=[]`, credentials were cleared,
network/process probes were blocked, Torch was not imported, and
`CUDA_VISIBLE_DEVICES=-1`. GPU utilization was 0% before and after; external
ambient memory moved from 2782 to 2913 MiB of 24463 MiB and is not attributed to
the CUDA-hidden worker.

The tracked final gate result is
[`results/2026-08-10-webcam-gaze-measurement-ceiling-v1-final-quality-gate.json`](results/2026-08-10-webcam-gaze-measurement-ceiling-v1-final-quality-gate.json),
byte SHA-256
`9a2043172e996a7599bac56281d6b068fde7ef66ea55593ce4ea1aea3c7d54c8`.
The frozen measurement protocol canonical SHA-256 remains
`be4dfb0956ce3594546336fe7a54da6ba878f2d6fcd457d36cbaf0159132fced`;
the frozen analysis definition canonical SHA-256 is
`d3118fb8a1cb4eff437ea45e2b9b4619ce78e856d2bfbf84a4acef80f278755a`.

No human 193-row capture has been completed yet. Therefore this checkpoint is
capture-ready software and integrity evidence, not a webcam accuracy result,
natural-reading result, population claim, selected abstention threshold, or
quality-band promotion.

The current-data audit itself is pure standard library. It was reproduced with
the same CPU-only Python 3.11 runtime used for the complete gate. Private input
paths are supplied locally; the generated result binds their contents by
SHA-256 without committing pseudonymous session/model identifiers:

```powershell
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONPATH = 'D:\projects\lexigaze\.venv\Lib\site-packages'
$env:CUDA_VISIBLE_DEVICES = '-1'
$python = 'C:\Users\morris\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe'
$participantSession = '<private local participant session.json>'
$calibrationSession = '<private local calibration session.json>'
$calibrationManifest = '<private local calibration manifest.jsonl>'
$modelArtifact = '<private local personalized model.json>'
& $python -X utf8 -m scripts.audit_webcam_gaze_measurement_ceiling `
  --participant-session $participantSession `
  --calibration-session-metadata $calibrationSession `
  --calibration-manifest $calibrationManifest `
  --model-artifact $modelArtifact `
  --line-gap-px 27.2 --median-word-width-px 40.9 `
  --bootstrap-resamples 20000 --bootstrap-seed 20260810 `
  --json-output 'docs\experiments\results\2026-08-10-webcam-gaze-measurement-ceiling-v1.json' `
  --markdown-output 'docs\experiments\2026-08-10-webcam-gaze-measurement-ceiling-v1.md'
```

The complete Flask/NumPy/OpenCV offline gate was also run with that Python 3.11
runtime and the project environment on `PYTHONPATH`; its worker denied network,
child-process, provider-credential, Torch, and CUDA access and reported no
tracked artifact changes:

```powershell
& $python -X utf8 -m scripts.run_offline_quality_gate `
  --timeout-seconds 300 `
  --json-output 'docs\experiments\results\2026-08-10-webcam-gaze-measurement-ceiling-v1-quality-gate.json'
```

The fixed-target browser round trip is now closed by server-issued, single-use
prediction receipts and server-side reconstruction. The remaining integrity
boundary is natural-reading telemetry: prediction values are still posted back
by the client without per-sample receipts. They are exported only as unverified
diagnostics and cannot support a formal gaze claim. Formal participant promotion
also remains forbidden because this is an unencrypted self-development
rehearsal, not a participant-ready dataset.

## Natural-reading provenance roadmap (design only)

Natural-reading predictions remain a separate blocker from the fixed-target
validation receipts. The smallest credible follow-up is a distinct `RS`
reading-segment contract: freeze and hash the assessment viewport and a
server-validated word-layout snapshot before sampling; bind each server
prediction to session, assessment, round, passage, segment, model artifact,
capture session/contract, and server elapsed time; and persist token hashes in
a small append-only, hash-chained ledger rather than rewriting `session.json`
for every sample. Batch consumption would be single-use and idempotent, with
the server reconstructing telemetry and deriving the current nearest-word
mapping only from the frozen rectangles.

Refresh, process restart, receipt gaps, replay, or a hard capture/model/runtime
failure must close or interrupt the segment and keep gaze behavioral-only;
behavioral reading and word review may continue. The rectangles are still
client-observed DOM geometry, even after structural server validation and
hash freezing. They are not eye-tracking ground truth or independent visual
attestation, so formal natural-reading gaze remains blocked until this design
is implemented and audited.

## Next evidence step

Run a fresh participant-compatible calibration and the new held-out five-point
start/end validation without resizing the browser after system check. The
result can assign the frozen rehearsal descriptive candidate band and decide
whether gaze remains eligible for exploratory analysis or the session should
remain behavioral-only. Fixed targets and layout ratios still do not establish
natural-reading line or word accuracy. The larger 193-sample motion-ceiling
protocol remains the next dedicated self-development capture when the user is
available.

Before opening formal participant collection, repeat the complete project
offline lane after every subsequent code change, complete a fresh
receipt-and-uncertainty-bound holdout, and keep natural-reading gaze disabled
unless its separate segment receipt design is implemented and audited.

Only after a matched-contract run isolates a repeatable posture, distance, or
occlusion failure should a synchronized phone or second-camera v2 be built.
