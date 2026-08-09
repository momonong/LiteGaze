# ADR 0006: Webcam gaze measurement boundary before text or cognitive fusion

- Status: Accepted
- Date: 2026-08-10
- Branch: `experiment/webcam-gaze-measurement-ceiling-v1`

## Context

The participant reading flow currently combines three different questions:

1. whether the webcam and personal calibrator can localize gaze on the screen;
2. whether screen geometry can narrow a sensor observation to a line or word
   occurrence; and
3. whether text difficulty or a cognitive profile can improve a downstream
   interpretation.

Those questions cannot be evaluated with the same labels. A text prior that
pulls an observation toward a difficult word may make a visualization look
plausible while hiding a poor sensor measurement. Cursor position and the
system's own nearest-word output are also not independent eye-tracking truth.
The existing rehearsal is useful development material, but its five fixed
validation regions are too close to calibration targets and its historical
model artifact conflates an inner hyperparameter score with the selected
nested-outer score. Its 1280x720 calibration camera and 640x480 participant
system-check camera also differ in aspect ratio by `0.444444`, so it cannot
support a matched-capture measurement claim.

## Decision

1. Measure the single-webcam sensor ceiling before adding another camera,
   changing the encoder, or tuning a fusion weight. Calibration, target
   decoding, and cognitive/text inference remain distinct stages.
2. Treat camera geometry as a versioned capture contract. Calibration,
   validation, and reading must preserve the actual aspect ratio, resize
   policy, mirror policy, facing mode, and inference transport geometry. A
   missing or incompatible contract disables gaze-dependent output while
   retaining behavioral collection.
3. Add the participant gaze measurement definition as a separate additive
   contract with its own ID, version, and hash. Do not rewrite the frozen
   general-collection v1 file or invalidate an invite already assigned to
   Visit 2. New collection state records the additive contract identity so the
   measurement remains reproducible.
4. Bind capture provenance on the server-created session. Browser-supplied run
   or source identifiers cannot override the session metadata.
5. Persist the screen targets that actually contributed to calibration. A
   validation target is independent only when its distance from every fitted
   target is at least 0.1 in viewport-fraction coordinates (0.2 in the signed
   `[-1, 1]` coordinates used by the calibrator). Missing or failed target
   independence is a gaze abstention, not a warning.
6. Keep live sensor mapping geometry-only. Text difficulty, token identity,
   cognitive profile, question correctness, and reading outcome cannot change
   raw coordinates, sensor acceptance, geometry coverage, or the sensor
   candidate set.
7. Develop the line-first occurrence decoder in shadow mode. It selects a line
   and then ranks unique word occurrences from bounding boxes and reading
   order, with explicit distance abstentions. It does not yet change the
   production mapping path.
8. Evaluate fixed targets only with target-held-out, target-macro metrics,
   directional error, drift, and target-cluster bootstrap intervals. Report
   layout-relative resolution as line-gap and word-width ratios, not as
   natural-reading line or word accuracy.
9. Keep selected nested-outer validation error distinct from hyperparameter
   cross-validation error in the calibration artifact and API.
10. Treat the current browser-to-validation sample round trip as rehearsal
    instrumentation only. It may display a provisional self-development band,
    but it cannot authorize formal promotion because predicted coordinates and
    success flags are posted back by the client. A formal protocol requires
    server-issued, single-use prediction receipts bound to session, model,
    phase, target, and capture-contract result, with metrics reconstructed on
    the server.
11. Run measurement-ceiling analysis offline on CPU. GPU training is not useful
   until independent evidence shows that the frozen encoder, rather than
   capture geometry or calibration, is the limiting component.
12. Consider a phone or second camera only after a matched-contract capture
    shows that errors reliably vary with posture, distance, or occlusion. A
    multiview design must introduce synchronized capture groups and a new
    held-out-camera protocol; it cannot reuse the single-camera result as its
    confirmation set.
13. Freeze the assessment viewport at system check. A resize or orientation
    change aborts fixed-target validation; during reading it permanently
    disables gaze analysis while preserving the behavioral round. Re-entering
    an active reading round without a versioned segment-resume contract keeps
    raw telemetry but cannot report a combined effective sampling rate.
14. Export behavioral data independently from gaze eligibility. Gaze-only
    tables require a verified session-frozen measurement contract, validation
    and telemetry bindings, capture compatibility, target independence, and
    integrity state. Visit pairs with different contract hashes remain usable
    separately but cannot be compared as a gaze pair.

## Existing-data audit

The 2026-08-10 self-development audit is deliberately retained despite failing
its integrity gate:

- capture provenance conflicts between session metadata and manifest rows;
- calibration used a 16:9 camera frame while the participant system check
  recorded a 4:3 frame, exceeding the frozen cross-phase aspect tolerance;
- all five evaluation regions are within the frozen exclusion radius of a
  calibration target;
- the historical top-level `199.18 px` value is hyperparameter CV, while the
  selected nested-outer M0 score is `204.58 px`;
- start and end median errors are `213.30 px` and `128.69 px`, respectively;
- the end median is still about `4.73` configured line gaps or `3.15` median
  word widths;
- a start-fitted translation worsens end median error by `19.74 px`; its
  target-cluster bootstrap interval crosses zero widely;
- natural-reading line accuracy, word accuracy, and uncertainty calibration
  remain not evaluable.

The coarse five-region nearest-target score is 100%, but this establishes only
large-region separation. It does not promote a line-level or word-level gaze
claim.

## Consequences

- New captures can fail closed on camera or target leakage while preserving
  useful behavioral reading data.
- The immediate geometry band remains a rehearsal aid, not tamper-resistant
  research evidence, until server-issued prediction receipts replace the
  client round trip.
- Existing legacy models without capture or fitted-target contracts remain
  usable only as unchecked development artifacts; they cannot receive a new
  gaze-quality claim.
- The geometry-only mapping removes circular cognitive attraction from sensor
  coverage. Any future text benefit must be evaluated downstream against an
  independent outcome.
- The next useful user action is a fresh protocol-compliant fixed-target
  capture. More natural-reading video or more frames at the same calibration
  points will not close the current measurement gap.
- A negative single-camera result is actionable: it identifies when to invest
  in multiview hardware without spending GPU time on an unmeasured bottleneck.

## Alternatives rejected for this stage

- Treat cursor position as eye-tracking ground truth: useful as an explicit
  pointing task, but not independent natural-reading gaze truth.
- Use text difficulty to attract gaze toward plausible words: contaminates the
  sensor measurement and can rescue an otherwise invalid observation.
- Tune a translation on the same end validation data used to report it:
  leakage; the start-only correction already failed to show stable benefit.
- Train a larger gaze encoder immediately: higher GPU cost without evidence
  that representation capacity is the limiting factor.
- Add a second camera before the single-camera ceiling test: increases capture
  and synchronization complexity without identifying the present failure.
