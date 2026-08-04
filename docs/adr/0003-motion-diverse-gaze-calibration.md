# ADR 0003: Motion-diverse gaze calibration and grouped validation

- Status: Accepted
- Date: 2026-08-05
- Branch: `research/gaze-motion-robustness`

## Context

The existing personalization path freezes UniGaze and fits a small polynomial
from predicted gaze pitch/yaw to screen coordinates. Its hyperparameters were
selected by leaving out one frame at a time. Frames captured seconds apart can
be nearly identical, so a neighbouring frame from the same posture can remain
in training and make validation look substantially better than an unseen
posture actually is.

The historical manifests contain many calibration rows, but they do not label
camera, device, distance, posture, motion block, lighting, or capture burst.
They therefore cannot separate target effects from posture effects. Adding more
unlabelled neutral frames would increase sample count without making motion
robustness identifiable.

## Decision

1. Keep the frozen UniGaze encoder as the baseline. Do not fine-tune it until a
   cheaper calibration-level intervention fails on frozen held-out data.
2. Add protocol `motion-diverse-v1`: the same 13 targets are captured in five
   blocks (`neutral`, `left`, `right`, `near`, and `far`), producing at least 65
   rows per session.
3. Persist a bounded allow-list of capture metadata on every row: `camera_id`,
   `device_class`, `motion_block_id`, `capture_burst_id`, posture, distance,
   lighting, and non-identifying camera geometry/rate.
4. Reject motion-diverse training unless the frozen metadata audit passes. The
   gate checks usable sample count, block/condition coverage, repeated targets,
   actual yaw separation, and near/far face-scale separation.
5. Select polynomial degree and ridge regularization using leave-one-motion-
   block-out validation. Never split one motion block across train and
   validation.
6. Compare two calibration candidates on exactly the same folds:

   - M0: gaze-only linear/quadratic ridge calibration;
   - M1: standardized ridge using gaze, head pose, normalized face location,
     face scale, and predeclared interactions.

   M1 is promoted only when its held-out improvement is at least both 5 px and
   5% relative to M0. Otherwise M0 remains the production artifact.
7. Report held-out error separately from train error. Existing legacy models
   retain their old value and are labelled as legacy rather than silently
   reinterpreted.
8. A motion-diverse session must start from the frozen base model. Cascaded
   calibration stages are not allowed until their leakage behaviour is covered
   by a separate decision and test.
9. Motion-diverse feature extraction defaults to CPU. CUDA requires the caller
   to set `allow_cuda=true` for a recorded, guarded experiment.

## Consequences

- The system can now collect data that identifies motion effects instead of
  merely increasing the number of similar frames.
- Failed physical movement is detected from measured head yaw and face scale,
  even if the UI labels say the requested movement was performed.
- A synthetic duplicate-frame regression demonstrates why sample-level LOOCV
  is not evidence of motion robustness.
- The collection takes longer (at least 65 targets) and requires user movement.
- No real-world accuracy gain is claimed until a new protocol-compliant session
  is captured and evaluated. The implementation creates the measurement path;
  it does not manufacture evidence from the historical dataset.
- A second camera or phone remains a compatible later extension. It should add
  a new synchronized camera domain and held-out-camera evaluation, not replace
  the single-camera motion baseline.

## Alternatives rejected for this stage

- More unlabelled frames from the current neutral workflow: does not identify
  posture or distance effects.
- Random frame split: leaks near-duplicate bursts.
- Immediate UniGaze fine-tuning: higher GPU cost and overfit risk before the
  calibration bottleneck is measured.
- Brightness/blur augmentation alone: useful for appearance variation but does
  not reproduce real 3D head movement or distance changes.
- Second camera first: interesting and still planned, but it adds another
  domain before the existing single-camera failure can be measured cleanly.
