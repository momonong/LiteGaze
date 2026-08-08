# LexiGaze System Optimization Roadmap — 2026-08-08

This roadmap was written while the frozen subject-heldout gaze experiment was
waiting for an uncontended GPU window. It does not use that experiment's final
outcomes to choose a model or threshold.

## 2026-08-08 milestone status

Subject-heldout gaze diversity v1 passed all frozen gates. On 15 held-out
MPIIGaze people, the tiny eye-image plus pose candidate reduced macro angular
error from `9.1250` to `6.9450` degrees versus pose-only (`23.89%` relative),
with a participant-bootstrap 95% CI of `[-2.7186, -1.5944]` degrees and
improvement on `14/15` people. This updates the decision branch, not the model:
the candidate remains research-only and the next milestone is independent
cross-dataset or real-capture confirmation.

## Evidence already established

### Text modeling

- Frozen GPT-2-small causal surprisal adds a small, same-direction signal beyond
  lexical controls on PROVO, English GECO L2, and the independently frozen
  OneStop confirmation corpus.
- Predictive entropy did not pass its preregistered incremental gate and should
  not be restored merely because the legacy artifact contains it.
- The provenance-complete Ridge text artifact has fixed training calibration,
  feature-distribution bounds, source hashes, and training-only scaling, but it
  remains a candidate rather than the production default.

The next text-model investment should therefore be calibration and independent
fusion evidence, not a larger language model or another public-corpus feature
search.

### Fusion

- Quality-aware fusion v2 improved aggregate synthetic-corruption MAE by about
  2.27%, handled jitter, dropout, and missing gaze safely, and retained exact
  text fallback.
- It failed its frozen drift condition: drift MAE increased by about 0.00448
  versus static fusion. The candidate correctly remained shadow-only.
- Synthetic corruption cannot establish real webcam benefit. A v3 must explain
  and measure drift from independent physical signals instead of retuning the
  v2 weight on the same synthetic benchmark.

### Gaze and calibration

- Motion-conditioned calibration is useful within the captured session, but
  historical data do not establish cross-person or cross-session robustness.
- The current v1 public-data experiment measures whether a small, diverse
  eye-image plus head-pose model transfers to unseen MPIIGaze people. It does
  not claim LexiGaze webcam improvement.
- Production UniGaze must remain unchanged until a separately frozen
  cross-dataset or real-capture confirmation passes.

## Recommended next evidence-producing milestone

Prepare one independent participant-capture protocol that can evaluate gaze,
text, and fusion without circular labels.

### Capture units

- At least ten independent participant-session-device capture groups.
- Complete capture groups held out in the primary analysis and complete
  articles held out in the secondary analysis.
- Multiple lighting, distance, posture, and head-motion blocks with immutable
  capture IDs.
- If a second camera or phone is added, treat it as an independent synchronized
  sensor/source and hold out device/source groups; do not merge near-duplicate
  frames across devices into different folds.

### Outcomes

- Use a short post-reading word familiarity or perceived-difficulty audit as
  the primary fusion target.
- Do not use public QA correctness, comprehension answers, reading time derived
  from the same gaze trace, text-model scores, or calibration labels as the
  promotion target.
- Keep reading behavior and adaptive-test performance as secondary descriptive
  measurements until their construct validity is separately established.

### Candidate comparisons

1. Text-only provenance-complete causal-surprisal artifact.
2. Gaze-only measurement with explicit tracking, calibration, and drift
   quality.
3. Existing static fusion.
4. A future drift-aware v3 whose reliability inputs are frozen before outcomes.

The v3 hypothesis should focus on detecting sustained spatial bias using
calibration residuals, head-pose change, face geometry, temporal consistency,
and optional cross-camera disagreement. It should abstain or fall back to text
rather than increasing gaze weight when those signals indicate drift.

## Literature implications for the two-camera idea

- [Gaze360](https://openaccess.thecvf.com/content_ICCV_2019/html/Kellnhofer_Gaze360_Physically_Unconstrained_Gaze_Estimation_in_the_Wild_ICCV_2019_paper.html)
  found value in temporal information and explicit uncertainty under broad head
  poses and distances. For LexiGaze, a temporal confidence output is likely a
  cheaper first step than adding another sensor.
- [Rotation-Constrained Cross-View Feature Fusion](https://openaccess.thecvf.com/content/WACV2024/html/Hisadome_Rotation-Constrained_Cross-View_Feature_Fusion_for_Multi-View_Appearance-Based_Gaze_Estimation_WACV_2024_paper.html)
  shows that multi-view benefit depends on paired images and the relative camera
  rotation, and can generalize to unseen camera pairs. Merely placing a phone
  beside the laptop is not enough: timestamps, extrinsics, capture identity,
  and missing-view behavior must be part of the protocol.
- [Analytical Gaze Generalization](https://openaccess.thecvf.com/content/CVPR2024/html/Bao_From_Feature_to_Gaze_A_Generalizable_Replacement_of_Linear_Layer_CVPR_2024_paper.html)
  identifies the high-dimensional feature-to-gaze fully connected mapping as a
  cross-domain overfit point. If the current tiny baseline fails, a constrained
  output geometry is a better v2 hypothesis than blindly widening the CNN.
- [UnReGA](https://openaccess.thecvf.com/content/CVPR2023/html/Cai_Source-Free_Adaptive_Gaze_Estimation_by_Uncertainty_Reduction_CVPR_2023_paper.html)
  supports separating sample and model uncertainty for target-domain
  adaptation. Any LexiGaze adaptation study should use fresh unlabeled capture
  groups and remain separate from the immutable v1 held-out outcomes.

## Decision branches after gaze-diversity v1

- **Observed branch — v1 passed:** retain the tiny candidate only as a research
  baseline and freeze a cross-dataset or real-capture confirmation. Do not
  promote it to the webcam path from MPIIGaze alone.
- Preserve the `p09` regression and the complete v1 result. Do not select a v2
  architecture, feature, or threshold against the same MPIIGaze subjects.
- Prioritize capture independence and fusion drift evidence over unconstrained
  model-size growth.

## Hardware policy

- Run Ridge, bootstrap, text-artifact, and quality-fusion work on CPU.
- Use GPU only for explicit gaze/vision training or label-free LM extraction
  that cannot be served efficiently by the existing cache.
- Require an idle-window preflight, per-process VRAM cap, thermal cutoff,
  resumable artifacts, and recorded telemetry for every long GPU run.
- Never overlap long LexiGaze work with another compute process on the same GPU;
  scheduling contention invalidates timing evidence and increases thermal risk.
