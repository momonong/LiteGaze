# ADR 0004: Provenance-complete text artifact and non-circular fusion gate

- Status: Accepted
- Date: 2026-08-06
- Branch: `feat/provenance-complete-fusion`

## Context

Frozen PROVO, English GECO L2, and OneStop experiments found that GPT-2-small
causal surprisal adds a small but consistent signal beyond lexical controls.
Predictive entropy did not pass its promotion gate. The committed product
XGBoost artifact predates those experiments, still consumes Rényi entropy, and
does not contain enough provenance to reconstruct its training run.

The historical fusion demo cannot close this gap. It constructs simulated gaze
from GECO total reading time and evaluates the fused score against that same
reading time. The cross-attention alternative is also untrained. Neither path
is eligible evidence that text improves real gaze measurements.

## Decision

1. Build English candidate `lexigaze-en-text-difficulty-m1-v1` from PROVO and
   English GECO L2 only. Give each corpus equal total training weight so the
   larger GECO table does not erase the PROVO population.
2. Reuse verified, label-free GPT-2 CPU feature caches. Include lexical controls
   plus current and previous causal surprisal. Exclude Shannon and Rényi entropy.
3. Keep OneStop outside artifact training, calibration, threshold selection,
   and further model choice. Its result remains immutable external confirmation.
4. Store feature order, Ridge coefficients, training-only scaling, fixed score
   calibration, feature-distribution bounds, source hashes, model revision,
   extractor hashes, and leakage controls in the candidate and its manifest.
5. Never normalize scores within an incoming document or request. Map raw
   predictions with fixed development-training 5th and 95th percentiles.
6. Mark observations outside the frozen 0.5th-to-99.5th percentile feature
   envelope instead of silently treating them as in-distribution.
7. Do not change the production default after the development build. Promotion
   requires protocol `production-text-fusion-v1` on future independent real
   captures.
8. Evaluate text-only, gaze-only, and combined standardized Ridge models on the
   same deterministic folds. Hold out complete participant-session-device
   capture groups in the primary analysis and complete articles in the
   secondary analysis. Balance training weight by the active holdout group.
9. Do not use reading time as the primary fusion target when gaze predictors
   contain dwell information. The promotion target must be an independent
   post-reading word familiarity or difficulty audit, recorded without using a
   public QA benchmark and not derived from gaze or text-model output.
10. Require at least ten usable groups on each holdout axis. Combined must beat
    gaze-only with a positive paired 95% bootstrap interval, improve in at least
    four of five folds, and avoid a strictly positive shuffled-target sentinel
    on both axes.
11. Keep all artifact fitting and fusion evaluation on CPU. GPU allocation is
    unnecessary for the frozen Ridge path.

## First candidate build

Run 001 used 509,428 positive development outcome rows and completed in 11.30
seconds on CPU. The GPU allocation did not change during the run.

| Corpus | Rows | Participant M1-M0 | 95% CI | Text M1-M0 | 95% CI | Positive folds |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| GECO L2 English | 359,314 | +0.0113 | [+0.0081, +0.0144] | +0.0181 | [+0.0158, +0.0206] | 5/5 |
| PROVO | 150,114 | +0.0112 | [+0.0089, +0.0135] | +0.0297 | [+0.0187, +0.0409] | 5/5 |

These are implementation-consistency diagnostics on previously inspected
development outcomes. They do not constitute a fourth confirmation corpus.

## Consequences

- The validated causal-surprisal signal now has a small, auditable candidate
  artifact that can be reproduced without language-model fine-tuning.
- Product behavior is unchanged until genuinely independent fusion evidence
  passes the frozen gate.
- A future audit capture needs a short post-reading familiarity/difficulty
  annotation. This is research instrumentation, not a public QA training set.
- Existing reading-time and simulated-fusion reports remain useful descriptive
  diagnostics but are not promotion evidence.
- The fixed distribution guard makes model extrapolation visible to downstream
  confidence and abstention logic.

## Alternatives rejected for this stage

- Silently replace the current XGBoost artifact: its behavior would change
  before real fusion benefit is established.
- Add entropy because it exists in the legacy product: the frozen replication
  gate did not support it.
- Retrain on OneStop: this would destroy the independent audit boundary.
- Select a fusion weight on the future audit set: this would tune directly on
  the evidence intended to validate promotion.
- Use dwell-derived reading time as the only fusion target: it cannot establish
  that the text branch contributes independent user-difficulty information.
- Train cross-attention immediately: higher compute and overfit risk without a
  sufficient independent labeled capture set.
