# PROVO Cross-Corpus Zero-Shot Execution Log

Date: 2026-08-03

Frozen result protocol: `docs/PROVO_ZERO_SHOT_PROTOCOL_V1_1_2026-08-03.md`

Result manifest: `output/provo_zero_shot_manifest.json`

## Question and completion rule

The confirmatory question was whether a fixed lexical model trained only on GECO
native-English readers transfers to the independent PROVO corpus without fitting,
scaling, filtering, or calibration on PROVO.

Engineering completion required a preregistered protocol, official-source
fingerprints, a CPU-only reproducible pipeline, participant-level uncertainty,
leakage tests, and publication of positive or negative results. Completion did not
depend on obtaining a favorable correlation.

## Chronology and outcome isolation

1. Protocol v1 was committed as `8356a44` before the complete PROVO file was
   downloaded and before any aggregate outcome was computed.
2. The official OSF object `a32be` was downloaded and matched the preregistered
   size and SHA-256.
3. Acquisition, schema validation, fixed models, inference, and synthetic leakage
   tests were committed as `3732288` before the first full execution.
4. The first v1 execution stopped during PROVO schema validation, before fitting a
   model, with a non-finite annotation-field error.
5. Schema-only diagnosis showed that the complete EyeLink fields are `IA_ID` and
   `IA_LABEL`, while some joined predictability-annotation fields are `NA`. It also
   showed that `IA_SKIP` is a first-pass measure and cannot define total fixation.
6. Protocol v1.1 was committed as `e66f846`; the matching implementation correction
   was committed as `2fef969`.
7. The first successful v1.1 execution produced the frozen result below. No model,
   feature, comparator, exclusion, statistic, or decision threshold was changed
   afterward.

The SR Research reading-measures reference defines `IA_DWELL_TIME` as total
duration and uses `IA_SKIP` for first-pass fixation ratio:
<https://www.sr-research.com/support/thread-336.html>.

## Data identity

### GECO L1 training corpus

- 18 native-English participants
- 3,052 participant-trials
- 324,240 word rows; 241,679 positive-total-dwell rows
- 6,104 source files / 24,905,820 bytes
- Tree SHA-256:
  `b9833e60b84ff49f8f8c9c7c4e7e990f091779bb1ef679933085c57f202b01bd`

### PROVO test corpus

- Official project: <https://osf.io/sjefs/>
- Official file GUID: `a32be`
- 84 participants × 55 texts = 4,620 participant-text pairs
- 2,743 unique displayed word items
- 230,412 rows; 153,566 positive-total-dwell rows
- 69,662,713 bytes
- SHA-256:
  `38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7`

The raw PROVO CSV is ignored by Git. The fingerprint, model parameters, per-subject
metrics, summary, report, plot, and manifest are tracked.

## Frozen results

The primary unit of inference is participant. Intervals use 10,000 participant
bootstraps; p-values use 100,000 participant sign flips.

| Frozen duration score | Macro participant Spearman rho | 95% bootstrap CI |
| --- | ---: | ---: |
| GECO lexical Ridge | 0.2205 | [0.2035, 0.2377] |
| Word length only | 0.2951 | [0.2758, 0.3144] |
| Lexical rarity only | 0.2811 | [0.2623, 0.2996] |

- Ridge versus zero: `p=0.000010`.
- Ridge minus word length: `-0.0746`, 95% CI `[-0.0853, -0.0634]`;
  paired sign-flip `p=0.000010`.
- Frozen decision: `basic_lexical_transfer_only`.

The positive Ridge interval supports transfer of some basic lexical signal, but the
model fails the preregistered incremental criterion because it is reliably worse than
raw word length. The result is therefore not evidence that combining the five frozen
features improves corpus-independent duration ranking.

The secondary GECO-trained fixation logistic model reached macro participant ROC
AUC `0.6486`, 95% CI `[0.6410, 0.6561]`, and macro Brier `0.2170`, 95% CI
`[0.2073, 0.2271]`. This is a modest any-fixation signal and cannot change the
primary conclusion.

## Post-hoc interpretation (not confirmatory)

These observations were computed only after the result was frozen and are not used
to revise v1.1:

- Ridge was below zero for 1 of 84 participants.
- Ridge beat word length for only 7 of 84 participants.
- Lexical rarity beat word length for 18 of 84 participants.
- Word length was non-negative for all 84 participants.
- The GECO Ridge coefficient for standardized Zipf frequency was slightly positive
  (`+0.0071`), whereas the cross-corpus rarity comparator assumes rarer words take
  longer. The relative-position coefficient was `-0.0509`.

This is a diagnostic clue, not a causal explanation: unconstrained directions learned
from one novel and apparatus can encode corpus-specific structure that weakens
transfer. The large and consistent word-length advantage is the stronger result.

## Reproducibility verification

The frozen execution took 38.38 seconds. Its manifest records:

- source commit `2fef969bbfde0ee5d0bd0ad7f44644c1f9544803`;
- clean Git state and no tracked diff;
- `LEXIGAZE_DEVICE=cpu`;
- Torch/CUDA not imported by the evaluator;
- source, dataset, configuration, model-result, and artifact hashes.

A second execution used the same code and an independent ignored output directory.
The following outputs were byte-identical:

- `provo_zero_shot_fingerprint.json`;
- `provo_zero_shot_subject_metrics.csv`;
- `provo_zero_shot_models.json`;
- `provo_zero_shot.png`.

The summary JSON was semantically identical after removing only timestamp and
wall-clock runtime. The rerun produced the same frozen decision.

## Leakage audit

- No question-answer data was loaded.
- PROVO supplied no training target, feature mean/scale, coefficient, offset,
  threshold, early-stopping signal, or exclusion decision.
- PROVO cloze, LSA, predictability, POS, and annotation word-length fields were not
  model inputs.
- GECO surprisal and attention were excluded because the same frozen generator was
  unavailable for PROVO.
- GECO `cognitive_mass` was excluded because its provenance can include GECO
  reading-time supervision.
- Statistics resampled participants, not individual word rows.
- The complete source file was used; no duration-magnitude outlier trimming occurred.

## Decision and next development boundary

1. Freeze PROVO v1.1 permanently; it is now test evidence, not development data.
2. Keep word length and lexical rarity as mandatory corpus-independent baselines.
3. Do not deploy the unconstrained GECO lexical Ridge as an improvement over the
   simple baseline.
4. If a richer lexical model is pursued, develop it on a third corpus that is neither
   GECO nor PROVO. Prespecify monotonic/sign constraints or robust multi-corpus
   training there.
5. Confirm any future candidate once on another untouched corpus. Do not select the
   candidate by checking it repeatedly against GECO or PROVO.
