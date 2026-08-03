# GECO-to-PROVO Zero-Shot Results

Protocol: `docs/PROVO_ZERO_SHOT_PROTOCOL_V1_1_2026-08-03.md` (v1.1)

Run completed: 2026-08-03T08:41:39.294185+00:00

Runtime: 38.38 seconds, CPU-only; Torch/CUDA not imported

## Data identity

- GECO L1 training: 18 participants,
  3,052 participant-trials,
  324,240 word rows
- GECO tree SHA-256: `b9833e60b84ff49f8f8c9c7c4e7e990f091779bb1ef679933085c57f202b01bd`
- PROVO test: 84 participants,
  55 texts,
  230,412 word rows
- PROVO source SHA-256: `38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7`

## Frozen primary evaluation

The model was fitted on GECO L1 only. No PROVO observation was used for fitting,
feature scaling, calibration, filtering, or model selection.

| Model | Macro participant Spearman rho | Participant bootstrap 95% CI |
| --- | ---: | ---: |
| `geco_lexical_ridge` | 0.2205 | [0.2035, 0.2377] |
| `word_length_only` | 0.2951 | [0.2758, 0.3144] |
| `lexical_rarity_only` | 0.2811 | [0.2623, 0.2996] |

Primary Ridge result: **rho = 0.2205**,
95% CI **[0.2035,
0.2377]**; participant sign-flip p-value against
zero **0.000010**.

Ridge minus word length: **-0.0746**,
95% bootstrap CI **[-0.0853,
-0.0634]**; paired sign-flip
p-value **0.000010**.

Frozen decision: **`basic_lexical_transfer_only`**.

## Secondary fixation occurrence

The GECO-trained logistic model reached macro participant ROC AUC
**0.6486**, 95% CI
**[0.6410,
0.6561]**, and macro Brier score
**0.2170**.

## Interpretation guardrails

- PROVO is a completely independent corpus and was never used to tune this model.
- No question-answer data, PROVO cloze norm, LSA value, or predictability field was
  used.
- GECO surprisal, attention, and `cognitive_mass` were excluded because an identical
  frozen feature generator was not available for both corpora or provenance was
  target-sensitive.
- Spearman is the primary endpoint because absolute reading-time calibration can
  change across eye trackers, preprocessing pipelines, and participant populations.
- PROVO v1 is now a frozen test result and must not become a development set.
