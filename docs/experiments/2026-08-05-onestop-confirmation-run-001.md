# OneStop Pristine Text-Model Confirmation - Run 001

- Completed: 2026-08-05T14:51:28.283657+00:00
- Runtime: 57.24 seconds
- Device: `cpu` (GPU used: false)
- Protocol: `onestop-ordinary-advanced-confirmation-v1`
- Decision: **`causal_surprisal_confirms_on_pristine_onestop`**

## Scope and provenance

- Source SHA-256: `8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9`
- Selected participants: 180
- Complete article groups: 30
- Display contexts: 173
- Unique displayed items: 20,788
- Ordinary reading, original Advanced Guardian paragraphs, first reading only.
- The protocol and source/header admission were committed before outcome access.

## Fixed design

- Five deterministic complete-article folds; no paragraph or row leakage.
- M0 lexical controls versus M1 frozen GPT-2-small causal surprisal.
- Ridge alpha 1.0 and training-fold-only standardization.
- A shuffled-training-target M1 sentinel was included.
- Secondary outcomes are descriptive and cannot change the decision.

## Total Reading Time

Rows: 371,443; participants: 180; articles: 30.

| Model | Macro participant rho | Macro article rho | Log-duration MAE |
| --- | ---: | ---: | ---: |
| `word_length_only` | 0.2438 | 0.5332 | 0.4889 |
| `m0_lexical` | 0.2788 | 0.5952 | 0.4828 |
| `m1_causal_surprisal` | 0.2851 | 0.6061 | 0.4818 |
| `target_shuffle_sentinel` | -0.0194 | 0.0400 | 0.5046 |

| Comparison | Participant delta [95% CI] | Article delta [95% CI] | Positive folds |
| --- | ---: | ---: | ---: |
| `m0_lexical_minus_word_length_only` | +0.0350 [+0.0312, +0.0388] | +0.0620 [+0.0516, +0.0726] | 5/5 |
| `m1_causal_surprisal_minus_m0_lexical` | +0.0063 [+0.0054, +0.0073] | +0.0108 [+0.0078, +0.0139] | 5/5 |
| `m1_causal_surprisal_minus_word_length_only` | +0.0413 [+0.0374, +0.0454] | +0.0728 [+0.0627, +0.0835] | 5/5 |

Shuffle sentinel macro rho versus zero: participant -0.0194 [-0.0265, -0.0123]; article +0.0400 [-0.0331, +0.1141].

## Gaze Duration

Rows: 371,443; participants: 180; articles: 30.

| Model | Macro participant rho | Macro article rho | Log-duration MAE |
| --- | ---: | ---: | ---: |
| `word_length_only` | 0.1641 | 0.3882 | 0.3550 |
| `m0_lexical` | 0.1864 | 0.4359 | 0.3536 |
| `m1_causal_surprisal` | 0.1900 | 0.4455 | 0.3534 |
| `target_shuffle_sentinel` | 0.0161 | 0.1336 | 0.3591 |

| Comparison | Participant delta [95% CI] | Article delta [95% CI] | Positive folds |
| --- | ---: | ---: | ---: |
| `m0_lexical_minus_word_length_only` | +0.0223 [+0.0198, +0.0248] | +0.0476 [+0.0405, +0.0552] | 5/5 |
| `m1_causal_surprisal_minus_m0_lexical` | +0.0036 [+0.0029, +0.0044] | +0.0096 [+0.0070, +0.0121] | 5/5 |
| `m1_causal_surprisal_minus_word_length_only` | +0.0259 [+0.0233, +0.0286] | +0.0573 [+0.0495, +0.0655] | 5/5 |

Shuffle sentinel macro rho versus zero: participant +0.0161 [+0.0109, +0.0212]; article +0.1336 [+0.0839, +0.1825].

## First Fixation Duration

Rows: 371,443; participants: 180; articles: 30.

| Model | Macro participant rho | Macro article rho | Log-duration MAE |
| --- | ---: | ---: | ---: |
| `word_length_only` | 0.0685 | 0.1750 | 0.3143 |
| `m0_lexical` | 0.1007 | 0.2618 | 0.3137 |
| `m1_causal_surprisal` | 0.1052 | 0.2745 | 0.3135 |
| `target_shuffle_sentinel` | 0.0097 | 0.0500 | 0.3150 |

| Comparison | Participant delta [95% CI] | Article delta [95% CI] | Positive folds |
| --- | ---: | ---: | ---: |
| `m0_lexical_minus_word_length_only` | +0.0322 [+0.0279, +0.0364] | +0.0868 [+0.0735, +0.1005] | 5/5 |
| `m1_causal_surprisal_minus_m0_lexical` | +0.0045 [+0.0031, +0.0059] | +0.0127 [+0.0074, +0.0176] | 5/5 |
| `m1_causal_surprisal_minus_word_length_only` | +0.0367 [+0.0321, +0.0411] | +0.0995 [+0.0852, +0.1142] | 5/5 |

Shuffle sentinel macro rho versus zero: participant +0.0097 [+0.0054, +0.0139]; article +0.0500 [+0.0248, +0.0762].

## Confirmation decision

The preregistered primary gate passed without changing a feature, filter,
threshold, fold, or model after outcome access:

1. M1-M0 participant delta was `+0.0063`, with 95% CI
   `[+0.0054, +0.0073]`.
2. M1-M0 complete-article delta was `+0.0108`, with 95% CI
   `[+0.0078, +0.0139]`.
3. The M1-M0 participant-macro difference was positive in all five outer
   folds (the frozen minimum was four).
4. The primary shuffle sentinel failed to establish positive signal: its
   participant estimate was negative, while its article CI crossed zero.

The effect is smaller than the earlier Provo and GECO L2 deltas, but it has the
same direction across an untouched corpus, both independent aggregation units,
and every complete-article fold. This supports keeping causal surprisal as a
modest auxiliary text signal; it does not support letting text evidence dominate
the gaze-text fusion score.

The two secondary outcomes also had positive M1-M0 effects in all folds, but
their shuffled-target sentinels were positively separated from zero. They were
preregistered as descriptive only and did not affect the decision. They should
not be presented as additional independent confirmation, nor used for tuning.

## Execution incidents and hardware findings

All incidents below occurred before the outcome pass and did not alter the
scientific protocol:

1. The archive initially appeared to contain two CSV files. The second was a
   176-byte macOS resource-fork sidecar under `__MACOSX`; the inspector was
   tightened to ignore only that packaging artifact. The verified main member
   and source hash did not change.
2. A first CPU attempt exposed an import-time accelerator side effect:
   `pipeline.py` imported spaCy eagerly, and spaCy/thinc queried CUDA even though
   the requested text device was CPU. spaCy is now loaded only by the full
   linguistic pipeline that uses it.
3. Transformers 5.14 entered CUDA tracing helpers from its GPT-2 padding and
   SDPA mask paths even with CUDA hidden. GPT-2 now uses its equivalent
   `inputs_embeds` + `attention_mask` forward contract and explicitly pinned
   eager attention. A real four-word GPT-2 probe passed with
   `CUDA_VISIBLE_DEVICES=-1` and unchanged GPU telemetry.

The unsuccessful library probes briefly changed desktop GPU memory from 168 to
276 MiB while utilization remained 0%; no formal feature or outcome result was
produced by those attempts. The successful label-only run stayed at 0% and 246
MiB before/after. The formal 57.24-second confirmation stayed at 0% and moved
from 246 to 211 MiB, with no GPU compute allocation claimed or required.

Formal environment: Python 3.11.9 from the project `.venv`, PyTorch
2.13.0+cu130, Transformers 5.14.1, pandas 3.0.5, NumPy 2.4.6, and SciPy 1.17.1.
The machine-wide Python 3.14 installation lacked project dependencies and was
not used for the experiment.

## Reproducibility audit

- Protocol freeze commit: `ea7d668`.
- Outcome-blind source/header admission commit: `b3402cc`.
- Formal source commit: `0b1a2e8a10fdfbb411ebc400848b98ffc488df8a`
  with a clean worktree.
- Source SHA-256:
  `8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9`.
- Display-item identity SHA-256:
  `d5fa4b538ef78664a3739be2b1be05f417e18f9ba5e6a85e2622a8b71e98aaa0`.
- Label-only feature SHA-256:
  `21ae475e79124a392ed353ebe6848cdddca8407a00c46eacb80ad0091705287b`.
- The feature cache contains 20,788 items over 173 displayed paragraph
  sequences and records that zero outcome, QA, or corpus-precomputed feature
  columns were read.
- The outcome pass read exactly the 15 frozen whitelist columns: 12 identity /
  label fields plus the three declared durations.
- Result audit rehashed all eight ignored local artifacts, reproduced the
  decision from the tracked summary, and confirmed five folds of six articles
  each with zero group overlap.

## Guardrails and interpretation boundary

- No question, answer, correctness, comprehension-score, or STARC span field was loaded.
- No corpus-provided surprisal, frequency, word-length, syntax, or semantic field was loaded.
- No gaze coordinate or duration was used as a predictor.
- No feature, model size, alpha, fold, filter, or threshold search was performed.
- Participants answered a question after each paragraph, so the result applies to ordinary reading for comprehension, not unrestricted browsing.
- "Pristine" means outcome-blind relative to this project. Possible overlap
  between GPT-2 pretraining data and the underlying Guardian articles cannot be
  ruled out, so this is not a language-model contamination audit.
- OneStop is not used for subsequent tuning regardless of the outcome.

## Decision and next boundary

- Keep frozen GPT-2 causal surprisal as the English text-model contribution.
- Do not revive entropy or add OneStop-specific features.
- Treat the confirmed contribution as small and complementary to real gaze
  evidence.
- The next production artifact must have a complete training-data and code
  manifest, be trained without OneStop, and freeze its fusion evaluation before
  viewing a held-out result.
- Preserve OneStop Run 001 as an immutable confirmation artifact rather than a
  new development benchmark.
