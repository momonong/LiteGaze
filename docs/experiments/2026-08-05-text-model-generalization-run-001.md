# Text-Model Generalization and Stable Cognitive Scoring - Run 001

- Date: 2026-08-05
- Branch: `experiment/text-model-generalization`
- Starting protocol commit: `48d2292`
- Final implementation commit before this report: `f9002a3`
- Compute policy: CPU-only model extraction and statistics; no new GPU workload
- Final status: causal GPT-2 surprisal replicated; entropy gate failed; product score stability fixed

## 1. Outcome

The experiment achieved its development objective without selecting against a
question-answer benchmark:

1. Frozen GPT-2-small causal surprisal added a small, consistent held-out signal
   beyond lexical controls on both Provo and English GECO L2.
2. The gain was positive in all five outer passage folds for total reading time,
   gaze duration, and first-fixation duration on both corpora.
3. Shannon and Rényi entropy did not pass the preregistered incremental gate for
   total reading time on either corpus. No larger entropy path is promoted.
4. Target-shuffle sentinels were negative, rather than reproducing the claimed
   effect.
5. The product's request-local min-max post-processing, not the XGBoost model,
   caused large score changes when unrelated text was appended. Fixed training-
   range calibration reduced the observed prefix `load_score` delta to zero.

This is strong development/replication evidence, not a final scientific
confirmation. Both corpora have historical project exposure, so a third corpus
must be frozen before outcome inspection for a final claim.

## 2. What changed

### 2.1 Runtime and metric correctness

- Added an explicit metric contract distinguishing causal surprisal from masked
  pseudo-surprisal, including units, context direction, and subtoken aggregation.
- Made cognition device selection explicit through `device`,
  `LEXIGAZE_COGNITION_DEVICE`, or `LEXIGAZE_DEVICE`.
- Explicit CPU mode does not probe CUDA.
- Removed the import-time CUDA/cuDNN side effect.
- Fixed causal GPT chunking to retain 64 words of left context, copy only new
  words, preserve Rényi entropy, and reject an individually oversized word.
- Configured the GPT-2 tokenizer for split-word alignment with prefix spaces.

### 2.2 Leakage-resistant benchmark

- Added deterministic, approximately balanced complete-group folds.
- Added standardized Ridge with all feature statistics fitted on training folds
  only.
- Added cross-fitted predictions, participant/text Spearman summaries, paired
  bootstrap confidence intervals, and a shuffled-training-target sentinel.
- Added frozen, hash-checked Provo and GECO L2 runners with artifact manifests.
- Kept all language models frozen and used no hyperparameter search.

### 2.3 Product stability

- Replaced request-local min-max scaling for XGBoost and Ridge predictions with
  the regressors' frozen 50-3000 ms GECO training range.
  - XGBoost is calibrated in `log(TRT_ms)` space.
  - Ridge is calibrated in millisecond space.
- Added a single model policy: English uses causal GPT-2; Chinese and Dutch use
  BERT. This fixed English fallback routes that had hard-coded BERT while the
  main cognition route and training pipeline used GPT-2.
- Made `core.cognition` exports lazy so importing the lightweight policy or
  calibration modules does not import Torch or initialize model runtime.
- Added the no-Torch policy/calibration tests to the local offline quality gate.

### 2.4 GitHub Actions

- Commit `bd14196` renamed
  `.github/workflows/offline-cpu-quality-gate.yml` to
  `.github/workflows/offline-cpu-quality-gate.yml.disabled`.
- There are zero active `.yml`/`.yaml` files under `.github/workflows` on this
  branch.
- The workflow content is preserved and can be re-enabled by restoring the
  `.yml` suffix. The disablement becomes the default-branch state only after
  merge.

## 3. Frozen protocol

### 3.1 Data

| Corpus | Source identity | Participants | Texts | Items | Evaluation rows |
| --- | --- | ---: | ---: | ---: | ---: |
| Provo | SHA-256 `38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7` | 84 | 55 | 2,743 | 150,114 |
| GECO L2 English | SHA-256 `cd551640cfd122b5e360d70c12998798125e7485c6deb900bc71b8e591c59b87` | 19 | 588 | 56,411 | 359,320 |

GECO input guards require exactly bilingual readers and English stimuli. The
runner never reads legacy prepared `surprisal_score`, `attention_score`, or
`cognitive_mass` fields. This matters because an older prepared GECO `L1`
directory contains Dutch stimuli and had previously been treated as an English
training path in project reports.

### 3.2 Models

- `word_length_only`: log character length.
- `M0 lexical`: current/previous length and Zipf frequency, relative position,
  capitalization, punctuation, and subtoken count.
- `M1 causal surprisal`: M0 plus current and previous GPT-2 causal surprisal.
- `M2 entropy`: M1 plus current/previous Shannon and Rényi entropy.
- `target_shuffle_sentinel`: M2 features fitted to a deterministically shuffled
  training target.

All models used Ridge alpha 1.0 and the same five complete-passage outer folds.
The first item of each passage was excluded because a causal model has no left
context for it. Positive durations were transformed with `log1p`.

### 3.3 Guardrails

- No QA, cloze, LSA, or Provo predictability field was read.
- No gaze coordinate or reading-time outcome was used as a predictor.
- No language-model fine-tuning or model-size search was performed.
- Scalers and Ridge weights were fitted on training passages only.
- Raw causal metrics were cached before any outcome evaluation.
- Provo set the protocol; GECO L2 reused the same features, folds, alpha, and
  decision gate before its outcomes were inspected in this run.

## 4. Data-quality incidents and invalidated diagnostics

Two early runs were deliberately invalidated before conclusions were accepted.

### 4.1 Provo display-padding whitespace

Provo `IA_LABEL` values contained trailing EyeLink display padding. Without
stripping it, GPT-2 interpreted the padding as an extra token: 2,312 of 2,743
items appeared to have exactly two subtokens. The diagnostic output was archived
locally as `provo-gpt2-small-run-001-untrimmed-labels`.

After NFKC normalization and stripping, 2,313 items were single-token, 368 were
two-token, and the maximum was five. The formal result was regenerated from
clean commit `9a85c9a`.

### 4.2 GECO punctuation mojibake

Five of 56,411 unique GECO L2 labels contained CP850-rendered UTF-8 punctuation:
three en dashes, one opening smart quote, and one closing smart quote. A narrow,
code-point-specific repair restored only those patterns and rejects residual
`ÔÇ` or Unicode replacement characters.

The first GECO output was archived locally as
`geco-l2-gpt2-small-run-001-mojibake-labels`. The formal result was regenerated
from clean commit `c151103`. Row counts were identical and all primary effect
changes were below approximately 0.00015.

## 5. Results

The table reports macro participant/text Spearman rho. Deltas are paired macro
rho differences for M1 minus M0 with 95% block-bootstrap confidence intervals.

### 5.1 Provo

| Outcome | M0 participant / text | M1 participant / text | M1-M0 participant delta | M1-M0 text delta | Positive folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total reading time | 0.3080 / 0.6918 | 0.3200 / 0.7219 | +0.0121 [0.0097, 0.0144] | +0.0301 [0.0196, 0.0408] | 5/5 |
| Gaze duration | 0.2599 / 0.6476 | 0.2654 / 0.6639 | +0.0055 [0.0041, 0.0068] | +0.0163 [0.0099, 0.0230] | 5/5 |
| First fixation | 0.1212 / 0.3917 | 0.1284 / 0.4126 | +0.0072 [0.0052, 0.0093] | +0.0209 [0.0096, 0.0324] | 5/5 |

For total reading time, M2 minus M1 was -0.0002 participant and -0.0008
text macro rho, with both confidence intervals crossing zero and only 1/5
positive folds. The TRT shuffle sentinel was -0.0387 participant and -0.0889
text rho.

### 5.2 GECO L2 English

| Outcome | M0 participant / text | M1 participant / text | M1-M0 participant delta | M1-M0 text delta | Positive folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total reading time | 0.2914 / 0.4815 | 0.3035 / 0.5007 | +0.0121 [0.0090, 0.0151] | +0.0191 [0.0166, 0.0216] | 5/5 |
| Gaze duration | 0.2472 / 0.4079 | 0.2533 / 0.4184 | +0.0061 [0.0048, 0.0074] | +0.0105 [0.0087, 0.0124] | 5/5 |
| First fixation | 0.1044 / 0.1800 | 0.1114 / 0.1942 | +0.0069 [0.0045, 0.0092] | +0.0142 [0.0108, 0.0175] | 5/5 |

For total reading time, M2 minus M1 was effectively zero (-0.00004 participant,
-0.00034 text), both intervals crossed zero, and only 2/5 folds were positive.
The TRT shuffle sentinel was -0.0301 participant and -0.0457 text rho.

GECO first fixation had a tiny positive text-level M2 delta, but its participant
interval crossed zero and it was not the primary entropy gate. This is a future
hypothesis, not evidence to promote entropy now.

### 5.3 Interpretation

- Causal surprisal has reproducible incremental information, but the effect is
  modest. It should augment lexical/gaze evidence rather than dominate it.
- Entropy is not justified as added product complexity by this run.
- The existing XGBoost artifact still contains a Rényi input for backward
  compatibility. It was not retrained or silently rewritten; a future artifact
  must use a clean ablation and provenance manifest before that input is removed.
- Masked BERT values retain their explicit `masked_pseudo` contract and must not
  be pooled with causal scores or cited as this result.

## 6. Product stability result

Probe text:

`The careful reader examined the ordinary report.`

The probe was rerun after appending an unrelated extreme word. GPT-2/XGBoost raw
predictions for the prefix were already identical; only normalization moved them.

| Version | Aligned prefix words | Changed scores | Mean absolute delta | Maximum absolute delta |
| --- | ---: | ---: | ---: | ---: |
| Request-local min-max | 8 | 6 | 0.228625 | 0.4637 |
| Fixed training-range calibration | 8 | 0 | 0.000000 | 0.0000 |

`load_score` is now appropriate for cross-request fusion. `load_level` remains a
document-relative summary because `_finalize_load_levels` uses a document
percentile; one label changed in the probe even though all scores were stable.
Consumers needing cross-session consistency must use the continuous score.

## 7. Compute and verification

### 7.1 GPU budget

| Run | Runtime | Device | GPU before | GPU after |
| --- | ---: | --- | --- | --- |
| Provo formal Run 001 | 18.52 s | CPU | 0%, 113 MiB | 0%, 113 MiB |
| GECO L2 formal Run 001 | 153.27 s | CPU | 0%, 195 MiB | 0%, 195 MiB |

No Python process appeared in the GPU compute-process list during the GECO run.
The earlier diagnostic GECO run changed desktop baseline from 188 to 195 MiB but
remained at 0% utilization and had no Python GPU process. Model extraction was
therefore kept on CPU; no overnight GPU allocation was needed.

### 7.2 Tests

- Text semantics/generalization/GECO schema suite: 13/13 passed before product
  integration.
- Focused metric, stability, and fusion suite after integration: 17/17 passed.
- Local offline CPU gate: 62/62 passed in 1.083 seconds.
  - `CUDA_VISIBLE_DEVICES=-1`
  - Torch not imported
  - GPU snapshot unchanged (174 MiB before/after in the final gate)
  - no network/process attempt
  - no artifact change
- Real GPT-2 + XGBoost CPU prefix probe: maximum score delta 0.0.
- `compileall`, Node syntax check, and `git diff --check` passed.
- Ruff was not installed in the active virtual environment, so no Ruff result is
  claimed.

## 8. Artifact provenance

Formal raw artifacts are intentionally ignored because cross-fitted predictions
are large. They remain reproducible from the tracked runners.

| Corpus | Feature SHA-256 | Source commit | Local manifest |
| --- | --- | --- | --- |
| Provo | `60e9af64f75a10bfb0ca8b1587a3e7bfa2290403428ea67c5069f92041107956` | `9a85c9a` | `data/provo/text_modeling/provo-gpt2-small-run-001/manifest.json` |
| GECO L2 | `8b952ce5951a1ad1e7a84f2be55281a48f6696a8089795214320e46fcfb0abe9` | `c151103` | `data/geco/text_modeling/geco-l2-gpt2-small-run-001/manifest.json` |

Tracked machine-readable summary:
[`results/2026-08-05-text-model-generalization-run-001.json`](results/2026-08-05-text-model-generalization-run-001.json).

## 9. Decisions

### Adopted now

- Causal GPT-2 as the consistent English default across cognition, fusion
  fallback, gaze-video analysis, and the offline orchestrator.
- Explicit metric contracts and device policy.
- Context-preserving GPT chunking.
- Fixed cross-request calibration for committed XGBoost/Ridge artifacts.
- Local, no-Torch quality gates as the current automated source of truth.

### Not adopted

- No LM fine-tuning.
- No larger model or hidden-state GPU extraction.
- No entropy promotion.
- No QA-based model selection.
- No claim that Provo or GECO is untouched final confirmation.
- No automatic removal/retraining of the provenance-poor XGBoost artifact.

## 10. Next frozen objective

Before inspecting another corpus's reading-time outcomes:

1. choose one independent English eye-tracking corpus and record its checksum,
   inclusion rules, outcomes, and complete-document split;
2. freeze M0/M1 and the current decision gate without adding features;
3. run label-free GPT-2 extraction once, CPU-first;
4. require same-direction M1-M0 evidence and a failed shuffle sentinel;
5. only after confirmation, retrain a provenance-complete production artifact
   and evaluate text-only, gaze-only, and combined fusion on the same groups.

Hardware work can proceed in parallel, but the new two-camera/phone path should
be evaluated on independent capture groups and must not use these text corpora to
tune gaze calibration.
