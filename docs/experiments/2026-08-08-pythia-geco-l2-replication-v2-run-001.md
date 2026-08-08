# Pythia-410M Full GECO L2 Replication v2 — Run 001

- Completed: 2026-08-08
- Branch: `research/non-cn-text-backbone-benchmark`
- Frozen v2 protocol: `pythia-full-geco-l2-replication-v2`
- Decision: **`does_not_replicate_over_gpt2`**
- Product decision: retain frozen GPT-2

## Outcome-blind v1 invalidation and v2 repair

v1 stopped before reading outcomes because GPT-2 produced one standalone ASCII
space token in `Text_ID 4:33`. The complete tokenizer-only diagnostic found
exactly one such GPT-2 separator token across 588 texts and zero for Pythia.

v2 was frozen before outcome access. It assigns a zero-overlap token to the
following word only when the source substring is entirely Unicode whitespace
inside the exact adjacent-word gap. All other ambiguity remains fail-closed.
Both backbones were recomputed from scratch; no v1 feature was reused.

## Primary result: total reading time

| Backbone | M1 participant rho | M1 text rho | M1-M0 participant delta [95% CI] | M1-M0 text delta [95% CI] | Positive folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | 0.3037 | 0.5011 | +0.0124 [+0.0093, +0.0154] | +0.0196 [+0.0172, +0.0221] | 5/5 |
| `pythia_410m_deduped_full` | 0.3042 | 0.5025 | +0.0129 [+0.0091, +0.0166] | +0.0214 [+0.0187, +0.0240] | 5/5 |

Both frozen language models again add a small, reproducible signal beyond the
lexical and candidate-tokenization baseline. Pythia is therefore a valid
research feature source, but that is not the same question as whether it should
replace GPT-2.

## Frozen Pythia-minus-GPT-2 contrast

- Participant delta: **+0.00042** [-0.00037, +0.00123], 19 paired readers.
- Text delta: **+0.00148** [+0.00029, +0.00268], 588 paired texts.
- Fold differences: +0.00106, +0.00134, -0.00069, -0.00084, +0.00125.
- Positive folds: **3/5**.

Pythia passed its own M1-M0 incremental gate, and its average values were
slightly higher. However, the participant interval includes zero and only three
folds favored it. It therefore failed both the frozen strong and directional
replication gates. The small Provo advantage did not replicate robustly enough
to justify a backbone change.

The secondary gaze-duration and first-fixation M1 point estimates were also
nearly indistinguishable. They were descriptive and cannot rescue the failed
primary decision.

## Evaluation incident

The first evaluation attempt opened the frozen outcome table but stopped before
producing a result because the shared evaluator expected the incremental gate
under a different JSON key. A schema-only bridge was added and tested: it passes
the exact preregistered GECO gate object without altering data, features, folds,
alpha, outcomes, thresholds, or decision logic. Commit `f780bff` records the
repair; the successful run used that clean commit.

## GPU and provenance

- v2 forward time: 43.65 seconds, or 0.01212 GPU-hours.
- v2 budget: 1.0 GPU-hour; about 1.21% consumed.
- Peak reserved memory: GPT-2 0.512 GiB; Pythia 1.057 GiB.
- One model was resident at a time; no training or fine-tuning occurred.
- Exact official revisions and the non-Chinese-source allowlist were enforced.
- No prepared GECO surprisal, attention, or cognitive-mass feature was used.
- No QA data or OneStop data was accessed.

## Final validation

- Text-backbone, GECO replication, and runtime source-policy tests: 24/24
  passed on the CPU validation path.
- Repository offline quality gate: 151/151 passed with zero failures, errors,
  skips, or unexpected successes.
- The offline gate imported no Torch, blocked network and child-process access,
  changed no tracked artifact scope, and left GPU memory unchanged at 742 MiB.

Machine-readable results are in
[`results/2026-08-08-pythia-geco-l2-replication-v2-run-001.json`](results/2026-08-08-pythia-geco-l2-replication-v2-run-001.json).

## Decision

Keep GPT-2 as the frozen production text backbone. Retain full Pythia-410M only
as a research comparator and do not open another public corpus merely to seek a
favorable result. The next meaningful text-model evidence must use the
independent Reader Assessment v3 `no_review / unsure / review_needed` outcome,
with participant and passage-family holdouts fixed before fitting.
