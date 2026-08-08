# Non-Chinese-Source Text Backbone Development Screen — Run 001

- Completed: 2026-08-08
- Branch: `research/non-cn-text-backbone-benchmark`
- Frozen protocol: `non-cn-text-backbone-provo-development-v1`
- Decision: **shortlist only `pythia_410m_deduped_full` for external replication**
- Product decision: keep GPT-2 frozen; no production model change

## What this run tested

Five immutable base-model identities were fixed before any new-backbone outcome
was read: GPT-2, two checkpoints of Pythia-410M-deduped, SmolLM2-360M, and OLMo
2 1B. The exact allowlist excludes Qwen and other disallowed source prefixes,
disables remote code, and permits only public ungated official repositories.

Every successful backbone produced left-only subtoken-sum surprisal. A shared
lexical baseline was extended first with the candidate tokenizer's subtoken
count and then with current/previous surprisal. Five complete Provo passages
folds were held out; all scaling and Ridge fitting occurred on training passages
only. No question, answer, or QA correctness was used.

Provo is historically inspected, so all findings below are development evidence
and cannot confirm a replacement.

## Primary result: total reading time

| Backbone | M1 participant rho | M1 text rho | M1-M0 participant delta [95% CI] | M1-M0 text delta [95% CI] | Positive folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | 0.3197 | 0.7199 | +0.0112 [+0.0089, +0.0135] | +0.0267 [+0.0167, +0.0370] | 5/5 |
| `pythia_410m_deduped_step1000` | 0.3193 | 0.7175 | +0.0103 [+0.0083, +0.0124] | +0.0241 [+0.0154, +0.0326] | 5/5 |
| `pythia_410m_deduped_full` | 0.3205 | 0.7238 | +0.0116 [+0.0091, +0.0140] | +0.0303 [+0.0200, +0.0406] | 5/5 |

All three causal backbones added a small, consistent signal beyond lexical and
tokenization controls. This reinforces the existing conclusion that surprisal
is useful as an auxiliary population-level feature rather than a dominant
personalized predictor.

## Challenger comparison

Relative to GPT-2:

| Challenger | Participant delta [95% CI] | Text delta [95% CI] | Positive folds | Frozen eligibility |
| --- | ---: | ---: | ---: | ---: |
| Pythia step1000 | -0.0004 [-0.0017, +0.0009] | -0.0024 [-0.0107, +0.0060] | 3/5 | No |
| Pythia full | +0.0008 [+0.0000, +0.0016] | +0.0039 [-0.0006, +0.0088] | 3/5 | Yes |

The full Pythia checkpoint passed the predeclared shortlist rule because both
point differences were nonnegative and its own surprisal increment passed the
gate. The advantage is nevertheless weak: the text interval includes zero and
only three of five folds favored it. This supports replication, not replacement.

The planned `step1000 - full` contrast was participant -0.0012
[-0.0026, +0.0001] and text -0.0063 [-0.0145, +0.0016], with 1/5 positive
folds. The direction favors full training, but neither interval excludes zero;
the run does not establish a training-token effect.

## Frozen technical failures

SmolLM2 and OLMo 2 both emitted a separator-only tokenizer offset at the same
display position. The v1 contract requires every token to overlap exactly one
displayed word, so both were stopped and recorded before outcome access. They
were not silently repaired or rerun. This is an adapter compatibility result,
not evidence that their language signal is inferior.

Any retry must use a separately frozen v2 rule for separator-only tokens and
must remain development-only.

## GPU and reproducibility

- Successful LM forward time: 31.04 seconds, or 0.00862 GPU-hours.
- Frozen budget: 1.5 GPU-hours; 0.57% consumed.
- Peak reserved memory: GPT-2 0.371 GiB; each Pythia run 0.924 GiB.
- One model was resident at a time; no training or fine-tuning occurred.
- Throughput was not a selection metric because first-run download and kernel
  warm-up conditions differed across the separate processes.
- OneStop was not accessed.

Machine-readable results are in
[`results/2026-08-08-non-cn-text-backbone-provo-dev-run-001.json`](results/2026-08-08-non-cn-text-backbone-provo-dev-run-001.json).

## Decision and next gate

GPT-2 remains the production/frozen baseline. Only full Pythia-410M-deduped may
advance to a newly preregistered GECO L2 comparison. The Pythia result must be
evaluated unchanged on GECO before it can be discussed as cross-corpus evidence,
and later requires the independent Reader Assessment v3 word-review outcome for
personalized utility.
