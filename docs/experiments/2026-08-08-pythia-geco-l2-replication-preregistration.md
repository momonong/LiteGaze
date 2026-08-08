# Pythia-410M Full GECO L2 Replication — Preregistration v1

- Frozen: 2026-08-08 (Asia/Taipei)
- Branch: `research/non-cn-text-backbone-benchmark`
- Status: Pythia GECO outcomes not inspected
- Parent result: Provo development screen Run 001

## Why this is the next test

The Provo screen admitted exactly one challenger: the fully trained
`EleutherAI/pythia-410m-deduped` checkpoint. Its advantage over GPT-2 was only
participant +0.0008 and text +0.0039, the text confidence interval included
zero, and three of five folds were positive. That is enough to justify one
replication, not enough to replace GPT-2.

GECO L2 is historically inspected for the project's GPT-2 baseline, so it is
not a pristine confirmation corpus. However, the Pythia features and outcomes
have not been used for challenger selection. This run is therefore a fixed
cross-corpus replication of one hypothesis, with no new model search.

## Frozen model and source identities

Only two official public, ungated base models are allowed:

| Role | Model | Immutable revision |
| --- | --- | --- |
| Baseline | `gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Challenger | `EleutherAI/pythia-410m-deduped` | `c4fc8d586d62df497f1f9b69d66d3ca419992d3e` |

The same non-Chinese-source exclusions remain active, including `Qwen/`,
`deepseek-ai/`, `THUDM/`, `baichuan-inc/`, `01-ai/`, and `uer/`. Remote code,
instruction/chat weights, derivatives, and fine-tuning are forbidden.

The frozen source is English GECO L2:

- SHA-256: `cd551640cfd122b5e360d70c12998798125e7485c6deb900bc71b8e591c59b87`
- 534,154 rows, 19 bilingual participants, 588 texts, 56,411 display items.
- No prepared surprisal, attention, cognitive-mass, question, answer, or QA
  field may be read.

## Frozen extraction and analysis

The token-to-word contract is unchanged from Provo: raw displayed text,
fast-tokenizer character offsets, no special tokens, left-only surprisal in
nats, subtoken sum, and a zero first-word boundary excluded from evaluation.
Five label-free texts must first pass complete coverage, finite features, at
least one token per word, and peak reserved CUDA memory below 12 GiB.

The feature sets are unchanged:

1. shared lexical M0;
2. M0 plus the candidate tokenizer's subtoken count;
3. M1 plus current and previous causal surprisal.

Positive reading times are transformed with `log1p`. Five deterministic
complete-text folds and Ridge alpha 1.0 are fixed. Training-fold-only scaling
and a shuffled-target sentinel remain mandatory. Total reading time is primary;
gaze duration and first-fixation duration are descriptive secondary outcomes.

## Frozen decision tiers

Pythia must first retain its own incremental M1-M0-tokenization gate: paired
participant and text bootstrap lower bounds above zero and at least four of five
positive folds.

Relative to GPT-2 on primary M1 predictions:

- **Strong replication:** participant and text 95% lower bounds above zero and
  at least 4/5 positive folds.
- **Directional replication:** both point differences above zero and at least
  4/5 positive folds, but the strong gate is not met.
- **Does not replicate:** every other result.

Secondary outcomes cannot rescue a failed primary decision. Even strong
replication cannot change production: independent Reader Assessment v3
word-review evidence is still required. OneStop cannot be reopened for another
model-selection test.

## GPU boundary

Frozen BF16 inference runs one model at a time with no training or fine-tuning.
Total forward time is capped at one GPU-hour and throughput cannot influence the
decision. Cached public weights from the Provo screen may be reused, but all
GECO word features are recomputed label-free under the frozen adapter.

The machine-readable contract is
[`protocols/2026-08-08-pythia-geco-l2-replication-v1.json`](protocols/2026-08-08-pythia-geco-l2-replication-v1.json).
