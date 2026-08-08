# Non-Chinese-Source Text Backbone Development Screen — Preregistration v1

- Frozen: 2026-08-08 (Asia/Taipei)
- Branch: `research/non-cn-text-backbone-benchmark`
- Status: new-backbone outcome blind
- Corpus role: development only; never final confirmation

## Decision being tested

GPT-2 small remains the immutable historical baseline. This run asks whether a
fixed set of non-Chinese-source, base causal language models provides a more
useful frozen word-level signal. It does not assume that a newer, larger, or
lower-perplexity model is a better model of human reading.

No result from this screen can change the production model. At most two
challengers may be shortlisted for a separately frozen GECO L2 replication.
The independent Reader Assessment v3 word-review cohort remains the required
confirmation for personalized utility.

## Exact source allowlist

Only the following public, ungated official repositories and immutable revisions
may be downloaded. `trust_remote_code` is disabled and only pretrained/base
weights are permitted.

| Key | Official model | Immutable revision | Developer | Role |
| --- | --- | --- | --- | --- |
| `gpt2` | `gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | OpenAI | Historical baseline |
| `pythia_410m_deduped_step1000` | `EleutherAI/pythia-410m-deduped` | `a3b3aff9a656ab34fec3474eb60bb5b487639539` | EleutherAI | Approx. 2.1B-token checkpoint |
| `pythia_410m_deduped_full` | `EleutherAI/pythia-410m-deduped` | `c4fc8d586d62df497f1f9b69d66d3ca419992d3e` | EleutherAI | Same-architecture full-training control |
| `smollm2_360m` | `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | Hugging Face | Compact modern English challenger |
| `olmo2_1b` | `allenai/OLMo-2-0425-1B` | `a1847dff35000b4271fa70afc5db10fd29fedbdf` | Ai2 | Fully open modern challenger |

Model IDs beginning with `Qwen/`, `deepseek-ai/`, `THUDM/`, `baichuan-inc/`,
`01-ai/`, or `uer/`, and derivatives of those bases, are excluded. This is a
model-supply-chain constraint, not an empirical claim about model quality.

## Outcome and overfit boundary

The frozen Provo source is already historically inspected, including known
GPT-2 results. It is therefore limited to development evidence. No question,
answer, QA correctness, or fixed question set is used.

- Primary outcome: positive total reading time, transformed with `log1p`.
- Secondary descriptive outcomes: gaze duration and first-fixation duration.
- Outer split: five deterministic complete-passage (`Text_ID`) folds.
- Every scaler and Ridge model fits on training passages only.
- The first displayed word is excluded because it has no left context.
- A shuffled-training-target sentinel must remain visible in the report.

## Frozen features

All backbones use the same token-to-word contract: no special tokens, left-only
context, negative log probability in nats, and sum across a word's subtokens.

1. `M0 lexical shared`: current and previous length/frequency plus position,
   capitalization, and punctuation.
2. `M0 tokenization`: M0 plus that backbone's subtoken count.
3. `M1 causal surprisal`: M0 tokenization plus current and previous surprisal.

The primary within-backbone comparison is M1 minus M0 tokenization. This
separates contextual probability from a tokenizer merely splitting rare words
into more pieces. Cross-backbone comparisons use paired held-out M1 predictions
under identical folds, alpha, outcomes, and preprocessing.

One planned mechanistic contrast compares the Pythia `step1000` M1 predictions
against the fully trained Pythia M1 predictions. Because architecture,
tokenizer, and data family are held constant, this tests the preregistered
hypothesis that substantially more language-model training need not improve fit
to human reading.

## Smoke, GPU, and stopping rules

Three passages are scored before a full run. A backbone advances only with
complete word coverage, finite context-scored features, at least one token per
word, and peak CUDA reserved memory below 12 GiB. A failure is recorded and is
not repaired after its outcome has been evaluated.

The run uses one frozen model at a time in BF16, performs no training or
fine-tuning, caches label-free features, and records load time, inference time,
throughput, allocated memory, and reserved memory. Total inference is capped at
1.5 GPU-hours. OneStop is not accessed.

## Interpretation boundary

A challenger passes the development incremental gate only when the primary
M1-M0-tokenization paired bootstrap lower bound is above zero for both
participants and texts and at least four of five folds are positive. Because
multiple candidates are screened on a known corpus, this is not confirmation.
To proceed, a challenger must also have nonnegative participant and text point
differences relative to GPT-2 under the held-out M1 model. Eligible challengers
are ranked by the smaller of those two differences, with the backbone key as a
fixed tie-breaker, and at most two can proceed. An unfavorable result is
reported without changing the model list, checkpoint, feature set, fold, or
threshold.

The complete executable contract is
[`protocols/2026-08-08-non-cn-text-backbone-provo-dev-v1.json`](protocols/2026-08-08-non-cn-text-backbone-provo-dev-v1.json).
