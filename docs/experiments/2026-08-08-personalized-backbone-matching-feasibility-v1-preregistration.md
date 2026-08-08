# Personalized Backbone Matching Feasibility v1 — Preregistration

- Frozen: 2026-08-08 (Asia/Taipei)
- Branch: `research/personalized-backbone-mixture-v1`
- Role: exploratory feasibility, not production confirmation
- Compute for the first gate: CPU only, zero GPU inference

## Question

Can a short calibration from a previously unseen reader tell us whether GPT-2
or Pythia better ranks that reader's word-level total reading time on completely
unseen texts?

The phrase “matching a model to a user” has a narrow operational meaning here:
choose the frozen backbone whose causal-surprisal features better align with the
reader's observed word-level timing ranks. It is **not** a cognitive, language
proficiency, CEFR, attention, fatigue, or personality classification.

## Why this is nested

Each of five outer folds holds out complete `Text_ID` passage units. Within each
outer fold, one of 19 participants is also held out from model fitting:

1. Fit each backbone's identical Ridge model on other participants and only
   non-evaluation texts.
2. Use a deterministic subset of the held-out participant's non-evaluation
   texts for calibration and model selection.
3. Evaluate the selected backbone on that participant's outer-fold texts.

Consequently, the target participant contributes zero fitting rows, and the
evaluation passage family contributes zero fitting or calibration rows. This
prevents the same participant data or passage family from both selecting and
testing a model.

## Realistic calibration budget and abstention

The primary budget is ten GECO texts. The frozen cache has a median of 96 words
per text, so the primary setting represents roughly 960 observed words rather
than hundreds of calibration passages. Five, 20, 40, 80, and all available
non-evaluation texts are secondary learning-curve points.

Selection is uncertainty-aware. For each calibration text, compute the
difference between Pythia and GPT-2 Spearman correlations. Select Pythia only
when a 2,000-sample paired bootstrap has a 95% lower bound strictly above zero;
otherwise abstain to GPT-2. A plain calibration argmax and an evaluation oracle
are descriptive diagnostics, not deployment rules.

## Frozen first-stage gate

At the ten-text budget, all of these must hold:

- the participant-bootstrap 95% lower bound for selected minus GPT-2 is above
  zero;
- at least four of five outer folds have a positive macro difference;
- at least 60% of participants improve on average;
- Pythia is selected in at least 5% of participant-fold cells;
- all participant and passage leakage checks pass.

If any clause fails, no additional backbone is downloaded or run. This protects
the GPU budget and avoids increasing candidate-selection variance when the
basic personalized-selection premise has not generalized.

## Conditional model bank

Only after a pass may the exact pinned, public, ungated, base-model bank be
expanded with SmolLM2-360M, OLMo-2-0425-1B, and IBM Granite 3.1 2B Base. New
feature extraction remains label-free, uses one model at a time, may consume at
most 0.25 GPU-hours and 12 GiB peak reserved VRAM, and cannot change production
without a newly collected confirmation cohort.

BLOOM-560M is excluded because its RAIL model card lists evaluating or scoring
individuals as out of scope. Gated Llama 3.2 1B and Gemma 3 1B are excluded from
this reproducible public-ungated bank. The source-prefix exclusions requested
for this project remain active, including Qwen, DeepSeek, THUDM, Baichuan,
01.AI, and UER.

## Integrity boundary

GECO aggregate GPT-2/Pythia results were already inspected in the parent
experiment. Therefore this analysis is explicitly exploratory even though the
nested personalized cells, calibration budgets, selector, and gate are frozen
before they are computed. Favorable evidence must be confirmed on newly
collected participants and unseen passage families. No question-answer dataset,
LM fine-tuning, or fixed-question tuning is permitted.

The complete machine-readable contract is
[`protocols/2026-08-08-personalized-backbone-matching-feasibility-v1.json`](protocols/2026-08-08-personalized-backbone-matching-feasibility-v1.json).
