# ADR 0005: Retain GPT-2 After the Non-Chinese-Source Backbone Screen

- Status: Accepted
- Date: 2026-08-08

## Context

LexiGaze uses frozen causal surprisal as one auxiliary text feature. GPT-2 small
has shown small positive increments beyond lexical controls on Provo, GECO L2,
and OneStop, but it was unclear whether a newer or larger base language model
would provide a more human-relevant signal.

Model provenance is also an explicit constraint. Qwen and other disallowed
Chinese-source model families may not enter the candidate or production path.
Instruction/chat models and participant-outcome fine-tuning are unsuitable for
the clean left-context probability contract and create avoidable overfit risk.

A frozen Provo development screen compared GPT-2 with two Pythia-410M training
checkpoints. SmolLM2 and OLMo 2 were outcome-blind technical failures under the
strict v1 offset contract and were not silently repaired. Full Pythia was the
only Provo shortlist candidate. A separately frozen GECO L2 replication then
compared only full Pythia and GPT-2 under an outcome-blind v2 separator rule.

## Decision

Retain GPT-2 small as the frozen production text backbone.

- Full Pythia-410M remains available only as a research comparator.
- Do not adopt an instruction/chat LLM for word surprisal or direct user-ability
  scoring.
- Do not reopen OneStop or add more public reading-time corpora to search for a
  favorable backbone result.
- Any future production challenger must first improve the independent Reader
  Assessment v3 word-review outcome under frozen participant and passage-family
  holdouts, calibration checks, and an explicit compute budget.
- Model loading must use an exact approved-source allowlist, immutable revisions
  in experiments, disabled remote code, and recorded provenance.
- The obsolete `uer/gpt2-chinese-cluecorpussmall` runtime mapping is removed;
  unsupported causal-model/language pairs fail before any model download.

## Rationale

On GECO total reading time, full Pythia had participant delta +0.00042 and text
delta +0.00148 relative to GPT-2. The participant confidence interval included
zero and only 3/5 passage folds favored Pythia, so it failed the preregistered
directional replication gate. The difference is too small and inconsistent to
justify tripling the parameter count or changing a stable artifact.

Both models retained positive incremental surprisal effects. The result rejects
a backbone replacement, not causal surprisal as an auxiliary feature.

## Consequences

- Existing GPT-2 artifacts remain comparable with prior Provo, GECO, and
  OneStop evidence.
- Pythia does not enter fusion or production merely because its average metric
  is fractionally higher.
- Engineering work shifts from public-corpus backbone selection toward the
  independent personalized word-review target and calibrated fusion.
- Separator-only tokenizer handling remains available as a governed adapter
  behavior, with reject-by-default semantics for protocols that do not opt in.
