# Production Text Artifact - Run 001

- Date: 2026-08-05T17:32:44.652332+00:00
- Protocol: `production-text-fusion-v1`
- Compute: CPU only; verified GPT-2 feature caches reused
- Status: candidate; the production default was not changed

## Outcome

A provenance-complete M1 text artifact was built from PROVO and English GECO L2 with equal total corpus weight. OneStop and all QA outcomes were excluded. This pooled development fit is an implementation diagnostic, not a new confirmation experiment.

| Corpus | Rows | Participant M1-M0 | 95% CI | Text M1-M0 | 95% CI | Positive folds |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| GECO_L2_English | 359314 | +0.0113 | [+0.0081, +0.0142] | +0.0181 | [+0.0158, +0.0205] | 5/5 |
| PROVO | 150114 | +0.0112 | [+0.0089, +0.0136] | +0.0297 | [+0.0187, +0.0406] | 5/5 |

## Decision

The candidate remains isolated from the product default. Promotion requires a future real-data evaluation where combined fusion beats gaze-only on both capture-held-out and article-held-out groups using an independent difficulty target that is not derived from gaze, text features, or a public QA benchmark.

## Leakage and compute controls

- OneStop was not loaded, trained on, or used for threshold selection.
- Entropy features were excluded.
- No language-model fine-tuning or inference was run.
- No request-local normalization is present in the artifact.
- GPU use was disabled with `CUDA_VISIBLE_DEVICES=-1`.

## Verification

- 13 focused text/fusion tests passed.
- The complete offline CPU gate passed 77/77 tests.
- The gate imported no Torch, made no network calls, and changed no artifacts.
- GPU utilization and allocated memory were unchanged across the gate.
- Full `core`, `scripts`, and `web` compilation passed.
- Ruff and `git diff --check` passed for every changed Python file.
- The fusion CLI completed an end-to-end synthetic contract test and wrote all
  four expected audit artifacts; this checks plumbing only, not real-world gain.
