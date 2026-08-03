# GECO Generalization v1.1 Execution Log

Execution date: 2026-08-03

Branch: `codex/feat/runtime-resource-guardrails`

Compute policy: CPU-only (`LEXIGAZE_DEVICE=cpu`, empty `CUDA_VISIBLE_DEVICES`)

## Chronology and protocol integrity

1. Commit `2f5b7aea02a30b0553c2269189b15555fd7b43ae` locked protocol v1, the implementation, fixed folds, fixed Ridge alpha, one primary endpoint, and its tests before a full population result was computed.
2. The first v1 run stopped on the first trial because `fixations.csv` contains repeated word IDs. It produced no fitted model or metric.
3. Commit `afadb779d01aa8287e3415a252088f98f272833c` locked v1.1 before results. Its only protocol amendment sums repeated positive fixation durations per word with `min_count=1`; all splits, features, models, metrics, and tests remained unchanged.
4. The first complete v1.1 run used source commit `afadb77`, completed in 123.68 seconds, and produced summary SHA-256 `c5ba96b109b75bb43a290e4bfeb20ea136b989761b1aa714675282183df602e4`.
5. A report-only Windows newline defect was found after the run. Commit `eb3795dba39493c1c552dcaa0120fa8d1bd097c7` changed only table-row newline composition.
6. The complete protocol was rerun from a clean `eb3795d` tree in 44.39 seconds. Fold metrics, participant metrics, dataset fingerprint, and plot were byte-identical to the first complete run. The JSON summary was identical after removing timestamps and runtime.

No parameter, exclusion, split, feature, or primary outcome was changed after results were observed.

## Population and provenance

- 37 participants: 18 L1 and 19 L2
- 5,892 participant-trial pairs
- 638,408 word observations
- 492,539 positive-duration observations
- 11,784 source CSV files, 49,641,749 bytes
- Dataset tree SHA-256: `ab15ba2f6b0974672445225a418f62421577c8ad17c54caa0ddbc0cb9e79029c`
- Final summary SHA-256: `732d47fc3f326ff283cd7d839de5df9a1b53502f2648859f3ef1387ecdb454fc`
- Final manifest SHA-256: `22bf71d7af877edb2336af13b42d3fd779628d708723604b5c48aabd818591c9`

The final manifest records source commit `eb3795d`, `dirty=false`, the v1.1 protocol hash, entry-point hash, dataset tree hash, runtime packages, CPU policy, all output hashes, and the complete fixed configuration.

## Prespecified primary result

Protocol: unseen reader plus unseen trial. Each prediction excludes the test participant fold and the test trial fold from training.

| Model | Macro participant Spearman rho | Stratified participant bootstrap 95% CI | Eligibility |
| --- | ---: | ---: | --- |
| Ridge text-only | 0.1216 | [0.0926, 0.1513] | Primary |
| Surprisal only | 0.0094 | [0.0029, 0.0157] | Prespecified comparator |
| Attention only | -0.0248 | [-0.0317, -0.0179] | Prespecified comparator |
| Word length only | 0.1225 | [0.0932, 0.1522] | Prespecified comparator |
| Cognitive mass | 0.0010 | [-0.0055, 0.0071] | Exploratory; upstream target-provenance risk |

- Ridge versus zero, participant sign-flip: `p=0.000010`
- Ridge minus surprisal: `Δrho=0.1122`, paired sign-flip `p=0.000010`

The statistically positive Ridge result is modest and does not beat the simple word-length rank baseline (`0.1216` versus `0.1225`). That comparison was not the sole primary hypothesis, so it is interpreted descriptively rather than promoted to a new confirmatory claim. The current cached surprisal and attention signals do not demonstrate meaningful incremental rank prediction beyond orthographic length under the strict double holdout.

`cognitive_mass` is not eligible for a generalization claim because its extraction path can use XGBoost/Ridge models trained on GECO total reading time. Its near-zero double-holdout correlation is reported rather than hidden.

## Secondary known-passage/new-reader result

The held-out reader's own outcome was subtracted from every population prior.

- Other-reader conditional duration prior: macro `rho=0.3105`, 95% CI `[0.3002, 0.3212]`, mean log-time MAE `0.4813`.
- Other-reader fixation-rate prior: macro ROC AUC `0.7766`, 95% CI `[0.7647, 0.7888]`, mean Brier score `0.1441`.

This is materially stronger than text-only prediction but applies only when the passage has already been observed in other readers. It supports a product direction based on privacy-preserving item priors or a short reader calibration phase, not a claim of zero-shot passage understanding.

## Decisions from this run

1. Freeze GECO v1.1 as a test protocol. Do not tune feature weights, Ridge alpha, or exclusions against these results.
2. Treat the prior pp01 / Trial 5 fusion correlations as descriptive calibration because the simulated dwell input is derived from the same TRT target.
3. Develop future feature changes on a separate development partition or corpus; the next confirmatory check should be a preregistered corpus-level zero-shot evaluation, such as PROVO.
4. Keep word length as a mandatory baseline for every cognitive-difficulty experiment.
5. Prefer population priors only for the explicitly scoped known-passage use case and always subtract the active reader from aggregates.

## Artifacts

- Protocol v1: `docs/GECO_GENERALIZATION_PROTOCOL_2026-08-03.md`
- Protocol v1.1 amendment: `docs/GECO_GENERALIZATION_PROTOCOL_V1_1_2026-08-03.md`
- Human-readable report: `output/geco_generalization_report.md`
- Machine-readable summary: `output/geco_generalization_summary.json`
- Reproducibility manifest: `output/geco_generalization_manifest.json`
- Participant metrics: `output/geco_generalization_subject_metrics.csv`
- Fold metrics: `output/geco_generalization_fold_metrics.csv`
- Dataset fingerprint: `output/geco_population_fingerprint.json`
- Figure: `output/geco_generalization.png`
