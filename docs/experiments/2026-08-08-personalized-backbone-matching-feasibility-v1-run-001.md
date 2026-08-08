# Personalized Backbone Matching Feasibility v1 — Run 001

- Completed: 2026-08-08T08:56:48.555683+00:00
- Runtime: 26.50 seconds
- Device: CPU only (GPU used: false)
- Decision: **`personalized_selection_feasibility_not_demonstrated`**
- Conditional model-bank expansion allowed: **false**
- Production backbone: `gpt2` (unchanged)

## Main finding

- With ten calibration texts, the abstaining selector chose Pythia in 5/95
  participant-fold cells and improved mean held-out Spearman by `+0.000115`
  (participant-bootstrap 95% CI `[+0.000020, +0.000234]`).
- The benefit was not broad or passage-stable: only 5/19 participants improved
  on average and only 3/5 outer folds were positive, so two frozen gate clauses
  failed.
- GPT-2's mean evaluation rho was `0.301845`; the selected gain was about
  `0.038%` of that value. Even the non-deployable evaluation oracle gained only
  `+0.001445` (`0.48%` of the GPT-2 rho), showing that this two-model bank has a
  very small practical routing ceiling.

## Design integrity

- Every evaluation cell holds out both the target participant and complete evaluation `Text_ID` passage units from fitting.
- Model selection uses only the target participant's deterministic calibration texts; evaluation texts never enter fitting or selection.
- Frozen label-free GPT-2 and Pythia caches were reused; no language model, QA set, fine-tuning, network, or GPU was used.
- GECO aggregate outcomes were previously inspected, so this remains exploratory and cannot confirm production personalization.
- An independent unchanged repeat produced byte-identical 570-row
  participant-fold detail (`SHA-256 04b8a360…ecccd9`) and exact matches for all
  budget summaries, gate decisions, and leakage diagnostics.
- The repository's offline quality gate passed 158/158 tests with zero network
  attempts, no Torch import, and unchanged GPU memory (75 MiB before and after).

The first execution stopped before any model fit because pandas interpreted the
literal GECO display word `null` as a missing CSV value. One of 56,411 feature
identities was affected. The cache reader was fixed to preserve labels
literally and convert only numeric feature columns; a regression test was added
in commit `31159c5`. No fold, outcome, model, feature, selector, or gate changed.

## Calibration learning curve

| Texts | Pythia selected | Selected − GPT-2 | 95% CI | Positive participants | Positive folds | Argmax − GPT-2 | Oracle upper bound |
|---:|---:|---:|:---:|---:|---:|---:|---:|
| 5 | 10/95 (10.5%) | +0.000090 | [-0.000114, +0.000284] | 6/19 | 4/5 | +0.000149 | +0.001445 |
| 10 | 5/95 (5.3%) | +0.000115 | [+0.000020, +0.000234] | 5/19 | 3/5 | +0.000255 | +0.001445 |
| 20 | 6/95 (6.3%) | +0.000066 | [-0.000088, +0.000214] | 4/19 | 2/5 | +0.000161 | +0.001445 |
| 40 | 6/95 (6.3%) | -0.000016 | [-0.000165, +0.000134] | 2/19 | 1/5 | +0.000494 | +0.001445 |
| 80 | 4/95 (4.2%) | +0.000069 | [+0.000000, +0.000195] | 2/19 | 3/5 | +0.000679 | +0.001445 |
| all | 8/95 (8.4%) | +0.000211 | [+0.000000, +0.000531] | 3/19 | 5/5 | +0.000542 | +0.001445 |

## Frozen primary gate

- PASS — `participant_bootstrap_ci_low_positive`
- FAIL — `positive_outer_folds`
- FAIL — `positive_participant_fraction`
- PASS — `challenger_selection_fraction`
- PASS — `leakage_checks`

## Interpretation

The ten-text uncertainty-aware selector did not clear every frozen clause. The conditional SmolLM2/OLMo/Granite extraction is therefore stopped: adding candidates now would increase selection variance without evidence that personalized model choice generalizes.

At the primary budget, the plain calibration argmax changed held-out Spearman by +0.000255; the non-deployable evaluation oracle upper bound was +0.001445. The gap between them distinguishes available reader-level heterogeneity from calibration reliability.

Detailed participant-cell results are stored only in the ignored local experiment directory; the tracked summary contains aggregates only.
