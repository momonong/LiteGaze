# Reliability-Aware Selective Fusion v1: Planning Simulation

- Status: **`completed_planning_only`**
- Protocol: `lexigaze-chi-reliability-aware-selective-fusion-v1`
- Seed: `20260817`; replications per cell: `200`
- Compute: CPU-only NumPy; GPU used: `false`; Torch imported: `false`
- Human, QA, and cognitive-profile outcomes used: **none**
- Runtime reference: `48.0438` seconds (`prior_identical_no_write_run`)

## Interpretation boundary

This is a deterministic crossed-cluster **planning sensitivity analysis**. It
does not estimate a human effect, authorize recruitment, promote F2, establish
webcam accuracy, or support cognitive, attention, fatigue, English-proficiency,
CEFR, or learning-benefit claims. The first-N diagnostics below are not sample
size recommendations.

The simulated outcome is the independently collected three-class word-review
response (`no_review`, `unsure`, `review_needed`). F0 is always-on
text-person-gaze fusion, F1 is the text-plus-person fallback, and F2 is
reliability-aware selective fusion. The primary estimands are F2 minus F1 and
F2 minus F0 mean multiclass NLL on a joint held-out participant,
passage-family, capture-session, and device-group cell. Brier score, normalized
ranked probability score (mean over the K-1 cumulative thresholds), and every
frozen risk-coverage cell are secondary.

The F2-vs-F0 sensitivity diagnostic includes only rows with synthetic noisy
gaze. It contains no true-missing gaze cases and cannot evaluate F0's frozen
future imputation branch. The generator also emits confirmation-only rows and
does not exercise the declared partition-assignment implementation.

## Yield and primary NLL sensitivity

| Scenario | Added gaze signal | Enrolled | Mean paired confirmation participants | Mean joint labels | Mean gaze eligibility | Structural evaluability | Mean F2-F1 NLL | Power vs F1 | Mean F2-F0 NLL | Power vs F0 | Same-replication joint power |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| optimistic | no_added_gaze_signal | 20 | 3.4 | 82.1 | 0.761 | 0.000 | 0.00000 | NE | 0.00000 | NE | NE |
| optimistic | no_added_gaze_signal | 40 | 6.9 | 165.6 | 0.745 | 0.265 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| optimistic | no_added_gaze_signal | 144 | 24.7 | 593.9 | 0.737 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| optimistic | no_added_gaze_signal | 300 | 51.4 | 1233.6 | 0.743 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| optimistic | no_added_gaze_signal | 600 | 102.9 | 2470.0 | 0.750 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| optimistic | no_added_gaze_signal | 900 | 153.5 | 3684.5 | 0.751 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| optimistic | weak_added_gaze_signal | 20 | 3.5 | 84.1 | 0.759 | 0.000 | -0.00413 | NE | -0.00022 | NE | NE |
| optimistic | weak_added_gaze_signal | 40 | 6.8 | 163.6 | 0.752 | 0.250 | -0.00376 | 0.240 | -0.00037 | 0.000 | 0.000 |
| optimistic | weak_added_gaze_signal | 144 | 24.9 | 598.3 | 0.748 | 1.000 | -0.00378 | 0.690 | -0.00002 | 0.010 | 0.005 |
| optimistic | weak_added_gaze_signal | 300 | 51.4 | 1234.4 | 0.745 | 1.000 | -0.00375 | 0.955 | -0.00049 | 0.025 | 0.020 |
| optimistic | weak_added_gaze_signal | 600 | 102.7 | 2464.8 | 0.753 | 1.000 | -0.00380 | 1.000 | -0.00018 | 0.035 | 0.035 |
| optimistic | weak_added_gaze_signal | 900 | 154.2 | 3700.6 | 0.752 | 1.000 | -0.00381 | 1.000 | -0.00053 | 0.045 | 0.045 |
| optimistic | moderate_added_gaze_signal | 20 | 3.5 | 83.6 | 0.747 | 0.000 | -0.00779 | NE | -0.01546 | NE | NE |
| optimistic | moderate_added_gaze_signal | 40 | 6.9 | 165.0 | 0.766 | 0.300 | -0.00781 | 0.167 | -0.01459 | 0.050 | 0.000 |
| optimistic | moderate_added_gaze_signal | 144 | 24.9 | 596.8 | 0.757 | 1.000 | -0.00757 | 0.545 | -0.01554 | 0.250 | 0.095 |
| optimistic | moderate_added_gaze_signal | 300 | 51.3 | 1231.7 | 0.751 | 1.000 | -0.00742 | 0.840 | -0.01583 | 0.575 | 0.470 |
| optimistic | moderate_added_gaze_signal | 600 | 102.9 | 2469.0 | 0.753 | 1.000 | -0.00767 | 0.995 | -0.01530 | 0.865 | 0.860 |
| optimistic | moderate_added_gaze_signal | 900 | 153.9 | 3693.6 | 0.748 | 1.000 | -0.00747 | 0.995 | -0.01531 | 0.980 | 0.975 |
| base | no_added_gaze_signal | 20 | 2.8 | 68.3 | 0.508 | 0.000 | 0.00000 | NE | 0.00000 | NE | NE |
| base | no_added_gaze_signal | 40 | 5.9 | 142.6 | 0.536 | 0.110 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| base | no_added_gaze_signal | 144 | 21.0 | 503.8 | 0.523 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| base | no_added_gaze_signal | 300 | 43.2 | 1037.8 | 0.523 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| base | no_added_gaze_signal | 600 | 86.1 | 2067.5 | 0.520 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| base | no_added_gaze_signal | 900 | 129.9 | 3117.2 | 0.524 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| base | weak_added_gaze_signal | 20 | 2.9 | 69.8 | 0.501 | 0.000 | -0.00332 | NE | -0.00544 | NE | NE |
| base | weak_added_gaze_signal | 40 | 5.7 | 136.0 | 0.510 | 0.050 | -0.00352 | 0.100 | -0.00642 | 0.000 | 0.000 |
| base | weak_added_gaze_signal | 144 | 20.9 | 502.3 | 0.522 | 1.000 | -0.00378 | 0.520 | -0.00519 | 0.075 | 0.015 |
| base | weak_added_gaze_signal | 300 | 43.4 | 1041.7 | 0.528 | 1.000 | -0.00368 | 0.870 | -0.00522 | 0.140 | 0.095 |
| base | weak_added_gaze_signal | 600 | 86.6 | 2079.2 | 0.519 | 1.000 | -0.00367 | 0.990 | -0.00475 | 0.190 | 0.185 |
| base | weak_added_gaze_signal | 900 | 129.4 | 3106.2 | 0.518 | 1.000 | -0.00367 | 1.000 | -0.00501 | 0.305 | 0.305 |
| base | moderate_added_gaze_signal | 20 | 2.9 | 68.6 | 0.513 | 0.000 | -0.00876 | NE | -0.06566 | NE | NE |
| base | moderate_added_gaze_signal | 40 | 5.9 | 141.4 | 0.503 | 0.055 | -0.00589 | 0.000 | -0.07145 | 0.182 | 0.000 |
| base | moderate_added_gaze_signal | 144 | 20.9 | 502.1 | 0.515 | 1.000 | -0.00728 | 0.370 | -0.06737 | 0.815 | 0.290 |
| base | moderate_added_gaze_signal | 300 | 43.1 | 1033.9 | 0.518 | 1.000 | -0.00703 | 0.600 | -0.06804 | 0.985 | 0.595 |
| base | moderate_added_gaze_signal | 600 | 86.1 | 2066.4 | 0.519 | 1.000 | -0.00710 | 0.910 | -0.06646 | 1.000 | 0.910 |
| base | moderate_added_gaze_signal | 900 | 129.8 | 3116.4 | 0.521 | 1.000 | -0.00689 | 0.980 | -0.06846 | 1.000 | 0.980 |
| pessimistic | no_added_gaze_signal | 20 | 2.1 | 49.8 | 0.264 | 0.000 | 0.00000 | NE | 0.00000 | NE | NE |
| pessimistic | no_added_gaze_signal | 40 | 4.2 | 101.5 | 0.317 | 0.015 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| pessimistic | no_added_gaze_signal | 144 | 14.9 | 356.8 | 0.295 | 0.995 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| pessimistic | no_added_gaze_signal | 300 | 30.5 | 731.9 | 0.291 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| pessimistic | no_added_gaze_signal | 600 | 62.6 | 1502.9 | 0.300 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| pessimistic | no_added_gaze_signal | 900 | 93.4 | 2242.7 | 0.300 | 1.000 | 0.00000 | 0.000 | 0.00000 | 0.000 | 0.000 |
| pessimistic | weak_added_gaze_signal | 20 | 2.1 | 51.6 | 0.282 | 0.000 | -0.00254 | NE | -0.00915 | NE | NE |
| pessimistic | weak_added_gaze_signal | 40 | 4.2 | 102.0 | 0.303 | 0.000 | -0.00187 | NE | -0.01150 | NE | NE |
| pessimistic | weak_added_gaze_signal | 144 | 15.2 | 364.0 | 0.309 | 1.000 | -0.00161 | 0.105 | -0.01611 | 0.160 | 0.000 |
| pessimistic | weak_added_gaze_signal | 300 | 31.0 | 744.4 | 0.301 | 1.000 | -0.00177 | 0.295 | -0.01413 | 0.220 | 0.065 |
| pessimistic | weak_added_gaze_signal | 600 | 62.7 | 1504.6 | 0.298 | 1.000 | -0.00171 | 0.545 | -0.01467 | 0.455 | 0.180 |
| pessimistic | weak_added_gaze_signal | 900 | 93.0 | 2232.2 | 0.305 | 1.000 | -0.00176 | 0.760 | -0.01306 | 0.560 | 0.395 |
| pessimistic | moderate_added_gaze_signal | 20 | 2.2 | 52.4 | 0.273 | 0.000 | -0.00301 | NE | -0.16053 | NE | NE |
| pessimistic | moderate_added_gaze_signal | 40 | 4.1 | 97.7 | 0.309 | 0.000 | -0.00364 | NE | -0.15871 | NE | NE |
| pessimistic | moderate_added_gaze_signal | 144 | 15.1 | 362.0 | 0.310 | 1.000 | -0.00345 | 0.040 | -0.15879 | 0.975 | 0.040 |
| pessimistic | moderate_added_gaze_signal | 300 | 30.9 | 740.9 | 0.303 | 1.000 | -0.00341 | 0.155 | -0.16031 | 1.000 | 0.155 |
| pessimistic | moderate_added_gaze_signal | 600 | 62.7 | 1505.6 | 0.297 | 1.000 | -0.00325 | 0.345 | -0.16111 | 1.000 | 0.345 |
| pessimistic | moderate_added_gaze_signal | 900 | 94.1 | 2258.5 | 0.300 | 1.000 | -0.00331 | 0.500 | -0.16044 | 1.000 | 0.500 |

`NE` means too few independent participant or passage clusters for the frozen
planning diagnostic. Marginal power is the fraction of structurally evaluable
synthetic replications whose conservative participant-by-passage 95% interval
excludes zero in the improvement direction. Joint power is the fraction where
both F2-vs-F1 and F2-vs-F0 succeed in the same replication; only joint power is
used for the first-N diagnostic.

## First tested N meeting the planning diagnostics

| Scenario | Added gaze signal | First tested N | Recruitment recommendation? |
| --- | --- | ---: | --- |
| optimistic | no_added_gaze_signal | none tested | no |
| optimistic | weak_added_gaze_signal | none tested | no |
| optimistic | moderate_added_gaze_signal | 600 | no |
| base | no_added_gaze_signal | none tested | no |
| base | weak_added_gaze_signal | none tested | no |
| base | moderate_added_gaze_signal | 600 | no |
| pessimistic | no_added_gaze_signal | none tested | no |
| pessimistic | weak_added_gaze_signal | none tested | no |
| pessimistic | moderate_added_gaze_signal | none tested | no |

These cells use assumed completion, gaze quality, class prevalence, crossed
ICC, and gaze signal. They must be replaced with blinded rehearsal estimates
and a pre-outcome product cost threshold before a formal sample size is frozen.

## Frozen risk-coverage example

Base assumptions, moderate sensitivity anchor, N=900:

| Target eligible coverage | Realized eligible coverage | Mean selected eligible | Conditional accepted F2-F1 NLL | All-row F1 NLL | All-row selective hybrid NLL | Hybrid-F1 NLL | Hybrid-F0 NLL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1.000 | 1623.1 | -0.01325 | 0.8149 | 0.8080 | -0.00689 | -0.06846 |
| 0.8 | 0.800 | 1298.9 | -0.01492 | 0.8149 | 0.8087 | -0.00621 | -0.06778 |
| 0.6 | 0.600 | 974.3 | -0.01644 | 0.8149 | 0.8098 | -0.00513 | -0.06670 |
| 0.4 | 0.400 | 649.6 | -0.01792 | 0.8149 | 0.8112 | -0.00373 | -0.06530 |
| 0.2 | 0.200 | 325.0 | -0.01961 | 0.8149 | 0.8129 | -0.00204 | -0.06361 |

Conditional accepted risk scores only the selected eligible rows. All-row
system risk uses F2 on those selected rows and exact F1 fallback on every
unselected eligible or ineligible row; no observation is dropped from system
risk. No threshold was selected from these results. All five predeclared
coverage cells are retained, including any non-monotonic outcome.

## Integrity and negative results

- Exact F1 fallback passed in every cell: `true`.
- Exact no-added-gaze null sentinel passed: `true`.
- Formal recruitment authorized: **false**.
- Model promotion authorized: **false**.

- optimistic/weak_added_gaze_signal: no candidate enrollment met both the 0.80 evaluability and 0.80 same-replication joint diagnostic-power thresholds.
- base/weak_added_gaze_signal: no candidate enrollment met both the 0.80 evaluability and 0.80 same-replication joint diagnostic-power thresholds.
- pessimistic/weak_added_gaze_signal: no candidate enrollment met both the 0.80 evaluability and 0.80 same-replication joint diagnostic-power thresholds.
- pessimistic/moderate_added_gaze_signal: no candidate enrollment met both the 0.80 evaluability and 0.80 same-replication joint diagnostic-power thresholds.
- N=20: at least one assumption bundle did not reach 0.80 structural evaluability; this cell remains feasibility-only.
- N=40: at least one assumption bundle did not reach 0.80 structural evaluability; this cell remains feasibility-only.
- optimistic/N=40: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- optimistic/N=144: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- optimistic/N=300: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- base/N=40: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- base/N=144: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- base/N=300: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- base/N=600: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- pessimistic/N=144: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- pessimistic/N=300: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- pessimistic/N=600: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.
- pessimistic/N=900: the moderate blend had lower F2-vs-F1 diagnostic power than the weak blend; more gaze weight is not uniformly better under the frozen noise model.

## Limitations

- No human outcome, gaze, question-answer, text-model, or cognitive-profile data are read.
- Effect blends are sensitivity anchors, not a smallest effect of interest.
- The H1 practical threshold is not yet frozen; this run diagnoses zero-bound interval exclusion only and therefore cannot test H1 as written.
- The no-added-signal cell is an exact F1 sentinel, not a calibrated type-I-error experiment.
- The generator supplies noisy gaze for every row and treats eligibility as a quality gate; F2-vs-F0 diagnostics exclude true-missing cases and are not informative about F0 imputation behavior.
- The conservative crossed-cluster interval is a planning approximation and is unstable with few independent participant or passage clusters.
- Synthetic device groups are nested one-per-participant, so they add a structural count but no independent variance axis; a separate device-class transfer study still requires enough independent device classes.
- The generator emits fit-free confirmation-only sensitivity rows and does not exercise the declared development-validation-confirmation partition assignment or discard implementation.
- NLL, Brier, RPS, and risk-coverage estimates are properties of the frozen synthetic generator, not evidence of human model benefit.
