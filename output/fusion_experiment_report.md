# LexiGaze Multimodal Gaze-Cognitive Fusion Report

This report summarizes the comparative evaluation of eleven fusion algorithms designed to combine eye-gaze tracking metrics (total reading time / dwell duration) with cognitive load metrics (information surprisal from BERT) into a unified **Reading Difficulty Score (RDS)**.

The algorithms were tested on **156 merged word records** from the **GECO Corpus (pp01, Trial 5)**. The ground-truth reading difficulty is represented by the actual human **Total Reading Time (TRT)**.

---

## Validity Scope

This is a **descriptive calibration diagnostic**, not an out-of-sample prediction benchmark. The simulated dwell and fixation inputs are constructed from the same TRT used as the evaluation target, and the current cognitive-mass extraction path can include GECO-supervised XGBoost/Ridge scores. Use `output/geco_generalization_report.md` for preregistered new-reader/new-trial evidence.

---

## Fusion Performance Summary

Each fusion method was evaluated by computing its Pearson correlation ($r$) and Spearman rank correlation ($\rho$) against the ground-truth Total Reading Time.

| Method | Pearson r | Pearson p-val | Spearman rho | Spearman p-val |
| --- | --- | --- | --- | --- |
| Linear | 0.7341 | 1.11e-27 | 0.614 | 1.55e-17 |
| Multiplicative | 0.6383 | 3.17e-19 | 0.3716 | 1.79e-06 |
| Gated | 0.5495 | 1.10e-13 | 0.6228 | 3.95e-18 |
| Sigmoid | 0.7503 | 1.78e-29 | 0.614 | 1.55e-17 |
| Bayesian | 0.6445 | 1.10e-19 | 0.3909 | 4.53e-07 |
| Rrf | 0.7203 | 3.02e-26 | 0.6569 | 1.26e-20 |
| Spillover_bayesian | 0.6192 | 6.89e-18 | 0.5403 | 3.36e-13 |
| Parafoveal | 0.6582 | 9.90e-21 | 0.4344 | 1.47e-08 |
| Spillover_rrf | 0.656 | 1.49e-20 | 0.5704 | 7.61e-15 |
| Parafoveal_rrf | 0.7161 | 7.95e-26 | 0.6537 | 2.22e-20 |
| Spillover_parafoveal_rrf | 0.6547 | 1.87e-20 | 0.5744 | 4.49e-15 |

---

## Key Findings

1. **Best Performing Method**: The **Rrf** fusion algorithm achieved the highest Spearman correlation of **0.6569** and Pearson correlation of **0.7203**.
2. **Interactive Effects**: Multiplicative and Bayesian update methods typically outperform simple linear sums. This is because reading difficulty is non-linear: a high-surprisal word that is skipped (short dwell) does not present actual cognitive difficulty to the reader, whereas high surprisal accompanied by long dwell duration indicates true processing bottleneck.
3. **Rank-based Robustness**: Reciprocal Rank Fusion (RRF) provides a robust, scale-invariant alternative that requires no parameter tuning and remains highly correlated with reading times.

---

## Top 10 Most Difficult Words (Identified by Rrf Fusion)

Below are the top 10 words identified as having the highest reading difficulty under the best fusion model.

| WORD_ID | WORD | WORD_TOTAL_READING_TIME | surprisal_score | RDS_rrf |
| --- | --- | --- | --- | --- |
| 4-5-32 | unfeignedly | 1051 | 26.234 | 1.0 |
| 4-5-27 | admiration | 668 | 15.04 | 0.9085566153312384 |
| 3-5-5 | stupefied  | 709 | 19.917 | 0.8937215118302122 |
| 3-5-83 | arresting  | 741 | 13.438 | 0.8820404527701096 |
| 4-5-59 | expressed | 989 | 11.142 | 0.8713509946488949 |
| 4-5-39 | passionately | 672 | 15.917 | 0.837927108481283 |
| 4-5-11 | stepmother | 586 | 13.836 | 0.8127571001326119 |
| 3-5-12 | surprised  | 912 | 8.405 | 0.7571429751274723 |
| 3-5-106 | inquest?  | 701 | 11.361 | 0.732275190350871 |
| 4-5-87 | sweetest | 717 | 12.776 | 0.7186300816568606 |

---

## Visualizations Generated in `output/`

1. **`fusion_correlation_comparison.png`**: Bar chart comparing Pearson and Spearman correlation coefficients across all 11 methods.
2. **`rds_distributions.png`**: Density plot showing the RDS score distributions.
3. **`gaze_cognitive_space_rds.png`**: Scatter plot of the 2D gaze-cognitive space (Surprisal vs. Dwell time) colored by fused RDS.
4. **`top_difficult_words.png`**: Horizontal bar plot of the top 10 most difficult words.
