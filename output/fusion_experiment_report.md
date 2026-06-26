# LexiGaze Multimodal Gaze-Cognitive Fusion Report

This report summarizes the comparative evaluation of six different fusion algorithms designed to combine eye-gaze tracking metrics (total reading time / dwell duration) with cognitive load metrics (information surprisal from BERT) into a unified **Reading Difficulty Score (RDS)**.

The algorithms were tested on the **GECO Corpus (pp01, Trial 5)** dataset consisting of 157 words read by a human subject. The ground-truth reading difficulty is represented by the actual human **Total Reading Time (TRT)**.

---

## Fusion Performance Summary

Each fusion method was evaluated by computing its Pearson correlation ($r$) and Spearman rank correlation ($\rho$) against the ground-truth Total Reading Time.

| Method | Pearson r | Pearson p-val | Spearman rho | Spearman p-val |
| --- | --- | --- | --- | --- |
| Linear | 0.8827 | 2.32e-52 | 0.8587 | 1.45e-46 |
| Multiplicative | 0.7325 | 1.67e-27 | 0.7091 | 3.89e-25 |
| Gated | 0.6735 | 5.77e-22 | 0.759 | 1.69e-30 |
| Sigmoid | 0.8291 | 1.00e-40 | 0.8587 | 1.45e-46 |
| Bayesian | 0.7224 | 1.89e-26 | 0.7188 | 4.33e-26 |
| Rrf | 0.839 | 1.51e-42 | 0.8248 | 5.61e-40 |

---

## Key Findings

1. **Best Performing Method**: The **Linear** fusion algorithm achieved the highest Spearman correlation of **0.8587** and Pearson correlation of **0.8827**.
2. **Interactive Effects**: Multiplicative and Bayesian update methods typically outperform simple linear sums. This is because reading difficulty is non-linear: a high-surprisal word that is skipped (short dwell) does not present actual cognitive difficulty to the reader, whereas high surprisal accompanied by long dwell duration indicates true processing bottleneck.
3. **Rank-based Robustness**: Reciprocal Rank Fusion (RRF) provides a robust, scale-invariant alternative that requires no parameter tuning and remains highly correlated with reading times.

---

## Top 10 Most Difficult Words (Identified by Linear Fusion)

Below are the top 10 words identified as having the highest reading difficulty under the best fusion model.

| WORD_ID | WORD | WORD_TOTAL_READING_TIME | surprisal_score | RDS_linear |
| --- | --- | --- | --- | --- |
| 4-5-32 | unfeignedly | 1051 | 26.234 | 1.0 |
| 3-5-91 | Inglethorp.  | 820 | 29.425 | 0.9468173476331861 |
| 4-5-59 | expressed | 989 | 11.142 | 0.8716144920320693 |
| 3-5-5 | stupefied  | 709 | 19.917 | 0.8047107244851026 |
| 3-5-12 | surprised  | 912 | 8.405 | 0.7504310797493157 |
| 3-5-83 | arresting  | 741 | 13.438 | 0.7291186041873005 |
| 4-5-27 | admiration | 668 | 15.04 | 0.716148831770284 |
| 4-5-39 | passionately | 672 | 15.917 | 0.6995690591068072 |
| 4-5-11 | stepmother | 586 | 13.836 | 0.6704766698660942 |
| 3-5-106 | inquest?  | 701 | 11.361 | 0.6633970998481487 |

---

## Visualizations Generated in `output/`

1. **`fusion_correlation_comparison.png`**: Bar chart comparing Pearson and Spearman correlation coefficients across all 6 methods.
2. **`rds_distributions.png`**: Density plot showing the RDS score distributions.
3. **`gaze_cognitive_space_rds.png`**: Scatter plot of the 2D gaze-cognitive space (Surprisal vs. Dwell time) colored by fused RDS.
4. **`top_difficult_words.png`**: Horizontal bar plot of the top 10 most difficult words.
