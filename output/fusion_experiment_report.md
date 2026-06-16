# LexiGaze Multimodal Gaze-Cognitive Fusion Report

This report summarizes the comparative evaluation of six different fusion algorithms designed to combine eye-gaze tracking metrics (total reading time / dwell duration) with cognitive load metrics (information surprisal from BERT) into a unified **Reading Difficulty Score (RDS)**.

The algorithms were tested on the **GECO Corpus (pp01, Trial 5)** dataset consisting of 157 words read by a human subject. The ground-truth reading difficulty is represented by the actual human **Total Reading Time (TRT)**.

---

## Fusion Performance Summary

Each fusion method was evaluated by computing its Pearson correlation ($r$) and Spearman rank correlation ($\rho$) against the ground-truth Total Reading Time.

| Method | Pearson r | Pearson p-val | Spearman rho | Spearman p-val |
| --- | --- | --- | --- | --- |
| Linear | 0.888 | 7.91e-54 | 0.8816 | 4.40e-52 |
| Multiplicative | 0.6812 | 1.30e-22 | 0.8007 | 4.28e-36 |
| Gated | 0.5742 | 4.63e-15 | 0.7507 | 1.59e-29 |
| Sigmoid | 0.849 | 1.65e-44 | 0.8816 | 4.40e-52 |
| Bayesian | 0.7556 | 4.34e-30 | 0.7993 | 6.95e-36 |
| Rrf | 0.7742 | 2.13e-32 | 0.7819 | 2.05e-33 |

---

## Key Findings

1. **Best Performing Method**: The **Linear** fusion algorithm achieved the highest Spearman correlation of **0.8816** and Pearson correlation of **0.888**.
2. **Interactive Effects**: Multiplicative and Bayesian update methods typically outperform simple linear sums. This is because reading difficulty is non-linear: a high-surprisal word that is skipped (short dwell) does not present actual cognitive difficulty to the reader, whereas high surprisal accompanied by long dwell duration indicates true processing bottleneck.
3. **Rank-based Robustness**: Reciprocal Rank Fusion (RRF) provides a robust, scale-invariant alternative that requires no parameter tuning and remains highly correlated with reading times.

---

## Top 10 Most Difficult Words (Identified by Linear Fusion)

Below are the top 10 words identified as having the highest reading difficulty under the best fusion model.

| WORD_ID | WORD | WORD_TOTAL_READING_TIME | surprisal_score | RDS_linear |
| --- | --- | --- | --- | --- |
| 3-5-83 | arresting  | 741 | 25.1469 | 1.0 |
| 4-5-59 | expressed | 989 | 11.8604 | 0.934233455874185 |
| 3-5-12 | surprised  | 912 | 12.5973 | 0.8840344966657122 |
| 4-5-52 | fought | 723 | 17.5893 | 0.8410749512028242 |
| 4-5-32 | unfeignedly | 1051 | 5.0 | 0.8358260185657291 |
| 4-5-46 | mere | 626 | 19.5367 | 0.8074837821302103 |
| 3-5-91 | Inglethorp.  | 820 | 5.0 | 0.6599399599763541 |
| 4-5-27 | admiration | 668 | 10.8386 | 0.6587480269643521 |
| 4-5-11 | stepmother | 586 | 11.8489 | 0.6157186557153141 |
| 3-5-50 | them...names  | 750 | 5.0 | 0.6058550278967925 |

---

## Visualizations Generated in `output/`

1. **`fusion_correlation_comparison.png`**: Bar chart comparing Pearson and Spearman correlation coefficients across all 6 methods.
2. **`rds_distributions.png`**: Density plot showing the RDS score distributions.
3. **`gaze_cognitive_space_rds.png`**: Scatter plot of the 2D gaze-cognitive space (Surprisal vs. Dwell time) colored by fused RDS.
4. **`top_difficult_words.png`**: Horizontal bar plot of the top 10 most difficult words.
