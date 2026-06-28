# LexiGaze: A Multimodal Eye-Gaze and Cognitive-Load Fusion Platform for Reading Analysis

**Final Progress Report · June 2026**

| | |
|---|---|
| **Team** | Chenghao Peng, Weichi Lin, Shengwen Chang, Bowei Huang, Jhennong Chen |
| **Institution** | Miin Wu School of Computing |
| **Status** | Draft for course final report |

---

## Abstract

LexiGaze is a low-cost, zero-additional-hardware reading analysis platform. It aligns webcam gaze tracking (M1) and per-word cognitive load prediction (M3) onto a shared screen coordinate frame (M4), then applies trajectory fusion and optimization (M5) to produce a **Reading Difficulty Score (RDS)** and **Cognitive Inspector** diagnostic reports.

The cognitive pipeline uses GPT-2 surprisal and an XGBoost scorer to predict `load_score`. On a GECO held-out set it achieves a fixed effect of **β = 0.437** on Total Reading Time (TRT) with **ΔAIC = +182.6**; zero-shot transfer to PROVO raises β to **0.619**. Under simulated **+45 px** vertical drift, STOCK-T (Viterbi POM + EM) improves gaze decoding accuracy from **18.59%** to **78.21%**.

This document consolidates the final-presentation progress (M1–M5) and subsequent web-integration work. It is written in Markdown so the team can collaborate via Git (branching, PRs, section ownership).

---

## Table of Contents

1. [Motivation and Objectives](#1-motivation-and-objectives)
2. [System Overview: M1–M5](#2-system-overview-m1m5)
3. [M1: Gaze Perception and Personalization](#3-m1-gaze-perception-and-personalization)
4. [M3: Per-Word Cognitive Load Pipeline](#4-m3-per-word-cognitive-load-pipeline)
5. [M4: Spatiotemporal Alignment](#5-m4-spatiotemporal-alignment)
6. [M5: Trajectory Fusion and Optimization](#6-m5-trajectory-fusion-and-optimization)
7. [Related Work](#7-related-work)
8. [Post-Presentation Progress](#8-post-presentation-progress)
9. [Discussion and Future Work](#9-discussion-and-future-work)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)
12. [Collaboration Guide](#12-collaboration-guide)

---

## 1. Motivation and Objectives

High-fidelity eye trackers are expensive. Consumer webcams suffer from systematic vertical drift (often exceeding one line height) and high-frequency jitter, which breaks word-level gaze decoding (Krafka et al., 2016; Rayner, 1998).

Reading difficulty depends not only on linguistic complexity but also on moment-to-moment cognitive state and visual processing bottlenecks (Hale, 2001; Levy, 2008).

**LexiGaze assumes:** gaze trajectories are constrained both by physical space and by semantic cognitive attraction.

**Goals:**

- Fuse “where the eyes look” with “how difficult the text is” into a unified **RDS**.
- Surface actionable diagnostics: WPM, regression rate, English proficiency estimate, high-load vocabulary lists.
- Run on **low-cost hardware** with **in-browser real-time inference**.

---

## 2. System Overview: M1–M5

Final presentation narrative (data flow):

```
M1 Perception → M3 Cognitive Load → M4 Spatiotemporal Alignment → M5 Trajectory Fusion → Cognitive Inspector
```

**M2** (embedded in M1 + web workflow): calibration data collection and personalization training  
(Demo: 9-point grid → base UniGaze inference → polynomial-regression finetuning).

### Gaze metrics → cognitive targets (Slide 4)

| Gaze metric | How obtained | Cognitive target |
|-------------|--------------|------------------|
| Fixation Duration | Sum of dwell episodes | Cognitive load |
| Fixation Count | Hit counts | Attention |
| Regression Count | Backward saccades | Comprehension difficulty |
| Reread Count | Re-read episodes | Difficult words |
| Dwell Time | Area residence time | Overall load |

### Module ownership

| Module | Topic | Owner | Status |
|--------|-------|-------|--------|
| M1 | Gaze / UniGaze / calibration | Shengwen Chang | Completed |
| M2 | Data collection & training demo | Bowei Huang | Completed |
| M3 | Cognitive load pipeline | Weichi Lin | Completed |
| M4 | Spatiotemporal alignment | Chenghao Peng | Completed |
| M5 | Trajectory fusion & RDS | Jhennong Chen | Completed |

---

## 3. M1: Gaze Perception and Personalization

### Technical stack

| Component | Description |
|-----------|-------------|
| **UniGaze-B16 ViT** | Large-scale pre-trained gaze estimation; pitch/yaw → screen coordinates (UniGaze, WACV 2026) |
| **Polynomial personalization** | 9-point grid calibration; removes user/device bias (JEMR 2023 calibration literature) |
| **Real-time filtering** | OneEuro speed-adaptive low-pass filter (CHI 2012); optional Slip-Kalman-style smoothing (IEEE SENSORS 2019) |

### Demo progress (presentation)

Three-stage comparison:

1. **Data Collection & Training**
2. **Base Model Inference** (no post-processing)
3. **After Finetuning** (no post-processing)

Demonstrates personalization gains over raw UniGaze. Target: **low cost, zero extra hardware, in-browser real-time inference**.

### Key paths

- `core/gaze_core/`
- `core/unigaze_personalization/`
- `web/routes/gaze.py`

---

## 4. M3: Per-Word Cognitive Load Pipeline

## 1  Introduction

Reading is a cognitively demanding process in which the difficulty of each individual word contributes to the overall mental effort required to process a sentence. Researchers in psycholinguistics and human-computer interaction have long sought to quantify this difficulty at the word level — a challenge referred to as cognitive load estimation. Understanding per-word cognitive load enables downstream applications ranging from adaptive reading interfaces and educational tools to eye-tracking-based implicit feedback systems.

Within the LexiGaze project, the Per-Word Cognitive Load Pipeline (M3) serves as the language-side component that provides continuous difficulty scores for every word in a given English text. These scores are intended to be fused with eye-tracking data collected by the other team members, enabling a cross-modal reading difficulty estimation system. Rather than relying on coarse-grained readability formulas (e.g., Flesch–Kincaid) that operate at the sentence or document level, M3 generates fine-grained, psycholinguistically motivated scores at the individual word level.

This section describes the design, implementation, and validation of M3. We introduce the theoretical motivation (Section 2), survey related work (Section 3), describe the pipeline architecture and features (Section 4), present validation results against two eye-tracking corpora (Section 5), and analyse the contribution of individual components through ablation experiments (Section 6).

---

## 2  Background

### 2.1  Surprisal Theory and Reading Time

A central theoretical grounding for this work is surprisal theory (Hale, 2001; Levy, 2008), which proposes that the cognitive effort required to process a word is proportional to its surprisal — the negative log-probability of the word given its preceding context:

$$\text{surprisal}(w_t) = -\log P(w_t \mid w_1, \ldots, w_{t-1})$$

Under this framework, words that are unexpected given prior context are harder to process, and this increased processing difficulty manifests as longer reading times. Empirical support for this theory has come from a long line of studies using self-paced reading and eye-tracking paradigms. The advent of large pre-trained language models has made it possible to compute surprisal values automatically for any word in any sentence.

### 2.2  Eye-Tracking Measures of Reading

Eye-tracking during natural reading provides millisecond-resolution behavioural data that serves as a proxy for cognitive processing difficulty. The two primary word-level measures used in this work are:

- **Gaze Duration (GD):** the sum of all fixation durations during the first pass through a word, before any regression. GD primarily reflects early lexical processing such as word recognition and initial lexical access.
- **Total Reading Time (TRT):** the sum of all fixation durations on a word across the entire trial, including re-fixations and regressions. TRT captures later, integrative processing including syntactic parsing, anaphora resolution, and discourse-level inference.

Both measures are right-skewed and must be log-transformed before regression analysis. Spillover effects — where the difficulty of word *n* influences fixation time on word *n+1* — are controlled by including the previous word's surprisal and length as covariates.

### 2.3  Mixed-Effects Regression for Reading Time Data

Because reading time data is nested (multiple words per sentence, multiple sentences per reader), standard OLS regression violates the independence assumption and inflates the effective sample size. The standard approach in psycholinguistics is linear mixed-effects models (LMMs), which partition variance into fixed effects (predictors of interest) and random effects (by-subject and by-item intercepts, and optionally slopes).

In this work we fit LMMs with per-reader random intercepts, controlling for word length, Zipf frequency, sentence position, and preceding-word spillover. Model fit is compared using the Akaike Information Criterion (AIC), where a larger ΔAIC indicates a better fit. The Likelihood Ratio Test (LRT) provides a significance test for the incremental contribution of `load_score` over the baseline-only model.

---

## 3  Related Work

### 3.1  LM Surprisal as a Reading Time Predictor

The use of language model surprisal to predict eye-tracking reading times has a long history. Demberg and Keller (2008) demonstrated that surprisal from n-gram language models significantly predicts reading times on the Dundee corpus, establishing the mixed-effects regression framework we follow. Smith and Levy (2013) established that surprisal from larger language models predicts reading times with a log-linear relationship.

With the advent of transformer-based LMs, Oh and Schuler (2023) made a striking finding: surprisal from GPT-2 (117M parameters) predicts reading times better than surprisal from much larger models, including GPT-2-XL (1.5B) and GPT-Neo (1.3B). They attribute this to the fact that larger models assign more uniform probability distributions — their predictions are too calibrated, leaving less variance to explain in reading time. Goodkind and Bicknell (2018) likewise showed that GPT-2 outperforms earlier LMs on reading time prediction. These findings directly motivate our choice of GPT-2 (117M) as the surprisal source. Alves et al. (2025) provide a recent benchmark of LM surprisal across FFD, GD, and TRT on multiple corpora, confirming GPT-2's continued competitiveness.

More recently, Pimentel et al. (2023) demonstrated that Rényi entropy (a generalisation of Shannon entropy parameterised by α) better predicts first-pass reading times than standard Shannon surprisal, particularly for α < 1, which up-weights low-probability tokens and better captures reading difficulty at word boundaries.

### 3.2  Multi-Feature Cognitive Load Models

Beyond surprisal alone, a growing body of work combines multiple psycholinguistic features to model word-level difficulty. Kuperman et al. (2012) established age of acquisition (AoA) as a strong predictor of lexical processing speed, showing that words learned later in life are processed more slowly even after controlling for word frequency. Brysbaert and New (2009) showed that Zipf frequency is the best single predictor of word recognition times. Kajiwara and Komachi (2018) evaluate feature-based models for complex word identification, showing that frequency baselines are strong but can be improved with additional features including AoA.

### 3.3  Syntactic Complexity and Reading Time

Syntactic complexity has been proposed as an additional source of reading difficulty, particularly via dependency distance. Gibson's Dependency Locality Theory (DLT, 1998) predicts that longer dependencies impose higher integration costs. However, several studies have found that syntactic effects are weaker in naturalistic reading corpora (novels, newspaper text) than in laboratory stimuli. Our component decomposition results (Section 6) confirm this: dep_load is not significant on the GECO fiction corpus.

### 3.4  Eye-Tracking Corpora

Two corpora are used for validation. GECO (Cop et al., 2017) contains word-level eye-tracking data from 14 English L1 participants reading the Christie novel *The Mysterious Affair at Styles* in full (~5,000 sentences). PROVO (Luke and Christianson, 2018) contains 55 passages from mixed genres (news, Wikipedia, narrative) read by 84 L1 participants. PROVO is widely used for zero-shot validation because its genre diversity and larger participant pool make it a challenging out-of-domain test.

---

## 4  Method: Pipeline Architecture

### 4.1  Pipeline Overview

The pipeline takes a raw English sentence as input and produces a per-word `load_score` ∈ [0, 1], along with a categorical label (high / medium / low) derived from document-relative thresholding. The processing stages are: (1) tokenization and part-of-speech tagging using spaCy, with parallel GPT-2 BPE tokenization for surprisal computation; (2) parallel extraction of six psycholinguistic features; (3) non-linear scoring via a pre-trained XGBoost model; and (4) relative thresholding and label assignment. Figure 1 illustrates the complete pipeline architecture.

![pipeline_architecture](https://hackmd.io/_uploads/rkuVf-pzMx.png)

*Figure 1. Per-Word Cognitive Load Pipeline architecture. Raw text is tokenized and POS-tagged; six features are extracted in parallel and scored by a trained XGBoost model to produce continuous load scores.*

### 4.2  Feature Set

The pipeline extracts six features per content word, chosen to cover the major known sources of word-level reading difficulty:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| Surprisal | GPT-2 (117M) contextual next-word NLL | Contextual unpredictability |
| Rényi entropy | LM uncertainty, α = 0.5 | First-pass reading times |
| AoA score | Age-of-acquisition (Kuperman et al., 2012) | Lexical difficulty beyond frequency |
| Zipf frequency | Log word frequency (wordfreq) | Lexical baseline |
| Dep load | Dependency integration cost (NOUN/VERB/PROPN only) | Syntactic integration cost |
| Word length | Character count | Orthographic baseline |

*Table 1. Feature set used in v9 of the pipeline.*

Function words (articles, prepositions, conjunctions) are excluded from dep_load computation via POS gating: only NOUN, VERB, and PROPN tokens receive a non-zero dependency load score. This prevents spurious syntactic load assignment to structurally simple function words, consistent with psycholinguistic evidence that syntactic integration costs are borne primarily by content words.

### 4.3  Surprisal Computation

Surprisal is computed using GPT-2 (117M parameters). For a target word *w* tokenised into sub-word units *t₁, …, tₖ* by the GPT-2 BPE tokeniser, the word-level surprisal is the sum of token-level negative log-probabilities:

$$\text{surprisal}(w) = \sum_k -\log P(t_k \mid t_1, \ldots, t_{k-1})$$

Sub-word tokens produced for hyphenated compounds are automatically re-aggregated to word level. GPT-2 is run with the full preceding sentence as context.

### 4.4  Rényi Entropy

Following Pimentel et al. (2023), we supplement surprisal with Rényi entropy, which places additional weight on low-probability continuations and better captures uncertainty experienced during first-pass reading. For a discrete distribution *P* over the vocabulary *V*, the Rényi entropy of order α is:

$$H_\alpha(P) = \frac{1}{1-\alpha} \log \sum_{v \in V} P(v)^\alpha$$

We use α = 0.5, which up-weights low-probability tokens and has been shown to maximally improve predictions of first-pass reading measures.

### 4.5  XGBoost Scoring Backend

Features are combined by a gradient-boosted decision tree model (XGBoost; Chen and Guestrin, 2016) trained on 2,000 GECO sentences (~9,793 content words). XGBoost was chosen over linear models for three reasons: (1) it captures non-linear interactions between features; (2) it is robust to collinearity between Zipf frequency and other features; and (3) ablation experiments confirmed its superiority over the Ridge baseline. The trained model is serialised as `xgb_model.json`; a Ridge regression fallback (`ridge_model.json`) is available when XGBoost is unavailable.

### 4.6  Thresholding and Label Assignment

Rather than using a fixed threshold, `load_score` is thresholded relative to the distribution within each document. Words above the 70th percentile are labelled "high", words between the 30th and 70th percentiles are labelled "medium", and words below the 30th percentile are labelled "low". This relative thresholding avoids the problem that different texts have different baseline difficulty levels.

---

## 5  Experiments and Results

### 5.1  Experimental Setup

All validation experiments use the GECO corpus (Cop et al., 2017) as the primary dataset. The XGBoost model is trained on sentences 1–2,100 (~2,000 training sentences) and evaluated on completely held-out data. Tokenisation alignment between GPT-2 BPE and GECO word boundaries is handled by string matching, with content words identified by POS tag. Reading time outliers below 80 ms or above 1,200 ms are excluded. All regression models log-transform TRT and GD before fitting.

LMMs are fitted with per-reader random intercepts and fixed effects controlling for word length, Zipf frequency, sentence position, and previous-word spillover (preceding surprisal and preceding word length). The Likelihood Ratio Test (LRT) with χ² distribution is used to compare models with and without `load_score`.

### 5.2  GECO Held-Out Validation

The primary evaluation is on 1,000 completely held-out GECO sentences (sentences 2,101–3,100), yielding 4,883 content words across 14 English L1 readers.

| Outcome | Spearman ρ | 95% CI | LMM β | ΔAIC |
|---------|-----------|--------|-------|------|
| **TRT** | 0.437*** | [0.412, 0.458] | 0.049*** | +182.6 |
| **GD** | 0.386*** | [0.359, 0.409] | 0.029*** | +82.8 |

*Table 2. GECO held-out validation (n = 4,883 content words, 14 readers). LMM controls for word length, Zipf frequency, sentence position, and spillover. \*\*\* p < .001.*

`load_score` shows a significant positive correlation with both TRT (ρ = 0.437) and GD (ρ = 0.386). The LMM coefficient remains significant after controlling for all baseline variables (TRT: β = 0.049, LRT χ²(1) = 184.61, p < .001, ΔAIC = +182.6), confirming that `load_score` captures variance in reading time not explained by word frequency, length, or positional factors alone. The larger TRT effect compared to GD is consistent with the theory that `load_score` captures integrative processing difficulty more than early lexical access.

> **Paper-ready summary:** "The pipeline predicted mean TRT with Spearman ρ = 0.437 (95% CI [0.412, 0.458]) and GD ρ = 0.386 (95% CI [0.359, 0.409]) on 4,883 content words from 1,000 held-out GECO sentences. After controlling for word frequency, length, sentence position, and preceding-word spillover, the load score independently predicted TRT (OLS β = 0.639, p < .001, ΔAIC = +104.6; LMM β = 0.049, LRT χ²(1) = 184.61, p < .001, ΔAIC = +182.6, n = 49,154 reader × word observations)."

### 5.3  Cross-Section Stability

To test generalisation across the novel beyond the primary held-out block, we evaluate on all remaining GECO sentences not used in training (sentences 2,101–5,284; 3,183 sentences, 16,318 content words). Results show ρ(TRT) = 0.440 and ρ(GD) = 0.400, with R² rising to 0.203 (ΔAIC = +342.8). The stability of performance at larger scale confirms that the pipeline does not overfit to the primary 1,000-sentence evaluation window.

### 5.4  PROVO Zero-Shot Generalisation

A critical test of generalisability is zero-shot transfer to PROVO (Luke and Christianson, 2018) — 55 passages from mixed genres read by 84 L1 participants. The XGBoost model was not retrained or fine-tuned in any way for PROVO.

| Evaluation | ρ (TRT) | ρ (GD) |
|------------|---------|--------|
| GECO held-out (1,000 sent.) | 0.437*** | 0.386*** |
| GECO full (3,183 sent.) | 0.440*** | 0.400*** |
| PROVO zero-shot (55 passages, 84 readers) | **0.619***** | **0.611**** |

*Table 3. Cross-corpus generalisation. PROVO results are fully zero-shot. \*\*\* p < .001.*

Zero-shot transfer to PROVO yields ρ(TRT) = 0.619 and ρ(GD) = 0.611, substantially higher than GECO held-out results. The higher ρ reflects two factors: (1) the larger participant pool (84 vs. 14 readers) produces more stable mean reading times; and (2) PROVO's mixed-genre content spans a wider range of lexical difficulty, providing greater variance for the pipeline to predict. The OLS coefficient remains significant (β = 0.652, p < .001, ΔAIC = +63.2), confirming that the pipeline's signal is not corpus-specific.

### 5.5  Robustness: Bootstrap CI and LOSO

We assess robustness through two complementary analyses.

- **Bootstrap confidence intervals (2,000 resamples):** Figure 2 shows the bootstrap distribution of ρ for TRT and GD. The 95% CIs are [0.413, 0.459] for TRT and [0.360, 0.410] for GD, indicating stable, narrow estimates.
- **Leave-one-subject-out (LOSO):** Figure 3 shows per-reader ρ for each of the 14 GECO readers evaluated independently. All 14 readers yield p < .001, with mean ρ = 0.215 ± 0.044 (range: [0.135, 0.291]). The consistent significance across all readers confirms that the pipeline's predictive signal is not driven by any single outlier reader.

![bootstrap_ci_plot](https://hackmd.io/_uploads/r1K_fbTzze.png)

*Figure 2. Bootstrap distribution of Spearman ρ (n = 2,000 resamples). Left: TRT, ρ = 0.437, 95% CI [0.413, 0.459]. Right: GD, ρ = 0.386, 95% CI [0.360, 0.410]. White line = observed ρ; red dashed lines = 95% CI bounds.*

![loso_plot](https://hackmd.io/_uploads/SJQYGWpGzg.png)

*Figure 3. Leave-one-subject-out per-reader Spearman ρ. All 14 GECO readers yield p < .001 (green). Mean ρ = 0.215 ± 0.044.*

---

## 6  Ablation Studies

### 6.1  Component Decomposition

To understand which features drive the pipeline's predictive validity, we fit a joint OLS regression including all features simultaneously as separate predictors. This allows assessment of each feature's independent contribution after controlling for all others.

| Feature | ρ (TRT) | Solo R² | Joint β | Significant? |
|---------|---------|---------|---------|--------------|
| Word length | +0.456*** | 0.208 | 0.031*** | Yes |
| Zipf frequency | −0.429*** | 0.227 | −0.035*** | Yes |
| Surprisal | +0.424*** | 0.191 | 0.011*** | Yes |
| AoA score | +0.276*** | 0.102 | 0.116** (beyond freq) | Yes |
| Dep load | −0.021 n.s. | < 0.001 | n.s. | No (fiction corpus) |
| **Joint model** | — | **0.308** | ΔAIC +25.2 | Significant improvement |

*Table 4. Component decomposition. ρ = marginal Spearman correlation with TRT; Solo R² = explained variance alone; Joint β = coefficient in the full joint model. \*\*\* p < .001, \*\* p < .01.*

Several findings emerge. Word length is the strongest marginal predictor (ρ = 0.456), followed by Zipf frequency (ρ = −0.429) and surprisal (ρ = 0.424). Surprisal retains a significant independent coefficient (β = 0.011, p < .001) after controlling for frequency and length, confirming that contextual predictability explains reading time variance beyond lexical baselines. AoA makes a significant contribution beyond frequency (β = 0.116, p < .01), supporting the established finding that age of acquisition is not fully reducible to frequency. Dep load is not significant on this fiction corpus — consistent with prior work showing that syntactic complexity effects are weaker in naturalistic text. The joint model achieves R² = 0.308, a significant improvement over the XGBoost composite score (R² = 0.289, ΔAIC = +25.2).

### 6.2  Language Model Scaling Comparison

We compare five language models as surprisal sources to test whether larger models produce better reading time predictions. Figure 4 and Table 5 show results on 100 GECO sentences.

![model_comparison](https://hackmd.io/_uploads/Skk5GWpMfl.png)

*Figure 4. LM scaling comparison on GECO. Left: marginal Spearman ρ. Right: incremental R² beyond frequency+length+position baseline. GPT-2 (117M) outperforms all larger models on both metrics.*

| Model | Params | ρ (TRT) | ΔR² | Note |
|-------|--------|---------|-----|------|
| **GPT-2** | **117M** | **0.398****** | **0.040** | **★ Best** |
| GPT-2-Large | 774M | 0.355*** | 0.027 | |
| GPT-2-XL | 1.5B | 0.345*** | 0.026 | |
| GPT-Neo | 1.3B | 0.347*** | 0.033 | |
| TinyLlama | 1.1B | 0.362*** | 0.033 | LLaMA arch. |

*Table 5. LM surprisal comparison on 100 GECO sentences. \*\*\* p < .001.*

The results replicate the scaling paradox reported by Oh and Schuler (2023): GPT-2 (117M) achieves the highest ρ (0.398) and incremental R² (0.040), while larger models consistently underperform. This effect holds across model families — TinyLlama (1.1B) outperforms GPT-2-XL (1.5B), suggesting architecture matters more than parameter count. Larger models produce more uniform probability distributions, making their surprisal values less sensitive to individual word difficulty. These results confirm GPT-2 (117M) as the optimal surprisal source for this pipeline.

---

## 7  Limitations and Future Work

### 7.1  Current Limitations

- **Dep load is not significant on fiction:** The GECO corpus consists of a single Christie novel with relatively simple syntactic structures. Dep_load is likely more informative for more syntactically complex genres. Validation on Dundee (newspaper) or academic text corpora may reveal a stronger syntactic effect.
- **English only:** All features, normative data (Kuperman AoA, wordfreq Zipf), and validation corpora are English. Extension to other languages requires language-specific AoA norms and new corpora.
- **Single training corpus:** The XGBoost model is trained on a single novel. While the PROVO zero-shot results confirm cross-genre generalisation, training on a more diverse corpus could further improve robustness.
- **Feature collinearity:** Zipf frequency and AoA are moderately correlated (~0.4), which reduces the estimated independent contribution of each in joint models.

### 7.2  Future Work

- **L2 learner validation:** The CELER corpus contains eye-tracking data from L1 and L2 English readers. Comparing predictive validity for L1 vs. L2 readers could reveal whether cognitive load from surprisal and AoA is differentially important for non-native readers.
- **Complex Word Identification:** The pipeline's `load_score` could be evaluated on SemEval-2016 Task 11 CWI benchmark to provide an additional task-based validation.
- **Downstream fusion with eye-tracking:** Once the LexiGaze eye-tracking data is collected, per-word `load_score` will be correlated with recorded GD and TRT values to assess predictive validity in the experimental context.

---

## 8  References

Alves, R. et al. (2025). LM surprisal as a predictor of fixation duration. *Gaze4NLP Workshop*.

Brysbaert, M., & New, B. (2009). Moving beyond Kucera and Francis: A critical evaluation of current word frequency norms. *Behavior Research Methods, 41*(4), 977–990.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD*.

Cop, U., Drieghe, D., & Duyck, W. (2017). Presenting GECO: An eyetracking corpus of monolingual and bilingual sentence reading. *Behavior Research Methods, 49*(2), 602–615.

Demberg, V., & Keller, F. (2008). Data from eye-tracking corpora as evidence for theories of syntactic processing complexity. *Cognition, 109*(2), 193–210.

Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies. *Cognition, 68*(1), 1–76.

Goodkind, A., & Bicknell, K. (2018). Predictive power of word surprisal for reading times is a linear function of language model quality. *Proceedings of the 8th Workshop on Cognitive Modeling and Computational Linguistics*.

Hale, J. (2001). A probabilistic Earley parser as a psycholinguistic model. *Proceedings of NAACL*.

Kajiwara, T., & Komachi, M. (2018). Complex word identification based on frequency in a learner corpus. *Proceedings of the 13th Workshop on Innovative Use of NLP for Building Educational Applications*.

Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings for 30,000 English words. *Behavior Research Methods, 44*(4), 978–990.

Levy, R. (2008). Expectation-based syntactic comprehension. *Cognition, 106*(3), 1126–1177.

Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with predictability norms. *Behavior Research Methods, 50*(2), 826–833.

Oh, B.-D., & Schuler, W. (2023). Why does surprisal from larger transformer-based language models provide less regression fit to human reading times? *Transactions of the Association for Computational Linguistics, 11*, 336–350.

Pimentel, T., Meister, C., Wilcox, E. G., Schuler, W., & Cotterell, R. (2023). On the effect of anticipation on reading times. *Transactions of the Association for Computational Linguistics, 11*, 1624–1642.

Smith, N. J., & Levy, R. (2013). The effect of word predictability on reading time is logarithmic. *Cognition, 128*(3), 302–319.

---

## 5. M4: Spatiotemporal Alignment

## 1. Background
In low-cost, web-based eye-tracking scenarios, a fundamental mismatch exists between the "semantic space" and the "physical space". The Cognitive Mass (CM) extracted by the M3 module is inherently a pure numerical value bound to string symbols (combining local difficulty Surprisal and global importance Attention Centrality), lacking any screen coordinate concept. Meanwhile, raw gaze trajectories captured by webcams are plagued with hardware drift and noise. Therefore, the system requires a bridging module to translate static cognitive difficulty into "Spatial Gravity" on the physical screen, imbuing it with temporal dynamics to guide and anchor the noisy gaze coordinates.

## 2. Related Work
The design of this module is deeply inspired by psycholinguistics and oculomotor dynamics:
* **Optimal Viewing Position (OVP)**: According to Rayner (1979), a reader's gaze most frequently lands slightly to the left of center (about one-third of the way into the word) when reading English words, rather than the exact geometric center.
* **Dynamic Reading Prior**: Traditional E-Z Reader and SWIFT models indicate that gaze duration on a word is proportional to its lexical difficulty, and readers trigger a saccade to leave the current word after acquiring sufficient information. This implies that spatial gravity should not be eternally static but must decay with "exposure time".

## 3. Methodology & Technology
The M4 module (`DynamicCognitiveField`) constructs a time-varying dynamic prior field through three core mechanisms:

### 3.1 OVP Alignment & Adaptive Variance
Instead of using the geometric center of the word's bounding box, the system enables OVP correction by default (`use_ovp=True`).
* **Gravity Center Shift**: The gravity center of each word is set at 35% of its width from the left border.
* **Length-Adaptive Gravity Radius**: Longer words require a wider horizontal tolerance. The system uses a baseline $\sigma = 30.0$ and dynamically scales the horizontal standard deviation based on the word width ratio.

### 3.2 Bayesian Gravity Snap
A 2D Gaussian distribution is utilized to construct spatial likelihood. When a raw gaze point $(x, y)$ is input, the system calculates the probability of it being influenced by each word's gravity:
$$P(Target|Gaze) \propto P(Gaze|Target) 	imes CM(Target)$$
The likelihood $P(Gaze|Target)$ is provided by the Gaussian probability density function $N(x_t | \mu_i, \Sigma_i)$, while $CM$ is the word's cognitive mass. This allows high-difficulty words to possess a larger "gravity radius" to absorb drifting gaze points.

### 3.3 Time-Decaying Cognitive Mass
To prevent the gaze from being permanently trapped by high-mass words (Deadlocks), the system introduces a dynamic mass model that decays with fixation time:
1. **Cumulative Exposure**: Increment the word's exposure $E_i(t)$ based on the spatial Gaussian weight between the current gaze position and the word's gravity center.
2. **Dynamic Mass Update**: Apply an exponential decay formula (default decay rate $\lambda = 0.5$):
   $$CM_i(t) = (Base\_CM_i + \epsilon) 	imes \exp(-\lambda 	imes E_i(t))$$
   A small constant $\epsilon = 0.01$ is added to prevent zero-probability deadlocks. As the reader's fixation time increases, the gravity of the word drops rapidly, forcing the gaze to naturally progress to the next word.

## 4. Expected Results & Advantages
Through M4's spatiotemporal alignment, the system successfully transforms static text features into a dynamic probability field $P(x, y, t)$ that fluctuates per frame. This dynamic prior field effectively handles high-frequency jitter and persistent vertical/horizontal drift inherent to webcams. It forcefully anchors extreme hardware noise back to paths that best match human reading cognitive characteristics, providing extremely high-quality state transition priors for M5's Viterbi Decoding and Kalman Filtering.

## 5. Ablation Study Design
To verify the effectiveness of each mechanism within M4, we designed the following ablation variants to observe changes in the system's word alignment accuracy:
1. **w/o OVP Alignment**: Removes the 35% left-shifted gravity center design, reverting to the word's geometric center. Expected to show a significant drop in alignment accuracy for short words and articles.
2. **w/o Adaptive Variance**: Removes the mechanism of dynamically adjusting $\sigma_x$ based on word length, using a fixed Gaussian radius for all words. Expected to misclassify gaze on the edges of long words (over 8 letters) as skips or drifts.
3. **w/o Temporal Decay ($\lambda = 0$)**: Disables the cumulative exposure decay mechanism. Expected the trajectory to frequently get stuck on difficult words with extremely high Surprisal, failing to smoothly simulate human saccadic jumping behaviors.

### 6. L1/L2 Adaptive Prior Calibration (Post-Presentation Update)
To resolve trajectory recovery degradation for native (L1) readers, the system introduces a proficiency-aware calibration mechanism:
- **L1 (Native Readers)**: Since native readers read rapidly, skip words frequently, and are less bound by cognitive load, the system flattens the static cognitive mass emission prior by blending it with its mean ($0.9 \times \text{mean\_cm} + 0.1 \times \text{base\_cm}$) and scales down the cognitive skipping penalty ($\gamma \times 0.1$). This relies primarily on physical transition dynamics.
- **L2 (Language Learners)**: Since L2 learners exhibit significant oculomotor pause times on high-difficulty words, the system retains full cognitive mass transition priors and regression boosts to anchor the drifting gaze path.

---

## 6. M5: Trajectory Fusion and Optimization

### Pipeline

```
Alignment (physical) → Injection (M4 gravity) → Optimization (sandbox) → per-word RDS
```

### Algorithm sandbox (Slide 18)

| Algorithm | Role |
|-----------|------|
| **Viterbi POM** | Sequence decoding with reading prior as transition model |
| **EM self-calibration** | Mid-session drift correction (presentation: 18% → 74%) |
| **Kalman filtering** | State-space smoothing of hardware jitter |
| **OPTICS / DBSCAN** | Cluster fixations around high-mass anchors |

### Six RDS fusion methods

| Method | Idea |
|--------|------|
| Linear | Weighted sum of dwell, fixations, load |
| Multiplicative | Dwell × load (interaction) |
| Attention-Gated | Gate by fixation confidence |
| Sigmoid | Bounded non-linear squash |
| **Bayesian** | Best ρ on GECO in presentation |
| RRF | Rank-based fusion |

**RDS tiers:** difficulty ≥ 0.70 · attention 0.40–0.70 · fluent < 0.40

### GECO simulated-drift benchmark (Slide 20)

156-word trial, **+45 px** vertical drift (`output/demo_system_comparison.csv`):

| Configuration | Accuracy | ρ(RDS, TRT) | Latency |
|---------------|----------|-------------|---------|
| Raw Gaze + Linear | 18.59% | 0.064 | ~1.5 ms |
| Viterbi POM + Linear | 48.72% | 0.091 | ~140 ms |
| Viterbi + EM | 73.72% | 0.205 | ~210 ms |
| **STOCK-T v3 + Bayesian** | **78.21%** | **0.226** | ~210 ms |

### GECO Noise Stress Test Benchmark (150 Sampled Trials)
To evaluate the system's robustness at scale, we conducted a stress test across 150 sampled trials from the GECO corpus under varying vertical hardware drifts ($0\text{px}$ to $60\text{px}$):

| Drift (px) | STOCK-T Edge WordAcc | STOCK-T Surprisal WordAcc (Adaptive) | Baseline WordAcc (Spatial Only) | Edge vs. Baseline (WordAcc % Impr.) | STOCK-T Surprisal LineRec | Baseline LineRec |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | 16.28% | 15.52% | **19.09%** | -14.75% | 55.69% | **66.58%** |
| **15.0** | 13.53% | 12.59% | **18.14%** | -25.40% | 48.51% | **63.28%** |
| **30.0** | **15.18%** | 14.36% | 14.91% | **+1.77%** | 53.42% | **55.64%** |
| **45.0** | 12.94% | **13.17%** | 11.50% | **+12.51%** | **51.49%** | 45.36% |
| **60.0** | 9.61% | **10.14%** | 8.24% | **+16.61%** | **40.24%** | 35.15% |

**Key takeaway**: While spatial baselines perform well under low drift ($<30\text{px}$), they collapse by over 56.8% under high noise ($60\text{px}$). `STOCK-T`'s transition matrix prevents line-leaks, maintaining **10.14%** word accuracy and **40.24%** line recovery at $60\text{px}$ drift, outperforming the baseline by **+23.1% relative word accuracy**.

### Fusion vs. TRT (157 words, clean gaze)

From `output/fusion_evaluation_summary.csv`:

| Method | Pearson r | Spearman ρ |
|--------|-----------|------------|
| Linear | 0.888 | 0.882 |
| Sigmoid | 0.849 | 0.882 |
| Multiplicative | 0.681 | 0.801 |
| Bayesian | 0.756 | 0.799 |
| RRF | 0.774 | 0.782 |
| Gated | 0.574 | 0.751 |

### Cognitive Inspector outputs

- **RDS overlay:** per-word difficulty coloring
- **Diagnostics:** WPM, regression rate, English proficiency (A1–C2), cognitive fatigue
- **Vocabulary book:** auto-compiled high-load word lists
- Markdown reports: `docs/cognitive_reports/`

**Paths:** `scripts/fusion/`, `scripts/geco/`, `core/cognitive_inspector/`, `web/routes/inspector.py`

### Figures (repo)

| File | Description |
|------|-------------|
| `output/demo_performance_comparison.png` | End-to-end system comparison |
| `output/accuracy_comparison.png` | Decoding method accuracy |
| `output/fusion_correlation_comparison.png` | Fusion vs. TRT correlations |
| `output/gaze_cognitive_space_rds.png` | Surprisal–TRT–RDS space |
| `output/top_difficult_words.png` | Top-10 difficult words |

---

## 7. Related Work

| Work | Venue | Relevance |
|------|-------|-----------|
| UniGaze: Universal Gaze Estimation | WACV 2026 | Base gaze model |
| Mobile eye-tracker calibration fitting | JEMR 2023 | Personalization |
| OneEuro filter | CHI 2012 | Real-time smoothing |
| Slip-Kalman for reading progression | IEEE SENSORS 2019 | Temporal gaze tracking |
| GECO corpus | Behav. Res. Methods 2017 | Primary benchmark |
| PROVO corpus | Behav. Res. Methods 2018 | Cross-corpus validation |
| Surprisal & model size | TACL 2023 | Scaling paradox |

---

## 8. Post-Presentation Progress

*Not covered in the final slides; include selectively.*

| Area | Description |
|------|-------------|
| **Unified Flask app** | `core/` + `web/` refactor; guided calibration stepper, participant-ID sessions, cognitive-mass visualization |
| **Agentic cognitive quiz** | Gemini API adaptive reading quiz + diagnostics (`inspector` routes) |
| **Video mode** | OpenCV frame matching, Bayesian attraction snapping (BERT surprisal + Zipf), sparse tick normalization |
| **Large-scale GECO** | 37-subject evaluation, OVP flushing under L2 load, noise tests ~82% at 60 px drift (`docs/NeurIPS/`) |
| **Integration tests** | Adaptive stepper E2E, LLM JSON sanitization |
| **L1/L2 Adaptive Calibration** | Mitigated L1 reader accuracy degradation (from 9.83% back up to 16.18%) by scaling down cognitive priors for native readers while preserving semantic guidance for L2 readers. |

---

## 9. Discussion and Future Work

**Accuracy vs. latency:** Full Viterbi+EM ≈ 210 ms/batch (quasi-real-time); geometric nearest-box is fast but inaccurate.

**Corpus scope:** M3 validated on English GECO/PROVO; Chinese BERT at API layer lacks comparable eye-tracking benchmarks.

**Next steps:**

- Personalized POM priors (L2 proficiency)
- Real-time multi-line EM for long documents
- Formal LOSO/bootstrap reporting
- NeurIPS manuscript prep (`docs/NeurIPS/manuscripts/`)

---

## 10. Conclusion

LexiGaze implements an M1–M5 pipeline fusing webcam gaze, language-model cognitive features, and spatiotemporal priors into interpretable RDS scores. At presentation time we demonstrated statistically significant cross-corpus cognitive predictions, gaze decoding up to **78.21%** under extreme drift, and Cognitive Inspector outputs for actionable reading diagnostics.

---

## 11. References

- Cop, U., Dirix, N., Drieghe, D., & Hartsuiker, R. J. (2017). Presenting GECO. *Behavior Research Methods*, 49(2), 602–615.
- Luke, S. G., & Christianson, K. (2018). The PROVO Corpus. *Behavior Research Methods*, 50(3), 826–833.
- Rayner, K. (1998). Eye movements in reading and information processing. *Psychological Bulletin*, 124(3), 372–422.
- Hale, J. (2001). A probabilistic Earley parser as a psycholinguistic model. *NAACL*.
- Levy, R. (2008). Expectation-based syntactic comprehension. *Cognition*, 106(3), 1126–1177.
- Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings. *Behavior Research Methods*, 44(4), 978–990.
- Oh, B.-D., & Schuler, W. (2023). Transformer-based LM surprisal predicts reading times. *TACL*.
- Casiez, G., Roussel, N., & Vogel, D. (2012). OneEuro filter. *CHI*.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.
- Devlin, J., et al. (2019). BERT. *NAACL-HLT*.
