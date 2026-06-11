# Component-wise Regression Report — Pipeline Feature Analysis

> n = 1044 content words (GECO, 142 sentences, 14 L1 readers)

---

## 1. Marginal Spearman Correlation (each feature vs mean TRT)

Each feature tested alone, without controlling for others.

| Feature | Spearman ρ | p-value | Sig. |
|---------|-----------|---------|------|
| Surprisal (GPT-2) | 0.424 | 0.0000 | *** |
| AoA score (Kuperman) | 0.276 | 0.0000 | *** |
| Dependency load | -0.021 | 0.5079 | n.s. |
| Zipf freq (inverse) | -0.429 | 0.0000 | *** |
| Word length | 0.456 | 0.0000 | *** |

---

## 2. Individual OLS (each feature as sole predictor)

| Feature | β | p(β) | R² |
|---------|---|------|-----|
| Surprisal (GPT-2) | 0.0251 | 0.0000 | 0.1905 |
| AoA score (Kuperman) | 0.4164 | 0.0000 | 0.1016 |
| Dependency load | -0.0061 | 0.8484 | 0.0000 |
| Zipf freq (inverse) | -0.1173 | 0.0000 | 0.2272 |
| Word length | 0.0560 | 0.0000 | 0.2328 |

---

## 3. Joint Model (all features together, no composite score)

Formula: `log_trt ~ surprisal + aoa_score + dep_load + zipf_score + WORD_LENGTH`
R² = 0.3077  |  AIC = -174.9

| Feature | Partial β | p-value | Sig. |
|---------|----------|---------|------|
| Surprisal (GPT-2) | 0.0112 | 0.0000 | *** |
| AoA score (Kuperman) | 0.1155 | 0.0027 | ** |
| Dependency load | -0.0283 | 0.2911 | n.s. |
| Zipf freq (inverse) | -0.0346 | 0.0005 | *** |
| Word length | 0.0307 | 0.0000 | *** |

---

## 4. Composite score vs Joint model

| Model | R² | AIC |
|-------|----|-----|
| Composite `load_score` + length + freq + pos | 0.2894 | -149.7 |
| Joint (all raw features) | 0.3077 | -174.9 |

---

## 5. Interpretation

- **Surprisal** and **Zipf frequency** are expected to dominate — they are the strongest
  psycholinguistic predictors in the reading-time literature.
- **AoA** should show independent contribution *beyond* frequency (Kuperman et al. 2012
  show AoA explains unique variance in naming and reading after controlling for frequency).
- **Dependency load** tests syntactic integration cost; effect is typically smaller
  for naturalistic text but meaningful for complex sentences.
- The joint model R² gives an upper bound on how much variance all pipeline features
  collectively explain in TRT.