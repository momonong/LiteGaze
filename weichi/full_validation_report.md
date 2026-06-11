# Full GECO Validation Report — Pipeline v9 (Paper-Level)

> Training: 600 sentences  |  XGB internal val: 100  |  Held-out test: 1000 sentences
> Test content words (≥3 readers): **4571**
> All test sentences completely unseen during training.

---

## Key Results

| Metric | Value | Sig. |
|--------|-------|------|
| Spearman ρ (TRT) | 0.393 | *** |
| Spearman ρ (GD)  | 0.362  | *** |
| Held-out R² (log TRT) | 0.1069 | — |
| OLS β(xgb_load) | 0.7657 | *** |
| OLS ΔR² | 0.0227 | — |
| OLS ΔAIC | 127.6 | — |

---

## Pipeline v9 Features
| Feature | Role | Source |
|---------|------|--------|
| GPT-2 surprisal | Contextual predictability | Oh & Schuler 2023 |
| Rényi entropy α=0.5 | Anticipatory load | Pimentel et al. 2023 |
| AoA (Kuperman 2012) | Lexical acquisition age | Dirix & Duyck 2017 |
| Zipf frequency | Lexical access speed | Brysbaert & New 2009 |
| POS-gated dep_load | Syntactic integration (NOUN/VERB only) | Rathi 2021 |
| XGBoost backend | Non-linear feature combination | Salicchi et al. 2022 |

---

## Comparison to SOTA
| System | ρ (TRT/GD) | Notes |
|--------|-----------|-------|
| **Pipeline v9 (this work)** | 0.393 / 0.362 | Held-out |
| Pipeline v8 (GPT-2+Ridge) | 0.420 / 0.375 | 150 sent. |
| GPT-2 surprisal only | ~0.35-0.40 | Literature |
| SOTA ceiling (ISC) | ~0.50-0.60 | Human upper bound |

---

## Paper-Ready Quote
> "The cognitive load pipeline (GPT-2 surprisal, Rényi entropy, AoA, syntactic
> dependency load, XGBoost) predicted mean TRT with Spearman ρ = 0.393
> (GD: ρ = 0.362, both p < .001) on 4571 content words
> from 1000 held-out GECO sentences. The load_score independently predicted
> TRT (OLS β = 0.766, p < .001, ΔR² = 0.0227)
> after controlling for word frequency, length, and sentence position."