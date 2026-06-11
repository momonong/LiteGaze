# Full GECO Validation Report — Pipeline v9 (Paper-Level)

> Training: 2000 sentences  |  XGB internal val: 100  |  Held-out test: 1000 sentences
> Test content words (≥3 readers): **4882**
> All test sentences completely unseen during training.
> OLS controls: word length, Zipf frequency, sentence position,
>   prev-word surprisal (spillover), prev-word length (spillover).

---

## Key Results

| Metric | Value | Sig. |
|--------|-------|------|
| Spearman ρ (TRT) | 0.437 | *** |
| Spearman ρ (GD)  | 0.388  | *** |
| Held-out R² (log TRT) | 0.1891 | — |
| OLS β(xgb_load) | 0.6622 | *** |
| OLS ΔR² | 0.0174 | — |
| OLS ΔAIC | 104.1 | — |

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
| **Pipeline v9 (this work)** | 0.437 / 0.388 | Held-out |
| Pipeline v8 (GPT-2+Ridge) | 0.420 / 0.375 | 150 sent. |
| GPT-2 surprisal only | ~0.35-0.40 | Literature |
| SOTA ceiling (ISC) | ~0.50-0.60 | Human upper bound |

---

## Paper-Ready Quote
> "The cognitive load pipeline (GPT-2 surprisal, Rényi entropy, AoA, syntactic
> dependency load, XGBoost) predicted mean TRT with Spearman ρ = 0.437
> (GD: ρ = 0.388, both p < .001) on 4882 content words
> from 1000 held-out GECO sentences. The load_score independently predicted
> TRT (OLS β = 0.662, p < .001, ΔR² = 0.0174)
> after controlling for word frequency, length, and sentence position."