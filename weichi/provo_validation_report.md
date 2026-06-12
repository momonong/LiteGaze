# PROVO External Validation Report — Pipeline v9

> **Corpus**: PROVO (Luke & Christianson, 2018) — 55 passages, 84 participants
> **Text genres**: newspaper, Wikipedia, narrative (mixed)
> **Model**: xgb_model.json trained on GECO (Christie novel) — **zero-shot transfer**
> **Content words evaluated**: 1592 (across 55 texts, ≥3 readers)

---

## Results

| Metric | Value | Sig. |
|--------|-------|------|
| Spearman ρ (TRT) | 0.619 | *** |
| Spearman ρ (GD)  | 0.611  | *** |
| OLS β(xgb_load)  | 0.6525 | *** |
| OLS ΔR²          | 0.0253 | — |
| OLS ΔAIC         | 63.2 | — |

---

## Cross-Corpus Comparison

| Corpus | Genre | Participants | Words | ρ (TRT) | ρ (GD) |
|--------|-------|-------------|-------|---------|--------|
| GECO (train) | Fiction (Christie) | 14 L1 | 9,793 | — | — |
| GECO (held-out) | Fiction (Christie) | 14 L1 | 4,882 | 0.437 | 0.388 |
| GECO (full remaining) | Fiction (Christie) | 14 L1 | 16,318 | 0.440 | 0.400 |
| **PROVO (zero-shot)** | **Mixed genres** | **84 L1** | **1592** | **0.619** | **0.611** |

---

## Interpretation
- **Strong zero-shot transfer**: ρ(TRT) = 0.619 on an entirely different corpus
  confirms the pipeline captures domain-general cognitive load signals.
- The model was trained exclusively on GECO (Christie fiction) yet generalizes
  to PROVO's mixed-genre passages (newspaper/Wikipedia/narrative).

## Paper-Ready Quote
> "Zero-shot transfer to the PROVO corpus (Luke & Christianson, 2018; 55 passages, 84 participants, mixed genres) yielded Spearman ρ = 0.619 (GD: ρ = 0.611, both p < .001) on 1592 content words, confirming cross-corpus generalization (OLS β = 0.652, p < .001, ΔAIC = 63.2)."