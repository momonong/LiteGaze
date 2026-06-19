# Remaining GECO Validation — Pipeline v9

> **Training range**: sentences 1–2100 (XGBoost trained on these)
> **Test range**: sentences 2101–5284 (2920 sentences, completely unseen)
> **Test content words**: 16318 (≥3 readers)
> **Purpose**: within-corpus cross-section generalization (different chapters of Christie novel)

## Results

| Metric | Value | Sig. |
|--------|-------|------|
| Spearman ρ (TRT) | 0.440 | *** |
| Spearman ρ (GD)  | 0.400  | *** |
| Held-out R² (log TRT) | 0.2027 | — |
| OLS β(xgb_load) | 0.7068 | *** |
| OLS ΔR² | 0.0168 | — |
| OLS ΔAIC | 342.8 | — |

## Comparison to Held-out Test (sentences 2101–3100)

| | Held-out (2101–3100) | Remaining (3101–5284) |
|-|---------------------|----------------------|
| n words | 4,882 | 16318 |
| ρ (TRT) | 0.437 *** | 0.440 *** |
| ρ (GD)  | 0.388 *** | 0.400 *** |
| R² | 0.189 | 0.203 |
| OLS ΔAIC | +104.1 | +342.8 |

## Interpretation
- Consistent ρ across different sections of GECO confirms pipeline stability
- OLS β significant in both sections → load_score has independent contribution
  after controlling for frequency, length, position, and spillover
- Next step: validate on a different corpus (PROVO or CELER) for true cross-corpus generalization