# Full GECO Validation Report — Pipeline v9 (Paper-Level)

> Training: 2000 sentences  |  XGB internal val: 100  |  Held-out: 1000 sentences
> Test content words (≥3 readers): **4883**
> OLS controls: word length, Zipf, sentence position,
>   prev-word surprisal + length (spillover).
> LMM: per-reader random intercepts (14 subjects, 49154 observations)
> Bootstrap: 1000 iterations, 95% CI via percentile method.

---

## Key Results

| Metric | Value | 95% CI | Sig. |
|--------|-------|--------|------|
| Spearman ρ (TRT) | 0.437 | [0.412, 0.458] | *** |
| Spearman ρ (GD)  | 0.386  | [0.359, 0.409] | *** |
| Held-out R² (log TRT) | 0.1882 | — | — |

## OLS Regression (Word-level Mean TRT)

| Parameter | Value | Sig. |
|-----------|-------|------|
| β(xgb_load) | 0.6388 | *** |
| ΔR²         | 0.0175 | — |
| ΔAIC        | 104.6 | — |

## LMM (Per-Reader, Random Intercepts)

| Parameter | Value | Sig. |
|-----------|-------|------|
| β(xgb_load_z) | 0.0490 | *** |
| LRT χ²(1)     | 184.61 | *** |
| ΔAIC          | 182.6 | — |
| N (obs)       | 49154 | — |
| N (subjects)  | 14 | — |

---

## Paper-Ready Quote
> "The pipeline predicted mean TRT with Spearman ρ = 0.437 (95% CI [0.412, 0.458])
> and GD ρ = 0.386 (95% CI [0.359, 0.409])
> on 4883 content words from 1000 held-out GECO sentences.
> The load score independently predicted TRT after controlling for
> word frequency, length, sentence position, and spillover
> (OLS β = 0.639, p < .001, ΔAIC = 104.6;
> LMM β = 0.049, LRT χ²(1) = 184.61, p < .001, ΔAIC = 182.6)."