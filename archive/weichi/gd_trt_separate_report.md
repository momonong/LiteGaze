# GD / TRT Separate Models — Pipeline v9

## OLS Regression (Word-level Mean, Spillover Controlled)

| Measure | β(xgb_load) | ΔR² | ΔAIC | Sig. | n |
|---------|------------|-----|------|------|---|
| TRT | 0.6388 | 0.0175 | 104.6 | *** | 4883 |
| GD  | 0.4055 | 0.0111 | 62.5 | *** | 4883 |

## LMM (Per-Reader Random Intercepts)

| Measure | β(xgb_load_z) | LRT χ²(1) | ΔAIC | Sig. | N obs | N subj |
|---------|--------------|-----------|------|------|-------|--------|
| TRT | 0.0490 | 184.61 | 182.6 | *** | 49154 | 14 |
| GD | 0.0292 | 84.77 | 82.8 | *** | 49077 | 14 |

---
## Paper-Ready Quote
> "The pipeline independently predicted both eye-tracking measures after
> controlling for word frequency, length, sentence position, and spillover.
> TRT: OLS β = 0.639, p < .001, ΔAIC = 104.6;
> GD: OLS β = 0.406, p < .001, ΔAIC = 62.5.
> Mixed-effects models confirmed both effects:
> TRT: LMM β = 0.049, LRT χ²(1) = 184.61, p < .001, ΔAIC = 182.6 (49154 reader×word obs);
> GD: LMM β = 0.029, LRT χ²(1) = 84.77, p < .001, ΔAIC = 82.8 (49077 obs)."