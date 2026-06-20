# LM Model Comparison Report — Surprisal Predictors of Eye-Tracking TRT

> 100 GECO sentences  |  665 content words (≥3 readers)  |  DV: mean TRT

---

## Results

| Model | Spearman ρ | Sig. | OLS β | OLS p | ΔR² (incremental) |
|-------|-----------|------|-------|-------|-------------------|
| GPT-2 (117M) | 0.398 | *** | 0.0182 | 0.0000 | 0.0396 |
| GPT-2-L (774M) | 0.355 | *** | 0.0143 | 0.0000 | 0.0273 |
| GPT-2-XL (1.5B) | 0.345 | *** | 0.0141 | 0.0000 | 0.0260 |
| GPT-Neo (1.3B) | 0.347 | *** | 0.0152 | 0.0000 | 0.0327 |
| TinyLlama (1.1B) | 0.362 | *** | 0.0146 | 0.0000 | 0.0334 |

---

## Interpretation

- **Spearman ρ (marginal)**: correlation between that model's per-word surprisal and mean TRT.
  Higher ρ = better alignment with human reading difficulty.
- **OLS β**: partial effect of surprisal on log TRT after controlling for word length,
  Zipf frequency, and sentence position.
- **ΔR²**: incremental variance in log TRT explained by adding surprisal to the baseline model.
  This is the cleanest measure of the surprisal's *unique* contribution.

### Reference: base pipeline
- Base pipeline uses `_compute_gpt` with `is_split_into_words=True` (sum BPE NLLs).
  Result: GPT-2 (117M) ρ = 0.424*** (n=1044, 150 sentences) — validated baseline.
- This comparison script uses SurprisalCalc with full-sentence `offset_mapping` for ALL models,
  enabling fair cross-model comparison.

### SOTA context
- Published benchmarks on GECO/Dundee: GPT-2 ρ ≈ 0.40–0.45;
  GPT-3 / LLaMA-7B ρ ≈ 0.45–0.52 (Pimentel et al. 2023; Oh & Schuler 2022).
- Model size helps up to ~1–3B params, then diminishing returns for reading-time prediction.

---

## Files
- `model_comparison.png` — ρ and ΔR² bar charts
- `component_report.md` — Phase A component breakdown
- `validation_report.md` — Base pipeline GECO validation (ρ = 0.42***)