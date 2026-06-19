# LexiGaze Cognitive Load Pipeline — weichi

An NLP pipeline for predicting **per-word cognitive load** in English text, validated against eye-tracking reading data. Designed to be integrated with eye-tracking reading research experiments.

---

## Pipeline v9 — What's inside

| Component | Implementation | Note |
|-----------|---------------|------|
| Language model | GPT-2 surprisal | Smaller = better: GPT-2 (117M) outperforms larger models |
| Entropy | Rényi entropy (α = 0.5) | Better than Shannon for first-pass reading prediction |
| Lexical features | AoA (Kuperman 2012) + Zipf | AoA has significant independent contribution beyond frequency |
| Syntactic load | dep_load (POS-gated: NOUN/VERB/PROPN only) | Non-content word dep_load set to 0.0 |
| Scoring backend | **XGBoost** (`xgb_model.json`) | Non-linear, breaks zipf collinearity; trained on 2,000 GECO sentences |

---

## Validation Results (Eye-Tracking)

All results are on **completely held-out data** — no data leakage.

### GECO Held-out (1,000 sentences, 4,883 content words)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Spearman ρ (TRT) | **0.437 \*\*\*** | [0.412, 0.458] |
| Spearman ρ (GD)  | **0.386 \*\*\*** | [0.359, 0.409] |
| LMM β — TRT | 0.049 \*\*\* | LRT χ²(1) = 184.61, ΔAIC = +182.6 |
| LMM β — GD  | 0.029 \*\*\* | LRT χ²(1) = 84.77,  ΔAIC = +82.8  |

> LMM = per-reader random intercepts, controlling for word length, Zipf, sentence position, and preceding-word spillover.

### PROVO Zero-Shot (55 passages, 84 L1 readers — never seen during training)

| Metric | GECO held-out | PROVO (zero-shot) |
|--------|--------------|-------------------|
| ρ (TRT) | 0.437 \*\*\* | **0.619 \*\*\*** |
| ρ (GD)  | 0.386 \*\*\* | **0.611 \*\*\*** |

### Robustness

- **Bootstrap 95% CI** (2,000 iterations): ρ(TRT) = [0.413, 0.459] — estimate is stable
- **LOSO**: 14/14 individual GECO readers all p < .001, mean ρ = 0.215 ± 0.044

---

## Quick Start

```python
from cognitive_load_pipeline import CognitiveLoadPipeline

# First run downloads GPT-2 (~5 sec); subsequent runs load from cache
pipeline = CognitiveLoadPipeline(model_type='gpt2', lang='en')

result = pipeline.run("The ubiquitous phenomenon completely bewildered the researchers.")

# High-load words
print(result["high_load_words"])
# → ['ubiquitous', 'phenomenon', 'bewildered']

# Word-level scores
for w in result["word_analysis"]:
    print(f"{w['word']:<15} score={w['load_score']:.3f}  level={w['load_level']}")
```

---

## Integration Guide for Group Members

This section explains how to plug this pipeline into your own reading/eye-tracking experiment.

### Step 1 — Required files

The following files must be present in `lexigaze/weichi/`:

| File | Source | Required? |
|------|--------|-----------|
| `cognitive_load_pipeline.py` | This repo | ✅ Yes |
| `xgb_model.json` | This repo (committed) | ✅ Yes (for XGBoost scoring) |
| `ridge_model.json` | This repo (committed) | ✅ Yes (Ridge fallback) |
| `GECO_data/AoA_Kuperman.csv` | **Not committed** — download separately | Recommended |
| `GECO_data/EnglishMaterial(ALL).csv` | **Not committed** — GECO corpus | Only if running validation scripts |
| `GECO_data/MonolingualReadingData.csv` | **Not committed** — GECO corpus | Only if running validation scripts |

> ⚠️ `GECO_data/` and `PROVO_data/` are excluded from git (large raw data).  
> Ask for the data files directly — do NOT commit them.

### Step 2 — Install dependencies

```bash
pip install torch transformers spacy numpy wordfreq xgboost statsmodels scipy pandas
python -m spacy download en_core_web_sm
```

### Step 3 — Get per-word load scores

```python
from cognitive_load_pipeline import CognitiveLoadPipeline

pipeline = CognitiveLoadPipeline(model_type='gpt2', lang='en')

def get_load_scores(sentence: str) -> list[dict]:
    """Returns word-level cognitive load scores (0–1) for a sentence."""
    result = pipeline.run(sentence)
    return result["word_analysis"]

words = get_load_scores("Quantum entanglement describes a non-local correlation.")
for w in words:
    print(f"{w['word']:<20} load={w['load_score']:.3f}  ({w['load_level']})")
```

### Step 4 — What `load_score` means

- Range: **0.0 – 1.0** (higher = more cognitively demanding)
- Threshold: **70th percentile** within the document (relative, not absolute)
- Label: `"high"` / `"medium"` / `"low"` based on threshold

Each word's score reflects:
- How **unpredictable** it is in context (GPT-2 surprisal)
- How **late it is acquired** (AoA score)
- How **rare** it is (Zipf frequency, inverse)
- How **syntactically complex** it is (dep_load, content words only)
- **Word length** (characters)

### Step 5 — Connecting to eye-tracking data

If you have per-word eye-tracking data (TRT, GD, etc.), match on word surface form + sentence position:

```python
# Your eye-tracking data format (example)
# eye_df: columns = [sentence_id, word, word_position, TRT, GD]

for sentence_id, group in eye_df.groupby("sentence_id"):
    sentence_text = " ".join(group["word"].tolist())
    load_scores = get_load_scores(sentence_text)

    for i, row in enumerate(group.itertuples()):
        if i < len(load_scores):
            load = load_scores[i]["load_score"]
            # Now correlate load with row.TRT or row.GD
```

> **Important**: Align by **word position**, not surface string — GPT-2 tokenisation may split hyphenated words differently. Always verify alignment before running statistics.

### Key things to watch out for

| Issue | What to do |
|-------|-----------|
| GPT-2 tokenisation splits a word (e.g. "neuro-symbolic" → 3 tokens) | Pipeline re-aggregates sub-tokens automatically |
| Function words (articles, prepositions) have near-zero load_score | Expected — POS gate excludes them from dep_load; surprisal is also low |
| `load_score = 0.0` for ALL words | Check if `xgb_model.json` is present — pipeline falls back to Ridge or heuristic if missing |
| Very short sentences (< 5 words) | Surprisal unreliable at sentence start; consider skipping first 2 words |
| Cross-sentence context | Use `cognitive_memory` parameter to carry forward momentum across sentences |

---

## Retraining the XGBoost model

If you want to train on your own eye-tracking corpus instead of GECO:

```bash
cd lexigaze/weichi
python train_xgb_geco.py   # modify TRAIN_N / data paths inside the script
```

Output: `xgb_model.json` — replace the existing file.

---

## Validation Scripts

| Script | What it does | Input needed |
|--------|-------------|--------------|
| `validate_geco.py` | Baseline validation on 150 GECO sentences | GECO corpus |
| `full_validation.py` | Full paper-level validation (2000 train / 1000 test) | GECO corpus |
| `validate_provo.py` | Zero-shot cross-corpus test on PROVO | PROVO corpus |
| `validate_remaining_geco.py` | Large-scale stability (3183 sentences) | GECO corpus |
| `gd_trt_separate_models.py` | GD / TRT separate OLS + LMM | Cached `test_predictions.csv` |
| `robustness_analysis.py` | Bootstrap CI + LOSO per-reader | Cached `test_predictions.csv` |
| `compare_models.py` | GPT-2 vs GPT-2-Large vs TinyLlama, etc. | GECO corpus |

---

## File Structure

```
lexigaze/weichi/
├── cognitive_load_pipeline.py   # ← Main pipeline (integration target)
├── xgb_model.json               # ← Trained XGBoost weights (committed)
├── ridge_model.json             # ← Trained Ridge weights (fallback, committed)
│
├── train_xgb_geco.py            # Retrain XGBoost on GECO
├── train_ridge_geco.py          # Retrain Ridge on GECO
│
├── validate_geco.py             # Core validation + merge_eye utility
├── full_validation.py           # Paper-level validation (OLS + LMM + Bootstrap)
├── validate_provo.py            # PROVO zero-shot
├── validate_remaining_geco.py   # Large-scale cross-section
├── gd_trt_separate_models.py    # GD / TRT separate OLS + LMM
├── robustness_analysis.py       # Bootstrap CI + LOSO
├── compare_models.py            # LM comparison
│
├── GECO_data/                   # ← NOT committed (ask for files)
│   ├── EnglishMaterial(ALL).csv
│   ├── MonolingualReadingData.csv
│   └── AoA_Kuperman.csv
└── PROVO_data/                  # ← NOT committed
    └── Provo_Corpus-Eyetracking_Data.csv
```

---

## Version History

| Version | Key Change | Eye-tracking ρ (TRT) |
|---------|-----------|----------------------|
| v1–v5 | BERT → GPT-2 → AoA → collocation boost | — (F1-based eval only) |
| v6 | Ridge Regression backend (GECO-trained) | 0.420 |
| v7 | Domain-adaptive scoring | 0.420 |
| v8 | ADJ gate + chunked `process_file` | 0.420 |
| **v9** | Rényi entropy + POS-gate dep_load + **XGBoost** | **0.437** (CI [0.412, 0.458]) |

---

## References

- Cop, U., et al. (2017). Presenting GECO. *Behavior Research Methods*. — GECO corpus
- Kuperman, V., et al. (2012). Age-of-acquisition ratings for 30,000 English words. *BRM*. — AoA norms
- Luke, S. G., & Christianson, K. (2018). The Provo Corpus. *BRM*. — PROVO corpus
- Oh, B.-D., & Schuler, W. (2023). Why Does Surprisal From Larger Transformer-Based Language Models Provide Less Regression Fit to Human Reading Times? *TACL*. — GPT-2 scaling paradox
- Pimentel, T., et al. (2023). On the Effect of Anticipation on Reading Times. *TACL*. — Rényi entropy
