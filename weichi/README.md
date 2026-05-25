# LexiGaze Cognitive Load Pipeline

An NLP pipeline for detecting **cognitively demanding words** in English text using GPT-2 surprisal, lexical frequency, Age-of-Acquisition (AoA), and syntactic dependency load. Designed to be integrated with eye-tracking reading research.

---

## Features

- **Surprisal-based scoring** via GPT-2 (or BERT for masked-LM mode)
- **Age-of-Acquisition (AoA)** features from Kuperman (2012) norms
- **Domain-adaptive scoring**: automatic detection of academic vs. general text, with Ridge Regression for general text
- **Collocation boost**: flags words that co-occur with adjacent high-load neighbors
- **ADJ precision gate**: prevents common adjectives from being over-flagged
- **Hyphenated compound re-aggregation**: `neuro` + `-` + `symbolic` → `neuro-symbolic`
- **Chunked document processing** (`process_file`): 2.9× faster than sentence-by-sentence inference

---

## Performance (v8)

### Benchmark Accuracy (English)

| Benchmark | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| Academic text (3 sentences, GT=9 words) | 88.9% | 88.9% | **88.9%** |
| GECO corpus (Christie novel, GT=17 words) | 56.0% | 82.4% | **66.7%** |

### Inference Speed

| Scenario | Time | Per word |
|----------|------|----------|
| Short sentence (~11 words) | ~32 ms | ~2.95 ms/word |
| Medium passage (~65 words) | ~74 ms | ~1.14 ms/word |
| Long document via `run()` × 25 sentences | ~834 ms | — |
| Long document via `process_file()` (v8) | ~289 ms | ~0.88 ms/word |

**Stage breakdown** (medium text, CPU):

| Stage | Time | Share |
|-------|------|-------|
| spaCy parsing | ~15 ms | 18% |
| GPT-2 inference | ~63 ms | 78% |
| Scoring + finalize | ~3 ms | 4% |

---

## Installation

### Requirements

```
torch
transformers
spacy
numpy
wordfreq
opencc-python-reimplemented
jieba
pymupdf          # for PDF support (optional)
```

### Setup

```bash
pip install torch transformers spacy numpy wordfreq opencc-python-reimplemented jieba pymupdf

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### Optional: AoA dictionary and Ridge model

Place the following files in the same directory as `cognitive_load_pipeline.py`:

| File | Purpose |
|------|---------|
| `GECO_data/AoA_Kuperman.csv` | Kuperman (2012) Age-of-Acquisition norms (enables AoA feature) |
| `ridge_model.json` | Trained Ridge Regression weights for general-domain text (enables domain-adaptive scoring) |

If these files are absent, the pipeline falls back to heuristic scoring only (still functional).

---

## Quick Start

```python
from cognitive_load_pipeline import CognitiveLoadPipeline

# Initialize once — loads GPT-2 and spaCy (~5 seconds on first run)
pipeline = CognitiveLoadPipeline(model_type='gpt2', lang='en')

# Single sentence
result = pipeline.run("The ubiquitous phenomenon completely bewildered the researchers.")

print(result["high_load_words"])
# → ['ubiquitous', 'phenomenon', 'bewildered']

# Detailed word-level output
for w in result["word_analysis"]:
    print(f"{w['word']:<15} score={w['load_score']:.3f}  level={w['load_level']}")
```

---

## API Reference

### `CognitiveLoadPipeline(model_type, lang)`

| Parameter | Options | Default |
|-----------|---------|---------|
| `model_type` | `'gpt2'`, `'gpt2-medium'`, `'bert'` | `'bert'` |
| `lang` | `'en'`, `'zh'` | `'zh'` |

**Recommended for English:** `model_type='gpt2', lang='en'`

---

### `pipeline.run(text, cognitive_memory=None, domain="auto") → dict`

Processes a single sentence or short paragraph.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Input text |
| `cognitive_memory` | `[float, float]` | Previous two words' load scores for carry-over momentum. Defaults to `[0.0, 0.0]`. |
| `domain` | `str` | `"auto"` (detect), `"academic"`, or `"general"` |

**Returns:**

```python
{
    "model": "gpt2",
    "lang": "en",
    "domain": "academic",          # detected domain
    "process_time_ms": 74,
    "high_load_words": ["ubiquitous", "phenomenon", "bewildered"],
    "word_analysis": [
        {
            "word": "ubiquitous",
            "pos": "ADJ",
            "position": 1,
            "surprisal": 8.432,
            "entropy": 3.21,
            "dependency_load": 0.15,
            "zipf_score": 3.2,
            "word_length": 10,
            "pos_score": 0.9,
            "load_level": "high",   # "high" / "medium" / "low"
            "load_score": 0.723,    # 0.0 – 1.0
            "aoa_score": 0.81       # 0.0 – 1.0 (higher = learned later)
        },
        ...
    ]
}
```

---

### `pipeline.process_file(file_path, output_path=None, domain="auto") → dict`

Processes a full document (`.txt`, `.md`, or `.pdf`). **Use this instead of calling `run()` in a loop** — it applies threshold and boosting at the document level, and is ~3× faster.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to `.txt`, `.md`, or `.pdf` |
| `output_path` | `str` or `None` | If provided, writes JSON result to this path |
| `domain` | `str` | `"auto"`, `"academic"`, or `"general"` |

**Returns:** Same structure as `run()`, without `process_time_ms`.

**Example:**

```python
result = pipeline.process_file("paper.pdf", output_path="result.json")
print(f"High-load words: {result['high_load_words']}")
```

---

## Integration Example

The typical integration pattern for eye-tracking research:

```python
from cognitive_load_pipeline import CognitiveLoadPipeline

pipeline = CognitiveLoadPipeline(model_type='gpt2', lang='en')

def get_cognitive_load(text: str) -> list[dict]:
    """Return word-level load scores for a given reading stimulus."""
    result = pipeline.run(text)
    return result["word_analysis"]

# Each entry has: word, load_score (0–1), load_level (high/medium/low)
words = get_cognitive_load("Quantum entanglement describes a non-local correlation.")
high = [w for w in words if w["load_level"] == "high"]
```

For pre-processing a stimulus set:

```python
stimuli = ["sentence one ...", "sentence two ..."]

pipeline = CognitiveLoadPipeline(model_type='gpt2', lang='en')
mem = [0.0, 0.0]

for sentence in stimuli:
    result = pipeline.run(sentence, cognitive_memory=mem)
    # carry cognitive momentum across sentences
    if result["word_analysis"]:
        last = result["word_analysis"][-1]
        mem = [last["load_score"], mem[0]]
    print(result["high_load_words"])
```

---

## File Structure

```
lexigaze/weichi/
├── cognitive_load_pipeline.py   # Main pipeline (this is the integration target)
├── benchmark_en.py              # Academic text benchmark (F1 evaluation)
├── bench_speed.py               # Inference speed measurement
├── plot_progress.py             # Version progress chart (v1–v8 accuracy)
├── plot_speed.py                # Speed analysis chart
├── ridge_model.json             # Trained Ridge weights (optional)
└── GECO_data/
    └── AoA_Kuperman.csv         # AoA norms (optional)
```

---

## Scoring Details

Each word is scored with seven features, weighted by part-of-speech:

| Feature | Description |
|---------|-------------|
| `surprisal` | GPT-2 negative log-probability (contextual predictability) |
| `entropy` | Token-level prediction uncertainty |
| `interaction` | `surprisal × frequency` — captures rare + unpredictable words |
| `dependency_load` | Syntactic integration cost from dependency tree |
| `tree_depth` | Depth in dependency parse tree |
| `aoa_score` | Age-of-Acquisition (later-learned → higher score) |
| `zipf_score` | Inverse word frequency (`7 − zipf_frequency`) |

**POS-based weight branches:**

- **Content words** (NOUN, VERB, PROPN): higher surprisal weight
- **Modifiers** (ADJ, ADV): lower surprisal, higher AoA and interaction

**Classification thresholds:** 70th percentile of document scores (floor: 0.18 for English, 0.30 for Chinese).

---

## Version History

| Version | Key Change | Academic F1 | GECO F1 |
|---------|-----------|-------------|---------|
| v1 | BERT baseline | 14.3% | — |
| v2 | Switch to GPT-2 | 38.7% | — |
| v3 | 70th percentile threshold + abbreviation filter | 88.9% | — |
| v4 | Collocation boost | 94.7% | 62.5% |
| v5 | AoA feature | 94.7% | 62.5% |
| v6 | Ridge Regression (GECO-trained) | 70.6% | 66.7% |
| v7 | Domain-adaptive scoring | 76.2% | 66.7% |
| **v8** | ADJ weight fix + boost proximity + ADJ gate + chunked `process_file` | **88.9%** | **66.7%** |

---

## Citation / References

- Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings for 30,000 English words. *Behavior Research Methods*.
- Radford, A., et al. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*.
- The GECO corpus: Cop, U., et al. (2017). Presenting GECO: An eye-tracking corpus of monolingual and bilingual sentence reading. *Behavior Research Methods*.
