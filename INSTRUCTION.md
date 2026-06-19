# 🔭 LexiGaze — How to Run

LexiGaze is a Flask web application that fuses three research pipelines into one tool:

- **Document coordinate extraction** — parse PDF/HTML/TXT/DOCX/MD into pixel-level word bounding boxes
- **Gaze tracking** — real-time UniGaze-B neural network + MediaPipe face detection via webcam
- **Cognitive load analysis** — GPT-2 (English) or BERT (Chinese) word-difficulty scoring

The server logic lives inside the `web/` package, and the core modules live under the `core/` package. The entry point of the project is the `run.py` script at the root of the repository.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Install Dependencies](#3-install-dependencies)
4. [Configure Environment Variables](#4-configure-environment-variables)
5. [Run the Server](#5-run-the-server)
6. [Using the Web Interface](#6-using-the-web-interface)
7. [API Reference](#7-api-reference)
8. [Research & Standalone Scripts](#8-research--standalone-scripts)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | **3.11** | Strictly required. 3.10 is acceptable. 3.9 or below will NOT work. |
| **pip** | Latest | `pip install --upgrade pip` inside your env |
| **Webcam** | Any USB / built-in | Required only for gaze tracking |
| **CUDA GPU** | Optional | Strongly recommended for gaze inference speed (RTX etc.) |

> **Windows-specific:** Always launch the server with `python -X utf8` (see step 5). This prevents `UnicodeEncodeError` crashes when library output contains non-ASCII characters on CP950/Big5 terminals.

---

## 2. Environment Setup

### Option A — Conda (Recommended for new setups)

```powershell
conda create -n lexigaze python=3.11
conda activate lexigaze
```

### Option B — uv (Fast, reproducible via lock file)

The repo ships a `uv.lock`. If you have [uv](https://github.com/astral-sh/uv):

```powershell
uv sync
```

This creates an isolated `.venv` and installs exact pinned versions from `uv.lock`.

### Option C — Standard venv

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

### Using uv (recommended, exact lock file)

```powershell
uv sync
```

### Using pip (from pyproject.toml)

```powershell
pip install -e .
```

### Using requirements.txt (fully pinned versions)

```powershell
pip install -r requirements.txt
```

### Additional: spaCy English model

Required for the English cognitive load pipeline:

```powershell
python -m spacy download en_core_web_sm
```

> **Core and Web logic packages are auto-discoverable.**
> All core logic submodules under `core/` and views under `web/` are loaded using standard Python import resolutions.

> **First-run download:** On the first cognitive load request, Hugging Face downloads GPT-2 (~500 MB for English) or BERT (~400 MB for Chinese) automatically into your `HF_HOME` directory.

---

## 4. Configure Environment Variables

Copy the example file and fill it in:

```powershell
copy .env.example .env
```

Edit `.env`:

```env
# Where Hugging Face model weights are cached (needs ~1 GB free space)
HF_HOME="D:/hf_models"

# Optional: HF token for private/gated models
HF_TOKEN=

# Required for Gemini-powered features (fusion analysis, etc.)
GEMINI_API_KEY=your_key_here

# Gemini model variant
MODEL_NAME="gemini-flash-lite-latest"
```

Get a free Gemini API key at: https://aistudio.google.com/apikey

---

## 5. Run the Server

The single entry point for the entire integrated application is `run.py`.

**Always run from the project root** (`D:\projects\lexigaze`), with the `-X utf8` flag:

```powershell
python -X utf8 run.py
```

On success you'll see:

```
================================================
  文件座標擷取工具  —  Flask Backend (Refactored)
================================================
  網址  : http://localhost:8080/
  資料  : data
  API   : http://localhost:8080/api/sessions
  認知負荷 : http://localhost:8080/api/cognitive/health
  停止  : Ctrl + C
================================================
```

The browser **opens automatically**. If not, go to `http://localhost:8080`.

Stop the server with **Ctrl + C**.

---

## 6. Using the Web Interface

### Main Tool — http://localhost:8080

The document analysis SPA (`word_track.html`). Features:

- **Upload** PDF, HTML, TXT, DOCX, or Markdown files
- **Extract** word/character bounding boxes with normalized coordinates (0.0–1.0)
- **Overlay** cognitive load scores as color-coded boxes or a Gaussian heatmap
  - Adjustable µ (threshold) and σ (spread) sliders
- **Save / Load** sessions via the REST API
- **Export** extracted coordinates as JSON or CSV
- **Evaluate** predictions against your own ground-truth word list (precision/recall/F1)
- **Archive** view of previous analysis results

### Gaze Calibration — http://localhost:8080/gaze

The gaze tracking page (`gaze_page.html`). Features:

- **Live inference** — webcam → MediaPipe face detection → UniGaze-B → screen coordinates
- **9-point calibration grid** — collect samples (configurable repeats per point)
- **Train personalization model** — polynomial regression calibration stored in `examples/models/`
- **Model selector** — switch between the frozen baseline and any trained personalization version
- **Filtering modes** — None, OneEuro, horizontal corridor lock, dwell/fixation detection, or combined

---

## 7. API Reference

### Health Checks

```
GET  /api/ping                    → General health + session count
GET  /api/gaze/health             → Gaze backend status
GET  /api/cognitive/health        → Cognitive load model status (shows loaded languages)
GET  /api/fuse/health             → Fusion pipeline status
```

### Document Sessions

```
GET    /api/sessions              → List saved sessions (summary only)
POST   /api/sessions              → Save extracted coordinate session
GET    /api/sessions/<id>         → Retrieve full session data
DELETE /api/sessions/<id>         → Delete a session
```

Sessions are stored as JSON files in `data/`.

### Gaze Tracking

```
GET  /api/gaze/models             → List trained personalization models
GET  /api/gaze/datasets           → List calibration datasets
POST /api/gaze/session            → Create a new calibration session
POST /api/gaze/sample             → Save a calibration sample (image + target point)
POST /api/gaze/train              → Train personalization model from a dataset
POST /api/gaze/predict            → Run gaze prediction on a webcam frame (base64)
```

Calibration data is stored in `data/sessions/`.

### Cognitive Load

```
GET  /api/cognitive/health        → Health check with loaded language list
POST /api/cognitive/warmup        → Pre-load a language model ("zh" or "en")
POST /api/cognitive/analyze/text  → Analyze text (auto-routes short vs long)
POST /api/cognitive/analyze/file  → Upload PDF/TXT/MD for batch analysis
POST /api/cognitive/evaluate      → Compare prediction vs ground truth (P/R/F1)
GET  /api/cognitive/archives      → List previously analyzed files
```

Analysis archives are stored in `archive/analysis_results/`.

### Fusion (Gaze × Cognitive Load)

```
GET  /api/fuse/health             → Fusion pipeline health
POST /api/fuse/                   → Compute per-word RDS (Reading Difficulty Score)
GET  /api/fuse/reports            → List saved fusion reports
GET  /api/fuse/reports/<id>       → Retrieve a specific fusion report
```

Fusion reports (optional) are saved to `docs/fusion_reports/`.

### Cognitive Capability Inspector

```
POST   /api/inspector/analyze      → Analyze chronological gaze history and generate user profile
POST   /api/inspector/report       → Compile detailed Markdown report (optional persist flag)
GET    /api/inspector/reports      → List all saved diagnostic reports
GET    /api/inspector/reports/<f>  → Get markdown content of a specific report
DELETE /api/inspector/reports/<f>  → Delete a specific report file
```

Persisted diagnostic reports are saved to `docs/cognitive_reports/`.

---

## 8. Research & Standalone Scripts

These scripts run independently of the server for benchmarking or research use.

### From the project root:

```powershell
# English cognitive load demo
python demo_english.py

# Benchmark the cognitive load pipeline
python benchmark_pipeline.py

# Visualization (BERT activations)
python run_visualize_bert.py
python visualize_load.py
```

### From the weichi/ module (cognitive load research - archived):

```powershell
# Quick demo with a few sentences
python archive/weichi/quick_demo.py

# Visualize load scores
python archive/weichi/visualize_load.py

# Retrain XGBoost scoring model on GECO corpus
python archive/weichi/train_xgb_geco.py

# Retrain Ridge fallback model
python archive/weichi/train_ridge_geco.py
```

### Validation Scripts (require GECO / PROVO corpus data)

> ⚠️ These scripts need corpus files that are **not committed** to the repo (`archive/weichi/GECO_data/`, `archive/weichi/PROVO_data/`). Ask a team member for these files.

```powershell
python archive/weichi/validate_geco.py          # Core GECO validation
python archive/weichi/full_validation.py        # Full paper-level (2000 train / 1000 test)
python archive/weichi/validate_provo.py         # Zero-shot PROVO cross-corpus test
python archive/weichi/robustness_analysis.py    # Bootstrap CI + LOSO per-reader
python archive/weichi/compare_models.py         # GPT-2 vs GPT-2-Large vs TinyLlama, etc.
```

---

## 9. Troubleshooting

### `UnicodeEncodeError` on Windows

**Cause:** Windows terminal encoding (CP950/Big5) can't represent certain characters printed by Hugging Face/transformers libraries.  
**Fix:** Launch with `-X utf8`:

```powershell
python -X utf8 run.py
```

The server already wraps stdout/stderr with a safe fallback, but `-X utf8` is more robust.

---

### `ModuleNotFoundError: No module named 'web'`

**Cause:** You ran the server from inside a subdirectory instead of the project root.  
**Fix:** Always run from the project root (`D:\projects\lexigaze`):

```powershell
# Correct ✅
python -X utf8 run.py

# Wrong ❌ — do NOT cd into subdirectories first
```

---

### `ModuleNotFoundError: No module named 'core.unigaze_personalization'`

**Cause:** The core modules directory structure is missing or corrupt.  
**Fix:** Check that `core/unigaze_personalization/` exists. If missing, please do a fresh clone:

```powershell
git submodule update --init --recursive
```

---

### `ModuleNotFoundError: No module named 'orchestrator'`

**Cause:** `scripts/fusion/orchestrator.py` is missing.  
**Fix:** Ensure the `scripts/fusion/` directory exists and contains `orchestrator.py`. This file is committed; a fresh clone should include it.

---

### Webcam not detected / gaze prediction fails

- Ensure no other app is using the webcam.
- Grant **camera permission** in your browser for `localhost:8080`.
- Navigate to `http://localhost:8080/gaze` (not the main page).

---

### Cognitive load model download very slow or fails

- Set `HF_HOME` in `.env` to a path with at least ~1.5 GB free.
- If behind a firewall, configure your proxy or authenticate via `HF_TOKEN`.
- You can pre-download English GPT-2 manually:

```powershell
python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; GPT2Tokenizer.from_pretrained('gpt2'); GPT2LMHeadModel.from_pretrained('gpt2')"
```

---

### Port 8080 already in use

```powershell
netstat -ano | findstr :8080
taskkill /PID <PID_NUMBER> /F
```

---

## Project Structure (Quick Reference)

```
lexigaze/
│
├── run.py                        ← Main server entry point (run this)
│
├── core/                         ← Core Business Logic Container
│   ├── cognition/                ← Cognitive load NLP pipeline & model weights
│   ├── cognitive_inspector/      ← User capability & reading diagnostics module
│   ├── gaze_core/                ← Gaze backend (inference, training, registry)
│   └── unigaze_personalization/  ← Preprocessing and deep learning transforms
│
├── web/                          ← Integrated web platform package
│   ├── routes/                   # Blueprints (cognitive, demo, fusion, gaze, inspector)
│   ├── static/                   # JS/CSS assets and MediaPipe landmarker task
│   └── templates/                # HTML layout templates (word_track, gaze_page)
│
├── data/                         # Saved document layout sessions
├── examples/models/              # Calibration datasets & personalization models
│
├── scripts/                      ← Diagnostic sandbox and test utilities
│   ├── test_cognitive_inspector.py # Unit and integration test suite
│   └── fusion/                   # Gaze×cognitive fusion orchestrator
│       └── orchestrator.py
│
├── archive/                      ← Archived legacy developer subdirectories
├── docs/                         ← Documentation directory
│   ├── refactor.md               # Moved refactoring guide
│   ├── cognitive_reports/        # Saved user diagnostic Markdown reports
│   └── fusion_reports/           # Persisted fusion RDS reports
│
├── .env                          ← Local secrets (never commit)
├── .env.example                  ← Template — copy to .env
├── pyproject.toml                <!-- Project metadata & dependencies -->
├── requirements.txt              ← Pinned dependency list
└── uv.lock                       ← Exact lock file for uv
```
