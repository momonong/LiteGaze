# 🔭 LexiGaze Operations, Setup, and System Testing Guide

This document provides a detailed setup guide, API references, troubleshooting tips, and a step-by-step end-to-end workflow to test and evaluate the entire LexiGaze platform.

---

## 📋 Table of Contents

1. [Environment Setup & Installation](#1-environment-setup--installation)
2. [Environment Variables Configuration](#2-environment-variables-configuration)
3. [Running the Flask Server](#3-running-the-flask-server)
4. [Step-by-Step System Testing Workflow (Calibration to Fusion)](#4-step-by-step-system-testing-workflow-calibration-to-fusion)
5. [Offline Performance Diagnostics & Directions for Improvement](#5-offline-performance-diagnostics--directions-for-improvement)
6. [REST API Reference](#6-rest-api-reference)
7. [Troubleshooting Guide](#7-troubleshooting-guide)

---

## 1. Environment Setup & Installation

### Option A — Fast Sync using uv (Recommended)
The repository contains a `uv.lock` file. If you have [uv](https://github.com/astral-sh/uv) installed:
```bash
# Create environment and install exact pinned dependencies
uv sync

# Download the English dependency parser model for spaCy
.venv/bin/python -m spacy download en_core_web_sm
```

### Option B — Conda Setup
```bash
conda create -n lexigaze python=3.11 -y
conda activate lexigaze

# Install package in editable mode
pip install -e .

# Download spaCy models
python -m spacy download en_core_web_sm
```

### Option C — Standard venv Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

*Note: The first time you execute a cognitive analysis request, the Hugging Face library will automatically download the language models (GPT-2 for English ~500 MB; BERT for Chinese ~400 MB) and cache them in your `HF_HOME` directory.*

---

## 2. Environment Variables Configuration

Duplicate the environment template file:
```bash
cp .env.example .env
```

Configure `.env` with the following variables:
```env
# Path where Hugging Face weights are cached
HF_HOME="/home/ubuntu/.cache/huggingface"

# Required for Google AI Studio API (Inspector analyses, summaries)
GEMINI_API_KEY=your_gemini_api_key_here

# Model variant for Google AI Studio
MODEL_NAME="gemma-4-26b-a4b-it"
```

---

## 3. Running the Flask Server

To run the Flask application locally:
```bash
# Always start from the project root with the UTF-8 flag to avoid console crashes
.venv/bin/python -X utf8 run.py
```

Upon starting successfully, you should see console logs resembling:
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
The browser will automatically open to `http://localhost:8080/`.

---

## 4. Step-by-Step System Testing Workflow

Follow this walkthrough to run a complete, end-to-end workflow on the platform:

```
[Server Running] ──► [Gaze Calibration] ──► [Personalization Model] ──► [Document Upload] ──► [Gaze & NLP Fusion]
```

### Phase 4.1: Perform Gaze Calibration & Model Personalization
1. Navigate to the calibration page at `http://localhost:8080/gaze`.
2. Expand the **收集設定 (Collection Settings)** panel at the bottom right.
3. Assign a custom **受試者 ID (Participant ID)** (e.g. `subject_test_01`).
4. Click **開始 (Start)** and keep your eyes focused on the red dot as it traverses the 9-point grid.
5. Once complete, the browser uploads calibration coordinates to the server. The server automatically fits a polynomial regression model and stores it in `examples/models/subject_test_01_model.json`.

### Phase 4.2: Upload Document & Extract Layout Coordinates
1. Open the main portal at `http://localhost:8080/`.
2. Upload a text or PDF document.
3. Click **Extract Coordinates** to parse the text and extract pixel-level word layout boxes.

### Phase 4.3: Execute Cognitive Load Analysis
1. Select the document language (English or Chinese) in the sidebar.
2. Click **Analyze Cognitive Load** to run the token-level suprisal and entropy models.
3. Toggle the **Heatmap** view to display the difficulty overlay over the parsed words.

### Phase 4.4: Read with Active Personalization
1. In the sidebar dropdown, select your trained personalization model (e.g. `subject_test_01_model`).
2. Toggle the **Live Gaze Tracking** connection on.
3. Read the document naturally. Your webcam feed is preprocessed, mapped to the screen, and matched to the word boxes on the screen in real-time.

### Phase 4.5: Execute Multimodal Fusion & Demo Results
1. Click **Analyze & Fuse** after finishing the reading session.
2. The frontend sends the accumulated dwell-times to `/api/fuse`.
3. The server runs the fusion algorithm (e.g. Linear, Bayesian, or Sigmoid) to combine gaze attention values with linguistic load.
4. The document view highlights words in red/orange/yellow (High/Medium/Low) matching calculated **Reading Difficulty Scores (RDS)**, identifying cognitive reading bottlenecks.

---

## 5. Offline Performance Diagnostics & Directions for Improvement

LexiGaze includes sandbox scripts to test and evaluate gaze correction, NLP modeling, and fusion pipelines offline on actual eye-tracking datasets (GECO). Use these to guide optimization:

### Diagnostic 5.1: Comparative Module Sandbox
To evaluate the combined performance of gaze decoders, NLP metrics, and fusion methods:
```bash
.venv/bin/python scripts/inspect_performance_demo.py
```
This simulates webcam drift (+45px vertical drift, 30-40px jitter) on a 156-word GECO reading trial, printing accuracy, Spearman correlation, and processing latency:

| Configuration | Gaze Accuracy (%) | RDS Correlation ($\rho$) | Latency (ms) | Target Optimization Direction |
| :--- | :---: | :---: | :---: | :--- |
| **Raw Gaze + No Cog + Linear** | 18.59% | 0.0636 | ~1.5 ms | **Baseline**: Heavily impacted by calibration drift. |
| **Viterbi + No Cog + Linear** | 48.72% | 0.0910 | ~140 ms | **Spatio-Temporal Prior**: Corrects drift but lacks online tuning. |
| **Viterbi + EM Calib + No Cog** | 73.72% | 0.2050 | ~210 ms | **EM Self-Calibration**: Re-calibrates offsets during reading. |
| **STOCK-T v3 + surprisal + Bayesian** | **78.21%** | **0.2258** | ~210 ms | **Optimal Joint System**: Max accuracy & difficulty correlation. |

### Diagnostic 5.2: Multimodal Fusion Calibration
To compare the six mathematical fusion methods on GECO:
```bash
.venv/bin/python scripts/experiment_fusion.py
```
This generates evaluation summaries and plots under `output/`:
* `fusion_correlation_comparison.png` - Compares Spearman $\rho$ and Pearson $r$.
* `rds_distributions.png` - Plots score distributions across algorithms.

### 📈 Directions for Future Improvement
1. **Gaze Correction Accuracy**: Raw webcam tracking suffers from vertical drift (~45px). Implementing **Auto-Calibrating EM Decoders** (`AutoCalibratingDecoder`) or **Psycholinguistic Transition Matrices** (`PsycholinguisticTransitionMatrix`) corrects this, increasing coordinate mapping accuracy from 18.5% to 78.2%.
2. **Computational Latency**: Running Viterbi and EM decoding sequentially adds ~200ms processing delay per page. Next-step optimizations should focus on compiling the transition matrix operations using **Cython/Numba** or vectorizing loops via **NumPy**.
3. **Cognitive Weight Calibration**: Currently, Bayesian and Multiplicative fusion yield higher correlation values ($\rho > 0.80$) on human reading times than simple Linear summation by modeling interaction effects (skipped words vs high surprisal). Tuning the prior boundaries in `scripts/fusion_module.py` will further improve prediction fidelity.

---

## 6. REST API Reference

### Health Diagnostics
* `GET  /api/ping`: General API status and session count.
* `GET  /api/gaze/health`: Gaze tracking module state.
* `GET  /api/cognitive/health`: Loaded NLP model lists and status.
* `GET  /api/fuse/health`: Fusion engine status.

### Document & Layout Sessions
* `GET  /api/sessions`: Returns metadata summaries of saved layouts.
* `POST /api/sessions`: Persists an extracted layout session.
* `GET  /api/sessions/<id>`: Fetches full layout details for a session.
* `DELETE /api/sessions/<id>`: Removes a layout session.

### Calibration & Personalization Models
* `GET  /api/gaze/models`: List all trained regression personalization models.
* `GET  /api/gaze/datasets`: List available calibration manifests.
* `POST /api/gaze/session`: Create a new calibration session.
* `POST /api/gaze/sample`: Save an individual calibration image sample.
* `POST /api/gaze/train`: Fit a regression model on a calibration dataset.
* `POST /api/gaze/predict`: Run real-time inference on a webcam frame (Base64).

### Cognitive Load & Inspector
* `POST /api/cognitive/warmup`: Load a specific language model ("en" or "zh").
* `POST /api/cognitive/analyze/text`: Compute cognitive metrics on a string.
* `POST /api/cognitive/analyze/file`: Batch compute cognitive metrics on a document.
* `POST /api/cognitive/evaluate`: Compare predicted difficulty against user ground-truths.
* `POST /api/inspector/analyze`: Analyze gaze history sequences to profile reading capability.
* `POST /api/inspector/report`: Compile diagnostic Markdown profiles.

### Joint Fusion
* `POST /api/fuse/`: Combine gaze logs and cognitive load scores to compute RDS.
* `GET  /api/fuse/reports`: List all saved fusion RDS reports.

---

## 7. Troubleshooting Guide

### 1. `UnicodeEncodeError` when starting Python
* **Cause**: System terminal (e.g. Windows CP950/Big5) cannot encode print output from Hugging Face or spacy.
* **Fix**: Ensure you start scripts using `python -X utf8 run.py`.

### 2. `ModuleNotFoundError: No module named 'web'`
* **Cause**: Python was executed from inside a subdirectory.
* **Fix**: Always execute scripts from the project root directory (e.g., `python -X utf8 run.py`).

### 3. Webcam feed not showing in browser
* **Cause**: Camera permission blocked, or another process is locking the webcam.
* **Fix**: Grant camera permission to `http://localhost:8080` in browser settings. Ensure background videoconferencing applications are closed.
