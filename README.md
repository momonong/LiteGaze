# lexigaze

Webcam gaze tracking alone is too inaccurate for language learning research. **lexigaze** tackles this by combining three pieces into one Flask app:

- **Document coordinate extraction** — Parses PDF, HTML, TXT, DOCX, and Markdown into pixel-level word/character bounding boxes. Normalized (0.0–1.0) so results are device-independent.

- **Gaze tracking** — UniGaze-B neural network + MediaPipe face detection runs in real time from any webcam. A personalization step collects a few calibration samples and trains a polynomial regression to fix per-user bias.

- **Cognitive load analysis** — GPT-2 (EN) or BERT (ZH) scores each word's difficulty. Long text auto-chunks and aggregates via ridge regression.

The key idea: extracted word coordinates are the shared reference frame for both gaze mapping and cognitive load overlay. This fuses **where** someone looks with **how hard** the text is — giving researchers a single coordinate space to do higher-dimensional analysis that raw gaze accuracy alone can't support.

## Use Cases

- **Reading research** — Record gaze patterns aligned to exact word positions during natural reading. Export mappings with confidence levels (high/medium/low) for offline analysis.

- **Readability assessment** — Upload a chapter or paper, run cognitive load analysis, and see which sentences pack the hardest vocabulary.

- **Personalized calibration** — A few seconds of webcam data trains a per-user model that meaningfully improves gaze accuracy, no special hardware needed.

- **Material comparison** — Overlay cognitive load heatmaps from two versions of the same document side by side to see which one reduces comprehension barriers.

- **Annotation QA** — After coordinate extraction, visually verify every bounding box lines up with rendered text before exporting.

## Directory Structure

```
chenghao/
├── server.py                  # Flask entry point; serves static files & REST APIs
├── word_track.html            # Main SPA: document upload, parsing, gaze overlay, cognitive load
├── gaze_page.html             # Gaze calibration & model training UI
├── gaze_page.js               # Calibration logic: data collection, training, live testing
├── gaze_integration.js        # Gaze inference loop: camera capture, prediction, filtering
├── mapping.js                 # Gaze-to-word overlay highlighting on extracted coordinates
├── gaze_routes.py             # Flask Blueprints for gaze APIs (predict, train, session, etc.)
├── cognitive_routes.py        # Flask Blueprints for cognitive load analysis APIs
├── gaze_core/                 # Backend Python package for gaze pipeline
│   ├── __init__.py            # sys.path injection for sibling module access
│   ├── inference.py           # UniGaze model inference (MediaPipe + ONNX → screen coords)
│   ├── training.py            # Personalization: polynomial regression calibration
│   ├── sample_store.py        # Session/dataset management, sample saving
│   ├── model_registry.py      # Model listing, naming, path resolution
│   └── filters.py             # Placeholder for server-side smoothing
├── wasm/                      # MediaPipe WebAssembly assets for face detection
│   ├── vision_wasm_internal.js
│   └── vision_wasm_internal.wasm
├── unigaze_b16.onnx           # Pre-trained UniGaze-B gaze estimation model
├── face_landmarker.task       # MediaPipe face landmarker model
├── data/                      # Saved session JSON files (uploaded document sessions)
└── gaze_data/                 # Calibration sessions & trained personalization models
    ├── sessions/
    └── runs/
```

## Quick Start

### Prerequisites

- Python 3.10+
- A webcam (for gaze tracking)

### Setup

```bash
# Install Python dependencies
pip install flask opencv-python numpy torch unigaze-personalization

# Start the server
python chenghao/server.py
```

The server starts at `http://localhost:8080`. Open it in your browser to access the main tool.

> **Note:** The first call to cognitive load analysis will download BERT / GPT-2 models (~500 MB) automatically.

## Features

### 1. Document Coordinate Extraction

Upload PDF, HTML, TXT, DOCX, or Markdown files and extract precise bounding boxes for every word or character. The parser renders each page and records position, size, and normalized coordinates (0.0–1.0, device-independent).

- **PDF**: Renders via pdf.js at configurable scale, extracts text layer coordinates
- **HTML / TXT / DOCX / MD**: Renders in-browser, uses DOM `Range.getClientRects()` for sub-pixel accurate positions
- Supports both **word** and **character** granularity
- Export results as **JSON** or **CSV**
- Save/load sessions via REST API
- Visual overlay with search highlight, hover inspection, and color legend

### 2. Gaze Tracking & Personalization

Real-time gaze estimation using the UniGaze-B neural network with MediaPipe face preprocessing.

- **Live inference**: Webcam feed → face detection → gaze angle prediction → screen coordinate mapping
- **Filtering modes**: None, OneEuro smoothing, horizontal corridor lock, dwell (fixation detection), or combined
- **Personalization**: Collect calibration samples (9-point grid with configurable repeats), train a polynomial regression model to map raw gaze → screen coordinates
- Model versions are versioned and selectable from a dropdown

### 3. Cognitive Load Analysis

Analyze text difficulty using transformer-based models to detect "deep/hard words" that impose high cognitive load.

- **English**: GPT-2 based pipeline
- **Chinese**: BERT based pipeline
- **Domain detection**: Auto, academic, or general
- Accepts short text (single inference) or long documents (auto-chunked, file-level threshold via ridge regression)
- Upload PDF / TXT / MD for batch analysis
- Visual overlay on extracted document coordinates: precise bounding boxes or Gaussian heatmap with adjustable µ (threshold) and σ (spread)
- Archive management for previous analysis results
- Evaluate against ground-truth annotation with precision / recall / F1 scoring

## API Reference

### Gaze Tracking (`/api/gaze/*`, `/api/*`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/gaze/health` | Backend health check |
| GET | `/api/gaze/models` | List trained personalization models |
| GET | `/api/gaze/datasets` | List calibration datasets |
| POST | `/api/gaze/session` | Create a new calibration session |
| POST | `/api/gaze/sample` | Save a calibration sample (image + target) |
| POST | `/api/gaze/train` | Train a personalization model from a dataset |
| POST | `/api/gaze/predict` | Run gaze prediction on a webcam frame |

### Cognitive Load (`/api/cognitive/*`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cognitive/health` | Health check with loaded languages |
| POST | `/api/cognitive/warmup` | Pre-load a language model (zh/en) |
| POST | `/api/cognitive/analyze/text` | Analyze short or long text |
| POST | `/api/cognitive/analyze/file` | Upload PDF/TXT/MD for analysis |
| POST | `/api/cognitive/evaluate` | Compare prediction with ground truth (precision/recall/F1) |
| GET | `/api/cognitive/archives` | List previously analyzed files |

### Document Sessions (`/api/*`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/ping` | Health check with session count |
| GET | `/api/sessions` | List saved document sessions |
| POST | `/api/sessions` | Save extracted coordinates as a session |
| GET | `/api/sessions/<id>` | Retrieve a saved session |
| DELETE | `/api/sessions/<id>` | Delete a session |