# 🏗️ LexiGaze: Repository Architecture & System Structure

This document provides a comprehensive overview of the **LexiGaze** repository architecture, describing its directory layout, module relationships, perception-cognition data flows, and internal database schemas.

---

## 📂 Repository Directory Tree

The repository is structured into modular subsystems representing different research branches (Perception, Cognition, Fusion, and Web Integration):

```
lexigaze/
├── core/                      # 🧠 CORE BUSINESS LOGIC CONTAINER
│   ├── cognition/             # Cognition pipeline & model JSON weights
│   │   ├── pipeline.py        # NLP Pipeline surprisal (GPT-2/BERT) calculation
│   │   ├── ridge_model.json   # Pretrained Ridge regression models
│   │   └── xgb_model.json     # Pretrained XGBoost model
│   │
│   ├── gaze_core/             # Gaze prediction filters & model registries
│   │   ├── inference.py       # Preprocesses frames & feeds to ONNX model
│   │   ├── training.py        # Trains polynomial regression on calibration datasets
│   │   ├── sample_store.py    # Dataset sessions, manifests, and image saving
│   │   └── model_registry.py  # Manages and lists trained per-user models
│   │
│   └── unigaze_personalization/ # MediaPipe preprocessing & model loading
│       ├── preprocess.py      # Face cropping & head pose estimation
│       ├── model.py           # Frozen UniGaze-B16 ViT model weights wrapper
│       ├── dataset.py         # Manifest helper for loading calibration sessions
│       └── transforms.py      # Image transformation pipeline
│
├── web/                       # 🌐 THE MAIN FLASK WEB APPLICATION PACKAGE
│   ├── routes/                # Modular backend endpoints (blueprints)
│   │   ├── cognitive.py       # Blueprints for text difficulty analysis
│   │   ├── gaze.py            # Blueprints for gaze prediction & models
│   │   ├── demo.py            # Blueprints for offline video-calibration
│   │   └── fusion.py          # Blueprints for combining gaze metrics & load
│   ├── static/                # Client-side JavaScript, CSS, & model weights
│   │   ├── mapping.js         # Gaze cursor smoothing and word mapping
│   │   ├── gaze_integration.js # Live loop & camera capture in browser
│   │   ├── gaze_page.js       # Calibration and timeline logic
│   │   └── face_landmarker.task # MediaPipe pretrained landmark task bundle
│   └── templates/             # Client-side HTML views
│       ├── word_track.html    # Main SPA: PDF coordinate extraction & reading dashboard
│       └── gaze_page.html     # Live calibration view
│
├── run.py                     # 🚀 Clean entrypoint at root (runs web)
├── refactor.md                # 📄 Refactoring documentation
├── archive/                   # 🗄️ ARCHIVED LEGACY MODULES (weichi, shengwen, BoWei)
│
├── scripts/                   # 🧪 RESEARCH SANDBOX & EXPERIMENTAL UTILITIES
│   ├── fusion/                # Core mathematical fusion helper modules
│   │   └── orchestrator.py    # Merges word layouts, gaze dwells, and NLP features
│   ├── fusion_module.py       # Implements 6 fusion algorithms (Linear, Bayesian, RRF, etc.)
│   ├── experiment_fusion.py   # Validation correlation tests on GECO corpus
│   ├── inspect_performance_demo.py # Joint pipeline (Viterbi/EM/Fusion) terminal inspection dashboard
│   ├── generate_web_demo_data.py   # Mock calibration session and model JSON generator
│   └── setup_remote_collection.py # Cross-platform ngrok tunnel launcher for remote laptops
│
├── docs/                      # 📄 SYSTEM DOCUMENTATION
│   ├── demo_integration_guide.md  # Demo walkthrough and integration architecture
│   └── fusion_reports/        # Saved joint gaze-cognitive fusion analysis logs
│
├── pyproject.toml             # Project dependency registry (Flask, Torch, OpenCV, Transformers)
├── uv.lock                    # Dependency lockfile for reproducible environments
├── instruction_data.md        # Setup guide for remote Ubuntu server + Windows clients
└── conclusion.md              # Registry of recent updates and accomplishments
```

---

## 🔄 Core Data Flows

LexiGaze combines perception tracking with linguistic cognition to map where a user looks and calculate how difficult the text is.

### 1. Perception Gaze Flow (Real-Time Tracking)
```
[Webcam Stream] 
       │ (640x480 Frame base64 JPEG)
       ▼
[MediaPipe Face Preprocessor] ──► Extracts 3D Face Bounding Box & Landmarks
       │
       ▼ (Normalized face image + crop)
[UniGaze-B16 ViT ONNX Model] ──► Predicts raw Gaze vector [Pitch, Yaw]
       │
       ▼ (Standardized [-1.0, 1.0] coordinates)
[Polynomial Personalization Model] ──► Corrects systematic offset & user bias
       │
       ▼ (Mapped viewport pixels)
[OneEuro / Dwell Filter] ──► Smooths coordinates, detects Dwells/Fixations
       │
       ▼
[Highlight Overlay] ──► Maps gaze cursor to active word layout boxes in mapping.js
```

### 2. Cognition NLP Flow (Language Complexity)
```
[Uploaded Document (PDF/MD/TXT)] 
       │
       ▼
[Document Coordinate Extraction] ──► Parses pages into word layout bounding boxes
       │
       ▼ (Extracted Text Content)
[HuggingFace Transformers (BERT/GPT-2)] ──► Computes per-word Surprisal & Attention Entropy
       │
       ▼
[Cognitive Load Overlay] ──► Visualizes text difficulty as a Gaussian Heatmap
```

### 3. Joint Multi-Multimodal Fusion Flow (RDS Output)
At the end of a reading session, the frontend gathers tracked gaze logs and sends them to the server to compute the Reading Difficulty Score (RDS):
```
[Gaze Dwell Logs] (Dwell time, fixations) ──┐
                                             ├──► [Orchestrator] ──► [Fusion Algorithms] ──► [RDS High/Med/Low]
[Cognitive Load Scores] (Surprisal) ─────────┘
```

---

## 📊 Database Schemas & File Structures

### 1. Calibration Session Manifest (`manifest.jsonl`)
Located in data/sessions/<session_id>/manifest.jsonl. Each line represents a recorded gaze calibration sample:
```json
{
  "ok": true,
  "sample_index": 0,
  "phase": "calibration",
  "point_index": 4,
  "repeat_index": 0,
  "target_x": 960.0,
  "target_y": 540.0,
  "target_x_norm": 0.0,
  "target_y_norm": 0.0,
  "viewport_width": 1920.0,
  "viewport_height": 1080.0,
  "raw_path": "raw/000000_calibration_04_00.jpg",
  "crop_path": "crop/000000_calibration_04_00.jpg",
  "normalized_face_path": "normalized_face/000000_calibration_04_00.jpg",
  "head_pose_pitch_yaw": [0.01, -0.02],
  "face_bbox": [480, 270, 960, 810]
}
```

### 2. Personalization Model JSON
Located in examples/models/<model_name>.json. Contains weights for mapping raw gaze angles (`[pitch, yaw]`) to screen locations:
```json
{
  "name": "subject_laptop_model",
  "created_at": "2026-06-19T01:31:00",
  "data_session_id": "session_20260619_013000",
  "mean_px_error": 12.5,
  "noise_level": 4.8,
  "train_samples": 18,
  "stages": [
    {
      "stage": 1,
      "poly_degree": 1,
      "W": [
        [0.85, 0.05],
        [0.05, 0.85],
        [0.01, 0.01]
      ],
      "mean_px_error": 12.5
    }
  ]
}
```

### 3. Document Coordinate Layout Session
Located in data/<session_id>.json. Extracted layout session generated during document uploading:
```json
{
  "id": "72c20283-7bd7-49cd-bbaa-b9d5d9ba5567",
  "filename": "sample.md",
  "filetype": "md",
  "created_at": "2026-06-19T01:30:00",
  "item_count": 134,
  "items": [
    {
      "page": 1,
      "index": 0,
      "text": "The",
      "top": 44.24,
      "left": 51.99,
      "width": 31.7,
      "height": 18.7,
      "norm_left": 0.0655,
      "norm_top": 0.0942,
      "norm_width": 0.0399,
      "norm_height": 0.04
    }
  ]
}
```
