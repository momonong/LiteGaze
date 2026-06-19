# LexiGaze Codebase Refactoring Report

This document details the architectural cleanup, modularization, and code unification process that restructured the LexiGaze repository into a clean, industry-standard package structure.

---

## 1. Directory Restructuring & Best Practices

Rather than placing all files in the root folder, the project has been refactored into structured, cohesive Python packages and layout templates following Flask application factory best practices.

### Core Changes
* **Application Factory Pattern**: Moved Flask app initialization and route blueprint registrations into `web_app/__init__.py`. 
* **Views & Templates Separation**:
  * Moved HTML views (`word_track.html`, `gaze_page.html`) to the `web_app/templates/` folder.
  * Moved CSS, JS (`mapping.js`, `gaze_integration.js`, `gaze_page.js`), and model weights (`face_landmarker.task`) into the `web_app/static/` folder.
  * Corrected HTML script src paths to reference `/static/<filename>`.
* **Cognitive Module Clean-up**:
  * Moved the active cognition pipeline and trained JSON model weights to a dedicated `cognition/` Python package (`cognition/pipeline.py`, `cognition/ridge_model.json`, `cognition/xgb_model.json`).
  * Simplified paths to automatically locate sibling model json weights.
* **Unified Entry Point**: Removed the redundant `server.py` and created a streamlined `run.py` entry point at the repository root.

### Directory Layout
```
lexigaze/
├── archive/                     # 🗄️ ARCHIVED LEGACY MODULES (weichi, shengwen, BoWei)
├── data/                        # 📊 Sessions and logs database
├── docs/                        # 📝 System documentation & design reports
├── examples/                    # 📂 Calibration data examples, reading passages, and models
├── scripts/                     # ⚙️ Offline scripts (calibration importer, fusion orchestrator, etc.)
│   ├── import_offline_calibration.py
│   ├── fusion/
│   │   ├── orchestrator.py
│   │   └── __init__.py
│   └── test_refactoring.py
│
├── gaze_core/                   # 👁️ Gaze tracking filters, inference, registry, & training
│   ├── __init__.py
│   ├── filters.py
│   ├── inference.py
│   ├── model_registry.py
│   ├── sample_store.py
│   └── training.py
│
├── unigaze_personalization/     # 🤖 Preprocessing pipelines and deep learning transforms
│   ├── assets/
│   │   ├── __init__.py
│   │   └── face_model.txt
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── preprocess.py
│   ├── server.py
│   └── transforms.py
│
├── cognition/                   # 🧠 COGNITIVE LOAD PIPELINE & MODELS
│   ├── __init__.py
│   ├── pipeline.py              # (contains CognitiveLoadPipeline)
│   ├── ridge_model.json         # Trained ridge regression weights
│   └── xgb_model.json           # Trained XGBoost model weights
│
├── web_app/                     # 🌐 THE MAIN FLASK WEB APPLICATION PACKAGE
│   ├── __init__.py              # Flask app factory (defines create_app)
│   ├── routes/                  # Modular backend endpoints (blueprints)
│   │   ├── __init__.py
│   │   ├── cognitive.py         # Cognitive API blueprint
│   │   ├── demo.py              # Calibration and processing blueprints
│   │   ├── fusion.py            # Event/RDS fusion blueprint
│   │   └── gaze.py              # Personalization model registry blueprints
│   ├── static/                  # Client-side JavaScript, CSS, & model weights
│   │   ├── face_landmarker.task # Face landmark detector model
│   │   ├── gaze_integration.js  # Frontend integration code
│   │   ├── gaze_page.js         # Gaze logging engine
│   │   └── mapping.js           # Gaze-to-word matching and highlight rendering
│   └── templates/               # Client-side HTML views
│       ├── gaze_page.html       # Eye-tracking calibration view
│       └── word_track.html      # Integrated reading dashboard
│
├── run.py                       # 🚀 Clean entrypoint at root (runs web_app)
└── refactor.md                  # 📄 [This file] Refactoring documentation
```

---

## 2. Path and Blueprint Resolutions

1. **Blueprints**:
   - `web_app/routes/*.py` define the blueprints.
   - `ROOT` inside the route scripts is updated to point to the repository root directory using `Path(__file__).resolve().parents[2]`.
   - References to the cognitive load pipeline now import directly from the unified package:
     ```python
     from cognition import CognitiveLoadPipeline
     ```

2. **Frontend Routing**:
   - Serving `/` renders `templates/word_track.html`.
   - Serving `/gaze` renders `templates/gaze_page.html`.
   - Script inclusions are served standardly from `/static/`.

---

## 3. How to Run the Project

### 1. Start the Flask App Server
```bash
uv run python -X utf8 run.py
```
- Dashboard View: [http://localhost:8080/](http://localhost:8080/)
- Calibration Page: [http://localhost:8080/gaze](http://localhost:8080/gaze)
- API Health Status: [http://localhost:8080/api/cognitive/health](http://localhost:8080/api/cognitive/health)

### 2. Run Personalization Model Training Offline
```bash
uv run python scripts/import_offline_calibration.py \
  examples/calibrations/test00_calibration.webm \
  examples/calibrations/test00_timeline.json \
  test00_user
```

### 3. Run Diagnostic Verification
To test that imports, model paths, routing, and configurations are functional:
```bash
uv run python scripts/test_refactoring.py
```
