# LexiGaze Codebase Refactoring Report

This document details the architectural cleanup, modularization, and code unification process that restructured the LexiGaze repository into a clean, industry-standard package structure.

---

## 1. Directory Restructuring & Best Practices

Rather than placing packages and layouts directly on the root repository directory, the codebase has been structured into:
1. A container package `core/` for business logic (eye-tracking gaze math, MediaPipe preprocessing, and NLP cognitive models).
2. A Flask frontend package `web/` following modular blueprints and template views patterns.

### Core Changes
* **`core/` logic package**:
  * **`core/cognition/`**: Cognition pipeline and trained JSON model weights (`pipeline.py`, `ridge_model.json`, `xgb_model.json`).
  * **`core/gaze_core/`**: Custom UniGaze prediction filters, model registries, and personalization trainers.
  * **`core/unigaze_personalization/`**: Image normalization, datasets creation, and face parsing preprocessors.
* **`web/` presentation package**:
  * **`web/__init__.py`**: Flask application factory setup (`create_app()`), registering blueprints and templates.
  * **`web/routes/`**: Blueprints for `cognitive`, `demo`, `fusion`, and `gaze` routes.
  * **`web/static/`**: Client-side CSS, JavaScript, and face landmarks detect assets (`mapping.js`, `gaze_integration.js`, `gaze_page.js`, `face_landmarker.task`).
  * **`web/templates/`**: Views (`word_track.html`, `gaze_page.html`).
* **Root Unified Entrypoint**: Streamlined run script [run.py](file://D:/projects/lexigaze/run.py) at the repository root.

### Directory Layout
```
lexigaze/
├── archive/                     # 🗄️ ARCHIVED LEGACY MODULES (weichi, shengwen, BoWei)
├── data/                        # 📊 Sessions and logs database
├── docs/                        # 📝 System documentation & design reports
├── examples/                    # 📂 Calibration data examples, reading passages, and models
├── scripts/                     # ⚙️ Offline scripts (calibration importer, fusion orchestrator, test suite)
│   ├── import_offline_calibration.py
│   ├── fusion/
│   │   ├── orchestrator.py
│   │   └── __init__.py
│   └── test_refactoring.py
│
├── core/                        # 🧠 CORE BUSINESS LOGIC CONTAINER
│   ├── __init__.py
│   ├── cognition/               # Cognition pipeline & model JSON weights
│   ├── gaze_core/               # Gaze prediction filters & model registries
│   └── unigaze_personalization/ # MediaPipe preprocessing and model loading
│
├── web/                         # 🌐 THE MAIN FLASK WEB APPLICATION PACKAGE
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
├── run.py                       # 🚀 Clean entrypoint at root (runs web)
└── refactor.md                  # 📄 [This file] Refactoring documentation
```

---

## 2. Path and Blueprint Resolutions

1. **Blueprints**:
   - `web/routes/*.py` define the blueprints.
   - `ROOT` inside the route scripts is updated to point to the repository root directory using `Path(__file__).resolve().parents[2]`.
   - References to the cognitive load pipeline now import from `core.cognition`:
     ```python
     from core.cognition import CognitiveLoadPipeline
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
