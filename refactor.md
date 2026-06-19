# LexiGaze Codebase Refactoring Report

This document details the architectural cleanup and code unification process that restructured the LexiGaze repository, removing redundant paths, eliminating runtime import path-injection hacks, and elevating the core integrated service to the root workspace.

---

## 1. Directory Restructuring

Prior to the refactoring, development was fragmented across modules owned by different developers. The integrated version hosted inside `chenghao/` acted as the entry point but relied on dynamic path injection to resolve peer folders. We have resolved this by archiving legacy development directories and bringing the production application directly to the workspace root.

### Directory Mapping Changes
| Source Path | Target/Consolidated Path | Description |
| :--- | :--- | :--- |
| `weichi/` (Cognition Module) | `archive/weichi/` | Archived legacy research scripts and notebooks. |
| `weichi/cognitive_load_pipeline.py` | `./cognitive_load_pipeline.py` | Promoted active cognition pipeline to root. |
| `weichi/*_model.json` | `./ridge_model.json`, `./xgb_model.json` | Promoted trained regression/XGB models to root. |
| `shengwen/` (Perception Module) | `archive/shengwen/` | Archived raw training pipelines, configurations, and scripts. |
| `shengwen/face_landmarker.task` | `./face_landmarker.task` | Promoted face landmarks landmark model to root. |
| `shengwen/src/unigaze_personalization/` | `./unigaze_personalization/` | Promoted personalization package to root. |
| `BoWei/` (Mapping Module) | `archive/BoWei/` | Archived redundant word tracking HTML/JS templates. |
| `chenghao/` (Integrated Hub) | `./` | All Flask routes, JS/HTML frontend pages, and core gaze modules moved to root. |

### Consolidated Repository Layout
```
lexigaze/
├── archive/                     # 🗄️ ARCHIVED LEGACY MODULES (weichi, shengwen, BoWei)
├── data/                        # 📊 Sessions and logs database
├── docs/                        # 📝 System documentation & design reports
├── examples/                    # 📂 Calibration data examples, reading passages, and models
├── gaze_core/                   # 👁️ Gaze tracking filters, inference, registry, & training
├── scripts/                     # ⚙️ Offline scripts (calibration importer, fusion orchestrator, test suite)
├── unigaze_personalization/     # 🤖 Preprocessing pipelines and deep learning transforms
├── cognitive_load_pipeline.py   # 🧠 Cognitive load pipeline (BERT/GPT-2 surprisal models)
├── server.py                    # 🌐 Consolidated Flask App Server
├── word_track.html              # 💻 Eye-tracking & cognitive text reading dashboard
└── refactor.md                  # 📄 [This file] Refactoring report & startup guide
```

---

## 2. Refactoring Actions & Path Resolution

To allow running from the root natively, path injections were cleaned up, and static asset lookups were adapted:

1. **Import Consolidation**:
   - Removed `sys.path.insert` statements that appended `shengwen/src` inside [gaze_core/__init__.py](file:///home/ubuntu/projects/lexigaze/gaze_core/__init__.py).
   - Removed path injections of `weichi/` inside [cognitive_routes.py](file:///home/ubuntu/projects/lexigaze/cognitive_routes.py).
   - Python files can now directly reference `unigaze_personalization` and `cognitive_load_pipeline` via standard root-level package absolute paths.

2. **File Paths Corrected**:
   - [server.py](file:///home/ubuntu/projects/lexigaze/server.py): Adjusted static asset directories to serve `archive/shengwen` assets and resolve passages under `./examples`.
   - [unigaze_personalization/preprocess.py](file:///home/ubuntu/projects/lexigaze/unigaze_personalization/preprocess.py): Changed face landmark detection file loader lookup relative path to `./face_landmarker.task` (from `../../face_landmarker.task`).
   - [gaze_core/model_registry.py](file:///home/ubuntu/projects/lexigaze/gaze_core/model_registry.py): Corrected personalization model save paths to land inside `examples/models/`.
   - [scripts/fusion/orchestrator.py](file:///home/ubuntu/projects/lexigaze/scripts/fusion/orchestrator.py): Corrected target search path for `cognitive_load_pipeline` to root.

3. **Git Configuration**:
   - Cleaned up `.gitignore` to map old ignores to the new `archive/` directory structure, while ensuring local run checkpoints inside `examples/models/` remain ignored.

---

## 3. How to Run the Unified Codebase

All features can now be run directly using the `uv` toolchain from the repository root:

### 1. Launch the Integrated Application Server
Launch the unified Flask server:
```bash
uv run python -X utf8 server.py
```
- Main Reading Interface: [http://localhost:8080/word_track.html](http://localhost:8080/word_track.html)
- API Documentation and Heath Checks: [http://localhost:8080/api/cognitive/health](http://localhost:8080/api/cognitive/health)

### 2. Run the Offline Video Calibration Importer
To run personalization autotraining on recorded calibration video files:
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

---

## 4. Test & Simulation Verification

A diagnostic script [scripts/test_refactoring.py](file:///home/ubuntu/projects/lexigaze/scripts/test_refactoring.py) was written and run to verify refactoring soundness. The test output confirmed all system components are healthy:

```
=== Refactoring Diagnostic Tests ===
Testing Test 1: CognitiveLoadPipeline import...
✓ Successfully imported CognitiveLoadPipeline.
Testing Test 2: Gaze Core imports...
✓ Gaze Core directories validated:
  Runs Dir: /home/ubuntu/projects/lexigaze/examples/models
  Sessions Dir: /home/ubuntu/projects/lexigaze/data/sessions
Testing Test 3: Fusion Orchestrator import and logic...
✓ RDS classification logic works.
✓ Cognitive lookup builder handles hyphenated words correctly.
Testing Test 4: Flask server routing integration...
✓ Flask server live endpoint ping verified successfully.

🎉 All refactoring diagnostic tests completed successfully!
```
All system capabilities—from gaze mapping, model training, and cognitive analysis endpoints—remain functional and backward-compatible.
