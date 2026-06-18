# LexiGaze: Module Integration & Demo Guide

This document explains how to integrate, run, and evaluate the joint performance of the LexiGaze eye-gaze tracking, cognitive load, and multimodal data fusion modules.

We have built a dedicated **Performance Inspection Sandbox** (`scripts/inspect_performance_demo.py`) that lets developers evaluate the combined effects of different gaze decoders, cognitive models, and fusion algorithms on system-level metrics (accuracy, latency, and difficulty correlation) at the same time.

---

## 🛠️ Sandbox Performance Inspection System

To check the effect of individual and combined modules on overall system performance, run the sandbox inspector from the project root:

```bash
# Run the joint performance inspection dashboard
.venv\Scripts\python.exe scripts/inspect_performance_demo.py
```

### 📊 System Performance Metrics
The system measures three key indicators on a GECO reading trial (156 words) with simulated webcam noise (+45px vertical drift and 30-40px random jitter):
1. **Gaze Decoding Accuracy (%)**: The percentage of gaze points correctly mapped to their true word targets on the screen.
2. **RDS Correlation (Spearman's $\rho$)**: How well the fused Reading Difficulty Score (RDS) correlates with the actual human reading time (TRT). Higher correlation indicates a more accurate representation of cognitive bottlenecks.
3. **Latency (ms)**: The total processing duration of the gaze-decoding, cognitive-feature, and data-fusion pipeline.

### 📈 Comparative Results (Simulated Sandbox)
The dashboard outputs the following results for the six default configurations:

| Configuration | Gaze Accuracy (%) | RDS Correlation ($\rho$) | Latency (ms) | Performance Impact |
| :--- | :---: | :---: | :---: | :--- |
| **1. Raw Gaze + No Cog + Linear** | 18.59% | 0.0636 | 1.51 ms | **Baseline**: Low accuracy due to systematic webcam drift; low correlation. |
| **2. Viterbi + No Cog + Linear** | 48.72% | 0.0910 | 198.76 ms | **Spatio-Temporal Prior**: Adding Viterbi sequence decoding corrects raw gaze drift. |
| **3. Viterbi + EM Calib + No Cog + Linear** | 73.72% | 0.2050 | 305.75 ms | **EM Self-Calibration**: Online calibration pushes gaze correction to 73% accuracy. |
| **4. STOCK-T v1 + surprisal + Linear** | 37.18% | 0.1438 | 178.13 ms | **Global Attention**: Incorporates BERT attention but has fixed prior limits. |
| **5. STOCK-T v2 + surprisal + Mult.** | 33.33% | 0.1066 | 170.75 ms | **Sparsified Attention**: Cognitive-gated top-k anchors; focuses on hard words. |
| **6. STOCK-T v3 + surprisal + Bayesian** | **78.21%** | **0.2258** | 285.25 ms | **Optimal Joint System**: Combines POM transition models, EM-calibration, and Bayesian fusion. |

---

## 🏗️ Modules Architecture & Interaction

LexiGaze fuses three research branches into a single system:

```
                  ┌───────────────────────────────┐
                  │   shengwen/ (Gaze Tracking)   │
                  │   - MediaPipe Preprocessor    │
                  │   - UniGaze-B16 Model         │
                  └───────────────┬───────────────┘
                                  │ Webcam Coordinates
                                  ▼
┌──────────────────┐      ┌────────────────────────┐      ┌───────────────────┐
│  weichi/ (NLP)   ├─────►│  chenghao/ (Flask App) ◄├─────┤ scripts/ (Fusion) │
│  - Surprisal     │      │  - word_track.html     │      │ - fusion_module.py│
│  - Rényi Entropy │      │  - server.py (API)     │      │ - Viterbi, EM     │
└──────────────────┘      └────────────────────────┘      └───────────────────┘
```

1. **Perception Module (`shengwen/`)**: Preprocesses webcam frames and estimates real-time screen coordinates via the UniGaze-B16 ViT model.
2. **Cognition Module (`weichi/`)**: Runs text through the transformer-based pipeline to calculate per-word cognitive features (information surprisal, entropy, syntactic load, AoA).
3. **Fusion Module (`scripts/fusion_module.py`)**: Fuses gaze occurrences with cognitive load to calculate the Reading Difficulty Score (RDS).
4. **Integration Server (`chenghao/`)**: Serves the frontend web interface, matches gaze coordinates with word layout bounding boxes, and handles HTTP API endpoints.

---

## 🔄 Web Platform Demo Steps

To demo the complete integrated system via the web browser:

### Step 1: Start the Integration Server
Launch the single entry point `chenghao/server.py` with UTF-8 support:
```bash
python -X utf8 chenghao/server.py
```
Open `http://localhost:8080` in your web browser.

### Step 2: Extract Bounding Boxes
1. On the main page (`word_track.html`), upload a sample PDF, HTML, or TXT document.
2. Click **Extract Coordinates** to render the document and extract pixel-level word coordinates.

### Step 3: Run Cognitive Load Analysis
1. Select the language (English or Chinese) and click **Analyze Cognitive Load**.
2. The backend runs the `weichi/` pipeline and returns `load_score` values.
3. Toggle the **Heatmap** slider to display the visual overlay highlighting difficult words.

### Step 4: Calibrate Gaze Tracking
1. Navigate to the Gaze Page at `http://localhost:8080/gaze`.
2. Follow the 9-point grid to collect calibration frames.
3. Click **Train Model** to run the personalization training, saving the polynomial model to `gaze_data/runs/`.

### Step 5: Test Gaze-Cognitive Fusion
1. Return to the main page, toggle the **Live Gaze Tracking** connection.
2. Read the text naturally. The browser captures your gaze points, aligns them to the word bounding boxes, and accumulates word dwell times.
3. At the end of the session, the frontend posts the aggregated gaze events to `/api/fuse`.
4. The server runs `scripts/fusion_module.py` to calculate per-word RDS and displays the fused difficulty highlights (High/Medium/Low) directly on the document.

---

## 🔗 How to Integrate Fusion Methods into the Flask Server

To integrate the different fusion algorithms from `scripts/fusion_module.py` directly into the web platform's API endpoint, update `chenghao/fusion_routes.py`:

```python
# In D:/projects/lexigaze/chenghao/fusion_routes.py

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fusion_module import LexiGazeFusion

# Initialize fusion class
fusion_engine = LexiGazeFusion()

@fusion_bp.post("/")
def fuse():
    body = request.get_json(force=True) or {}
    gaze_events = body.get("gaze_events", [])
    cognitive_result = body.get("cognitive_result", {})
    algorithm = body.get("algorithm", "linear")  # Allow selecting fusion method from client
    
    # Extract dwell, fixations, and load scores
    # ... (aggregate gaze events as we do in server)
    
    # Run the selected fusion algorithm
    if algorithm == "multiplicative":
        rds = fusion_engine.fuse_multiplicative(g_dwell, g_fix, l_score)
    elif algorithm == "bayesian":
        rds = fusion_engine.fuse_bayesian(g_dwell, l_score)
    elif algorithm == "rrf":
        rds = fusion_engine.fuse_rrf(g_dwell, l_score)
    else:
        rds = fusion_engine.fuse_linear(g_dwell, g_fix, l_score)
        
    return jsonify({"ok": True, "rds": rds})
```
