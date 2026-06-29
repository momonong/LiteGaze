# 🏗️ LexiGaze: System Architecture & Data Schema Specification

This document details the software architecture, module interaction models, data flows, and database schemas of the **LexiGaze** platform.

---

## 🏛️ Subsystem Design

LexiGaze is divided into four decoupled subsystems, each responsible for a distinct aspect of the multimodal perception-cognition fusion pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           WEB INTEGRATION PORTAL                        │
│                                 (web/)                                  │
│  - Routes Blueprints          - Templates (word_track)  - styles        │
│  - mapping.js                 - gaze_integration.js     - gaze_page     │
└──────┬────────────────────────────┬────────────────────────────▲────────┘
       │ raw frames                 │ doc text                   │ RDS / Inspector
       ▼                            ▼                            │ reports
┌──────────────────────┐    ┌──────────────────────┐    ┌────────┴────────┐
│  PERCEPTION MODULE   │    │   COGNITION MODULE   │    │  FUSION ENGINE  │
│   (core/gaze_core)   │    │   (core/cognition)   │    │    (scripts)    │
│  - Landmark Detect   │    │  - HuggingFace LLMs  │    │  - Viterbi POM  │
│  - UniGaze ONNX ViT  │    │  - Surprisal Engine  │    │  - EM Calibrate │
│  - Polynomial Reg    │    │  - XGBoost Scorer    │    │  - Multi. Sum   │
└──────────────────────┘    └──────────────────────┘    └─────────────────┘
```

### 1. Perception Module (`core/gaze_core/` & `core/unigaze_personalization/`)
Processes real-time webcam streams to track eye coordinates:
* **Preprocessing**: Face landmarker extracts 3D facial landmarks and crops the face region.
* **Neural Gaze Prediction**: The cropped face is transformed and passed to a pre-trained UniGaze-B16 ViT model (loaded via ONNX Runtime), outputting pitch/yaw angles.
* **Calibration Adaptation**: Corrects eye-tracking systematic bias (e.g. laptop-screen posture changes) using a polynomial regression model fit to a 9-point grid.
* **Smoothing**: Appling OneEuro and horizontal corridor filters to minimize jitter.

### 2. Cognition Module (`core/cognition/`)
Analyzes text linguistic complexity to determine reading difficulty:
* **Linguistic Pipeline**: Extracts word tokens and dependency-parsed relations using spaCy.
* **Surprisal & Entropy Engine**: Passes the text through local language models (GPT-2 for English, BERT for Chinese) to compute lexical surprisal and contextual entropy.
* **XGBoost Classifier**: Merges surprisal, Age-of-Acquisition (AoA) norms, Zipf word frequency, and dependency load to output a single cognitive load score per word.

### 3. Fusion Engine (`scripts/` / `scripts/fusion_module.py` & `scripts/geco/core/`)
Fuses tracking inputs with cognitive models offline or at the end of a session:
* **STOCK-T Sequence Decoding**: Implements the Spatio-Temporal Sequence Decoder (`viterbi_decoder.py`) and Auto-Calibrating EM (`em_calibration.py`) to resolve systematic drift and vertical line-locking.
* **OSTMC & PAOAT Constraints**: Employs **Oculomotor Spatio-Temporal Monotonicity Constraints** to enforce directional reading layouts (penalizing backward/multi-line skips) and **Proficiency-Adaptive OVP Anchor Tuning** to scale foveal target anchors dynamically.
* **Math Fusion**: Implements six combination metrics (Linear, Multiplicative, Gated, Sigmoid, Bayesian, and Reciprocal Rank Fusion) to generate a unified Reading Difficulty Score (RDS).

### 4. Web Portal (`web/`)
The web presentation framework (built on Flask) providing interactive reading dashboards, real-time webcam rendering, and detailed markdown diagnostic reports.

---

## 🔄 Core Data Pipelines

### 1. Eye-Gaze Tracking Flow
```
[Webcam Frame]
       │ (640x480 Base64 JPEG)
       ▼
[MediaPipe Landmark Processor] ──► Crop & Pose Estimation
       │
       ▼ (Normalized Face Image)
[UniGaze ONNX Model] ────────────► Raw Gaze Vector [Pitch, Yaw]
       │
       ▼
[Polynomial Adapter Model] ──────► Corrected Screen Coordinates [X, Y]
       │
       ▼
[OneEuro / Corridor Filter] ─────► Smoothed Coordinates (Dwell/Fixations)
```

### 2. Cognitive Analyzer Flow
```
[Uploaded Document (PDF/MD/TXT)]
       │
       ▼
[Document Coordinate Extraction] ──► DOM Word Bounding Boxes [x, y, w, h]
       │
       ▼ (Extracted Text Sequences)
[HuggingFace GPT-2/BERT] ──────────► Surprisal & Contextual Entropy
       │
       ▼
[XGBoost Classifier Model] ────────► Predicted Cognitive Load Scores
```

### 3. Joint Multimodal Fusion Flow
```
[Raw Coordinates (x, y)] ──► [Dynamic Sliding EM Calibration] ──► [Corrected Gaze] ──┐
                                                                                   ├──► [Viterbi Sequencer (POM + OSTMC + PAOAT)] ──► [Snapped Word Indices] ──► [Word RDS]
[Linguistic Features] ────► [XGBoost Cognitive load_score] ────────────────────────┘
```

---

## 📊 Database & File Schemas

### 1. Calibration Session Manifest (`manifest.jsonl`)
Located under `data/sessions/<session_id>/manifest.jsonl`. Tracks collected calibration frames:
```json
{
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

### 2. Personalization Model Config (`<model_name>.json`)
Located under `examples/models/<model_name>.json`. Contains calibration regression coefficients:
```json
{
  "name": "user_laptop_01_model",
  "created_at": "2026-06-20T16:00:00",
  "data_session_id": "session_20260620_160000",
  "mean_px_error": 18.4,
  "noise_level": 5.2,
  "train_samples": 18,
  "stages": [
    {
      "stage": 1,
      "poly_degree": 2,
      "W": [
        [0.82, 0.04],
        [0.03, 0.81],
        [0.01, 0.02]
      ],
      "mean_px_error": 18.4
    }
  ]
}
```

### 3. Document Coordinate Layout (`<session_id>.json`)
Located under `data/<session_id>.json`. Details pixel-level coordinate boxes for each word:
```json
{
  "id": "72c20283-7bd7-49cd-bbaa-b9d5d9ba5567",
  "filename": "document.md",
  "filetype": "md",
  "created_at": "2026-06-20T16:05:00",
  "item_count": 1,
  "items": [
    {
      "page": 1,
      "index": 0,
      "text": "LexiGaze",
      "top": 44.24,
      "left": 51.99,
      "width": 80.0,
      "height": 18.7,
      "norm_left": 0.0655,
      "norm_top": 0.0942,
      "norm_width": 0.0399,
      "norm_height": 0.04
    }
  ]
}
```
