# UniGaze-B Personalization

This project uses `unigaze_b16_joint` as the non-personalized webcam gaze baseline, then trains a lightweight per-user adapter from calibration samples.

## Baseline Model

- **Model**: `unigaze_b16_joint`
- **Source**: `unigaze` PyPI loader
- **Weights**: `UniGaze/UniGaze-models` on Hugging Face
- **Input**: 224x224 normalized RGB face image
- **Output**: gaze pitch/yaw

## Architecture

```
Input: 224x224 Webcam Face Image
  → UniGaze-B16 (frozen) → gaze pitch/yaw + visual features
  → [optional] + head_pose
  → ScreenAdapter (trained) → normalized screen (x, y) in [-1, 1]
```

The adapter is intentionally tiny (162 trainable parameters) to avoid overfitting with minimal calibration samples.

## Install

```powershell
# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

Note: Requires Python 3.10+ and a CUDA-capable GPU for training.

## Quick Start

### 1. Verify Installation

```powershell
ugaze-smoke
```

First run downloads `unigaze_b16_joint.safetensors` from Hugging Face.

### 2. Collect Calibration Data

```powershell
ugaze-collect --host 127.0.0.1 --port 8000 --data-dir data/sessions
```

Open http://127.0.0.1:8000 in your browser.

**Collection Flow:**
1. Browser requests webcam permission
2. Click `New Session` to create `data/sessions/<session_id>/`
3. Click `Calibration` to start 13-point gaze collection
4. Look at each target point - the system captures webcam frames
5. Data is processed via MediaPipe + face normalization and saved to `manifest.jsonl`

**13 Calibration Points:**
```
(0.08, 0.10)  (0.50, 0.10)  (0.92, 0.10)
(0.08, 0.50)  (0.50, 0.50)  (0.92, 0.50)
(0.08, 0.90)  (0.50, 0.90)  (0.92, 0.90)
(0.29, 0.30)  (0.71, 0.30)
(0.29, 0.70)  (0.71, 0.70)
```

### 3. Train Personalization Adapter

```powershell
ugaze-train-adapter `
  --manifest data\sessions\<session_id>\manifest.jsonl `
  --output-dir runs\<session_id>_adapter
```

Options:
- `--adapter-input` - Choose adapter input: `gaze` (default), `gaze_pose`, or `features`
- `--epochs` - Training epochs (default: 80)
- `--hidden-dim` - Adapter hidden dimension (default: 32)

### 4. Evaluate

```powershell
ugaze-eval-adapter `
  --manifest data\sessions\<session_id>\manifest.jsonl `
  --adapter-checkpoint runs\<session_id>_adapter\adapter_best.pt
```

### 5. Inference

```powershell
ugaze-infer --image path\to\webcam_frame.jpg --json
```

## Training Data Flow

### Raw Data Collection

```
User looks at screen target → Webcam captures frame → raw/*.jpg
```

### Preprocessing (MediaPipeUniGazePreprocessor)

```
raw image
  → MediaPipe FaceMesh (468 landmarks)
  → solvePnP → head pose (rvec, tvec)
  → compute face center (nose + eyes)
  → perspective warp → normalized 224x224 face
```

### Output Files per Sample

| File | Description |
|------|-------------|
| `raw/000000.jpg` | Original webcam capture |
| `crop/000000.jpg` | Face region crop |
| `normalized_face/000000.jpg` | 224x224 perspective-normalized face |

### manifest.jsonl Record

```json
{
  "normalized_face_path": "normalized_face/000000.jpg",
  "head_pose_pitch_yaw": [0.123, -0.456],
  "target_x_norm": -0.8,
  "target_y_norm": -0.8,
  "viewport_width": 1280,
  "viewport_height": 720
}
```

### Model Input (CalibrationDataset)

```python
{
    'image':      tensor (3, 224, 224)  # ImageNet normalized
    'head_pose':  tensor (2,)           # [pitch, yaw]
    'target':     tensor (2,)           # target (x, y) in [-1, 1]
    'viewport':   tensor (2,)           # (width, height)
}
```

## Project Structure

```
.
├── src/unigaze_personalization/
│   ├── model.py          # UniGazeFeatureWrapper, ScreenAdapter, PersonalizedUniGaze
│   ├── dataset.py        # CalibrationDataset, manifest reading
│   ├── transforms.py     # ImageNet normalization, coordinate transforms
│   ├── preprocess.py     # MediaPipe face detection & normalization
│   ├── train_adapter.py  # Training loop
│   ├── eval_adapter.py   # Evaluation
│   ├── infer.py          # Inference
│   ├── server.py         # FastAPI server for data collection
│   └── smoke_test.py     # Quick sanity check
├── web/static/
│   ├── index.html        # Collection UI
│   ├── app.js            # Frontend logic (13-point calibration)
│   └── styles.css        # Styling
├── data/
│   └── sessions/         # Collected calibration data
└── runs/                 # Trained adapters
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ugaze-smoke` | Verify model loads correctly |
| `ugaze-collect` | Start data collection server |
| `ugaze-train-adapter` | Train personalization adapter |
| `ugaze-eval-adapter` | Evaluate trained adapter |
| `ugaze-infer` | Run inference on a single image |
