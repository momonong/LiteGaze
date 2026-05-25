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

> [!IMPORTANT]
> **Environment Requirements:**
> - **Python Version:** **Python 3.10 or 3.11** is strictly required (Python 3.9 or lower is **not** supported).
> - **CUDA GPU:** A CUDA-capable GPU is recommended for training.
> - **Pip Version:** An up-to-date `pip` is required to correctly install modern dependencies.

Follow these steps to set up the environment and install dependencies:

### 1. Verify Python Version
Before installing, check your system's Python version:
```powershell
python --version
```
If your version is lower than `3.10` (e.g., Python 3.9), please download and install **Python 3.10 or 3.11** from the [official Python website](https://www.python.org/downloads/) or the Microsoft Store.

### 2. Create and Set Up the Virtual Environment

Create the virtual environment using Python 3.10+:
```powershell
# If your default 'python' command is already Python 3.10 or 3.11:
python -m venv .venv

# OR, if you installed multiple versions and have the Python launcher (py) installed:
py -3.10 -m venv .venv
# or
py -3.11 -m venv .venv
```

### 3. Upgrade Pip & Install Dependencies

To avoid errors caused by outdated `pip` versions, upgrade `pip`, `setuptools`, and `wheel` inside the virtual environment first, then install the package:
```powershell
# Upgrade pip and key packaging utilities
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

# Install the project in editable mode with all dependencies
.venv\Scripts\python.exe -m pip install -e .
```

## Quick Start

```powershell
.venv\Scripts\activate
python -m src.unigaze_personalization.server
```
