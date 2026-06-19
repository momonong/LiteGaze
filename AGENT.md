# 🤖 LexiGaze: Developer Rules & AI Agent Guidelines

This document defines guidelines, design principles, and coding standards for all AI agents and developers working on the **LexiGaze** repository.

---

## 🎯 Product Vision: Unobtrusive English Vocabulary Acquisition

LexiGaze is built to be a next-generation English learning system that merges **webcam-based eye-tracking** and **NLP cognitive text modeling** to enable effortless learning:
* **Zero Disruption**: Instead of requiring readers to pause, highlight, or manually look up unfamiliar words, the system identifies vocabulary gaps *implicitly*.
* **Fixation & Dwell Detection**: By analyzing real-time gaze trajectories (long dwell times, multiple fixations, and backward reading regressions), the system detects words where the reader is struggling.
* **Linguistic Surprisal & Complexity**: Using NLP models (BERT, GPT-2), the system computes lexical surprisal and entropy to determine if the target word is complex or rare.
* **Automated Vocabulary Book**: High-load words identified by the gaze-cognitive fusion pipeline are automatically compiled into a personalized vocabulary list. This allows readers to maintain their natural reading flow while building a study guide behind the scenes.

---

## 🏛️ System Architecture Overview

LexiGaze is divided into four main subsystems:
1. **Perception Module (`shengwen/`)**: Captures video frames via webcam, runs MediaPipe face mesh and landmark preprocessing, passes normalized face crops to the UniGaze-B16 ViT neural net, and fits polynomial calibration models to correct systematic eye-tracking offsets.
2. **Cognition Module (`weichi/`)**: Computes word-level linguistic complexity metrics (surprisal, contextual entropy, zipf frequency) to determine text difficulty.
3. **Fusion & Orchestrator (`scripts/`)**: Integrates raw gaze dwell duration and cognitive load weights using math fusion models (RRF, Bayesian Integration, etc.) to yield a final Reading Difficulty Score (RDS).
4. **Web UI & Server (`chenghao/`)**: Renders the document reading dashboard (word layout mapping and highlight overlay) and handles calibration recording and automated post-calibration training.

---

## ⚙️ Coding Standards & Constraints

### 1. Module Path Injection
Do NOT install `shengwen` or `weichi` as global or editable packages in `pyproject.toml`. Instead, load them dynamically at runtime by injecting their paths into `sys.path`:
* Gaze submodules append `shengwen/src/` inside [chenghao/gaze_core/__init__.py](file:///home/ubuntu/projects/lexigaze/chenghao/gaze_core/__init__.py).
* Cognitive submodules insert `weichi/` inside [chenghao/cognitive_routes.py](file:///home/ubuntu/projects/lexigaze/chenghao/cognitive_routes.py).

### 2. Relative API Routes
All API endpoints inside HTML or Javascript files under `chenghao/` must use relative paths (e.g., `/api/gaze/predict`). This allows the frontend to run seamlessly through public `ngrok` tunnels without domain rewrites.

### 3. Unicode & Console Printing
* Always launch Python scripts containing non-ASCII print outputs using the UTF-8 flag:
  ```bash
  python -X utf8 chenghao/server.py
  ```
* Always open project text files with explicit UTF-8 encoding:
  ```python
  with open(filepath, 'w', encoding='utf-8') as f:
  ```

### 4. Running Backend Headlessly
Ensure OpenCV face extraction processes do not rely on X11 display environments, using `opencv-python-headless` for server tasks.

---

## 🧠 Planned Feature: Cognitive Ability Inspector Agent

We are designing a specialized **Cognitive Ability Inspector** agent that operates on LexiGaze reading logs to evaluate and guide readers.

### Core Responsibilities
1. **Gaze Trajectory Analysis**: Scan gaze coordinate time series to detect abnormal dwell spikes, frequent regression steps (saccades jumping backward), and slow reading speed segments.
2. **Linguistic Diagnostic Mapping**: Analyze reading logs against different paragraphs (comparing behavior on easy vs. hard text templates in `examples/`) to locate the specific syntactic structures or vocabulary lists triggering cognitive overload.
3. **User Profile Generation**: Score the reader's attention retention, fatigue levels, vocabulary tier, and reading speed.
4. **Actionable Remedial Reporting**: Compile report documents offering tailored feedback, such as target reading speed exercises, readability suggestions, or text simplifications based on the user's implicit vocabulary capture logs.
