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
1. **Perception Module (`core/unigaze_personalization/`)**: Captures video frames via webcam, runs MediaPipe face mesh and landmark preprocessing, passes normalized face crops to the UniGaze-B16 ViT neural net, and fits polynomial calibration models to correct systematic eye-tracking offsets.
2. **Cognition Module (`core/cognition/`)**: Computes word-level linguistic complexity metrics (surprisal, contextual entropy, zipf frequency) to determine text difficulty.
3. **Fusion & Orchestrator (`scripts/`)**: Integrates raw gaze dwell duration and cognitive load weights using math fusion models (RRF, Bayesian Integration, etc.) to yield a final Reading Difficulty Score (RDS).
4. **Web UI & Server (`web/`)**: Renders the document reading dashboard (word layout mapping and highlight overlay) and handles calibration recording and automated post-calibration training.

---

## ⚙️ Coding Standards & Constraints

### 1. Unified Modular Packages
Do NOT use legacy path-injection hacks inside Python files. All modules must be imported natively using the consolidated package layouts:
* Core logic and submodules (gaze calculation, deep learning preprocessing, NLP pipelines) are imported from the `core` package.
* Flask routes and web application elements are imported from the `web` package.

### 2. Relative API Routes
All API endpoints inside HTML or Javascript files under `web/` must use relative paths (e.g., `/api/gaze/predict`). This allows the frontend to run seamlessly through public `ngrok` tunnels without domain rewrites.

### 3. Unicode & Console Printing
* Always launch Python scripts containing non-ASCII print outputs using the UTF-8 flag:
  ```bash
  python -X utf8 run.py
  ```
* Always open project text files with explicit UTF-8 encoding:
  ```python
  with open(filepath, 'w', encoding='utf-8') as f:
  ```

### 4. Running Backend Headlessly
Ensure OpenCV face extraction processes do not rely on X11 display environments, using `opencv-python-headless` for server tasks.

---

## 🧠 Core Feature: Cognitive Ability Inspector Agent (Fully Implemented)

The **Cognitive Ability Inspector** is a fully realized, integrated core module that analyzes sequential reading trajectories to assess user cognitive load, vocabulary level, reading ability, and fatigue.

### Features & Capabilities
1. **Gaze Trajectory & Saccade Sequence Analysis**: Evaluates chronological eye-tracking streams to compute:
   * **Fixation Duration** (dwell time per word group) $\rightarrow$ Reflects **Cognitive Load**.
   * **Fixation Count** (hit counts) $\rightarrow$ Reflects **Attention/Focus**.
   * **Regression Count** (backtracking transitions) $\rightarrow$ Reflects **Comprehension Difficulty**.
   * **Reread Count** (returning to previously read words) $\rightarrow$ Reflects **Vocabulary Difficulty**.
   * **Dwell Time** (total region/paragraph duration) $\rightarrow$ Reflects **Overall Burden**.
2. **Linguistic Diagnostic Profiles**: Correlates long fixations and regressions against word lexical characteristics (Zipf frequency) to determine:
   * **Reading Ability Score & Level**: Measures words per minute (WPM), regression rates, and avg fixation duration.
   * **English Proficiency Score & Level**: Evaluates if reader pauses are on rare technical words vs. common basic syntax.
   * **Fatigue Level (Low/Medium/High)**: Compares average fixation times in the second half of reading vs. the first half.
3. **Reports History & CRUD Panel**: Generates and persists detailed Markdown reports with actionable remedial advice to [`docs/cognitive_reports/`](file:///home/ubuntu/projects/lexigaze/docs/cognitive_reports/). The reading dashboard includes a history manager to reload, delete, or download reports.
