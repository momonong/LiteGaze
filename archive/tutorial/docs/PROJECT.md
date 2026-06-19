# LexiGaze: Neuro-Symbolic & Vibe Coding Project Overview

## Current Project Status

LexiGaze is a research-oriented framework designed to demonstrate the application of Neuro-Symbolic AI in Edge computing environments. The project addresses the inherent limitations of low-cost hardware perception (e.g., webcam gaze tracking) by integrating Large Language Model (LLM) derived linguistic priors to calibrate systemic drift and sensor noise.

The workflow has been optimized into a modular pipeline that separates heavy neural computation (BERT-based quantitative analysis) from local symbolic logic and LLM-assisted knowledge translation.

### Core Workflow Architecture

| Component | Category | Task | Primary Tools |
| :--- | :--- | :--- | :--- |
| **Perception** | Neural | Hardware Gaze Capture | `server.py`, `index.html` |
| **Cognition** | Symbolic | Quantitative Prior Extraction | `text_model.py` (via Colab) |
| **Translation** | LLM | Knowledge-to-Skill Mapping | `skill_builder.py` |
| **Fusion** | Hybrid | Neuro-Symbolic Calibration | `calibrate.py` |
| **Verification** | Analytics | Trajectory & Density Analysis | `visualize.py`, `heatmap.py` |

---

### Directory Architecture

- **tutorial/**: Core workshop resources and implementation scripts.
    - **server.py**: Local HTTP server for eye-tracking data collection.
    - **text_model.py**: BERT-based engine for calculating Word Surprisal (PLL).
    - **skill_builder.py**: Asynchronous LLM translator that converts academic papers into structured Agent Skills.
    - **calibrate.py**: The central fusion engine where students implement the "Gravity Snap" algorithm.
    - **visualize.py**: Diagnostic tool for generating high-fidelity gaze trajectory GIFs.
    - **heatmap.py**: Statistical tool for Kernel Density Estimation (KDE) comparison.
    - **knowledge/**: Repository of academic literature and prompt configurations for skill building.
    - **data/**: Centralized storage for raw, baseline, and calibrated CSV datasets.
    - **figures/**: Output directory for diagnostic visualizations and dashboard results.

---

### Technical Challenges & Mitigations

1. **Environmental Decoupling**: To ensure stability in classroom environments, the heavy `torch` and `transformers` dependencies have been moved to Google Colab, while the local environment remains lightweight.
2. **API Rate Limiting**: The `skill_builder.py` utilizes an asynchronous semaphore mechanism to respect the 15 RPM limits of free-tier Gemini API keys.
3. **Drift Compensation**: The static 150px radius in current snap algorithms remains a point of iteration; the project is moving toward dynamic radius models based on fixation duration.

---

### Strategic Roadmap

#### 1. Dynamic Gravity Models
Developing algorithms that adjust attraction radius based on real-time cognitive load indicators and fixation stability.

#### 2. On-Device Symbolic Reasoning
Exploring the conversion of linguistic priors into ONNX or TFLite formats for fully local, browser-based real-time calibration.

#### 3. Adaptive User Interfaces
Implementing UI components that respond to high-surprisal gaze patterns, providing automated visual assistance or content simplification.

---
*Last Updated: 2026-04-27*
