# LexiGaze: Multimodal Eye-Gaze & Cognitive Load Integration Platform

Webcam gaze tracking alone is often too inaccurate for high-fidelity reading research. **LexiGaze** solves this by fusing real-time neural eye-gaze tracking (where the user looks) with symbolic natural language processing cognitive load models (how difficult the text is) onto a shared coordinate reference frame.

By mapping gaze coordinates directly to pixel-level bounding boxes of words, researchers can run higher-dimensional analysis, correcting gaze noise and identifying true processing bottlenecks during natural reading.

---

## 🚀 Key Features

* **Gaze Tracking & Personalization**: Real-time webcam gaze prediction using MediaPipe and UniGaze-B16, with 13-target standard calibration or a five-block motion-diverse protocol and leakage-resistant grouped validation.
* **Cognitive Load Analysis**: NLP analysis utilizing GPT-2 (English) and BERT (Chinese) to calculate word surprisal, contextual entropy, lexical frequency, and syntactic complexity.
* **Multimodal Data Fusion**: Joint algorithms (Linear, Multiplicative, Attention-Gated, Sigmoid, Bayesian, and Reciprocal Rank Fusion) that combine gaze attention and cognitive features into a unified **Reading Difficulty Score (RDS)**.
* **Cognitive Inspector**: Diagnoses user reading proficiency, words-per-minute (WPM), regression trends, English capability levels, and cognitive fatigue.

---

## 📂 Project Structure & Module Ownership

The project is structured into three main research modules and an integration hub:

```
lexigaze/
├── core/                           # 🧠 CORE BUSINESS LOGIC CONTAINER
│   ├── cognition/                  # NLP cognitive pipeline and pre-trained weights
│   ├── cognitive_inspector/        # Diagnostic analyzer and Markdown report generator
│   ├── gaze_core/                  # Gaze prediction filters, training, and registries
│   └── unigaze_personalization/    # MediaPipe preprocessing, datasets, and ONNX models
│
├── web/                            # 🌐 THE MAIN FLASK WEB APPLICATION PACKAGE
│   ├── routes/                     # Blueprints (cognitive, gaze, demo, fusion, inspector)
│   ├── static/                     # Frontend scripts (mapping.js, gaze_integration.js) and styles
│   └── templates/                  # HTML views (word_track.html, gaze_page.html)
│
├── scripts/                        # 🧪 BENCHMARKS & EXPERIMENTAL UTILITIES
│   ├── fusion/                     # Offline data fusion orchestrator
│   ├── geco/                       # NeurIPS gaze correction and Viterbi decoding benchmark
│   ├── inspect_performance_demo.py  # Sandboxed comparative dashboard for system metrics
│   └── experiment_fusion.py        # Compares fusion methods on GECO human eye-tracking data
│
├── data/                           # Document coordinate layouts and calibration databases
├── output/                         # Auto-generated experiment plots and evaluation reports
└── docs/                           # 📄 SYSTEM DOCUMENTATION & ARCHIVE
```

---

## 📖 Documentation Sitemap

To avoid mixed purposes, our documentation is structured as follows:

| Document | Purpose | Key Contents |
| :--- | :--- | :--- |
| **[README.md](file://D:/projects/lexigaze/README.md)** | Project Landing Page | High-level overview, key features, folder structure, quick links. |
| **[ARCHITECTURE.md](file://D:/projects/lexigaze/ARCHITECTURE.md)** | Technical Architecture | Subsystems, detailed perception/cognition data flows, JSON schemas. |
| **[INSTRUCTION.md](file://D:/projects/lexigaze/INSTRUCTION.md)** | Operations & Walkthrough | Setup, step-by-step testing workflow, performance diagnostic guide. |
| **[INSTRUCTION_DATA.md](file://D:/projects/lexigaze/INSTRUCTION_DATA.md)** | Distributed Setup | Setup guide for Ubuntu server and Windows laptop client. |
| **[AGENT.md](file://D:/projects/lexigaze/AGENT.md)** | Developer Rules | Code quality standards, imports, relative API rules. |
| **[CONTRIBUTING.md](file://D:/projects/lexigaze/CONTRIBUTING.md)** | Git Guidelines | Branch names, commit messages, and collaborative pull requests. |
| **[docs/testing.md](docs/testing.md)** | Safe Validation | Offline CPU gate, CI boundary, and heavy-test policy. |
| **[Gaze benchmark contract](docs/performance/gaze-inference-benchmark.md)** | Hardware Validation | Guarded workloads, parity, telemetry, and production TF32 scope. |
| **[Hardware methodology ADR](docs/adr/0001-hardware-performance-methodology.md)** | Performance Decisions | Acceptance gates, data isolation, and reproducibility requirements. |
| **[Motion-diverse calibration ADR](docs/adr/0003-motion-diverse-gaze-calibration.md)** | Gaze Robustness Decision | Five-block capture protocol, grouped validation, and promotion rules. |
| **[Motion robustness experiment](docs/experiments/2026-08-05-gaze-motion-robustness.md)** | Experiment Record | Historical-data audit, implementation, tests, GPU gates, and next capture. |

---

## ⚡ Quick Start

For detailed requirements and alternative installations (such as Standard `venv` or `conda`), please see **[INSTRUCTION.md](file://D:/projects/lexigaze/INSTRUCTION.md)**.

### 1. Fast Setup with uv
```bash
# Sync dependency environment
uv sync

# Download spaCy English model for NLP pipeline
.venv/bin/python -m spacy download en_core_web_sm
```

### 2. Configure Environment
Create a `.env` file in the project root:
```env
# Optional: Path where Hugging Face weights are cached (defaults to ~/.cache/huggingface)
# HF_HOME="/home/ubuntu/.cache/huggingface"

GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME="gemma-4-26b-a4b-it"
```

### 3. Run the Server
Launch the Flask backend:
```bash
# Use the UTF-8 flag to avoid console printing crashes
.venv/bin/python -X utf8 run.py
```
Open **`http://localhost:8080`** in your browser.

### 4. Run the safe regression gate

```bash
python -X utf8 -m scripts.run_offline_quality_gate
```

This lane is CPU-only, blocks test-time network access, clears provider credentials, and does not install or import PyTorch. See [docs/testing.md](docs/testing.md) for its scope and the separate heavy-test policy.
