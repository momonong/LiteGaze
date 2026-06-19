# LexiGaze: A Neuro-Symbolic & Vibe Coding Workshop Portfolio

## 1. Project Vision
LexiGaze is a pioneering educational framework that bridges the gap between hardware-constrained Neural Perception and high-level Symbolic Cognition. In the era of Edge AI, raw data from low-cost sensors (such as webcams) is inherently noisy and prone to systemic drift. This project demonstrates how to use the "Symbolic Prior" extracted by Large Language Models (LLMs) to calibrate and stabilize edge eye-tracking data.

## 2. Core Philosophy: The Neuro-Symbolic Bridge
Traditional AI often relies on massive neural networks to solve all problems, which is often unfeasible on edge devices. LexiGaze adopts a hybrid approach:
- **Neural Perception (The Eye)**: Uses WebGazer.js to capture real-time gaze coordinates. This is fast and accessible but mathematically "dirty."
- **Symbolic Cognition (The Brain)**: Uses BERT and LLMs to understand the linguistic importance of words in a sentence (Surprisal Theory).
- **Fusion**: Calibrates the noisy eye-tracker by "snapping" gaze points toward high-surprisal words, treating human language as a gravitational prior.

## 3. Methodology: Vibe Coding & Skill Implementation
LexiGaze moves beyond traditional syntax-heavy programming into the realm of **Vibe Coding**. This paradigm focuses on:
- **Contextual Engineering**: Providing AI agents with rich academic context (papers, formulas, and domain logic) rather than just coding instructions.
- **Skill Translation**: Using the `skill_builder.py` pipeline to convert raw academic PDF literature into structured, machine-readable "Agent Skills."
- **Declarative Development**: Defining *what* knowledge is needed via `questions.yaml` and allowing the AI to architect the *how* of the implementation.

## 4. Technical Architecture: The Five-Phase Pipeline

### Phase 0: Knowledge Foundation
Setting up the environment and API keys to power the LLM "Brain."

### Phase 1: Neural Perception (Data Collection)
Deploying a local server to capture raw webcam gaze data while a user reads a text containing complex linguistic structures.
- **Output**: `raw.csv`

### Phase 2: Quantitative Cognition (Information Theory)
Utilizing Google Colab and BERT-Tiny to calculate the Pseudo-Log-Likelihood (PLL) of words. This identifies "cognitive anchors" in the text.
- **Output**: `cognitive_weights.json`

### Phase 3: Skill Building (Symbolic Extraction)
Running an asynchronous translation pipeline that interrogated academic papers to extract calibration logic and domain-specific parameters.
- **Output**: `skill_*.md`

### Phase 4: Hybrid Fusion (Calibration)
Implementing the "Gravity Snap" algorithm where neural coordinates are attracted to symbolic anchors based on their calculated cognitive weight (Alpha).
- **Output**: `calibrated.csv`

### Phase 5: Visual Analytics (Verification)
Generating high-fidelity trajectory animations and Kernel Density Estimation (KDE) heatmaps to verify the reduction in systemic drift.
- **Output**: `trajectory.gif`, `dashboard.png`

## 5. Scientific Pillars
- **Surprisal Theory**: Quantifying the information content of language.
- **Eye-Tracking Science**: Understanding Fixations vs. Saccades.
- **Information Theory**: Using entropy-based metrics to guide AI calibration.
- **Edge AI Optimization**: Making sophisticated AI logic run on low-power hardware through symbolic intervention.

## 6. Technology Stack
- **Language**: Python 3.11+
- **Deep Learning**: PyTorch, HuggingFace Transformers (BERT)
- **AI Infrastructure**: Google Gemini API, Google AI Studio
- **Data Science**: Pandas, NumPy, Scipy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: WebGazer.js, Flask/HTTP Server
- **Environment**: uv, npm, dotenv
