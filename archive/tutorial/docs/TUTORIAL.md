# LexiGaze: Neuro-Symbolic Workshop

## Laptop Setup
Prepare your environment before the session starts:

1. Editor: Use any editor you prefer (VSCode, Cursor, Zed, etc.). For minimalist enthusiasts, Vim or Nano are also perfectly fine.
2. Google Account: Required for accessing Google AI Studio and Colab. A standard free-tier account is sufficient.
3. Environment: Ensure you have Python 3.11+ installed. We recommend using a virtual environment.

## Getting Started

```bash
# 1. Download the specific tutorial branch
git clone -b tutorial/lets-goooo https://github.com/momonong/lexigaze.git

# 2. Enter the project directory
cd lexigaze

# 3. Install Gemini CLI and complete login
npm install -g @google/gemini-cli
gemini login
```

## Overview
This workshop implements a Neuro-Symbolic AI architecture. We use Neural Perception (WebGazer) to capture gaze and Symbolic Cognition (LLM/BERT) to extract linguistic priors. These are fused via Vibe Coding to calibrate hardware errors.

---

## Core Methodology: Vibe Coding & Skill Building
Unlike traditional coding, we use LLMs to translate academic knowledge into executable code. This process applies to every module (Webcam, Text Model, and Calibration).

### 1. Build Skills (Knowledge Translation)
Translate PDF papers into structured Markdown "Skills":
```bash
python tutorial/skill_builder.py
```

### 2. Agentic Refinement (Continuous Improvement)
Use the generated skills in tutorial/skills/ as context for your AI agent (Cursor, Gemini, etc.) to optimize:
- Webcam: Improve noise filtering based on eye-tracking literature.
- Text Model: Refine surprisal normalization based on linguistic papers.
- Calibration: Design advanced snap algorithms based on cognitive science.

---

## Hands-on Workshop Steps

### Step 0: Environment & API Setup
Configure your Gemini API key:
1. Obtain a key from Google AI Studio (https://aistudio.google.com/).
2. Create a .env file in the root: GEMINI_API_KEY="your_key".

### Step 1: Neural Perception (Data Collection)
Capture raw, noisy gaze coordinates using a web-based eye-tracker.
1. Start server: python tutorial/server.py.
2. Open http://localhost:8000 and read the text.
3. Export data to tutorial/data/raw.csv.

### Step 2: Symbolic Cognition (Quantitative Analysis)
Analyze linguistic difficulty to guide calibration using deep learning.
1. Open the Google Colab notebook.
2. Run the bert-tiny analysis to compute "Word Surprisal".
3. Save cognitive_weights.json to tutorial/data/.

### Step 3: Neuro-Symbolic Fusion (Calibration)
Apply symbolic priors to noisy neural perception data.
1. Execute engine: python tutorial/calibrate.py.
2. Implements "Gravity Snap" using cognitive weights.
3. Outputs results to tutorial/data/calibrated.csv.

### Step 4: Verification (Visual Analytics)
Verify the performance improvement via trajectory animation.
1. Generate GIFs: python tutorial/visualize.py --target all.
2. View results in tutorial/figures/.

## Core Component Summary
- `server.py`: Local host for eye-tracking frontend.
- `text_model.py`: Quantitative engine for computing Word Surprisal using BERT models.
- `skill_builder.py`: Asynchronous LLM-powered knowledge translator.
- `calibrate.py`: Fusion engine for perception and cognition.
- `visualize.py`: Gaze trajectory animator.

