# 🧠 IntelligentGaze: Neuro-Symbolic & Edge AI Hands-on Workshop

**Supervising Professor:** [Insert Name] | **Instructor:** Morris (Gemi)
**Estimated Time:** 90 Minutes

## 🎯 Course Objective (The "Why")
In Edge AI, due to hardware compute limitations and environmental interference, pure Neural Network perception is often noisy and prone to systematic errors. 

Today, we are not training massive models. Instead, we will guide you in implementing a cutting-edge **Neuro-Symbolic AI** architecture: 
Using the "Symbolic Prior" extracted by a Large Language Model to calibrate the "Neural Perception" of an edge eye-tracker.

---

## 🛠️ Phase 1: The Harsh Reality — Collecting Edge Perception Data
Signals from edge devices are inherently dirty (noisy).

1. Start the local data collection server:
   ```bash
   python server.py
   ```
2. Open your browser to **`http://localhost:8000`** and read the short text.
3. The system will capture your gaze coordinates via WebGazer.
4. Export your personal dataset: `raw.csv`.

*(Reflection point: Can a traditional Moving Average filter solve the issue of your entire gaze trajectory systematically drifting downwards?)*

---

## 📖 Phase 2: Extracting the Symbolic Brain — Surprisal Theory
When humans read, the brain automatically allocates higher cognitive load to "rare or difficult words," causing our gaze to linger (Fixation). We will use AI to mathematically quantify this process.

**Your Vibe Coding Task:**
Open Colab. We will load the ultra-lightweight `bert-tiny` model (only 18MB, ensuring no network crashes for a class of 50).
1. Use Masked Language Modeling (MLM) to mask difficult words in the text (e.g., *phenomenon*).
2. Write a Python snippet to calculate the model's predicted probability ($p$) for that word.
3. Use the formula $-ln(p)$ to calculate the word's "Surprisal."
4. Normalize the Surprisal score into a gravitational weight ($\alpha$) between 0.0 and 1.0.

✅ **Deliverable:** `cognitive_weights.json` (Your symbolic rulebase)

---

## 🧲 Phase 3: Neuro-Symbolic Fusion Engine (The Vibe Coding Challenge)
Now, you possess both the noisy raw eye trajectory (`p1_raw.csv`) and the purely rational linguistic prior (`cognitive_weights.json`).

**Your Vibe Coding Task:**
In the sandbox section of `calibrate.py`, use natural language prompts to have the AI write a "gravitational snap" algorithm for you.
* **Challenge 1:** If the gravitational radius is set too small (e.g., 150px) and the hardware gaze drift is too severe, the snap won't trigger. Try designing a dynamic radius.
* **Challenge 2:** Combine the Moving Average baseline with the $\alpha$ weight to successfully pull the red dots (Raw) toward the green star (Target Word).

✅ **Deliverable:** Successfully plot your `heatmap_calibrated.png` comparison chart and witness firsthand how the brain's prior corrects hardware flaws!