# Real-Time System Evaluation Framework for LexiGaze

This document outlines a comprehensive framework for evaluating the real-time performance of LexiGaze across its constituent modules (M1–M5) and user interaction layers. 

While offline benchmarks (such as Pearson/Spearman correlation on static corpora) establish a baseline for algorithm development, they are insufficient for real-time systems. A real-time reading inspector must be evaluated for **responsiveness, oculomotor stability, snapping reliability, and actual human cognitive benefits**.

---

## 1. Evaluation Architecture: The Three Pillars

To evaluate LexiGaze comprehensively, we define a three-pillar testing architecture:

```
                  +-----------------------------------+
                  | LexiGaze Real-Time Evaluation     |
                  +-----------------+-----------------+
                                    |
         +--------------------------+--------------------------+
         |                          |                          |
+--------v--------+        +--------v--------+        +--------v--------+
|    Pillar 1     |        |    Pillar 2     |        |    Pillar 3     |
|  Oculomotor &   |        | System Latency  |        |  Human-in-the-  |
|  Snapping Dyn.  |        | & Resource Prof. |        |  Loop Usability |
+--------+--------+        +--------+--------+        +--------+--------+
         |                          |                          |
         +--> Snap Jitter Rate      +--> End-to-End Latency    +--> A/B Comp. Test
         +--> Regression Capture    +--> Frame Rate (FPS)      +--> NASA-TLX Index
         +--> Line Snap Stability   +--> Memory Footprint      +--> Reading Speed (WPM)
```

---

## 2. Pillar 1: Oculomotor & Snapping Dynamics (M1, M4, M5)

Webcam eye-gaze tracking is noisy. The role of M4 (Spatiotemporal Snapping) and M5 (Trajectory Fusion) is to reconstruct a stable reading path. We evaluate this using the following real-time metrics:

### 2.1 Snap Jitter Rate (SJR)
* **What it measures**: The frequency of vertical snapping flips when a user is fixating on a single line of text. High jitter forces the Snapper to flicker between adjacent lines, causing visual fatigue.
* **Metric**:
  $$\text{SJR} = \frac{\text{Number of vertical line swaps}}{\text{Total fixation time on line (sec)}}$$
* **Evaluation Target**: $\text{SJR} < 0.1\,\text{Hz}$ (no more than one vertical flip every 10 seconds during stable reading).

### 2.2 Line Snapping Stability (LSS)
* **What it measures**: How well the system keeps the snapped gaze point on the current line of text, even in the presence of natural vertical drifting (up to 45px).
* **Methodology**: Users read a 500-word text with high vertical drift injected manually into the raw gaze coordinates. We measure the proportion of gazes that remain snapped to the correct line boundary.
* **Target**: $\ge 95\%$ alignment accuracy under $\pm 35\,\text{px}$ drift.

### 2.3 Dynamic Regression Capture (DRC)
* **What it measures**: When a reader performs a regression (looks back to a previous word), the horizontal saccade is rapid. This metric evaluates whether the snapping algorithm (M4) lags behind or snaps to the wrong adjacent line during a regression.
* **Metric**:
  $$\text{DRC} = \frac{\text{Correctly aligned regression destinations}}{\text{Total regressions detected}}$$
* **Target**: $\ge 85\%$ accuracy on backward saccades exceeding 3 words.

---

## 3. Pillar 2: System Latency & Resource Profiling

A real-time system must process webcam input concurrently with language processing without dropping frames or stalling the CPU.

### 3.1 End-to-End Latency Budget
To maintain an interactive experience, the latency budget from camera capture to UI highlighting must remain under **100 ms**.

```
[Camera Frame] ---> (M1: Face Preprocess: 15ms) ---> (M1: UniGaze ViT Inference: 25ms) 
               ---> (M4: Bayesian Snapping: 5ms)  ---> (M5: Fusion & UI Highlight: 5ms)
               ========================================================================
               Total Budget = 50ms (leaves 50ms buffer for rendering / network)
```

### 3.2 Frame Processing Rate (FPS)
* **Target**: Stable **$30\,\text{FPS}$** for webcam capture and UniGaze tracking on consumer laptops.
* **Evaluation**: Profiling FPS under active reading conditions while M3 (Cognitive pipeline) extracts features in the background.

### 3.3 Memory Footprint Stability
* **Target**: Leak-free operation during long reading sessions ($>30$ minutes).
* **Evaluation**: We evaluate whether the Viterbi path window buffer or the EM calibration accumulator causes a memory build-up. The window buffer must be a fixed size ($N = 20$ events) and discard older observations in $O(1)$ time.

---

## 4. Pillar 3: Human-in-the-Loop HCI Usability (User Study)

Ultimately, the platform is designed to assist humans. We propose an A/B user study framework to evaluate the real-world utility of the integrated platform.

### 4.1 A/B Reading Efficiency Test
* **Setup**: 20 participants are split into two groups reading the same technical papers:
  * **Group A (Control)**: Reading with standard PDF layout (no snapping or RDS feedback).
  * **Group B (LexiGaze)**: Reading with live snaps, dynamic RDS coloring (highlighting high-difficulty regions), and adaptive scrolling.
* **Evaluated Metrics**:
  1. **Words Per Minute (WPM)**: Measure if dynamic highlights speed up reading or cause distraction.
  2. **Comprehension Quiz**: A 10-question quiz checking text understanding and retention.
  3. **Regression Saccade Rate**: Do users repeat-read less because the platform clarified difficult words on the first pass?

### 4.2 NASA-TLX Workload Assessment
After the reading session, participants fill out a NASA-TLX survey measuring:
1. **Mental Demand**: How much mental and perceptual activity was required?
2. **Temporal Demand**: Did they feel rushed or hurried?
3. **Frustration Level**: Did the snapping jitter or calibration errors cause annoyance?

---

## 5. Summary Matrix for Modules

| Module | What to Evaluate (Real-Time) | Evaluation Method | Success Criteria |
| :--- | :--- | :--- | :--- |
| **M1: Gaze Perception** | Jitter & Tracking FPS | Chrome Performance DevTools | $\ge 30\,\text{FPS}$, Jitter $\le 10\,\text{px}$ |
| **M3: Cognitive Load** | Token Surprisal Latency | Python Backend `cProfile` | $\le 15\,\text{ms}$ per sentence |
| **M4: Snapping** | Alignment Snapping Accuracy | High-Speed Camera Video Sync | SJR $\le 0.1\,\text{Hz}$ |
| **M5: Fusion** | RDS Update Responsiveness | API Endpoint `time()` Logger | $\le 10\,\text{ms}$ execution latency |
| **End-to-End** | Comprehension & Usability | User A/B testing & NASA-TLX | Higher quiz scores, SUS score $> 75$ |
