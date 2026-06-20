# Domain Knowledge & Scientific References

## 1. Neural Perception: Eye-Tracking Fundamentals

### Fixation vs. Saccade
- **Fixation**: The period when the eyes are relatively still, lingering on a specific point (typically 200–500ms). This is when the brain processes visual information.
- **Saccade**: Rapid, ballistic eye movements between fixations. During a saccade, visual sensitivity is significantly reduced (saccadic suppression).
- **Smooth Pursuit**: Slow eye movements used to track a moving object (less common in static text reading).

### Edge AI Challenges
- **Jitter**: High-frequency noise caused by low-light conditions or sensor limitations in webcams.
- **Systematic Drift**: Low-frequency errors often caused by head movement, glasses, or improper calibration, leading the gaze trajectory to shift away from the true target.

## 2. Symbolic Cognition: Linguistic Priors

### Surprisal Theory (Information Theory)
- Based on the principle that the "information content" of a word is inversely proportional to its probability: $I(w) = -\ln P(w | \text{context})$.
- **Cognitive Load**: Harder-to-predict words (high surprisal) require more neural processing time, leading to longer **Fixation Durations**.

### BERT as a Cognitive Proxy
- **Pseudo-Log-Likelihood (PLL)**: We use Masked Language Models (MLM) like BERT to calculate the probability of a word within its sentence.
- **Symbolic Alpha**: We normalize surprisal scores into a weight ($\alpha$) that represents the "gravitational pull" a word exerts on the noisy gaze data.

## 3. Neuro-Symbolic Fusion: Calibration Filters

### Traditional Filters (Neural Smoothing)
- **Moving Average**: A simple temporal filter that averages the last $N$ coordinates to reduce high-frequency jitter.
- **Kalman Filter**: A more advanced recursive filter that estimates the state of a dynamic system from a series of noisy measurements.

### Neuro-Symbolic "Gravity Snap"
- **The Concept**: Instead of purely temporal smoothing, we use linguistic "attractors." 
- **The Logic**: If a noisy gaze point falls within a certain radius of a high-surprisal word, the system "snaps" the coordinate toward that word, weighted by its cognitive importance ($\alpha$).
- **Advantage**: Corrects systematic drift that traditional filters (which have no knowledge of the text content) cannot detect.
