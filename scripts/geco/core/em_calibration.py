import numpy as np
from .viterbi_decoder import viterbi_gaze_decode
from .dynamic_field import DynamicCognitiveField

class AutoCalibratingDecoder:
    """
    Skill 6 & 14: EM-based Dynamic Drift Auto-Calibration with Multi-Hypothesis Initialization.
    Iteratively estimates systematic drift and re-decodes the sequence.
    """
    def __init__(self, calibration_window_size=40, hypotheses=[0, 40, -40]):
        self.window_size = calibration_window_size
        self.hypotheses = hypotheses # Skill 14: Vertical shift hypotheses to fix line-locking

    def calibrate_and_decode(self, raw_gaze_sequence, word_boxes, base_cm, transition_matrix, sigma_gaze=[40, 30], use_ovp=True, is_L2=False):
        # Step 1: E-Step (Expectation) with Skill 14 Multi-Hypothesis
        window = raw_gaze_sequence[:self.window_size]
        
        best_initial_indices = None
        best_likelihood = -np.inf
        best_h = 0
        
        # print(f"Skill 14: Evaluating {len(self.hypotheses)} drift hypotheses...")
        
        for h in self.hypotheses:
            # Temporarily shift window by hypothesis h (vertical)
            hyp_window = window.copy()
            hyp_window[:, 1] += h
            
            indices, likelihood = viterbi_gaze_decode(hyp_window, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp, is_L2=is_L2)
            
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_initial_indices = indices
                best_h = h
        
        # print(f"Skill 14: Best hypothesis h={best_h}px (Likelihood: {best_likelihood:.2f})")
        
        # Step 2: M-Step (Maximization / Drift Estimation)
        # Access biologically aligned OVP centers for drift estimation
        if use_ovp:
            dfield = DynamicCognitiveField(word_boxes, base_cm, use_ovp=True, is_L2=is_L2)
            word_centers = dfield.word_centers # OVP Centers (35% width)
        else:
            word_centers = np.array([[ (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ] for box in word_boxes])
            
        predicted_centers = word_centers[best_initial_indices]
        
        # Calculate error vector (Corrected by hypothesis h)
        # Error = (RawGaze + h) - PredictedWordCenter
        # Drift = Error - h = RawGaze - PredictedWordCenter
        errors = window - predicted_centers
        
        # Robust Mean Drift (Median)
        drift_x = np.nanmedian(errors[:, 0])
        drift_y = np.nanmedian(errors[:, 1])
        
        # Handle case where user isn't reading (high variance)
        if np.nanstd(errors) > 150:
            print("⚠️ High variance detected in initial window. Skipping auto-calibration.")
            drift_x, drift_y = 0, 0
            
        # print(f"Auto-Calibration: Detected Drift ({drift_x:.1f}px, {drift_y:.1f}px)")
        
        # Step 3: Update & Final Decode
        corrected_gaze = raw_gaze_sequence - np.array([drift_x, drift_y])
        final_indices, _ = viterbi_gaze_decode(corrected_gaze, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp, is_L2=is_L2)
        
        return final_indices, (drift_x, drift_y)
