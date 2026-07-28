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

    def calibrate_and_decode(self, raw_gaze_sequence, word_boxes, base_cm, transition_matrix, sigma_gaze=[40, 30], use_ovp=True, is_L2=False, alpha_cm=None):
        # Step 1: E-Step (Expectation) with Skill 14 Multi-Hypothesis
        window = raw_gaze_sequence[:self.window_size]
        
        best_initial_indices = None
        best_likelihood = -np.inf
        best_h = 0
        
        for h in self.hypotheses:
            # Temporarily shift window by hypothesis h (vertical)
            hyp_window = window.copy()
            hyp_window[:, 1] += h
            
            indices, likelihood = viterbi_gaze_decode(hyp_window, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp, is_L2=is_L2, alpha_cm=alpha_cm)
            
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_initial_indices = indices
                best_h = h
        
        # Step 2: M-Step (Maximization / Drift Estimation)
        if use_ovp:
            dfield = DynamicCognitiveField(word_boxes, base_cm, use_ovp=True, is_L2=is_L2, alpha_cm=alpha_cm)
            word_centers = dfield.word_centers
        else:
            word_centers = np.array([[ (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ] for box in word_boxes])
            
        predicted_centers = word_centers[best_initial_indices]
        errors = window - predicted_centers
        
        # Robust Mean Drift (Median)
        drift_x = np.nanmedian(errors[:, 0])
        drift_y = np.nanmedian(errors[:, 1])
        
        if np.nanstd(errors) > 150:
            drift_x, drift_y = 0, 0
            
        # Step 3: Update & Final Decode
        corrected_gaze = raw_gaze_sequence - np.array([drift_x, drift_y])
        final_indices, _ = viterbi_gaze_decode(corrected_gaze, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp, is_L2=is_L2, alpha_cm=alpha_cm)
        
        return final_indices, (drift_x, drift_y)


class MultiLineAdaptiveEMDecoder:
    """
    Multi-Line Adaptive EM Anchoring Decoder.
    Partitions paragraph word bounding boxes into vertical line clusters and fits
    line-specific vertical drift offsets with a spatial smoothness prior across adjacent lines.
    Eliminates multi-line vertical drift and line-jumping decoding errors.
    """
    def __init__(self, hypotheses=[0, 30, -30, 60, -60], smoothness_lambda=0.5, max_em_iters=3):
        self.hypotheses = hypotheses
        self.smoothness_lambda = smoothness_lambda
        self.max_em_iters = max_em_iters

    @staticmethod
    def cluster_words_into_lines(word_boxes: np.ndarray, line_threshold: float = 15.0) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray]:
        """
        Cluster word bounding boxes into discrete line indices based on vertical midpoint.
        Returns:
          - word_line_indices: Array of line index per word box (N,)
          - line_to_words: Dict mapping line_idx -> list of word indices
          - line_y_centers: Line baseline vertical centers (K,)
        """
        N = len(word_boxes)
        y_centers = (word_boxes[:, 1] + word_boxes[:, 3]) / 2.0
        
        sorted_indices = np.argsort(y_centers)
        line_to_words = {}
        word_line_indices = np.zeros(N, dtype=int)
        line_y_list = []

        current_line_idx = 0
        current_line_words = [sorted_indices[0]]
        current_y_sum = y_centers[sorted_indices[0]]

        for idx in sorted_indices[1:]:
            y_val = y_centers[idx]
            avg_y = current_y_sum / len(current_line_words)
            if abs(y_val - avg_y) <= line_threshold:
                current_line_words.append(idx)
                current_y_sum += y_val
            else:
                line_to_words[current_line_idx] = current_line_words
                line_y_list.append(avg_y)
                current_line_idx += 1
                current_line_words = [idx]
                current_y_sum = y_val

        line_to_words[current_line_idx] = current_line_words
        line_y_list.append(current_y_sum / len(current_line_words))

        for l_idx, w_list in line_to_words.items():
            for w in w_list:
                word_line_indices[w] = l_idx

        return word_line_indices, line_to_words, np.array(line_y_list)

    def calibrate_and_decode(self, raw_gaze_sequence: np.ndarray, word_boxes: np.ndarray, base_cm, transition_matrix, sigma_gaze=[40, 30], use_ovp=True, is_L2=False, alpha_cm=None):
        # 1. Cluster words into paragraph lines
        word_line_indices, line_to_words, line_y_centers = self.cluster_words_into_lines(word_boxes)
        num_lines = len(line_y_centers)

        # Target word centers (OVP or bounding box centers)
        if use_ovp:
            dfield = DynamicCognitiveField(word_boxes, base_cm, use_ovp=True, is_L2=is_L2, alpha_cm=alpha_cm)
            word_centers = dfield.word_centers
        else:
            word_centers = np.array([[ (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0 ] for box in word_boxes])

        # 2. Multi-Hypothesis initial E-step to find optimal global starting hypothesis
        best_initial_indices = None
        best_likelihood = -np.inf

        for h in self.hypotheses:
            hyp_gaze = raw_gaze_sequence.copy()
            hyp_gaze[:, 1] += h
            indices, likelihood = viterbi_gaze_decode(
                hyp_gaze, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp, is_L2=is_L2, alpha_cm=alpha_cm
            )
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_initial_indices = indices

        curr_indices = best_initial_indices
        line_drift_y = np.zeros(num_lines)
        global_drift_x = 0.0

        # 3. Iterative Multi-Line EM Loop
        for iter_idx in range(self.max_em_iters):
            # M-Step: Compute raw vertical drift per line cluster
            pred_word_centers = word_centers[curr_indices]
            gaze_y = raw_gaze_sequence[:, 1]
            gaze_x = raw_gaze_sequence[:, 0]

            global_drift_x = np.nanmedian(gaze_x - pred_word_centers[:, 0])

            raw_line_drifts = np.zeros(num_lines)
            counts = np.zeros(num_lines)

            for t_idx, w_idx in enumerate(curr_indices):
                l_idx = word_line_indices[w_idx]
                dy = gaze_y[t_idx] - pred_word_centers[t_idx, 1]
                raw_line_drifts[l_idx] += dy
                counts[l_idx] += 1.0

            for l_idx in range(num_lines):
                if counts[l_idx] > 0:
                    raw_line_drifts[l_idx] /= counts[l_idx]
                else:
                    # Fallback to nearest neighbor or median
                    raw_line_drifts[l_idx] = np.nanmedian(gaze_y - pred_word_centers[:, 1])

            # Apply Spatial Smoothness Prior across lines:
            # Min sum_k (line_drift_k - raw_k)^2 + lambda * sum_{k} (line_drift_{k+1} - line_drift_k)^2
            smoothed_line_drifts = raw_line_drifts.copy()
            if num_lines > 1:
                for _ in range(5):  # Smoothness relaxation
                    for k in range(num_lines):
                        neighbors = []
                        if k > 0:
                            neighbors.append(smoothed_line_drifts[k - 1])
                        if k < num_lines - 1:
                            neighbors.append(smoothed_line_drifts[k + 1])
                        if neighbors:
                            smoothed_line_drifts[k] = (1.0 - self.smoothness_lambda) * raw_line_drifts[k] + self.smoothness_lambda * np.mean(neighbors)

            line_drift_y = smoothed_line_drifts

            # E-Step: Correct gaze points line-by-line and re-decode
            corrected_gaze = np.asarray(raw_gaze_sequence, dtype=np.float64).copy()
            corrected_gaze[:, 0] -= global_drift_x

            # Apply continuous vertical correction function based on gaze y-position
            for t_idx in range(len(corrected_gaze)):
                # Interpolate line vertical drift based on raw y position
                y_raw = raw_gaze_sequence[t_idx, 1]
                # Nearest line drift
                nearest_l = np.argmin(np.abs(line_y_centers - y_raw))
                corrected_gaze[t_idx, 1] -= line_drift_y[nearest_l]

            new_indices, new_likelihood = viterbi_gaze_decode(
                corrected_gaze, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp, is_L2=is_L2, alpha_cm=alpha_cm
            )

            if new_indices == curr_indices:
                break
            curr_indices = new_indices

        return curr_indices, (global_drift_x, line_drift_y)

