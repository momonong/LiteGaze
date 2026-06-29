import numpy as np
from scipy.stats import multivariate_normal
from .dynamic_field import DynamicCognitiveField
from .transition_model import ReadingTransitionMatrix

def viterbi_gaze_decode(gaze_sequence, word_boxes, base_cm, transition_matrix, sigma_gaze=[40, 30], lambda_decay=0.5, use_ovp=True, is_L2=False, alpha_cm=None):
    """
    Skill 3 & 7: Spatio-Temporal Viterbi Gaze Decoder with OVP.
    """
    num_words = len(word_boxes)
    num_steps = len(gaze_sequence)
    
    # Initialize DP table
    viterbi = np.full((num_words, num_steps), -np.inf)
    backpointer = np.zeros((num_words, num_steps), dtype=int)
    
    # Initialize Dynamic Cognitive Field (supports OVP)
    dynamic_field = DynamicCognitiveField(word_boxes, base_cm, lambda_decay=lambda_decay, use_ovp=use_ovp, is_L2=is_L2, alpha_cm=alpha_cm)
    word_centers = dynamic_field.word_centers
    sigma_x_words = dynamic_field.sigma_x
    sigma_y_words = dynamic_field.sigma_y
    
    # 1. Initialization Step
    current_gaze = gaze_sequence[0]
    cm_t0 = dynamic_field.update(current_gaze[0], current_gaze[1])
    
    for i in range(num_words):
        # Combined variance (sensor noise + word-specific tolerance)
        sx = np.sqrt(sigma_gaze[0]**2 + sigma_x_words[i]**2)
        sy = np.sqrt(sigma_gaze[1]**2 + sigma_y_words[i]**2)
        
        diff = current_gaze - word_centers[i]
        prob_spatial = np.exp(-0.5 * ( (diff[0]**2 / sx**2) + (diff[1]**2 / sy**2) ))
        emission_prob = prob_spatial * cm_t0[i]
        
        # Skill 3: Uniform start probability to allow data to drive initialization
        start_prob = 1.0 / num_words 
        
        if emission_prob > 0:
            viterbi[i, 0] = np.log(start_prob) + np.log(emission_prob)

    # 2. Recursion Step
    for t in range(1, num_steps):
        current_gaze = gaze_sequence[t]
        cm_t = dynamic_field.update(current_gaze[0], current_gaze[1])
        
        for j in range(num_words):
            start_i = max(0, j - 3)
            end_i = min(num_words, j + 4)
            
            # Emissions with OVP-adjusted variance
            sx = np.sqrt(sigma_gaze[0]**2 + sigma_x_words[j]**2)
            sy = np.sqrt(sigma_gaze[1]**2 + sigma_y_words[j]**2)
            diff = current_gaze - word_centers[j]
            prob_spatial = np.exp(-0.5 * ( (diff[0]**2 / sx**2) + (diff[1]**2 / sy**2) ))
            emission_prob = prob_spatial * cm_t[j]
            
            if emission_prob <= 0:
                continue
                
            log_b = np.log(emission_prob)
            
            # Vectorized transition search for speed
            prev_scores = viterbi[start_i:end_i, t-1]
            trans_probs = transition_matrix[start_i:end_i, j]
            
            # Filter out -inf and zero probs
            valid_mask = (prev_scores != -np.inf) & (trans_probs > 0)
            if not np.any(valid_mask):
                continue
                
            combined_scores = prev_scores[valid_mask] + np.log(trans_probs[valid_mask]) + log_b
            max_idx_local = np.argmax(combined_scores)
            
            viterbi[j, t] = combined_scores[max_idx_local]
            backpointer[j, t] = np.arange(start_i, end_i)[valid_mask][max_idx_local]

    # 3. Termination & Path Backtracking
    best_path = []
    if np.all(viterbi[:, -1] == -np.inf):
        # Fallback if no path found (unlikely with enough noise margin)
        # Returns a rough guess and a very low score
        guess = [np.argmax([multivariate_normal.pdf(g, mean=word_centers[np.argmin(np.sum((word_centers - g)**2, axis=1))], cov=np.diag([sigma_gaze[0]**2, sigma_gaze[1]**2])) for g in gaze_sequence])] * num_steps
        return guess, -np.inf

    last_state = np.argmax(viterbi[:, -1])
    final_score = viterbi[last_state, -1]
    best_path.append(last_state)
    
    for t in range(num_steps - 1, 0, -1):
        last_state = backpointer[last_state, t]
        best_path.insert(0, last_state)
        
    return best_path, final_score
