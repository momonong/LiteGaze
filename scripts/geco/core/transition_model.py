import numpy as np

class ReadingTransitionMatrix:
    """
    Skill 2: Oculomotor Transition Matrix Builder.
    Builds a transition probability matrix P(w_j | w_i) for a Hidden Markov Model.
    """
    def __init__(self, momentum_forward=1.0, skip_weight=0.5, regression_weight=0.1):
        self.momentum_forward = momentum_forward
        self.skip_weight = skip_weight
        self.regression_weight = regression_weight

    def build_matrix(self, base_cm_array, is_L2_reader=False):
        """
        Constructs an N x N transition matrix.
        base_cm_array: Array of static cognitive mass (higher means more difficult).
        is_L2_reader: If True, increases regression probability.
        """
        n = len(base_cm_array)
        t_matrix = np.zeros((n, n))
        
        reg_w = self.regression_weight * (2.0 if is_L2_reader else 1.0)
        
        for i in range(n):
            # 1. Saccadic Momentum (Forward)
            if i + 1 < n:
                t_matrix[i, i + 1] = 1.0 * self.momentum_forward
            if i + 2 < n:
                # Word Skipping logic: higher prob if i+1 is easy (low CM)
                skip_boost = 1.0 - base_cm_array[i + 1]
                t_matrix[i, i + 2] = 0.5 * self.momentum_forward * (1.0 + self.skip_weight * skip_boost)
            
            # 2. Stay (Refixation)
            # Higher base_cm (difficult word) means higher stay probability
            t_matrix[i, i] = 0.3 * base_cm_array[i]
            
            # 3. Regressions (Backwards)
            for j in range(i):
                # Regression probability decays with distance
                dist_penalty = np.exp(-(i - j))
                # Boost regression if current word is hard
                t_matrix[i, j] = reg_w * base_cm_array[i] * dist_penalty

        # Apply softmax to each row to ensure they sum to 1.0
        row_sums = t_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        t_matrix /= row_sums
        
        return t_matrix

class PsycholinguisticTransitionMatrix:
    """
    Skill 10: Psycholinguistic-Oculomotor Transition Matrix (POM).
    Purely mathematically driven model inspired by SWIFT and E-Z Reader.
    Removes LLM attention to focus on causal physical momentum and cognitive modulation.
    """
    def __init__(self, sigma_fwd=1.0, sigma_reg=1.5, gamma=0.5):
        """
        sigma_fwd: Spread of forward saccades (centered at i+1).
        sigma_reg: Spread of regressions (centered at i-1).
        gamma: Scaling factor for CM-based skipping penalty.
        """
        self.sigma_fwd = sigma_fwd
        self.sigma_reg = sigma_reg
        self.gamma = gamma

    def build_matrix(self, num_words, base_cm_array, is_L2=False, word_boxes=None):
        """
        Builds the N x N transition matrix using POM logic.
        num_words: Number of words in sentence.
        base_cm_array: Array of Cognitive Mass (Surprisal/Difficulty).
        is_L2: If True, applies full cognitive mass modulation. If False (L1), scales down cognitive influence.
        word_boxes: Optional word bounding boxes [x_min, y_min, x_max, y_max] to enforce layout constraints.
        """
        n = num_words
        t_matrix = np.zeros((n, n))
        
        # Skill 10 Fix: Ensure CM is normalized to [0, 1] for the formula to work
        cm_min = base_cm_array.min()
        cm_max = base_cm_array.max()
        if cm_max > cm_min:
            norm_cm = (base_cm_array - cm_min) / (cm_max - cm_min)
        else:
            # Neutral baseline (0.5) if CM is uniform, preventing bias towards/against regressions
            norm_cm = np.full_like(base_cm_array, 0.5)
        
        # Adaptive scaling based on L1/L2 reader profile
        effective_gamma = self.gamma if is_L2 else (self.gamma * 0.1) # L1 readers are less restricted by difficulty
        
        for i in range(n):
            for j in range(n):
                if j > i:
                    # 1. Base Physical Saccade (Forward Momentum)
                    # Gaussian centered at i+1
                    p_fwd = np.exp(-((j - (i + 1))**2) / (2 * self.sigma_fwd**2))
                    # 2. Cognitive Modulation (Skipping)
                    # Penalize transition to j if j is cognitively heavy (high CM)
                    t_matrix[i, j] = p_fwd * max(0.1, 1.0 - effective_gamma * norm_cm[j])
                else:
                    # 3. Cognitive Modulation (Regressions & Stays)
                    # Exponential decay from i-1
                    p_reg = np.exp(-abs(j - (i - 1)) / self.sigma_reg)
                    
                    if is_L2:
                        reg_mod = norm_cm[i] + 0.1
                    else:
                        # For L1 (native), regressions are mostly oculomotor/random drift-based rather than difficulty-driven
                        reg_mod = 0.5
                    
                    t_matrix[i, j] = p_reg * reg_mod
            
            # Enforce Oculomotor Spatio-Temporal Monotonicity Constraints (OSTMC)
            # People read text left-to-right, top-to-bottom. We heavily penalize:
            # - Transitions to previous lines (y_j < y_i - 15px)
            # - Transitions skipping a line downwards (y_j > y_i + 45px)
            if word_boxes is not None and len(word_boxes) == n:
                word_y = np.array([(box[1] + box[3]) / 2.0 for box in word_boxes])
                for j in range(n):
                    if word_y[j] < word_y[i] - 15.0:  # Upward reading violation
                        t_matrix[i, j] *= 1e-4
                    elif word_y[j] > word_y[i] + 45.0: # Downward line-skipping violation
                        t_matrix[i, j] *= 1e-4

            # Row-wise normalization (sum to 1.0)
            row_sum = t_matrix[i].sum()
            if row_sum > 0:
                t_matrix[i] /= row_sum
                
        return t_matrix
