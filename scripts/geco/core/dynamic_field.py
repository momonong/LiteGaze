import numpy as np
from scipy.stats import multivariate_normal

class DynamicCognitiveField:
    """
    Skill 1: Dynamic Cognitive Field Generator.
    Implements a time-decaying cognitive mass model based on cumulative exposure.
    """
    def __init__(self, word_boxes, base_cm, lambda_decay=0.5, sigma=30.0, use_ovp=True, is_L2=False):
        """
        word_boxes: List of [x_min, y_min, x_max, y_max].
        base_cm: NumPy array of pre-computed static cognitive mass.
        lambda_decay: Decay rate for exposure.
        sigma: Default standard deviation for the spatial Gaussian.
        use_ovp: If True, uses Optimal Viewing Position (35% of width).
        is_L2: If True, applies full cognitive mass modulation. If False (L1), flattens cognitive prior.
        """
        self.word_boxes = np.array(word_boxes)
        base_cm_arr = np.array(base_cm)
        if not is_L2:
            # Flatten cognitive mass for native readers to reduce false semantic bias
            mean_cm = np.mean(base_cm_arr)
            self.base_cm = 0.9 * mean_cm + 0.1 * base_cm_arr
        else:
            self.base_cm = base_cm_arr
        self.lambda_decay = lambda_decay
        self.use_ovp = use_ovp
        
        # Calculate centers or OVP
        if use_ovp:
            # Skill 7: OVP is at ~35-40% of the word width from left
            widths = self.word_boxes[:, 2] - self.word_boxes[:, 0]
            ovp_x = self.word_boxes[:, 0] + widths * 0.35
            center_y = (self.word_boxes[:, 1] + self.word_boxes[:, 3]) / 2
            self.word_centers = np.stack([ovp_x, center_y], axis=1)
            
            # Skill 7: Length-based variance (longer words = wider horizontal tolerance)
            # Base sigma is 30, scaled by width ratio (relative to average word width ~100px)
            self.sigma_x = sigma * (1.0 + (widths / 200.0))
            self.sigma_y = np.full(len(base_cm), sigma)
        else:
            self.word_centers = np.array([[ (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ] for box in word_boxes])
            self.sigma_x = np.full(len(base_cm), sigma)
            self.sigma_y = np.full(len(base_cm), sigma)
        
        self.num_words = len(base_cm)
        self.exposure = np.zeros(self.num_words)
        self.current_cm = np.copy(self.base_cm)

    def update(self, gaze_x, gaze_y):
        """
        Updates exposure based on current gaze and recalculates dynamic CM.
        """
        gaze_pos = np.array([gaze_x, gaze_y])
        
        # Calculate spatial Gaussian update for exposure with per-word sigma
        # Using vectorized approach for multiple words
        diff = self.word_centers - gaze_pos
        # N(x_t | mu_i, Sigma_i)
        # exp(-0.5 * ( (dx^2 / sx^2) + (dy^2 / sy^2) ))
        exponent = -0.5 * ( (diff[:, 0]**2 / self.sigma_x**2) + (diff[:, 1]**2 / self.sigma_y**2) )
        update_inc = np.exp(exponent)
        
        # Update cumulative exposure
        self.exposure += update_inc
        
        # CM_i(t) = (Base_CM_i + epsilon) * exp(-lambda * E_i(t))
        # Skill 1: Adding epsilon to prevent zero-probability deadlocks
        decay_factor = np.exp(-self.lambda_decay * self.exposure)
        self.current_cm = (self.base_cm + 0.01) * decay_factor
        
        # Normalize across the sentence (softmax-like or simple sum normalization)
        if np.sum(self.current_cm) > 0:
            self.current_cm /= np.sum(self.current_cm)
        else:
            # Fallback if all CMs become zero (unlikely)
            self.current_cm = np.ones(self.num_words) / self.num_words
            
        return self.current_cm

    def get_cm(self):
        return self.current_cm
