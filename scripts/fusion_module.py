"""
fusion_module.py - LexiGaze Multimodal Fusion Algorithms
=========================================================
This module implements six different algorithms for fusing eye-tracking gaze data
(perception) with cognitive load metrics (cognition) into a unified
Reading Difficulty Score (RDS).
"""

import numpy as np
import pandas as pd

def normalize(array):
    """Min-max normalize array to [0, 1]."""
    arr = np.array(array, dtype=float)
    amin, amax = np.min(arr), np.max(arr)
    if amax - amin == 0:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)

class LexiGazeFusion:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def fuse_linear(self, gaze_dwell, gaze_fix, load_score, w1=0.35, w2=0.25, w3=0.40):
        """
        1. Weighted Linear Fusion (Baseline)
        RDS = w1 * gaze_dwell + w2 * gaze_fix + w3 * load_score
        """
        g_dwell = normalize(gaze_dwell)
        g_fix = normalize(gaze_fix)
        l_score = normalize(load_score)
        
        rds = w1 * g_dwell + w2 * g_fix + w3 * l_score
        return normalize(rds)

    def fuse_multiplicative(self, gaze_dwell, gaze_fix, load_score, w_dwell=0.6, w_fix=0.4):
        """
        2. Multiplicative / Interaction-based Fusion
        RDS = load_score * (w_dwell * gaze_dwell + w_fix * gaze_fix)
        Emphasizes words that have both high gaze attention and high cognitive difficulty.
        """
        g_dwell = normalize(gaze_dwell)
        g_fix = normalize(gaze_fix)
        l_score = normalize(load_score)
        
        gaze_attn = w_dwell * g_dwell + w_fix * g_fix
        rds = l_score * gaze_attn
        return normalize(rds)

    def fuse_gated(self, gaze_dwell, gaze_fix, load_score, threshold=0.2, alpha=0.1):
        """
        3. Attention-Gated Cognitive Fusion
        Only applies cognitive load if the gaze dwell is above a threshold.
        Otherwise, load_score is scaled down by alpha.
        """
        g_dwell = normalize(gaze_dwell)
        l_score = normalize(load_score)
        
        rds = np.zeros_like(g_dwell)
        for i in range(len(g_dwell)):
            if g_dwell[i] >= threshold:
                rds[i] = l_score[i]
            else:
                rds[i] = alpha * l_score[i]
        return normalize(rds)

    def fuse_sigmoid(self, gaze_dwell, gaze_fix, load_score, w1=0.35, w2=0.25, w3=0.40, k=10, x0=0.5):
        """
        4. Non-linear Sigmoidal Fusion
        Applies a logistic activation function over the linear combination.
        """
        g_dwell = normalize(gaze_dwell)
        g_fix = normalize(gaze_fix)
        l_score = normalize(load_score)
        
        linear_rds = w1 * g_dwell + w2 * g_fix + w3 * l_score
        rds = 1.0 / (1.0 + np.exp(-k * (linear_rds - x0)))
        return normalize(rds)

    def fuse_bayesian(self, gaze_dwell, load_score):
        """
        5. Bayesian Posterior Fusion
        Treats load_score as the prior P(Diff) and gaze_dwell as the likelihood P(Gaze|Diff).
        RDS = P(Diff|Gaze) = (P(Gaze|Diff) * P(Diff)) / P(Gaze)
        """
        g_dwell = normalize(gaze_dwell)
        l_score = normalize(load_score)
        
        # Bayesian update formula
        numerator = l_score * g_dwell
        denominator = numerator + (1.0 - l_score) * (1.0 - g_dwell) + self.epsilon
        rds = numerator / denominator
        return normalize(rds)

    def fuse_rrf(self, gaze_dwell, load_score, k=60):
        """
        6. Reciprocal Rank Fusion (RRF)
        Fuses rankings of words from gaze dwell and cognitive load.
        """
        df = pd.DataFrame({
            'dwell': gaze_dwell,
            'load': load_score
        })
        
        # Calculate descending ranks (1-based)
        rank_g = df['dwell'].rank(ascending=False, method='min')
        rank_l = df['load'].rank(ascending=False, method='min')
        
        rrf_score = 1.0 / (rank_g + k) + 1.0 / (rank_l + k)
        return normalize(rrf_score)
