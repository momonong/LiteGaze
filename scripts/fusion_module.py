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

    def fuse_spillover_bayesian(self, gaze_dwell, load_score):
        """
        7. Spillover-Aware Bayesian Fusion
        Applies a Gaussian smoothing filter over sequential gaze dwell times
        to simulate oculomotor spillover, and then applies Bayesian fusion.
        """
        g_dwell = np.array(gaze_dwell, dtype=float)
        # Apply Gaussian filter for spillover (rolling window of size 3)
        smoothed = np.copy(g_dwell)
        n = len(g_dwell)
        for i in range(n):
            left = g_dwell[i-1] if i > 0 else g_dwell[i]
            right = g_dwell[i+1] if i < n-1 else g_dwell[i]
            smoothed[i] = 0.25 * left + 0.5 * g_dwell[i] + 0.25 * right
            
        g_norm = normalize(smoothed)
        l_score = normalize(load_score)
        
        numerator = l_score * g_norm
        denominator = numerator + (1.0 - l_score) * (1.0 - g_norm) + self.epsilon
        rds = numerator / denominator
        return normalize(rds)

    def fuse_parafoveal(self, gaze_dwell, load_score, threshold=0.5, alpha=0.3):
        """
        8. Parafoveal Skip-Corrected Fusion
        If a word has high cognitive load but zero gaze dwell (skipped),
        interpolates a small fraction of the neighboring words' dwell times
        to account for parafoveal processing or visual misalignment.
        """
        g_dwell = np.array(gaze_dwell, dtype=float)
        l_score = normalize(load_score)
        n = len(g_dwell)
        
        corrected_gaze = np.copy(g_dwell)
        for i in range(n):
            if g_dwell[i] == 0 and l_score[i] >= threshold:
                # Interpolate from neighbors
                neighbors = []
                if i > 0 and g_dwell[i-1] > 0:
                    neighbors.append(g_dwell[i-1])
                if i < n-1 and g_dwell[i+1] > 0:
                    neighbors.append(g_dwell[i+1])
                if neighbors:
                    corrected_gaze[i] = alpha * np.mean(neighbors)
                    
        g_norm = normalize(corrected_gaze)
        
        numerator = l_score * g_norm
        denominator = numerator + (1.0 - l_score) * (1.0 - g_norm) + self.epsilon
        rds = numerator / denominator
        return normalize(rds)

    def fuse_spillover_rrf(self, gaze_dwell, load_score, k=60):
        """
        9. Spillover-Corrected Reciprocal Rank Fusion (Spillover-RRF)
        Smooths sequential gaze dwell times using a rolling Gaussian kernel
        before computing rank and fusing.
        """
        g_dwell = np.array(gaze_dwell, dtype=float)
        # Apply Gaussian filter for spillover (rolling window of size 3)
        smoothed = np.copy(g_dwell)
        n = len(g_dwell)
        for i in range(n):
            left = g_dwell[i-1] if i > 0 else g_dwell[i]
            right = g_dwell[i+1] if i < n-1 else g_dwell[i]
            smoothed[i] = 0.25 * left + 0.5 * g_dwell[i] + 0.25 * right
            
        df = pd.DataFrame({
            'dwell': smoothed,
            'load': load_score
        })
        
        # Calculate descending ranks (1-based)
        rank_g = df['dwell'].rank(ascending=False, method='min')
        rank_l = df['load'].rank(ascending=False, method='min')
        
        rrf_score = 1.0 / (rank_g + k) + 1.0 / (rank_l + k)
        return normalize(rrf_score)

    def fuse_parafoveal_rrf(self, gaze_dwell, load_score, threshold=0.5, alpha=0.3, k=60):
        """
        10. Parafoveal-Corrected Reciprocal Rank Fusion
        Interpolates skipped gaze dwell times for difficult words before ranking and fusing.
        """
        g_dwell = np.array(gaze_dwell, dtype=float)
        l_score = normalize(load_score)
        n = len(g_dwell)
        
        corrected_gaze = np.copy(g_dwell)
        for i in range(n):
            if g_dwell[i] == 0 and l_score[i] >= threshold:
                # Interpolate from neighbors
                neighbors = []
                if i > 0 and g_dwell[i-1] > 0:
                    neighbors.append(g_dwell[i-1])
                if i < n-1 and g_dwell[i+1] > 0:
                    neighbors.append(g_dwell[i+1])
                if neighbors:
                    corrected_gaze[i] = alpha * np.mean(neighbors)
                    
        df = pd.DataFrame({
            'dwell': corrected_gaze,
            'load': load_score
        })
        
        # Calculate descending ranks (1-based)
        rank_g = df['dwell'].rank(ascending=False, method='min')
        rank_l = df['load'].rank(ascending=False, method='min')
        
        rrf_score = 1.0 / (rank_g + k) + 1.0 / (rank_l + k)
        return normalize(rrf_score)

    def fuse_spillover_parafoveal_rrf(self, gaze_dwell, load_score, threshold=0.5, alpha=0.3, k=60):
        """
        11. Spillover-Parafoveal Reciprocal Rank Fusion (SP-RRF)
        Combines parafoveal skip interpolation and spillover smoothing
        before applying Reciprocal Rank Fusion.
        """
        g_dwell = np.array(gaze_dwell, dtype=float)
        l_score = normalize(load_score)
        n = len(g_dwell)
        
        # Phase 1: Parafoveal skip correction
        corrected = np.copy(g_dwell)
        for i in range(n):
            if g_dwell[i] == 0 and l_score[i] >= threshold:
                neighbors = []
                if i > 0 and g_dwell[i-1] > 0:
                    neighbors.append(g_dwell[i-1])
                if i < n-1 and g_dwell[i+1] > 0:
                    neighbors.append(g_dwell[i+1])
                if neighbors:
                    corrected[i] = alpha * np.mean(neighbors)
                    
        # Phase 2: Spillover smoothing
        smoothed = np.copy(corrected)
        for i in range(n):
            left = corrected[i-1] if i > 0 else corrected[i]
            right = corrected[i+1] if i < n-1 else corrected[i]
            smoothed[i] = 0.25 * left + 0.5 * corrected[i] + 0.25 * right
            
        df = pd.DataFrame({
            'dwell': smoothed,
            'load': load_score
        })
        
        rank_g = df['dwell'].rank(ascending=False, method='min')
        rank_l = df['load'].rank(ascending=False, method='min')
        
        rrf_score = 1.0 / (rank_g + k) + 1.0 / (rank_l + k)
        return normalize(rrf_score)

    def fuse_cross_attention(self, gaze_dwell, gaze_fix, load_score):
        """
        12. Experimental untrained neural projection (research archive only).

        The weights below are initialized from a fixed random seed and have no
        trained checkpoint. Production orchestration rejects this method; this
        implementation remains only so historical exploratory outputs can be
        reproduced and clearly labelled.
        """
        import torch
        from torch import nn
        
        g_dwell = normalize(gaze_dwell)
        g_fix = normalize(gaze_fix)
        l_score = normalize(load_score)
        
        n_samples = len(g_dwell)
        if n_samples == 0:
            return np.zeros(0)
            
        # Structure gaze features: [dwell, fixation, normalized_index]
        gaze_feat = torch.tensor([
            [g_dwell[i], g_fix[i], i / float(n_samples)] 
            for i in range(n_samples)
        ], dtype=torch.float32)
        
        # Structure NLP features: [load_score, load_score * 0.7, 1.0 - load_score]
        nlp_feat = torch.tensor([
            [l_score[i], l_score[i] * 0.7, 1.0 - l_score[i]]
            for i in range(n_samples)
        ], dtype=torch.float32)
        
        class CrossAttentionFusionModel(nn.Module):
            def __init__(self, d_gaze, d_nlp, d_model):
                super().__init__()
                self.q_proj = nn.Linear(d_gaze, d_model)
                self.k_proj = nn.Linear(d_nlp, d_model)
                self.v_proj = nn.Linear(d_nlp, d_model)
                self.scale = np.sqrt(d_model)
                self.out_layer = nn.Sequential(
                    nn.Linear(d_model, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, g, n):
                Q = self.q_proj(g)
                K = self.k_proj(n)
                V = self.v_proj(n)
                attn = torch.sum(Q * K, dim=-1, keepdim=True) / self.scale
                attn_w = torch.sigmoid(attn)
                fused = attn_w * V
                return self.out_layer(fused)
                
        # Initialize model with a fixed seed for reproducibility
        torch.manual_seed(42)
        model = CrossAttentionFusionModel(d_gaze=3, d_nlp=3, d_model=8)
        
        # Perform quick inference (no training gradient needed)
        with torch.no_grad():
            output = model(gaze_feat, nlp_feat).numpy().flatten()
            
        return normalize(output)

    def fuse_fatigue_adaptive(self, gaze_dwell, load_score):
        """
        13. Fatigue-Adaptive Gaze Weighting
        Dynamically scales down gaze confidence weighting as the chronological reading
        sequence progresses to filter out accumulated webcam visual jitter.
        """
        g_dwell = normalize(gaze_dwell)
        l_score = normalize(load_score)
        n = len(g_dwell)
        if n == 0:
            return np.zeros(0)
            
        rds = np.zeros(n)
        for i in range(n):
            # Trust gaze heavily at the start (alpha=0.8), fade to prior (alpha=0.15) at the end
            alpha = max(0.15, 0.8 - 0.015 * i)
            rds[i] = alpha * g_dwell[i] + (1.0 - alpha) * l_score[i]
            
        return normalize(rds)



