import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = "archive/data/geco/geco_pp01_cognitive_mass.csv"
OUTPUT_DIR = "output"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # Inject noise (Drift and Jitter)
    np.random.seed(42)
    SIGMA_X = 40.0
    SIGMA_Y = 30.0
    DRIFT_Y = 45.0
    
    df['noisy_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, len(df))
    df['noisy_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    
    # Calculate corrected coordinates using Cognitive Mass attraction snapping
    # We snap each noisy point to the word box that minimizes effective_dist
    corrected_x = []
    corrected_y = []
    
    for idx, row in df.iterrows():
        gx, gy = row['noisy_x'], row['noisy_y']
        
        # Find the best target word minimizing effective distance
        best_word = None
        min_eff_dist = float('inf')
        
        for t_idx, t_row in df.iterrows():
            tx, ty = t_row['true_x'], t_row['true_y']
            dist = np.sqrt((gx - tx)**2 + (gy - ty)**2)
            
            # CM pull factor: 1.0 + 1.8 * cognitive_mass
            cm = 1.0 + 1.8 * t_row['cognitive_mass']
            eff_dist = dist / cm
            
            if eff_dist < min_eff_dist:
                min_eff_dist = eff_dist
                best_word = t_row
                
        corrected_x.append(best_word['true_x'])
        corrected_y.append(best_word['true_y'])
        
    df['corrected_x'] = corrected_x
    df['corrected_y'] = corrected_y
    
    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    sns.set_theme(style="white")
    
    # Plot 1: Raw Noisy Gaze (with Drift) Heatmap
    sns.kdeplot(
        x=df['noisy_x'], y=df['noisy_y'],
        fill=True, thresh=0.05, levels=30, cmap="Reds",
        ax=axes[0], alpha=0.85
    )
    # Scatter true word positions for comparison
    axes[0].scatter(df['true_x'], df['true_y'], color='blue', s=15, alpha=0.4, label='True Word Positions')
    axes[0].set_title("1. Raw Noisy Gaze (Systemic +45px Vertical Drift & Jitter)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Screen X (px)")
    axes[0].set_ylabel("Screen Y (px)")
    axes[0].legend(loc="upper right")
    
    # Plot 2: Cognitive Mass Snapped Gaze Heatmap
    sns.kdeplot(
        x=df['corrected_x'], y=df['corrected_y'],
        fill=True, thresh=0.05, levels=30, cmap="Purples",
        ax=axes[1], alpha=0.85
    )
    axes[1].scatter(df['true_x'], df['true_y'], color='blue', s=15, alpha=0.4, label='True Word Positions')
    axes[1].set_title("2. Fused Gaze (Cognitive Mass Dynamic Snapping Correction)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Screen X (px)")
    axes[1].legend(loc="upper right")
    
    # Invert Y axis because screen space coordinates start from top-left (Y grows downward)
    plt.gca().invert_yaxis()
    
    plt.suptitle("LexiGaze Multimodal Gaze-Word Snapping Correction Heatmap", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_file = os.path.join(OUTPUT_DIR, "gaze_correction_heatmap.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"Heatmap visualization saved to: {out_file}")

if __name__ == "__main__":
    main()
