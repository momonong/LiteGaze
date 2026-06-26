import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.geco.core.cm_algorithm import BayesianGravitySnap
from scripts.geco.core.transition_model import ReadingTransitionMatrix, PsycholinguisticTransitionMatrix
from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder, StandardKalmanDecoder, StaticBayesianDecoder
from scripts.geco.core.em_calibration import AutoCalibratingDecoder
from scripts.geco.core.attention_transition import AttentionGuidedMatrix, print_sample_matrix

# 1. Configuration & Parameters
DATA_PATH = "data/geco/geco_pp01_cognitive_mass.csv"
ATTN_PATH = "data/geco/geco_pp01_cognitive_mass_attention.npy"
OUTPUT_DIR = "output"
OS_NAME = "linux"

# Noise Model
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

def inject_noise(df):
    """Inject systematic drift and Gaussian jitter."""
    np.random.seed(42)
    df['noisy_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, len(df))
    df['noisy_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    return df

def evaluate_accuracy(target_words, predicted_words):
    """Calculate word-level accuracy."""
    correct = sum(1 for t, p in zip(target_words, predicted_words) if str(t).strip() == str(p).strip())
    return (correct / len(target_words)) * 100

def run_evaluation():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Run extract_cognitive_mass.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    df = inject_noise(df)
    
    # Preparation for Algorithms
    word_boxes = [[row['true_x']-20, row['true_y']-15, row['true_x']+20, row['true_y']+15] for _, row in df.iterrows()]
    base_cm = df['cognitive_mass'].values
    gaze_sequence = df[['noisy_x', 'noisy_y']].values
    true_words = df['WORD'].values

    # --- Config A: Nearest Box (Baseline 1) ---
    print("Running Config A: Nearest Bounding Box...")
    decoder_a = NearestBoundingBoxDecoder()
    indices_a = decoder_a.decode(gaze_sequence, word_boxes)
    acc_a = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_a])

    # --- Config B: Standard Kalman (Baseline 2) ---
    print("Running Config B: Standard Kalman...")
    decoder_b = StandardKalmanDecoder()
    indices_b = decoder_b.decode(gaze_sequence, word_boxes)
    acc_b = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_b])

    # --- Config C: Static Bayesian (Baseline 3) ---
    print("Running Config C: Static Bayesian...")
    decoder_c = StaticBayesianDecoder(sigma_x=SIGMA_X, sigma_y=SIGMA_Y)
    indices_c = decoder_c.decode(gaze_sequence, word_boxes, base_cm)
    acc_c = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_c])

    # --- Config D: Spatio-Temporal Viterbi (Our Breakthrough) ---
    print("Running Config D: Spatio-Temporal Viterbi (No OVP)...")
    transition_builder = ReadingTransitionMatrix()
    t_matrix = transition_builder.build_matrix(base_cm, is_L2_reader=True)
    indices_d, _ = viterbi_gaze_decode(gaze_sequence, word_boxes, base_cm, t_matrix, sigma_gaze=[SIGMA_X, SIGMA_Y], use_ovp=False)
    acc_d = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_d])

    # --- Config E: Viterbi + OVP (Skill 7) ---
    print("Running Config E: Viterbi + OVP Optimization...")
    indices_e, _ = viterbi_gaze_decode(gaze_sequence, word_boxes, base_cm, t_matrix, sigma_gaze=[SIGMA_X, SIGMA_Y], use_ovp=True)
    acc_e = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_e])

    # --- Config F: Viterbi + EM Auto-Calibration (Skill 6) ---
    print("Running Config F: Viterbi + EM Auto-Calibration...")
    calibrator = AutoCalibratingDecoder(calibration_window_size=30)
    indices_f, drift = calibrator.calibrate_and_decode(gaze_sequence, word_boxes, base_cm, t_matrix, sigma_gaze=[SIGMA_X, SIGMA_Y])
    acc_f = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_f])

    # --- Config G: STOCK-T v1 (Skill 8) ---
    print("Running Config G: STOCK-T v1 (Global Attention)...")
    if os.path.exists(ATTN_PATH):
        attn_matrix = np.load(ATTN_PATH)
        # Skill 8: Fixed alpha fusion (0.5 for demonstration of the regression)
        stock_t_v1_builder = AttentionGuidedMatrix(mu_saccade=1.0)
        # Manually mimic v1 behavior by passing None for base_cm to trigger fallback alpha
        t_matrix_v1 = stock_t_v1_builder.build_matrix(len(df), attn_matrix)
        indices_g, _ = viterbi_gaze_decode(gaze_sequence, word_boxes, base_cm, t_matrix_v1, sigma_gaze=[SIGMA_X, SIGMA_Y])
        acc_g = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_g])
    else:
        acc_g = 0

    # --- Config H: STOCK-T v2 (Skill 9) ---
    print("Running Config H: STOCK-T v2 (Cognitive-Gated Sparse Attention)...")
    if os.path.exists(ATTN_PATH):
        # Skill 9: Dynamic alpha + Sparsification
        stock_t_v2_builder = AttentionGuidedMatrix(regression_sensitivity=0.8, top_k_anchors=2)
        t_matrix_v2 = stock_t_v2_builder.build_matrix(len(df), attn_matrix, base_cm)
        
        # Verify matrix
        print_sample_matrix(t_matrix_v2, n=8)
        
        indices_h, _ = viterbi_gaze_decode(gaze_sequence, word_boxes, base_cm, t_matrix_v2, sigma_gaze=[SIGMA_X, SIGMA_Y])
        acc_h = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_h])
    else:
        acc_h = 0

    # --- Config I: STOCK-T v3 (Skill 10 - POM) ---
    print("Running Config I: STOCK-T v3 (Psycholinguistic POM)...")
    # Skill 10: Purely math-driven transition
    pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=1.0, sigma_reg=1.5, gamma=0.5)
    t_matrix_pom = pom_builder.build_matrix(len(df), base_cm)
    
    # Run with EM-Calibration for maximum accuracy
    calibrator_pom = AutoCalibratingDecoder(calibration_window_size=30)
    indices_i, drift_pom = calibrator_pom.calibrate_and_decode(gaze_sequence, word_boxes, base_cm, t_matrix_pom, sigma_gaze=[SIGMA_X, SIGMA_Y])
    acc_i = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_i])

    # 2. Results Summary
    results = {
        "Configuration": [
            "Nearest Box", 
            "Kalman Filter", 
            "Static Bayesian", 
            "Viterbi (Base)", 
            "Viterbi + OVP", 
            "Viterbi + EM-AutoCal",
            "STOCK-T v1",
            "STOCK-T v2",
            "STOCK-T v3 (POM)"
        ],
        "Accuracy (%)": [acc_a, acc_b, acc_c, acc_d, acc_e, acc_f, acc_g, acc_h, acc_i]
    }
    res_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("📊 Advanced Evaluation Results (NeurIPS Benchmark)")
    print("="*50)
    print(res_df)
    print("="*50)

    # 3. Visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Configuration", y="Accuracy (%)", data=res_df, hue="Configuration", palette="viridis", legend=False)
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.title(f"Gaze Correction Accuracy Comparison (+{int(DRIFT_Y)}px Vertical Drift)", fontsize=14)
    plt.ylim(0, max(acc_f, acc_g, 50) + 10)
    plt.xticks(rotation=15)
    plt.ylabel("Word-Level Fixation Accuracy (%)")
    plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png", dpi=300)
    print(f"✅ Chart saved to {OUTPUT_DIR}/accuracy_comparison.png")

    # Save CSV
    res_df.to_csv("data/geco/geco_pp01_final_evaluation.csv", index=False)

if __name__ == "__main__":
    run_evaluation()
