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

from scripts.geco.core.transition_model import ReadingTransitionMatrix, PsycholinguisticTransitionMatrix
from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode
from scripts.geco.core.baseline_decoders import NearestBoundingBoxDecoder, StandardKalmanDecoder, StaticBayesianDecoder
from scripts.geco.core.em_calibration import AutoCalibratingDecoder

# 1. Configuration & Parameters
DATA_PATH = "data/geco/geco_l1_pp01_cognitive_mass.csv"
OUTPUT_DIR = "output"

# Noise Model (Same as L2 for fair comparison)
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0

def inject_noise(df):
    np.random.seed(42)
    df['noisy_x'] = df['true_x'] + np.random.normal(0, SIGMA_X, len(df))
    df['noisy_y'] = df['true_y'] + np.random.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    return df

def evaluate_accuracy(target_words, predicted_words):
    correct = sum(1 for t, p in zip(target_words, predicted_words) if str(t).strip().lower() == str(p).strip().lower())
    return (correct / len(target_words)) * 100

def run_l1_evaluation():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df = inject_noise(df)
    
    word_boxes = [[row['true_x']-20, row['true_y']-15, row['true_x']+20, row['true_y']+15] for _, row in df.iterrows()]
    base_cm = df['cognitive_mass'].values
    gaze_sequence = df[['noisy_x', 'noisy_y']].values
    true_words = df['WORD'].values

    # Baseline 1: Nearest Box
    decoder_a = NearestBoundingBoxDecoder()
    indices_a = decoder_a.decode(gaze_sequence, word_boxes)
    acc_a = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_a])

    # Baseline 2: Static Bayesian
    decoder_c = StaticBayesianDecoder(sigma_x=SIGMA_X, sigma_y=SIGMA_Y)
    indices_c = decoder_c.decode(gaze_sequence, word_boxes, base_cm)
    acc_c = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_c])

    # Config D: Base Viterbi (Native Settings)
    transition_builder = ReadingTransitionMatrix()
    # is_L2_reader=False for Native Dutch
    t_matrix = transition_builder.build_matrix(base_cm, is_L2_reader=False)
    indices_d, _ = viterbi_gaze_decode(gaze_sequence, word_boxes, base_cm, t_matrix, sigma_gaze=[SIGMA_X, SIGMA_Y])
    acc_d = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_d])

    # Config E: STOCK-T v3 (POM) with Default L2 Params
    pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3)
    t_matrix_pom = pom_builder.build_matrix(len(df), base_cm)
    calibrator = AutoCalibratingDecoder(calibration_window_size=30)
    indices_e, _ = calibrator.calibrate_and_decode(gaze_sequence, word_boxes, base_cm, t_matrix_pom, sigma_gaze=[SIGMA_X, SIGMA_Y])
    acc_e = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_e])

    # Config F: STOCK-T v3 (POM) with "Native Optimized" Params
    # Native readers: Larger saccades (87.6px vs 72.1px), so sigma_fwd should be larger?
    # Actually, they skip more, so gamma should be higher.
    pom_builder_nat = PsycholinguisticTransitionMatrix(sigma_fwd=1.2, sigma_reg=1.0, gamma=0.6)
    t_matrix_nat = pom_builder_nat.build_matrix(len(df), base_cm)
    indices_f, _ = calibrator.calibrate_and_decode(gaze_sequence, word_boxes, base_cm, t_matrix_nat, sigma_gaze=[SIGMA_X, SIGMA_Y])
    acc_f = evaluate_accuracy(true_words, [df.iloc[idx]['WORD'] for idx in indices_f])

    results = {
        "Configuration": ["Nearest Box", "Static Bayesian", "Viterbi (Native)", "STOCK-T (L2 Params)", "STOCK-T (Native Opt)"],
        "Accuracy (%)": [acc_a, acc_c, acc_d, acc_e, acc_f]
    }
    res_df = pd.DataFrame(results)
    print("\n📊 L1 (Native Dutch) Evaluation Results")
    print(res_df.to_string(index=False))

    # Save CSV
    res_df.to_csv("data/geco/geco_l1_final_evaluation.csv", index=False)

if __name__ == "__main__":
    run_l1_evaluation()
