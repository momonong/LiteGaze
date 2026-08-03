"""
inspect_performance_demo.py - LexiGaze Interactive Module Performance Inspector
=============================================================================
This script runs a sandbox demo that evaluates the combination of different
LexiGaze modules (gaze tracking decoders, cognitive load pipelines, and
multimodal fusion models) side-by-side to inspect their joint effects on
system accuracy, difficulty correlation, and latency.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

if __package__:
    from .experiment_manifest import write_experiment_manifest
    from .fusion_module import LexiGazeFusion
    from .geco.core.attention_transition import AttentionGuidedMatrix
    from .geco.core.baseline_decoders import NearestBoundingBoxDecoder
    from .geco.core.em_calibration import AutoCalibratingDecoder
    from .geco.core.transition_model import (
        PsycholinguisticTransitionMatrix,
        ReadingTransitionMatrix,
    )
    from .geco.core.viterbi_decoder import viterbi_gaze_decode
else:
    from experiment_manifest import write_experiment_manifest
    from fusion_module import LexiGazeFusion
    from geco.core.attention_transition import AttentionGuidedMatrix
    from geco.core.baseline_decoders import NearestBoundingBoxDecoder
    from geco.core.em_calibration import AutoCalibratingDecoder
    from geco.core.transition_model import (
        PsycholinguisticTransitionMatrix,
        ReadingTransitionMatrix,
    )
    from geco.core.viterbi_decoder import viterbi_gaze_decode

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "archive/data/geco/geco_pp01_cognitive_mass.csv"
ATTN_PATH = PROJECT_ROOT / "archive/data/geco/geco_pp01_cognitive_mass_attention.npy"
CLEAN_TRIAL_PATH = PROJECT_ROOT / "archive/data/geco/geco_pp01_trial5_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Noise injection parameters (simulating typical webcam inaccuracies)
RANDOM_SEED = 42
DRIFT_Y = 45.0
SIGMA_X = 40.0
SIGMA_Y = 30.0
CALIBRATION_WINDOW_SIZE = 30

def inject_noise(df, rng):
    """Inject systematic vertical drift and random Gaussian horizontal/vertical jitter."""
    df = df.copy()
    df['noisy_x'] = df['true_x'] + rng.normal(0, SIGMA_X, len(df))
    df['noisy_y'] = df['true_y'] + rng.normal(0, SIGMA_Y, len(df)) + DRIFT_Y
    return df

def calculate_accuracy(target_indices, predicted_indices):
    """Compute the accuracy of matching gaze points to the correct words."""
    correct = sum(1 for t, p in zip(target_indices, predicted_indices) if t == p)
    return (correct / len(target_indices)) * 100

def main():
    experiment_started = time.perf_counter()
    print("=========================================================")
    print("      LexiGaze Module Performance Inspection Demo         ")
    print("=========================================================")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Ensure raw data exists in archive/data/geco/")
        write_experiment_manifest(
            OUTPUT_DIR / "demo_system_comparison_manifest.json",
            "joint_system_performance_geco_pp01_trial5",
            root=PROJECT_ROOT,
            datasets=[DATA_PATH, ATTN_PATH, CLEAN_TRIAL_PATH],
            config=_base_manifest_config(),
            metrics={"error": f"missing required dataset: {DATA_PATH}"},
            seed=RANDOM_SEED,
            status="failed",
            duration_seconds=round(time.perf_counter() - experiment_started, 6),
        )
        return 1
        
    # Load dataset
    rng = np.random.RandomState(RANDOM_SEED)
    df = pd.read_csv(DATA_PATH)
    df = inject_noise(df, rng)
    
    # Extract actual reading times from GECO (we merge with the clean trial data to get TRT)
    if CLEAN_TRIAL_PATH.exists():
        df_clean = pd.read_csv(CLEAN_TRIAL_PATH)
        df = pd.merge(df, df_clean[["WORD_ID", "WORD_TOTAL_READING_TIME"]], on="WORD_ID", how="left")
        # Fill missing values if any
        df["WORD_TOTAL_READING_TIME"] = df["WORD_TOTAL_READING_TIME"].fillna(df["WORD_TOTAL_READING_TIME"].median())
    else:
        # Fallback to simulated reading times if clean CSV is missing
        df["WORD_TOTAL_READING_TIME"] = df["cognitive_mass"] * 100 + rng.normal(100, 50, len(df))
        df["WORD_TOTAL_READING_TIME"] = np.clip(df["WORD_TOTAL_READING_TIME"], 50, None)

    # Word layout coordinates
    word_boxes = [[row['true_x']-20, row['true_y']-15, row['true_x']+20, row['true_y']+15] for _, row in df.iterrows()]
    gaze_sequence = df[['noisy_x', 'noisy_y']].values
    true_indices = list(range(len(df)))
    trt = df["WORD_TOTAL_READING_TIME"].values
    
    # Load attention matrix
    attn_matrix = np.load(ATTN_PATH) if ATTN_PATH.exists() else None
    
    # Define system configurations
    # We will compare seven distinct system configurations
    configs = {
        "1. Raw Gaze + No Cog + Linear": {
            "gaze_decoder": "nearest_box",
            "cog_load": "none",
            "fusion": "linear"
        },
        "2. Viterbi + No Cog + Linear": {
            "gaze_decoder": "viterbi_base",
            "cog_load": "none",
            "fusion": "linear"
        },
        "3. Viterbi + EM Calib + No Cog + Linear": {
            "gaze_decoder": "viterbi_em",
            "cog_load": "none",
            "fusion": "linear"
        },
        "4. STOCK-T v1 + surprisal + Linear": {
            "gaze_decoder": "stock_t_v1",
            "cog_load": "surprisal",
            "fusion": "linear"
        },
        "5. STOCK-T v2 + surprisal + Multiplicative": {
            "gaze_decoder": "stock_t_v2",
            "cog_load": "surprisal",
            "fusion": "multiplicative"
        },
        "6. STOCK-T v3 + surprisal + Bayesian": {
            "gaze_decoder": "stock_t_v3_pom",
            "cog_load": "surprisal",
            "fusion": "bayesian"
        },
        "7. STOCK-T v3 + CogMass + Bayesian": {
            "gaze_decoder": "stock_t_v3_pom",
            "cog_load": "cognitive_mass",
            "fusion": "bayesian"
        }
    }
    
    fusion_module = LexiGazeFusion()
    results = []
    
    for name, c in configs.items():
        print(f"Running Configuration: {name}...")
        t0 = time.time()
        
        # A. Gaze Decoder Step
        dec_name = c["gaze_decoder"]
        if dec_name == "nearest_box":
            decoder = NearestBoundingBoxDecoder()
            decoded_indices = decoder.decode(gaze_sequence, word_boxes)
        elif dec_name == "viterbi_base":
            transition_builder = ReadingTransitionMatrix()
            t_matrix = transition_builder.build_matrix(df['cognitive_mass'].values, is_L2_reader=True)
            decoded_indices, _ = viterbi_gaze_decode(gaze_sequence, word_boxes, df['cognitive_mass'].values, t_matrix, sigma_gaze=[SIGMA_X, SIGMA_Y])
        elif dec_name == "viterbi_em":
            transition_builder = ReadingTransitionMatrix()
            t_matrix = transition_builder.build_matrix(df['cognitive_mass'].values, is_L2_reader=True)
            calibrator = AutoCalibratingDecoder(calibration_window_size=CALIBRATION_WINDOW_SIZE)
            decoded_indices, _ = calibrator.calibrate_and_decode(gaze_sequence, word_boxes, df['cognitive_mass'].values, t_matrix, sigma_gaze=[SIGMA_X, SIGMA_Y])
        elif dec_name == "stock_t_v1":
            stock_t_v1_builder = AttentionGuidedMatrix(mu_saccade=1.0)
            t_matrix_v1 = stock_t_v1_builder.build_matrix(len(df), attn_matrix)
            decoded_indices, _ = viterbi_gaze_decode(gaze_sequence, word_boxes, df['cognitive_mass'].values, t_matrix_v1, sigma_gaze=[SIGMA_X, SIGMA_Y])
        elif dec_name == "stock_t_v2":
            stock_t_v2_builder = AttentionGuidedMatrix(regression_sensitivity=0.8, top_k_anchors=2)
            t_matrix_v2 = stock_t_v2_builder.build_matrix(len(df), attn_matrix, df['cognitive_mass'].values)
            decoded_indices, _ = viterbi_gaze_decode(gaze_sequence, word_boxes, df['cognitive_mass'].values, t_matrix_v2, sigma_gaze=[SIGMA_X, SIGMA_Y])
        elif dec_name == "stock_t_v3_pom":
            pom_builder = PsycholinguisticTransitionMatrix(sigma_fwd=1.0, sigma_reg=1.5, gamma=0.5)
            t_matrix_pom = pom_builder.build_matrix(len(df), df['cognitive_mass'].values)
            calibrator_pom = AutoCalibratingDecoder(calibration_window_size=CALIBRATION_WINDOW_SIZE)
            decoded_indices, _ = calibrator_pom.calibrate_and_decode(gaze_sequence, word_boxes, df['cognitive_mass'].values, t_matrix_pom, sigma_gaze=[SIGMA_X, SIGMA_Y])
        else:
            decoded_indices = true_indices
            
        gaze_acc = calculate_accuracy(true_indices, decoded_indices)
        
        # B. Cognitive Load Step
        cog_type = c["cog_load"]
        if cog_type == "none":
            load_scores = np.zeros(len(df))
        elif cog_type == "surprisal":
            load_scores = df["surprisal_score"].values
        else:
            load_scores = df["cognitive_mass"].values
            
        # C. Fusion Step
        fuse_type = c["fusion"]
        # Convert index sequence to simulated dwell/fixation events
        # We accumulate simulated gaze occurrences for decoded words
        sim_dwell = np.zeros(len(df))
        sim_fix = np.zeros(len(df))
        for idx in decoded_indices:
            if idx < len(df):
                sim_dwell[idx] += 120 # 120ms per gaze frame
                sim_fix[idx] += 1
                
        if fuse_type == "linear":
            rds = fusion_module.fuse_linear(sim_dwell, sim_fix, load_scores)
        elif fuse_type == "multiplicative":
            rds = fusion_module.fuse_multiplicative(sim_dwell, sim_fix, load_scores)
        elif fuse_type == "bayesian":
            rds = fusion_module.fuse_bayesian(sim_dwell, load_scores)
        else:
            rds = sim_dwell
            
        # Evaluate Fused RDS vs. Human Reading Time (TRT)
        rho, _ = spearmanr(rds, trt)
        latency_ms = (time.time() - t0) * 1000
        
        results.append({
            "Configuration": name,
            "Gaze Accuracy (%)": round(gaze_acc, 2),
            "RDS Correlation (rho)": round(rho, 4),
            "Latency (ms)": round(latency_ms, 2)
        })
        
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("COMPARATIVE PERFORMANCE DASHBOARD (JOINT SYSTEM EFFECTS)")
    print("="*70)
    print(df_res.to_string(index=False))
    print("="*70)
    
    # Save statistics as CSV
    df_res.to_csv(OUTPUT_DIR / "demo_system_comparison.csv", index=False)
    
    # Plot performance figures
    plot_comparison(df_res)
    best_accuracy_row = df_res.loc[df_res["Gaze Accuracy (%)"].idxmax()]
    best_rds_row = df_res.loc[df_res["RDS Correlation (rho)"].idxmax()]
    manifest_path = write_experiment_manifest(
        OUTPUT_DIR / "demo_system_comparison_manifest.json",
        "joint_system_performance_geco_pp01_trial5",
        root=PROJECT_ROOT,
        datasets=[DATA_PATH, ATTN_PATH, CLEAN_TRIAL_PATH],
        artifacts=[
            OUTPUT_DIR / "demo_system_comparison.csv",
            OUTPUT_DIR / "demo_performance_comparison.png",
        ],
        config={
            **_base_manifest_config(),
            "reading_time_source": "geco_clean_trial" if CLEAN_TRIAL_PATH.exists() else "simulated",
            "attention_matrix_available": ATTN_PATH.exists(),
            "system_configurations": configs,
        },
        metrics={
            "row_count": len(df),
            "best_gaze_accuracy_configuration": best_accuracy_row["Configuration"],
            "best_gaze_accuracy_percent": best_accuracy_row["Gaze Accuracy (%)"],
            "best_rds_configuration": best_rds_row["Configuration"],
            "best_spearman_rho": best_rds_row["RDS Correlation (rho)"],
            "configuration_results": df_res.to_dict(orient="records"),
        },
        seed=RANDOM_SEED,
        duration_seconds=round(time.perf_counter() - experiment_started, 6),
    )
    
    print(f"\nSaved comparison statistics to: {OUTPUT_DIR}/demo_system_comparison.csv")
    print(f"Saved performance chart to: {OUTPUT_DIR}/demo_performance_comparison.png")
    print(f"Saved reproducibility manifest to: {manifest_path}")
    print("=========================================================")
    return 0


def _base_manifest_config():
    return {
        "corpus": "GECO",
        "participant": "pp01",
        "trial": 5,
        "drift_y_px": DRIFT_Y,
        "sigma_x_px": SIGMA_X,
        "sigma_y_px": SIGMA_Y,
        "calibration_window_size": CALIBRATION_WINDOW_SIZE,
    }

def plot_comparison(df):
    """Draw a side-by-side bar chart showing system performance trade-offs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style="whitegrid")
    
    # Gaze Accuracy
    sns.barplot(x="Gaze Accuracy (%)", y="Configuration", data=df, ax=axes[0], hue="Configuration", palette="viridis", legend=False)
    axes[0].set_title("Gaze Decoding Fixation Accuracy (%)")
    axes[0].set_xlim(0, 105)
    for p in axes[0].patches:
        width = p.get_width()
        axes[0].annotate(f'{width:.1f}%', (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=10, color='black', xytext=(5, 0),
                    textcoords='offset points')
                    
    # RDS Correlation
    sns.barplot(x="RDS Correlation (rho)", y="Configuration", data=df, ax=axes[1], hue="Configuration", palette="plasma", legend=False)
    axes[1].set_title("Difficulty correlation (Spearman rho)")
    axes[1].set_xlim(0, 1.05)
    for p in axes[1].patches:
        width = p.get_width()
        axes[1].annotate(f'{width:.3f}', (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=10, color='black', xytext=(5, 0),
                    textcoords='offset points')
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")

    # Latency
    sns.barplot(x="Latency (ms)", y="Configuration", data=df, ax=axes[2], hue="Configuration", palette="rocket", legend=False)
    axes[2].set_title("Processing Latency (ms)")
    for p in axes[2].patches:
        width = p.get_width()
        axes[2].annotate(f'{width:.1f}ms', (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=10, color='black', xytext=(5, 0),
                    textcoords='offset points')
    axes[2].set_yticklabels([])
    axes[2].set_ylabel("")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "demo_performance_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    raise SystemExit(main())
