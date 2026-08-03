"""
experiment_fusion.py - Run and evaluate gaze-cognitive fusion methods on GECO data
==================================================================================
This script loads GECO trial 5 clean reading data and its corresponding BERT
cognitive mass features, applies eleven fusion algorithms, correlates the
results with actual human reading times, and saves the output plots and report.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

if __package__:
    from .experiment_manifest import capture_source_snapshot, write_experiment_manifest
    from .fusion_module import LexiGazeFusion
else:
    from experiment_manifest import capture_source_snapshot, write_experiment_manifest
    from fusion_module import LexiGazeFusion

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GECO_CLEAN_PATH = PROJECT_ROOT / "archive/data/geco/geco_pp01_trial5_clean.csv"
GECO_COG_PATH = PROJECT_ROOT / "archive/data/geco/geco_pp01_cognitive_mass.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

RANDOM_SEED = 42
SKIP_RATE = 0.30
DWELL_JITTER_SIGMA = 40.0
FIXATION_JITTER_SIGMA = 10.0
FIXATION_SCALE = 0.85
GATED_THRESHOLD = 0.25

def main():
    experiment_started = time.perf_counter()
    source_snapshot = capture_source_snapshot(PROJECT_ROOT)
    print("Starting Multimodal Fusion Experiment...")
    
    # 1. Check directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")
    
    if not GECO_CLEAN_PATH.exists() or not GECO_COG_PATH.exists():
        error = "Required data files not found in archive/data/geco/"
        print(f"Error: {error}")
        write_experiment_manifest(
            OUTPUT_DIR / "fusion_experiment_manifest.json",
            "fusion_comparison_geco_pp01_trial5",
            root=PROJECT_ROOT,
            datasets=[GECO_CLEAN_PATH, GECO_COG_PATH],
            config=_manifest_config(),
            metrics={"error": error},
            seed=RANDOM_SEED,
            status="failed",
            duration_seconds=round(time.perf_counter() - experiment_started, 6),
            source_snapshot=source_snapshot,
        )
        return 1
        
    # 2. Load and merge datasets
    df_clean = pd.read_csv(GECO_CLEAN_PATH)
    df_cog = pd.read_csv(GECO_COG_PATH)
    
    print(f"Loaded clean trial data: {len(df_clean)} rows")
    print(f"Loaded cognitive features: {len(df_cog)} rows")
    
    # Merge on WORD_ID and WORD
    df_merged = pd.merge(df_clean, df_cog, on=["WORD_ID", "WORD", "true_x", "true_y"])
    print(f"Merged dataset: {len(df_merged)} rows")
    
    # 3. Extract variables
    # Ground truth: actual human reading time
    trt = df_merged["WORD_TOTAL_READING_TIME"].values
    
    # Simulate realistic webcam gaze tracking with skips and noise
    # (30% word skip rate, plus Gaussian measurement jitter)
    # Local legacy RNG preserves the historical seed-42 sequence without
    # mutating NumPy's process-wide random state.
    rng = np.random.RandomState(RANDOM_SEED)
    mask = rng.choice([0, 1], size=len(trt), p=[SKIP_RATE, 1.0 - SKIP_RATE])
    g_dwell = trt * mask + rng.normal(0, DWELL_JITTER_SIGMA, len(trt)) * mask
    g_dwell = np.clip(g_dwell, 0, None)
    
    g_fix = (trt * FIXATION_SCALE + rng.normal(0, FIXATION_JITTER_SIGMA, len(trt))) * mask
    g_fix = np.clip(g_fix, 0, None)
    
    # Cognitive variable: full cognitive mass (ensemble combination of all linguistic features)
    c_load = df_merged["cognitive_mass"].values
    
    # 4. Instantiate fusion class and apply algorithms
    fusion = LexiGazeFusion()
    
    df_merged["RDS_linear"] = fusion.fuse_linear(g_dwell, g_fix, c_load)
    df_merged["RDS_multiplicative"] = fusion.fuse_multiplicative(g_dwell, g_fix, c_load)
    df_merged["RDS_gated"] = fusion.fuse_gated(g_dwell, g_fix, c_load, threshold=GATED_THRESHOLD)
    df_merged["RDS_sigmoid"] = fusion.fuse_sigmoid(g_dwell, g_fix, c_load)
    df_merged["RDS_bayesian"] = fusion.fuse_bayesian(g_dwell, c_load)
    df_merged["RDS_rrf"] = fusion.fuse_rrf(g_dwell, c_load)
    df_merged["RDS_spillover_bayesian"] = fusion.fuse_spillover_bayesian(g_dwell, c_load)
    df_merged["RDS_parafoveal"] = fusion.fuse_parafoveal(g_dwell, c_load)
    df_merged["RDS_spillover_rrf"] = fusion.fuse_spillover_rrf(g_dwell, c_load)
    df_merged["RDS_parafoveal_rrf"] = fusion.fuse_parafoveal_rrf(g_dwell, c_load)
    df_merged["RDS_spillover_parafoveal_rrf"] = fusion.fuse_spillover_parafoveal_rrf(g_dwell, c_load)
    
    # 5. Evaluate correlation with actual reading time (TRT)
    methods = [
        "RDS_linear", "RDS_multiplicative", "RDS_gated", 
        "RDS_sigmoid", "RDS_bayesian", "RDS_rrf",
        "RDS_spillover_bayesian", "RDS_parafoveal",
        "RDS_spillover_rrf", "RDS_parafoveal_rrf",
        "RDS_spillover_parafoveal_rrf"
    ]
    
    eval_results = []
    for m in methods:
        p_coef, p_pval = pearsonr(df_merged[m], trt)
        s_coef, s_pval = spearmanr(df_merged[m], trt)
        
        eval_results.append({
            "Method": m.replace("RDS_", "").capitalize(),
            "Pearson r": round(p_coef, 4),
            "Pearson p-val": f"{p_pval:.2e}",
            "Spearman rho": round(s_coef, 4),
            "Spearman p-val": f"{s_pval:.2e}"
        })
        
    df_eval = pd.DataFrame(eval_results)
    print("\nEvaluation Results (Correlation with Actual Human Reading Time):")
    print(df_eval.to_string(index=False))
    
    # Save evaluation summary to CSV
    df_eval.to_csv(OUTPUT_DIR / "fusion_evaluation_summary.csv", index=False)
    
    # Save detailed merged output
    df_merged.to_csv(OUTPUT_DIR / "fused_rds_dataset.csv", index=False)
    print(f"Saved fused dataset and evaluation summary to {OUTPUT_DIR}/")
    
    # 6. Generate Visualizations
    sns.set_theme(style="whitegrid")
    
    # A. Barplot of Correlations
    plt.figure(figsize=(10, 6))
    df_plot = df_eval.melt(id_vars=["Method"], value_vars=["Pearson r", "Spearman rho"], 
                           var_name="Metric", value_name="Correlation")
    ax = sns.barplot(x="Method", y="Correlation", hue="Metric", data=df_plot, palette="muted")
    plt.title("Correlation of Fused RDS Methods with Actual Human Reading Time (TRT)", fontsize=14)
    plt.ylim(0.0, 1.1)
    plt.ylabel("Correlation Coefficient")
    plt.xlabel("Fusion Algorithm")
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                        textcoords='offset points')
            
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fusion_correlation_comparison.png", dpi=300)
    plt.close()
    
    # B. Distribution Plot of RDS
    plt.figure(figsize=(12, 7))
    for m in methods:
        sns.kdeplot(df_merged[m], label=m.replace("RDS_", "").capitalize(), fill=True, alpha=0.1)
    plt.title("Reading Difficulty Score (RDS) Distribution by Fusion Method", fontsize=14)
    plt.xlabel("Fused RDS Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rds_distributions.png", dpi=300)
    plt.close()
    
    # C. Scatter Plot of Best Method (Multiplicative or Bayesian)
    best_method = df_eval.loc[df_eval["Spearman rho"].idxmax()]["Method"]
    best_col = f"RDS_{best_method.lower()}"
    print(f"\nBest Performing Method: {best_method} (Spearman rho = {df_eval.loc[df_eval['Method'] == best_method, 'Spearman rho'].values[0]})")
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_merged["surprisal_score"], df_merged["WORD_TOTAL_READING_TIME"], 
                          c=df_merged[best_col], cmap="viridis", s=60, alpha=0.8, edgecolors="none")
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"Fused RDS ({best_method})", rotation=270, labelpad=15)
    plt.xlabel("Cognitive Surprisal (BERT)")
    plt.ylabel("Actual Human Reading Time (ms)")
    plt.title(f"Gaze-Cognitive Space colored by Fused RDS ({best_method} Method)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gaze_cognitive_space_rds.png", dpi=300)
    plt.close()
    
    # D. Top 10 Difficult Words Table & Barplot
    df_sorted = df_merged.sort_values(by=best_col, ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=best_col, y="WORD", data=df_sorted, palette="flare", hue="WORD", legend=False)
    plt.title(f"Top 10 Most Difficult Words Identified by {best_method} Fusion", fontsize=14)
    plt.xlabel(f"Fused RDS ({best_method})")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_difficult_words.png", dpi=300)
    plt.close()
    
    # 7. Write Experiment Report
    write_report(df_eval, best_method, df_sorted, len(df_merged))
    best_spearman_row = df_eval.loc[df_eval["Method"] == best_method].iloc[0]
    best_pearson_row = df_eval.loc[df_eval["Pearson r"].idxmax()]
    artifacts = [
        OUTPUT_DIR / "fusion_evaluation_summary.csv",
        OUTPUT_DIR / "fused_rds_dataset.csv",
        OUTPUT_DIR / "fusion_correlation_comparison.png",
        OUTPUT_DIR / "rds_distributions.png",
        OUTPUT_DIR / "gaze_cognitive_space_rds.png",
        OUTPUT_DIR / "top_difficult_words.png",
        OUTPUT_DIR / "fusion_experiment_report.md",
    ]
    manifest_path = write_experiment_manifest(
        OUTPUT_DIR / "fusion_experiment_manifest.json",
        "fusion_comparison_geco_pp01_trial5",
        root=PROJECT_ROOT,
        datasets=[GECO_CLEAN_PATH, GECO_COG_PATH],
        artifacts=artifacts,
        config=_manifest_config(),
        metrics={
            "row_count": len(df_merged),
            "best_spearman_method": best_method,
            "best_spearman_rho": best_spearman_row["Spearman rho"],
            "pearson_r_for_best_spearman": best_spearman_row["Pearson r"],
            "best_pearson_method": best_pearson_row["Method"],
            "best_pearson_r": best_pearson_row["Pearson r"],
            "method_results": df_eval.to_dict(orient="records"),
        },
        seed=RANDOM_SEED,
        duration_seconds=round(time.perf_counter() - experiment_started, 6),
        source_snapshot=source_snapshot,
    )
    print(f"Saved experiment report to {OUTPUT_DIR}/fusion_experiment_report.md")
    print(f"Saved reproducibility manifest to {manifest_path}")
    print("Fusion Experiment Completed successfully!")
    return 0


def _manifest_config():
    return {
        "corpus": "GECO",
        "participant": "pp01",
        "trial": 5,
        "skip_rate": SKIP_RATE,
        "dwell_jitter_sigma": DWELL_JITTER_SIGMA,
        "fixation_jitter_sigma": FIXATION_JITTER_SIGMA,
        "fixation_scale": FIXATION_SCALE,
        "gated_threshold": GATED_THRESHOLD,
        "evaluation_scope": "single_participant_descriptive_calibration",
        "same_event_gaze_derived_from_target_trt": True,
        "predictive_generalization_eligible": False,
    }

def df_to_markdown(df):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_strs = [str(val) for val in row.values]
        lines.append("| " + " | ".join(row_strs) + " |")
    return "\n".join(lines)

def write_report(df_eval, best_method, df_top10, row_count):
    report_path = OUTPUT_DIR / "fusion_experiment_report.md"
    
    table_eval = df_to_markdown(df_eval)
    top10_cols = ["WORD_ID", "WORD", "WORD_TOTAL_READING_TIME", "surprisal_score", f"RDS_{best_method.lower()}"]
    table_top10 = df_to_markdown(df_top10[top10_cols])
    
    report_content = f"""# LexiGaze Multimodal Gaze-Cognitive Fusion Report

This report summarizes the comparative evaluation of eleven fusion algorithms designed to combine eye-gaze tracking metrics (total reading time / dwell duration) with cognitive load metrics (information surprisal from BERT) into a unified **Reading Difficulty Score (RDS)**.

The algorithms were tested on **{row_count} merged word records** from the **GECO Corpus (pp01, Trial 5)**. The ground-truth reading difficulty is represented by the actual human **Total Reading Time (TRT)**.

---

## Validity Scope

This is a **descriptive calibration diagnostic**, not an out-of-sample prediction benchmark. The simulated dwell and fixation inputs are constructed from the same TRT used as the evaluation target, and the current cognitive-mass extraction path can include GECO-supervised XGBoost/Ridge scores. Use `output/geco_generalization_report.md` for preregistered new-reader/new-trial evidence.

---

## Fusion Performance Summary

Each fusion method was evaluated by computing its Pearson correlation ($r$) and Spearman rank correlation ($\\rho$) against the ground-truth Total Reading Time.

{table_eval}

---

## Key Findings

1. **Best Performing Method**: The **{best_method}** fusion algorithm achieved the highest Spearman correlation of **{df_eval.loc[df_eval['Method'] == best_method, 'Spearman rho'].values[0]}** and Pearson correlation of **{df_eval.loc[df_eval['Method'] == best_method, 'Pearson r'].values[0]}**.
2. **Interactive Effects**: Multiplicative and Bayesian update methods typically outperform simple linear sums. This is because reading difficulty is non-linear: a high-surprisal word that is skipped (short dwell) does not present actual cognitive difficulty to the reader, whereas high surprisal accompanied by long dwell duration indicates true processing bottleneck.
3. **Rank-based Robustness**: Reciprocal Rank Fusion (RRF) provides a robust, scale-invariant alternative that requires no parameter tuning and remains highly correlated with reading times.

---

## Top 10 Most Difficult Words (Identified by {best_method} Fusion)

Below are the top 10 words identified as having the highest reading difficulty under the best fusion model.

{table_top10}

---

## Visualizations Generated in `output/`

1. **`fusion_correlation_comparison.png`**: Bar chart comparing Pearson and Spearman correlation coefficients across all 11 methods.
2. **`rds_distributions.png`**: Density plot showing the RDS score distributions.
3. **`gaze_cognitive_space_rds.png`**: Scatter plot of the 2D gaze-cognitive space (Surprisal vs. Dwell time) colored by fused RDS.
4. **`top_difficult_words.png`**: Horizontal bar plot of the top 10 most difficult words.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

if __name__ == "__main__":
    raise SystemExit(main())

