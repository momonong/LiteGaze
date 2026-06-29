"""
scripts/optimize_viterbi_parameters.py
══════════════════════════════════════════════════════════════════════════════
Performs a grid search optimization over Viterbi decoding and drift-correction
hyperparameters using real subject ground truth files to maximize snapping accuracy.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Setup root path for import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode

# Real subject details map
SUBJECT_METRIC_MAP = {
    "gt_1782692940276.json": {"id": "subject001", "wpm": 73.7, "reg": 6 / 20.0},
    "gt_1782693148405.json": {"id": "subject002", "wpm": 43.5, "reg": 9 / 20.0},
    "gt_1782693226352.json": {"id": "subject003", "wpm": 33.3, "reg": 1 / 20.0},
    "gt_1782693294940.json": {"id": "subject004", "wpm": 29.0, "reg": 2 / 20.0},
    "gt_1782693357844.json": {"id": "subject005", "wpm": 31.6, "reg": 3 / 20.0},
}

def load_subject_data():
    gt_dir = ROOT / "data" / "ground_truth"
    gt_files = list(gt_dir.glob("gt_*.json"))
    
    loaded_trials = []
    
    for gt_file in gt_files:
        with open(gt_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        pairs = data.get("pairs", [])
        if not pairs:
            continue
            
        gaze_seq = np.array([[p["gaze_x"], p["gaze_y"]] for p in pairs], dtype=float)
        true_targets = np.array([[p["cursor_x"], p["cursor_y"]] for p in pairs], dtype=float)
        
        unique_targets = []
        target_mapping = []
        for t in true_targets:
            found = False
            for u_idx, u in enumerate(unique_targets):
                if np.allclose(t, u, atol=5.0):
                    target_mapping.append(u_idx)
                    found = True
                    break
            if not found:
                target_mapping.append(len(unique_targets))
                unique_targets.append(t)
                
        unique_targets = np.array(unique_targets, dtype=float)
        num_targets = len(unique_targets)
        target_mapping = np.array(target_mapping, dtype=int)
        
        # Create foveal fsd group clustering labels
        group_labels = np.full(num_targets, -1, dtype=int)
        group_count = 0
        for i in range(num_targets):
            if group_labels[i] == -1:
                group_labels[i] = group_count
                for j in range(i + 1, num_targets):
                    if abs(unique_targets[i][1] - unique_targets[j][1]) < 25.0 and abs(unique_targets[i][0] - unique_targets[j][0]) < 120.0:
                        group_labels[j] = group_count
                group_count += 1
                
        # Word boxes
        word_boxes = np.zeros((num_targets, 4))
        for idx, ut in enumerate(unique_targets):
            word_boxes[idx] = [ut[0] - 45, ut[1] - 15, ut[0] + 45, ut[1] + 15]
            
        base_cm = np.ones(num_targets) / num_targets
        
        loaded_trials.append({
            "subject_id": SUBJECT_METRIC_MAP.get(gt_file.name, {"id": "unknown"})["id"],
            "gaze_seq": gaze_seq,
            "word_boxes": word_boxes,
            "base_cm": base_cm,
            "target_mapping": target_mapping,
            "group_labels": group_labels
        })
        
    return loaded_trials

def run_grid_search():
    trials = load_subject_data()
    if not trials:
        print("❌ No subject data loaded.")
        return
        
    print(f"Loaded {len(trials)} subjects. Starting grid search optimization...")
    
    # Grid spaces
    sigma_gaze_options = [15.0, 25.0, 35.0, 45.0, 55.0]
    alpha_cm_options = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    best_overall_acc = -1.0
    best_config = None
    
    results = []
    
    for sig in sigma_gaze_options:
        for alpha in alpha_cm_options:
            strict_accs = []
            group_accs = []
            
            for tr in trials:
                num_targets = len(tr["word_boxes"])
                t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3)
                transition_matrix = t_pom.build_matrix(num_targets, tr["base_cm"], word_boxes=tr["word_boxes"])
                
                try:
                    path, _ = viterbi_gaze_decode(
                        tr["gaze_seq"], tr["word_boxes"], tr["base_cm"], transition_matrix,
                        sigma_gaze=[sig, sig * 0.75], use_ovp=True, is_L2=True, alpha_cm=alpha
                    )
                    hits = np.sum(path == tr["target_mapping"])
                    acc = hits / len(tr["gaze_seq"])
                    
                    # Group hits
                    group_hits = 0
                    for t, pred_idx in enumerate(path):
                        true_idx = tr["target_mapping"][t]
                        if tr["group_labels"][pred_idx] == tr["group_labels"][true_idx]:
                            group_hits += 1
                    group_acc = group_hits / len(tr["gaze_seq"])
                except Exception:
                    acc = 0.0
                    group_acc = 0.0
                    
                strict_accs.append(acc)
                group_accs.append(group_acc)
                
            mean_strict = np.mean(strict_accs)
            mean_group = np.mean(group_accs)
            
            results.append({
                "sigma_gaze": sig,
                "alpha_cm": alpha,
                "mean_strict_acc": mean_strict,
                "mean_group_acc": mean_group
            })
            
            # We optimize for Group Snapping Accuracy first due to horizontal webcam noise
            if mean_group > best_overall_acc:
                best_overall_acc = mean_group
                best_config = {
                    "sigma_gaze": sig,
                    "alpha_cm": alpha,
                    "mean_strict_acc": mean_strict,
                    "mean_group_acc": mean_group
                }
                
    print("\n" + "="*50)
    print("🏆 OPTIMAL HYPERPARAMETER CONFIGURATION:")
    print(f"  * Sigma Gaze (Search Radius): {best_config['sigma_gaze']} px")
    print(f"  * Alpha Cognitive Mass:     {best_config['alpha_cm']}")
    print(f"  * Best Mean Group Accuracy:  {best_config['mean_group_acc']*100:.2f}%")
    print(f"  * Strict Snapping Accuracy:  {best_config['mean_strict_acc']*100:.2f}%")
    print("="*50 + "\n")
    
    # Save the optimization table
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "hyperparameter_optimization_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🔬 LexiGaze Decoder Hyperparameter Optimization Report\n\n")
        f.write("This report presents grid search evaluation results over Viterbi snapping parameters using empirical subject webcam trials.\n\n")
        f.write("## 1. Grid Search Performance Table\n\n")
        f.write("| Sigma Gaze (px) | Alpha CM (Prior Weight) | Mean Strict Acc (%) | Mean Group Acc (%) |\n")
        f.write("|---|---|---|---|\n")
        for res in sorted(results, key=lambda x: -x["mean_group_acc"]):
            f.write(f"| {res['sigma_gaze']:.1f} | {res['alpha_cm']:.1f} | {res['mean_strict_acc']*100:.2f}% | {res['mean_group_acc']*100:.2f}% |\n")
            
        f.write("\n## 2. Conclusion & Integration\n")
        f.write(f"- The mathematically optimal configuration is **Sigma Gaze = {best_config['sigma_gaze']} px** and **Alpha CM = {best_config['alpha_cm']}**.\n")
        f.write("- Setting a moderate foveal search radius filters out minor gaze deviations while preserving layout sequence alignment.\n")
        
    print(f"Saved optimization report to {report_path}")

if __name__ == "__main__":
    run_grid_search()
