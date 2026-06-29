"""
scripts/evaluate_empirical_fusion.py
══════════════════════════════════════════════════════════════════════════════
Runs a comparative experiment on the 5 collected subjects (subject001-subject005)
comparing baseline direct foveal snapping against our newly integrated
Adaptive POM Viterbi Auto-Calibration Decoder.
"""

import os
import sys
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd

# Bootstrap ROOT path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
from scripts.geco.core.em_calibration import AutoCalibratingDecoder

# Mapping from ground truth files to subject IDs (chronologically matching)
SUBJECT_METRIC_MAP = {
    "gt_1782692940276.json": {"id": "subject001", "wpm": 73.7, "reg": 6 / 20.0},
    "gt_1782693148405.json": {"id": "subject002", "wpm": 43.5, "reg": 9 / 20.0},
    "gt_1782693226352.json": {"id": "subject003", "wpm": 33.3, "reg": 1 / 20.0},
    "gt_1782693294940.json": {"id": "subject004", "wpm": 29.0, "reg": 2 / 20.0},
    "gt_1782693357844.json": {"id": "subject005", "wpm": 31.6, "reg": 3 / 20.0},
}

def dynamic_sliding_window_decode(gaze_seq, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp, is_L2, alpha_cm, window_size=30):
    """
    Proposed Method: Dynamic Sliding Window EM Drift Correction.
    Estimates systematic drift vector at each step using a sliding window median of errors.
    """
    from scripts.geco.core.viterbi_decoder import viterbi_gaze_decode
    from scripts.geco.core.dynamic_field import DynamicCognitiveField
    
    # Pass 1: Run Viterbi once to get initial raw alignment path
    initial_path, _ = viterbi_gaze_decode(gaze_seq, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp, is_L2=is_L2, alpha_cm=alpha_cm)
    
    # Extract centers
    if use_ovp:
        dfield = DynamicCognitiveField(word_boxes, base_cm, use_ovp=True, is_L2=is_L2, alpha_cm=alpha_cm)
        word_centers = dfield.word_centers
    else:
        word_centers = np.array([[ (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ] for box in word_boxes])
        
    predicted_centers = word_centers[initial_path]
    errors = gaze_seq - predicted_centers
    
    # Calculate rolling median of errors to correct drift dynamically over time
    corrected_gaze = np.copy(gaze_seq)
    n = len(gaze_seq)
    for t in range(n):
        start = max(0, t - window_size // 2)
        end = min(n, t + window_size // 2 + 1)
        
        # Estimate drift median for local window
        drift_x = np.nanmedian(errors[start:end, 0])
        drift_y = np.nanmedian(errors[start:end, 1])
        
        # Correct local gaze coordinates
        if np.nanstd(errors[start:end]) < 150:
            corrected_gaze[t] -= np.array([drift_x, drift_y])
            
    # Pass 2: Re-decode on dynamically corrected coordinates
    final_path, _ = viterbi_gaze_decode(corrected_gaze, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp, is_L2=is_L2, alpha_cm=alpha_cm)
    return final_path


def main():
    print("=" * 70)
    print("🔬 LexiGaze Empirical Snapping & Drift Correction Experiment")
    print("=" * 70)

    gt_dir = ROOT / "data" / "ground_truth"
    gt_files = glob.glob(str(gt_dir / "gt_*.json"))

    if not gt_files:
        print("❌ Error: No ground-truth subject files found in data/ground_truth/")
        return

    results = []

    for filepath in sorted(gt_files):
        filename = os.path.basename(filepath)
        sub_info = SUBJECT_METRIC_MAP.get(filename, {"id": "unknown", "wpm": 40.0, "reg": 0.15})
        subject_id = sub_info["id"]

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs = data.get("pairs", [])
        if not pairs:
            continue

        print(f"\nProcessing {subject_id} ({filename}) - {len(pairs)} gaze samples...")

        # Extract sequence
        gaze_seq = np.array([[p["gaze_x"], p["gaze_y"]] for p in pairs], dtype=float)
        true_targets = np.array([[p["cursor_x"], p["cursor_y"]] for p in pairs], dtype=float)

        # Form unique word boxes from true targets to simulate text layout
        # In actual reading, words are static on the screen. Let's find unique target coordinates.
        unique_targets = []
        target_mapping = [] # Map each sample to unique target index
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

        # Cluster unique targets into Foveal Groups (within 25px vertically and 120px horizontally)
        group_labels = np.full(num_targets, -1, dtype=int)
        group_count = 0
        for i in range(num_targets):
            if group_labels[i] == -1:
                group_labels[i] = group_count
                for j in range(i + 1, num_targets):
                    if abs(unique_targets[i][1] - unique_targets[j][1]) < 25.0 and abs(unique_targets[i][0] - unique_targets[j][0]) < 120.0:
                        group_labels[j] = group_count
                group_count += 1

        # Create word boxes around unique targets
        word_boxes = np.zeros((num_targets, 4))
        for idx, ut in enumerate(unique_targets):
            word_boxes[idx] = [ut[0] - 45, ut[1] - 15, ut[0] + 45, ut[1] + 15]

        # 1. Baseline snap (Direct Euclidean distance to closest target)
        baseline_hits = 0
        baseline_group_hits = 0
        for i, gaze in enumerate(gaze_seq):
            dists = np.linalg.norm(unique_targets - gaze, axis=1)
            pred_idx = np.argmin(dists)
            if pred_idx == target_mapping[i]:
                baseline_hits += 1
            if group_labels[pred_idx] == group_labels[target_mapping[i]]:
                baseline_group_hits += 1
        baseline_acc = baseline_hits / len(gaze_seq)
        baseline_group_acc = baseline_group_hits / len(gaze_seq)

        # 2. Viterbi POM + EM drift-correction
        # Compute proficiency & alpha_cm
        wpm = sub_info["wpm"]
        reg_rate = sub_info["reg"]
        wpm_factor = min(1.0, max(0.0, (wpm - 80.0) / (200.0 - 80.0)))
        reg_factor = min(1.0, max(0.0, (reg_rate - 0.05) / (0.25 - 0.05)))
        proficiency = 0.7 * wpm_factor + 0.3 * (1.0 - reg_factor)
        alpha_cm = 1.0 - proficiency

        # Base cognitive mass (simulated with uniform default since we don't have text words here)
        base_cm = np.ones(num_targets) / num_targets

        # Snapping thresholds
        snap_threshold = 35.0 + (1.0 - proficiency) * 20.0

        try:
            t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3)
            transition_matrix = t_pom.build_matrix(num_targets, base_cm, word_boxes=word_boxes)

            calibrator = AutoCalibratingDecoder(calibration_window_size=min(30, len(gaze_seq)))
            final_indices, drift = calibrator.calibrate_and_decode(
                gaze_seq, word_boxes, base_cm, transition_matrix,
                sigma_gaze=[snap_threshold, snap_threshold * 0.75], use_ovp=True, is_L2=True, alpha_cm=alpha_cm
            )

            viterbi_hits = 0
            viterbi_group_hits = 0
            for i, pred_idx in enumerate(final_indices):
                if pred_idx == target_mapping[i]:
                    viterbi_hits += 1
                if group_labels[pred_idx] == group_labels[target_mapping[i]]:
                    viterbi_group_hits += 1
            viterbi_acc = viterbi_hits / len(gaze_seq)
            viterbi_group_acc = viterbi_group_hits / len(gaze_seq)
            drift_str = f"({drift[0]:.1f}px, {drift[1]:.1f}px)"
        except Exception as e:
            viterbi_acc = 0.0
            viterbi_group_acc = 0.0
            drift_str = "Error"
            print(f"  Viterbi failed: {e}")

        # 3. Dynamic Sliding Window drift-correction
        try:
            dynamic_indices = dynamic_sliding_window_decode(
                gaze_seq, word_boxes, base_cm, transition_matrix,
                sigma_gaze=[snap_threshold, snap_threshold * 0.75], use_ovp=True, is_L2=True, alpha_cm=alpha_cm, window_size=30
            )
            
            dynamic_hits = 0
            dynamic_group_hits = 0
            for i, pred_idx in enumerate(dynamic_indices):
                if pred_idx == target_mapping[i]:
                    dynamic_hits += 1
                if group_labels[pred_idx] == group_labels[target_mapping[i]]:
                    dynamic_group_hits += 1
            dynamic_acc = dynamic_hits / len(gaze_seq)
            dynamic_group_acc = dynamic_group_hits / len(gaze_seq)
        except Exception as e:
            dynamic_acc = 0.0
            dynamic_group_acc = 0.0
            print(f"  Dynamic Sliding Window failed: {e}")

        improvement = viterbi_acc - baseline_acc
        group_improvement = viterbi_group_acc - baseline_group_acc
        dyn_improvement = dynamic_group_acc - baseline_group_acc
        
        print(f"  Baseline Accuracy:      {baseline_acc*100:.2f}% (Group: {baseline_group_acc*100:.2f}%)")
        print(f"  Adaptive Viterbi:       {viterbi_acc*100:.2f}% (Group: {viterbi_group_acc*100:.2f}%) (Drift corrected: {drift_str})")
        print(f"  Dynamic Sliding Viterbi:{dynamic_acc*100:.2f}% (Group: {dynamic_group_acc*100:.2f}%)")
        print(f"  Group Improvement:      {group_improvement*100:+.2f}% (Dynamic Group: {dyn_improvement*100:+.2f}%)")

        results.append({
            "Subject ID": subject_id,
            "WPM": wpm,
            "Proficiency": round(proficiency, 3),
            "Baseline Group Acc": f"{baseline_group_acc*100:.2f}%",
            "Static Viterbi Group Acc": f"{viterbi_group_acc*100:.2f}%",
            "Dynamic Sliding Group Acc": f"{dynamic_group_acc*100:.2f}%",
            "Systematic Drift (Y)": drift_str,
            "Dynamic vs Baseline Improvement": f"{dyn_improvement*100:+.2f}%",
        })

    print("\n" + "=" * 70)
    print("📊 COMPARATIVE SUMMARY")
    print("=" * 70)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

    # Save report
    report_path = ROOT / "output" / "empirical_evaluation_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🔬 Real Subject Snapping Accuracy Experiment\n\n")
        f.write("This report evaluates the accuracy of gaze-to-word snapped mapping comparing baseline Euclidean snapping against the newly integrated **Adaptive POM Viterbi Auto-Calibration Decoder**.\n\n")
        
        # Write markdown table manually
        f.write("| " + " | ".join(df_res.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df_res.columns)) + " |\n")
        for _, row in df_res.iterrows():
            f.write("| " + " | ".join(str(val) for val in row.values) + " |\n")
            
        f.write("\n\n*Evaluation conducted on real participant ground-truth coordinates (`subject001`-`subject005`).*")

    print(f"\nSaved report to {report_path}")

if __name__ == "__main__":
    main()
