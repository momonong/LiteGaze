"""
scripts/train_all_session_models.py
==============================================================================
Batch Trainer for LexiGaze Personalization Models.
Iterates over all calibration dataset sessions in `data/sessions/`, extracts
facial landmarks and UniGaze feature vectors, fits polynomial regression models
via LOOCV, and saves trained weight JSON artifacts into `examples/models/`.
==============================================================================
"""

import os
import sys
import time
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ensure UTF-8 output encoding for Windows CP950 console safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.gaze_core.training import train_placeholder

def main():
    sessions_dir = PROJECT_ROOT / "data" / "sessions"
    models_dir = PROJECT_ROOT / "examples" / "models"
    output_dir = PROJECT_ROOT / "output"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if not sessions_dir.exists():
        print(f"❌ Sessions directory not found: {sessions_dir}")
        return

    session_folders = [d for d in os.listdir(sessions_dir) if (sessions_dir / d).is_dir()]
    session_folders = sorted(session_folders)
    
    print("=" * 70)
    print("🎯 LEXIGAZE BATCH PERSONALIZATION MODEL TRAINER")
    print("=" * 70)
    print(f"Total calibration sessions found: {len(session_folders)}")
    print(f"Target models directory: {models_dir}")
    print("=" * 70)

    results = []
    
    start_time = time.time()

    for idx, session_id in enumerate(tqdm(session_folders, desc="Training Personalization Models"), 1):
        manifest_path = sessions_dir / session_id / "manifest.jsonl"
        if not manifest_path.exists():
            continue

        model_name = f"model_{session_id}"
        payload = {
            "data_session_id": session_id,
            "base_model_name": "0",
            "output_model_name": model_name
        }

        try:
            res_dict, status_code = train_placeholder(PROJECT_ROOT, payload)
            if status_code == 200 and res_dict.get("ok"):
                results.append({
                    "session_id": session_id,
                    "model_name": res_dict.get("model_name"),
                    "status": "success",
                    "train_samples": res_dict.get("train_samples", 0),
                    "mean_px_error": round(res_dict.get("best_val_px_error", 0.0), 2),
                    "noise_level": round(res_dict.get("noise_level", 0.0), 2),
                })
            else:
                results.append({
                    "session_id": session_id,
                    "model_name": model_name,
                    "status": f"failed: {res_dict.get('error', 'unknown error')}",
                    "train_samples": 0,
                    "mean_px_error": np.nan,
                    "noise_level": np.nan,
                })
        except Exception as exc:
            results.append({
                "session_id": session_id,
                "model_name": model_name,
                "status": f"exception: {exc}",
                "train_samples": 0,
                "mean_px_error": np.nan,
                "noise_level": np.nan,
            })

    elapsed = time.time() - start_time
    df_results = pd.DataFrame(results)
    
    csv_path = output_dir / "batch_calibration_training_summary.csv"
    df_results.to_csv(csv_path, index=False)
    
    success_df = df_results[df_results["status"] == "success"]
    
    print("\n" + "=" * 70)
    print("🏁 BATCH TRAINING COMPLETED")
    print(f"Total Sessions Processed: {len(session_folders)}")
    print(f"Models Successfully Trained: {len(success_df)} / {len(session_folders)}")
    print(f"Total Elapsed Time: {elapsed:.2f} seconds ({elapsed/60.0:.2f} mins)")
    
    if not success_df.empty:
        avg_err = success_df["mean_px_error"].mean()
        avg_noise = success_df["noise_level"].mean()
        print(f"Average Pixel Error Across Models: {avg_err:.2f} px")
        print(f"Average Noise Level Across Models: {avg_noise:.2f} px")
        
    print(f"Training summary saved to: {csv_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
