#!/usr/bin/env python3
import json
import os
from pathlib import Path

def generate_mock_data():
    print("Generating mock data for LexiGaze demo...")
    
    # Paths are relative to the project root
    ROOT = Path(__file__).parent.parent / "chenghao"
    
    # 1. Ensure runs and sessions directories exist
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    sessions_base_dir = ROOT / "data" / "sessions"
    sessions_base_dir.mkdir(parents=True, exist_ok=True)
    
    mock_session_id = "mock_calibration_session"
    session_dir = sessions_base_dir / mock_session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    for sub in ["raw", "crop", "normalized_face"]:
        (session_dir / sub).mkdir(parents=True, exist_ok=True)
        
    print(f"Created directories in {ROOT}")

    # 2. Write mock personalization model
    model_data = {
        "name": "mock_user_model",
        "created_at": "2026-06-19T00:00:00",
        "data_session_id": mock_session_id,
        "stages": [
            {
                "stage": 1,
                "W": [
                    [0.85, 0.05],
                    [0.05, 0.85],
                    [0.01, 0.01]
                ],
                "poly_degree": 1,
                "mean_px_error": 12.5
            }
        ],
        "mean_px_error": 12.5,
        "noise_level": 4.8,
        "train_samples": 18
    }
    model_path = runs_dir / "mock_user_model.json"
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)
    print(f"Generated mock trained model: {model_path}")

    # 3. Write session.json for the calibration dataset
    session_meta = {
        "session_id": mock_session_id,
        "participant_id": "mock_user",
        "created_at": "20260619_000000"
    }
    with open(session_dir / "session.json", "w", encoding="utf-8") as f:
        json.dump(session_meta, f, ensure_ascii=False, indent=2)
        
    # 4. Write manifest.jsonl with simulated calibration points
    # 9-point grid target positions in normalized space (-1.0 to 1.0)
    grid_points = [
        (-0.8, -0.8), (0.0, -0.8), (0.8, -0.8),
        (-0.8,  0.0), (0.0,  0.0), (0.8,  0.0),
        (-0.8,  0.8), (0.0,  0.8), (0.8,  0.8)
    ]
    
    manifest_path = session_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for repeat in range(2):
            for idx, (tx, ty) in enumerate(grid_points):
                sample_idx = repeat * len(grid_points) + idx
                stem = f"{sample_idx:06d}_calibration_{idx:02d}_{repeat:02d}"
                
                # Write dummy files to avoid missing file errors
                (session_dir / "raw" / f"{stem}.jpg").write_bytes(b"")
                (session_dir / "crop" / f"{stem}.jpg").write_bytes(b"")
                (session_dir / "normalized_face" / f"{stem}.jpg").write_bytes(b"")
                
                record = {
                    "ok": True,
                    "sample_index": sample_idx,
                    "phase": "calibration",
                    "point_index": idx,
                    "repeat_index": repeat,
                    "target_x": (tx + 1.0) * 0.5 * 1920.0,
                    "target_y": (ty + 1.0) * 0.5 * 1080.0,
                    "target_x_norm": tx,
                    "target_y_norm": ty,
                    "viewport_width": 1920.0,
                    "viewport_height": 1080.0,
                    "screen_width": 1920.0,
                    "screen_height": 1080.0,
                    "raw_path": f"raw/{stem}.jpg",
                    "crop_path": f"crop/{stem}.jpg",
                    "normalized_face_path": f"normalized_face/{stem}.jpg",
                    "head_pose_pitch_yaw": [0.01, -0.02],
                    "face_bbox": [480, 270, 960, 810],
                    "created_at_unix": 1749912000.0 + sample_idx
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
    print(f"Generated calibration dataset: {manifest_path} (18 samples)")
    print("Mock data setup complete. You can now start the server and run full-pipeline simulations!")

if __name__ == "__main__":
    generate_mock_data()
