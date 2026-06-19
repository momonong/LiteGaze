#!/usr/bin/env python3
# scripts/import_offline_calibration.py
import sys
import os
import json
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure local packages in the root can be found
ROOT = Path(__file__).parent.parent
CHENGHAO_DIR = ROOT
sys.path.insert(0, str(ROOT))

from gaze_core.sample_store import ensure_sessions_dir, create_session, get_preprocessor
from gaze_core.model_registry import ensure_runs_dir
from gaze_core.training import train_placeholder
from demo_routes import extract_all_targets_sequential

def main():
    if len(sys.argv) < 3:
        print("Usage: uv run python scripts/import_offline_calibration.py <video_path> <timeline_path> [participant_id]")
        sys.exit(1)
        
    video_file = Path(sys.argv[1]).resolve()
    timeline_file = Path(sys.argv[2]).resolve()
    
    if not video_file.exists():
        print(f"Error: Video file not found at: {video_file}")
        sys.exit(1)
    if not timeline_file.exists():
        print(f"Error: Timeline file not found at: {timeline_file}")
        sys.exit(1)
        
    with open(timeline_file, 'r', encoding='utf-8') as f:
        timeline = json.load(f)
        
    participant_id = sys.argv[3] if len(sys.argv) > 3 else timeline.get("participant_id", "anonymous")
    targets = timeline.get("targets", [])
    viewport_width = float(timeline.get("viewport_width", 1920.0))
    viewport_height = float(timeline.get("viewport_height", 1080.0))
    
    print(f"==================================================")
    print(f"  LexiGaze Offline Calibration Importer")
    print(f"==================================================")
    print(f"Participant ID: {participant_id}")
    print(f"Video File    : {video_file.name}")
    print(f"Timeline File : {timeline_file.name}")
    print(f"Gaze Targets  : {len(targets)}")
    print(f"==================================================")
    
    # 1. Create a calibration session directory
    session_res = create_session(CHENGHAO_DIR, participant_id)
    if not session_res.get("ok"):
        print("Error: Failed to create session directory structure.")
        sys.exit(1)
        
    session_id = session_res["session_id"]
    sessions_dir = ensure_sessions_dir(CHENGHAO_DIR)
    session_dir = sessions_dir / session_id
    
    # 2. Copy raw video file to session directory
    video_path = session_dir / f"raw_video{video_file.suffix}"
    import shutil
    shutil.copy2(video_file, video_path)
    
    # 3. Open video using OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: OpenCV failed to open video file.")
        sys.exit(1)
        
    preprocessor = get_preprocessor()
    manifest_path = session_dir / "manifest.jsonl"
    processed_count = 0
    failed_count = 0
    
    print("Processing video frames (extracting face landmarks sequentially)...")
    extracted_results = extract_all_targets_sequential(cap, targets, preprocessor)
    cap.release()

    for i, target in enumerate(targets):
        frame, processed = extracted_results[i]
        timestamp_ms = float(target.get("timestamp_ms", 0.0))
        phase = target.get("phase", "calibration")
        point_index = int(target.get("point_index", 0))
        repeat_index = int(target.get("repeat_index", 0))
        target_x = float(target.get("target_x", 0.0))
        target_y = float(target.get("target_y", 0.0))
        
        target_x_norm = target.get("target_x_norm")
        if target_x_norm is None:
            target_x_norm = (target_x / viewport_width) * 2.0 - 1.0
        target_y_norm = target.get("target_y_norm")
        if target_y_norm is None:
            target_y_norm = (target_y / viewport_height) * 2.0 - 1.0
            
        stem = f"{processed_count:06d}_{phase}_{point_index:02d}_{repeat_index:02d}"
        
        if frame is not None and processed is not None:
            raw_path = session_dir / "raw" / f"{stem}.jpg"
            crop_path = session_dir / "crop" / f"{stem}.jpg"
            norm_path = session_dir / "normalized_face" / f"{stem}.jpg"
            
            cv2.imwrite(str(raw_path), frame)
            cv2.imwrite(str(crop_path), processed.crop_bgr)
            cv2.imwrite(str(norm_path), processed.image_bgr)
            
            record = {
                "ok": True,
                "sample_index": processed_count,
                "phase": phase,
                "point_index": point_index,
                "repeat_index": repeat_index,
                "target_x": target_x,
                "target_y": target_y,
                "target_x_norm": target_x_norm,
                "target_y_norm": target_y_norm,
                "viewport_width": viewport_width,
                "viewport_height": viewport_height,
                "screen_width": target.get("screen_width", viewport_width),
                "screen_height": target.get("screen_height", viewport_height),
                "raw_path": raw_path.relative_to(session_dir).as_posix(),
                "crop_path": crop_path.relative_to(session_dir).as_posix(),
                "normalized_face_path": norm_path.relative_to(session_dir).as_posix(),
                "head_pose_pitch_yaw": processed.head_pose_pitch_yaw.tolist(),
                "face_bbox": processed.face_bbox,
                "created_at_unix": time.time(),
                "extracted_from_video": True,
                "extracted_timestamp_ms": timestamp_ms
            }
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_count += 1
        else:
            failed_count += 1
    print(f"Face extraction completed. Success: {processed_count}, Failed: {failed_count}.")
    
    if processed_count == 0:
        print("Error: Failed to extract any face landmarks from the video.")
        sys.exit(1)
        
    # 4. Train personalization regression model
    output_model_name = f"{participant_id}_video_model"
    train_payload = {
        "data_session_id": session_id,
        "output_model_name": output_model_name,
        "base_model_name": "0"
    }
    
    print(f"Training polynomial personalization regression model '{output_model_name}'...")
    train_res, train_status = train_placeholder(CHENGHAO_DIR, train_payload)
    
    if train_status == 200:
        print("\n🎉 Import & Personalization Autotraining Complete!")
        print(f"Model ID    : {output_model_name}")
        print(f"Mean Error  : {train_res.get('best_val_px_error', 0.0):.2f} px")
        print(f"Noise Level : {train_res.get('noise_level', 0.0):.2f}")
    else:
        print(f"Error: Personalization model training failed (Status {train_status}): {train_res}")

if __name__ == "__main__":
    main()
