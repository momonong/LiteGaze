# chenghao/demo_routes.py
from __future__ import annotations

import json
import time
from pathlib import Path
from flask import Blueprint, jsonify, request
import cv2
import numpy as np

from gaze_core.model_registry import ensure_runs_dir, model_path
from gaze_core.sample_store import ensure_sessions_dir, create_session
from gaze_core.training import train_placeholder

ROOT = Path(__file__).parent
demo_bp = Blueprint("demo", __name__, url_prefix="/api/demo")


def extract_best_face_frame(cap, target_time_ms, preprocessor, max_search_ms=400, search_step_ms=33):
    # Try exact timestamp first
    cap.set(cv2.CAP_PROP_POS_MSEC, target_time_ms)
    ret, frame = cap.read()
    if ret:
        try:
            processed = preprocessor.process(frame)
            if processed and processed.image_bgr is not None:
                return frame, processed
        except Exception:
            pass

    # Search window around target timestamp
    offsets = []
    for ms in range(search_step_ms, max_search_ms + 1, search_step_ms):
        offsets.extend([ms, -ms])

    for offset in offsets:
        curr_time = target_time_ms + offset
        if curr_time < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, curr_time)
        ret, frame = cap.read()
        if ret:
            try:
                processed = preprocessor.process(frame)
                if processed and processed.image_bgr is not None:
                    return frame, processed
            except Exception:
                pass
                
    return None, None


@demo_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "backend": "lexigaze-demo-router", "status": "active"})


@demo_bp.route("/upload_video", methods=["POST"])
def upload_video():
    # 1. Verify files and timeline metadata
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "Missing video file in request ('video')"}), 400
        
    video_file = request.files["video"]
    timeline_str = request.form.get("timeline")
    
    if not timeline_str:
        return jsonify({"ok": False, "error": "Missing timeline JSON string"}), 400
        
    try:
        timeline = json.loads(timeline_str)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Invalid timeline JSON format: {exc}"}), 400

    participant_id = timeline.get("participant_id", "anonymous")
    targets = timeline.get("targets", [])
    viewport_width = float(timeline.get("viewport_width", 1920.0))
    viewport_height = float(timeline.get("viewport_height", 1080.0))

    if not targets:
        return jsonify({"ok": False, "error": "Timeline targets list is empty"}), 400

    # 2. Create a calibration session folder
    session_res = create_session(ROOT, participant_id)
    if not session_res.get("ok"):
        return jsonify({"ok": False, "error": "Failed to create session directory structures"}), 500
        
    session_id = session_res["session_id"]
    sessions_dir = ensure_sessions_dir(ROOT)
    session_dir = sessions_dir / session_id

    # 3. Save raw video file to session directory for offline verification / retraining
    video_extension = Path(video_file.filename).suffix or ".webm"
    video_path = session_dir / f"raw_video{video_extension}"
    video_file.save(str(video_path))
    
    # 4. Open video using OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return jsonify({"ok": False, "error": f"Failed to open video file: {video_path.name}"}), 400

    # Read preprocessor (MediaPipe UniGazePreprocessor)
    try:
        from gaze_core.sample_store import get_preprocessor
        preprocessor = get_preprocessor()
    except Exception as exc:
        cap.release()
        return jsonify({"ok": False, "error": f"Failed to initialize MediaPipe preprocessor: {exc}"}), 500

    manifest_path = session_dir / "manifest.jsonl"
    processed_count = 0
    failed_count = 0

    # 5. Extract frames and process faces
    for i, target in enumerate(targets):
        timestamp_ms = float(target.get("timestamp_ms", 0.0))
        phase = target.get("phase", "calibration")
        point_index = int(target.get("point_index", 0))
        repeat_index = int(target.get("repeat_index", 0))
        
        target_x = float(target.get("target_x", 0.0))
        target_y = float(target.get("target_y", 0.0))
        
        target_x_norm = target.get("target_x_norm")
        if target_x_norm is None:
            target_x_norm = (target_x / viewport_width) * 2.0 - 1.0
        else:
            target_x_norm = float(target_x_norm)
            
        target_y_norm = target.get("target_y_norm")
        if target_y_norm is None:
            target_y_norm = (target_y / viewport_height) * 2.0 - 1.0
        else:
            target_y_norm = float(target_y_norm)

        # Robust extraction with sliding window search around target timestamp
        frame, processed = extract_best_face_frame(cap, timestamp_ms, preprocessor)
        
        stem = f"{processed_count:06d}_{phase}_{point_index:02d}_{repeat_index:02d}"
        
        if frame is not None and processed is not None:
            raw_path = session_dir / "raw" / f"{stem}.jpg"
            crop_path = session_dir / "crop" / f"{stem}.jpg"
            norm_path = session_dir / "normalized_face" / f"{stem}.jpg"
            
            # Save image files
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

    cap.release()

    if processed_count == 0:
        return jsonify({
            "ok": False, 
            "error": "No face samples could be extracted from the video. Please check your video layout, lighting, or webcam quality."
        }), 400

    # 6. Train personalization model from the newly extracted dataset
    output_model_name = f"{participant_id}_video_model"
    train_payload = {
        "data_session_id": session_id,
        "output_model_name": output_model_name,
        "base_model_name": "0"
    }
    
    train_res, train_status = train_placeholder(ROOT, train_payload)

    return jsonify({
        "ok": True,
        "session_id": session_id,
        "processed_samples": processed_count,
        "failed_samples": failed_count,
        "model_name": output_model_name,
        "training": train_res,
        "video_saved_path": video_path.name
    })
