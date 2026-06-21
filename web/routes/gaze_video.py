"""
gaze_video.py — Offline video-mode gaze analysis for reading sessions.

This endpoint receives:
  - A WebM video of the user's webcam during the reading session
  - A JSON "reading_timeline": list of { word, index, timestamp_ms, viewport_x, viewport_y }
    representing when each word was in the user's viewport (from scroll tracking)

It processes each frame with the gaze model and returns a gaze_events list
compatible with the /api/fuse endpoint (same format as gazeBuffer from gaze_integration.js).
"""
from __future__ import annotations

import json
import base64
import time
import traceback
import tempfile
from pathlib import Path
from flask import Blueprint, jsonify, request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
gaze_video_bp = Blueprint("gaze_video", __name__, url_prefix="/api/gaze")


def _frame_to_base64_jpeg(frame: np.ndarray, quality: int = 70) -> str:
    """Encode an OpenCV BGR frame to base64 JPEG string (data URL format)."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@gaze_video_bp.post("/analyze_reading_video")
def analyze_reading_video():
    """
    Offline reading-session video analysis.

    Multipart form data:
      video          : WebM/MP4 webcam recording
      reading_timeline: JSON string — list of reading events:
        [
          {
            "word":         "the",
            "index":        42,
            "timestamp_ms": 3200,      // ms since recording started
            "viewport_x":   450.5,     // estimated word centre X on screen (px)
            "viewport_y":   312.0,     // estimated word centre Y on screen (px)
            "confidence":   "medium"   // optional hint
          }, ...
        ]
      model_name     : (optional) gaze model name, defaults to "before"
      participant_id : (optional) participant label
      viewport_width : screen width in px
      viewport_height: screen height in px
    """
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "Missing 'video' file"}), 400

    timeline_str = request.form.get("reading_timeline", "[]")
    try:
        reading_timeline = json.loads(timeline_str)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Invalid reading_timeline JSON: {exc}"}), 400

    if not reading_timeline:
        return jsonify({"ok": False, "error": "reading_timeline is empty"}), 400

    model_name      = request.form.get("model_name", "before")
    participant_id  = request.form.get("participant_id", "anonymous")
    viewport_width  = float(request.form.get("viewport_width", 1920))
    viewport_height = float(request.form.get("viewport_height", 1080))

    # ── Save video to a temp file ──────────────────────────────────────────
    video_file = request.files["video"]
    suffix     = Path(video_file.filename or "reading.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        # ── Load gaze predictor ────────────────────────────────────────────
        from core.gaze_core.inference import predict as gaze_predict

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            return jsonify({"ok": False, "error": "Cannot open video file"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Sort reading_timeline chronologically
        reading_timeline.sort(key=lambda e: float(e.get("timestamp_ms", 0)))

        gaze_events = []
        gaze_buffer: dict[str, dict] = {}  # key → {word, dwell_count, fixation_count, confidence}
        gaze_history = []                  # chronological trace of hits
        prev_word_key: str | None = None

        timeline_idx = 0
        frame_idx = 0

        while timeline_idx < len(reading_timeline) and cap.isOpened():
            event = reading_timeline[timeline_idx]
            ts_ms      = float(event.get("timestamp_ms", 0))
            word       = str(event.get("word", ""))
            word_index = int(event.get("index", -1))
            confidence_hint = event.get("confidence", "medium")

            # Sequential frame extraction to avoid OpenCV seek issues on browser-recorded streams
            frame = None
            matched_ts_ms = 0.0

            while cap.isOpened():
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms <= 0 and frame_idx > 0:
                    pos_ms = frame_idx * (1000.0 / fps)

                # Match frame closest to ts_ms (with a half-frame tolerance of ~16.6ms)
                if pos_ms >= ts_ms - 16.6:
                    ret, f = cap.read()
                    if not ret or f is None:
                        break
                    frame = f
                    matched_ts_ms = pos_ms
                    frame_idx += 1
                    break
                else:
                    # Skip frame
                    ret = cap.grab()
                    if not ret:
                        break
                    frame_idx += 1

            if frame is None:
                break

            # Resize to 240×180 for speed (same as gaze_integration.js captureFrame)
            h, w = frame.shape[:2]
            target_w = 240
            target_h = max(1, int(target_w * h / w))
            frame_small = cv2.resize(frame, (target_w, target_h))

            # Convert to base64 JPEG (same format as captureFrame() in browser)
            image_data = _frame_to_base64_jpeg(frame_small, quality=50)

            # Run gaze prediction
            body = {
                "image_data":      image_data,
                "model_name":      model_name,
                "viewport_width":  viewport_width,
                "viewport_height": viewport_height,
            }
            result, status_code = gaze_predict(ROOT, body)

            if not result.get("ok") or status_code != 200:
                # Face not detected
                confidence = "low"
            else:
                # Map gaze position to confidence based on distance from expected word position
                gaze_xy = result.get("screen_xy_px")
                if gaze_xy:
                    gx, gy = gaze_xy
                    exp_x = float(event.get("viewport_x", viewport_width / 2))
                    exp_y = float(event.get("viewport_y", viewport_height / 2))
                    dist  = ((gx - exp_x) ** 2 + (gy - exp_y) ** 2) ** 0.5
                    if dist < 80:
                        confidence = "high"
                    elif dist < 200:
                        confidence = "medium"
                    else:
                        confidence = "low"
                else:
                    confidence = confidence_hint

            # Filter: only record hits with high/medium confidence
            if confidence in ("high", "medium"):
                # Chronological trace
                gaze_history.append({
                    "word":         word,
                    "index":        word_index,
                    "confidence":   confidence,
                    "timestamp_ms": int(matched_ts_ms),
                })

                # Aggregated buffer (same structure as gazeBuffer in JS)
                key = word.lower()
                if key:
                    if key not in gaze_buffer:
                        gaze_buffer[key] = {
                            "word":           word,
                            "dwell_count":    0,
                            "fixation_count": 0,
                            "confidence":     confidence,
                        }
                    gaze_buffer[key]["dwell_count"] += 1
                    if prev_word_key != key:
                        gaze_buffer[key]["fixation_count"] += 1
                        prev_word_key = key
                    rank = {"high": 2, "medium": 1, "low": 0}
                    if rank.get(confidence, 0) > rank.get(gaze_buffer[key]["confidence"], 0):
                        gaze_buffer[key]["confidence"] = confidence

            timeline_idx += 1

        cap.release()

        # Format output identical to flushGazeBuffer() in JS
        gaze_events = [
            {
                "word":           v["word"],
                "confidence":     v["confidence"],
                "dwell_count":    v["dwell_count"],
                "fixation_count": v["fixation_count"],
                "timestamp_ms":   int(time.time() * 1000),
            }
            for v in gaze_buffer.values()
        ]

        return jsonify({
            "ok":           True,
            "participant":  participant_id,
            "model_name":   model_name,
            "event_count":  len(gaze_events),
            "gaze_events":  gaze_events,
            "gaze_history": gaze_history,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)
