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


def get_participant_profile(participant_id: str) -> tuple[float, float]:
    """
    Search docs/cognitive_reports/ for the most recent report of the participant
    to retrieve their recorded WPM and regression count.
    Returns (wpm, regression_rate). Defaults to (40.0, 0.15) if not found.
    """
    if not participant_id or participant_id == "anonymous":
        return 40.0, 0.15
    try:
        import glob
        import re
        reports_dir = ROOT / "docs" / "cognitive_reports"
        files = glob.glob(str(reports_dir / f"{participant_id}_*.md"))
        if not files:
            return 40.0, 0.15
        # Get the most recent report by sorting
        latest_file = sorted(files)[-1]
        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract WPM
        wpm = 40.0
        m_wpm = re.search(r'閱讀速度 \(WPM\)\*\* \| `(.*?)`', content)
        if m_wpm:
            val = m_wpm.group(1).replace('words/min', '').replace('字/分', '').strip()
            wpm = float(val)
            
        # Extract Regressions
        reg_rate = 0.15
        m_reg = re.search(r'回看次數 \(Regression Count\)\*\* \| `(.*?)`', content)
        if m_reg:
            val = m_reg.group(1).replace('次', '').strip()
            reg_rate = int(val) / 20.0
            
        return wpm, reg_rate
    except Exception:
        return 40.0, 0.15


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

    # Calculate adaptive foveal snapping boundaries based on past user proficiency metrics
    wpm, reg_rate = get_participant_profile(participant_id)
    wpm_factor = min(1.0, max(0.0, (wpm - 80.0) / (200.0 - 80.0)))
    reg_factor = min(1.0, max(0.0, (reg_rate - 0.05) / (0.25 - 0.05)))
    proficiency = 0.7 * wpm_factor + 0.3 * (1.0 - reg_factor)
    
    # Fluent readers (proficiency -> 1): tighten snapping window to prevent false captures (e.g. 35px)
    # Struggling readers (proficiency -> 0): relax snapping window to capture backtracking and regressions (e.g. 55px)
    snap_threshold = 35.0 + (1.0 - proficiency) * 20.0
    outer_threshold = 120.0 + (1.0 - proficiency) * 100.0
    print(f"[gaze_video] Adaptive Snapping: participant={participant_id}, wpm={wpm:.1f}, reg={reg_rate:.2f}, proficiency={proficiency:.2f} -> thresholds: snap<{snap_threshold:.1f}px, outer<{outer_threshold:.1f}px")

    # ── Initialize Cognitive Load Pipeline for Attraction Snapping Prior ──
    cog_lookup = {}
    lang = "en"
    try:
        import re
        from core.cognition import CognitiveLoadPipeline
        # Detect if text contains Chinese
        is_zh = any(re.search(r'[\u4e00-\u9fff]', str(e.get("word", ""))) for e in reading_timeline)
        lang = "zh" if is_zh else "en"
        
        # Extract full reconstructed reading text
        words_seq = [str(e.get("word", "")) for e in reading_timeline if e.get("word")]
        if words_seq:
            text_str = " ".join(words_seq) if lang == "en" else "".join(words_seq)
            # Use BERT model for dynamic surprisal
            pipeline = CognitiveLoadPipeline(model_type='bert', lang=lang)
            cog_result = pipeline.run(text_str)
            word_analysis = cog_result.get("word_analysis", [])
            for item in word_analysis:
                w_key = item.get("word", "").lower().strip()
                if w_key:
                    cog_lookup[w_key] = item
            print(f"[gaze_video] Successfully ran CognitiveLoadPipeline. Found {len(cog_lookup)} analyzed words.")
    except Exception as exc:
        print(f"[gaze_video] Warning: Failed to run CognitiveLoadPipeline: {exc}")

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

        # Collect raw predicted coordinates and matched frames metadata
        raw_gaze_list = []
        valid_indices = []
        predictions = []

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

            gaze_xy = result.get("screen_xy_px") if (result.get("ok") and status_code == 200) else None
            predictions.append({
                "gaze_xy": gaze_xy,
                "matched_ts_ms": matched_ts_ms,
                "event": event,
                "confidence_hint": confidence_hint
            })
            if gaze_xy:
                raw_gaze_list.append(gaze_xy)
                valid_indices.append(timeline_idx)

            timeline_idx += 1

        cap.release()

        # ── Port of GECO POM + EMdrift Auto-Calibration Decoding ──
        decoded_via_viterbi = False
        if len(raw_gaze_list) >= 5:
            try:
                raw_gaze_sequence = np.array(raw_gaze_list, dtype=float)
                
                # Construct synthetic word boxes and base cognitive mass from the timeline
                word_boxes = []
                base_cm = []
                for event in reading_timeline:
                    vx = float(event.get("viewport_x", 0.0))
                    vy = float(event.get("viewport_y", 0.0))
                    # 90x30px standard bounding box size
                    word_boxes.append([vx - 45.0, vy - 15.0, vx + 45.0, vy + 15.0])
                    
                    word = str(event.get("word", ""))
                    clean_w = word.strip(".,;:?!'\"()").lower()
                    cog_item = cog_lookup.get(clean_w) if cog_lookup else None
                    load = float(cog_item["load_score"]) if (cog_item and "load_score" in cog_item) else 0.5
                    base_cm.append(load)
                    
                word_boxes = np.array(word_boxes, dtype=float)
                base_cm = np.array(base_cm, dtype=float)
                
                from scripts.geco.core.transition_model import PsycholinguisticTransitionMatrix
                from scripts.geco.core.em_calibration import AutoCalibratingDecoder
                
                # Build psycholinguistic transition priors
                t_pom = PsycholinguisticTransitionMatrix(sigma_fwd=0.8, sigma_reg=1.5, gamma=0.3)
                transition_matrix = t_pom.build_matrix(len(reading_timeline), base_cm)
                
                # Run dynamic auto-calibration with adaptive proficiency weighting
                alpha_cm_val = float(1.0 - proficiency)
                calibrator = AutoCalibratingDecoder(calibration_window_size=min(30, len(raw_gaze_sequence)))
                final_indices, drift = calibrator.calibrate_and_decode(
                    raw_gaze_sequence, word_boxes, base_cm, transition_matrix,
                    sigma_gaze=[snap_threshold, snap_threshold * 0.75], use_ovp=True, is_L2=True, alpha_cm=alpha_cm_val
                )
                
                print(f"[gaze_video] Viterbi AutoCalibratingDecoder successfully corrected systematic drift: x={drift[0]:.1f}px, y={drift[1]:.1f}px")
                
                # Populate gaze_buffer and gaze_history using corrected Viterbi path target indices
                for i, t_idx in enumerate(valid_indices):
                    corrected_word_idx = final_indices[i]
                    if 0 <= corrected_word_idx < len(reading_timeline):
                        target_event = reading_timeline[corrected_word_idx]
                        word_name = target_event.get("word", "")
                        ts_ms = int(reading_timeline[t_idx].get("timestamp_ms", 0))
                        
                        gaze_history.append({
                            "word":         word_name,
                            "index":        corrected_word_idx,
                            "confidence":   "high",
                            "timestamp_ms": ts_ms,
                        })
                        
                        key = word_name.lower()
                        if key:
                            if key not in gaze_buffer:
                                gaze_buffer[key] = {
                                    "word":           word_name,
                                    "dwell_count":    0,
                                    "fixation_count": 0,
                                    "confidence":     "high",
                                }
                            gaze_buffer[key]["dwell_count"] += 1
                            if prev_word_key != key:
                                gaze_buffer[key]["fixation_count"] += 1
                                prev_word_key = key
                decoded_via_viterbi = True
            except Exception as e:
                print(f"[gaze_video] Warning: Viterbi drift-correction failed: {e}. Falling back to default heuristics.")

        if not decoded_via_viterbi:
            # Fall back to original distance-based snap matching
            for pred in predictions:
                gaze_xy = pred["gaze_xy"]
                matched_ts_ms = pred["matched_ts_ms"]
                event = pred["event"]
                confidence_hint = pred["confidence_hint"]
                word = str(event.get("word", ""))
                word_index = int(event.get("index", -1))
                
                if not gaze_xy:
                    confidence = "low"
                else:
                    gx, gy = gaze_xy
                    exp_x = float(event.get("viewport_x", viewport_width / 2))
                    exp_y = float(event.get("viewport_y", viewport_height / 2))
                    dist  = ((gx - exp_x) ** 2 + (gy - exp_y) ** 2) ** 0.5
                    
                    try:
                        clean_w = word.strip(".,;:?!'\"()").lower()
                        cog_item = cog_lookup.get(clean_w) if cog_lookup else None
                        if cog_item and "load_score" in cog_item:
                            load = float(cog_item["load_score"])
                            mass = 1.0 + load * 3.0
                        elif cog_item and "surprisal" in cog_item:
                            surp = float(cog_item["surprisal"])
                            mass = 1.0 + max(0.0, surp) * 0.20
                        else:
                            from wordfreq import zipf_frequency
                            zipf = zipf_frequency(clean_w, lang) if clean_w else 5.0
                            mass = 1.0 + max(0.0, (5.0 - zipf)) * 0.35 if zipf > 0 else 1.0
                    except Exception:
                        mass = 1.0
                        
                    effective_dist = dist / mass
                    if effective_dist < snap_threshold:
                        confidence = "high"
                    elif effective_dist < outer_threshold:
                        confidence = "medium"
                    else:
                        confidence = "low"
                        
                if confidence in ("high", "medium"):
                    gaze_history.append({
                        "word":         word,
                        "index":        word_index,
                        "confidence":   confidence,
                        "timestamp_ms": int(matched_ts_ms),
                    })
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
