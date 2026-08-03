from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import cv2
import numpy as np

from .model_registry import model_path

# Thread-safe caching structures
_preprocessor_lock = threading.Lock()
_model_cache_lock = threading.Lock()
_preprocessor = None
_model_cache = {}


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        with _preprocessor_lock:
            if _preprocessor is None:
                from core.unigaze_personalization.preprocess import MediaPipeUniGazePreprocessor
                _preprocessor = MediaPipeUniGazePreprocessor()
    return _preprocessor


def get_base_model():
    with _model_cache_lock:
        base_model = _model_cache.get("base_model")
        if base_model is None:
            import torch
            from core.device import resolve_torch_device
            from core.unigaze_personalization.model import UniGazeFeatureWrapper, load_unigaze_b16

            device = resolve_torch_device()

            try:
                base_model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()
            except Exception:
                if device.type == "cpu":
                    raise
                device = torch.device("cpu")
                base_model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()

            _model_cache["base_model"] = base_model
        return base_model


def predict(root: Path, payload: dict) -> tuple[dict, int]:
    model_name = payload.get("model_name", "before")
    viewport_width = float(payload.get("viewport_width", 0) or 0)
    viewport_height = float(payload.get("viewport_height", 0) or 0)

    # 1. Load model configuration if not using standard frozen baseline
    calibration_file = None
    if model_name != "before":
        calibration_file = model_path(root, model_name)
        if not calibration_file.exists():
            return {"ok": False, "error": f"model {model_name} not found"}, 404

    # 2. Decode the incoming webcam frame
    image_data = payload.get("image_data", "")
    if not image_data:
        return {"ok": False, "error": "missing image_data"}, 400

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data)
        np_arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cannot decode image")
    except Exception as exc:
        return {"ok": False, "error": f"failed to decode image: {exc}"}, 400

    # 3. Extract baseline prediction using UniGaze neural network
    try:
        import torch
        from core.unigaze_personalization.transforms import to_unigaze_tensor

        # Run MediaPipe face detection and facial normalization
        preprocessor = get_preprocessor()
        try:
            processed = preprocessor.process(img)
        except ValueError as ve:
            if "no face detected" in str(ve):
                return {"ok": False, "error": "no face detected in frame"}, 400
            raise

        # Feed image tensor to neural network
        base_model = get_base_model()
        device = next(base_model.parameters()).device
        image_tensor = to_unigaze_tensor(processed.image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            gaze = base_model(image_tensor)
            gaze = gaze.squeeze(0).cpu().tolist()  # [pitch, yaw]

        # 4. Map raw gaze angles to screen coordinates
        if model_name == "before":
            pitch, yaw = gaze[0], gaze[1]
            scale_x = 4.5
            scale_y = 4.5
            pred_x = max(-1.0, min(1.0, yaw * scale_x))
            pred_y = max(-1.0, min(1.0, pitch * scale_y))
            pred_xy = [pred_x, pred_y]
        else:
            with calibration_file.open("r", encoding="utf-8") as handle:
                cal_data = json.load(handle)

            if "stages" in cal_data:
                stages = cal_data["stages"]
            else:
                stages = [{
                    "stage": 1,
                    "W": cal_data["W"],
                    "poly_degree": cal_data.get("poly_degree", 2)
                }]

            p_curr, y_curr = gaze[0], gaze[1]
            for stage_meta in stages:
                W_stage = np.array(stage_meta["W"])
                deg = stage_meta.get("poly_degree", 2)

                if deg == 1:
                    feat = np.array([y_curr, p_curr, 1.0])
                else:
                    feat = np.array([y_curr, p_curr, y_curr * y_curr, p_curr * p_curr, y_curr * p_curr, 1.0])

                pred = feat @ W_stage
                y_curr = float(pred[0])
                p_curr = float(pred[1])

            pred_x = max(-1.0, min(1.0, y_curr))
            pred_y = max(-1.0, min(1.0, p_curr))
            pred_xy = [pred_x, pred_y]

        # Map standardized [-1, 1] coordinates back to viewport pixels if viewport dimensions are provided
        if viewport_width > 0 and viewport_height > 0:
            pixel_x = ((pred_xy[0] + 1.0) * 0.5) * viewport_width
            pixel_y = ((pred_xy[1] + 1.0) * 0.5) * viewport_height
            screen_xy_px = [pixel_x, pixel_y]
        else:
            screen_xy_px = [0.0, 0.0]

        return {
            "ok": True,
            "screen_xy_norm": pred_xy,
            "screen_xy_px": screen_xy_px,
            "gaze_pitch_yaw": gaze,
            "head_pose_pitch_yaw": processed.head_pose_pitch_yaw.tolist(),
            "face_bbox": processed.face_bbox,
            "model_name": model_name,
            "source": "unigaze",
        }, 200

    except Exception as exc:
        return {"ok": False, "error": f"prediction pipeline failed: {exc}"}, 500
