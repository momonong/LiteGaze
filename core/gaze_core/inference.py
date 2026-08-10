from __future__ import annotations

import base64
import binascii
import json
import threading
from pathlib import Path

import cv2
import numpy as np

from .capture_contract import (
    compare_capture_contracts,
    normalize_capture_contract,
    validate_transport_frame,
)
from .calibration_regression import face_geometry_from_bbox
from .model_registry import model_path
from .stage_pipeline import apply_stage_chain
from .torch_runtime import (
    cuda_runtime_available,
    enable_process_wide_cuda_tf32,
    restore_matmul_precision,
)
from .uncertainty import (
    normalize_uncertainty_observation,
    score_samples,
    unavailable_uncertainty,
    uncertainty_status_for_model,
)

# Thread-safe caching structures
_preprocessor_lock = threading.Lock()
_model_cache_lock = threading.Lock()
_preprocessor = None
_model_cache = {}
MAX_INFERENCE_IMAGE_BYTES = 10 * 1024 * 1024


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        with _preprocessor_lock:
            if _preprocessor is None:
                from core.unigaze_personalization.preprocess import (
                    MediaPipeUniGazePreprocessor,
                )

                _preprocessor = MediaPipeUniGazePreprocessor()
    return _preprocessor


def get_base_model(*, allow_cuda: bool = True):
    cache_key = "base_model_auto" if allow_cuda else "base_model_cpu"
    with _model_cache_lock:
        base_model = _model_cache.get(cache_key)
        if base_model is None:
            import torch

            from core.unigaze_personalization.model import (
                UniGazeFeatureWrapper,
                load_unigaze_b16,
            )

            device = "cpu"
            if allow_cuda and cuda_runtime_available(torch):
                try:
                    t = torch.zeros((1, 3, 224, 224), device="cuda")
                    conv = torch.nn.Conv2d(3, 16, kernel_size=16, stride=16).to("cuda")
                    _ = conv(t)
                    device = "cuda"
                except Exception:
                    device = "cpu"

            previous_matmul_precision = enable_process_wide_cuda_tf32(torch, device)
            try:
                base_model = (
                    UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()
                )
            except Exception:
                restore_matmul_precision(torch, previous_matmul_precision)
                device = "cpu"
                base_model = (
                    UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()
                )

            _model_cache[cache_key] = base_model
        return base_model


def predict(root: Path, payload: dict) -> tuple[dict, int]:
    model_name = payload.get("model_name", "before")
    viewport_width = float(payload.get("viewport_width", 0) or 0)
    viewport_height = float(payload.get("viewport_height", 0) or 0)

    # 1. Load model configuration if not using standard frozen baseline
    calibration_file = None
    cal_data = None
    normalized_observed_contract = None
    capture_contract_check = {
        "status": "not_applicable",
        "compatible": None,
        "reasons": [],
        "warnings": [],
    }
    if model_name != "before":
        calibration_file = model_path(root, model_name)
        if not calibration_file.exists():
            return {
                "ok": False,
                "error": f"model {model_name} not found",
                "failure_stage": "model_hard_error",
            }, 404
        try:
            with calibration_file.open("r", encoding="utf-8") as handle:
                cal_data = json.load(handle)
        except (OSError, ValueError, TypeError) as exc:
            return {
                "ok": False,
                "error": f"failed to read model {model_name}: {exc}",
                "failure_stage": "model_hard_error",
            }, 500

        observed_contract = payload.get("capture_contract")
        if cal_data.get("capture_contract") is not None and observed_contract is None:
            return {
                "ok": False,
                "error": (
                    "capture contract is required by this calibrated model; "
                    "recalibration or a contract-aware capture client is required"
                ),
                "failure_code": "capture_contract_mismatch",
                "failure_stage": "capture_hard_error",
                "capture_contract_check": {
                    "status": "mismatch",
                    "compatible": False,
                    "reasons": ["observed_capture_contract_missing"],
                    "warnings": [],
                },
            }, 409
        if observed_contract is not None:
            try:
                normalized_observed_contract = normalize_capture_contract(
                    observed_contract
                )
                capture_contract_check = compare_capture_contracts(
                    cal_data.get("capture_contract"),
                    normalized_observed_contract,
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "error": str(exc),
                    "failure_stage": "capture_hard_error",
                }, 400
            if capture_contract_check["compatible"] is False:
                return {
                    "ok": False,
                    "error": (
                        "capture contract does not match calibration; "
                        "recalibration is recommended before gaze use"
                    ),
                    "failure_code": "capture_contract_mismatch",
                    "failure_stage": "capture_hard_error",
                    "capture_contract_check": capture_contract_check,
                }, 409

    # 2. Decode the incoming webcam frame
    image_data = payload.get("image_data", "")
    if not isinstance(image_data, str) or not image_data:
        return {
            "ok": False,
            "error": "missing image_data",
            "failure_stage": "capture_hard_error",
        }, 400

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data, validate=True)
        if not raw or len(raw) > MAX_INFERENCE_IMAGE_BYTES:
            return {
                "ok": False,
                "error": "image payload size is invalid",
                "failure_stage": "capture_hard_error",
            }, 413
        np_arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cannot decode image")
    except (binascii.Error, TypeError, ValueError) as exc:
        return {
            "ok": False,
            "error": f"failed to decode image: {exc}",
            "failure_stage": "capture_hard_error",
        }, 400

    if normalized_observed_contract is not None:
        try:
            validate_transport_frame(
                normalized_observed_contract,
                frame_width_px=int(img.shape[1]),
                frame_height_px=int(img.shape[0]),
            )
        except ValueError as exc:
            capture_contract_check = {
                **capture_contract_check,
                "status": "mismatch",
                "compatible": False,
                "reasons": [
                    *capture_contract_check.get("reasons", []),
                    "decoded_frame_dimensions_mismatch",
                ],
            }
            return {
                "ok": False,
                "error": str(exc),
                "failure_code": "capture_contract_mismatch",
                "failure_stage": "capture_hard_error",
                "capture_contract_check": capture_contract_check,
            }, 409

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
                return {
                    "ok": False,
                    "error": "no face detected in frame",
                    "failure_code": "no_face_detected",
                    "failure_stage": "attributable_sensor_failure",
                    "capture_contract_check": capture_contract_check,
                    "model_name": model_name,
                    "uncertainty": unavailable_uncertainty(
                        "unavailable_sensor_failure",
                        "no face was detected, so no sensor observation was scored",
                    ),
                }, 400
            raise

        # Feed image tensor to neural network
        base_model = get_base_model(allow_cuda=payload.get("allow_cuda") is not False)
        device = next(base_model.parameters()).device
        image_tensor = to_unigaze_tensor(processed.image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            gaze = base_model(image_tensor)
            gaze = gaze.squeeze(0).cpu().tolist()  # [pitch, yaw]

        # 4. Map raw gaze angles to screen coordinates
        uncertainty = unavailable_uncertainty(
            "unavailable_frozen_baseline",
            "the frozen uncalibrated baseline has no uncertainty_v2 state",
        )
        if model_name == "before":
            pitch, yaw = gaze[0], gaze[1]
            scale_x = 4.5
            scale_y = 4.5
            pred_x = max(-1.0, min(1.0, yaw * scale_x))
            pred_y = max(-1.0, min(1.0, pitch * scale_y))
            pred_xy = [pred_x, pred_y]
        else:
            if "stages" in cal_data:
                stages = cal_data["stages"]
            else:
                stages = [
                    {
                        "stage": 1,
                        "W": cal_data["W"],
                        "poly_degree": cal_data.get("poly_degree", 2),
                    }
                ]

            gaze_rows = np.asarray([gaze], dtype=np.float64)
            production_requires_motion = any(
                stage.get("calibrator_type") == "motion_conditioned_ridge_v1"
                for stage in stages
            )
            head_rows = None
            face_rows = None
            if production_requires_motion:
                head_rows = np.asarray(
                    [processed.head_pose_pitch_yaw], dtype=np.float64
                )
                face_rows = np.asarray(
                    [face_geometry_from_bbox(processed.face_bbox)],
                    dtype=np.float64,
                )
            pred_xy = apply_stage_chain(
                gaze_rows,
                stages,
                head_pitch_yaw=head_rows,
                face_geometry=face_rows,
                clamp=True,
            )[0].astype(float).tolist()

            uncertainty = uncertainty_status_for_model(cal_data)
            if uncertainty["status"] == "scored_no_threshold":
                if viewport_width <= 0 or viewport_height <= 0:
                    uncertainty = unavailable_uncertainty(
                        "unavailable_viewport_missing",
                        (
                            "positive viewport dimensions are required to report "
                            "uncertainty covariance in pixels"
                        ),
                    )
                else:
                    try:
                        score_head_rows = np.asarray(
                            [processed.head_pose_pitch_yaw], dtype=np.float64
                        )
                        score_face_rows = np.asarray(
                            [face_geometry_from_bbox(processed.face_bbox)],
                            dtype=np.float64,
                        )
                        scored = score_samples(
                            gaze_rows,
                            stages,
                            cal_data["uncertainty_v2"]["final_score_state"],
                            [[viewport_width, viewport_height]],
                            head_pitch_yaw=score_head_rows,
                            face_geometry=score_face_rows,
                        )
                        uncertainty = normalize_uncertainty_observation({
                            "schema_version": 1,
                            "status": scored["status"],
                            "definition_sha256": scored["definition_sha256"],
                            "score": float(scored["uncertainty_score"][0]),
                            "components": {
                                name: {
                                    "value": float(values[0]),
                                    "percentile": float(
                                        scored["component_percentiles"][name][0]
                                    ),
                                }
                                for name, values in scored["components"].items()
                            },
                            "jackknife_disagreement_covariance_norm": scored[
                                "jackknife_disagreement_covariance_norm"
                            ][0].astype(float).tolist(),
                            "jackknife_disagreement_covariance_px": scored[
                                "jackknife_disagreement_covariance_px"
                            ][0].astype(float).tolist(),
                            "abstention": {
                                "status": scored["abstention_status"],
                                "threshold": None,
                            },
                        }, viewport=(viewport_width, viewport_height))
                    except Exception:
                        uncertainty = unavailable_uncertainty(
                            "unavailable_invalid_state",
                            (
                                "stored uncertainty state did not satisfy the "
                                "frozen runtime contract"
                            ),
                        )

        # Map standardized [-1, 1] coordinates back to viewport pixels if viewport dimensions are provided
        if viewport_width > 0 and viewport_height > 0:
            pixel_x = ((pred_xy[0] + 1.0) * 0.5) * viewport_width
            pixel_y = ((pred_xy[1] + 1.0) * 0.5) * viewport_height
            screen_xy_px = [pixel_x, pixel_y]
        else:
            screen_xy_px = [0.0, 0.0]

        try:
            uncertainty = normalize_uncertainty_observation(
                uncertainty,
                viewport=(viewport_width, viewport_height)
                if viewport_width > 0 and viewport_height > 0
                else None,
            )
        except Exception:
            uncertainty = unavailable_uncertainty(
                "unavailable_invalid_observation",
                "runtime uncertainty observation failed its stable contract",
            )

        return {
            "ok": True,
            "screen_xy_norm": pred_xy,
            "screen_xy_px": screen_xy_px,
            "gaze_pitch_yaw": gaze,
            "head_pose_pitch_yaw": processed.head_pose_pitch_yaw.tolist(),
            "face_bbox": processed.face_bbox,
            "capture_contract_check": capture_contract_check,
            "model_name": model_name,
            "source": "unigaze",
            "uncertainty": uncertainty,
        }, 200

    except Exception as exc:
        return {
            "ok": False,
            "error": f"prediction pipeline failed: {exc}",
            "failure_code": "prediction_pipeline_failed",
            "failure_stage": "inference_hard_error",
            "capture_contract_check": capture_contract_check,
            "model_name": model_name,
        }, 500
