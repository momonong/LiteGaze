from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .calibration_regression import (
    MOTION_FEATURE_NAMES,
    face_geometry_from_bbox,
    fit_best_stage,
    fit_standardized_ridge,
    motion_challenger_decision,
    motion_conditioned_features,
    standardized_design,
)
from .model_registry import clean_model_name, model_path
from .motion_robustness import audit_payload, load_motion_samples
from .sample_store import ensure_sessions_dir
from .torch_runtime import cuda_runtime_available

def train_placeholder(root: Path, payload: dict) -> tuple[dict, int]:
    dataset_id = payload.get("data_session_id", "")
    base_model_name = payload.get("base_model_name", "0")
    output_name = clean_model_name(payload.get("output_model_name", "gaze_model"))
    
    session_dir = ensure_sessions_dir(root) / dataset_id
    manifest_path = session_dir / "manifest.jsonl"
    
    if not dataset_id or not manifest_path.exists():
        return {"ok": False, "error": "dataset session or manifest not found"}, 404

    motion_samples, motion_diagnostics = load_motion_samples(
        ensure_sessions_dir(root),
        session_ids=(dataset_id,),
    )
    uses_motion_protocol = any(
        sample.collection_protocol == "motion-diverse-v1"
        for sample in motion_samples
    )
    if uses_motion_protocol:
        motion_audit = audit_payload(motion_samples, motion_diagnostics)
        if motion_audit["status"] != "ready":
            return {
                "ok": False,
                "error": (
                    "motion-diverse calibration failed its frozen coverage gates; "
                    "collect the missing conditions before training"
                ),
                "motion_audit": motion_audit,
            }, 400
        if base_model_name != "0":
            return {
                "ok": False,
                "error": (
                    "motion-diverse calibration must start from the frozen base "
                    "model; staged recalibration is not leakage-audited"
                ),
            }, 400

    try:
        from core.unigaze_personalization.dataset import read_manifest
        from core.unigaze_personalization.transforms import to_unigaze_tensor
        from core.unigaze_personalization.model import UniGazeFeatureWrapper, load_unigaze_b16

        # 1. Load calibration data records
        records = read_manifest(manifest_path)
        if not records:
            return {"ok": False, "error": "no valid calibration samples found"}, 400

        # Load baseline UniGaze-B model (CPU or GPU with safe fallback)
        device = "cpu"
        allow_cuda_value = payload.get("allow_cuda")
        allow_cuda = (
            not uses_motion_protocol
            if allow_cuda_value is None
            else allow_cuda_value is True
        )
        if allow_cuda and cuda_runtime_available(torch):
            try:
                t = torch.zeros((1, 3, 224, 224), device="cuda")
                conv = torch.nn.Conv2d(3, 16, kernel_size=16, stride=16).to("cuda")
                _ = conv(t)
                device = "cuda"
            except Exception:
                device = "cpu"

        try:
            base_model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()
        except Exception:
            device = "cpu"
            base_model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()

        gaze_list = []
        target_list = []
        viewport_list = []
        validation_groups = []
        head_pose_list = []
        face_geometry_list = []

        # 2. Extract baseline predictions
        for record in records:
            if not record.get("normalized_face_path"):
                continue
            if uses_motion_protocol:
                try:
                    head_pose = [
                        float(record["head_pose_pitch_yaw"][0]),
                        float(record["head_pose_pitch_yaw"][1]),
                    ]
                    face_geometry = face_geometry_from_bbox(record["face_bbox"])
                except (KeyError, TypeError, ValueError):
                    continue
            image_path = session_dir / record["normalized_face_path"]
            if not image_path.exists():
                continue
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = to_unigaze_tensor(image_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                gaze_tensor = base_model(image_tensor)
                gaze_vec = gaze_tensor.squeeze(0).cpu().tolist()  # [pitch, yaw]

            gaze_list.append(gaze_vec)
            target_list.append([record["target_x_norm"], record["target_y_norm"]])
            viewport_list.append([
                float(record.get("viewport_width", 1920.0)),
                float(record.get("viewport_height", 1080.0))
            ])
            validation_groups.append(record.get("motion_block_id", ""))
            if uses_motion_protocol:
                head_pose_list.append(head_pose)
                face_geometry_list.append(face_geometry)

        N = len(gaze_list)
        if N == 0:
            return {"ok": False, "error": "no valid face images processed successfully"}, 400

        # 3. Fit stages sequentially
        stages = []
        if base_model_name != "0":
            base_model_file = model_path(root, base_model_name)
            if not base_model_file.exists():
                return {"ok": False, "error": f"base model {base_model_name} not found."}, 400
            try:
                with base_model_file.open("r", encoding="utf-8") as handle:
                    old_data = json.load(handle)
                    if "stages" in old_data:
                        stages = old_data["stages"]
                    elif "W" in old_data:
                        stages = [{
                            "stage": 1,
                            "W": old_data["W"],
                            "poly_degree": old_data.get("poly_degree", 2),
                            "mean_px_error": old_data.get("mean_px_error", 0.0)
                        }]
                    if any(
                        stage.get("calibrator_type")
                        == "motion_conditioned_ridge_v1"
                        for stage in stages
                    ):
                        return {
                            "ok": False,
                            "error": (
                                "motion-conditioned models cannot be used as a "
                                "cascaded calibration base"
                            ),
                        }, 400
            except Exception as exc:
                return {"ok": False, "error": f"failed to read base model: {exc}"}, 500

        unique_targets = len(set(tuple(t) for t in target_list))
        grouped_validation = validation_groups if uses_motion_protocol else None
        validation_scheme = (
            "leave_one_motion_block_out"
            if grouped_validation is not None
            else "leave_one_sample_out"
        )
        candidate_comparison = None

        if len(stages) == 0:
            # Motion-diverse sessions hold out a complete posture block. Legacy
            # sessions retain sample-level LOOCV for backward compatibility.
            X_raw = np.array(gaze_list)
            Y = np.array(target_list)
            W, poly_degree, best_alpha, baseline_validation_error = fit_best_stage(
                X_raw,
                Y,
                viewport_list,
                unique_targets,
                is_stage_1=True,
                validation_groups=grouped_validation,
            )

            # Recompute predictions on train set for metrics
            yaw = X_raw[:, 1]
            pitch = X_raw[:, 0]
            if poly_degree == 1:
                X = np.column_stack([yaw, pitch, np.ones(N)])
            else:
                X = np.column_stack([yaw, pitch, yaw * yaw, pitch * pitch, yaw * pitch, np.ones(N)])

            baseline_predictions = X @ W
            best_validation_error = baseline_validation_error

            if uses_motion_protocol:
                motion_features = motion_conditioned_features(
                    X_raw,
                    np.array(head_pose_list),
                    np.array(face_geometry_list),
                )
                (
                    conditioned_weights,
                    feature_mean,
                    feature_scale,
                    conditioned_alpha,
                    conditioned_validation_error,
                ) = fit_standardized_ridge(
                    motion_features,
                    Y,
                    viewport_list,
                    validation_groups=validation_groups,
                )
                (
                    select_conditioned,
                    required_improvement,
                    observed_improvement,
                ) = motion_challenger_decision(
                    baseline_validation_error,
                    conditioned_validation_error,
                )
                candidate_comparison = {
                    "baseline_gaze_only_px": baseline_validation_error,
                    "motion_conditioned_px": conditioned_validation_error,
                    "required_improvement_px": required_improvement,
                    "observed_improvement_px": observed_improvement,
                    "selected": (
                        "motion_conditioned_ridge_v1"
                        if select_conditioned
                        else "gaze_polynomial"
                    ),
                }
            else:
                select_conditioned = False

            if select_conditioned:
                pred_Y = (
                    standardized_design(
                        motion_features,
                        feature_mean,
                        feature_scale,
                    )
                    @ conditioned_weights
                )
                best_validation_error = conditioned_validation_error
                stages = [{
                    "stage": 1,
                    "calibrator_type": "motion_conditioned_ridge_v1",
                    "feature_names": list(MOTION_FEATURE_NAMES),
                    "feature_mean": feature_mean.tolist(),
                    "feature_scale": feature_scale.tolist(),
                    "W": conditioned_weights.tolist(),
                    "alpha": conditioned_alpha,
                    "validation_px_error": conditioned_validation_error,
                    "validation_scheme": validation_scheme,
                    "mean_px_error": 0.0,
                }]
            else:
                pred_Y = baseline_predictions
                stages = [{
                    "stage": 1,
                    "calibrator_type": "gaze_polynomial",
                    "W": W.tolist(),
                    "poly_degree": poly_degree,
                    "alpha": best_alpha,
                    "validation_px_error": baseline_validation_error,
                    "validation_scheme": validation_scheme,
                    "mean_px_error": 0.0,
                }]
        else:
            # Stage 2+: Secondary Calibration using self-tuning LOOCV
            current_inputs = gaze_list
            for stage_idx, stage_meta in enumerate(stages):
                W_stage = np.array(stage_meta["W"])
                s_degree = stage_meta["poly_degree"]
                
                next_inputs = []
                for idx in range(N):
                    if stage_idx == 0:
                        p_i, y_i = current_inputs[idx][0], current_inputs[idx][1]
                    else:
                        y_i, p_i = current_inputs[idx][0], current_inputs[idx][1]
                        
                    if s_degree == 1:
                        feat = np.array([y_i, p_i, 1.0])
                    else:
                        feat = np.array([y_i, p_i, y_i * y_i, p_i * p_i, y_i * p_i, 1.0])
                    
                    pred = feat @ W_stage
                    next_inputs.append([float(pred[0]), float(pred[1])])
                current_inputs = next_inputs

            s1_arr = np.array(current_inputs)
            Y = np.array(target_list)
            W2, poly_degree, best_alpha, best_validation_error = fit_best_stage(
                s1_arr,
                Y,
                viewport_list,
                unique_targets,
                is_stage_1=False,
                validation_groups=grouped_validation,
            )

            # Recompute predictions on train set for metrics
            s1_x = s1_arr[:, 0]
            s1_y = s1_arr[:, 1]
            if poly_degree == 1:
                X = np.column_stack([s1_x, s1_y, np.ones(N)])
            else:
                X = np.column_stack([s1_x, s1_y, s1_x * s1_x, s1_y * s1_y, s1_x * s1_y, np.ones(N)])

            pred_Y = X @ W2
            
            stages = list(stages) + [{
                "stage": len(stages) + 1,
                "W": W2.tolist(),
                "poly_degree": poly_degree,
                "alpha": best_alpha,
                "validation_px_error": best_validation_error,
                "validation_scheme": validation_scheme,
                "mean_px_error": 0.0
            }]

        # 4. Calculate error and noise metrics
        errors = []
        target_to_preds = {}
        for i in range(N):
            w_w = viewport_list[i][0]
            h_h = viewport_list[i][1]
            
            pred_x_px = (pred_Y[i, 0] + 1.0) * 0.5 * w_w
            pred_y_px = (pred_Y[i, 1] + 1.0) * 0.5 * h_h
            
            target_x_px = (Y[i, 0] + 1.0) * 0.5 * w_w
            target_y_px = (Y[i, 1] + 1.0) * 0.5 * h_h
            
            err = np.sqrt((pred_x_px - target_x_px) ** 2 + (pred_y_px - target_y_px) ** 2)
            errors.append(err)

            t_norm_tuple = (float(Y[i, 0]), float(Y[i, 1]))
            if t_norm_tuple not in target_to_preds:
                target_to_preds[t_norm_tuple] = []
            target_to_preds[t_norm_tuple].append([pred_x_px, pred_y_px])

        mean_px_error = float(np.mean(errors))
        stages[-1]["mean_px_error"] = mean_px_error

        std_devs = []
        for t_norm, preds in target_to_preds.items():
            if len(preds) > 1:
                preds_arr = np.array(preds)
                std_x = np.std(preds_arr[:, 0])
                std_y = np.std(preds_arr[:, 1])
                std_devs.append(float(np.sqrt(std_x ** 2 + std_y ** 2)))
        noise_level = float(np.mean(std_devs)) if len(std_devs) > 0 else 0.0

        # Save model JSON artifact to the chenghao/gaze_data/runs/ directory
        output_model_path = model_path(root, output_name)
        calibration_data = {
            "name": output_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_session_id": dataset_id,
            "stages": stages,
            "num_stages": len(stages),
            "mean_px_error": mean_px_error,
            "validation_px_error": best_validation_error,
            "validation_scheme": validation_scheme,
            "candidate_comparison": candidate_comparison,
            "training_device": device,
            "noise_level": noise_level,
            "train_samples": N
        }
        
        output_model_path.write_text(
            json.dumps(calibration_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        return {
            "ok": True,
            "model_name": output_name,
            "train_samples": N,
            "best_val_px_error": best_validation_error,
            "train_px_error": mean_px_error,
            "validation_scheme": validation_scheme,
            "training_device": device,
            "noise_level": noise_level
        }, 200

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": f"training pipeline failed: {exc}"}, 500
