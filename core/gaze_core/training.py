from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .capture_contract import (
    build_fit_target_contract,
    representative_capture_contract,
)
from .calibration_regression import (
    MOTION_FEATURE_NAMES,
    face_geometry_from_bbox,
    fit_best_stage,
    fit_standardized_ridge,
    motion_conditioned_features,
)
from .model_registry import clean_model_name, model_path
from .motion_experiment import (
    BASELINE_MODEL,
    CHALLENGER_MODEL,
    VALIDATION_SCHEME,
    build_uncertainty_v2_bundle,
    evaluate_motion_candidates,
)
from .motion_robustness import audit_payload, load_motion_samples
from .sample_store import ensure_sessions_dir
from .stage_pipeline import apply_stage_chain
from .torch_runtime import cuda_runtime_available
from .uncertainty import validate_complete_motion_grid


def _selected_motion_validation_metrics(
    selected_model: str,
    *,
    baseline_error: float,
    challenger_error: float,
    baseline_hyperparameter_error: float,
    challenger_hyperparameter_error: float,
) -> dict[str, float]:
    """Keep selected nested-outer and hyperparameter CV metrics distinct."""

    if selected_model == BASELINE_MODEL:
        validation_error = baseline_error
        hyperparameter_error = baseline_hyperparameter_error
    elif selected_model == CHALLENGER_MODEL:
        validation_error = challenger_error
        hyperparameter_error = challenger_hyperparameter_error
    else:
        raise ValueError(f"unknown motion calibration model: {selected_model}")
    return {
        "validation_px_error": float(validation_error),
        "hyperparameter_cv_px_error": float(hyperparameter_error),
    }


def _stable_training_sample_id(
    dataset_id: str,
    manifest_index: int,
    record: dict,
) -> str:
    """Create an opaque, deterministic id without persisting a source path."""

    identity = {
        "dataset_id": str(dataset_id),
        "manifest_index": int(manifest_index),
        "capture_burst_id": str(record.get("capture_burst_id", "")),
        "motion_block_id": str(record.get("motion_block_id", "")),
        "point_index": str(record.get("point_index", "")),
    }
    serialized = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


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
        import torch

        from core.unigaze_personalization.dataset import read_manifest
        from core.unigaze_personalization.model import (
            UniGazeFeatureWrapper,
            load_unigaze_b16,
        )
        from core.unigaze_personalization.transforms import to_unigaze_tensor

        # 1. Load calibration data records
        records = read_manifest(manifest_path)
        if not records:
            return {"ok": False, "error": "no valid calibration samples found"}, 400
        capture_contract = representative_capture_contract(records)

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
        uncertainty_sample_ids = []
        uncertainty_target_ids = []

        # 2. Extract baseline predictions
        for manifest_index, record in enumerate(records):
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
                uncertainty_sample_ids.append(
                    _stable_training_sample_id(dataset_id, manifest_index, record)
                )
                uncertainty_target_ids.append(str(record.get("point_index", "")))

        N = len(gaze_list)
        if N == 0:
            return {"ok": False, "error": "no valid face images processed successfully"}, 400
        if uses_motion_protocol:
            try:
                validate_complete_motion_grid(
                    uncertainty_sample_ids,
                    uncertainty_target_ids,
                    validation_groups,
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "error": (
                        "uncertainty v2 requires the exact frozen 13x5 processed "
                        f"grid: {exc}"
                    ),
                    "uncertainty_v2_status": "failed_closed_incomplete_grid",
                }, 400

        # 3. Fit stages sequentially
        stages = []
        inherited_fit_target_contract = None
        if base_model_name != "0":
            base_model_file = model_path(root, base_model_name)
            if not base_model_file.exists():
                return {"ok": False, "error": f"base model {base_model_name} not found."}, 400
            try:
                with base_model_file.open("r", encoding="utf-8") as handle:
                    old_data = json.load(handle)
                    inherited_fit_target_contract = old_data.get(
                        "fit_target_contract"
                    )
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
            VALIDATION_SCHEME
            if grouped_validation is not None
            else "leave_one_sample_out"
        )
        candidate_comparison = None

        if len(stages) == 0:
            # Motion-diverse sessions hold out a complete posture block. Legacy
            # sessions retain sample-level LOOCV for backward compatibility.
            X_raw = np.array(gaze_list)
            Y = np.array(target_list)
            (
                W,
                poly_degree,
                best_alpha,
                baseline_hyperparameter_error,
            ) = fit_best_stage(
                X_raw,
                Y,
                viewport_list,
                unique_targets,
                is_stage_1=True,
                validation_groups=grouped_validation,
            )

            # Preserve the historical unclamped training metric while sharing
            # the same stage-order/design implementation as nested evaluation.
            baseline_predictions = apply_stage_chain(
                X_raw,
                [{
                    "stage": 1,
                    "calibrator_type": "gaze_polynomial",
                    "W": W.tolist(),
                    "poly_degree": poly_degree,
                    "alpha": best_alpha,
                }],
                clamp=False,
            )
            baseline_validation_error = baseline_hyperparameter_error
            best_validation_error = baseline_hyperparameter_error
            selected_validation_metrics = {
                "validation_px_error": float(baseline_hyperparameter_error),
                "hyperparameter_cv_px_error": float(baseline_hyperparameter_error),
            }

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
                    conditioned_hyperparameter_error,
                ) = fit_standardized_ridge(
                    motion_features,
                    Y,
                    viewport_list,
                    validation_groups=validation_groups,
                )
                nested_comparison = evaluate_motion_candidates(
                    X_raw,
                    np.array(head_pose_list),
                    np.array(face_geometry_list),
                    Y,
                    viewport_list,
                    validation_groups,
                )
                promotion_gate = nested_comparison["promotion_gate"]
                baseline_validation_error = nested_comparison["candidates"][
                    BASELINE_MODEL
                ]["macro_mean_px"]
                conditioned_validation_error = nested_comparison["candidates"][
                    CHALLENGER_MODEL
                ]["macro_mean_px"]
                selected_model = promotion_gate["selected_model"]
                select_conditioned = selected_model == CHALLENGER_MODEL
                selected_validation_metrics = _selected_motion_validation_metrics(
                    selected_model,
                    baseline_error=baseline_validation_error,
                    challenger_error=conditioned_validation_error,
                    baseline_hyperparameter_error=baseline_hyperparameter_error,
                    challenger_hyperparameter_error=conditioned_hyperparameter_error,
                )
                best_validation_error = selected_validation_metrics[
                    "validation_px_error"
                ]
                candidate_comparison = {
                    "baseline_gaze_only_px": baseline_validation_error,
                    "motion_conditioned_px": conditioned_validation_error,
                    "required_improvement_px": promotion_gate[
                        "required_improvement_px"
                    ],
                    "observed_improvement_px": promotion_gate[
                        "observed_improvement_px"
                    ],
                    "selected": promotion_gate["selected_model"],
                    "validation_scheme": nested_comparison["validation_scheme"],
                    "metrics": nested_comparison["candidates"],
                    "promotion_gate": promotion_gate,
                    "folds": nested_comparison["folds"],
                    "final_fit_hyperparameters": {
                        BASELINE_MODEL: {
                            "alpha": best_alpha,
                            "degree": poly_degree,
                            "group_cv_mean_px": baseline_hyperparameter_error,
                        },
                        CHALLENGER_MODEL: {
                            "alpha": conditioned_alpha,
                            "group_cv_mean_px": conditioned_hyperparameter_error,
                        },
                    },
                }
            else:
                select_conditioned = False

            if select_conditioned:
                stages = [{
                    "stage": 1,
                    "calibrator_type": "motion_conditioned_ridge_v1",
                    "feature_names": list(MOTION_FEATURE_NAMES),
                    "feature_mean": feature_mean.tolist(),
                    "feature_scale": feature_scale.tolist(),
                    "W": conditioned_weights.tolist(),
                    "alpha": conditioned_alpha,
                    "validation_px_error": selected_validation_metrics[
                        "validation_px_error"
                    ],
                    "hyperparameter_cv_px_error": selected_validation_metrics[
                        "hyperparameter_cv_px_error"
                    ],
                    "validation_scheme": validation_scheme,
                    "mean_px_error": 0.0,
                }]
                pred_Y = apply_stage_chain(
                    X_raw,
                    stages,
                    head_pitch_yaw=np.array(head_pose_list),
                    face_geometry=np.array(face_geometry_list),
                    clamp=False,
                )
            else:
                pred_Y = baseline_predictions
                stages = [{
                    "stage": 1,
                    "calibrator_type": "gaze_polynomial",
                    "W": W.tolist(),
                    "poly_degree": poly_degree,
                    "alpha": best_alpha,
                    "validation_px_error": selected_validation_metrics[
                        "validation_px_error"
                    ],
                    "hyperparameter_cv_px_error": selected_validation_metrics[
                        "hyperparameter_cv_px_error"
                    ],
                    "validation_scheme": validation_scheme,
                    "mean_px_error": 0.0,
                }]
        else:
            # Stage 2+: Secondary Calibration using self-tuning LOOCV
            s1_arr = apply_stage_chain(
                np.array(gaze_list),
                stages,
                clamp=False,
            )
            Y = np.array(target_list)
            W2, poly_degree, best_alpha, best_validation_error = fit_best_stage(
                s1_arr,
                Y,
                viewport_list,
                unique_targets,
                is_stage_1=False,
                validation_groups=grouped_validation,
            )

            stages = list(stages) + [{
                "stage": len(stages) + 1,
                "calibrator_type": "gaze_polynomial",
                "W": W2.tolist(),
                "poly_degree": poly_degree,
                "alpha": best_alpha,
                "validation_px_error": best_validation_error,
                "hyperparameter_cv_px_error": best_validation_error,
                "validation_scheme": validation_scheme,
                "mean_px_error": 0.0
            }]
            pred_Y = apply_stage_chain(
                np.array(gaze_list),
                stages,
                clamp=False,
            )

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
        fit_target_contract = build_fit_target_contract(
            target_list,
            inherited_contract=inherited_fit_target_contract,
            inherited_targets_required=base_model_name != "0",
        )
        uncertainty_v2_bundle = None
        if uses_motion_protocol:
            uncertainty_v2_bundle = build_uncertainty_v2_bundle(
                np.array(gaze_list),
                np.array(head_pose_list),
                np.array(face_geometry_list),
                np.array(target_list),
                viewport_list,
                validation_groups,
                uncertainty_sample_ids,
                uncertainty_target_ids,
                stages,
            )

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
            "train_samples": N,
            "fit_target_contract": fit_target_contract,
        }
        if uncertainty_v2_bundle is not None:
            calibration_data["model_artifact_schema_version"] = 2
            calibration_data["uncertainty_v2"] = uncertainty_v2_bundle
        if "hyperparameter_cv_px_error" in stages[-1]:
            calibration_data["hyperparameter_cv_px_error"] = stages[-1][
                "hyperparameter_cv_px_error"
            ]
        if capture_contract is not None:
            calibration_data["capture_contract"] = capture_contract
        
        output_model_path.write_text(
            json.dumps(calibration_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        response = {
            "ok": True,
            "model_name": output_name,
            "train_samples": N,
            "best_val_px_error": best_validation_error,
            "train_px_error": mean_px_error,
            "validation_scheme": validation_scheme,
            "training_device": device,
            "noise_level": noise_level,
            "capture_contract": capture_contract,
            "fit_target_contract": fit_target_contract,
        }
        if uncertainty_v2_bundle is not None:
            response["uncertainty_v2"] = {
                "status": uncertainty_v2_bundle["status"],
                "definition_sha256": uncertainty_v2_bundle[
                    "definition_sha256"
                ],
                "threshold": None,
                "abstention_status": uncertainty_v2_bundle[
                    "abstention_policy"
                ]["status"],
                "fresh_matched_contract_capture_required": True,
            }
        if "hyperparameter_cv_px_error" in stages[-1]:
            response["hyperparameter_cv_px_error"] = stages[-1][
                "hyperparameter_cv_px_error"
            ]
        return response, 200

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": f"training pipeline failed: {exc}"}, 500
