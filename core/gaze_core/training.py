from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .model_registry import clean_model_name, model_path
from .sample_store import ensure_sessions_dir


def fit_best_stage(inputs: np.ndarray, Y: np.ndarray, viewport_list: list[list[float]], unique_targets: int, is_stage_1: bool = True) -> tuple[np.ndarray, int, float]:
    """
    Fits the best Ridge regression model using LOOCV (Leave-One-Out Cross Validation)
    to select the optimal polynomial degree (1 or 2) and regularization parameter alpha.
    """
    N = len(inputs)
    if is_stage_1:
        # Stage 1: inputs is [pitch, yaw], we want yaw first, pitch second
        val1 = inputs[:, 1]  # yaw
        val2 = inputs[:, 0]  # pitch
    else:
        # Stage 2+: inputs is [x, y], we want x first, y second
        val1 = inputs[:, 0]  # x
        val2 = inputs[:, 1]  # y

    candidate_degrees = [1]
    if unique_targets > 5 and N >= 6:
        candidate_degrees.append(2)

    candidate_alphas = [1e-4, 1e-3, 1e-2, 0.1]

    best_degree = 1
    best_alpha = 1e-3
    best_cv_error = float('inf')

    for degree in candidate_degrees:
        if degree == 1:
            X = np.column_stack([val1, val2, np.ones(N)])
        else:
            X = np.column_stack([val1, val2, val1 * val1, val2 * val2, val1 * val2, np.ones(N)])

        for alpha in candidate_alphas:
            errors = []
            for i in range(N):
                X_train = np.delete(X, i, axis=0)
                Y_train = np.delete(Y, i, axis=0)
                X_test = X[i, :].reshape(1, -1)
                Y_test = Y[i, :].reshape(1, -1)

                w_w, h_h = viewport_list[i]

                try:
                    XT_X = X_train.T @ X_train
                    I = np.eye(X_train.shape[1])
                    I[-1, -1] = 0.0  # Do not regularize bias
                    W = np.linalg.solve(XT_X + alpha * I, X_train.T @ Y_train)
                    pred = X_test @ W

                    pred_x_px = (pred[0, 0] + 1.0) * 0.5 * w_w
                    pred_y_px = (pred[0, 1] + 1.0) * 0.5 * h_h
                    target_x_px = (Y_test[0, 0] + 1.0) * 0.5 * w_w
                    target_y_px = (Y_test[0, 1] + 1.0) * 0.5 * h_h

                    err = np.sqrt((pred_x_px - target_x_px)**2 + (pred_y_px - target_y_px)**2)
                    errors.append(err)
                except np.linalg.LinAlgError:
                    continue

            if errors:
                mean_cv = np.mean(errors)
                if mean_cv < best_cv_error:
                    best_cv_error = mean_cv
                    best_degree = degree
                    best_alpha = alpha

    # Train final model on all data
    if best_degree == 1:
        X_all = np.column_stack([val1, val2, np.ones(N)])
    else:
        X_all = np.column_stack([val1, val2, val1 * val1, val2 * val2, val1 * val2, np.ones(N)])

    XT_X = X_all.T @ X_all
    I = np.eye(X_all.shape[1])
    I[-1, -1] = 0.0
    W_final = np.linalg.solve(XT_X + best_alpha * I, X_all.T @ Y)

    return W_final, best_degree, best_alpha


def train_placeholder(root: Path, payload: dict) -> tuple[dict, int]:
    dataset_id = payload.get("data_session_id", "")
    base_model_name = payload.get("base_model_name", "0")
    output_name = clean_model_name(payload.get("output_model_name", "gaze_model"))
    
    session_dir = ensure_sessions_dir(root) / dataset_id
    manifest_path = session_dir / "manifest.jsonl"
    
    if not dataset_id or not manifest_path.exists():
        return {"ok": False, "error": "dataset session or manifest not found"}, 404

    try:
        from core.unigaze_personalization.dataset import read_manifest
        from core.unigaze_personalization.transforms import to_unigaze_tensor
        from core.unigaze_personalization.model import UniGazeFeatureWrapper, load_unigaze_b16

        # 1. Load calibration data records
        records = read_manifest(manifest_path)
        if not records:
            return {"ok": False, "error": "no valid calibration samples found"}, 400

        # Load baseline UniGaze-B model (CPU or GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()

        gaze_list = []
        target_list = []
        viewport_list = []

        # 2. Extract baseline predictions
        for record in records:
            if not record.get("normalized_face_path"):
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
            except Exception as exc:
                return {"ok": False, "error": f"failed to read base model: {exc}"}, 500

        unique_targets = len(set(tuple(t) for t in target_list))

        if len(stages) == 0:
            # Stage 1: Fit fresh Stage 1 model using self-tuning LOOCV
            X_raw = np.array(gaze_list)
            Y = np.array(target_list)
            W, poly_degree, best_alpha = fit_best_stage(X_raw, Y, viewport_list, unique_targets, is_stage_1=True)

            # Recompute predictions on train set for metrics
            yaw = X_raw[:, 1]
            pitch = X_raw[:, 0]
            if poly_degree == 1:
                X = np.column_stack([yaw, pitch, np.ones(N)])
            else:
                X = np.column_stack([yaw, pitch, yaw * yaw, pitch * pitch, yaw * pitch, np.ones(N)])

            pred_Y = X @ W
            
            stages = [{
                "stage": 1,
                "W": W.tolist(),
                "poly_degree": poly_degree,
                "alpha": best_alpha,
                "mean_px_error": 0.0
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
            W2, poly_degree, best_alpha = fit_best_stage(s1_arr, Y, viewport_list, unique_targets, is_stage_1=False)

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
            "mean_px_error": mean_px_error,
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
            "best_val_px_error": mean_px_error,
            "noise_level": noise_level
        }, 200

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": f"training pipeline failed: {exc}"}, 500
