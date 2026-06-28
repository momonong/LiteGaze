import os
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_calibration_samples():
    sessions_dir = ROOT / "data" / "sessions"
    datasets = []
    
    for sess_path in sessions_dir.iterdir():
        if not sess_path.is_dir():
            continue
        manifest_path = sess_path / "manifest.jsonl"
        if not manifest_path.exists():
            continue
            
        gaze_list = []
        target_list = []
        viewport_list = []
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("ok") and "head_pose_pitch_yaw" in record:
                        gaze_list.append(record["head_pose_pitch_yaw"])
                        target_list.append([record["target_x_norm"], record["target_y_norm"]])
                        viewport_list.append([
                            float(record.get("viewport_width", 1920.0)),
                            float(record.get("viewport_height", 1080.0))
                        ])
                except Exception:
                    continue
                    
        if len(gaze_list) >= 6:
            datasets.append({
                "session_id": sess_path.name,
                "gaze": np.array(gaze_list),
                "target": np.array(target_list),
                "viewport": np.array(viewport_list)
            })
            
    print(f"Found {len(datasets)} valid calibration datasets with enough samples.")
    return datasets

def fit_ridge(X, Y, alpha):
    # Standard Ridge Regression solver with bias
    XT_X = X.T @ X
    I = np.eye(X.shape[1])
    I[-1, -1] = 0.0  # Do not regularize the bias term
    W = np.linalg.solve(XT_X + alpha * I, X.T @ Y)
    return W

def evaluate_cross_validation(datasets, degree=2, alpha=1e-4):
    cv_errors_px = []
    train_errors_px = []
    
    for data in datasets:
        gaze = data["gaze"]
        target = data["target"]
        viewport = data["viewport"]
        n = len(gaze)
        
        # Build features
        pitch = gaze[:, 0]
        yaw = gaze[:, 1]
        if degree == 1:
            X = np.column_stack([yaw, pitch, np.ones(n)])
        else:
            X = np.column_stack([
                yaw, pitch, yaw * yaw, pitch * pitch, yaw * pitch, np.ones(n)
            ])
            
        Y = target
        
        # Leave-One-Out Cross-Validation (LOOCV)
        dataset_cv_errs = []
        for i in range(n):
            # Split train/test
            X_train = np.delete(X, i, axis=0)
            Y_train = np.delete(Y, i, axis=0)
            X_test = X[i, :].reshape(1, -1)
            Y_test = Y[i, :].reshape(1, -1)
            
            w_w, h_h = viewport[i]
            
            try:
                W = fit_ridge(X_train, Y_train, alpha)
                pred_norm = X_test @ W
                
                # Convert back to pixels
                pred_x_px = (pred_norm[0, 0] + 1.0) * 0.5 * w_w
                pred_y_px = (pred_norm[0, 1] + 1.0) * 0.5 * h_h
                
                target_x_px = (Y_test[0, 0] + 1.0) * 0.5 * w_w
                target_y_px = (Y_test[0, 1] + 1.0) * 0.5 * h_h
                
                err_px = np.sqrt((pred_x_px - target_x_px) ** 2 + (pred_y_px - target_y_px) ** 2)
                dataset_cv_errs.append(err_px)
            except np.linalg.LinAlgError:
                continue
                
        # Train error on all samples
        try:
            W_all = fit_ridge(X, Y, alpha)
            pred_all = X @ W_all
            for idx in range(n):
                w_w, h_h = viewport[idx]
                p_x = (pred_all[idx, 0] + 1.0) * 0.5 * w_w
                p_y = (pred_all[idx, 1] + 1.0) * 0.5 * h_h
                t_x = (Y[idx, 0] + 1.0) * 0.5 * w_w
                t_y = (Y[idx, 1] + 1.0) * 0.5 * h_h
                train_errors_px.append(np.sqrt((p_x - t_x)**2 + (p_y - t_y)**2))
        except np.linalg.LinAlgError:
            pass
            
        if dataset_cv_errs:
            cv_errors_px.append(np.mean(dataset_cv_errs))
            
    return np.mean(train_errors_px) if train_errors_px else 0.0, np.mean(cv_errors_px) if cv_errors_px else 0.0

def main():
    datasets = load_calibration_samples()
    if not datasets:
        print("No valid datasets found to evaluate.")
        return
        
    print("\nGrid Search over Regularization Parameter alpha (Degree = 2):")
    print("-" * 65)
    print(f"{'alpha':<12} | {'Mean Train Error (px)':<22} | {'Mean Cross-Val Error (px)':<25}")
    print("-" * 65)
    
    alphas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]
    best_cv = float('inf')
    best_alpha = 1e-4
    
    for a in alphas:
        t_err, cv_err = evaluate_cross_validation(datasets, degree=2, alpha=a)
        print(f"{a:<12g} | {t_err:<22.4f} | {cv_err:<25.4f}")
        if cv_err < best_cv and cv_err > 0.0:
            best_cv = cv_err
            best_alpha = a
            
    print("-" * 65)
    print(f"Optimal Alpha under Degree=2: {best_alpha} (Validation Error: {best_cv:.4f} px)")
    
    print("\nComparing Polynomial Degrees with optimal alpha:")
    print("-" * 65)
    for deg in [1, 2]:
        t_err, cv_err = evaluate_cross_validation(datasets, degree=deg, alpha=best_alpha)
        print(f"Degree {deg} | Train Error: {t_err:.4f} px | Cross-Val Error: {cv_err:.4f} px")

if __name__ == "__main__":
    main()
