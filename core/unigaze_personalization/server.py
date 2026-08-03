from __future__ import annotations

import argparse
import base64
import json
import time
import uuid
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .preprocess import MediaPipeUniGazePreprocessor


class SessionRequest(BaseModel):
    participant_id: str = "anonymous"


class SampleRequest(BaseModel):
    session_id: str
    image_data: str = Field(description="Data URL or base64-encoded JPEG/PNG")
    target_x: float
    target_y: float
    viewport_width: float
    viewport_height: float
    target_x_norm: float | None = None
    target_y_norm: float | None = None
    screen_width: float | None = None
    screen_height: float | None = None
    phase: str = "calibration"
    point_index: int = 0
    repeat_index: int = 0


class TrainRequest(BaseModel):
    data_session_id: str
    base_model_name: str
    output_model_name: str
    epochs: int = 80
    val_ratio: float = 0.2


class PredictRequest(BaseModel):
    image_data: str
    model_name: str = "before"


# Model cache for real-time inference
_model_cache = {}
_model_cache_lock = Lock()


def _decode_image(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot decode image")
    return image


def _decode_binary_image(raw: bytes) -> np.ndarray:
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot decode binary image")
    return image


def _rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def create_app(data_dir: str | Path = "data/sessions") -> FastAPI:
    root = Path(__file__).resolve().parents[2]
    static_dir = root / "web" / "static"
    data_root = Path(data_dir).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    preprocessor = MediaPipeUniGazePreprocessor()
    lock = Lock()

    app = FastAPI(title="UniGaze Personalization Collector")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "model": "unigaze_b16_joint", "data_dir": str(data_root)}

    @app.get("/api/list_datasets")
    def list_datasets() -> dict:
        datasets = []
        if data_root.exists():
            for folder in sorted(data_root.iterdir(), key=lambda p: p.name, reverse=True):
                if folder.is_dir() and (folder / "manifest.jsonl").exists():
                    manifest_path = folder / "manifest.jsonl"
                    try:
                        num_samples = sum(1 for _ in manifest_path.open("r", encoding="utf-8"))
                    except Exception:
                        num_samples = 0
                    
                    participant = "unknown"
                    session_json = folder / "session.json"
                    if session_json.exists():
                        try:
                            meta = json.loads(session_json.read_text(encoding="utf-8"))
                            participant = meta.get("participant_id", "unknown")
                        except Exception:
                            pass
                    
                    datasets.append({
                        "id": folder.name,
                        "display_name": f"{folder.name} ({participant}, {num_samples} 點)",
                        "num_samples": num_samples,
                        "participant": participant
                    })
        return {"ok": True, "datasets": datasets}

    @app.get("/api/list_models")
    def list_models() -> dict:
        models = []
        runs_root = root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        if runs_root.exists():
            for file in sorted(runs_root.glob("*.json"), key=lambda p: p.name):
                model_name = file.stem
                try:
                    meta = json.loads(file.read_text(encoding="utf-8"))
                    mean_px_error = meta.get("mean_px_error", 0.0)
                    num_stages = len(meta.get("stages", []))
                    noise_level = meta.get("noise_level", 0.0)
                    if noise_level > 0:
                        display_name = f"{model_name} (階數: {num_stages}, 誤差: {mean_px_error:.1f} px, 噪聲: {noise_level:.1f} px)"
                    else:
                        display_name = f"{model_name} (階數: {num_stages}, 誤差: {mean_px_error:.1f} px)"
                except Exception:
                    display_name = model_name
                    num_stages = 1
                    mean_px_error = 0.0
                    noise_level = 0.0
                
                models.append({
                    "name": model_name,
                    "display_name": display_name,
                    "num_stages": num_stages,
                    "mean_px_error": mean_px_error,
                    "noise_level": noise_level
                })
        return {"ok": True, "models": models}

    @app.post("/api/session")
    def create_session(request: SessionRequest) -> dict:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        clean_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request.participant_id)
        session_id = f"{timestamp}_{clean_id}_{uuid.uuid4().hex[:8]}"
        session_dir = data_root / session_id
        for child in ["raw", "crop", "normalized_face"]:
            (session_dir / child).mkdir(parents=True, exist_ok=True)
        meta = {"session_id": session_id, "participant_id": request.participant_id, "created_at": timestamp}
        (session_dir / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"ok": True, "session_id": session_id, "session_dir": str(session_dir)}

    @app.post("/api/sample")
    def add_sample(request: SampleRequest) -> dict:
        session_dir = data_root / request.session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="session not found")
        try:
            image = _decode_image(request.image_data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with lock:
            manifest_path = session_dir / "manifest.jsonl"
            sample_index = 0
            if manifest_path.exists():
                sample_index = sum(1 for _ in manifest_path.open("r", encoding="utf-8"))
            stem = f"{sample_index:06d}_{request.phase}_{request.point_index:02d}_{request.repeat_index:02d}"
            raw_path = session_dir / "raw" / f"{stem}.jpg"
            cv2.imwrite(str(raw_path), image)
            record = {
                "ok": True,
                "sample_index": sample_index,
                "session_id": request.session_id,
                "phase": request.phase,
                "point_index": request.point_index,
                "repeat_index": request.repeat_index,
                "target_x": request.target_x,
                "target_y": request.target_y,
                "viewport_width": request.viewport_width,
                "viewport_height": request.viewport_height,
                "screen_width": request.screen_width,
                "screen_height": request.screen_height,
                "target_x_norm": request.target_x_norm
                if request.target_x_norm is not None
                else (request.target_x / request.viewport_width) * 2.0 - 1.0,
                "target_y_norm": request.target_y_norm
                if request.target_y_norm is not None
                else (request.target_y / request.viewport_height) * 2.0 - 1.0,
                "raw_path": _rel(raw_path, session_dir),
                "created_at_unix": time.time(),
            }
            try:
                processed = preprocessor.process(image)
                crop_path = session_dir / "crop" / f"{stem}.jpg"
                norm_path = session_dir / "normalized_face" / f"{stem}.jpg"
                cv2.imwrite(str(crop_path), processed.crop_bgr)
                cv2.imwrite(str(norm_path), processed.image_bgr)
                record.update(
                    {
                        "crop_path": _rel(crop_path, session_dir),
                        "normalized_face_path": _rel(norm_path, session_dir),
                        "head_pose_pitch_yaw": processed.head_pose_pitch_yaw.tolist(),
                        "face_bbox": processed.face_bbox,
                    }
                )
            except ValueError as exc:
                record["ok"] = False
                record["error"] = str(exc)
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return {
            "ok": record["ok"],
            "sample_index": sample_index,
            "error": record.get("error", ""),
            "manifest": str(manifest_path),
        }

    @app.post("/api/train")
    def train_session(request: TrainRequest) -> dict:
        session_dir = data_root / request.data_session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="dataset session not found")
        manifest_path = session_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise HTTPException(status_code=400, detail="no calibration data found in selected dataset.")
        try:
            import json
            import numpy as np
            import torch
            import cv2
            from .dataset import read_manifest
            from .transforms import to_unigaze_tensor
            from .model import UniGazeFeatureWrapper, device_from_arg, load_unigaze_b16

            # 1. Load calibration data points
            records = read_manifest(manifest_path)
            if not records:
                raise ValueError("No records found in manifest")

            # Load base unigaze model
            device = device_from_arg("auto")
            base_model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()

            gaze_list = []
            target_list = []
            viewport_list = []

            # 2. Feed-forward each crop once to get raw eye angles [pitch, yaw]
            for record in records:
                image_path = session_dir / record["normalized_face_path"]
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
                raise ValueError("No valid images processed successfully")

            # Load base_model_name:
            # If base_model_name == "0": starts fresh Stage 1
            # If base_model_name != "0": load runs/{base_model_name}.json to get stages
            stages = []
            runs_root = root / "runs"
            runs_root.mkdir(parents=True, exist_ok=True)
            
            if request.base_model_name != "0":
                base_model_file = runs_root / f"{request.base_model_name}.json"
                if not base_model_file.exists():
                    raise HTTPException(status_code=400, detail=f"base model {request.base_model_name} not found.")
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
                    raise HTTPException(status_code=500, detail=f"failed to read base model: {exc}")

            unique_targets = len(set(tuple(t) for t in target_list))

            if len(stages) == 0:
                # Fresh fitting of Stage 1
                if unique_targets <= 5:
                    poly_degree = 1
                else:
                    poly_degree = 2 if N >= 6 else 1
                X_raw = np.array(gaze_list)
                pitch = X_raw[:, 0]
                yaw = X_raw[:, 1]

                if poly_degree == 1:
                    X = np.column_stack([yaw, pitch, np.ones(N)])
                else:
                    X = np.column_stack([
                        yaw,
                        pitch,
                        yaw * yaw,
                        pitch * pitch,
                        yaw * pitch,
                        np.ones(N)
                     ])

                Y = np.array(target_list)
                alpha = 1e-4
                XT_X = X.T @ X
                I = np.eye(X.shape[1])
                I[-1, -1] = 0.0
                W = np.linalg.solve(XT_X + alpha * I, X.T @ Y)

                pred_Y = X @ W
                
                stages = [{
                    "stage": 1,
                    "W": W.tolist(),
                    "poly_degree": poly_degree,
                    "mean_px_error": 0.0
                }]
            else:
                # Secondary Calibration (Stage 2)
                # First, pass gaze_list through stages sequentially to get base model predictions
                current_inputs = gaze_list
                for stage_idx, stage_meta in enumerate(stages):
                    W_stage = np.array(stage_meta["W"])
                    s_degree = stage_meta["poly_degree"]
                    
                    next_inputs = []
                    for idx in range(N):
                        if stage_idx == 0:
                            # Stage 1 input is raw [pitch, yaw] from UniGaze
                            p_i, y_i = current_inputs[idx][0], current_inputs[idx][1]
                        else:
                            # Subsequent stages input is previous stage mapped output
                            y_i, p_i = current_inputs[idx][0], current_inputs[idx][1]
                            
                        if s_degree == 1:
                            feat = np.array([y_i, p_i, 1.0])
                        else:
                            feat = np.array([y_i, p_i, y_i * y_i, p_i * p_i, y_i * p_i, 1.0])
                        
                        pred = feat @ W_stage
                        next_inputs.append([float(pred[0]), float(pred[1])])
                    current_inputs = next_inputs

                s1_arr = np.array(current_inputs)
                s1_x = s1_arr[:, 0]
                s1_y = s1_arr[:, 1]
                
                # Fit the next stage
                if unique_targets <= 5:
                    poly_degree = 1
                else:
                    poly_degree = 2 if N >= 6 else 1

                if poly_degree == 1:
                    X = np.column_stack([s1_x, s1_y, np.ones(N)])
                else:
                    X = np.column_stack([
                        s1_x,
                        s1_y,
                        s1_x * s1_x,
                        s1_y * s1_y,
                        s1_x * s1_y,
                        np.ones(N)
                    ])
                    
                Y = np.array(target_list)
                alpha = 1e-4
                XT_X = X.T @ X
                I = np.eye(X.shape[1])
                I[-1, -1] = 0.0
                W2 = np.linalg.solve(XT_X + alpha * I, X.T @ Y)
                
                pred_Y = X @ W2
                
                stages = list(stages) + [{
                    "stage": len(stages) + 1,
                    "W": W2.tolist(),
                    "poly_degree": poly_degree,
                    "mean_px_error": 0.0
                }]

            # Calculate validation error
            errors = []
            target_to_preds = {}
            for i in range(N):
                w_w = viewport_list[i][0]
                h_h = viewport_list[i][1]
                
                pred_x_px = (pred_Y[i, 0] + 1.0) * 0.5 * w_w
                pred_y_px = (pred_Y[i, 1] + 1.0) * 0.5 * h_h
                
                target_x_px = (Y[i, 0] + 1.0) * 0.5 * w_w
                target_y_px = (Y[i, 1] + 1.0) * 0.5 * h_h
                
                err = np.sqrt((pred_x_px - target_x_px)**2 + (pred_y_px - target_y_px)**2)
                errors.append(err)

                # Group predictions by unique normalized targets for noise analysis
                t_norm_tuple = (float(Y[i, 0]), float(Y[i, 1]))
                if t_norm_tuple not in target_to_preds:
                    target_to_preds[t_norm_tuple] = []
                target_to_preds[t_norm_tuple].append([pred_x_px, pred_y_px])

            mean_px_error = float(np.mean(errors))
            stages[-1]["mean_px_error"] = mean_px_error

            # Calculate coordinate standard deviation (noise floor) across target clusters
            std_devs = []
            for t_norm, preds in target_to_preds.items():
                if len(preds) > 1:
                    preds_arr = np.array(preds)
                    std_x = np.std(preds_arr[:, 0])
                    std_y = np.std(preds_arr[:, 1])
                    std_devs.append(float(np.sqrt(std_x**2 + std_y**2)))
            noise_level = float(np.mean(std_devs)) if len(std_devs) > 0 else 0.0

            # Save under designated name in runs/
            output_model_path = runs_root / f"{request.output_model_name}.json"
            calibration_data = {
                "stages": stages,
                "mean_px_error": mean_px_error,
                "noise_level": noise_level,
                "train_samples": N
            }
            with output_model_path.open("w", encoding="utf-8") as handle:
                json.dump(calibration_data, handle, indent=2)

            return {
                "ok": True,
                "best_val_px_error": mean_px_error,
                "noise_level": noise_level,
                "train_samples": N,
                "val_samples": 0,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/predict")
    def predict_gaze(request: PredictRequest) -> dict:
        # Load the base model with thread-safe caching (shared across all sessions to save RAM!)
        with _model_cache_lock:
            base_model = _model_cache.get("base_model")
            if base_model is None:
                try:
                    import torch
                    from .model import UniGazeFeatureWrapper, device_from_arg, load_unigaze_b16

                    device = device_from_arg("auto")
                    base_model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()
                    _model_cache["base_model"] = base_model
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"failed to load base model: {exc}")

        # Decode image
        try:
            image = _decode_image(request.image_data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Preprocess and predict
        try:
            import torch
            import numpy as np
            import json
            from .transforms import to_unigaze_tensor

            processed = preprocessor.process(image)
            device = next(base_model.parameters()).device
            image_tensor = to_unigaze_tensor(processed.image_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                gaze = base_model(image_tensor)
                gaze = gaze.squeeze(0).cpu().tolist()  # [pitch, yaw]

            # If testing before training, project the raw gaze angles (radians) directly
            if request.model_name == "before":
                # Raw gaze is [pitch, yaw] in radians
                pitch, yaw = gaze[0], gaze[1]
                
                # Standard linear scaling to map raw radians to [-1, 1] screen coordinates
                scale_x = 4.5
                scale_y = 4.5
                
                pred_x = max(-1.0, min(1.0, yaw * scale_x))
                pred_y = max(-1.0, min(1.0, pitch * scale_y))
                pred_xy = [pred_x, pred_y]
            else:
                # Calibrated mode using the selected regression matrix from runs/
                runs_root = root / "runs"
                calibration_path = runs_root / f"{request.model_name}.json"
                if not calibration_path.exists():
                    raise HTTPException(status_code=400, detail="Model not found. Please train/calibrate first.")

                with calibration_path.open("r", encoding="utf-8") as handle:
                    cal_data = json.load(handle)

                if "stages" in cal_data:
                    stages = cal_data["stages"]
                else:
                    # Migrate legacy schema
                    stages = [{
                        "stage": 1,
                        "W": cal_data["W"],
                        "poly_degree": cal_data.get("poly_degree", 2)
                    }]

                # Chain stages sequentially
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

            return {
                "ok": True,
                "screen_xy_norm": pred_xy,
                "gaze_pitch_yaw": gaze,
                "head_pose_pitch_yaw": processed.head_pose_pitch_yaw.tolist(),
                "face_bbox": processed.face_bbox,
            }
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": f"prediction failed: {exc}"}

    @app.websocket("/api/predict/ws")
    async def predict_gaze_ws(websocket: WebSocket):
        await websocket.accept()
        model_name = "before"

        # Load the base model with thread-safe caching
        with _model_cache_lock:
            base_model = _model_cache.get("base_model")
            if base_model is None:
                try:
                    import torch
                    from .model import UniGazeFeatureWrapper, device_from_arg, load_unigaze_b16

                    device = device_from_arg("auto")
                    base_model = UniGazeFeatureWrapper(load_unigaze_b16(device)).to(device).eval()
                    _model_cache["base_model"] = base_model
                except Exception as exc:
                    await websocket.send_json({"ok": False, "error": f"failed to load base model: {exc}"})
                    await websocket.close()
                    return

        try:
            import torch
            import numpy as np
            import json
            from .transforms import to_unigaze_tensor

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                
                if "text" in message:
                    try:
                        cfg = json.loads(message["text"])
                        if "model_name" in cfg:
                            model_name = cfg["model_name"]
                    except Exception as exc:
                        await websocket.send_json({"ok": False, "error": f"invalid config: {exc}"})
                        
                elif "bytes" in message:
                    t_start = time.perf_counter()
                    raw_bytes = message["bytes"]
                    try:
                        image = _decode_binary_image(raw_bytes)
                    except Exception as exc:
                        await websocket.send_json({"ok": False, "error": f"image decode failed: {exc}"})
                        continue
                    t_decode = (time.perf_counter() - t_start) * 1000.0

                    try:
                        t_mp_start = time.perf_counter()
                        processed = preprocessor.process(image)
                        t_mediapipe = (time.perf_counter() - t_mp_start) * 1000.0

                        t_nn_start = time.perf_counter()
                        device = next(base_model.parameters()).device
                        image_tensor = to_unigaze_tensor(processed.image_rgb).unsqueeze(0).to(device)

                        with torch.no_grad():
                            gaze = base_model(image_tensor)
                            gaze = gaze.squeeze(0).cpu().tolist()  # [pitch, yaw]

                        if model_name == "before":
                            pitch, yaw = gaze[0], gaze[1]
                            scale_x = 4.5
                            scale_y = 4.5
                            pred_x = max(-1.0, min(1.0, yaw * scale_x))
                            pred_y = max(-1.0, min(1.0, pitch * scale_y))
                            pred_xy = [pred_x, pred_y]
                        else:
                            runs_root = root / "runs"
                            calibration_path = runs_root / f"{model_name}.json"
                            if not calibration_path.exists():
                                await websocket.send_json({"ok": False, "error": f"model {model_name} not found"})
                                continue

                            with calibration_path.open("r", encoding="utf-8") as handle:
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

                        t_nn = (time.perf_counter() - t_nn_start) * 1000.0
                        t_total = (time.perf_counter() - t_start) * 1000.0
                        fps = 1000.0 / t_total if t_total > 0 else 0.0
                        print(
                            f"[WebSocket Info] Decode: {t_decode:.1f}ms | "
                            f"MediaPipe Face: {t_mediapipe:.1f}ms | "
                            f"Gaze NN: {t_nn:.1f}ms | "
                            f"Total: {t_total:.1f}ms | "
                            f"FPS: {fps:.1f}"
                        )

                        await websocket.send_json({
                            "ok": True,
                            "screen_xy_norm": pred_xy,
                            "gaze_pitch_yaw": gaze,
                            "head_pose_pitch_yaw": processed.head_pose_pitch_yaw.tolist(),
                            "face_bbox": processed.face_bbox,
                        })
                    except Exception as exc:
                        await websocket.send_json({"ok": False, "error": f"prediction failed: {exc}"})
        except WebSocketDisconnect:
            pass

    return app


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UniGaze webcam calibration collector.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="data/sessions")
    return parser


def main() -> None:
    import uvicorn

    args = build_argparser().parse_args()
    app = create_app(args.data_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
