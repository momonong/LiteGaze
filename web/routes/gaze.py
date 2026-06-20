from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from core.gaze_core.inference import predict
from core.gaze_core.model_registry import ensure_runs_dir, list_models, delete_model, rename_model
from core.gaze_core.sample_store import create_session, list_datasets, save_sample, delete_dataset, rename_dataset
from core.gaze_core.training import train_placeholder


ROOT = Path(__file__).resolve().parents[2]
gaze_bp = Blueprint("gaze", __name__, url_prefix="/api/gaze")
gaze_api_bp = Blueprint("gaze_api", __name__, url_prefix="/api")


@gaze_bp.get("/health")
def health():
    ensure_runs_dir(ROOT)
    return jsonify({"ok": True, "backend": "chenghao-gaze", "mode": "http-polling"})


@gaze_bp.get("/models")
def models():
    return jsonify({"ok": True, "models": list_models(ROOT)})


@gaze_bp.get("/datasets")
def datasets():
    return jsonify({"ok": True, "datasets": list_datasets(ROOT)})


@gaze_bp.post("/session")
def session():
    body = request.get_json(force=True) or {}
    return jsonify(create_session(ROOT, body.get("participant_id", "anonymous")))


@gaze_bp.post("/sample")
def sample():
    body = request.get_json(force=True) or {}
    response, status = save_sample(ROOT, body)
    return jsonify(response), status


@gaze_bp.post("/train")
def train():
    body = request.get_json(force=True) or {}
    response, status = train_placeholder(ROOT, body)
    return jsonify(response), status


@gaze_bp.post("/predict")
def predict_gaze():
    body = request.get_json(force=True) or {}
    response, status = predict(ROOT, body)
    return jsonify(response), status


@gaze_bp.route("/datasets/<session_id>", methods=["DELETE", "PUT"])
def dataset_ops(session_id):
    if request.method == "DELETE":
        return jsonify(delete_dataset(ROOT, session_id))
    body = request.get_json(force=True) or {}
    new_name = body.get("new_name", "")
    if not new_name:
        return jsonify({"ok": False, "error": "new_name required"}), 400
    return jsonify(rename_dataset(ROOT, session_id, new_name))


@gaze_bp.post("/datasets/<session_id>/reprocess")
def reprocess_dataset(session_id):
    """Re-run MediaPipe on raw frames for sessions that had face detection failures."""
    import json as _json
    import cv2 as _cv2
    from core.gaze_core.sample_store import get_preprocessor, ensure_sessions_dir

    session_dir = ensure_sessions_dir(ROOT) / session_id
    manifest_path = session_dir / "manifest.jsonl"
    if not session_dir.exists() or not manifest_path.exists():
        return jsonify({"ok": False, "error": "session not found"}), 404

    try:
        preprocessor = get_preprocessor()
        records = [_json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        fixed = 0
        still_failed = 0
        updated = []
        for rec in records:
            if rec.get("normalized_face_path"):  # already processed
                updated.append(rec)
                continue
            raw_p = session_dir / rec.get("raw_path", "")
            img = _cv2.imread(str(raw_p), _cv2.IMREAD_COLOR)
            if img is None:
                still_failed += 1
                updated.append(rec)
                continue
            try:
                processed = preprocessor.process(img)
                stem = Path(rec["raw_path"]).stem
                crop_path = session_dir / "crop" / f"{stem}.jpg"
                norm_path = session_dir / "normalized_face" / f"{stem}.jpg"
                _cv2.imwrite(str(crop_path), processed.crop_bgr)
                _cv2.imwrite(str(norm_path), processed.image_bgr)
                rec["crop_path"] = crop_path.relative_to(session_dir).as_posix()
                rec["normalized_face_path"] = norm_path.relative_to(session_dir).as_posix()
                rec["head_pose_pitch_yaw"] = processed.head_pose_pitch_yaw.tolist()
                rec["face_bbox"] = processed.face_bbox
                rec.pop("warning", None)
                rec["face_detected"] = True
                rec["ok"] = True
                fixed += 1
            except Exception as exc:
                rec["warning"] = str(exc)
                still_failed += 1
            updated.append(rec)
        manifest_path.write_text(
            "\n".join(_json.dumps(r, ensure_ascii=False) for r in updated) + "\n",
            encoding="utf-8",
        )
        return jsonify({"ok": True, "fixed": fixed, "still_failed": still_failed, "total": len(updated)})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500



@gaze_bp.route("/models/<model_name>", methods=["DELETE", "PUT"])
def model_ops(model_name):
    if request.method == "DELETE":
        return jsonify(delete_model(ROOT, model_name))
    body = request.get_json(force=True) or {}
    new_name = body.get("new_name", "")
    if not new_name:
        return jsonify({"ok": False, "error": "new_name required"}), 400
    return jsonify(rename_model(ROOT, model_name, new_name))


@gaze_api_bp.get("/health")
def api_health():
    ensure_runs_dir(ROOT)
    return jsonify({"ok": True, "backend": "chenghao-gaze", "mode": "http-polling"})


@gaze_api_bp.get("/list_models")
def api_list_models():
    models_data = []
    for model in list_models(ROOT):
        models_data.append(
            {
                **model,
                "num_stages": model.get("num_stages", 1),
                "noise_level": model.get("noise_level", 0.0),
            }
        )
    return jsonify({"ok": True, "models": models_data})


@gaze_api_bp.get("/list_datasets")
def api_list_datasets():
    return jsonify({"ok": True, "datasets": list_datasets(ROOT)})


@gaze_api_bp.post("/session")
def api_session():
    body = request.get_json(force=True) or {}
    return jsonify(create_session(ROOT, body.get("participant_id", "anonymous")))


@gaze_api_bp.post("/sample")
def api_sample():
    body = request.get_json(force=True) or {}
    response, status = save_sample(ROOT, body)
    return jsonify(response), status


@gaze_api_bp.post("/train")
def api_train():
    body = request.get_json(force=True) or {}
    response, status = train_placeholder(ROOT, body)
    return jsonify(response), status


@gaze_api_bp.post("/predict")
def api_predict():
    body = request.get_json(force=True) or {}
    response, status = predict(ROOT, body)
    return jsonify(response), status
