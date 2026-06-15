from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from gaze_core.inference import predict
from gaze_core.model_registry import ensure_runs_dir, list_models, delete_model, rename_model
from gaze_core.sample_store import create_session, list_datasets, save_sample, delete_dataset, rename_dataset
from gaze_core.training import train_placeholder


ROOT = Path(__file__).parent
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
