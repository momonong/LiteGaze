from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from gaze_core.inference import predict
from gaze_core.model_registry import ensure_runs_dir, list_models
from gaze_core.sample_store import create_session, list_datasets, save_sample
from gaze_core.training import train_placeholder


ROOT = Path(__file__).parent
gaze_bp = Blueprint("gaze", __name__, url_prefix="/api/gaze")


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
