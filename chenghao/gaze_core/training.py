from __future__ import annotations

import json
import time
from pathlib import Path

from .model_registry import clean_model_name, model_path
from .sample_store import ensure_sessions_dir


def train_placeholder(root: Path, payload: dict) -> tuple[dict, int]:
    dataset_id = payload.get("data_session_id", "")
    output_name = clean_model_name(payload.get("output_model_name", "gaze_model"))
    session_dir = ensure_sessions_dir(root) / dataset_id
    manifest = session_dir / "manifest.jsonl"
    if not dataset_id or not manifest.exists():
        return {"ok": False, "error": "dataset not found"}, 404

    sample_count = sum(1 for _ in manifest.open("r", encoding="utf-8"))
    if sample_count == 0:
        return {"ok": False, "error": "dataset has no samples"}, 400

    meta = {
        "name": output_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_session_id": dataset_id,
        "train_samples": sample_count,
        "mean_px_error": 0.0,
        "backend": "chenghao-placeholder",
        "note": "This file reserves the model contract. Replace gaze_core/inference.py and training.py with the real model pipeline.",
    }
    path = model_path(root, output_name)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "model_name": output_name, "train_samples": sample_count, "best_val_px_error": 0.0}, 200
