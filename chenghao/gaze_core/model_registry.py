from __future__ import annotations

import json
from pathlib import Path


def ensure_runs_dir(root: Path) -> Path:
    runs_dir = root / "gaze_data" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def clean_model_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return cleaned or "gaze_model"


def list_models(root: Path) -> list[dict]:
    runs_dir = ensure_runs_dir(root)
    models = []
    for file in sorted(runs_dir.glob("*.json"), key=lambda p: p.name):
        try:
            meta = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        name = file.stem
        mean_px_error = float(meta.get("mean_px_error", 0.0) or 0.0)
        train_samples = int(meta.get("train_samples", 0) or 0)
        models.append(
            {
                "name": name,
                "display_name": f"{name} ({train_samples} samples, {mean_px_error:.1f} px)",
                "mean_px_error": mean_px_error,
                "num_stages": int(meta.get("num_stages", 1) or 1),
                "noise_level": float(meta.get("noise_level", 0.0) or 0.0),
                "train_samples": train_samples,
                "created_at": meta.get("created_at", ""),
            }
        )
    return models


def model_path(root: Path, model_name: str) -> Path:
    return ensure_runs_dir(root) / f"{clean_model_name(model_name)}.json"
