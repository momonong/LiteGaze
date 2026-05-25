from __future__ import annotations

from pathlib import Path

from .model_registry import model_path


def predict(root: Path, payload: dict) -> tuple[dict, int]:
    model_name = payload.get("model_name", "before")
    viewport_width = float(payload.get("viewport_width", 0) or 0)
    viewport_height = float(payload.get("viewport_height", 0) or 0)

    if model_name != "before" and not model_path(root, model_name).exists():
        return {"ok": False, "error": "model not found"}, 404

    # First integration pass: keep the HTTP contract stable while the real
    # UniGaze pipeline is wired into this module.
    return {
        "ok": True,
        "screen_xy_norm": [0.0, 0.0],
        "screen_xy_px": [viewport_width / 2.0, viewport_height / 2.0],
        "model_name": model_name,
        "source": "placeholder",
    }, 200
