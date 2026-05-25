from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path


def ensure_sessions_dir(root: Path) -> Path:
    sessions_dir = root / "gaze_data" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def clean_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return cleaned or "anonymous"


def create_session(root: Path, participant_id: str) -> dict:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_{clean_id(participant_id)}_{uuid.uuid4().hex[:8]}"
    session_dir = ensure_sessions_dir(root) / session_id
    (session_dir / "raw").mkdir(parents=True, exist_ok=True)
    meta = {
        "session_id": session_id,
        "participant_id": participant_id or "anonymous",
        "created_at": timestamp,
    }
    (session_dir / "session.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "session_id": session_id}


def list_datasets(root: Path) -> list[dict]:
    datasets = []
    sessions_dir = ensure_sessions_dir(root)
    for folder in sorted(sessions_dir.iterdir(), key=lambda p: p.name, reverse=True):
        manifest = folder / "manifest.jsonl"
        if not folder.is_dir() or not manifest.exists():
            continue
        try:
            sample_count = sum(1 for _ in manifest.open("r", encoding="utf-8"))
        except Exception:
            sample_count = 0
        participant = "unknown"
        session_json = folder / "session.json"
        if session_json.exists():
            try:
                participant = json.loads(session_json.read_text(encoding="utf-8")).get(
                    "participant_id",
                    "unknown",
                )
            except Exception:
                pass
        datasets.append(
            {
                "id": folder.name,
                "display_name": f"{folder.name} ({participant}, {sample_count} samples)",
                "participant": participant,
                "sample_count": sample_count,
            }
        )
    return datasets


def save_sample(root: Path, payload: dict) -> dict:
    session_id = payload.get("session_id", "")
    session_dir = ensure_sessions_dir(root) / session_id
    if not session_id or not session_dir.exists():
        return {"ok": False, "error": "session not found"}, 404

    image_data = payload.get("image_data", "")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data)
    except Exception:
        return {"ok": False, "error": "invalid image data"}, 400

    manifest = session_dir / "manifest.jsonl"
    sample_index = 0
    if manifest.exists():
        sample_index = sum(1 for _ in manifest.open("r", encoding="utf-8"))

    phase = payload.get("phase", "calibration")
    point_index = int(payload.get("point_index", 0) or 0)
    repeat_index = int(payload.get("repeat_index", 0) or 0)
    stem = f"{sample_index:06d}_{phase}_{point_index:02d}_{repeat_index:02d}"
    raw_path = session_dir / "raw" / f"{stem}.jpg"
    raw_path.write_bytes(raw)

    record = {
        "sample_index": sample_index,
        "phase": phase,
        "point_index": point_index,
        "repeat_index": repeat_index,
        "target_x": payload.get("target_x"),
        "target_y": payload.get("target_y"),
        "target_x_norm": payload.get("target_x_norm"),
        "target_y_norm": payload.get("target_y_norm"),
        "viewport_width": payload.get("viewport_width"),
        "viewport_height": payload.get("viewport_height"),
        "raw_path": raw_path.relative_to(session_dir).as_posix(),
        "created_at_unix": time.time(),
    }
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"ok": True, "sample_index": sample_index}, 200
