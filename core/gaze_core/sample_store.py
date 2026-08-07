from __future__ import annotations

import base64
import binascii
import json
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from .motion_robustness import capture_metadata

# Thread-safe locks
_preprocessor_lock = threading.Lock()
_manifest_lock = threading.Lock()
_preprocessor = None
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        with _preprocessor_lock:
            if _preprocessor is None:
                from core.unigaze_personalization.preprocess import (
                    MediaPipeUniGazePreprocessor,
                )

                _preprocessor = MediaPipeUniGazePreprocessor()
    return _preprocessor


def ensure_sessions_dir(root: Path) -> Path:
    sessions_dir = root / "data" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def clean_id(value: object) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return cleaned or "anonymous"


def _bounded_metadata_text(value: object, *, default: str = "") -> str:
    text = str(value or "").strip()
    return (text or default)[:128]


def safe_session_dir(
    root: Path, session_id: object, *, require_exists: bool = False
) -> Path:
    sessions_dir = ensure_sessions_dir(root).resolve()
    safe_id = str(session_id or "").strip()
    if not safe_id or safe_id != clean_id(safe_id):
        raise ValueError("invalid session ID")
    session_dir = (sessions_dir / safe_id).resolve()
    if session_dir.parent != sessions_dir:
        raise ValueError("invalid session path")
    if require_exists and not session_dir.is_dir():
        raise FileNotFoundError("session not found")
    return session_dir


def read_session_metadata(root: Path, session_id: object) -> dict:
    session_dir = safe_session_dir(root, session_id, require_exists=True)
    path = session_dir / "session.json"
    if not path.exists():
        raise FileNotFoundError("session metadata not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid session metadata")
    return payload


def create_session(
    root: Path,
    participant_id: str,
    *,
    capture_run_id: str | None = None,
    capture_source: str = "direct-frame",
    source_session_id: str | None = None,
    study_metadata: dict[str, object] | None = None,
) -> dict:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_{clean_id(participant_id)}_{uuid.uuid4().hex[:8]}"
    session_dir = safe_session_dir(root, session_id)
    resolved_capture_run_id = _bounded_metadata_text(
        capture_run_id,
        default=f"capture-{uuid.uuid4().hex}",
    )

    # Create required subdirectories for calibration pipeline
    (session_dir / "raw").mkdir(parents=True, exist_ok=True)
    (session_dir / "crop").mkdir(parents=True, exist_ok=True)
    (session_dir / "normalized_face").mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": session_id,
        "participant_id": clean_id(participant_id),
        "created_at": timestamp,
        "capture_run_id": resolved_capture_run_id,
        "capture_source": _bounded_metadata_text(
            capture_source,
            default="direct-frame",
        ),
    }
    resolved_source_session_id = _bounded_metadata_text(source_session_id)
    if resolved_source_session_id:
        meta["source_session_id"] = resolved_source_session_id
    if study_metadata:
        allowed_study_keys = {
            "study_session_id",
            "study_protocol_id",
            "study_protocol_version",
            "study_consent_version",
            "study_consent_digest_sha256",
            "study_mode",
        }
        for key in sorted(allowed_study_keys):
            value = _bounded_metadata_text(study_metadata.get(key))
            if value:
                meta[key] = value
    (session_dir / "session.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "session_id": session_id,
        "capture_run_id": resolved_capture_run_id,
    }


def list_datasets(root: Path) -> list[dict]:
    datasets = []
    sessions_dir = ensure_sessions_dir(root)
    if not sessions_dir.exists():
        return datasets
    for folder in sorted(sessions_dir.iterdir(), key=lambda p: p.name, reverse=True):
        manifest = folder / "manifest.jsonl"
        if not folder.is_dir() or not manifest.exists():
            continue
        try:
            sample_count = sum(1 for _ in manifest.open("r", encoding="utf-8"))
        except Exception:
            sample_count = 0
        participant = "unknown"
        capture_run_id = None
        capture_source = None
        source_session_id = None
        session_json = folder / "session.json"
        if session_json.exists():
            try:
                session_meta = json.loads(session_json.read_text(encoding="utf-8"))
                participant = session_meta.get("participant_id", "unknown")
                capture_run_id = session_meta.get("capture_run_id")
                capture_source = session_meta.get("capture_source")
                source_session_id = session_meta.get("source_session_id")
            except Exception:
                pass
        datasets.append(
            {
                "id": folder.name,
                "display_name": f"{participant} ({sample_count} samples)",
                "participant": participant,
                "sample_count": sample_count,
                "capture_run_id": capture_run_id,
                "capture_source": capture_source,
                "source_session_id": source_session_id,
            }
        )
    return datasets


def save_sample(root: Path, payload: dict) -> tuple[dict, int]:
    session_id = payload.get("session_id", "")
    try:
        session_dir = safe_session_dir(root, session_id, require_exists=True)
    except (FileNotFoundError, ValueError):
        return {"ok": False, "error": "session not found"}, 404

    image_data = payload.get("image_data", "")
    if not isinstance(image_data, str):
        return {"ok": False, "error": "invalid image data"}, 400
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data, validate=True)
    except (binascii.Error, TypeError, ValueError):
        return {"ok": False, "error": "invalid image data"}, 400
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        return {"ok": False, "error": "image payload size is invalid"}, 413

    # Ensure thread-safe index reading and writing to manifest
    with _manifest_lock:
        manifest = session_dir / "manifest.jsonl"
        sample_index = 0
        if manifest.exists():
            sample_index = sum(1 for _ in manifest.open("r", encoding="utf-8"))

        phase = payload.get("phase", "calibration")
        point_index = int(payload.get("point_index", 0) or 0)
        repeat_index = int(payload.get("repeat_index", 0) or 0)
        stem = f"{sample_index:06d}_{phase}_{point_index:02d}_{repeat_index:02d}"

        # Save raw frame
        raw_path = session_dir / "raw" / f"{stem}.jpg"
        raw_path.write_bytes(raw)

        viewport_width = float(payload.get("viewport_width", 1920) or 1920)
        viewport_height = float(payload.get("viewport_height", 1080) or 1080)
        target_x = float(payload.get("target_x") or 0.0)
        target_y = float(payload.get("target_y") or 0.0)

        target_x_norm = payload.get("target_x_norm")
        if target_x_norm is None:
            target_x_norm = (target_x / viewport_width) * 2.0 - 1.0
        else:
            target_x_norm = float(target_x_norm)

        target_y_norm = payload.get("target_y_norm")
        if target_y_norm is None:
            target_y_norm = (target_y / viewport_height) * 2.0 - 1.0
        else:
            target_y_norm = float(target_y_norm)

        record = {
            "ok": True,
            "sample_index": sample_index,
            "phase": phase,
            "point_index": point_index,
            "repeat_index": repeat_index,
            "target_x": target_x,
            "target_y": target_y,
            "target_x_norm": target_x_norm,
            "target_y_norm": target_y_norm,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "screen_width": payload.get("screen_width"),
            "screen_height": payload.get("screen_height"),
            "raw_path": raw_path.relative_to(session_dir).as_posix(),
            "created_at_unix": time.time(),
        }
        record.update(capture_metadata(payload))

        # Decode image and run the MediaPipe facial-normalization preprocessor.
        try:
            np_arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cannot decode image")

            preprocessor = get_preprocessor()
            processed = preprocessor.process(img)

            crop_path = session_dir / "crop" / f"{stem}.jpg"
            norm_path = session_dir / "normalized_face" / f"{stem}.jpg"

            cv2.imwrite(str(crop_path), processed.crop_bgr)
            cv2.imwrite(str(norm_path), processed.image_bgr)

            record.update(
                {
                    "crop_path": crop_path.relative_to(session_dir).as_posix(),
                    "normalized_face_path": norm_path.relative_to(
                        session_dir
                    ).as_posix(),
                    "head_pose_pitch_yaw": processed.head_pose_pitch_yaw.tolist(),
                    "face_bbox": processed.face_bbox,
                }
            )
        except Exception as exc:
            # Face detection failure on one frame should not abort the whole loop.
            # Mark this sample as skipped but keep ok=True so the JS loop continues.
            record["ok"] = True
            record["face_detected"] = False
            record["warning"] = str(exc)

        # Write manifest entry
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "ok": True,
        "sample_index": sample_index,
        "face_detected": record.get("face_detected", True),
        "warning": record.get("warning", ""),
    }, 200


def delete_dataset(root: Path, session_id: str) -> dict:
    import shutil

    try:
        session_dir = safe_session_dir(root, session_id, require_exists=True)
    except (FileNotFoundError, ValueError):
        return {"ok": False, "error": "dataset not found"}
    shutil.rmtree(session_dir)
    return {"ok": True}


def rename_dataset(root: Path, session_id: str, new_name: str) -> dict:
    try:
        session_dir = safe_session_dir(root, session_id, require_exists=True)
    except (FileNotFoundError, ValueError):
        return {"ok": False, "error": "dataset not found"}
    meta_file = session_dir / "session.json"
    if not meta_file.exists():
        return {"ok": False, "error": "dataset not found"}
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["participant_id"] = clean_id(new_name)
    meta_file.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True}


def purge_session_images(root: Path, session_id: str) -> dict[str, object]:
    """Delete imagery while retaining the derived manifest and audit metadata."""

    import shutil

    session_dir = safe_session_dir(root, session_id, require_exists=True)
    removed: list[str] = []
    for name in ("raw", "crop", "normalized_face"):
        target = (session_dir / name).resolve()
        if target.parent != session_dir:
            raise ValueError("invalid calibration image path")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(name)
    meta_path = session_dir / "session.json"
    metadata = read_session_metadata(root, session_id)
    metadata["calibration_images_purged_at_unix"] = time.time()
    metadata["calibration_images_retained"] = False
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "removed_directories": removed}
