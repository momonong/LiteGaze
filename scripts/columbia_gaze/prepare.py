"""One-pass CPU preparation of frozen Columbia candidate and production inputs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.columbia_gaze.data import (
    EXPECTED_HEIGHT,
    EXPECTED_IMAGES,
    EXPECTED_WIDTH,
    MediaPipeEyeCornerFallback,
    SourceBundle,
    candidate_eye_pair,
)


def prepare_inputs(
    *,
    bundle: SourceBundle,
    work_dir: Path,
    landmark_model_path: Path,
    integrity: dict[str, str],
) -> tuple[Path, Path, dict[str, Any]]:
    """Decode once, prepare both paths, and persist integrity-bound local caches."""
    cache_id = hashlib.sha256(
        json.dumps(integrity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = work_dir / f"prepared-{cache_id}"
    manifest_path = cache_dir / "manifest.json"
    candidate_path = cache_dir / "candidate_inputs.npz"
    production_path = cache_dir / "production_faces.npy"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("integrity") != integrity:
            raise ValueError("prepared-input manifest integrity mismatch")
        _validate_prepared_files(candidate_path, production_path)
        return candidate_path, production_path, manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    count = len(bundle.samples)
    if count != EXPECTED_IMAGES:
        raise ValueError("prepared input requires the full Columbia grid")
    eye_images = np.empty((count, 2, 1, 36, 60), dtype=np.uint8)
    raw_eye_poses = np.empty((count, 2, 2), dtype=np.float32)
    targets = np.empty((count, 2), dtype=np.float32)
    subject_indices = np.empty(count, dtype=np.int16)
    head_poses = np.empty(count, dtype=np.int8)
    vertical_gazes = np.empty(count, dtype=np.int8)
    horizontal_gazes = np.empty(count, dtype=np.int8)
    fallback_mask = np.zeros(count, dtype=bool)
    production_success = np.zeros(count, dtype=bool)
    subject_order = sorted({sample.subject for sample in bundle.samples})
    subject_to_index = {subject: index for index, subject in enumerate(subject_order)}

    production_temp = production_path.with_suffix(".npy.tmp")
    production_faces = np.lib.format.open_memmap(
        production_temp,
        mode="w+",
        dtype=np.uint8,
        shape=(count, 224, 224, 3),
    )
    production_faces[:] = 0
    from core.unigaze_personalization.preprocess import MediaPipeUniGazePreprocessor

    production_preprocessor = MediaPipeUniGazePreprocessor()
    fallback = MediaPipeEyeCornerFallback(landmark_model_path)
    production_failures: Counter[str] = Counter()
    fallback_failures: Counter[str] = Counter()
    fallback_attempted = 0
    fallback_succeeded = 0
    started = time.monotonic()
    try:
        for index, sample in enumerate(bundle.samples):
            image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
            if image is None or image.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
                raise ValueError("audited Columbia image changed during preparation")

            corners = bundle.official_corners.get(sample.identity)
            if corners is None:
                if sample.identity not in bundle.missing_annotation_identities:
                    raise ValueError("unexpected annotation absence during preparation")
                fallback_mask[index] = True
                fallback_attempted += 1
                try:
                    corners = fallback.detect(image)
                    fallback_succeeded += 1
                except Exception as exc:
                    fallback_failures[type(exc).__name__] += 1
                    raise ValueError("frozen eye-corner fallback failed") from exc
            left, right = candidate_eye_pair(image, corners)
            eye_images[index, 0, 0] = left
            eye_images[index, 1, 0] = right
            base_pose = sample.head_pitch_yaw
            raw_eye_poses[index, 0] = base_pose
            raw_eye_poses[index, 1] = (base_pose[0], -base_pose[1])
            targets[index] = sample.target_pitch_yaw
            subject_indices[index] = subject_to_index[sample.subject]
            head_poses[index] = sample.head_pose_degrees
            vertical_gazes[index] = sample.vertical_gaze_degrees
            horizontal_gazes[index] = sample.horizontal_gaze_degrees

            try:
                resized_width = 640
                resized_height = round(EXPECTED_HEIGHT * resized_width / EXPECTED_WIDTH)
                resized = cv2.resize(
                    image,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )
                processed = production_preprocessor.process(resized)
                normalized = np.asarray(processed.image_rgb, dtype=np.uint8)
                if normalized.shape != (224, 224, 3):
                    raise ValueError("unexpected production normalized-face shape")
                production_faces[index] = normalized
                production_success[index] = True
            except Exception as exc:
                production_failures[_failure_code(exc)] += 1

            if (index + 1) % 100 == 0 or index + 1 == count:
                elapsed = time.monotonic() - started
                print(
                    f"PREPARE rows={index + 1}/{count} "
                    f"production_ok={int(production_success[: index + 1].sum())} "
                    f"seconds={elapsed:.1f}",
                    flush=True,
                )
    finally:
        fallback.close()
        face_mesh = getattr(production_preprocessor, "_face_mesh", None)
        if face_mesh is not None and hasattr(face_mesh, "close"):
            face_mesh.close()
        production_faces.flush()
        del production_faces

    if fallback_attempted != 15 or fallback_succeeded != 15 or fallback_failures:
        raise ValueError("frozen fallback completeness gate failed")
    candidate_temp = candidate_path.with_suffix(".npz.tmp")
    with candidate_temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            eye_images=eye_images,
            raw_eye_poses=raw_eye_poses,
            targets=targets,
            subject_indices=subject_indices,
            head_poses=head_poses,
            vertical_gazes=vertical_gazes,
            horizontal_gazes=horizontal_gazes,
            fallback_mask=fallback_mask,
            production_success=production_success,
        )
    os.replace(candidate_temp, candidate_path)
    os.replace(production_temp, production_path)
    duration = time.monotonic() - started
    success_count = int(production_success.sum())
    manifest = {
        "schema_version": 1,
        "integrity": dict(integrity),
        "rows": count,
        "duration_seconds": duration,
        "candidate": {
            "rows": count,
            "eyes_per_row": 2,
            "official_annotation_rows": int((~fallback_mask).sum()),
            "fallback": {
                "eligible": len(bundle.missing_annotation_identities),
                "attempted": fallback_attempted,
                "succeeded": fallback_succeeded,
                "failed": sum(fallback_failures.values()),
                "failure_reasons": dict(sorted(fallback_failures.items())),
            },
        },
        "production": {
            "attempted": count,
            "succeeded": success_count,
            "failed": count - success_count,
            "coverage": success_count / count,
            "failure_reasons": dict(sorted(production_failures.items())),
        },
    }
    _atomic_json(manifest_path, manifest)
    _validate_prepared_files(candidate_path, production_path)
    return candidate_path, production_path, manifest


def _validate_prepared_files(candidate_path: Path, production_path: Path) -> None:
    if not candidate_path.is_file() or not production_path.is_file():
        raise FileNotFoundError("prepared Columbia input cache is incomplete")
    with np.load(candidate_path, allow_pickle=False) as candidate:
        if candidate["eye_images"].shape != (EXPECTED_IMAGES, 2, 1, 36, 60):
            raise ValueError("cached candidate eye shape mismatch")
        if candidate["raw_eye_poses"].shape != (EXPECTED_IMAGES, 2, 2):
            raise ValueError("cached candidate pose shape mismatch")
        if candidate["targets"].shape != (EXPECTED_IMAGES, 2):
            raise ValueError("cached candidate target shape mismatch")
    production = np.load(production_path, mmap_mode="r", allow_pickle=False)
    if production.shape != (EXPECTED_IMAGES, 224, 224, 3):
        raise ValueError("cached production face shape mismatch")


def _failure_code(exc: Exception) -> str:
    text = str(exc).lower()
    if "no face detected" in text:
        return "no_face_detected"
    if isinstance(exc, ValueError):
        return "value_error"
    return type(exc).__name__


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
