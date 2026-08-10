"""Offline-only resolution of local UniGaze preprocessing assets."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FACE_LANDMARKER_CANDIDATES = (
    Path("web/static/face_landmarker.task"),
    Path("archive/shengwen/face_landmarker.task"),
)


def resolve_face_landmarker_asset(root: str | Path | None = None) -> Path:
    """Return a repository-local model asset without downloading anything."""

    repository_root = Path(root or REPOSITORY_ROOT).resolve()
    for relative_path in FACE_LANDMARKER_CANDIDATES:
        candidate = (repository_root / relative_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(repository_root):
            return candidate
    raise FileNotFoundError(
        "face_landmarker.task is unavailable in the repository; "
        "network download is disabled"
    )


__all__ = [
    "FACE_LANDMARKER_CANDIDATES",
    "resolve_face_landmarker_asset",
]
