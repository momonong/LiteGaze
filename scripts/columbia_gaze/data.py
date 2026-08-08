"""Audited loading and frozen preprocessing for Columbia Gaze v2."""

from __future__ import annotations

import csv
import hashlib
import itertools
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

EXPECTED_HEAD_POSES = (-30, -15, 0, 15, 30)
EXPECTED_VERTICAL_GAZES = (-10, 0, 10)
EXPECTED_HORIZONTAL_GAZES = (-15, -10, -5, 0, 5, 10, 15)
EXPECTED_SUBJECTS = 56
EXPECTED_IMAGES_PER_SUBJECT = 105
EXPECTED_IMAGES = EXPECTED_SUBJECTS * EXPECTED_IMAGES_PER_SUBJECT
EXPECTED_WIDTH = 5184
EXPECTED_HEIGHT = 3456
OFFICIAL_ANNOTATIONS = 5865
MISSING_ANNOTATIONS = 15
FILENAME_PATTERN = re.compile(
    r"^(?P<subject>\d{4})_2m_(?P<head>-?\d+)P_"
    r"(?P<vertical>-?\d+)V_(?P<horizontal>-?\d+)H\.jpg$",
    re.IGNORECASE,
)
ANNOTATION_FIELDS = (
    "IMAGE",
    "RIGHT_EYE_IN_X",
    "RIGHT_EYE_IN_Y",
    "RIGHT_EYE_OUT_X",
    "RIGHT_EYE_OUT_Y",
    "LEFT_EYE_IN_X",
    "LEFT_EYE_IN_Y",
    "LEFT_EYE_OUT_X",
    "LEFT_EYE_OUT_Y",
)


@dataclass(frozen=True)
class ColumbiaSample:
    """One public Columbia image and its filename-encoded factors."""

    path: Path
    identity: str
    subject: str
    head_pose_degrees: int
    vertical_gaze_degrees: int
    horizontal_gaze_degrees: int

    @property
    def target_pitch_yaw(self) -> np.ndarray:
        return np.radians(
            [self.vertical_gaze_degrees, self.horizontal_gaze_degrees]
        ).astype(np.float32)

    @property
    def head_pitch_yaw(self) -> np.ndarray:
        return np.radians([0.0, self.head_pose_degrees]).astype(np.float32)


@dataclass(frozen=True)
class EyeCorners:
    """Subject-anatomical eye corners in original image coordinates."""

    right_in: tuple[float, float]
    right_out: tuple[float, float]
    left_in: tuple[float, float]
    left_out: tuple[float, float]


@dataclass(frozen=True)
class SourceBundle:
    """Validated source records retained in memory for a formal run."""

    samples: tuple[ColumbiaSample, ...]
    official_corners: dict[str, EyeCorners]
    missing_annotation_identities: frozenset[str]
    audit: dict[str, Any]


def parse_filename(path: Path) -> ColumbiaSample:
    """Parse and validate the exact public filename factor contract."""
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unparseable Columbia filename: {path.name}")
    subject = match.group("subject")
    head = int(match.group("head"))
    vertical = int(match.group("vertical"))
    horizontal = int(match.group("horizontal"))
    if head not in EXPECTED_HEAD_POSES:
        raise ValueError(f"unexpected head-pose label: {path.name}")
    if vertical not in EXPECTED_VERTICAL_GAZES:
        raise ValueError(f"unexpected vertical-gaze label: {path.name}")
    if horizontal not in EXPECTED_HORIZONTAL_GAZES:
        raise ValueError(f"unexpected horizontal-gaze label: {path.name}")
    if path.parent.name != subject:
        raise ValueError(f"subject directory mismatch: {path.name}")
    return ColumbiaSample(
        path=path,
        identity=path.stem,
        subject=subject,
        head_pose_degrees=head,
        vertical_gaze_degrees=vertical,
        horizontal_gaze_degrees=horizontal,
    )


def load_eye_corner_annotations(path: Path) -> dict[str, EyeCorners]:
    """Load the official 5,865-row eye-corner CSV with exact columns."""
    result: dict[str, EyeCorners] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ANNOTATION_FIELDS:
            raise ValueError("unexpected Columbia eye-corner CSV columns")
        for line_number, row in enumerate(reader, start=2):
            identity = str(row["IMAGE"]).strip()
            if not identity or identity in result:
                raise ValueError(
                    f"invalid or duplicate eye annotation at line {line_number}"
                )
            values = {
                field: _annotation_coordinate(row[field], field, line_number)
                for field in ANNOTATION_FIELDS[1:]
            }
            result[identity] = EyeCorners(
                right_in=(values["RIGHT_EYE_IN_X"], values["RIGHT_EYE_IN_Y"]),
                right_out=(
                    values["RIGHT_EYE_OUT_X"],
                    values["RIGHT_EYE_OUT_Y"],
                ),
                left_in=(values["LEFT_EYE_IN_X"], values["LEFT_EYE_IN_Y"]),
                left_out=(values["LEFT_EYE_OUT_X"], values["LEFT_EYE_OUT_Y"]),
            )
    return result


def audit_columbia_source(
    image_root: Path,
    annotation_csv: Path,
    *,
    decode_images: bool,
) -> SourceBundle:
    """Validate the exact image grid, annotations, and optional JPEG decoding."""
    paths = sorted(image_root.rglob("*.jpg"), key=lambda item: item.as_posix())
    samples = tuple(parse_filename(path) for path in paths)
    identities = [sample.identity for sample in samples]
    duplicate_count = len(identities) - len(set(identities))
    subjects = sorted({sample.subject for sample in samples})
    per_subject_counts = Counter(sample.subject for sample in samples)
    expected_cells = set(
        itertools.product(
            EXPECTED_HEAD_POSES,
            EXPECTED_VERTICAL_GAZES,
            EXPECTED_HORIZONTAL_GAZES,
        )
    )
    grid_failures = 0
    for subject in subjects:
        cells = {
            (
                sample.head_pose_degrees,
                sample.vertical_gaze_degrees,
                sample.horizontal_gaze_degrees,
            )
            for sample in samples
            if sample.subject == subject
        }
        if cells != expected_cells or per_subject_counts[subject] != len(
            expected_cells
        ):
            grid_failures += 1

    official = load_eye_corner_annotations(annotation_csv)
    identity_set = set(identities)
    annotation_set = set(official)
    missing = identity_set - annotation_set
    extra = annotation_set - identity_set
    corrupt = 0
    dimension_mismatch = 0
    if decode_images:
        for sample in samples:
            image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
            if image is None:
                corrupt += 1
                continue
            if image.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
                dimension_mismatch += 1

    audit = {
        "status": "passed",
        "subject_count": len(subjects),
        "image_count": len(samples),
        "images_per_subject_min": min(per_subject_counts.values(), default=0),
        "images_per_subject_max": max(per_subject_counts.values(), default=0),
        "duplicate_image_identity_count": duplicate_count,
        "subject_grid_failure_count": grid_failures,
        "official_annotation_count": len(official),
        "missing_annotation_count": len(missing),
        "extra_annotation_count": len(extra),
        "missing_annotation_identity_sha256": _identity_sha256(missing),
        "image_identity_sha256": _identity_sha256(identity_set),
        "decoded_all_images": bool(decode_images),
        "corrupt_image_count": corrupt,
        "dimension_mismatch_count": dimension_mismatch,
    }
    expected = (
        len(subjects) == EXPECTED_SUBJECTS
        and len(samples) == EXPECTED_IMAGES
        and min(per_subject_counts.values(), default=0) == EXPECTED_IMAGES_PER_SUBJECT
        and max(per_subject_counts.values(), default=0) == EXPECTED_IMAGES_PER_SUBJECT
        and duplicate_count == 0
        and grid_failures == 0
        and len(official) == OFFICIAL_ANNOTATIONS
        and len(missing) == MISSING_ANNOTATIONS
        and not extra
        and (not decode_images or (corrupt == 0 and dimension_mismatch == 0))
    )
    if not expected:
        audit["status"] = "failed"
        raise ValueError(f"Columbia source audit failed: {audit}")
    return SourceBundle(
        samples=samples,
        official_corners=official,
        missing_annotation_identities=frozenset(missing),
        audit=audit,
    )


def affine_eye_crop(
    image_bgr: np.ndarray,
    first_corner: tuple[float, float],
    second_corner: tuple[float, float],
) -> np.ndarray:
    """Apply the frozen two-corner affine eye normalization to 60x36 gray."""
    if image_bgr is None or image_bgr.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
        raise ValueError("eye crop requires one original 5184x3456 BGR image")
    pair = sorted(
        (
            np.asarray(first_corner, dtype=np.float32),
            np.asarray(second_corner, dtype=np.float32),
        ),
        key=lambda point: float(point[0]),
    )
    delta = pair[1] - pair[0]
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance) or distance < 2.0:
        raise ValueError("eye corners are degenerate")
    source = np.stack((pair[0], pair[1], pair[0] + np.array([-delta[1], delta[0]])))
    target = np.asarray(((15.0, 18.0), (45.0, 18.0), (15.0, 48.0)), dtype=np.float32)
    transform = cv2.getAffineTransform(source.astype(np.float32), target)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    crop = cv2.warpAffine(
        gray,
        transform,
        (60, 36),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape != (36, 60) or crop.dtype != np.uint8:
        raise ValueError("unexpected normalized eye crop")
    return crop


def candidate_eye_pair(
    image_bgr: np.ndarray,
    corners: EyeCorners,
) -> tuple[np.ndarray, np.ndarray]:
    """Return anatomical left plus flipped anatomical right eye crops."""
    left = affine_eye_crop(image_bgr, corners.left_in, corners.left_out)
    right = affine_eye_crop(image_bgr, corners.right_in, corners.right_out)
    right_flipped = np.ascontiguousarray(right[:, ::-1])
    return left, right_flipped


class MediaPipeEyeCornerFallback:
    """One-attempt v2 fallback for the 15 officially unannotated images."""

    def __init__(self, model_path: Path, *, confidence: float = 0.5) -> None:
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

        if not model_path.is_file():
            raise FileNotFoundError(f"pinned face landmarker is missing: {model_path}")
        options = FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
            num_faces=1,
            min_face_detection_confidence=confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> EyeCorners:
        """Detect fixed landmarks once at width 1280 and map to original pixels."""
        from mediapipe import Image, ImageFormat

        if image_bgr.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
            raise ValueError("fallback requires one original Columbia image")
        resized_width = 1280
        resized_height = round(EXPECTED_HEIGHT * resized_width / EXPECTED_WIDTH)
        resized = cv2.resize(
            image_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
        if len(result.face_landmarks) != 1:
            raise ValueError("fallback did not detect exactly one face")
        landmarks = result.face_landmarks[0]
        scale_x = EXPECTED_WIDTH / resized_width
        scale_y = EXPECTED_HEIGHT / resized_height

        def point(index: int) -> tuple[float, float]:
            landmark = landmarks[index]
            return (
                float(landmark.x * resized_width * scale_x),
                float(landmark.y * resized_height * scale_y),
            )

        return EyeCorners(
            right_in=point(133),
            right_out=point(33),
            left_in=point(362),
            left_out=point(263),
        )

    def close(self) -> None:
        self._landmarker.close()


def _annotation_coordinate(raw: str | None, field: str, line_number: int) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} at annotation line {line_number}") from exc
    limit = EXPECTED_WIDTH if field.endswith("_X") else EXPECTED_HEIGHT
    if not np.isfinite(value) or value < 0.0 or value >= limit:
        raise ValueError(f"out-of-frame {field} at annotation line {line_number}")
    return value


def _identity_sha256(identities: set[str] | frozenset[str]) -> str:
    digest = hashlib.sha256()
    for identity in sorted(identities):
        digest.update(identity.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
