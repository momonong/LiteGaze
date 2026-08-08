"""Audited MPIIGaze evaluation-subset loading for subject-heldout experiments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SampleRef:
    """One official MPIIGaze eye-image evaluation reference."""

    subject: str
    day: str
    filename: str
    side: str

    @property
    def identity(self) -> str:
        return f"{self.subject}/{self.day}/{self.filename}:{self.side}"


@dataclass(frozen=True)
class SubjectSplit:
    """One nested leave-one-subject-out fold."""

    fold: int
    train_subjects: tuple[str, ...]
    validation_subject: str
    test_subject: str


@dataclass
class EvaluationData:
    """In-memory balanced evaluation subset."""

    images: np.ndarray
    poses: np.ndarray
    targets: np.ndarray
    subject_ids: np.ndarray
    days: np.ndarray
    sides: np.ndarray
    references: tuple[SampleRef, ...]
    subjects: tuple[str, ...]

    def indices_for(self, selected_subjects: tuple[str, ...] | list[str]) -> np.ndarray:
        selected = {self.subjects.index(subject) for subject in selected_subjects}
        mask = np.fromiter(
            (int(subject_id) in selected for subject_id in self.subject_ids),
            dtype=bool,
            count=len(self.subject_ids),
        )
        return np.flatnonzero(mask)


def build_nested_subject_splits(
    subjects: list[str] | tuple[str, ...],
) -> list[SubjectSplit]:
    """Build the frozen outer-test/next-subject-validation split schedule."""
    ordered = tuple(subjects)
    if len(ordered) < 3 or len(set(ordered)) != len(ordered):
        raise ValueError("subjects must contain at least three unique identifiers")
    splits: list[SubjectSplit] = []
    for fold, test_subject in enumerate(ordered):
        validation_subject = ordered[(fold + 1) % len(ordered)]
        train_subjects = tuple(
            subject
            for subject in ordered
            if subject not in {test_subject, validation_subject}
        )
        split = SubjectSplit(
            fold=fold,
            train_subjects=train_subjects,
            validation_subject=validation_subject,
            test_subject=test_subject,
        )
        validate_subject_split(split, expected_subjects=ordered)
        splits.append(split)
    return splits


def validate_subject_split(
    split: SubjectSplit, *, expected_subjects: tuple[str, ...]
) -> None:
    """Reject subject overlap or omission before any model can run."""
    train = set(split.train_subjects)
    validation = {split.validation_subject}
    test = {split.test_subject}
    if train & validation or train & test or validation & test:
        raise ValueError(f"subject overlap detected in fold {split.fold}")
    observed = train | validation | test
    if observed != set(expected_subjects):
        raise ValueError(f"subject omission detected in fold {split.fold}")


def parse_sample_list(path: Path, subject: str) -> tuple[SampleRef, ...]:
    """Parse one official 3,000-row eye-image sample list."""
    references: list[SampleRef] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.replace("\\", "/").split()
        if len(parts) != 2 or parts[1] not in {"left", "right"}:
            raise ValueError(f"invalid sample reference at {path}:{line_number}")
        relative, side = parts
        reference_parts = Path(relative).parts
        if len(reference_parts) != 2:
            raise ValueError(f"invalid sample path at {path}:{line_number}")
        day, filename = reference_parts
        if not day.startswith("day") or not filename.lower().endswith(".jpg"):
            raise ValueError(f"unexpected sample path at {path}:{line_number}")
        references.append(
            SampleRef(
                subject=subject,
                day=day,
                filename=filename,
                side=side,
            )
        )
    return tuple(references)


def gaze_vector_to_angles(vectors: np.ndarray) -> np.ndarray:
    """Convert normalized 3D gaze vectors to [pitch, yaw] radians."""
    values = _validated_vectors(vectors, "gaze")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("gaze vector contains a zero norm")
    values = values / norms
    pitch = np.arcsin(np.clip(-values[:, 1], -1.0, 1.0))
    yaw = np.arctan2(-values[:, 0], -values[:, 2])
    return np.column_stack((pitch, yaw)).astype(np.float32)


def pose_rotation_vector_to_angles(vectors: np.ndarray) -> np.ndarray:
    """Convert Rodrigues head rotations to normalized [pitch, yaw] radians."""
    values = _validated_vectors(vectors, "pose")
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec(values).as_matrix()
    forward = rotation[:, :, 2]
    pitch = np.arcsin(np.clip(forward[:, 1], -1.0, 1.0))
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    return np.column_stack((pitch, yaw)).astype(np.float32)


def load_evaluation_data(
    root: Path,
    subjects: list[str] | tuple[str, ...],
    *,
    expected_samples_per_subject: int,
) -> tuple[EvaluationData, dict[str, Any]]:
    """Load and audit the balanced official evaluation subset."""
    dataset_root = root.resolve()
    normalized_root = dataset_root / "Data" / "Normalized"
    sample_root = dataset_root / "Evaluation Subset" / "sample list for eye image"
    if not normalized_root.is_dir() or not sample_root.is_dir():
        raise FileNotFoundError("MPIIGaze normalized data or sample list is missing")

    ordered_subjects = tuple(subjects)
    all_images: list[np.ndarray] = []
    all_poses: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_subject_ids: list[np.ndarray] = []
    all_days: list[str] = []
    all_sides: list[str] = []
    all_references: list[SampleRef] = []
    audit_subjects: dict[str, Any] = {}
    source_files: set[Path] = set()

    for subject_id, subject in enumerate(ordered_subjects):
        sample_path = sample_root / f"{subject}.txt"
        references = parse_sample_list(sample_path, subject)
        if len(references) != expected_samples_per_subject:
            raise ValueError(
                f"{subject} has {len(references)} official samples; "
                f"expected {expected_samples_per_subject}"
            )
        side_counts = {
            "left": sum(reference.side == "left" for reference in references),
            "right": sum(reference.side == "right" for reference in references),
        }
        if side_counts != {"left": 1500, "right": 1500}:
            raise ValueError(
                f"{subject} has unexpected eye-side balance: {side_counts}"
            )

        subject_images: list[np.ndarray] = []
        subject_poses: list[np.ndarray] = []
        subject_targets: list[np.ndarray] = []
        day_cache: dict[str, dict[str, Any]] = {}
        day_counts: dict[str, int] = {}

        for reference in references:
            if reference.day not in day_cache:
                mat_path = normalized_root / subject / f"{reference.day}.mat"
                day_cache[reference.day] = _load_normalized_day(mat_path)
                source_files.add(mat_path)
            day = day_cache[reference.day]
            index = day["filename_to_index"].get(reference.filename)
            if index is None:
                raise ValueError(
                    f"sample {reference.identity} is absent from normalized MAT data"
                )
            side_data = day[reference.side]
            image = np.asarray(side_data["image"][index], dtype=np.uint8)
            if image.shape != (36, 60):
                raise ValueError(
                    f"sample {reference.identity} has image shape {image.shape}"
                )
            pose = pose_rotation_vector_to_angles(side_data["pose"][index : index + 1])[
                0
            ]
            target = gaze_vector_to_angles(side_data["gaze"][index : index + 1])[0]
            if reference.side == "right":
                image = np.ascontiguousarray(image[:, ::-1])
                pose = pose.copy()
                target = target.copy()
                pose[1] *= -1.0
                target[1] *= -1.0
            subject_images.append(image[None, :, :])
            subject_poses.append(pose)
            subject_targets.append(target)
            all_days.append(reference.day)
            all_sides.append(reference.side)
            day_counts[reference.day] = day_counts.get(reference.day, 0) + 1

        all_images.append(np.stack(subject_images))
        all_poses.append(np.stack(subject_poses))
        all_targets.append(np.stack(subject_targets))
        all_subject_ids.append(
            np.full(expected_samples_per_subject, subject_id, dtype=np.int16)
        )
        all_references.extend(references)
        source_files.add(sample_path)
        audit_subjects[subject] = {
            "samples": len(references),
            "unique_sample_identities": len(
                {reference.identity for reference in references}
            ),
            "duplicate_reference_rows": len(references)
            - len({reference.identity for reference in references}),
            "left_samples": side_counts["left"],
            "right_samples": side_counts["right"],
            "days": len(day_counts),
            "minimum_day_samples": min(day_counts.values()),
            "maximum_day_samples": max(day_counts.values()),
        }

    identities = [reference.identity for reference in all_references]
    expected_total = expected_samples_per_subject * len(ordered_subjects)
    if len(identities) != expected_total:
        raise ValueError("evaluation subset row count is not exact")

    data = EvaluationData(
        images=np.concatenate(all_images, axis=0),
        poses=np.concatenate(all_poses, axis=0).astype(np.float32, copy=False),
        targets=np.concatenate(all_targets, axis=0).astype(np.float32, copy=False),
        subject_ids=np.concatenate(all_subject_ids),
        days=np.asarray(all_days),
        sides=np.asarray(all_sides),
        references=tuple(all_references),
        subjects=ordered_subjects,
    )
    _validate_loaded_arrays(data, expected_total)
    split_schedule = build_nested_subject_splits(ordered_subjects)
    audit = {
        "status": "passed",
        "subject_count": len(ordered_subjects),
        "samples_per_subject": expected_samples_per_subject,
        "total_samples": expected_total,
        "unique_sample_identities": len(set(identities)),
        "duplicate_reference_rows": expected_total - len(set(identities)),
        "image_shape": list(data.images.shape[1:]),
        "subjects": audit_subjects,
        "split_overlap_count": _split_overlap_count(split_schedule),
        "source_file_count": len(source_files),
        "source_sha256": _combined_file_sha256(source_files, dataset_root),
    }
    return data, audit


def fit_pose_standardization(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit pose standardization using outer-training rows only."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("pose matrix must be finite with shape (n, 2)")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize_pose(
    poses: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply already-fitted outer-training pose statistics."""
    result = (np.asarray(poses, dtype=np.float32) - mean) / scale
    if not np.isfinite(result).all():
        raise ValueError("standardized poses contain non-finite values")
    return result.astype(np.float32, copy=False)


def permute_targets_within_subjects(
    targets: np.ndarray, subject_ids: np.ndarray, *, seed: int
) -> np.ndarray:
    """Break input-target correspondence while preserving each subject distribution."""
    values = np.asarray(targets, dtype=np.float32)
    groups = np.asarray(subject_ids).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) != len(groups):
        raise ValueError("sentinel target inputs are invalid")
    result = values.copy()
    rng = np.random.default_rng(seed)
    for subject_id in sorted(set(int(value) for value in groups)):
        indices = np.flatnonzero(groups == subject_id)
        if len(indices) < 2:
            raise ValueError("each sentinel subject must contain at least two rows")
        result[indices] = values[rng.permutation(indices)]
    return result


def _load_normalized_day(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    from scipy.io import loadmat

    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    root = payload.get("data")
    filenames = np.atleast_1d(payload.get("filenames"))
    if root is None or filenames.size == 0:
        raise ValueError(f"invalid normalized MAT structure: {path}")
    filename_values = tuple(str(value) for value in filenames.tolist())
    if len(set(filename_values)) != len(filename_values):
        raise ValueError(f"duplicate filenames in normalized MAT: {path}")

    result: dict[str, Any] = {
        "filename_to_index": {
            filename: index for index, filename in enumerate(filename_values)
        }
    }
    for side in ("left", "right"):
        value = getattr(root, side, None)
        if value is None:
            raise ValueError(f"normalized MAT lacks {side} data: {path}")
        image = np.asarray(getattr(value, "image", None))
        gaze = np.asarray(getattr(value, "gaze", None), dtype=np.float64)
        pose = np.asarray(getattr(value, "pose", None), dtype=np.float64)
        if len(filename_values) == 1:
            image = image.reshape(1, 36, 60)
            gaze = gaze.reshape(1, 3)
            pose = pose.reshape(1, 3)
        if image.shape != (len(filename_values), 36, 60):
            raise ValueError(f"invalid {side} image shape in {path}: {image.shape}")
        if gaze.shape != (len(filename_values), 3):
            raise ValueError(f"invalid {side} gaze shape in {path}: {gaze.shape}")
        if pose.shape != (len(filename_values), 3):
            raise ValueError(f"invalid {side} pose shape in {path}: {pose.shape}")
        if not np.isfinite(gaze).all() or not np.isfinite(pose).all():
            raise ValueError(f"non-finite labels in {path}")
        result[side] = {"image": image, "gaze": gaze, "pose": pose}
    return result


def _validated_vectors(vectors: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} vectors must have shape (n, 3)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} vectors contain non-finite values")
    return values


def _validate_loaded_arrays(data: EvaluationData, expected_total: int) -> None:
    if data.images.shape != (expected_total, 1, 36, 60):
        raise ValueError(f"unexpected image matrix shape: {data.images.shape}")
    if data.poses.shape != (expected_total, 2):
        raise ValueError(f"unexpected pose matrix shape: {data.poses.shape}")
    if data.targets.shape != (expected_total, 2):
        raise ValueError(f"unexpected target matrix shape: {data.targets.shape}")
    if not np.isfinite(data.poses).all() or not np.isfinite(data.targets).all():
        raise ValueError("loaded evaluation arrays contain non-finite values")
    if data.images.dtype != np.uint8:
        raise ValueError("loaded evaluation images must remain uint8")
    if np.max(np.abs(data.targets[:, 0])) > math.pi / 2 + 1e-6:
        raise ValueError("gaze pitch is outside its physical angle range")
    if np.max(np.abs(data.targets[:, 1])) > math.pi + 1e-6:
        raise ValueError("gaze yaw is outside its physical angle range")


def _split_overlap_count(splits: list[SubjectSplit]) -> int:
    overlap = 0
    for split in splits:
        train = set(split.train_subjects)
        validation = {split.validation_subject}
        test = {split.test_subject}
        overlap += len(train & validation)
        overlap += len(train & test)
        overlap += len(validation & test)
    return overlap


def _combined_file_sha256(paths: set[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
