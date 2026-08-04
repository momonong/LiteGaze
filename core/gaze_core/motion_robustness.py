"""Dependency-free audits for motion-diverse gaze calibration datasets.

The production gaze model is intentionally not imported here.  This module
inspects only JSON/JSONL metadata, so data coverage and validation leakage can
be checked without opening participant images, using a camera, or consuming a
GPU.  It also provides deterministic grouped folds for later model comparison.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, pi, sqrt
from pathlib import Path
from typing import Any


MOTION_METADATA_FIELDS = (
    "camera_id",
    "capture_burst_id",
    "collect_mode",
    "collection_protocol",
    "device_class",
    "distance_condition",
    "lighting_condition",
    "motion_block_id",
    "posture_condition",
)

CAPTURE_METADATA_TEXT_FIELDS = MOTION_METADATA_FIELDS
CAPTURE_METADATA_NUMBER_FIELDS = (
    "camera_frame_rate",
    "camera_height",
    "camera_width",
)


@dataclass(frozen=True)
class MotionSample:
    session_id: str
    participant_id: str
    point_index: int
    repeat_index: int
    phase: str
    head_pitch_deg: float | None
    head_yaw_deg: float | None
    face_scale: float | None
    camera_id: str | None
    capture_burst_id: str | None
    device_class: str | None
    distance_condition: str | None
    lighting_condition: str | None
    motion_block_id: str | None
    posture_condition: str | None
    collect_mode: str | None
    collection_protocol: str | None

    @property
    def head_pose_usable(self) -> bool:
        return self.head_pitch_deg is not None and self.head_yaw_deg is not None


@dataclass(frozen=True)
class LoadDiagnostics:
    sessions_seen: int
    manifests_seen: int
    lines_seen: int
    malformed_lines: int
    source_sha256: str


@dataclass(frozen=True)
class CoverageRequirements:
    min_usable_samples: int = 50
    min_motion_blocks: int = 5
    min_replicated_targets: int = 9
    min_head_yaw_span_deg: float = 20.0
    min_lateral_pose_separation_deg: float = 15.0
    min_distance_scale_change_ratio: float = 0.05
    required_distance_conditions: tuple[str, ...] = ("nominal", "near", "far")
    required_posture_conditions: tuple[str, ...] = ("neutral", "left", "right")
    required_metadata_fields: tuple[str, ...] = MOTION_METADATA_FIELDS

    def __post_init__(self) -> None:
        if self.min_usable_samples <= 0:
            raise ValueError("min_usable_samples must be positive")
        if self.min_motion_blocks <= 1:
            raise ValueError("min_motion_blocks must be greater than one")
        if self.min_replicated_targets <= 0:
            raise ValueError("min_replicated_targets must be positive")
        if (
            not isfinite(self.min_head_yaw_span_deg)
            or self.min_head_yaw_span_deg <= 0
        ):
            raise ValueError("min_head_yaw_span_deg must be finite and positive")
        if (
            not isfinite(self.min_lateral_pose_separation_deg)
            or self.min_lateral_pose_separation_deg <= 0
        ):
            raise ValueError(
                "min_lateral_pose_separation_deg must be finite and positive"
            )
        if not 0 < self.min_distance_scale_change_ratio < 1:
            raise ValueError("min_distance_scale_change_ratio must be between 0 and 1")
        invalid = set(self.required_metadata_fields) - set(MOTION_METADATA_FIELDS)
        if invalid:
            raise ValueError(f"unknown required metadata fields: {sorted(invalid)}")
        if not self.required_distance_conditions:
            raise ValueError("required_distance_conditions must not be empty")
        if not self.required_posture_conditions:
            raise ValueError("required_posture_conditions must not be empty")


@dataclass(frozen=True)
class CoverageIssue:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class ValidationFold:
    group_name: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def capture_metadata(record: Mapping[str, Any]) -> dict[str, str | float]:
    """Return a bounded allow-list of capture metadata safe to persist."""

    metadata: dict[str, str | float] = {}
    for field in CAPTURE_METADATA_TEXT_FIELDS:
        value = _optional_text(record.get(field))
        if value is not None:
            metadata[field] = value[:128]
    for field in CAPTURE_METADATA_NUMBER_FIELDS:
        value = _finite_float(record.get(field))
        if value is not None and value >= 0:
            metadata[field] = value
    return metadata


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _motion_sample(
    record: Mapping[str, Any],
    *,
    session_id: str,
    participant_id: str,
) -> MotionSample:
    pitch_deg = None
    yaw_deg = None
    pose = record.get("head_pose_pitch_yaw")
    if isinstance(pose, Sequence) and not isinstance(pose, (str, bytes)):
        if len(pose) >= 2:
            pitch = _finite_float(pose[0])
            yaw = _finite_float(pose[1])
            if pitch is not None and yaw is not None:
                pitch_deg = pitch * 180.0 / pi
                yaw_deg = yaw * 180.0 / pi

    face_scale = None
    face_bbox = record.get("face_bbox")
    if isinstance(face_bbox, Mapping):
        width = _finite_float(face_bbox.get("w_norm"))
        height = _finite_float(face_bbox.get("h_norm"))
        if width is not None and height is not None and width >= 0 and height >= 0:
            face_scale = sqrt(width * height)

    def optional_int(key: str) -> int:
        try:
            return int(record.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    return MotionSample(
        session_id=session_id,
        participant_id=participant_id,
        point_index=optional_int("point_index"),
        repeat_index=optional_int("repeat_index"),
        phase=_optional_text(record.get("phase")) or "unknown",
        head_pitch_deg=pitch_deg,
        head_yaw_deg=yaw_deg,
        face_scale=face_scale,
        camera_id=_optional_text(record.get("camera_id")),
        capture_burst_id=_optional_text(record.get("capture_burst_id")),
        device_class=_optional_text(record.get("device_class")),
        distance_condition=_optional_text(record.get("distance_condition")),
        lighting_condition=_optional_text(record.get("lighting_condition")),
        motion_block_id=_optional_text(record.get("motion_block_id")),
        posture_condition=_optional_text(record.get("posture_condition")),
        collect_mode=_optional_text(record.get("collect_mode")),
        collection_protocol=_optional_text(record.get("collection_protocol")),
    )


def load_motion_samples(
    sessions_dir: str | Path,
    *,
    session_ids: Iterable[str] | None = None,
) -> tuple[tuple[MotionSample, ...], LoadDiagnostics]:
    """Read all session manifests without opening any image paths."""

    root = Path(sessions_dir)
    selected_sessions = set(session_ids) if session_ids is not None else None
    samples: list[MotionSample] = []
    sessions_seen = 0
    manifests_seen = 0
    lines_seen = 0
    malformed_lines = 0
    digest = hashlib.sha256()

    if root.exists():
        for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if selected_sessions is not None and session_dir.name not in selected_sessions:
                continue
            sessions_seen += 1
            manifest_path = session_dir / "manifest.jsonl"
            if not manifest_path.is_file():
                continue
            manifests_seen += 1
            participant_id = "unknown"
            session_meta_path = session_dir / "session.json"
            if session_meta_path.is_file():
                try:
                    raw_session_meta = session_meta_path.read_bytes()
                    digest.update(
                        session_meta_path.relative_to(root).as_posix().encode("utf-8")
                    )
                    digest.update(b"\0")
                    digest.update(raw_session_meta)
                    digest.update(b"\0")
                    session_meta = json.loads(raw_session_meta.decode("utf-8"))
                    participant_id = (
                        _optional_text(session_meta.get("participant_id")) or "unknown"
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    participant_id = "unknown"

            relative_name = manifest_path.relative_to(root).as_posix()
            raw_manifest = manifest_path.read_bytes()
            digest.update(relative_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(raw_manifest)
            digest.update(b"\0")

            # Invalid UTF-8 is treated like any other malformed manifest row
            # instead of aborting the complete audit.
            for raw_line in raw_manifest.decode("utf-8", errors="replace").splitlines():
                if not raw_line.strip():
                    continue
                lines_seen += 1
                try:
                    record = json.loads(raw_line)
                except (json.JSONDecodeError, TypeError):
                    malformed_lines += 1
                    continue
                if not isinstance(record, Mapping):
                    malformed_lines += 1
                    continue
                samples.append(
                    _motion_sample(
                        record,
                        session_id=session_dir.name,
                        participant_id=participant_id,
                    )
                )

    diagnostics = LoadDiagnostics(
        sessions_seen=sessions_seen,
        manifests_seen=manifests_seen,
        lines_seen=lines_seen,
        malformed_lines=malformed_lines,
        source_sha256=digest.hexdigest(),
    )
    return tuple(samples), diagnostics


def _numeric_summary(values: Iterable[float | None]) -> dict[str, float] | None:
    ordered = sorted(value for value in values if value is not None and isfinite(value))
    if not ordered:
        return None

    def percentile(fraction: float) -> float:
        index = int((len(ordered) - 1) * fraction)
        return round(ordered[index], 3)

    return {
        "min": round(ordered[0], 3),
        "p05": percentile(0.05),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": round(ordered[-1], 3),
    }


def summarize_motion_coverage(
    samples: Sequence[MotionSample],
    diagnostics: LoadDiagnostics,
) -> dict[str, Any]:
    """Return aggregate-only coverage statistics with no participant labels."""

    metadata_coverage = {
        field: sum(getattr(sample, field) is not None for sample in samples)
        for field in MOTION_METADATA_FIELDS
    }
    target_conditions: dict[tuple[str, int], set[str]] = defaultdict(set)
    for sample in samples:
        condition = sample.motion_block_id or sample.posture_condition
        if condition:
            target_conditions[(sample.session_id, sample.point_index)].add(condition)

    replicated_targets = sum(
        len(conditions) >= 2 for conditions in target_conditions.values()
    )
    motion_blocks = sorted(
        {sample.motion_block_id for sample in samples if sample.motion_block_id}
    )
    participant_labels = {sample.participant_id for sample in samples}

    return {
        "schema_version": 1,
        "sessions_seen": diagnostics.sessions_seen,
        "manifests_seen": diagnostics.manifests_seen,
        "participant_label_count": len(participant_labels),
        "manifest_lines": diagnostics.lines_seen,
        "samples": len(samples),
        "head_pose_usable": sum(sample.head_pose_usable for sample in samples),
        "malformed_lines": diagnostics.malformed_lines,
        "source_sha256": diagnostics.source_sha256,
        "repeat_distribution": dict(
            sorted(Counter(sample.repeat_index for sample in samples).items())
        ),
        "phase_distribution": dict(
            sorted(Counter(sample.phase for sample in samples).items())
        ),
        "head_pitch_deg": _numeric_summary(
            sample.head_pitch_deg for sample in samples
        ),
        "head_yaw_deg": _numeric_summary(sample.head_yaw_deg for sample in samples),
        "face_scale": _numeric_summary(sample.face_scale for sample in samples),
        "metadata_coverage": metadata_coverage,
        "distance_condition_distribution": dict(
            sorted(
                Counter(
                    sample.distance_condition
                    for sample in samples
                    if sample.distance_condition
                ).items()
            )
        ),
        "posture_condition_distribution": dict(
            sorted(
                Counter(
                    sample.posture_condition
                    for sample in samples
                    if sample.posture_condition
                ).items()
            )
        ),
        "collection_protocol_distribution": dict(
            sorted(
                Counter(
                    sample.collection_protocol
                    for sample in samples
                    if sample.collection_protocol
                ).items()
            )
        ),
        "head_yaw_by_posture": {
            condition: _numeric_summary(
                sample.head_yaw_deg
                for sample in samples
                if sample.posture_condition == condition
            )
            for condition in ("neutral", "left", "right")
        },
        "face_scale_by_distance": {
            condition: _numeric_summary(
                sample.face_scale
                for sample in samples
                if sample.distance_condition == condition
            )
            for condition in ("nominal", "near", "far")
        },
        "unique_motion_blocks": len(motion_blocks),
        "replicated_session_targets": replicated_targets,
    }


def audit_motion_coverage(
    summary: Mapping[str, Any],
    requirements: CoverageRequirements | None = None,
) -> tuple[CoverageIssue, ...]:
    """Evaluate only predeclared coverage gates; never infer model accuracy."""

    requirements = requirements or CoverageRequirements()
    issues: list[CoverageIssue] = []
    usable = int(summary.get("head_pose_usable", 0) or 0)
    if usable < requirements.min_usable_samples:
        issues.append(
            CoverageIssue(
                code="INSUFFICIENT_USABLE_SAMPLES",
                severity="error",
                message=(
                    f"Only {usable} samples contain usable head pose; "
                    f"at least {requirements.min_usable_samples} are required."
                ),
            )
        )

    motion_blocks = int(summary.get("unique_motion_blocks", 0) or 0)
    if motion_blocks < requirements.min_motion_blocks:
        issues.append(
            CoverageIssue(
                code="INSUFFICIENT_MOTION_BLOCKS",
                severity="error",
                message=(
                    f"Only {motion_blocks} labeled motion blocks are present; "
                    f"at least {requirements.min_motion_blocks} are required."
                ),
            )
        )

    replicated_targets = int(summary.get("replicated_session_targets", 0) or 0)
    if replicated_targets < requirements.min_replicated_targets:
        issues.append(
            CoverageIssue(
                code="NO_CROSS_CONDITION_TARGET_REPLICATION",
                severity="error",
                message=(
                    f"Only {replicated_targets} session-target pairs repeat across "
                    "motion conditions, so gaze target and posture cannot be "
                    "identified separately."
                ),
            )
        )

    yaw = summary.get("head_yaw_deg")
    if isinstance(yaw, Mapping):
        yaw_min = _finite_float(yaw.get("min"))
        yaw_max = _finite_float(yaw.get("max"))
        if yaw_min is not None and yaw_max is not None:
            yaw_span = yaw_max - yaw_min
            if yaw_span < requirements.min_head_yaw_span_deg:
                issues.append(
                    CoverageIssue(
                        code="NARROW_HEAD_YAW_COVERAGE",
                        severity="error",
                        message=(
                            f"Observed head-yaw span is {yaw_span:.1f} degrees; "
                            f"the frozen minimum is "
                            f"{requirements.min_head_yaw_span_deg:.1f}."
                        ),
                    )
                )

    metadata_coverage = summary.get("metadata_coverage")
    if isinstance(metadata_coverage, Mapping):
        total_samples = int(summary.get("samples", 0) or 0)
        for field in requirements.required_metadata_fields:
            populated = int(metadata_coverage.get(field, 0) or 0)
            if populated < total_samples:
                issues.append(
                    CoverageIssue(
                        code=f"INCOMPLETE_{field.upper()}",
                        severity="error",
                        message=(
                            f"{field} is populated for {populated}/{total_samples} "
                            "samples."
                        ),
                    )
                )

    condition_gates = (
        (
            "distance_condition_distribution",
            requirements.required_distance_conditions,
            "MISSING_DISTANCE_CONDITIONS",
        ),
        (
            "posture_condition_distribution",
            requirements.required_posture_conditions,
            "MISSING_POSTURE_CONDITIONS",
        ),
    )
    for summary_key, required_conditions, issue_code in condition_gates:
        distribution = summary.get(summary_key)
        observed = set(distribution) if isinstance(distribution, Mapping) else set()
        missing = sorted(set(required_conditions) - observed)
        if missing:
            issues.append(
                CoverageIssue(
                    code=issue_code,
                    severity="error",
                    message=f"Required conditions are missing: {', '.join(missing)}.",
                )
            )

    yaw_by_posture = summary.get("head_yaw_by_posture")
    if isinstance(yaw_by_posture, Mapping):
        left = yaw_by_posture.get("left")
        right = yaw_by_posture.get("right")
        left_median = _finite_float(left.get("p50")) if isinstance(left, Mapping) else None
        right_median = (
            _finite_float(right.get("p50")) if isinstance(right, Mapping) else None
        )
        if left_median is None or right_median is None:
            issues.append(
                CoverageIssue(
                    code="UNVERIFIED_LATERAL_POSE_SEPARATION",
                    severity="error",
                    message="Left/right posture rows lack usable head-yaw measurements.",
                )
            )
        else:
            separation = abs(right_median - left_median)
            if separation < requirements.min_lateral_pose_separation_deg:
                issues.append(
                    CoverageIssue(
                        code="INSUFFICIENT_LATERAL_POSE_SEPARATION",
                        severity="error",
                        message=(
                            f"Left/right median head yaw differs by only "
                            f"{separation:.1f} degrees."
                        ),
                    )
                )

    scale_by_distance = summary.get("face_scale_by_distance")
    if isinstance(scale_by_distance, Mapping):
        condition_medians: dict[str, float] = {}
        for condition in ("nominal", "near", "far"):
            condition_summary = scale_by_distance.get(condition)
            if isinstance(condition_summary, Mapping):
                median = _finite_float(condition_summary.get("p50"))
                if median is not None:
                    condition_medians[condition] = median
        if len(condition_medians) < 3 or condition_medians.get("nominal", 0) <= 0:
            issues.append(
                CoverageIssue(
                    code="UNVERIFIED_DISTANCE_SEPARATION",
                    severity="error",
                    message=(
                        "Nominal/near/far rows lack usable normalized face-scale "
                        "measurements."
                    ),
                )
            )
        else:
            nominal = condition_medians["nominal"]
            minimum_change = requirements.min_distance_scale_change_ratio
            if not (
                condition_medians["near"] >= nominal * (1.0 + minimum_change)
                and condition_medians["far"] <= nominal * (1.0 - minimum_change)
            ):
                issues.append(
                    CoverageIssue(
                        code="INSUFFICIENT_DISTANCE_SEPARATION",
                        severity="error",
                        message=(
                            "Near/far face scale does not differ from nominal by "
                            f"at least {minimum_change:.0%} in the expected direction."
                        ),
                    )
                )

    if int(summary.get("malformed_lines", 0) or 0) > 0:
        issues.append(
            CoverageIssue(
                code="MALFORMED_MANIFEST_LINES",
                severity="error",
                message="One or more manifest rows could not be parsed.",
            )
        )
    return tuple(issues)


def grouped_validation_folds(
    samples: Sequence[MotionSample],
    *,
    group_field: str,
) -> tuple[ValidationFold, ...]:
    """Create deterministic leave-one-group-out folds with zero group leakage."""

    allowed_fields = {
        "capture_burst_id",
        "motion_block_id",
        "participant_id",
        "session_id",
    }
    if group_field not in allowed_fields:
        raise ValueError(f"group_field must be one of {sorted(allowed_fields)}")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        value = _optional_text(getattr(sample, group_field))
        if value is None:
            raise ValueError(f"sample {index} is missing {group_field}")
        groups[value].append(index)
    if len(groups) < 2:
        raise ValueError(f"at least two {group_field} groups are required")

    all_indices = set(range(len(samples)))
    folds: list[ValidationFold] = []
    for group_name in sorted(groups):
        validation = tuple(groups[group_name])
        train = tuple(sorted(all_indices - set(validation)))
        if not train or not validation:
            raise ValueError("each fold must contain train and validation samples")
        folds.append(
            ValidationFold(
                group_name=group_name,
                train_indices=train,
                validation_indices=validation,
            )
        )
    return tuple(folds)


def audit_payload(
    samples: Sequence[MotionSample],
    diagnostics: LoadDiagnostics,
    requirements: CoverageRequirements | None = None,
) -> dict[str, Any]:
    """Build one stable, JSON-ready audit result."""

    requirements = requirements or CoverageRequirements()
    summary = summarize_motion_coverage(samples, diagnostics)
    issues = audit_motion_coverage(summary, requirements)
    return {
        "status": "ready" if not any(issue.severity == "error" for issue in issues) else "not_ready",
        "requirements": asdict(requirements),
        "summary": summary,
        "issues": [asdict(issue) for issue in issues],
    }
