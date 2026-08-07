"""Server-side quality checks for participant calibration sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.gaze_core.motion_robustness import audit_payload, load_motion_samples
from core.gaze_core.sample_store import read_session_metadata, safe_session_dir


def audit_participant_calibration(
    root: Path,
    gaze_session_id: str,
    *,
    expected_study_session_id: str,
) -> dict[str, Any]:
    session_dir = safe_session_dir(root, gaze_session_id, require_exists=True)
    metadata = read_session_metadata(root, gaze_session_id)
    if metadata.get("study_session_id") != expected_study_session_id:
        raise ValueError("calibration dataset does not belong to the study session")
    manifest = session_dir / "manifest.jsonl"
    if not manifest.exists():
        raise ValueError("calibration manifest is missing")

    samples, diagnostics = load_motion_samples(
        session_dir.parent,
        session_ids=(gaze_session_id,),
    )
    motion_audit = audit_payload(samples, diagnostics)
    sample_count = diagnostics.lines_seen - diagnostics.malformed_lines
    unique_points = len({sample.point_index for sample in samples})
    face_detected_count = sum(sample.head_pose_usable for sample in samples)
    collection_protocols = sorted(
        {
            sample.collection_protocol
            for sample in samples
            if sample.collection_protocol is not None
        }
    )
    reasons = []
    if sample_count < 65:
        reasons.append("fewer_than_65_calibration_samples")
    if unique_points < 13:
        reasons.append("fewer_than_13_unique_targets")
    if "motion-diverse-v1" not in collection_protocols:
        reasons.append("motion_diverse_protocol_missing")
    if motion_audit["status"] != "ready":
        reasons.append("motion_coverage_gate_failed")
    if diagnostics.malformed_lines:
        reasons.append("malformed_manifest_rows")

    return {
        "passed": not reasons,
        "gaze_session_id": gaze_session_id,
        "sample_count": sample_count,
        "face_detected_count": face_detected_count,
        "face_detected_fraction": round(
            face_detected_count / sample_count if sample_count else 0.0,
            4,
        ),
        "unique_target_count": unique_points,
        "collection_protocols": collection_protocols,
        "source_sha256": diagnostics.source_sha256,
        "motion_audit_status": motion_audit["status"],
        "motion_audit_issues": motion_audit["issues"],
        "reasons": reasons,
    }
