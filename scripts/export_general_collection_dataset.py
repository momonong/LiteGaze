"""Export completed rehearsal sessions into a versioned private analysis bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gaze_core.capture_contract import (
    compare_capture_contracts,
    normalize_capture_contract,
)
from core.participant_study.general_collection import (
    canonical_sha256,
    classify_gaze_quality,
    evaluate_validation_target_independence,
    load_general_bank,
    load_general_protocol,
    summarize_validation_samples,
    validate_general_design,
)
from core.participant_study.protocol import load_protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _canonical_equal(left: object, right: object) -> bool:
    try:
        return canonical_sha256(left) == canonical_sha256(right)
    except (TypeError, ValueError):
        return False


def _metric_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(_metric_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _normalized_viewport(payload: object) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        raise ValueError("assessment viewport is unavailable")
    normalized: dict[str, int] = {}
    for field in ("width_px", "height_px"):
        value = payload.get(field)
        if isinstance(value, bool):
            raise ValueError("assessment viewport dimension is invalid")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("assessment viewport dimension is invalid") from exc
        if not number.is_integer() or not 1 <= number <= 16384:
            raise ValueError("assessment viewport dimension is invalid")
        normalized[field] = int(number)
    return normalized


def _add_reason(audit: dict[str, Any], reason: str) -> None:
    if reason not in audit["reasons"]:
        audit["reasons"].append(reason)


def _audit_session_gaze(
    session_path: Path,
    session: Mapping[str, object],
    general_protocol: Mapping[str, object],
    bank: Mapping[str, object],
) -> dict[str, Any]:
    """Reconstruct every gaze provenance decision before export.

    This intentionally treats absent legacy fields as gaze-unavailable while
    leaving behavioral records exportable.
    """

    collection = dict(session.get("general_collection") or {})
    quality_root = dict(session.get("quality") or {})
    calibration = dict(quality_root.get("calibration") or {})
    final_quality = dict(quality_root.get("general_collection") or {})
    linked_data = dict(session.get("linked_data") or {})
    assignment = dict(session.get("collection_assignment") or {})
    session_id = str(session.get("study_session_id") or session_path.parent.name)
    audit: dict[str, Any] = {
        "study_session_id": session_id,
        "participant_id": str(session.get("participant_id") or ""),
        "pair_id": str(assignment.get("pair_id") or ""),
        "visit_index": assignment.get("visit_index"),
        "contract_id": None,
        "contract_version": None,
        "contract_sha256": None,
        "contract_snapshot_valid": False,
        "assessment_viewport": None,
        "validation_integrity_status": "unavailable",
        "validation_payload_sha256": {"start": None, "end": None},
        "capture_contract_eligible": False,
        "target_independence_eligible": False,
        "gaze_integrity_eligible": False,
        "geometry_contract_eligible": False,
        "recomputed_quality": {},
        "source_telemetry_sample_count": 0,
        "source_validation_sample_count": 0,
        "pair_gaze_comparison_status": "not_evaluated",
        "pair_gaze_comparable": False,
        "reasons": [],
    }

    if session.get("state") != "completed":
        _add_reason(audit, "session_not_completed")

    measurement_contract: dict[str, Any] | None = None
    snapshot = collection.get("gaze_measurement_contract")
    if not isinstance(snapshot, Mapping):
        _add_reason(audit, "measurement_contract_snapshot_unavailable")
    else:
        raw_contract = snapshot.get("contract")
        if not isinstance(raw_contract, Mapping):
            _add_reason(audit, "measurement_contract_snapshot_incomplete")
        else:
            measurement_contract = dict(raw_contract)
            try:
                calculated_sha256 = canonical_sha256(measurement_contract)
            except (TypeError, ValueError):
                calculated_sha256 = None
                _add_reason(audit, "measurement_contract_not_canonical")
            audit["contract_id"] = str(snapshot.get("contract_id") or "") or None
            audit["contract_version"] = (
                str(snapshot.get("contract_version") or "") or None
            )
            audit["contract_sha256"] = str(snapshot.get("sha256") or "") or None
            identity_matches = (
                audit["contract_id"] == measurement_contract.get("contract_id")
                and audit["contract_version"]
                == measurement_contract.get("contract_version")
            )
            hash_matches = (
                calculated_sha256 is not None
                and audit["contract_sha256"] == calculated_sha256
            )
            if not identity_matches:
                _add_reason(audit, "measurement_contract_identity_mismatch")
            if not hash_matches:
                _add_reason(audit, "measurement_contract_hash_mismatch")
            design_valid = False
            if calculated_sha256 is not None:
                try:
                    measurement_design = validate_general_design(
                        general_protocol,
                        bank,
                        measurement_contract,
                    )
                except (TypeError, ValueError):
                    _add_reason(audit, "measurement_contract_design_invalid")
                else:
                    design_valid = (
                        measurement_design.get("gaze_measurement_contract_sha256")
                        == calculated_sha256
                    )
                    if not design_valid:
                        _add_reason(audit, "measurement_contract_design_hash_mismatch")
            audit["contract_snapshot_valid"] = (
                identity_matches and hash_matches and design_valid
            )

    assessment_viewport: dict[str, int] | None = None
    try:
        assessment_viewport = _normalized_viewport(
            collection.get("assessment_viewport")
        )
    except ValueError:
        _add_reason(audit, "assessment_viewport_unavailable_or_invalid")
    else:
        audit["assessment_viewport"] = assessment_viewport
        device = dict(
            dict(quality_root.get("general_system_check") or {}).get("device") or {}
        )
        try:
            system_check_viewport = _normalized_viewport(
                {
                    "width_px": device.get("viewport_width"),
                    "height_px": device.get("viewport_height"),
                }
            )
        except ValueError:
            _add_reason(audit, "system_check_viewport_unavailable_or_invalid")
        else:
            if system_check_viewport != assessment_viewport:
                _add_reason(audit, "assessment_viewport_system_check_mismatch")

    phase_integrity: dict[str, bool] = {"start": False, "end": False}
    recomputed_validations: dict[str, dict[str, Any]] = {}
    capture_checks: list[bool] = []
    independence_checks: list[bool] = []
    validations = collection.get("validations")
    validation_map = dict(validations) if isinstance(validations, Mapping) else {}
    for phase in ("start", "end"):
        phase_ok = True
        summary_raw = validation_map.get(phase)
        if not isinstance(summary_raw, Mapping):
            _add_reason(audit, f"{phase}_validation_unavailable")
            continue
        summary = dict(summary_raw)
        raw_samples = summary.get("samples")
        if isinstance(raw_samples, list):
            audit["source_validation_sample_count"] += len(raw_samples)
        else:
            _add_reason(audit, f"{phase}_validation_samples_unavailable")
            phase_ok = False

        recomputed: dict[str, Any] | None = None
        if (
            measurement_contract is not None
            and assessment_viewport is not None
            and isinstance(raw_samples, list)
        ):
            try:
                recomputed = summarize_validation_samples(
                    raw_samples,
                    viewport_width_px=assessment_viewport["width_px"],
                    viewport_height_px=assessment_viewport["height_px"],
                    measurement_contract=measurement_contract,
                )
            except (TypeError, ValueError):
                _add_reason(audit, f"{phase}_validation_samples_invalid")
                phase_ok = False
        else:
            phase_ok = False

        if recomputed is not None:
            recomputed_validations[phase] = recomputed
            if any(
                not _canonical_equal(summary.get(key), value)
                for key, value in recomputed.items()
            ):
                _add_reason(audit, f"{phase}_validation_summary_mismatch")
                phase_ok = False
            expected_samples_sha256 = canonical_sha256(recomputed["samples"])
            if summary.get("samples_sha256") != expected_samples_sha256:
                _add_reason(audit, f"{phase}_validation_samples_hash_mismatch")
                phase_ok = False

        expected_provenance = {
            "contract_id": audit["contract_id"],
            "contract_version": audit["contract_version"],
            "sha256": audit["contract_sha256"],
        }
        if not audit["contract_snapshot_valid"] or not _canonical_equal(
            summary.get("gaze_measurement_contract"), expected_provenance
        ):
            _add_reason(audit, f"{phase}_measurement_contract_binding_mismatch")
            phase_ok = False
        if summary.get("gaze_measurement_contract_sha256") != audit["contract_sha256"]:
            _add_reason(audit, f"{phase}_measurement_contract_hash_mismatch")
            phase_ok = False
        if assessment_viewport is None or not _canonical_equal(
            summary.get("assessment_viewport"), assessment_viewport
        ):
            _add_reason(audit, f"{phase}_assessment_viewport_mismatch")
            phase_ok = False

        normalized_capture: dict[str, Any] | None = None
        try:
            normalized_capture = normalize_capture_contract(
                summary.get("capture_contract")  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            _add_reason(audit, f"{phase}_capture_contract_unavailable_or_invalid")
            phase_ok = False

        payload_sha256 = summary.get("validation_payload_sha256")
        audit["validation_payload_sha256"][phase] = (
            str(payload_sha256) if payload_sha256 else None
        )
        if (
            normalized_capture is not None
            and assessment_viewport is not None
            and audit["contract_sha256"]
            and recomputed is not None
        ):
            expected_payload_sha256 = canonical_sha256(
                {
                    "samples": recomputed["samples"],
                    "capture_contract": normalized_capture,
                    "gaze_measurement_contract_sha256": audit["contract_sha256"],
                    "assessment_viewport": assessment_viewport,
                }
            )
            if payload_sha256 != expected_payload_sha256:
                _add_reason(audit, f"{phase}_validation_payload_hash_mismatch")
                phase_ok = False
        else:
            phase_ok = False

        capture_compatible = False
        if normalized_capture is not None:
            try:
                recomputed_capture_check = compare_capture_contracts(
                    calibration.get("capture_contract"),
                    normalized_capture,
                )
            except (TypeError, ValueError):
                recomputed_capture_check = None
            if recomputed_capture_check is None:
                _add_reason(audit, f"{phase}_capture_contract_check_invalid")
                phase_ok = False
            else:
                capture_compatible = (
                    recomputed_capture_check.get("compatible") is True
                    and recomputed_capture_check.get("status") == "compatible"
                )
                if not _canonical_equal(
                    summary.get("capture_contract_check"),
                    recomputed_capture_check,
                ):
                    _add_reason(audit, f"{phase}_capture_contract_check_mismatch")
                    phase_ok = False
                if not capture_compatible:
                    _add_reason(audit, f"{phase}_capture_contract_ineligible")
                    phase_ok = False
        capture_checks.append(capture_compatible)

        target_independent = False
        if recomputed is not None and measurement_contract is not None:
            try:
                recomputed_independence = evaluate_validation_target_independence(
                    recomputed,
                    calibration.get("fit_target_contract"),
                    measurement_contract=measurement_contract,
                )
            except (TypeError, ValueError):
                recomputed_independence = None
            if recomputed_independence is None:
                _add_reason(audit, f"{phase}_target_independence_check_invalid")
                phase_ok = False
            else:
                target_independent = (
                    recomputed_independence.get("status") == "passed"
                    and recomputed_independence.get("independent") is True
                )
                if not _canonical_equal(
                    summary.get("target_independence_check"),
                    recomputed_independence,
                ):
                    _add_reason(audit, f"{phase}_target_independence_check_mismatch")
                    phase_ok = False
                if not target_independent:
                    _add_reason(audit, f"{phase}_target_independence_ineligible")
                    phase_ok = False
        independence_checks.append(target_independent)
        phase_integrity[phase] = phase_ok

    audit["capture_contract_eligible"] = (
        len(capture_checks) == 2 and all(capture_checks)
    )
    audit["target_independence_eligible"] = (
        len(independence_checks) == 2 and all(independence_checks)
    )
    audit["validation_integrity_status"] = (
        "passed"
        if audit["contract_snapshot_valid"]
        and assessment_viewport is not None
        and all(phase_integrity.values())
        else "failed"
    )

    gaze_integrity = collection.get("gaze_integrity")
    if isinstance(gaze_integrity, Mapping):
        integrity_reasons = list(gaze_integrity.get("reasons") or [])
        audit["gaze_integrity_eligible"] = (
            gaze_integrity.get("eligible") is True and not integrity_reasons
        )
        if not audit["gaze_integrity_eligible"]:
            _add_reason(audit, "gaze_integrity_ineligible")
    else:
        _add_reason(audit, "gaze_integrity_unavailable")

    required_flags = {
        "capture_contract_eligible": audit["capture_contract_eligible"],
        "target_independence_eligible": audit["target_independence_eligible"],
        "gaze_integrity_eligible": audit["gaze_integrity_eligible"],
    }
    for field, recomputed_value in required_flags.items():
        if final_quality.get(field) is not recomputed_value:
            _add_reason(audit, f"stored_{field}_mismatch")
    audit["geometry_contract_eligible"] = all(required_flags.values())
    if final_quality.get("geometry_contract_eligible") is not audit[
        "geometry_contract_eligible"
    ]:
        _add_reason(audit, "stored_geometry_contract_eligible_mismatch")
    if not audit["geometry_contract_eligible"]:
        _add_reason(audit, "geometry_contract_ineligible")

    telemetry_root = session_path.parent / "collection" / "telemetry"
    telemetry_batch_count = 0
    telemetry_attempt_count = 0
    telemetry_success_count = 0
    successful_poses: list[list[float]] = []
    successful_face_scales: list[float] = []
    for batch_path in sorted(telemetry_root.glob("*/*.json")):
        try:
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _add_reason(audit, "telemetry_batch_unreadable")
            continue
        samples = batch.get("samples")
        if not isinstance(samples, list):
            _add_reason(audit, "telemetry_samples_unavailable")
            continue
        telemetry_batch_count += 1
        telemetry_attempt_count += len(samples)
        audit["source_telemetry_sample_count"] += len(samples)
        forbidden = {"image", "image_data", "frame", "video", "audio"}
        if forbidden & set(batch) or any(
            isinstance(sample, Mapping) and forbidden & set(sample)
            for sample in samples
        ):
            _add_reason(audit, "telemetry_raw_media_prohibited")
        try:
            telemetry_viewport = _normalized_viewport(batch.get("viewport"))
        except ValueError:
            _add_reason(audit, "telemetry_viewport_unavailable_or_invalid")
        else:
            if assessment_viewport is None or telemetry_viewport != assessment_viewport:
                _add_reason(audit, "telemetry_viewport_mismatch")
        if str(batch.get("study_session_id") or "") != session_id:
            _add_reason(audit, "telemetry_session_binding_mismatch")
        if batch.get("visit_index") != assignment.get("visit_index"):
            _add_reason(audit, "telemetry_visit_binding_mismatch")
        if batch.get("capture_session_id") != linked_data.get("gaze_session_id"):
            _add_reason(audit, "telemetry_capture_session_binding_mismatch")
        normalized_payload = {
            "batch_id": batch.get("batch_id"),
            "passage_id": batch.get("passage_id"),
            "viewport": batch.get("viewport"),
            "samples": samples,
        }
        try:
            expected_telemetry_sha256 = canonical_sha256(normalized_payload)
        except (TypeError, ValueError):
            expected_telemetry_sha256 = None
        if (
            expected_telemetry_sha256 is None
            or batch.get("payload_sha256") != expected_telemetry_sha256
        ):
            _add_reason(audit, "telemetry_payload_hash_mismatch")

        for sample in samples:
            if not isinstance(sample, Mapping):
                _add_reason(audit, "telemetry_sample_invalid")
                continue
            if sample.get("prediction_success") is not True:
                continue
            telemetry_success_count += 1
            pose = sample.get("head_pose_pitch_yaw")
            bbox = sample.get("normalized_face_bbox")
            try:
                if not isinstance(pose, list) or len(pose) != 2:
                    raise ValueError
                normalized_pose = [float(value) for value in pose]
                if not all(math.isfinite(value) for value in normalized_pose):
                    raise ValueError
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError
                normalized_bbox = [float(value) for value in bbox]
                if not all(math.isfinite(value) for value in normalized_bbox):
                    raise ValueError
            except (TypeError, ValueError):
                _add_reason(audit, "telemetry_success_geometry_invalid")
                continue
            successful_poses.append(normalized_pose)
            successful_face_scales.append(
                max(0.0, normalized_bbox[2] - normalized_bbox[0])
                * max(0.0, normalized_bbox[3] - normalized_bbox[1])
            )

    if session.get("state") == "completed" and final_quality and (
        telemetry_batch_count == 0 or telemetry_attempt_count == 0
    ):
        _add_reason(audit, "completed_gaze_session_has_no_telemetry")

    head_pose_min = [
        min((pose[index] for pose in successful_poses), default=None)
        for index in range(2)
    ]
    head_pose_max = [
        max((pose[index] for pose in successful_poses), default=None)
        for index in range(2)
    ]
    face_scale_min = min(successful_face_scales, default=None)
    face_scale_max = max(successful_face_scales, default=None)
    recomputed_telemetry_stats = {
        "batch_count": telemetry_batch_count,
        "attempt_count": telemetry_attempt_count,
        "successful_count": telemetry_success_count,
        "head_pose_min": head_pose_min,
        "head_pose_max": head_pose_max,
        "face_scale_min": face_scale_min,
        "face_scale_max": face_scale_max,
    }
    stored_telemetry_stats = collection.get("telemetry_stats")
    if not isinstance(stored_telemetry_stats, Mapping):
        _add_reason(audit, "stored_telemetry_stats_unavailable")
    else:
        for field, expected in recomputed_telemetry_stats.items():
            if not _metric_equal(stored_telemetry_stats.get(field), expected):
                _add_reason(audit, f"stored_telemetry_{field}_mismatch")

    reading_elapsed_ms = 0.0
    round_file_count = 0
    for round_path in sorted(
        (session_path.parent / "collection" / "rounds").glob("R*.json")
    ):
        try:
            observation = json.loads(round_path.read_text(encoding="utf-8"))
            elapsed_ms = float(observation.get("reading_elapsed_ms"))
            if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
                raise ValueError
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            _add_reason(audit, "round_timing_unavailable_or_invalid")
            continue
        reading_elapsed_ms += elapsed_ms
        round_file_count += 1
    stored_rounds = collection.get("rounds")
    if isinstance(stored_rounds, list):
        try:
            stored_reading_elapsed_ms = sum(
                float(dict(item).get("reading_elapsed_ms", 0.0))
                for item in stored_rounds
            )
        except (TypeError, ValueError):
            stored_reading_elapsed_ms = None
        if (
            stored_reading_elapsed_ms is None
            or not math.isfinite(stored_reading_elapsed_ms)
            or not _metric_equal(stored_reading_elapsed_ms, reading_elapsed_ms)
            or len(stored_rounds) != round_file_count
        ):
            _add_reason(audit, "round_timing_binding_mismatch")
    else:
        _add_reason(audit, "stored_round_timing_unavailable")

    validation_summaries = [
        recomputed_validations[phase]
        for phase in ("start", "end")
        if phase in recomputed_validations
    ]
    medians = [
        float(summary["median_spatial_error_px"])
        for summary in validation_summaries
        if summary.get("median_spatial_error_px") is not None
    ]
    p90_values = [
        float(summary["p90_spatial_error_px"])
        for summary in validation_summaries
        if summary.get("p90_spatial_error_px") is not None
    ]
    precision_values = [
        float(summary["precision_rms_px"])
        for summary in validation_summaries
        if summary.get("precision_rms_px") is not None
    ]
    reading_seconds = reading_elapsed_ms / 1000.0
    raw_effective_sampling_hz = (
        telemetry_success_count / reading_seconds if reading_seconds > 0 else 0.0
    )
    telemetry_segments_contiguous = (
        "reading_active_reentered_without_segment_contract"
        not in set(integrity_reasons if isinstance(gaze_integrity, Mapping) else [])
    )
    recomputed_quality: dict[str, Any] = {
        "median_spatial_error_px": max(medians) if medians else None,
        "p90_spatial_error_px": max(p90_values) if p90_values else None,
        "precision_rms_px": max(precision_values) if precision_values else None,
        "prediction_success_fraction": (
            telemetry_success_count / telemetry_attempt_count
            if telemetry_attempt_count
            else 0.0
        ),
        "effective_sampling_hz": (
            raw_effective_sampling_hz if telemetry_segments_contiguous else None
        ),
        "head_pose_range": [
            (
                float(head_pose_max[index]) - float(head_pose_min[index])
                if head_pose_min[index] is not None
                and head_pose_max[index] is not None
                else None
            )
            for index in range(2)
        ],
        "face_scale_range": (
            float(face_scale_max) - float(face_scale_min)
            if face_scale_min is not None and face_scale_max is not None
            else None
        ),
        "drift_change_px": (
            float(recomputed_validations["end"]["median_spatial_error_px"])
            - float(recomputed_validations["start"]["median_spatial_error_px"])
            if len(recomputed_validations) == 2
            and recomputed_validations["end"].get("median_spatial_error_px")
            is not None
            and recomputed_validations["start"].get("median_spatial_error_px")
            is not None
            else None
        ),
        **required_flags,
        "telemetry_segments_contiguous": telemetry_segments_contiguous,
    }
    if not telemetry_segments_contiguous:
        recomputed_quality["raw_effective_sampling_hz"] = raw_effective_sampling_hz
    if (
        not audit["geometry_contract_eligible"]
        or recomputed_quality["median_spatial_error_px"] is None
        or recomputed_quality["p90_spatial_error_px"] is None
    ):
        recomputed_band = "behavioral_only"
    else:
        try:
            recomputed_band = classify_gaze_quality(recomputed_quality)
        except (TypeError, ValueError):
            recomputed_band = "behavioral_only"
            _add_reason(audit, "recomputed_gaze_quality_invalid")
    audit["recomputed_quality"] = {
        **recomputed_quality,
        "gaze_quality_band": recomputed_band,
    }

    stored_collection_metrics = collection.get("gaze_quality_metrics")
    if not isinstance(stored_collection_metrics, Mapping):
        _add_reason(audit, "stored_collection_gaze_quality_metrics_unavailable")
    else:
        for field, expected in recomputed_quality.items():
            if not _metric_equal(stored_collection_metrics.get(field), expected):
                _add_reason(audit, f"stored_collection_{field}_mismatch")
    if collection.get("gaze_quality_band") != recomputed_band:
        _add_reason(audit, "stored_collection_gaze_quality_band_mismatch")
    for field, expected in recomputed_quality.items():
        if not _metric_equal(final_quality.get(field), expected):
            _add_reason(audit, f"stored_final_{field}_mismatch")
    if final_quality.get("gaze_quality_band") != recomputed_band:
        _add_reason(audit, "stored_final_gaze_quality_band_mismatch")

    audit["reasons"].sort()
    audit["base_eligible"] = not audit["reasons"]
    audit["eligible"] = audit["base_eligible"]
    return audit


def _apply_pair_gaze_policy(audits: list[dict[str, Any]]) -> dict[str, str]:
    """Allow Visit 1/2 comparison only under one identical frozen contract."""

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in audits:
        if audit["pair_id"]:
            by_pair[audit["pair_id"]].append(audit)
        else:
            audit["pair_gaze_comparison_status"] = "pair_id_unavailable"

    pair_status: dict[str, str] = {}
    for pair_id, pair_audits in sorted(by_pair.items()):
        if len(pair_audits) == 1:
            status = "single_visit_only"
        else:
            try:
                visits = {int(audit["visit_index"]) for audit in pair_audits}
            except (TypeError, ValueError):
                visits = set()
            if len(pair_audits) != 2 or visits != {1, 2}:
                status = "pair_visit_set_invalid"
            elif len({audit["participant_id"] for audit in pair_audits}) != 1:
                status = "pair_participant_mismatch"
            elif not all(audit["base_eligible"] for audit in pair_audits):
                status = "paired_visit_gaze_ineligible"
            elif len({audit["contract_sha256"] for audit in pair_audits}) != 1:
                status = "pair_measurement_contract_mismatch"
            else:
                status = "comparable_same_measurement_contract"
                for audit in pair_audits:
                    audit["pair_gaze_comparable"] = True
        pair_status[pair_id] = status
        for audit in pair_audits:
            audit["pair_gaze_comparison_status"] = status

    for audit in audits:
        audit["reasons"].sort()
        audit["eligible"] = audit["base_eligible"]
    return pair_status


def export_bundle(
    root: Path,
    output: Path,
    *,
    include_incomplete: bool = False,
) -> dict[str, Any]:
    general_protocol = load_general_protocol()
    bank = load_general_bank()
    # Layer 1 preserves the existing whole-design health gate for the runtime
    # protocol and bank. Layer 2 below audits each session only against its own
    # frozen additive gaze contract; this current contract is never substituted
    # for a missing or older session snapshot.
    runtime_design = validate_general_design(general_protocol, bank)
    design = {
        "protocol_sha256": runtime_design["protocol_sha256"],
        "bank_sha256": runtime_design["bank_sha256"],
    }
    consent_protocol = load_protocol()
    rehearsal_root = (
        root
        / "data"
        / "participant_studies"
        / consent_protocol["protocol_id"]
        / "rehearsals"
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)

    participants: dict[str, dict[str, Any]] = {}
    session_rows: list[dict[str, Any]] = []
    passage_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    layout_rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    reading_video_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    gaze_excluded_rows: list[dict[str, Any]] = []
    storage_security_modes: set[str] = set()
    retention_policies: set[str] = set()

    session_records: list[
        tuple[Path, dict[str, Any], str, dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    seen_session_ids: set[str] = set()
    for session_path in sorted(rehearsal_root.glob("ST-*/session.json")):
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session_id = str(session.get("study_session_id") or session_path.parent.name)
        if session.get("state") == "withdrawn":
            excluded.append({"study_session_id": session_id, "reason": "withdrawn"})
            continue
        if not include_incomplete and session.get("state") != "completed":
            excluded.append({"study_session_id": session_id, "reason": "incomplete"})
            continue
        assignment = dict(session.get("collection_assignment") or {})
        collection = dict(session.get("general_collection") or {})
        if not assignment or not collection.get("assessment_id"):
            excluded.append({"study_session_id": session_id, "reason": "not_general_collection"})
            continue
        if assignment.get("protocol_sha256") != design["protocol_sha256"]:
            raise ValueError(f"session {session_id} has a different protocol digest")
        if assignment.get("bank_sha256") != design["bank_sha256"]:
            raise ValueError(f"session {session_id} has a different bank digest")
        if session_id in seen_session_ids:
            raise ValueError(f"duplicate study session ID: {session_id}")
        seen_session_ids.add(session_id)
        audit = _audit_session_gaze(
            session_path,
            session,
            general_protocol,
            bank,
        )
        session_records.append(
            (session_path, session, session_id, assignment, collection, audit)
        )

    pair_status = _apply_pair_gaze_policy(
        [record[-1] for record in session_records]
    )

    for session_path, session, session_id, assignment, collection, audit in session_records:
        participant_id = str(session["participant_id"])
        governance = dict(session.get("data_governance") or {})
        storage_security_modes.add(
            str(governance.get("storage_security") or "legacy_unspecified")
        )
        retention_policies.add(
            str(governance.get("retention_policy") or "legacy_unspecified")
        )
        profile = dict(collection.get("profile") or {})
        participant_row = {
            "participant_id": participant_id,
            "pair_id": assignment.get("pair_id"),
            "schedule_cell": assignment.get("schedule_cell"),
            "sequence": assignment.get("sequence"),
            **profile,
        }
        existing_participant = participants.get(participant_id)
        if existing_participant and existing_participant != participant_row:
            raise ValueError(f"participant {participant_id} changed frozen profile or assignment")
        participants[participant_id] = participant_row
        system_check = dict(
            dict(session.get("quality") or {}).get("general_system_check") or {}
        )
        device = dict(system_check.get("device", {}))
        gaze_eligible = bool(audit["eligible"])
        recomputed_quality = dict(audit.get("recomputed_quality") or {})
        assessment_viewport = dict(audit.get("assessment_viewport") or {})
        session_rows.append(
            {
                "participant_id": participant_id,
                "study_session_id": session_id,
                "visit_index": assignment.get("visit_index"),
                "capture_session_id": dict(session.get("linked_data") or {}).get("gaze_session_id"),
                "form_id": assignment.get("form_id"),
                "order_cell": assignment.get("order_cell"),
                "state": session.get("state"),
                "storage_security": governance.get("storage_security"),
                "retention_policy": governance.get("retention_policy"),
                "self_only": governance.get("self_only"),
                "formal_promotion_allowed": False,
                "created_at_utc": session.get("created_at_utc"),
                "completed_at_utc": next(
                    (
                        event.get("at_utc")
                        for event in reversed(session.get("events", []))
                        if event.get("event") == "general_collection_completed"
                    ),
                    None,
                ),
                "device_class": device.get("device_class"),
                "browser_family": device.get("browser_family"),
                "viewport_width": device.get("viewport_width"),
                "viewport_height": device.get("viewport_height"),
                "camera_width": device.get("camera_width"),
                "camera_height": device.get("camera_height"),
                "estimated_camera_fps_band": device.get("estimated_camera_fps_band"),
                "gaze_measurement_contract_id": audit["contract_id"],
                "gaze_measurement_contract_version": audit["contract_version"],
                "gaze_measurement_contract_sha256": audit["contract_sha256"],
                "gaze_contract_snapshot_valid": audit["contract_snapshot_valid"],
                "assessment_viewport_width": assessment_viewport.get("width_px"),
                "assessment_viewport_height": assessment_viewport.get("height_px"),
                "validation_integrity_status": audit["validation_integrity_status"],
                "capture_contract_eligible": audit["capture_contract_eligible"],
                "target_independence_eligible": audit[
                    "target_independence_eligible"
                ],
                "gaze_integrity_eligible": audit["gaze_integrity_eligible"],
                "geometry_contract_eligible": audit[
                    "geometry_contract_eligible"
                ],
                "gaze_export_status": "eligible" if gaze_eligible else "excluded",
                "gaze_export_eligible": gaze_eligible,
                "gaze_exclusion_reasons": _json_cell(audit["reasons"]),
                "pair_gaze_comparison_status": audit[
                    "pair_gaze_comparison_status"
                ],
                "pair_gaze_comparable": audit["pair_gaze_comparable"],
                "start_validation_payload_sha256": audit[
                    "validation_payload_sha256"
                ]["start"],
                "end_validation_payload_sha256": audit[
                    "validation_payload_sha256"
                ]["end"],
                "gaze_quality_band": (
                    recomputed_quality.get("gaze_quality_band")
                    if gaze_eligible
                    else "unavailable"
                ),
                "median_spatial_error_px": (
                    recomputed_quality.get("median_spatial_error_px")
                    if gaze_eligible
                    else None
                ),
                "p90_spatial_error_px": (
                    recomputed_quality.get("p90_spatial_error_px")
                    if gaze_eligible
                    else None
                ),
                "precision_rms_px": (
                    recomputed_quality.get("precision_rms_px")
                    if gaze_eligible
                    else None
                ),
                "prediction_success_fraction": (
                    recomputed_quality.get("prediction_success_fraction")
                    if gaze_eligible
                    else None
                ),
                "effective_sampling_hz": (
                    recomputed_quality.get("effective_sampling_hz")
                    if gaze_eligible
                    else None
                ),
                "head_pose_range": (
                    _json_cell(recomputed_quality.get("head_pose_range"))
                    if gaze_eligible
                    else None
                ),
                "face_scale_range": (
                    recomputed_quality.get("face_scale_range")
                    if gaze_eligible
                    else None
                ),
                "drift_change_px": (
                    recomputed_quality.get("drift_change_px")
                    if gaze_eligible
                    else None
                ),
            }
        )
        if not gaze_eligible:
            gaze_excluded_rows.append(
                {
                    "participant_id": participant_id,
                    "study_session_id": session_id,
                    "pair_id": assignment.get("pair_id"),
                    "visit_index": assignment.get("visit_index"),
                    "gaze_measurement_contract_id": audit["contract_id"],
                    "gaze_measurement_contract_version": audit["contract_version"],
                    "gaze_measurement_contract_sha256": audit["contract_sha256"],
                    "pair_gaze_comparison_status": audit[
                        "pair_gaze_comparison_status"
                    ],
                    "exclusion_reasons": _json_cell(audit["reasons"]),
                    "source_telemetry_sample_count": audit[
                        "source_telemetry_sample_count"
                    ],
                    "source_validation_sample_count": audit[
                        "source_validation_sample_count"
                    ],
                    "behavioral_fields_retained": True,
                }
            )

        for round_path in sorted((session_path.parent / "collection" / "rounds").glob("R*.json")):
            observation = json.loads(round_path.read_text(encoding="utf-8"))
            base = {
                "participant_id": participant_id,
                "study_session_id": session_id,
                "visit_index": assignment.get("visit_index"),
                "form_id": assignment.get("form_id"),
                "round_number": observation.get("round_number"),
                "passage_id": observation.get("passage_id"),
                "passage_family_id": observation.get("passage_family_id"),
            }
            report = dict(observation.get("passage_self_report") or {})
            passage_rows.append(
                {
                    **base,
                    "difficulty_band": observation.get("difficulty_band"),
                    "reading_elapsed_ms": observation.get("reading_elapsed_ms"),
                    "scroll_occurred": observation.get("scroll_occurred"),
                    "zoom_ratio": observation.get("zoom_ratio"),
                    "understanding": report.get("understanding"),
                    "mental_effort": report.get("mental_effort"),
                    "read_complete": report.get("read_complete"),
                    "interrupted": report.get("interrupted"),
                    "word_layout_sha256": observation.get("word_layout_sha256"),
                    "probe_order_sha256": observation.get("probe_order_sha256"),
                }
            )
            for order_index, review in enumerate(observation.get("word_reviews", [])):
                review_rows.append(
                    {
                        **base,
                        "probe_order_index": order_index,
                        "probe_id": review.get("probe_id"),
                        "surface": review.get("surface"),
                        "stratum": review.get("stratum"),
                        "label": review.get("label"),
                    }
                )
            for layout in observation.get("word_layout", []):
                layout_rows.append({**base, **layout})

        if gaze_eligible:
            telemetry_root = session_path.parent / "collection" / "telemetry"
            for batch_path in sorted(telemetry_root.glob("*/*.json")):
                batch = json.loads(batch_path.read_text(encoding="utf-8"))
                for sample_index, sample in enumerate(batch.get("samples", [])):
                    telemetry_rows.append(
                        {
                            "participant_id": participant_id,
                            "study_session_id": session_id,
                            "visit_index": assignment.get("visit_index"),
                            "capture_session_id": batch.get("capture_session_id"),
                            "gaze_measurement_contract_id": audit["contract_id"],
                            "gaze_measurement_contract_version": audit[
                                "contract_version"
                            ],
                            "gaze_measurement_contract_sha256": audit[
                                "contract_sha256"
                            ],
                            "pair_gaze_comparison_status": audit[
                                "pair_gaze_comparison_status"
                            ],
                            "pair_gaze_comparable": audit[
                                "pair_gaze_comparable"
                            ],
                            "assessment_viewport_width": assessment_viewport.get(
                                "width_px"
                            ),
                            "assessment_viewport_height": assessment_viewport.get(
                                "height_px"
                            ),
                            "passage_id": batch.get("passage_id"),
                            "batch_id": batch.get("batch_id"),
                            "sample_index": sample_index,
                            "monotonic_elapsed_ms": sample.get(
                                "monotonic_elapsed_ms"
                            ),
                            "prediction_success": sample.get("prediction_success"),
                            "coarse_failure_code": sample.get("coarse_failure_code"),
                            "screen_xy_norm": _json_cell(sample.get("screen_xy_norm")),
                            "screen_xy_px": _json_cell(sample.get("screen_xy_px")),
                            "gaze_pitch_yaw": _json_cell(sample.get("gaze_pitch_yaw")),
                            "head_pose_pitch_yaw": _json_cell(
                                sample.get("head_pose_pitch_yaw")
                            ),
                            "normalized_face_bbox": _json_cell(
                                sample.get("normalized_face_bbox")
                            ),
                            "nearest_word_index": sample.get("nearest_word_index"),
                            "viewport": _json_cell(batch.get("viewport")),
                        }
                    )
            validations_for_export = dict(collection.get("validations") or {})
            for phase in ("start", "end"):
                summary = dict(validations_for_export[phase])
                for sample_index, sample in enumerate(summary.get("samples", [])):
                    validation_rows.append(
                        {
                            "participant_id": participant_id,
                            "study_session_id": session_id,
                            "visit_index": assignment.get("visit_index"),
                            "capture_session_id": dict(
                                session.get("linked_data") or {}
                            ).get("gaze_session_id"),
                            "gaze_measurement_contract_id": audit["contract_id"],
                            "gaze_measurement_contract_version": audit[
                                "contract_version"
                            ],
                            "gaze_measurement_contract_sha256": audit[
                                "contract_sha256"
                            ],
                            "validation_payload_sha256": audit[
                                "validation_payload_sha256"
                            ][phase],
                            "pair_gaze_comparison_status": audit[
                                "pair_gaze_comparison_status"
                            ],
                            "pair_gaze_comparable": audit[
                                "pair_gaze_comparable"
                            ],
                            "assessment_viewport_width": assessment_viewport.get(
                                "width_px"
                            ),
                            "assessment_viewport_height": assessment_viewport.get(
                                "height_px"
                            ),
                            "phase": phase,
                            "sample_index": sample_index,
                            **sample,
                        }
                    )

        reading_video_root = session_path.parent / "collection" / "reading_video"
        for metadata_path in sorted(reading_video_root.glob("R*.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            media_paths = [
                candidate
                for candidate in metadata_path.parent.glob(f"{metadata_path.stem}.*")
                if candidate.suffix.lower() != ".json"
            ]
            if len(media_paths) != 1:
                raise ValueError(
                    f"session {session_id} reading video {metadata_path.stem} "
                    "does not have exactly one media file"
                )
            media_path = media_paths[0]
            if media_path.stat().st_size != int(metadata.get("bytes", -1)):
                raise ValueError(f"reading video size mismatch: {media_path}")
            if _sha256(media_path) != metadata.get("sha256"):
                raise ValueError(f"reading video digest mismatch: {media_path}")
            reading_video_rows.append(
                {
                    "participant_id": participant_id,
                    "study_session_id": session_id,
                    "visit_index": assignment.get("visit_index"),
                    "round_number": metadata.get("round_number"),
                    "passage_id": metadata.get("passage_id"),
                    "recording_id": metadata.get("recording_id"),
                    "duration_ms": metadata.get("duration_ms"),
                    "mime_type": metadata.get("mime_type"),
                    "bytes": metadata.get("bytes"),
                    "sha256": metadata.get("sha256"),
                    "video_track_count": metadata.get("video_track_count"),
                    "audio_track_count": metadata.get("audio_track_count"),
                    "storage_security": metadata.get("storage_security"),
                    "dataset_role": metadata.get("dataset_role"),
                    "source_relative_path": media_path.relative_to(root).as_posix(),
                }
            )

    tables: dict[str, tuple[list[str], list[dict[str, Any]]]] = {
        "participants.csv": (
            [
                "participant_id", "pair_id", "schedule_cell", "sequence", "english_l1",
                "english_age_of_acquisition_band", "weekly_english_reading_band",
                "vision_correction", "education_band",
            ],
            sorted(participants.values(), key=lambda row: str(row["participant_id"])),
        ),
        "sessions.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "capture_session_id",
                "form_id", "order_cell", "state", "storage_security",
                "retention_policy", "self_only", "formal_promotion_allowed",
                "created_at_utc", "completed_at_utc",
                "device_class", "browser_family", "viewport_width", "viewport_height",
                "camera_width", "camera_height", "estimated_camera_fps_band",
                "gaze_measurement_contract_id", "gaze_measurement_contract_version",
                "gaze_measurement_contract_sha256", "gaze_contract_snapshot_valid",
                "assessment_viewport_width", "assessment_viewport_height",
                "validation_integrity_status", "capture_contract_eligible",
                "target_independence_eligible", "gaze_integrity_eligible",
                "geometry_contract_eligible", "gaze_export_status",
                "gaze_export_eligible", "gaze_exclusion_reasons",
                "pair_gaze_comparison_status", "pair_gaze_comparable",
                "start_validation_payload_sha256", "end_validation_payload_sha256",
                "gaze_quality_band", "median_spatial_error_px", "p90_spatial_error_px",
                "precision_rms_px", "prediction_success_fraction", "effective_sampling_hz",
                "head_pose_range", "face_scale_range", "drift_change_px",
            ],
            session_rows,
        ),
        "passages.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "form_id", "round_number",
                "passage_id", "passage_family_id", "difficulty_band", "reading_elapsed_ms",
                "scroll_occurred", "zoom_ratio", "understanding", "mental_effort",
                "read_complete", "interrupted", "word_layout_sha256", "probe_order_sha256",
            ],
            passage_rows,
        ),
        "word_reviews.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "form_id", "round_number",
                "passage_id", "passage_family_id", "probe_order_index", "probe_id", "surface",
                "stratum", "label",
            ],
            review_rows,
        ),
        "word_layout.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "form_id", "round_number",
                "passage_id", "passage_family_id", "word_index", "left_px", "top_px",
                "right_px", "bottom_px",
            ],
            layout_rows,
        ),
        "gaze_telemetry.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "capture_session_id",
                "gaze_measurement_contract_id", "gaze_measurement_contract_version",
                "gaze_measurement_contract_sha256", "pair_gaze_comparison_status",
                "pair_gaze_comparable", "assessment_viewport_width",
                "assessment_viewport_height",
                "passage_id", "batch_id", "sample_index", "monotonic_elapsed_ms",
                "prediction_success", "coarse_failure_code", "screen_xy_norm", "screen_xy_px",
                "gaze_pitch_yaw", "head_pose_pitch_yaw", "normalized_face_bbox",
                "nearest_word_index", "viewport",
            ],
            telemetry_rows,
        ),
        "validation_samples.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "capture_session_id",
                "gaze_measurement_contract_id", "gaze_measurement_contract_version",
                "gaze_measurement_contract_sha256", "validation_payload_sha256",
                "pair_gaze_comparison_status", "pair_gaze_comparable",
                "assessment_viewport_width", "assessment_viewport_height", "phase",
                "sample_index", "target_id", "target_x_px", "target_y_px",
                "target_x_norm", "target_y_norm", "prediction_success",
                "predicted_x_px", "predicted_y_px", "spatial_error_px",
            ],
            validation_rows,
        ),
        "reading_video_index.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "round_number",
                "passage_id", "recording_id", "duration_ms", "mime_type", "bytes",
                "sha256", "video_track_count", "audio_track_count", "storage_security",
                "dataset_role", "source_relative_path",
            ],
            reading_video_rows,
        ),
        "excluded_sessions.csv": (
            ["study_session_id", "reason"],
            excluded,
        ),
        "gaze_excluded_sessions.csv": (
            [
                "participant_id", "study_session_id", "pair_id", "visit_index",
                "gaze_measurement_contract_id", "gaze_measurement_contract_version",
                "gaze_measurement_contract_sha256", "pair_gaze_comparison_status",
                "exclusion_reasons", "source_telemetry_sample_count",
                "source_validation_sample_count", "behavioral_fields_retained",
            ],
            gaze_excluded_rows,
        ),
    }
    for filename, (fields, rows) in tables.items():
        _write_csv(output / filename, fields, rows)
    files = {
        filename: {
            "sha256": _sha256(output / filename),
            "row_count": len(rows),
        }
        for filename, (_, rows) in tables.items()
    }
    all_audits = [record[-1] for record in session_records]
    eligible_audits = [audit for audit in all_audits if audit["eligible"]]
    gaze_exclusion_reason_counts = Counter(
        reason
        for audit in all_audits
        if not audit["eligible"]
        for reason in audit["reasons"]
    )
    eligible_contract_counts = Counter(
        (
            str(audit["contract_id"]),
            str(audit["contract_version"]),
            str(audit["contract_sha256"]),
        )
        for audit in eligible_audits
    )
    pair_status_counts = Counter(pair_status.values())
    manifest = {
        "schema_version": 2,
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": general_protocol["dataset_role"],
        "formal_promotion_allowed": False,
        "claim_boundary": general_protocol["claim_boundary"],
        "protocol_id": general_protocol["protocol_id"],
        "protocol_version": general_protocol["protocol_version"],
        "protocol_sha256": design["protocol_sha256"],
        "bank_id": bank["bank_id"],
        "bank_version": bank["bank_version"],
        "bank_sha256": design["bank_sha256"],
        "bank_status": bank["status"],
        "participant_count": len(participants),
        "session_count": len(session_rows),
        "files": files,
        "gaze_provenance": {
            "policy": "eligible_only_gaze_tables_behavioral_rows_retained",
            "session_gaze_eligible_count": len(eligible_audits),
            "session_gaze_excluded_count": len(all_audits) - len(eligible_audits),
            "gaze_tables_contain_only_eligible_sessions": True,
            "legacy_or_unavailable_gaze_mixed_with_eligible": False,
            "validation_payload_hash_required": True,
            "validation_payload_hash_scope": (
                "stored_payload_integrity_not_client_prediction_authenticity"
            ),
            "prediction_values_tamper_resistant": False,
            "server_issued_prediction_receipts_required_before_formal_promotion": True,
            "frozen_measurement_contract_snapshot_required": True,
            "assessment_viewport_binding_required": True,
            "capture_contract_eligibility_required": True,
            "target_independence_eligibility_required": True,
            "nonempty_telemetry_required_for_completed_gaze_session": True,
            "stored_telemetry_stats_must_match_raw_batches": True,
            "final_gaze_quality_metrics_recomputed_before_export": True,
            "pair_comparison_requires_same_contract_sha256": True,
            "eligible_measurement_contracts": [
                {
                    "contract_id": contract_id,
                    "contract_version": contract_version,
                    "sha256": contract_sha256,
                    "session_count": count,
                }
                for (
                    contract_id,
                    contract_version,
                    contract_sha256,
                ), count in sorted(eligible_contract_counts.items())
            ],
            "pair_comparison_status_counts": dict(sorted(pair_status_counts.items())),
            "pair_comparable_count": pair_status_counts.get(
                "comparable_same_measurement_contract",
                0,
            ),
            "pair_not_comparable_count": len(pair_status) - pair_status_counts.get(
                "comparable_same_measurement_contract",
                0,
            ),
            "exclusion_reason_counts": dict(
                sorted(gaze_exclusion_reason_counts.items())
            ),
        },
        "split_policy": {
            "same_participant_same_partition": True,
            "participant_holdout_required": True,
            "passage_family_holdout_required": True,
            "probe_holdout_required": True,
            "capture_session_and_device_holdout_required": True,
            "split_manifest_status": "not_created_by_exporter",
        },
        "privacy": {
            "direct_identifiers_exported": False,
            "raw_images_or_video_exported": False,
            "bundle_classification": "private_pseudonymous_research_data",
        },
        "source_reading_videos": {
            "present": bool(reading_video_rows),
            "count": len(reading_video_rows),
            "raw_media_files_exported": False,
            "index_exported": True,
            "dataset_role": "self_development_only_not_confirmation",
        },
        "storage_governance": {
            "security_modes": sorted(storage_security_modes),
            "retention_policies": sorted(retention_policies),
            "unencrypted_self_development_present": (
                "unencrypted_self_development" in storage_security_modes
            ),
            "formal_promotion_allowed": False,
        },
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = args.root.resolve()
    output = (
        args.output or (root / "artifacts" / f"general_collection_{timestamp}")
    ).resolve()
    manifest = export_bundle(
        root,
        output,
        include_incomplete=args.include_incomplete,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "participant_count": manifest["participant_count"],
                "session_count": manifest["session_count"],
                "formal_promotion_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
