"""Export completed rehearsal sessions into a versioned private analysis bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
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
from core.gaze_core.uncertainty_contract import (
    RUNTIME_OBSERVATION_SCHEMA_VERSION,
    canonical_json_bytes as uncertainty_canonical_json_bytes,
    normalize_uncertainty_observation,
    unavailable_uncertainty,
)
from core.participant_study.general_collection import (
    canonical_sha256,
    classify_gaze_quality,
    evaluate_validation_target_independence,
    load_general_bank,
    load_general_protocol,
    summarize_validation_samples,
    validation_target_definitions,
    validate_general_design,
)
from core.participant_study.protocol import load_protocol


PREDICTION_RECEIPT_SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
READING_TELEMETRY_EVIDENCE_STATUS = "client_roundtrip_unverified"
SUCCESS_CONTRADICTORY_UNCERTAINTY_STATUSES = frozenset(
    {
        "unavailable_capture_failure",
        "unavailable_prediction_failure",
        "unavailable_sensor_failure",
    }
)


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


def _add_reason(
    audit: dict[str, Any],
    reason: str,
    *,
    scope: str = "validation",
) -> None:
    if reason not in audit["reasons"]:
        audit["reasons"].append(reason)
    scoped_key = f"{scope}_reasons"
    if scoped_key in audit and reason not in audit[scoped_key]:
        audit[scoped_key].append(reason)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _finite_receipt_number(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("outcome_invalid")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome_invalid") from exc
    if not math.isfinite(number):
        raise ValueError("outcome_invalid")
    return number


def _normalized_receipt_uncertainty(
    prediction: Mapping[str, object],
    *,
    prediction_success: bool,
    viewport: Mapping[str, int],
) -> dict[str, Any]:
    """Revalidate nested receipt uncertainty without requiring it for geometry."""

    schema_version = prediction.get("uncertainty_schema_version")
    raw_observation = prediction.get("uncertainty")
    if schema_version is None and raw_observation is None:
        return unavailable_uncertainty(
            (
                "unavailable_receipt_missing"
                if prediction_success
                else "unavailable_sensor_failure"
            ),
            "legacy prediction receipt did not contain runtime uncertainty evidence",
        )
    if (
        type(schema_version) is not int
        or schema_version != RUNTIME_OBSERVATION_SCHEMA_VERSION
    ):
        raise ValueError("uncertainty_schema_mismatch")
    try:
        normalized = normalize_uncertainty_observation(
            raw_observation,  # type: ignore[arg-type]
            viewport=(viewport["width_px"], viewport["height_px"]),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("uncertainty_observation_invalid") from exc
    if (
        not prediction_success
        and normalized.get("status") != "unavailable_sensor_failure"
    ):
        raise ValueError("uncertainty_outcome_contradiction")
    if (
        prediction_success
        and normalized.get("status")
        in SUCCESS_CONTRADICTORY_UNCERTAINTY_STATUSES
    ):
        raise ValueError("uncertainty_outcome_contradiction")
    return normalized


def _receipt_uncertainty_observation(
    *,
    issued_record_sha256: str,
    phase: str,
    receipt_ordinal: int,
    target_id: str,
    target_repeat_index: int,
    prediction_success: bool,
    uncertainty: Mapping[str, object],
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_OBSERVATION_SCHEMA_VERSION,
        "receipt_record_sha256": issued_record_sha256,
        "phase": phase,
        "receipt_ordinal": receipt_ordinal,
        "target_id": target_id,
        "target_repeat_index": target_repeat_index,
        "prediction_success": prediction_success,
        "uncertainty": json.loads(
            uncertainty_canonical_json_bytes(uncertainty).decode("utf-8")
        ),
    }


def _receipt_uncertainty_summary(
    observations: Sequence[Mapping[str, object]],
    *,
    status: str,
) -> dict[str, Any]:
    normalized = [
        json.loads(
            uncertainty_canonical_json_bytes(observation).decode("utf-8")
        )
        for observation in observations
    ]
    scored_count = sum(
        dict(observation.get("uncertainty") or {}).get("status")
        == "scored_no_threshold"
        for observation in normalized
    )
    return {
        "schema_version": RUNTIME_OBSERVATION_SCHEMA_VERSION,
        "status": status,
        "count": len(normalized),
        "scored_count": scored_count,
        "unavailable_count": len(normalized) - scored_count,
        "observation_sha256s": [
            canonical_sha256(observation) for observation in normalized
        ],
        "observations_sha256": canonical_sha256(normalized),
    }


def _uncertainty_export_fields(
    observation: Mapping[str, object],
) -> dict[str, Any]:
    uncertainty = dict(observation.get("uncertainty") or {})
    status = str(uncertainty.get("status") or "")
    scored = status == "scored_no_threshold"
    sensor_failure = status == "unavailable_sensor_failure"
    components = dict(uncertainty.get("components") or {}) if scored else {}

    def component(name: str, field: str) -> object:
        return dict(components.get(name) or {}).get(field) if scored else None

    abstention = dict(uncertainty.get("abstention") or {}) if scored else {}
    return {
        "uncertainty_observation_sha256": canonical_sha256(observation),
        "uncertainty_observation_json": _json_cell(observation),
        "uncertainty_schema_version": uncertainty.get("schema_version"),
        "uncertainty_status": status,
        "uncertainty_evidence_status": (
            "verified_scored_no_threshold" if scored else "not_evaluable"
        ),
        "uncertainty_evidence_eligible": scored,
        "uncertainty_coverage_risk_status": (
            "conditional_input_eligible_no_threshold"
            if scored
            else (
                "excluded_sensor_failure_reported_in_capture_coverage"
                if sensor_failure
                else "not_evaluable"
            )
        ),
        "uncertainty_definition_sha256": uncertainty.get("definition_sha256"),
        "uncertainty_score": uncertainty.get("score"),
        "uncertainty_ood_value": component("ood", "value"),
        "uncertainty_ood_percentile": component("ood", "percentile"),
        "uncertainty_leverage_value": component("leverage", "value"),
        "uncertainty_leverage_percentile": component(
            "leverage", "percentile"
        ),
        "uncertainty_disagreement_value": component("disagreement", "value"),
        "uncertainty_disagreement_percentile": component(
            "disagreement", "percentile"
        ),
        "uncertainty_jackknife_disagreement_covariance_norm": (
            _json_cell(uncertainty.get("jackknife_disagreement_covariance_norm"))
            if scored
            else ""
        ),
        "uncertainty_jackknife_disagreement_covariance_px": (
            _json_cell(uncertainty.get("jackknife_disagreement_covariance_px"))
            if scored
            else ""
        ),
        "uncertainty_abstention_status": abstention.get("status"),
        "uncertainty_abstention_threshold_json": (
            _json_cell(abstention.get("threshold")) if scored else ""
        ),
        "uncertainty_reason": uncertainty.get("reason"),
    }


def _expected_receipt_target(
    target: Mapping[str, object],
    viewport: Mapping[str, int],
) -> dict[str, object]:
    return {
        "target_id": target["target_id"],
        "target_x_viewport_fraction": float(
            target["target_x_viewport_fraction"]
        ),
        "target_y_viewport_fraction": float(
            target["target_y_viewport_fraction"]
        ),
        "target_x_norm": float(target["target_x_norm"]),
        "target_y_norm": float(target["target_y_norm"]),
        "target_x_px": float(
            math.floor(
                float(target["target_x_viewport_fraction"])
                * viewport["width_px"]
                + 0.5
            )
        ),
        "target_y_px": float(
            math.floor(
                float(target["target_y_viewport_fraction"])
                * viewport["height_px"]
                + 0.5
            )
        ),
    }


def _receipt_outcome_sample(
    issued: Mapping[str, object],
    *,
    expected_target: Mapping[str, object],
    viewport: Mapping[str, int],
    issued_record_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    prediction = issued.get("prediction")
    if not isinstance(prediction, Mapping):
        raise ValueError("outcome_invalid")
    success = prediction.get("success")
    if not isinstance(success, bool):
        raise ValueError("outcome_invalid")
    sample: dict[str, Any] = {
        "target_id": expected_target["target_id"],
        "target_x_px": expected_target["target_x_px"],
        "target_y_px": expected_target["target_y_px"],
        "target_x_norm": expected_target["target_x_norm"],
        "target_y_norm": expected_target["target_y_norm"],
        "prediction_success": success,
    }
    if success:
        raw_px = prediction.get("screen_xy_px")
        raw_norm = prediction.get("screen_xy_norm")
        if (
            not isinstance(raw_px, Sequence)
            or isinstance(raw_px, (str, bytes))
            or len(raw_px) != 2
            or not isinstance(raw_norm, Sequence)
            or isinstance(raw_norm, (str, bytes))
            or len(raw_norm) != 2
        ):
            raise ValueError("outcome_invalid")
        predicted_x = _finite_receipt_number(raw_px[0])
        predicted_y = _finite_receipt_number(raw_px[1])
        normalized_x = _finite_receipt_number(raw_norm[0])
        normalized_y = _finite_receipt_number(raw_norm[1])
        if (
            not 0.0 <= predicted_x <= viewport["width_px"]
            or not 0.0 <= predicted_y <= viewport["height_px"]
            or not -1.0 <= normalized_x <= 1.0
            or not -1.0 <= normalized_y <= 1.0
        ):
            raise ValueError("outcome_viewport_mismatch")
        expected_x = ((normalized_x + 1.0) * 0.5) * viewport["width_px"]
        expected_y = ((normalized_y + 1.0) * 0.5) * viewport["height_px"]
        if not math.isclose(
            predicted_x, expected_x, rel_tol=0.0, abs_tol=1e-6
        ) or not math.isclose(
            predicted_y, expected_y, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("outcome_coordinate_transform_mismatch")
        http_status = prediction.get("http_status")
        if (
            isinstance(http_status, bool)
            or not isinstance(http_status, int)
            or not 200 <= http_status < 300
            or prediction.get("failure_stage") is not None
            or prediction.get("failure_code") is not None
            or prediction.get("error") is not None
        ):
            raise ValueError("outcome_invalid")
        sample["predicted_x_px"] = predicted_x
        sample["predicted_y_px"] = predicted_y
        sample["spatial_error_px"] = math.hypot(
            predicted_x - float(expected_target["target_x_px"]),
            predicted_y - float(expected_target["target_y_px"]),
        )
        return sample, None

    http_status = prediction.get("http_status")
    if (
        prediction.get("failure_stage") != "attributable_sensor_failure"
        or prediction.get("failure_code") != "no_face_detected"
        or isinstance(http_status, bool)
        or http_status != 400
        or prediction.get("screen_xy_px") is not None
        or prediction.get("screen_xy_norm") is not None
    ):
        raise ValueError("failure_outcome_invalid")
    sample["predicted_x_px"] = None
    sample["predicted_y_px"] = None
    sample["spatial_error_px"] = None
    return sample, {
        "receipt_record_sha256": issued_record_sha256,
        "failure_stage": "attributable_sensor_failure",
        "failure_code": "no_face_detected",
        "http_status": 400,
    }


def _audit_prediction_receipts(
    session: Mapping[str, object],
    collection: Mapping[str, object],
    *,
    measurement_contract: Mapping[str, object] | None,
    contract_sha256: str | None,
    assessment_viewport: Mapping[str, int] | None,
    calibration_capture_contract: Mapping[str, object] | None,
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Verify stored receipt evidence without accepting any client round trip."""

    result: dict[str, Any] = {
        "schema_version": None,
        "eligible": False,
        "status": "unavailable",
        "record_count": 0,
        "phases": {
            phase: {
                "eligible": False,
                "bundle": None,
                "bundle_sha256": None,
                "receipt_record_sha256s": [],
                "reconstructed_samples": [],
                "reconstructed_failures": [],
                "capture_contract": None,
                "capture_contract_warnings": [],
                "uncertainty_observations": [],
                "uncertainty_summary": None,
                "uncertainty_evidence_eligible": False,
                "uncertainty_evidence_status": "not_evaluable",
                "uncertainty_reasons": [],
                "uncertainty_successful_prediction_count": 0,
                "uncertainty_no_face_count": 0,
                "uncertainty_capture_coverage_fraction": 0.0,
                "uncertainty_conditional_scored_fraction": None,
                "uncertainty_coverage_risk_evaluable": False,
            }
            for phase in ("start", "end")
        },
    }
    if (
        measurement_contract is None
        or not contract_sha256
        or assessment_viewport is None
    ):
        _add_reason(audit, "prediction_receipt_prerequisites_unavailable")
        return result

    registry = collection.get("prediction_receipts")
    if not isinstance(registry, Mapping):
        _add_reason(audit, "prediction_receipt_registry_unavailable")
        return result
    schema_version = registry.get("schema_version")
    result["schema_version"] = schema_version
    if (
        isinstance(schema_version, bool)
        or schema_version != PREDICTION_RECEIPT_SCHEMA_VERSION
    ):
        _add_reason(audit, "prediction_receipt_registry_schema_mismatch")
        return result
    raw_records = registry.get("records")
    if not isinstance(raw_records, Mapping):
        _add_reason(audit, "prediction_receipt_registry_records_invalid")
        return result
    result["record_count"] = len(raw_records)

    registry_ok = True
    records: dict[str, Mapping[str, object]] = {}
    by_record_sha256: dict[str, list[tuple[str, Mapping[str, object]]]] = defaultdict(
        list
    )
    for raw_key, raw_record in raw_records.items():
        registry_key = str(raw_key)
        if not _valid_sha256(registry_key) or not isinstance(raw_record, Mapping):
            _add_reason(audit, "prediction_receipt_registry_record_invalid")
            registry_ok = False
            continue
        record = dict(raw_record)
        issued = record.get("issued")
        issued_record_sha256 = record.get("issued_record_sha256")
        if "token" in record or (
            isinstance(issued, Mapping) and "token" in issued
        ):
            _add_reason(audit, "prediction_receipt_raw_token_persisted")
            registry_ok = False
        if not isinstance(issued, Mapping) or not _valid_sha256(
            issued_record_sha256
        ):
            _add_reason(audit, "prediction_receipt_registry_record_invalid")
            registry_ok = False
            continue
        try:
            calculated_record_sha256 = canonical_sha256(issued)
        except (TypeError, ValueError):
            calculated_record_sha256 = None
        if calculated_record_sha256 != issued_record_sha256:
            _add_reason(audit, "prediction_receipt_registry_record_hash_mismatch")
            registry_ok = False
        if issued.get("receipt_id_sha256") != registry_key:
            _add_reason(audit, "prediction_receipt_registry_identity_mismatch")
            registry_ok = False
        records[registry_key] = record
        by_record_sha256[str(issued_record_sha256)].append((registry_key, record))

    linked_data = dict(session.get("linked_data") or {})
    frozen_artifact_sha256 = collection.get("model_artifact_sha256")
    expected_common = {
        "study_session_id": session.get("study_session_id"),
        "authorization_fingerprint_sha256": session.get("access_token_sha256"),
        "assessment_id": collection.get("assessment_id"),
        "model_name": linked_data.get("model_name"),
        "model_artifact_sha256": frozen_artifact_sha256,
        "capture_session_id": linked_data.get("gaze_session_id"),
        "viewport": dict(assessment_viewport),
        "measurement_contract_sha256": contract_sha256,
    }
    if not _valid_sha256(expected_common["authorization_fingerprint_sha256"]):
        _add_reason(audit, "prediction_receipt_authorization_binding_unavailable")
        registry_ok = False
    if not _valid_sha256(frozen_artifact_sha256):
        _add_reason(audit, "prediction_receipt_model_artifact_binding_unavailable")
        registry_ok = False
    if not expected_common["model_name"]:
        _add_reason(audit, "prediction_receipt_model_binding_unavailable")
        registry_ok = False
    if linked_data.get("assessment_id") != expected_common["assessment_id"]:
        _add_reason(audit, "prediction_receipt_assessment_linkage_mismatch")
        registry_ok = False
    for field in ("study_session_id", "assessment_id", "capture_session_id"):
        value = expected_common[field]
        if not isinstance(value, str) or not value.strip():
            _add_reason(audit, f"prediction_receipt_{field}_binding_unavailable")
            registry_ok = False

    targets = validation_target_definitions(measurement_contract)
    validation_map = dict(collection.get("validations") or {})
    used_registry_keys: set[str] = set()
    used_record_sha256s: set[str] = set()
    for phase in ("start", "end"):
        phase_result = result["phases"][phase]
        phase_ok = registry_ok
        summary = validation_map.get(phase)
        if not isinstance(summary, Mapping):
            _add_reason(audit, f"{phase}_prediction_receipt_summary_unavailable")
            continue
        if (
            summary.get("prediction_receipt_status") != "verified"
            or summary.get("prediction_receipts_verified") is not True
        ):
            _add_reason(audit, f"{phase}_prediction_receipt_status_unverified")
            phase_ok = False
        bundle = summary.get("prediction_receipt_bundle")
        if not isinstance(bundle, Mapping):
            _add_reason(audit, f"{phase}_prediction_receipt_bundle_unavailable")
            continue
        record_sha256s = bundle.get("receipt_record_sha256s")
        bundle_count = bundle.get("count")
        if (
            bundle.get("schema_version") != PREDICTION_RECEIPT_SCHEMA_VERSION
            or bundle.get("status") != "verified"
            or bundle.get("phase") != phase
            or isinstance(bundle_count, bool)
            or bundle_count != len(targets) * 3
            or not isinstance(record_sha256s, list)
            or len(record_sha256s) != len(targets) * 3
            or any(not _valid_sha256(value) for value in record_sha256s)
            or len(set(record_sha256s)) != len(record_sha256s)
        ):
            _add_reason(audit, f"{phase}_prediction_receipt_bundle_invalid")
            continue
        bundle_core = {
            "schema_version": PREDICTION_RECEIPT_SCHEMA_VERSION,
            "status": "verified",
            "phase": phase,
            "count": len(record_sha256s),
            "receipt_record_sha256s": record_sha256s,
        }
        expected_bundle = {
            **bundle_core,
            "bundle_sha256": canonical_sha256(bundle_core),
        }
        if not _canonical_equal(bundle, expected_bundle):
            _add_reason(audit, f"{phase}_prediction_receipt_bundle_hash_mismatch")
            phase_ok = False
        phase_result["bundle_sha256"] = expected_bundle["bundle_sha256"]
        phase_result["bundle"] = expected_bundle
        phase_result["receipt_record_sha256s"] = list(record_sha256s)

        reconstructed_samples: list[dict[str, Any]] = []
        reconstructed_failures: list[dict[str, Any]] = []
        uncertainty_observations: list[dict[str, Any]] = []
        uncertainty_reasons: list[str] = []
        phase_capture: dict[str, Any] | None = None
        phase_capture_warnings: list[str] = []
        for ordinal, issued_record_sha256 in enumerate(record_sha256s):
            if issued_record_sha256 in used_record_sha256s:
                _add_reason(audit, "prediction_receipt_record_reused_across_phases")
                phase_ok = False
            used_record_sha256s.add(issued_record_sha256)
            matches = by_record_sha256.get(issued_record_sha256, [])
            if len(matches) != 1:
                _add_reason(
                    audit,
                    f"{phase}_prediction_receipt_registry_record_missing_or_duplicated",
                )
                phase_ok = False
                continue
            registry_key, record = matches[0]
            used_registry_keys.add(registry_key)
            issued = record.get("issued")
            if not isinstance(issued, Mapping):
                _add_reason(audit, f"{phase}_prediction_receipt_record_invalid")
                phase_ok = False
                continue
            consumed_at = record.get("consumed_at_utc")
            if (
                not isinstance(consumed_at, str)
                or not consumed_at.strip()
                or record.get("consumed_validation_phase") != phase
            ):
                _add_reason(audit, f"{phase}_prediction_receipt_consumption_mismatch")
                phase_ok = False
            if issued.get("schema_version") != PREDICTION_RECEIPT_SCHEMA_VERSION:
                _add_reason(audit, f"{phase}_prediction_receipt_schema_mismatch")
                phase_ok = False
            issued_at = issued.get("issued_at_utc")
            if not isinstance(issued_at, str) or not issued_at.strip():
                _add_reason(audit, f"{phase}_prediction_receipt_issue_time_invalid")
                phase_ok = False
            for field, expected in expected_common.items():
                if not _canonical_equal(issued.get(field), expected):
                    _add_reason(
                        audit,
                        f"{phase}_prediction_receipt_{field}_binding_mismatch",
                    )
                    phase_ok = False
            if issued.get("phase") != phase:
                _add_reason(audit, f"{phase}_prediction_receipt_phase_binding_mismatch")
                phase_ok = False
            target = targets[ordinal // 3]
            expected_target = _expected_receipt_target(target, assessment_viewport)
            if (
                issued.get("receipt_ordinal") != ordinal
                or issued.get("target_repeat_index") != ordinal % 3
                or not _canonical_equal(issued.get("target"), expected_target)
            ):
                _add_reason(audit, f"{phase}_prediction_receipt_target_sequence_mismatch")
                phase_ok = False
            capture_check = issued.get("capture_contract_check")
            if not isinstance(capture_check, Mapping) or capture_check.get(
                "compatible"
            ) is not True:
                _add_reason(audit, f"{phase}_prediction_receipt_capture_check_failed")
                phase_ok = False
            try:
                normalized_capture = normalize_capture_contract(
                    issued.get("capture_contract")  # type: ignore[arg-type]
                )
            except (TypeError, ValueError):
                _add_reason(audit, f"{phase}_prediction_receipt_capture_invalid")
                phase_ok = False
                normalized_capture = None
            if normalized_capture is not None:
                if phase_capture is None:
                    phase_capture = normalized_capture
                else:
                    try:
                        within_phase_check = compare_capture_contracts(
                            phase_capture,
                            normalized_capture,
                        )
                    except (TypeError, ValueError):
                        within_phase_check = None
                    if (
                        within_phase_check is None
                        or within_phase_check.get("compatible") is not True
                    ):
                        _add_reason(
                            audit,
                            f"{phase}_prediction_receipt_capture_changed_within_phase",
                        )
                        phase_ok = False
                    else:
                        phase_capture_warnings.extend(
                            str(value)
                            for value in within_phase_check.get("warnings", [])
                        )
                try:
                    calibration_check = compare_capture_contracts(
                        calibration_capture_contract,
                        normalized_capture,
                    )
                except (TypeError, ValueError):
                    calibration_check = None
                if (
                    calibration_check is None
                    or calibration_check.get("compatible") is not True
                ):
                    _add_reason(
                        audit,
                        f"{phase}_prediction_receipt_calibration_capture_incompatible",
                    )
                    phase_ok = False
                else:
                    phase_capture_warnings.extend(
                        str(value)
                        for value in calibration_check.get("warnings", [])
                    )
            try:
                sample, failure = _receipt_outcome_sample(
                    issued,
                    expected_target=expected_target,
                    viewport=assessment_viewport,
                    issued_record_sha256=issued_record_sha256,
                )
            except ValueError as exc:
                _add_reason(audit, f"{phase}_prediction_receipt_{exc}")
                phase_ok = False
                continue
            reconstructed_samples.append(sample)
            if failure is not None:
                reconstructed_failures.append(failure)
            prediction = issued.get("prediction")
            try:
                uncertainty = _normalized_receipt_uncertainty(
                    prediction,  # type: ignore[arg-type]
                    prediction_success=sample["prediction_success"],
                    viewport=assessment_viewport,
                )
            except (TypeError, ValueError) as exc:
                reason = f"{phase}_prediction_receipt_{exc}"
                _add_reason(audit, reason)
                uncertainty_reasons.append(str(exc))
                phase_ok = False
            else:
                uncertainty_observations.append(
                    _receipt_uncertainty_observation(
                        issued_record_sha256=issued_record_sha256,
                        phase=phase,
                        receipt_ordinal=ordinal,
                        target_id=str(expected_target["target_id"]),
                        target_repeat_index=ordinal % 3,
                        prediction_success=sample["prediction_success"],
                        uncertainty=uncertainty,
                    )
                )
                uncertainty_status = str(uncertainty.get("status") or "")
                if uncertainty_status != "scored_no_threshold":
                    uncertainty_reasons.append(uncertainty_status or "unavailable")

        phase_result["capture_contract"] = phase_capture
        phase_result["capture_contract_warnings"] = sorted(
            set(phase_capture_warnings)
        )
        phase_result["reconstructed_samples"] = reconstructed_samples
        phase_result["reconstructed_failures"] = reconstructed_failures
        phase_result["uncertainty_observations"] = uncertainty_observations
        expected_uncertainty_summary = _receipt_uncertainty_summary(
            uncertainty_observations,
            status="verified",
        )
        phase_result["uncertainty_summary"] = expected_uncertainty_summary
        phase_result["uncertainty_reasons"] = sorted(set(uncertainty_reasons))
        successful_prediction_count = sum(
            sample.get("prediction_success") is True
            for sample in reconstructed_samples
        )
        no_face_count = len(reconstructed_failures)
        expected_observation_count = len(targets) * 3
        phase_result["uncertainty_successful_prediction_count"] = (
            successful_prediction_count
        )
        phase_result["uncertainty_no_face_count"] = no_face_count
        phase_result["uncertainty_capture_coverage_fraction"] = (
            expected_uncertainty_summary["scored_count"]
            / expected_observation_count
        )
        phase_result["uncertainty_conditional_scored_fraction"] = (
            expected_uncertainty_summary["scored_count"]
            / successful_prediction_count
            if successful_prediction_count > 0
            else None
        )
        phase_result["uncertainty_evidence_eligible"] = (
            len(uncertainty_observations) == expected_observation_count
            and len(reconstructed_samples) == expected_observation_count
            and expected_uncertainty_summary["scored_count"]
            == successful_prediction_count
            and expected_uncertainty_summary["unavailable_count"]
            == no_face_count
        )
        phase_result["uncertainty_coverage_risk_evaluable"] = bool(
            phase_result["uncertainty_evidence_eligible"]
            and successful_prediction_count > 0
        )
        phase_result["uncertainty_evidence_status"] = (
            "verified_scored_no_threshold"
            if phase_result["uncertainty_evidence_eligible"]
            else "not_evaluable"
        )
        if phase_capture is None or not _canonical_equal(
            summary.get("capture_contract"), phase_capture
        ):
            _add_reason(audit, f"{phase}_prediction_receipt_capture_binding_mismatch")
            phase_ok = False
        if not _canonical_equal(summary.get("samples"), reconstructed_samples):
            _add_reason(audit, f"{phase}_prediction_receipt_outcome_sample_mismatch")
            phase_ok = False
        if not _canonical_equal(
            summary.get("prediction_failures"), reconstructed_failures
        ):
            _add_reason(audit, f"{phase}_prediction_receipt_failure_summary_mismatch")
            phase_ok = False
        if summary.get("model_artifact_sha256") != frozen_artifact_sha256:
            _add_reason(audit, f"{phase}_prediction_receipt_model_artifact_mismatch")
            phase_ok = False
        if not _canonical_equal(
            summary.get("uncertainty_observations"), uncertainty_observations
        ):
            _add_reason(
                audit,
                f"{phase}_prediction_receipt_uncertainty_observations_mismatch",
            )
            phase_ok = False
        if not _canonical_equal(
            summary.get("uncertainty_summary"), expected_uncertainty_summary
        ):
            _add_reason(
                audit,
                f"{phase}_prediction_receipt_uncertainty_summary_mismatch",
            )
            phase_ok = False
        phase_result["eligible"] = phase_ok

    if set(records) != used_registry_keys or len(records) != len(targets) * 6:
        _add_reason(audit, "prediction_receipt_registry_contains_unbound_records")
        for phase in ("start", "end"):
            result["phases"][phase]["eligible"] = False
    result["eligible"] = all(
        result["phases"][phase]["eligible"] for phase in ("start", "end")
    )
    result["status"] = "verified" if result["eligible"] else "unavailable"
    return result


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
        "prediction_receipt_registry_schema_version": None,
        "prediction_receipt_status": "unavailable",
        "prediction_receipt_eligible": False,
        "prediction_receipt_record_count": 0,
        "prediction_receipt_bundle_sha256": {"start": None, "end": None},
        "prediction_receipt_record_sha256s": {"start": [], "end": []},
        "prediction_receipt_capture_warnings": {"start": [], "end": []},
        "prediction_receipt_uncertainty_observations": {"start": [], "end": []},
        "prediction_receipt_uncertainty_summary": {"start": None, "end": None},
        "uncertainty_evidence_eligible": False,
        "uncertainty_evidence_status": "not_evaluable",
        "uncertainty_reasons": [],
        "uncertainty_successful_prediction_count": {"start": 0, "end": 0},
        "uncertainty_no_face_count": {"start": 0, "end": 0},
        "uncertainty_capture_coverage_fraction": {"start": 0.0, "end": 0.0},
        "uncertainty_conditional_scored_fraction": {"start": None, "end": None},
        "uncertainty_coverage_risk_evaluable": False,
        "capture_contract_eligible": False,
        "target_independence_eligible": False,
        "gaze_integrity_eligible": False,
        "validation_geometry_contract_eligible": False,
        "geometry_contract_eligible": False,
        "validation_gaze_export_eligible": False,
        "reading_gaze_export_eligible": False,
        "reading_telemetry_evidence_status": READING_TELEMETRY_EVIDENCE_STATUS,
        "recomputed_validation_quality": {},
        "recomputed_quality": {},
        "source_telemetry_sample_count": 0,
        "source_validation_sample_count": 0,
        "pair_validation_gaze_comparison_status": "not_evaluated",
        "pair_validation_gaze_comparable": False,
        "pair_gaze_comparison_status": "reading_telemetry_unverified",
        "pair_gaze_comparable": False,
        "validation_reasons": [],
        "reading_reasons": [],
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
    receipt_audit = _audit_prediction_receipts(
        session,
        collection,
        measurement_contract=measurement_contract,
        contract_sha256=audit["contract_sha256"],
        assessment_viewport=assessment_viewport,
        calibration_capture_contract=calibration.get("capture_contract"),
        audit=audit,
    )
    audit["prediction_receipt_registry_schema_version"] = receipt_audit[
        "schema_version"
    ]
    audit["prediction_receipt_status"] = receipt_audit["status"]
    audit["prediction_receipt_eligible"] = receipt_audit["eligible"]
    audit["prediction_receipt_record_count"] = receipt_audit["record_count"]
    for phase in ("start", "end"):
        audit["prediction_receipt_bundle_sha256"][phase] = receipt_audit["phases"][
            phase
        ]["bundle_sha256"]
        audit["prediction_receipt_record_sha256s"][phase] = receipt_audit["phases"][
            phase
        ]["receipt_record_sha256s"]
        audit["prediction_receipt_capture_warnings"][phase] = receipt_audit[
            "phases"
        ][phase]["capture_contract_warnings"]
        audit["prediction_receipt_uncertainty_observations"][phase] = (
            receipt_audit["phases"][phase]["uncertainty_observations"]
        )
        audit["prediction_receipt_uncertainty_summary"][phase] = receipt_audit[
            "phases"
        ][phase]["uncertainty_summary"]
        audit["uncertainty_reasons"].extend(
            f"{phase}:{reason}"
            for reason in receipt_audit["phases"][phase]["uncertainty_reasons"]
        )
        for field in (
            "uncertainty_successful_prediction_count",
            "uncertainty_no_face_count",
            "uncertainty_capture_coverage_fraction",
            "uncertainty_conditional_scored_fraction",
        ):
            audit[field][phase] = receipt_audit["phases"][phase][field]
    audit["uncertainty_reasons"] = sorted(set(audit["uncertainty_reasons"]))
    audit["uncertainty_evidence_eligible"] = all(
        receipt_audit["phases"][phase]["uncertainty_evidence_eligible"]
        for phase in ("start", "end")
    )
    audit["uncertainty_evidence_status"] = (
        "verified_scored_no_threshold"
        if audit["uncertainty_evidence_eligible"]
        else "not_evaluable"
    )
    audit["uncertainty_coverage_risk_evaluable"] = all(
        receipt_audit["phases"][phase]["uncertainty_coverage_risk_evaluable"]
        for phase in ("start", "end")
    )
    if (
        not audit["uncertainty_evidence_eligible"]
        and not audit["uncertainty_reasons"]
    ):
        audit["uncertainty_reasons"] = [
            "prediction_receipt_uncertainty_unavailable"
        ]
    for phase in ("start", "end"):
        receipt_phase = receipt_audit["phases"][phase]
        phase_ok = receipt_phase["eligible"] is True
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
                    prediction_receipt_status="verified",
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
            and receipt_phase["bundle"] is not None
        ):
            expected_payload_sha256 = canonical_sha256(
                {
                    "samples": recomputed["samples"],
                    "capture_contract": normalized_capture,
                    "prediction_receipt_bundle": receipt_phase["bundle"],
                    "uncertainty_observations": receipt_phase[
                        "uncertainty_observations"
                    ],
                    "uncertainty_summary": receipt_phase["uncertainty_summary"],
                    "prediction_receipt_status": "verified",
                    "prediction_receipts_verified": True,
                    "model_artifact_sha256": collection.get(
                        "model_artifact_sha256"
                    ),
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
            _add_reason(audit, "gaze_integrity_ineligible", scope="reading")
    else:
        _add_reason(audit, "gaze_integrity_unavailable", scope="reading")

    validation_flags = {
        "prediction_receipt_eligible": audit["prediction_receipt_eligible"],
        "capture_contract_eligible": audit["capture_contract_eligible"],
        "target_independence_eligible": audit["target_independence_eligible"],
    }
    audit["validation_geometry_contract_eligible"] = all(
        validation_flags.values()
    )
    required_flags = {
        **validation_flags,
        "gaze_integrity_eligible": audit["gaze_integrity_eligible"],
    }
    for field, recomputed_value in required_flags.items():
        if final_quality.get(field) is not recomputed_value:
            _add_reason(audit, f"stored_{field}_mismatch", scope="reading")
    audit["geometry_contract_eligible"] = all(required_flags.values())
    if final_quality.get("geometry_contract_eligible") is not audit[
        "geometry_contract_eligible"
    ]:
        _add_reason(
            audit,
            "stored_geometry_contract_eligible_mismatch",
            scope="reading",
        )
    if not audit["geometry_contract_eligible"]:
        _add_reason(audit, "geometry_contract_ineligible", scope="reading")
    _add_reason(
        audit,
        "reading_prediction_receipts_unavailable",
        scope="reading",
    )

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
            _add_reason(audit, "telemetry_batch_unreadable", scope="reading")
            continue
        if not isinstance(batch, Mapping):
            _add_reason(audit, "telemetry_batch_invalid", scope="reading")
            continue
        samples = batch.get("samples")
        if not isinstance(samples, list):
            _add_reason(audit, "telemetry_samples_unavailable", scope="reading")
            continue
        telemetry_batch_count += 1
        telemetry_attempt_count += len(samples)
        audit["source_telemetry_sample_count"] += len(samples)
        forbidden = {"image", "image_data", "frame", "video", "audio"}
        if forbidden & set(batch) or any(
            isinstance(sample, Mapping) and forbidden & set(sample)
            for sample in samples
        ):
            _add_reason(audit, "telemetry_raw_media_prohibited", scope="reading")
        try:
            telemetry_viewport = _normalized_viewport(batch.get("viewport"))
        except ValueError:
            _add_reason(
                audit,
                "telemetry_viewport_unavailable_or_invalid",
                scope="reading",
            )
        else:
            if assessment_viewport is None or telemetry_viewport != assessment_viewport:
                _add_reason(audit, "telemetry_viewport_mismatch", scope="reading")
        if str(batch.get("study_session_id") or "") != session_id:
            _add_reason(
                audit, "telemetry_session_binding_mismatch", scope="reading"
            )
        if batch.get("visit_index") != assignment.get("visit_index"):
            _add_reason(audit, "telemetry_visit_binding_mismatch", scope="reading")
        if batch.get("capture_session_id") != linked_data.get("gaze_session_id"):
            _add_reason(
                audit,
                "telemetry_capture_session_binding_mismatch",
                scope="reading",
            )
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
            _add_reason(audit, "telemetry_payload_hash_mismatch", scope="reading")

        for sample in samples:
            if not isinstance(sample, Mapping):
                _add_reason(audit, "telemetry_sample_invalid", scope="reading")
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
                _add_reason(
                    audit, "telemetry_success_geometry_invalid", scope="reading"
                )
                continue
            successful_poses.append(normalized_pose)
            successful_face_scales.append(
                max(0.0, normalized_bbox[2] - normalized_bbox[0])
                * max(0.0, normalized_bbox[3] - normalized_bbox[1])
            )

    if session.get("state") == "completed" and final_quality and (
        telemetry_batch_count == 0 or telemetry_attempt_count == 0
    ):
        _add_reason(
            audit, "completed_gaze_session_has_no_telemetry", scope="reading"
        )

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
        _add_reason(audit, "stored_telemetry_stats_unavailable", scope="reading")
    else:
        for field, expected in recomputed_telemetry_stats.items():
            if not _metric_equal(stored_telemetry_stats.get(field), expected):
                _add_reason(
                    audit, f"stored_telemetry_{field}_mismatch", scope="reading"
                )

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
            _add_reason(
                audit, "round_timing_unavailable_or_invalid", scope="reading"
            )
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
            _add_reason(audit, "round_timing_binding_mismatch", scope="reading")
    else:
        _add_reason(audit, "stored_round_timing_unavailable", scope="reading")

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
    validation_success_fractions = [
        float(summary["prediction_success_fraction"])
        for summary in validation_summaries
    ]
    audit["recomputed_validation_quality"] = {
        "median_spatial_error_px": max(medians) if medians else None,
        "p90_spatial_error_px": max(p90_values) if p90_values else None,
        "precision_rms_px": max(precision_values) if precision_values else None,
        "prediction_success_fraction": (
            min(validation_success_fractions)
            if validation_success_fractions
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
    }
    recomputed_quality: dict[str, Any] = {
        "median_spatial_error_px": max(medians) if medians else None,
        "p90_spatial_error_px": max(p90_values) if p90_values else None,
        "precision_rms_px": max(precision_values) if precision_values else None,
        "prediction_receipt_status": audit["prediction_receipt_status"],
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
            _add_reason(audit, "recomputed_gaze_quality_invalid", scope="reading")
    audit["recomputed_quality"] = {
        **recomputed_quality,
        "gaze_quality_band": recomputed_band,
    }

    stored_collection_metrics = collection.get("gaze_quality_metrics")
    if not isinstance(stored_collection_metrics, Mapping):
        _add_reason(
            audit,
            "stored_collection_gaze_quality_metrics_unavailable",
            scope="reading",
        )
    else:
        for field, expected in recomputed_quality.items():
            if not _metric_equal(stored_collection_metrics.get(field), expected):
                _add_reason(
                    audit, f"stored_collection_{field}_mismatch", scope="reading"
                )
    if collection.get("gaze_quality_band") != recomputed_band:
        _add_reason(
            audit,
            "stored_collection_gaze_quality_band_mismatch",
            scope="reading",
        )
    for field, expected in recomputed_quality.items():
        if not _metric_equal(final_quality.get(field), expected):
            _add_reason(audit, f"stored_final_{field}_mismatch", scope="reading")
    if final_quality.get("gaze_quality_band") != recomputed_band:
        _add_reason(
            audit, "stored_final_gaze_quality_band_mismatch", scope="reading"
        )

    audit["validation_reasons"].sort()
    audit["reading_reasons"].sort()
    audit["reasons"].sort()
    audit["validation_gaze_export_eligible"] = (
        not audit["validation_reasons"]
        and audit["validation_integrity_status"] == "passed"
        and audit["validation_geometry_contract_eligible"]
    )
    audit["uncertainty_evidence_eligible"] = bool(
        audit["uncertainty_evidence_eligible"]
        and audit["validation_gaze_export_eligible"]
    )
    audit["uncertainty_evidence_status"] = (
        "verified_scored_no_threshold"
        if audit["uncertainty_evidence_eligible"]
        else "not_evaluable"
    )
    audit["uncertainty_coverage_risk_evaluable"] = bool(
        audit["uncertainty_coverage_risk_evaluable"]
        and audit["validation_gaze_export_eligible"]
    )
    audit["reading_gaze_export_eligible"] = False
    audit["base_eligible"] = audit["validation_gaze_export_eligible"]
    # A generic session-wide gaze claim remains false until reading predictions
    # have their own server-issued, single-use receipt contract.
    audit["eligible"] = False
    return audit


def _apply_pair_gaze_policy(audits: list[dict[str, Any]]) -> dict[str, str]:
    """Compare receipt-verified fixed-target validation only, never reading gaze."""

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in audits:
        if audit["pair_id"]:
            by_pair[audit["pair_id"]].append(audit)
        else:
            audit["pair_validation_gaze_comparison_status"] = "pair_id_unavailable"

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
                    audit["pair_validation_gaze_comparable"] = True
        pair_status[pair_id] = status
        for audit in pair_audits:
            audit["pair_validation_gaze_comparison_status"] = status

    for audit in audits:
        audit["validation_reasons"].sort()
        audit["reading_reasons"].sort()
        audit["reasons"].sort()
        audit["validation_gaze_export_eligible"] = audit["base_eligible"]
        audit["reading_gaze_export_eligible"] = False
        audit["eligible"] = False
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
    # This table remains eligible-only. It is intentionally empty until reading
    # predictions have their own server-issued receipt contract.
    telemetry_rows: list[dict[str, Any]] = []
    unverified_telemetry_rows: list[dict[str, Any]] = []
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
        validation_gaze_eligible = bool(audit["validation_gaze_export_eligible"])
        reading_gaze_eligible = bool(audit["reading_gaze_export_eligible"])
        validation_quality = dict(
            audit.get("recomputed_validation_quality") or {}
        )
        assessment_viewport = dict(audit.get("assessment_viewport") or {})
        uncertainty_summaries = {
            phase: dict(
                audit["prediction_receipt_uncertainty_summary"].get(phase) or {}
            )
            for phase in ("start", "end")
        }
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
                "prediction_receipt_registry_schema_version": audit[
                    "prediction_receipt_registry_schema_version"
                ],
                "prediction_receipt_status": audit["prediction_receipt_status"],
                "prediction_receipt_eligible": audit[
                    "prediction_receipt_eligible"
                ],
                "prediction_receipt_record_count": audit[
                    "prediction_receipt_record_count"
                ],
                "start_prediction_receipt_bundle_sha256": audit[
                    "prediction_receipt_bundle_sha256"
                ]["start"],
                "end_prediction_receipt_bundle_sha256": audit[
                    "prediction_receipt_bundle_sha256"
                ]["end"],
                "start_prediction_receipt_capture_warnings": _json_cell(
                    audit["prediction_receipt_capture_warnings"]["start"]
                ),
                "end_prediction_receipt_capture_warnings": _json_cell(
                    audit["prediction_receipt_capture_warnings"]["end"]
                ),
                "start_uncertainty_observations_sha256": uncertainty_summaries[
                    "start"
                ].get("observations_sha256"),
                "end_uncertainty_observations_sha256": uncertainty_summaries[
                    "end"
                ].get("observations_sha256"),
                "start_uncertainty_scored_count": uncertainty_summaries[
                    "start"
                ].get("scored_count"),
                "end_uncertainty_scored_count": uncertainty_summaries["end"].get(
                    "scored_count"
                ),
                "start_uncertainty_unavailable_count": uncertainty_summaries[
                    "start"
                ].get("unavailable_count"),
                "end_uncertainty_unavailable_count": uncertainty_summaries["end"].get(
                    "unavailable_count"
                ),
                "start_uncertainty_successful_prediction_count": audit[
                    "uncertainty_successful_prediction_count"
                ]["start"],
                "end_uncertainty_successful_prediction_count": audit[
                    "uncertainty_successful_prediction_count"
                ]["end"],
                "start_uncertainty_no_face_count": audit[
                    "uncertainty_no_face_count"
                ]["start"],
                "end_uncertainty_no_face_count": audit[
                    "uncertainty_no_face_count"
                ]["end"],
                "start_uncertainty_capture_coverage_fraction": audit[
                    "uncertainty_capture_coverage_fraction"
                ]["start"],
                "end_uncertainty_capture_coverage_fraction": audit[
                    "uncertainty_capture_coverage_fraction"
                ]["end"],
                "start_uncertainty_conditional_scored_fraction": audit[
                    "uncertainty_conditional_scored_fraction"
                ]["start"],
                "end_uncertainty_conditional_scored_fraction": audit[
                    "uncertainty_conditional_scored_fraction"
                ]["end"],
                "uncertainty_evidence_status": audit[
                    "uncertainty_evidence_status"
                ],
                "uncertainty_evidence_eligible": audit[
                    "uncertainty_evidence_eligible"
                ],
                "uncertainty_reasons": _json_cell(audit["uncertainty_reasons"]),
                "uncertainty_abstention_threshold_selected": False,
                "uncertainty_coverage_risk_evaluable": audit[
                    "uncertainty_coverage_risk_evaluable"
                ],
                "uncertainty_risk_population": "successful_predictions_only",
                "capture_contract_eligible": audit["capture_contract_eligible"],
                "target_independence_eligible": audit[
                    "target_independence_eligible"
                ],
                "gaze_integrity_eligible": audit["gaze_integrity_eligible"],
                "geometry_contract_eligible": audit[
                    "geometry_contract_eligible"
                ],
                "validation_geometry_contract_eligible": audit[
                    "validation_geometry_contract_eligible"
                ],
                "validation_gaze_export_status": (
                    "eligible" if validation_gaze_eligible else "excluded"
                ),
                "validation_gaze_export_eligible": validation_gaze_eligible,
                "validation_gaze_exclusion_reasons": _json_cell(
                    audit["validation_reasons"]
                ),
                "reading_gaze_export_status": READING_TELEMETRY_EVIDENCE_STATUS,
                "reading_gaze_export_eligible": reading_gaze_eligible,
                "reading_gaze_exclusion_reasons": _json_cell(
                    audit["reading_reasons"]
                ),
                "gaze_export_status": (
                    "validation_only"
                    if validation_gaze_eligible
                    else "behavioral_only"
                ),
                "gaze_export_eligible": False,
                "gaze_exclusion_reasons": _json_cell(audit["reasons"]),
                "pair_gaze_comparison_status": audit[
                    "pair_gaze_comparison_status"
                ],
                "pair_gaze_comparable": audit["pair_gaze_comparable"],
                "pair_validation_gaze_comparison_status": audit[
                    "pair_validation_gaze_comparison_status"
                ],
                "pair_validation_gaze_comparable": audit[
                    "pair_validation_gaze_comparable"
                ],
                "start_validation_payload_sha256": audit[
                    "validation_payload_sha256"
                ]["start"],
                "end_validation_payload_sha256": audit[
                    "validation_payload_sha256"
                ]["end"],
                "gaze_quality_band": "unavailable",
                "median_spatial_error_px": (
                    validation_quality.get("median_spatial_error_px")
                    if validation_gaze_eligible
                    else None
                ),
                "p90_spatial_error_px": (
                    validation_quality.get("p90_spatial_error_px")
                    if validation_gaze_eligible
                    else None
                ),
                "precision_rms_px": (
                    validation_quality.get("precision_rms_px")
                    if validation_gaze_eligible
                    else None
                ),
                "validation_prediction_success_fraction": (
                    validation_quality.get("prediction_success_fraction")
                    if validation_gaze_eligible
                    else None
                ),
                "prediction_success_fraction": None,
                "effective_sampling_hz": None,
                "head_pose_range": None,
                "face_scale_range": None,
                "drift_change_px": (
                    validation_quality.get("drift_change_px")
                    if validation_gaze_eligible
                    else None
                ),
            }
        )
        if not reading_gaze_eligible or not validation_gaze_eligible:
            gaze_excluded_rows.append(
                {
                    "participant_id": participant_id,
                    "study_session_id": session_id,
                    "pair_id": assignment.get("pair_id"),
                    "visit_index": assignment.get("visit_index"),
                    "gaze_measurement_contract_id": audit["contract_id"],
                    "gaze_measurement_contract_version": audit["contract_version"],
                    "gaze_measurement_contract_sha256": audit["contract_sha256"],
                    "prediction_receipt_status": audit[
                        "prediction_receipt_status"
                    ],
                    "uncertainty_evidence_status": audit[
                        "uncertainty_evidence_status"
                    ],
                    "uncertainty_evidence_eligible": audit[
                        "uncertainty_evidence_eligible"
                    ],
                    "uncertainty_reasons": _json_cell(audit["uncertainty_reasons"]),
                    "validation_gaze_export_eligible": validation_gaze_eligible,
                    "reading_gaze_export_eligible": reading_gaze_eligible,
                    "reading_telemetry_evidence_status": (
                        READING_TELEMETRY_EVIDENCE_STATUS
                    ),
                    "pair_gaze_comparison_status": audit[
                        "pair_gaze_comparison_status"
                    ],
                    "pair_validation_gaze_comparison_status": audit[
                        "pair_validation_gaze_comparison_status"
                    ],
                    "validation_exclusion_reasons": _json_cell(
                        audit["validation_reasons"]
                    ),
                    "reading_exclusion_reasons": _json_cell(
                        audit["reading_reasons"]
                    ),
                    "exclusion_reasons": _json_cell(audit["reasons"]),
                    "source_telemetry_sample_count": audit[
                        "source_telemetry_sample_count"
                    ],
                    "source_validation_sample_count": audit[
                        "source_validation_sample_count"
                    ],
                    "behavioral_fields_retained": True,
                    "receipt_verified_validation_fields_retained": (
                        validation_gaze_eligible
                    ),
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

        telemetry_root = session_path.parent / "collection" / "telemetry"
        for batch_path in sorted(telemetry_root.glob("*/*.json")):
            try:
                batch = json.loads(batch_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(batch, Mapping) or not isinstance(
                batch.get("samples"), list
            ):
                continue
            for sample_index, sample in enumerate(batch["samples"]):
                if not isinstance(sample, Mapping):
                    continue
                unverified_telemetry_rows.append(
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
                        "evidence_status": READING_TELEMETRY_EVIDENCE_STATUS,
                        "prediction_receipt_bound": False,
                        "reading_gaze_export_eligible": False,
                        "formal_evidence_eligible": False,
                        "pair_gaze_comparison_status": audit[
                            "pair_gaze_comparison_status"
                        ],
                        "pair_gaze_comparable": False,
                        "assessment_viewport_width": assessment_viewport.get(
                            "width_px"
                        ),
                        "assessment_viewport_height": assessment_viewport.get(
                            "height_px"
                        ),
                        "passage_id": batch.get("passage_id"),
                        "batch_id": batch.get("batch_id"),
                        "batch_payload_sha256": batch.get("payload_sha256"),
                        "sample_index": sample_index,
                        "monotonic_elapsed_ms": sample.get("monotonic_elapsed_ms"),
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

        if validation_gaze_eligible:
            validations_for_export = dict(collection.get("validations") or {})
            for phase in ("start", "end"):
                summary = dict(validations_for_export[phase])
                receipt_record_sha256s = audit[
                    "prediction_receipt_record_sha256s"
                ][phase]
                uncertainty_observations = audit[
                    "prediction_receipt_uncertainty_observations"
                ][phase]
                uncertainty_summary = dict(
                    audit["prediction_receipt_uncertainty_summary"][phase] or {}
                )
                for sample_index, sample in enumerate(summary.get("samples", [])):
                    uncertainty_fields = _uncertainty_export_fields(
                        uncertainty_observations[sample_index]
                    )
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
                            "prediction_receipt_status": "verified",
                            "prediction_receipt_bundle_sha256": audit[
                                "prediction_receipt_bundle_sha256"
                            ][phase],
                            "prediction_receipt_record_sha256": (
                                receipt_record_sha256s[sample_index]
                            ),
                            "uncertainty_phase_observations_sha256": (
                                uncertainty_summary.get("observations_sha256")
                            ),
                            "evidence_status": (
                                "receipt_verified_fixed_target_validation"
                            ),
                            "validation_gaze_export_eligible": True,
                            "formal_evidence_eligible": False,
                            "pair_gaze_comparison_status": audit[
                                "pair_gaze_comparison_status"
                            ],
                            "pair_gaze_comparable": audit[
                                "pair_gaze_comparable"
                            ],
                            "pair_validation_gaze_comparison_status": audit[
                                "pair_validation_gaze_comparison_status"
                            ],
                            "pair_validation_gaze_comparable": audit[
                                "pair_validation_gaze_comparable"
                            ],
                            "assessment_viewport_width": assessment_viewport.get(
                                "width_px"
                            ),
                            "assessment_viewport_height": assessment_viewport.get(
                                "height_px"
                            ),
                            "phase": phase,
                            "sample_index": sample_index,
                            **uncertainty_fields,
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
                "validation_integrity_status",
                "prediction_receipt_registry_schema_version",
                "prediction_receipt_status", "prediction_receipt_eligible",
                "prediction_receipt_record_count",
                "start_prediction_receipt_bundle_sha256",
                "end_prediction_receipt_bundle_sha256",
                "start_prediction_receipt_capture_warnings",
                "end_prediction_receipt_capture_warnings",
                "start_uncertainty_observations_sha256",
                "end_uncertainty_observations_sha256",
                "start_uncertainty_scored_count", "end_uncertainty_scored_count",
                "start_uncertainty_unavailable_count",
                "end_uncertainty_unavailable_count",
                "start_uncertainty_successful_prediction_count",
                "end_uncertainty_successful_prediction_count",
                "start_uncertainty_no_face_count", "end_uncertainty_no_face_count",
                "start_uncertainty_capture_coverage_fraction",
                "end_uncertainty_capture_coverage_fraction",
                "start_uncertainty_conditional_scored_fraction",
                "end_uncertainty_conditional_scored_fraction",
                "uncertainty_evidence_status", "uncertainty_evidence_eligible",
                "uncertainty_reasons",
                "uncertainty_abstention_threshold_selected",
                "uncertainty_coverage_risk_evaluable",
                "uncertainty_risk_population",
                "capture_contract_eligible",
                "target_independence_eligible", "gaze_integrity_eligible",
                "validation_geometry_contract_eligible", "geometry_contract_eligible",
                "validation_gaze_export_status", "validation_gaze_export_eligible",
                "validation_gaze_exclusion_reasons", "reading_gaze_export_status",
                "reading_gaze_export_eligible", "reading_gaze_exclusion_reasons",
                "gaze_export_status", "gaze_export_eligible", "gaze_exclusion_reasons",
                "pair_gaze_comparison_status", "pair_gaze_comparable",
                "pair_validation_gaze_comparison_status",
                "pair_validation_gaze_comparable",
                "start_validation_payload_sha256", "end_validation_payload_sha256",
                "gaze_quality_band", "median_spatial_error_px", "p90_spatial_error_px",
                "precision_rms_px", "validation_prediction_success_fraction",
                "prediction_success_fraction", "effective_sampling_hz",
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
                "gaze_measurement_contract_sha256", "evidence_status",
                "prediction_receipt_bound", "reading_gaze_export_eligible",
                "formal_evidence_eligible", "pair_gaze_comparison_status",
                "pair_gaze_comparable", "assessment_viewport_width",
                "assessment_viewport_height",
                "passage_id", "batch_id", "batch_payload_sha256", "sample_index",
                "monotonic_elapsed_ms",
                "prediction_success", "coarse_failure_code", "screen_xy_norm", "screen_xy_px",
                "gaze_pitch_yaw", "head_pose_pitch_yaw", "normalized_face_bbox",
                "nearest_word_index", "viewport",
            ],
            telemetry_rows,
        ),
        "reading_telemetry_unverified.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "capture_session_id",
                "gaze_measurement_contract_id", "gaze_measurement_contract_version",
                "gaze_measurement_contract_sha256", "evidence_status",
                "prediction_receipt_bound", "reading_gaze_export_eligible",
                "formal_evidence_eligible", "pair_gaze_comparison_status",
                "pair_gaze_comparable", "assessment_viewport_width",
                "assessment_viewport_height", "passage_id", "batch_id",
                "batch_payload_sha256", "sample_index", "monotonic_elapsed_ms",
                "prediction_success", "coarse_failure_code", "screen_xy_norm",
                "screen_xy_px", "gaze_pitch_yaw", "head_pose_pitch_yaw",
                "normalized_face_bbox", "nearest_word_index", "viewport",
            ],
            unverified_telemetry_rows,
        ),
        "validation_samples.csv": (
            [
                "participant_id", "study_session_id", "visit_index", "capture_session_id",
                "gaze_measurement_contract_id", "gaze_measurement_contract_version",
                "gaze_measurement_contract_sha256", "validation_payload_sha256",
                "prediction_receipt_status", "prediction_receipt_bundle_sha256",
                "prediction_receipt_record_sha256",
                "uncertainty_phase_observations_sha256",
                "uncertainty_observation_sha256", "uncertainty_observation_json",
                "uncertainty_schema_version", "uncertainty_status",
                "uncertainty_evidence_status", "uncertainty_evidence_eligible",
                "uncertainty_coverage_risk_status",
                "uncertainty_definition_sha256", "uncertainty_score",
                "uncertainty_ood_value", "uncertainty_ood_percentile",
                "uncertainty_leverage_value", "uncertainty_leverage_percentile",
                "uncertainty_disagreement_value",
                "uncertainty_disagreement_percentile",
                "uncertainty_jackknife_disagreement_covariance_norm",
                "uncertainty_jackknife_disagreement_covariance_px",
                "uncertainty_abstention_status",
                "uncertainty_abstention_threshold_json", "uncertainty_reason",
                "evidence_status",
                "validation_gaze_export_eligible", "formal_evidence_eligible",
                "pair_gaze_comparison_status", "pair_gaze_comparable",
                "pair_validation_gaze_comparison_status",
                "pair_validation_gaze_comparable",
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
                "gaze_measurement_contract_sha256", "prediction_receipt_status",
                "uncertainty_evidence_status", "uncertainty_evidence_eligible",
                "uncertainty_reasons",
                "validation_gaze_export_eligible", "reading_gaze_export_eligible",
                "reading_telemetry_evidence_status", "pair_gaze_comparison_status",
                "pair_validation_gaze_comparison_status", "validation_exclusion_reasons",
                "reading_exclusion_reasons", "exclusion_reasons",
                "source_telemetry_sample_count", "source_validation_sample_count",
                "behavioral_fields_retained", "receipt_verified_validation_fields_retained",
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
    validation_eligible_audits = [
        audit for audit in all_audits if audit["validation_gaze_export_eligible"]
    ]
    uncertainty_eligible_audits = [
        audit for audit in all_audits if audit["uncertainty_evidence_eligible"]
    ]
    uncertainty_definition_sha256s = sorted(
        {
            str(row["uncertainty_definition_sha256"])
            for row in validation_rows
            if row.get("uncertainty_evidence_eligible")
            and row.get("uncertainty_definition_sha256")
        }
    )
    gaze_exclusion_reason_counts = Counter(
        reason
        for audit in all_audits
        for reason in audit["reasons"]
    )
    validation_exclusion_reason_counts = Counter(
        reason
        for audit in all_audits
        if not audit["validation_gaze_export_eligible"]
        for reason in audit["validation_reasons"]
    )
    reading_exclusion_reason_counts = Counter(
        reason for audit in all_audits for reason in audit["reading_reasons"]
    )
    eligible_contract_counts = Counter(
        (
            str(audit["contract_id"]),
            str(audit["contract_version"]),
            str(audit["contract_sha256"]),
        )
        for audit in validation_eligible_audits
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
            "policy": (
                "receipt_verified_fixed_target_validation_separated_from_"
                "client_roundtrip_reading_telemetry"
            ),
            "session_gaze_eligible_count": 0,
            "session_gaze_excluded_count": len(all_audits),
            "validation_gaze_eligible_count": len(validation_eligible_audits),
            "validation_gaze_excluded_count": (
                len(all_audits) - len(validation_eligible_audits)
            ),
            "reading_gaze_eligible_count": 0,
            "reading_gaze_unverified_session_count": len(all_audits),
            "gaze_tables_contain_only_eligible_sessions": True,
            "eligible_reading_gaze_table_row_count": len(telemetry_rows),
            "client_roundtrip_unverified_reading_row_count": len(
                unverified_telemetry_rows
            ),
            "unverified_reading_telemetry_separate_from_eligible_gaze": True,
            "legacy_or_unavailable_gaze_mixed_with_eligible": False,
            "validation_payload_hash_required": True,
            "validation_payload_hash_scope": (
                "server_receipt_outcomes_capture_bundle_uncertainty_model_"
                "measurement_viewport"
            ),
            "prediction_values_tamper_resistant": False,
            "prediction_tamper_resistance_scope": (
                "receipt_hashes_detect_integrity_drift_but_are_not_keyed_signatures"
            ),
            "client_posted_validation_prediction_values_trusted": False,
            "server_issued_validation_prediction_receipts_required": True,
            "validation_prediction_receipt_registry_verified_before_export": True,
            "receipt_uncertainty_schema_version": (
                RUNTIME_OBSERVATION_SCHEMA_VERSION
            ),
            "receipt_uncertainty_observations_bound_to_validation_payload": True,
            "uncertainty_geometry_eligibility_independent_of_availability": True,
            "uncertainty_evidence_eligible_session_count": len(
                uncertainty_eligible_audits
            ),
            "uncertainty_evidence_not_evaluable_session_count": (
                len(all_audits) - len(uncertainty_eligible_audits)
            ),
            "uncertainty_coverage_risk_evaluable_session_count": sum(
                bool(audit["uncertainty_coverage_risk_evaluable"])
                for audit in all_audits
            ),
            "uncertainty_scored_validation_row_count": sum(
                bool(row.get("uncertainty_evidence_eligible"))
                for row in validation_rows
            ),
            "uncertainty_unavailable_validation_row_count": sum(
                not bool(row.get("uncertainty_evidence_eligible"))
                for row in validation_rows
            ),
            "uncertainty_no_face_validation_row_count": sum(
                row.get("uncertainty_status") == "unavailable_sensor_failure"
                for row in validation_rows
            ),
            "uncertainty_capture_coverage_definition": (
                "scored_successful_predictions_divided_by_all_fixed_target_attempts"
            ),
            "uncertainty_risk_population": "successful_predictions_only",
            "uncertainty_definition_sha256s": uncertainty_definition_sha256s,
            "uncertainty_abstention_threshold_selected": False,
            "uncertainty_threshold_may_be_selected_from_this_export": False,
            "raw_prediction_receipt_tokens_exported": False,
            "authorization_fingerprints_exported": False,
            "reading_prediction_receipts_available": False,
            "reading_prediction_receipts_required_before_gaze_eligibility": True,
            "reading_telemetry_evidence_status": READING_TELEMETRY_EVIDENCE_STATUS,
            "frozen_measurement_contract_snapshot_required": True,
            "assessment_viewport_binding_required": True,
            "capture_contract_eligibility_required": True,
            "target_independence_eligibility_required": True,
            "nonempty_telemetry_required_for_reading_diagnostics": True,
            "stored_telemetry_stats_must_match_raw_batches": True,
            "final_gaze_quality_metrics_recomputed_for_diagnostics_only": True,
            "pair_validation_comparison_requires_same_contract_sha256": True,
            "pair_reading_gaze_comparison_allowed": False,
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
            "pair_validation_comparison_status_counts": dict(
                sorted(pair_status_counts.items())
            ),
            "pair_validation_comparable_count": pair_status_counts.get(
                "comparable_same_measurement_contract",
                0,
            ),
            "pair_validation_not_comparable_count": (
                len(pair_status)
                - pair_status_counts.get("comparable_same_measurement_contract", 0)
            ),
            "exclusion_reason_counts": dict(
                sorted(gaze_exclusion_reason_counts.items())
            ),
            "validation_exclusion_reason_counts": dict(
                sorted(validation_exclusion_reason_counts.items())
            ),
            "reading_exclusion_reason_counts": dict(
                sorted(reading_exclusion_reason_counts.items())
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
