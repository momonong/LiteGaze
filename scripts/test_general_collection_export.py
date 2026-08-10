"""Focused stdlib tests for gaze provenance in the private exporter."""

from __future__ import annotations

import copy
import csv
import importlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from core.gaze_core.capture_contract import (
    build_fit_target_contract,
    compare_capture_contracts,
    load_participant_gaze_measurement_contract,
)

# System Python in the lightweight lane lacks OpenCV, which the package facade
# imports for unrelated calibration helpers. Temporarily install a namespace
# only while loading the pure exporter dependencies, then restore *all* module
# and parent-package state so a shared unittest worker can import the real
# participant-study facade afterward.
import core as core_package

_PARTICIPANT_PREFIX = "core.participant_study"
_saved_participant_modules = {
    name: module
    for name, module in sys.modules.items()
    if name == _PARTICIPANT_PREFIX or name.startswith(_PARTICIPANT_PREFIX + ".")
}
_had_core_participant_attribute = hasattr(core_package, "participant_study")
_saved_core_participant_attribute = getattr(core_package, "participant_study", None)
_installed_temporary_namespace = _PARTICIPANT_PREFIX not in sys.modules
if _installed_temporary_namespace:
    participant_study_package = types.ModuleType(_PARTICIPANT_PREFIX)
    participant_study_package.__path__ = [
        str(ROOT / "core" / "participant_study")
    ]
    sys.modules[_PARTICIPANT_PREFIX] = participant_study_package

try:
    _general_collection = importlib.import_module(
        "core.participant_study.general_collection"
    )
    _participant_protocol = importlib.import_module("core.participant_study.protocol")
    _exporter = importlib.import_module("scripts.export_general_collection_dataset")
finally:
    if _installed_temporary_namespace:
        for module_name in list(sys.modules):
            if module_name == _PARTICIPANT_PREFIX or module_name.startswith(
                _PARTICIPANT_PREFIX + "."
            ):
                del sys.modules[module_name]
        sys.modules.update(_saved_participant_modules)
        if _had_core_participant_attribute:
            core_package.participant_study = _saved_core_participant_attribute
        elif hasattr(core_package, "participant_study"):
            delattr(core_package, "participant_study")

canonical_sha256 = _general_collection.canonical_sha256
evaluate_validation_target_independence = (
    _general_collection.evaluate_validation_target_independence
)
load_general_bank = _general_collection.load_general_bank
load_general_protocol = _general_collection.load_general_protocol
summarize_validation_samples = _general_collection.summarize_validation_samples
validation_target_definitions = _general_collection.validation_target_definitions
load_protocol = _participant_protocol.load_protocol
export_bundle = _exporter.export_bundle


VIEWPORT = {"width_px": 1280, "height_px": 800}
PREDICTION_RECEIPT_SCHEMA_VERSION = 1
RUNTIME_UNCERTAINTY_SCHEMA_VERSION = 1
UNCERTAINTY_DEFINITION_SHA256 = (
    "75a24c436e9a89024462268812ecc9be149a1958b3911e5cd71c3974b235a180"
)


def _scored_uncertainty_observation() -> dict[str, object]:
    covariance_norm = [[0.0004, 0.0], [0.0, 0.0001]]
    covariance_px = [[163.84, 0.0], [0.0, 16.0]]
    return {
        "schema_version": RUNTIME_UNCERTAINTY_SCHEMA_VERSION,
        "status": "scored_no_threshold",
        "definition_sha256": UNCERTAINTY_DEFINITION_SHA256,
        "score": 0.6,
        "components": {
            "ood": {"value": 1.25, "percentile": 0.2},
            "leverage": {"value": 0.75, "percentile": 0.4},
            "disagreement": {
                "value": math.sqrt((0.0004 + 0.0001) / 2.0),
                "percentile": 0.6,
            },
        },
        "jackknife_disagreement_covariance_norm": covariance_norm,
        "jackknife_disagreement_covariance_px": covariance_px,
        "abstention": {"status": "not_selected", "threshold": None},
    }


def _legacy_unavailable_uncertainty(
    *,
    prediction_success: bool,
) -> dict[str, object]:
    return {
        "schema_version": RUNTIME_UNCERTAINTY_SCHEMA_VERSION,
        "status": (
            "unavailable_receipt_missing"
            if prediction_success
            else "unavailable_sensor_failure"
        ),
        "reason": (
            "legacy prediction receipt did not contain runtime uncertainty evidence"
        ),
    }


def _uncertainty_summary(
    observations: list[dict[str, object]],
    *,
    status: str,
) -> dict[str, object]:
    scored_count = sum(
        observation["uncertainty"]["status"] == "scored_no_threshold"
        for observation in observations
    )
    return {
        "schema_version": RUNTIME_UNCERTAINTY_SCHEMA_VERSION,
        "status": status,
        "count": len(observations),
        "scored_count": scored_count,
        "unavailable_count": len(observations) - scored_count,
        "observation_sha256s": [
            canonical_sha256(observation) for observation in observations
        ],
        "observations_sha256": canonical_sha256(observations),
    }


def _refresh_validation_payload_sha256(summary: dict[str, object]) -> None:
    summary["validation_payload_sha256"] = canonical_sha256(
        {
            "samples": summary["samples"],
            "capture_contract": summary["capture_contract"],
            "prediction_receipt_bundle": summary["prediction_receipt_bundle"],
            "uncertainty_observations": summary["uncertainty_observations"],
            "uncertainty_summary": summary["uncertainty_summary"],
            "prediction_receipt_status": "verified",
            "prediction_receipts_verified": True,
            "model_artifact_sha256": summary["model_artifact_sha256"],
            "gaze_measurement_contract_sha256": summary[
                "gaze_measurement_contract_sha256"
            ],
            "assessment_viewport": summary["assessment_viewport"],
        }
    )


def _capture_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "intent_width_px": 1280,
        "intent_height_px": 720,
        "intent_frame_rate_hz": 30.0,
        "source_width_px": 1280,
        "source_height_px": 720,
        "source_frame_rate_hz": 30.0,
        "transport_width_px": 640,
        "transport_height_px": 360,
        "resize_policy": "fit_width_preserve_aspect",
        "mime_type": "image/jpeg",
        "jpeg_quality": 0.86,
        "mirror_applied": False,
        "facing_mode": "user",
    }


def _fit_target_contract(measurement_contract: dict[str, object]) -> dict[str, object]:
    participant_calibration = dict(
        measurement_contract["participant_calibration"]  # type: ignore[arg-type]
    )
    targets = [
        {
            "target_x_norm": float(target["target_x_viewport_fraction"]) * 2.0 - 1.0,
            "target_y_norm": float(target["target_y_viewport_fraction"]) * 2.0 - 1.0,
        }
        for target in participant_calibration["frozen_targets"]  # type: ignore[index]
    ]
    return build_fit_target_contract(targets)


def _validation_summary(
    measurement_contract: dict[str, object],
    *,
    offset_px: float,
    calibration_capture: dict[str, object],
    fit_target_contract: dict[str, object],
    no_face_ordinal: int | None = None,
) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for target in validation_target_definitions(measurement_contract):
        target_x_px = float(
            math.floor(
                float(target["target_x_viewport_fraction"])
                * VIEWPORT["width_px"]
                + 0.5
            )
        )
        target_y_px = float(
            math.floor(
                float(target["target_y_viewport_fraction"])
                * VIEWPORT["height_px"]
                + 0.5
            )
        )
        for _ in range(3):
            samples.append(
                {
                    "target_id": target["target_id"],
                    "target_x_norm": target["target_x_norm"],
                    "target_y_norm": target["target_y_norm"],
                    "target_x_px": target_x_px,
                    "target_y_px": target_y_px,
                    "prediction_success": True,
                    "predicted_x_px": target_x_px + offset_px,
                    "predicted_y_px": target_y_px,
                }
            )
    if no_face_ordinal is not None:
        failed_sample = samples[no_face_ordinal]
        failed_sample["prediction_success"] = False
        failed_sample.pop("predicted_x_px")
        failed_sample.pop("predicted_y_px")
    summary = summarize_validation_samples(
        samples,
        viewport_width_px=VIEWPORT["width_px"],
        viewport_height_px=VIEWPORT["height_px"],
        measurement_contract=measurement_contract,
        prediction_receipt_status="verified",
    )
    contract_sha256 = canonical_sha256(measurement_contract)
    provenance = {
        "contract_id": measurement_contract["contract_id"],
        "contract_version": measurement_contract["contract_version"],
        "sha256": contract_sha256,
    }
    observed_capture = _capture_contract()
    summary["capture_contract"] = observed_capture
    summary["assessment_viewport"] = dict(VIEWPORT)
    summary["gaze_measurement_contract"] = provenance
    summary["gaze_measurement_contract_sha256"] = contract_sha256
    summary["samples_sha256"] = canonical_sha256(summary["samples"])
    summary["capture_contract_check"] = compare_capture_contracts(
        calibration_capture,
        observed_capture,
    )
    summary["target_independence_check"] = evaluate_validation_target_independence(
        summary,
        fit_target_contract,
        measurement_contract=measurement_contract,
    )
    return summary


def _receipt_records_for_summary(
    summary: dict[str, object],
    measurement_contract: dict[str, object],
    *,
    phase: str,
    study_session_id: str,
    authorization_fingerprint_sha256: str,
    assessment_id: str,
    model_name: str,
    model_artifact_sha256: str,
    capture_session_id: str,
    uncertainty_mode: str = "scored",
) -> dict[str, object]:
    record_sha256s: list[str] = []
    records: dict[str, object] = {}
    uncertainty_observations: list[dict[str, object]] = []
    prediction_failures: list[dict[str, object]] = []
    targets = validation_target_definitions(measurement_contract)
    samples = list(summary["samples"])  # type: ignore[arg-type]
    capture_contract = _capture_contract()
    for ordinal, sample_raw in enumerate(samples):
        sample = dict(sample_raw)
        target = targets[ordinal // 3]
        target_x_px = float(sample["target_x_px"])
        target_y_px = float(sample["target_y_px"])
        prediction_success = sample.get("prediction_success") is True
        predicted_x_px = (
            float(sample["predicted_x_px"]) if prediction_success else None
        )
        predicted_y_px = (
            float(sample["predicted_y_px"]) if prediction_success else None
        )
        receipt_id_sha256 = canonical_sha256(
            {
                "fixture_receipt": study_session_id,
                "phase": phase,
                "ordinal": ordinal,
            }
        )
        issued = {
            "schema_version": PREDICTION_RECEIPT_SCHEMA_VERSION,
            "receipt_id_sha256": receipt_id_sha256,
            "issued_at_utc": "2026-08-10T00:10:00+00:00",
            "study_session_id": study_session_id,
            "authorization_fingerprint_sha256": (
                authorization_fingerprint_sha256
            ),
            "assessment_id": assessment_id,
            "model_name": model_name,
            "model_artifact_sha256": model_artifact_sha256,
            "capture_session_id": capture_session_id,
            "phase": phase,
            "receipt_ordinal": ordinal,
            "target_repeat_index": ordinal % 3,
            "target": {
                "target_id": target["target_id"],
                "target_x_viewport_fraction": float(
                    target["target_x_viewport_fraction"]
                ),
                "target_y_viewport_fraction": float(
                    target["target_y_viewport_fraction"]
                ),
                "target_x_norm": float(target["target_x_norm"]),
                "target_y_norm": float(target["target_y_norm"]),
                "target_x_px": target_x_px,
                "target_y_px": target_y_px,
            },
            "viewport": dict(VIEWPORT),
            "measurement_contract_sha256": canonical_sha256(
                measurement_contract
            ),
            "capture_contract": capture_contract,
            "capture_contract_check": {
                "status": "compatible",
                "compatible": True,
                "reasons": [],
                "warnings": [],
            },
            "prediction": (
                {
                    "success": True,
                    "screen_xy_px": [predicted_x_px, predicted_y_px],
                    "screen_xy_norm": [
                        predicted_x_px / VIEWPORT["width_px"] * 2.0 - 1.0,
                        predicted_y_px / VIEWPORT["height_px"] * 2.0 - 1.0,
                    ],
                    "http_status": 200,
                    "failure_stage": None,
                    "failure_code": None,
                    "error": None,
                }
                if prediction_success
                else {
                    "success": False,
                    "screen_xy_px": None,
                    "screen_xy_norm": None,
                    "http_status": 400,
                    "failure_stage": "attributable_sensor_failure",
                    "failure_code": "no_face_detected",
                    "error": "No face detected",
                }
            ),
        }
        prediction = issued["prediction"]
        if uncertainty_mode == "scored":
            prediction["uncertainty_schema_version"] = (
                RUNTIME_UNCERTAINTY_SCHEMA_VERSION
            )
            prediction["uncertainty"] = (
                _scored_uncertainty_observation()
                if prediction_success
                else {
                    "schema_version": RUNTIME_UNCERTAINTY_SCHEMA_VERSION,
                    "status": "unavailable_sensor_failure",
                    "reason": "no face was detected, so no sensor observation was scored",
                }
            )
        elif uncertainty_mode != "legacy_missing":
            raise ValueError("fixture uncertainty mode is invalid")
        issued_record_sha256 = canonical_sha256(issued)
        record_sha256s.append(issued_record_sha256)
        records[receipt_id_sha256] = {
            "issued": issued,
            "issued_record_sha256": issued_record_sha256,
            "consumed_at_utc": "2026-08-10T00:20:00+00:00",
            "consumed_validation_phase": phase,
        }
        if not prediction_success:
            prediction_failures.append(
                {
                    "receipt_record_sha256": issued_record_sha256,
                    "failure_stage": "attributable_sensor_failure",
                    "failure_code": "no_face_detected",
                    "http_status": 400,
                }
            )
        uncertainty_observations.append(
            {
                "schema_version": RUNTIME_UNCERTAINTY_SCHEMA_VERSION,
                "receipt_record_sha256": issued_record_sha256,
                "phase": phase,
                "receipt_ordinal": ordinal,
                "target_id": target["target_id"],
                "target_repeat_index": ordinal % 3,
                "prediction_success": prediction_success,
                "uncertainty": (
                    copy.deepcopy(prediction["uncertainty"])
                    if uncertainty_mode == "scored"
                    else _legacy_unavailable_uncertainty(
                        prediction_success=prediction_success
                    )
                ),
            }
        )
    bundle_core = {
        "schema_version": PREDICTION_RECEIPT_SCHEMA_VERSION,
        "status": "verified",
        "phase": phase,
        "count": len(record_sha256s),
        "receipt_record_sha256s": record_sha256s,
    }
    summary["prediction_receipt_bundle"] = {
        **bundle_core,
        "bundle_sha256": canonical_sha256(bundle_core),
    }
    summary["model_artifact_sha256"] = model_artifact_sha256
    summary["prediction_failures"] = prediction_failures
    summary["uncertainty_observations"] = uncertainty_observations
    summary["uncertainty_summary"] = _uncertainty_summary(
        uncertainty_observations,
        status="verified",
    )
    _refresh_validation_payload_sha256(summary)
    return records


def _session(
    root: Path,
    *,
    session_id: str,
    participant_id: str,
    pair_id: str,
    visit_index: int,
    measurement_contract: dict[str, object] | None = None,
    uncertainty_mode: str = "scored",
    no_face_ordinal: int | None = None,
    device_overrides: dict[str, object] | None = None,
) -> Path:
    protocol = load_general_protocol()
    bank = load_general_bank()
    contract = copy.deepcopy(
        measurement_contract or load_participant_gaze_measurement_contract()
    )
    contract_sha256 = canonical_sha256(contract)
    calibration_capture = _capture_contract()
    fit_target_contract = _fit_target_contract(contract)
    assessment_id = f"GC-{session_id}"
    authorization_fingerprint_sha256 = canonical_sha256(
        {"fixture_access": session_id}
    )
    model_name = f"fixture-model-{session_id}"
    model_artifact_sha256 = canonical_sha256({"fixture_model": session_id})
    capture_session_id = f"GAZE-{session_id}"
    start = _validation_summary(
        contract,
        offset_px=6.0,
        calibration_capture=calibration_capture,
        fit_target_contract=fit_target_contract,
        no_face_ordinal=no_face_ordinal,
    )
    end = _validation_summary(
        contract,
        offset_px=9.0,
        calibration_capture=calibration_capture,
        fit_target_contract=fit_target_contract,
        no_face_ordinal=no_face_ordinal,
    )
    prediction_receipt_records: dict[str, object] = {}
    for phase, summary in (("start", start), ("end", end)):
        prediction_receipt_records.update(
            _receipt_records_for_summary(
                summary,
                contract,
                phase=phase,
                study_session_id=session_id,
                authorization_fingerprint_sha256=(
                    authorization_fingerprint_sha256
                ),
                assessment_id=assessment_id,
                model_name=model_name,
                model_artifact_sha256=model_artifact_sha256,
                capture_session_id=capture_session_id,
                uncertainty_mode=uncertainty_mode,
            )
        )
    face_scale = (0.6 - 0.2) * (0.6 - 0.2)
    expected_quality = {
        "median_spatial_error_px": 9.0,
        "p90_spatial_error_px": 9.0,
        "precision_rms_px": 0.0,
        "prediction_success_fraction": 1.0,
        "effective_sampling_hz": 10.0,
        "head_pose_range": [0.0, 0.0],
        "face_scale_range": 0.0,
        "drift_change_px": 3.0,
        "capture_contract_eligible": True,
        "target_independence_eligible": True,
        "gaze_integrity_eligible": True,
        "telemetry_segments_contiguous": True,
    }
    session = {
        "schema_version": 1,
        "study_session_id": session_id,
        "access_token_sha256": authorization_fingerprint_sha256,
        "participant_id": participant_id,
        "state": "completed",
        "created_at_utc": "2026-08-10T00:00:00+00:00",
        "events": [
            {
                "event": "general_collection_completed",
                "at_utc": "2026-08-10T01:00:00+00:00",
            }
        ],
        "collection_assignment": {
            "protocol_sha256": canonical_sha256(protocol),
            "bank_sha256": canonical_sha256(bank),
            "pair_id": pair_id,
            "visit_index": visit_index,
            "schedule_cell": 0,
            "sequence": "AB",
            "form_id": "form_a" if visit_index == 1 else "form_b",
            "order_cell": 0,
        },
        "linked_data": {
            "gaze_session_id": capture_session_id,
            "assessment_id": assessment_id,
            "model_name": model_name,
            "model_artifact_sha256": model_artifact_sha256,
        },
        "data_governance": {
            "storage_security": "unencrypted_self_development",
            "retention_policy": "manual_until_researcher_deletes",
            "self_only": True,
        },
        "general_collection": {
            "assessment_id": assessment_id,
            "model_artifact_sha256": model_artifact_sha256,
            "prediction_receipts": {
                "schema_version": PREDICTION_RECEIPT_SCHEMA_VERSION,
                "records": prediction_receipt_records,
            },
            "profile": {
                "english_l1": "yes",
                "english_age_of_acquisition_band": "0_5",
                "weekly_english_reading_band": "5_plus_hours",
                "vision_correction": "none",
                "education_band": "graduate",
            },
            "gaze_measurement_contract": {
                "contract_id": contract["contract_id"],
                "contract_version": contract["contract_version"],
                "sha256": contract_sha256,
                "contract": contract,
            },
            "assessment_viewport": dict(VIEWPORT),
            "validations": {"start": start, "end": end},
            "gaze_integrity": {"eligible": True, "reasons": []},
            "rounds": [{"reading_elapsed_ms": 1000.0}],
            "telemetry_stats": {
                "batch_count": 1,
                "attempt_count": 10,
                "successful_count": 10,
                "head_pose_min": [0.0, 0.0],
                "head_pose_max": [0.0, 0.0],
                "face_scale_min": face_scale,
                "face_scale_max": face_scale,
            },
            "gaze_quality_metrics": dict(expected_quality),
            "gaze_quality_band": "word_level_candidate",
        },
        "quality": {
            "general_system_check": {
                "device": {
                    "device_class": "desktop",
                    "browser_family": "chromium",
                    "viewport_width": VIEWPORT["width_px"],
                    "viewport_height": VIEWPORT["height_px"],
                    "device_pixel_ratio_bucket": "1_2",
                    "camera_width": 1280,
                    "camera_height": 720,
                    "estimated_camera_fps_band": "20_30",
                    **(device_overrides or {}),
                }
            },
            "calibration": {
                "model_artifact_sha256": model_artifact_sha256,
                "capture_contract": calibration_capture,
                "fit_target_contract": fit_target_contract,
            },
            "general_collection": {
                **expected_quality,
                "geometry_contract_eligible": True,
                "gaze_quality_band": "word_level_candidate",
            },
        },
    }
    session_dir = (
        root
        / "data"
        / "participant_studies"
        / load_protocol()["protocol_id"]
        / "rehearsals"
        / session_id
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    telemetry_payload = {
        "batch_id": "B-12345678",
        "passage_id": "fixture-passage",
        "viewport": dict(VIEWPORT),
        "samples": [
            {
                "monotonic_elapsed_ms": float(sample_index * 100),
                "prediction_success": True,
                "coarse_failure_code": None,
                "screen_xy_norm": [0.0, 0.0],
                "screen_xy_px": [640.0, 400.0],
                "gaze_pitch_yaw": [0.0, 0.0],
                "head_pose_pitch_yaw": [0.0, 0.0],
                "normalized_face_bbox": [0.2, 0.2, 0.6, 0.6],
                "nearest_word_index": None,
            }
            for sample_index in range(10)
        ],
    }
    telemetry = {
        "schema_version": 1,
        "participant_id": participant_id,
        "study_session_id": session_id,
        "visit_index": visit_index,
        "capture_session_id": f"GAZE-{session_id}",
        "payload_sha256": canonical_sha256(telemetry_payload),
        **telemetry_payload,
    }
    telemetry_path = (
        session_dir / "collection" / "telemetry" / "fixture-passage" / "B-12345678.json"
    )
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.write_text(
        json.dumps(telemetry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    round_path = session_dir / "collection" / "rounds" / "R01.json"
    round_path.parent.mkdir(parents=True, exist_ok=True)
    round_path.write_text(
        json.dumps(
            {
                "round_number": 1,
                "passage_id": "fixture-passage",
                "passage_family_id": "fixture-family",
                "difficulty_band": "standard",
                "reading_elapsed_ms": 1000,
                "passage_self_report": {
                    "understanding": 4,
                    "mental_effort": 2,
                    "read_complete": True,
                    "interrupted": False,
                },
                "word_reviews": [
                    {
                        "probe_id": "fixture-probe",
                        "surface": "fixture",
                        "stratum": "known",
                        "label": "known",
                    }
                ],
                "word_layout": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return session_dir / "session.json"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _receipt_record_at(
    session: dict[str, object],
    *,
    phase: str,
    ordinal: int,
) -> dict[str, object]:
    collection = session["general_collection"]  # type: ignore[index]
    summary = collection["validations"][phase]  # type: ignore[index]
    record_sha256 = summary["prediction_receipt_bundle"][  # type: ignore[index]
        "receipt_record_sha256s"
    ][ordinal]
    records = collection["prediction_receipts"]["records"]  # type: ignore[index]
    matches = [
        record
        for record in records.values()
        if record["issued_record_sha256"] == record_sha256
    ]
    if len(matches) != 1:
        raise AssertionError("fixture receipt record is not unique")
    return matches[0]


def _rehash_receipt_record(
    session: dict[str, object],
    *,
    phase: str,
    ordinal: int,
) -> None:
    collection = session["general_collection"]  # type: ignore[index]
    summary = collection["validations"][phase]  # type: ignore[index]
    bundle = summary["prediction_receipt_bundle"]
    record = _receipt_record_at(session, phase=phase, ordinal=ordinal)
    old_record_sha256 = bundle["receipt_record_sha256s"][ordinal]
    new_record_sha256 = canonical_sha256(record["issued"])
    record["issued_record_sha256"] = new_record_sha256
    if bundle["receipt_record_sha256s"][ordinal] != old_record_sha256:
        raise AssertionError("fixture receipt order changed unexpectedly")
    bundle["receipt_record_sha256s"][ordinal] = new_record_sha256
    summary["uncertainty_observations"][ordinal][
        "receipt_record_sha256"
    ] = new_record_sha256
    summary["uncertainty_summary"] = _uncertainty_summary(
        summary["uncertainty_observations"],
        status="verified",
    )
    bundle_core = {
        "schema_version": bundle["schema_version"],
        "status": bundle["status"],
        "phase": bundle["phase"],
        "count": bundle["count"],
        "receipt_record_sha256s": bundle["receipt_record_sha256s"],
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle_core)
    _refresh_validation_payload_sha256(summary)


class GeneralCollectionExportProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-export-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_dependency_bootstrap_does_not_leak_fake_participant_package(self) -> None:
        participant_module = sys.modules.get(_PARTICIPANT_PREFIX)
        self.assertTrue(
            participant_module is None
            or bool(getattr(participant_module, "__file__", None)),
            "temporary participant-study namespace leaked into the unittest worker",
        )

    def test_valid_session_exports_bound_gaze_with_provenance(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-VALID",
            participant_id="P-VALID",
            pair_id="PAIR-VALID",
            visit_index=1,
        )
        output = self.root / "export"
        manifest = export_bundle(self.root, output)
        stored_session = json.loads(session_path.read_text(encoding="utf-8"))
        contract_sha256 = stored_session["general_collection"][
            "gaze_measurement_contract"
        ]["sha256"]

        session_row = _rows(output / "sessions.csv")[0]
        self.assertEqual(session_row["gaze_export_status"], "validation_only")
        self.assertEqual(session_row["gaze_export_eligible"], "False")
        self.assertEqual(session_row["validation_gaze_export_status"], "eligible")
        self.assertEqual(session_row["validation_gaze_export_eligible"], "True")
        self.assertEqual(
            session_row["reading_gaze_export_status"],
            "client_roundtrip_unverified",
        )
        self.assertEqual(session_row["reading_gaze_export_eligible"], "False")
        self.assertEqual(session_row["prediction_receipt_status"], "verified")
        self.assertEqual(
            session_row["uncertainty_evidence_status"],
            "verified_scored_no_threshold",
        )
        self.assertEqual(session_row["uncertainty_evidence_eligible"], "True")
        self.assertEqual(
            session_row["uncertainty_abstention_threshold_selected"],
            "False",
        )
        self.assertEqual(
            session_row["gaze_measurement_contract_sha256"],
            contract_sha256,
        )
        self.assertEqual(
            session_row["pair_gaze_comparison_status"],
            "reading_telemetry_unverified",
        )
        self.assertEqual(
            session_row["pair_validation_gaze_comparison_status"],
            "single_visit_only",
        )
        validation_rows = _rows(output / "validation_samples.csv")
        self.assertEqual(len(validation_rows), 30)
        self.assertEqual(
            {row["uncertainty_status"] for row in validation_rows},
            {"scored_no_threshold"},
        )
        self.assertEqual(
            {row["uncertainty_definition_sha256"] for row in validation_rows},
            {UNCERTAINTY_DEFINITION_SHA256},
        )
        self.assertEqual(
            {row["uncertainty_abstention_threshold_json"] for row in validation_rows},
            {"null"},
        )
        self.assertEqual(
            {row["uncertainty_evidence_eligible"] for row in validation_rows},
            {"True"},
        )
        self.assertEqual(
            {row["formal_evidence_eligible"] for row in validation_rows},
            {"False"},
        )
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        unverified_rows = _rows(output / "reading_telemetry_unverified.csv")
        self.assertEqual(len(unverified_rows), 10)
        self.assertEqual(
            {row["evidence_status"] for row in unverified_rows},
            {"client_roundtrip_unverified"},
        )
        self.assertEqual(
            {row["prediction_receipt_bound"] for row in unverified_rows},
            {"False"},
        )
        self.assertEqual(len(_rows(output / "gaze_excluded_sessions.csv")), 1)
        self.assertEqual(
            manifest["gaze_provenance"]["session_gaze_eligible_count"],
            0,
        )
        self.assertEqual(
            manifest["gaze_provenance"]["validation_gaze_eligible_count"],
            1,
        )
        self.assertEqual(
            manifest["gaze_provenance"]["reading_gaze_eligible_count"],
            0,
        )
        self.assertEqual(
            manifest["gaze_provenance"][
                "uncertainty_evidence_eligible_session_count"
            ],
            1,
        )
        self.assertFalse(
            manifest["gaze_provenance"][
                "uncertainty_abstention_threshold_selected"
            ]
        )
        self.assertFalse(
            manifest["gaze_provenance"]["prediction_values_tamper_resistant"]
        )
        private_fingerprint = stored_session["access_token_sha256"]
        exported_text = "\n".join(
            path.read_text(encoding="utf-8") for path in output.iterdir()
        )
        self.assertNotIn(private_fingerprint, exported_text)
        self.assertNotIn("PR-", exported_text)

    def test_legacy_gaze_is_excluded_while_behavior_is_retained(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-LEGACY",
            participant_id="P-LEGACY",
            pair_id="PAIR-LEGACY",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        del session["general_collection"]["gaze_measurement_contract"]
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        manifest = export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertEqual(session_row["gaze_export_status"], "behavioral_only")
        self.assertEqual(session_row["validation_gaze_export_eligible"], "False")
        self.assertEqual(session_row["reading_gaze_export_eligible"], "False")
        self.assertEqual(session_row["gaze_quality_band"], "unavailable")
        self.assertEqual(session_row["median_spatial_error_px"], "")
        self.assertIn("measurement_contract_snapshot_unavailable", excluded["exclusion_reasons"])
        self.assertEqual(len(_rows(output / "passages.csv")), 1)
        self.assertEqual(len(_rows(output / "word_reviews.csv")), 1)
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(len(_rows(output / "reading_telemetry_unverified.csv")), 10)
        self.assertEqual(
            manifest["gaze_provenance"]["session_gaze_excluded_count"],
            1,
        )

    def test_training_time_artifact_bindings_are_revalidated(self) -> None:
        cases = (
            (
                "linked",
                ("linked_data", "model_artifact_sha256"),
                "training_time_model_artifact_linkage_mismatch",
            ),
            (
                "calibration",
                ("quality", "calibration", "model_artifact_sha256"),
                "training_time_calibration_artifact_binding_mismatch",
            ),
        )
        for name, path, reason in cases:
            with self.subTest(name=name):
                session_path = _session(
                    self.root,
                    session_id=f"ST-TRAINING-FREEZE-{name.upper()}",
                    participant_id=f"P-TRAINING-FREEZE-{name.upper()}",
                    pair_id=f"PAIR-TRAINING-FREEZE-{name.upper()}",
                    visit_index=1,
                )
                session = json.loads(session_path.read_text(encoding="utf-8"))
                target = session
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = "f" * 64
                session_path.write_text(
                    json.dumps(session, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                output = self.root / f"export-{name}"
                export_bundle(self.root, output)
                excluded_rows = _rows(output / "gaze_excluded_sessions.csv")
                matching = next(
                    row
                    for row in excluded_rows
                    if row["study_session_id"]
                    == f"ST-TRAINING-FREEZE-{name.upper()}"
                )
                self.assertIn(reason, matching["validation_exclusion_reasons"])
                self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)

    def test_missing_receipt_registry_excludes_only_validation_gaze(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-NO-RECEIPTS",
            participant_id="P-NO-RECEIPTS",
            pair_id="PAIR-NO-RECEIPTS",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        del session["general_collection"]["prediction_receipts"]
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertEqual(session_row["gaze_export_status"], "behavioral_only")
        self.assertEqual(session_row["validation_gaze_export_eligible"], "False")
        self.assertIn(
            "prediction_receipt_registry_unavailable",
            excluded["validation_exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "passages.csv")), 1)
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)
        self.assertEqual(len(_rows(output / "reading_telemetry_unverified.csv")), 10)

    def test_legacy_uncertainty_is_not_evaluable_but_geometry_is_retained(self) -> None:
        _session(
            self.root,
            session_id="ST-LEGACY-UNCERTAINTY",
            participant_id="P-LEGACY-UNCERTAINTY",
            pair_id="PAIR-LEGACY-UNCERTAINTY",
            visit_index=1,
            uncertainty_mode="legacy_missing",
        )
        output = self.root / "export"
        manifest = export_bundle(self.root, output)

        session_row = _rows(output / "sessions.csv")[0]
        validation_rows = _rows(output / "validation_samples.csv")
        self.assertEqual(session_row["prediction_receipt_status"], "verified")
        self.assertEqual(session_row["validation_gaze_export_eligible"], "True")
        self.assertEqual(session_row["uncertainty_evidence_status"], "not_evaluable")
        self.assertEqual(session_row["uncertainty_evidence_eligible"], "False")
        self.assertEqual(len(validation_rows), 30)
        self.assertEqual(
            {row["uncertainty_status"] for row in validation_rows},
            {"unavailable_receipt_missing"},
        )
        self.assertEqual(
            {row["uncertainty_evidence_eligible"] for row in validation_rows},
            {"False"},
        )
        self.assertEqual(
            {row["uncertainty_coverage_risk_status"] for row in validation_rows},
            {"not_evaluable"},
        )
        self.assertEqual(
            {row["uncertainty_score"] for row in validation_rows},
            {""},
        )
        self.assertEqual(
            manifest["gaze_provenance"][
                "uncertainty_evidence_not_evaluable_session_count"
            ],
            1,
        )
        self.assertEqual(
            manifest["gaze_provenance"][
                "uncertainty_unavailable_validation_row_count"
            ],
            30,
        )

    def test_no_face_is_capture_coverage_not_missing_uncertainty_evidence(self) -> None:
        _session(
            self.root,
            session_id="ST-NO-FACE-UNCERTAINTY",
            participant_id="P-NO-FACE-UNCERTAINTY",
            pair_id="PAIR-NO-FACE-UNCERTAINTY",
            visit_index=1,
            no_face_ordinal=0,
        )
        output = self.root / "export"
        manifest = export_bundle(self.root, output)

        session_row = _rows(output / "sessions.csv")[0]
        validation_rows = _rows(output / "validation_samples.csv")
        no_face_rows = [
            row
            for row in validation_rows
            if row["uncertainty_status"] == "unavailable_sensor_failure"
        ]
        self.assertEqual(session_row["validation_gaze_export_eligible"], "True")
        self.assertEqual(
            session_row["uncertainty_evidence_status"],
            "verified_scored_no_threshold",
        )
        self.assertEqual(session_row["uncertainty_evidence_eligible"], "True")
        self.assertEqual(session_row["uncertainty_coverage_risk_evaluable"], "True")
        self.assertEqual(session_row["start_uncertainty_no_face_count"], "1")
        self.assertEqual(session_row["end_uncertainty_no_face_count"], "1")
        self.assertEqual(
            float(session_row["start_uncertainty_capture_coverage_fraction"]),
            14 / 15,
        )
        self.assertEqual(
            session_row["start_uncertainty_conditional_scored_fraction"],
            "1.0",
        )
        self.assertEqual(len(validation_rows), 30)
        self.assertEqual(len(no_face_rows), 2)
        self.assertEqual(
            {row["uncertainty_coverage_risk_status"] for row in no_face_rows},
            {"excluded_sensor_failure_reported_in_capture_coverage"},
        )
        self.assertEqual(
            manifest["gaze_provenance"][
                "uncertainty_scored_validation_row_count"
            ],
            28,
        )
        self.assertEqual(
            manifest["gaze_provenance"][
                "uncertainty_no_face_validation_row_count"
            ],
            2,
        )
        self.assertEqual(
            manifest["gaze_provenance"][
                "uncertainty_coverage_risk_evaluable_session_count"
            ],
            1,
        )

    def test_rehashed_invalid_uncertainty_observation_fails_closed(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-UNCERTAINTY-TAMPER",
            participant_id="P-UNCERTAINTY-TAMPER",
            pair_id="PAIR-UNCERTAINTY-TAMPER",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        record = _receipt_record_at(session, phase="start", ordinal=0)
        record["issued"]["prediction"]["uncertainty"]["score"] = 0.1
        start_summary = session["general_collection"]["validations"]["start"]
        start_summary["uncertainty_observations"][0]["uncertainty"][
            "score"
        ] = 0.1
        _rehash_receipt_record(session, phase="start", ordinal=0)
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertIn(
            "start_prediction_receipt_uncertainty_observation_invalid",
            excluded["validation_exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)
        self.assertEqual(len(_rows(output / "passages.csv")), 1)

    def test_rehashed_semantic_receipt_tamper_still_fails_closed(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-RECEIPT-TAMPER",
            participant_id="P-RECEIPT-TAMPER",
            pair_id="PAIR-RECEIPT-TAMPER",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        record = _receipt_record_at(session, phase="start", ordinal=0)
        record["issued"]["assessment_id"] = "GC-ATTACKER"
        _rehash_receipt_record(session, phase="start", ordinal=0)
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertIn(
            "start_prediction_receipt_assessment_id_binding_mismatch",
            excluded["validation_exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)
        self.assertEqual(len(_rows(output / "passages.csv")), 1)

    def test_rehashed_receipt_bindings_fail_closed_independently(self) -> None:
        cases = (
            (
                "study_session_id",
                ("study_session_id",),
                "ST-ATTACKER",
                "start_prediction_receipt_study_session_id_binding_mismatch",
            ),
            (
                "authorization",
                ("authorization_fingerprint_sha256",),
                "0" * 64,
                "start_prediction_receipt_authorization_fingerprint_sha256_binding_mismatch",
            ),
            (
                "model",
                ("model_name",),
                "attacker-model",
                "start_prediction_receipt_model_name_binding_mismatch",
            ),
            (
                "artifact",
                ("model_artifact_sha256",),
                "1" * 64,
                "start_prediction_receipt_model_artifact_sha256_binding_mismatch",
            ),
            (
                "capture_session",
                ("capture_session_id",),
                "GAZE-ATTACKER",
                "start_prediction_receipt_capture_session_id_binding_mismatch",
            ),
            (
                "viewport",
                ("viewport",),
                {"width_px": 1281, "height_px": 800},
                "start_prediction_receipt_viewport_binding_mismatch",
            ),
            (
                "measurement_contract",
                ("measurement_contract_sha256",),
                "2" * 64,
                "start_prediction_receipt_measurement_contract_sha256_binding_mismatch",
            ),
            (
                "target_order",
                ("receipt_ordinal",),
                1,
                "start_prediction_receipt_target_sequence_mismatch",
            ),
            (
                "target_identity",
                ("target", "target_id"),
                "attacker-target",
                "start_prediction_receipt_target_sequence_mismatch",
            ),
            (
                "outcome",
                ("prediction", "screen_xy_px"),
                [1.0, 1.0],
                "start_prediction_receipt_outcome_coordinate_transform_mismatch",
            ),
        )

        for case_index, (name, path, value, expected_reason) in enumerate(cases):
            with self.subTest(binding=name):
                with tempfile.TemporaryDirectory(
                    prefix=f"lexigaze-export-{case_index}-"
                ) as directory:
                    root = Path(directory)
                    session_path = _session(
                        root,
                        session_id=f"ST-BINDING-{case_index}",
                        participant_id=f"P-BINDING-{case_index}",
                        pair_id=f"PAIR-BINDING-{case_index}",
                        visit_index=1,
                    )
                    session = json.loads(session_path.read_text(encoding="utf-8"))
                    record = _receipt_record_at(session, phase="start", ordinal=0)
                    target = record["issued"]
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = value
                    _rehash_receipt_record(session, phase="start", ordinal=0)
                    session_path.write_text(
                        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    output = root / "export"
                    export_bundle(root, output)
                    excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
                    self.assertIn(
                        expected_reason,
                        excluded["validation_exclusion_reasons"],
                    )
                    self.assertEqual(
                        len(_rows(output / "validation_samples.csv")),
                        0,
                    )

    def test_replayed_receipt_consumption_phase_fails_closed(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-RECEIPT-REPLAY",
            participant_id="P-RECEIPT-REPLAY",
            pair_id="PAIR-RECEIPT-REPLAY",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        record = _receipt_record_at(session, phase="start", ordinal=0)
        record["consumed_validation_phase"] = "end"
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertIn(
            "start_prediction_receipt_consumption_mismatch",
            excluded["validation_exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)

    def test_receipt_resolution_change_is_warning_not_geometry_failure(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-CAPTURE-WARNING",
            participant_id="P-CAPTURE-WARNING",
            pair_id="PAIR-CAPTURE-WARNING",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        warning_capture = _capture_contract()
        warning_capture["source_width_px"] = 640
        warning_capture["source_height_px"] = 360
        calibration_capture = session["quality"]["calibration"]["capture_contract"]
        capture_check = compare_capture_contracts(
            calibration_capture,
            warning_capture,
        )
        self.assertTrue(capture_check["compatible"])
        self.assertIn("source_resolution_changed", capture_check["warnings"])

        for phase in ("start", "end"):
            for ordinal in range(15):
                record = _receipt_record_at(
                    session,
                    phase=phase,
                    ordinal=ordinal,
                )
                record["issued"]["capture_contract"] = copy.deepcopy(
                    warning_capture
                )
                _rehash_receipt_record(
                    session,
                    phase=phase,
                    ordinal=ordinal,
                )
            summary = session["general_collection"]["validations"][phase]
            summary["capture_contract"] = copy.deepcopy(warning_capture)
            summary["capture_contract_check"] = copy.deepcopy(capture_check)
            _refresh_validation_payload_sha256(summary)
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        self.assertEqual(session_row["validation_gaze_export_eligible"], "True")
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 30)
        self.assertIn(
            "source_resolution_changed",
            session_row["start_prediction_receipt_capture_warnings"],
        )

    def test_receipt_record_and_bundle_hash_tamper_fail_closed(self) -> None:
        cases = (
            (
                "record",
                "prediction_receipt_registry_record_hash_mismatch",
            ),
            (
                "bundle",
                "start_prediction_receipt_bundle_hash_mismatch",
            ),
        )
        for case_index, (kind, expected_reason) in enumerate(cases):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory(
                    prefix=f"lexigaze-export-hash-{case_index}-"
                ) as directory:
                    root = Path(directory)
                    session_path = _session(
                        root,
                        session_id=f"ST-HASH-{case_index}",
                        participant_id=f"P-HASH-{case_index}",
                        pair_id=f"PAIR-HASH-{case_index}",
                        visit_index=1,
                    )
                    session = json.loads(session_path.read_text(encoding="utf-8"))
                    if kind == "record":
                        record = _receipt_record_at(
                            session,
                            phase="start",
                            ordinal=0,
                        )
                        record["issued"]["issued_at_utc"] = (
                            "2026-08-10T23:59:59+00:00"
                        )
                    else:
                        start_summary = session["general_collection"]["validations"][
                            "start"
                        ]
                        start_summary["prediction_receipt_bundle"][
                            "bundle_sha256"
                        ] = "0" * 64
                        _refresh_validation_payload_sha256(start_summary)
                    session_path.write_text(
                        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    output = root / "export"
                    export_bundle(root, output)
                    excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
                    self.assertIn(
                        expected_reason,
                        excluded["validation_exclusion_reasons"],
                    )
                    self.assertEqual(
                        len(_rows(output / "validation_samples.csv")),
                        0,
                    )

    def test_tampered_validation_payload_hash_excludes_only_gaze(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-TAMPER",
            participant_id="P-TAMPER",
            pair_id="PAIR-TAMPER",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["general_collection"]["validations"]["start"][
            "validation_payload_sha256"
        ] = "0" * 64
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertIn(
            "start_validation_payload_hash_mismatch",
            excluded["exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "passages.csv")), 1)
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 0)
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(len(_rows(output / "reading_telemetry_unverified.csv")), 10)

    def test_completed_session_without_telemetry_excludes_gaze(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-NO-TELEMETRY",
            participant_id="P-NO-TELEMETRY",
            pair_id="PAIR-NO-TELEMETRY",
            visit_index=1,
        )
        telemetry_paths = list(
            (session_path.parent / "collection" / "telemetry").rglob("*.json")
        )
        self.assertEqual(len(telemetry_paths), 1)
        telemetry_paths[0].unlink()

        output = self.root / "export"
        export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertEqual(session_row["gaze_export_status"], "validation_only")
        self.assertEqual(session_row["validation_gaze_export_eligible"], "True")
        self.assertEqual(session_row["reading_gaze_export_eligible"], "False")
        self.assertIn(
            "completed_gaze_session_has_no_telemetry",
            excluded["exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(len(_rows(output / "reading_telemetry_unverified.csv")), 0)
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 30)
        self.assertEqual(len(_rows(output / "passages.csv")), 1)

    def test_tampered_final_quality_metrics_exclude_gaze(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-QUALITY-TAMPER",
            participant_id="P-QUALITY-TAMPER",
            pair_id="PAIR-QUALITY-TAMPER",
            visit_index=1,
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        final_quality = session["quality"]["general_collection"]
        final_quality["median_spatial_error_px"] = 999999.0
        final_quality["effective_sampling_hz"] = 999999.0
        session_path.write_text(
            json.dumps(session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertEqual(session_row["gaze_export_status"], "validation_only")
        self.assertEqual(session_row["validation_gaze_export_eligible"], "True")
        self.assertEqual(session_row["median_spatial_error_px"], "9.0")
        self.assertEqual(session_row["effective_sampling_hz"], "")
        self.assertIn(
            "stored_final_median_spatial_error_px_mismatch",
            excluded["exclusion_reasons"],
        )
        self.assertIn(
            "stored_final_effective_sampling_hz_mismatch",
            excluded["exclusion_reasons"],
        )
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 30)

    def test_client_telemetry_tamper_never_becomes_eligible_evidence(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-CLIENT-TELEMETRY-TAMPER",
            participant_id="P-CLIENT-TELEMETRY-TAMPER",
            pair_id="PAIR-CLIENT-TELEMETRY-TAMPER",
            visit_index=1,
        )
        telemetry_path = next(
            (session_path.parent / "collection" / "telemetry").rglob("*.json")
        )
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        telemetry["samples"][0]["screen_xy_px"] = [1279.0, 799.0]
        telemetry_payload = {
            key: telemetry[key]
            for key in ("batch_id", "passage_id", "viewport", "samples")
        }
        telemetry["payload_sha256"] = canonical_sha256(telemetry_payload)
        telemetry_path.write_text(
            json.dumps(telemetry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        unverified = _rows(output / "reading_telemetry_unverified.csv")
        self.assertEqual(session_row["validation_gaze_export_eligible"], "True")
        self.assertEqual(session_row["reading_gaze_export_eligible"], "False")
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 30)
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(len(unverified), 10)
        self.assertEqual(unverified[0]["screen_xy_px"], "[1279.0,799.0]")
        self.assertEqual(unverified[0]["formal_evidence_eligible"], "False")
        self.assertEqual(unverified[0]["prediction_receipt_bound"], "False")

    def test_malformed_reading_telemetry_cannot_block_validation_export(self) -> None:
        session_path = _session(
            self.root,
            session_id="ST-MALFORMED-TELEMETRY",
            participant_id="P-MALFORMED-TELEMETRY",
            pair_id="PAIR-MALFORMED-TELEMETRY",
            visit_index=1,
        )
        telemetry_path = next(
            (session_path.parent / "collection" / "telemetry").rglob("*.json")
        )
        telemetry_path.write_text("[]\n", encoding="utf-8")

        output = self.root / "export"
        export_bundle(self.root, output)
        session_row = _rows(output / "sessions.csv")[0]
        excluded = _rows(output / "gaze_excluded_sessions.csv")[0]
        self.assertEqual(session_row["validation_gaze_export_eligible"], "True")
        self.assertIn("telemetry_batch_invalid", excluded["reading_exclusion_reasons"])
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 30)
        self.assertEqual(len(_rows(output / "reading_telemetry_unverified.csv")), 0)
        self.assertEqual(len(_rows(output / "passages.csv")), 1)

    def test_pair_contract_mismatch_keeps_sessions_but_forbids_pair_comparison(self) -> None:
        first_contract = load_participant_gaze_measurement_contract()
        second_contract = copy.deepcopy(first_contract)
        second_contract["export_fixture_revision"] = "different-frozen-contract"
        _session(
            self.root,
            session_id="ST-PAIR1",
            participant_id="P-PAIR",
            pair_id="PAIR-2",
            visit_index=1,
            measurement_contract=first_contract,
        )
        _session(
            self.root,
            session_id="ST-PAIR2",
            participant_id="P-PAIR",
            pair_id="PAIR-2",
            visit_index=2,
            measurement_contract=second_contract,
        )

        output = self.root / "export"
        manifest = export_bundle(self.root, output)
        sessions = _rows(output / "sessions.csv")
        self.assertEqual(
            {row["gaze_export_status"] for row in sessions},
            {"validation_only"},
        )
        self.assertEqual(
            {row["pair_validation_gaze_comparison_status"] for row in sessions},
            {"pair_measurement_contract_mismatch"},
        )
        self.assertEqual(
            {row["pair_validation_gaze_comparable"] for row in sessions},
            {"False"},
        )
        self.assertEqual(
            {row["pair_gaze_comparison_status"] for row in sessions},
            {"reading_telemetry_unverified"},
        )
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 60)
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(len(_rows(output / "reading_telemetry_unverified.csv")), 20)
        self.assertEqual(
            manifest["gaze_provenance"]["pair_validation_comparable_count"],
            0,
        )
        self.assertEqual(
            manifest["gaze_provenance"][
                "pair_validation_comparison_status_counts"
            ],
            {"pair_measurement_contract_mismatch": 1},
        )

    def test_pair_requires_same_device_class_and_browser_family(self) -> None:
        _session(
            self.root,
            session_id="ST-DEVICE1",
            participant_id="P-DEVICE",
            pair_id="PAIR-DEVICE",
            visit_index=1,
        )
        _session(
            self.root,
            session_id="ST-DEVICE2",
            participant_id="P-DEVICE",
            pair_id="PAIR-DEVICE",
            visit_index=2,
            device_overrides={"browser_family": "firefox"},
        )

        output = self.root / "export"
        manifest = export_bundle(self.root, output)
        sessions = _rows(output / "sessions.csv")
        self.assertEqual(
            {row["pair_validation_gaze_comparison_status"] for row in sessions},
            {"pair_device_policy_mismatch"},
        )
        self.assertEqual(
            {row["pair_validation_gaze_comparable"] for row in sessions},
            {"False"},
        )
        self.assertEqual(
            {row["pair_device_policy_match"] for row in sessions},
            {"False"},
        )
        self.assertEqual(
            manifest["gaze_provenance"][
                "pair_validation_comparison_status_counts"
            ],
            {"pair_device_policy_mismatch": 1},
        )

    def test_pair_geometry_changes_are_diagnostic_not_policy_expansion(self) -> None:
        _session(
            self.root,
            session_id="ST-GEOMETRY1",
            participant_id="P-GEOMETRY",
            pair_id="PAIR-GEOMETRY",
            visit_index=1,
        )
        _session(
            self.root,
            session_id="ST-GEOMETRY2",
            participant_id="P-GEOMETRY",
            pair_id="PAIR-GEOMETRY",
            visit_index=2,
            device_overrides={"device_pixel_ratio_bucket": "1_5"},
        )

        output = self.root / "export"
        export_bundle(self.root, output)
        sessions = _rows(output / "sessions.csv")
        self.assertEqual(
            {row["pair_validation_gaze_comparison_status"] for row in sessions},
            {"comparable_same_measurement_contract"},
        )
        self.assertEqual(
            {row["pair_validation_gaze_comparable"] for row in sessions},
            {"True"},
        )
        self.assertEqual(
            {row["pair_device_policy_match"] for row in sessions},
            {"True"},
        )
        self.assertEqual(
            {row["pair_device_diagnostics_status"] for row in sessions},
            {"changed"},
        )
        for row in sessions:
            self.assertEqual(
                json.loads(row["pair_device_diagnostic_mismatches"]),
                ["device_pixel_ratio_bucket"],
            )

    def test_ineligible_sibling_does_not_exclude_valid_visit_gaze(self) -> None:
        _session(
            self.root,
            session_id="ST-SIBLING1",
            participant_id="P-SIBLING",
            pair_id="PAIR-SIBLING",
            visit_index=1,
        )
        invalid_path = _session(
            self.root,
            session_id="ST-SIBLING2",
            participant_id="P-SIBLING",
            pair_id="PAIR-SIBLING",
            visit_index=2,
        )
        invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
        del invalid["general_collection"]["gaze_measurement_contract"]
        invalid_path.write_text(
            json.dumps(invalid, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output = self.root / "export"
        manifest = export_bundle(self.root, output)
        sessions = {
            row["study_session_id"]: row for row in _rows(output / "sessions.csv")
        }
        self.assertEqual(
            sessions["ST-SIBLING1"]["gaze_export_status"],
            "validation_only",
        )
        self.assertEqual(
            sessions["ST-SIBLING2"]["gaze_export_status"],
            "behavioral_only",
        )
        self.assertEqual(
            {
                row["pair_validation_gaze_comparison_status"]
                for row in sessions.values()
            },
            {"paired_visit_gaze_ineligible"},
        )
        self.assertEqual(len(_rows(output / "validation_samples.csv")), 30)
        self.assertEqual(len(_rows(output / "gaze_telemetry.csv")), 0)
        self.assertEqual(len(_rows(output / "reading_telemetry_unverified.csv")), 20)
        self.assertEqual(
            manifest["gaze_provenance"]["validation_gaze_eligible_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
