"""Tests for the CPU-only webcam gaze measurement-ceiling audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from core.gaze_core.measurement_ceiling import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_LINE_GAP_PX,
    DEFAULT_MEDIAN_WORD_WIDTH_PX,
    DEFAULT_PREFLIGHT_PROTOCOL_PATH,
    DEFAULT_TARGET_OVERLAP_TOLERANCE_SIGNED,
    MeasurementCeilingError,
    REPEATABILITY_PROXY_COVERAGE_LEVELS,
    UNCERTAINTY_V2_COVERAGE_LEVELS,
    _fixed_uncertainty_coverage_scope,
    _nearest_rank,
    _temporal_repeatability_proxy,
    build_measurement_ceiling_result,
    deterministic_json,
    render_measurement_ceiling_markdown,
)
from core.gaze_core.uncertainty_contract import (
    canonical_sha256,
    unavailable_uncertainty,
    verified_definition,
)
from scripts.audit_webcam_gaze_measurement_ceiling import _parse_args, main


class WebcamGazeMeasurementCeilingTests(unittest.TestCase):
    def test_repeatability_proxy_uses_start_score_and_end_risk_only(self) -> None:
        start_records: list[dict[str, object]] = []
        end_records: list[dict[str, object]] = []
        for index in range(5):
            target_id = f"target-{index + 1}"
            spread = float(index + 1)
            for predicted_x in (-spread, 0.0, spread):
                start_records.append(
                    {
                        "target_id": target_id,
                        "predicted_x_px": predicted_x,
                        "predicted_y_px": 0.0,
                        # Start target error is forbidden as the proxy score.
                        "spatial_error_px": 999_999.0,
                    }
                )
            for _ in range(3):
                end_records.append(
                    {
                        "target_id": target_id,
                        "predicted_x_px": 0.0,
                        "predicted_y_px": 0.0,
                        "spatial_error_px": float((index + 1) * 10),
                    }
                )

        aligned = _temporal_repeatability_proxy(start_records, end_records)
        reversed_end = [dict(record) for record in end_records]
        for record in reversed_end:
            target_index = int(str(record["target_id"]).split("-")[-1])
            record["spatial_error_px"] = float((6 - target_index) * 10)
        reversed_risk = _temporal_repeatability_proxy(start_records, reversed_end)

        self.assertEqual(
            aligned["claim_boundary"],
            "proxy_not_predictive_uncertainty",
        )
        self.assertEqual(
            aligned["fixed_requested_coverages"],
            list(REPEATABILITY_PROXY_COVERAGE_LEVELS),
        )
        self.assertEqual(
            [
                point["requested_coverage"]
                for point in aligned["coverage_risk_curve"]
            ],
            [0.2, 0.4, 0.6, 0.8, 1.0],
        )
        self.assertEqual(
            aligned["ordered_target_ids_low_to_high_proxy"],
            reversed_risk["ordered_target_ids_low_to_high_proxy"],
        )
        self.assertEqual(
            {
                target_id: values["start_repeatability_rms_px"]
                for target_id, values in aligned["targets"].items()
            },
            {
                target_id: values["start_repeatability_rms_px"]
                for target_id, values in reversed_risk["targets"].items()
            },
        )
        changed_non_score_fields = [dict(record) for record in start_records]
        for index, record in enumerate(changed_non_score_fields):
            record["spatial_error_px"] = float(index * 123_457)
        for index, record in enumerate(reversed_end):
            record["predicted_x_px"] = float(index * 98_765)
            record["predicted_y_px"] = float(-index * 54_321)
        non_score_mutation = _temporal_repeatability_proxy(
            changed_non_score_fields,
            reversed_end,
        )
        self.assertEqual(
            reversed_risk["ordered_target_ids_low_to_high_proxy"],
            non_score_mutation["ordered_target_ids_low_to_high_proxy"],
        )
        self.assertEqual(
            reversed_risk["coverage_risk_curve"],
            non_score_mutation["coverage_risk_curve"],
        )
        self.assertAlmostEqual(aligned["association"]["value"], 1.0)
        self.assertAlmostEqual(reversed_risk["association"]["value"], -1.0)
        self.assertEqual(
            aligned["coverage_risk_curve"][0]["retained_target_ids"],
            ["target-1"],
        )
        self.assertFalse(aligned["score_uses_target_error_or_end_data"])
        self.assertFalse(aligned["threshold_selection_authorized"])
        self.assertFalse(aligned["quality_band_change_authorized"])
        self.assertFalse(aligned["per_sample_abstention_authorized"])

    def _write_fixture(
        self,
        root: Path,
        *,
        overlapping_calibration_target: bool = False,
        boundary_calibration_target: bool = False,
        conflicting_capture_source: bool = False,
        calibration_camera_width: int = 1280,
        calibration_camera_height: int = 720,
        calibration_camera_frame_rate: float = 30.0,
        participant_camera_width: int = 1280,
        participant_camera_height: int = 720,
        participant_camera_fps_band: str = "24_30",
    ) -> tuple[Path, Path, Path, Path]:
        gaze_session_id = "calibration-1"
        model_name = "fixture-model"
        validation_targets = {
            "left": (100.0, 100.0),
            "center": (500.0, 400.0),
            "right": (900.0, 100.0),
        }

        def validation_samples(
            *,
            bias_x: float,
            bias_y: float,
        ) -> list[dict[str, object]]:
            samples: list[dict[str, object]] = []
            for target_id, (target_x, target_y) in validation_targets.items():
                samples.append(
                    {
                        "target_id": target_id,
                        "target_x_px": target_x,
                        "target_y_px": target_y,
                        "predicted_x_px": target_x + bias_x,
                        "predicted_y_px": target_y + bias_y,
                        "prediction_success": True,
                        # This browser-derived assignment must never become truth.
                        "nearest_word_index": 999_999,
                    }
                )
            return samples

        participant = {
            "schema_version": 7,
            "protocol_id": "fixture-protocol",
            "protocol_version": "1",
            "mode": "rehearsal",
            "state": "completed",
            "dataset_role": "development",
            "study_session_id": "study-1",
            "linked_data": {
                "gaze_session_id": gaze_session_id,
                "model_name": model_name,
            },
            "quality": {
                "general_system_check": {
                    "device": {
                        "viewport_width": 1000,
                        "viewport_height": 800,
                        "camera_width": participant_camera_width,
                        "camera_height": participant_camera_height,
                        "estimated_camera_fps_band": participant_camera_fps_band,
                    }
                }
            },
            "general_collection": {
                "validations": {
                    "start": {
                        "samples": validation_samples(
                            bias_x=-10.0,
                            bias_y=20.0,
                        )
                    },
                    "end": {
                        "samples": validation_samples(
                            bias_x=-10.0,
                            bias_y=20.0,
                        )
                    },
                }
            },
        }
        participant_path = root / "participant-session.json"
        participant_path.write_text(
            json.dumps(participant, ensure_ascii=False),
            encoding="utf-8",
        )

        manifest_path = root / gaze_session_id / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True)
        session_metadata_path = manifest_path.parent / "session.json"
        session_metadata_path.write_text(
            json.dumps(
                {
                    "session_id": gaze_session_id,
                    "capture_run_id": "capture-1",
                    "capture_source": "study-direct-frame",
                    "study_session_id": "study-1",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        calibration_targets = [(-0.4, -0.4), (0.4, 0.4)]
        if overlapping_calibration_target:
            # Pixel target (500, 400) is normalized to (0, 0).
            calibration_targets.append((0.0, 0.0))
        if boundary_calibration_target:
            # Exactly 0.2 signed units from the center target is allowed.
            calibration_targets.append((0.2, 0.0))
        manifest_records = [
            {
                "ok": True,
                "target_x_norm": x,
                "target_y_norm": y,
                "viewport_width": 1000,
                "viewport_height": 800,
                "camera_width": calibration_camera_width,
                "camera_height": calibration_camera_height,
                "camera_frame_rate": calibration_camera_frame_rate,
                "capture_run_id": "capture-1",
                "capture_source": (
                    "direct-frame"
                    if conflicting_capture_source
                    else "study-direct-frame"
                ),
                "collection_protocol": "fixture",
                "motion_block_id": "neutral",
            }
            for x, y in calibration_targets
        ]
        manifest_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in manifest_records
            ),
            encoding="utf-8",
        )

        model = {
            "name": model_name,
            "data_session_id": gaze_session_id,
            "validation_px_error": 25.0,
            "validation_scheme": "nested_leave_one_motion_block_out",
            "candidate_comparison": {
                "selected": "gaze_polynomial",
                "baseline_gaze_only_px": 25.0,
                "motion_conditioned_px": 30.0,
            },
            "stages": [
                {
                    "stage": 1,
                    "calibrator_type": "gaze_polynomial",
                    "validation_px_error": 25.0,
                    "hyperparameter_cv_px_error": 22.0,
                }
            ],
        }
        model_path = root / "model.json"
        model_path.write_text(
            json.dumps(model, ensure_ascii=False),
            encoding="utf-8",
        )
        return participant_path, session_metadata_path, manifest_path, model_path

    def _add_v2_receipt_evidence(
        self,
        paths: tuple[Path, Path, Path, Path],
        *,
        no_face: tuple[str, int] | None = None,
        successful_unavailable: tuple[str, int] | None = None,
        definition_sha256: str | None = None,
        target_x_shift_px: float = 0.0,
        include_model_v2: bool = True,
        noncanonical_model_state: bool = False,
    ) -> None:
        participant_path, _, _, model_path = paths
        _, frozen_definition_sha256 = verified_definition()
        selected_definition_sha256 = (
            frozen_definition_sha256
            if definition_sha256 is None
            else definition_sha256
        )
        model = json.loads(model_path.read_text(encoding="utf-8"))
        score_state = {
            "schema_version": 2,
            "status": "scored_no_threshold",
            "definition_sha256": selected_definition_sha256,
            "fit_scope": "all_training_motion_blocks",
            "training_sample_count": 65,
            "threshold": None,
            "abstention_status": "not_selected",
            "component_reference": {
                "ood": [0.1, 0.2],
                "leverage": [0.1, 0.2],
                "disagreement": [0.01, 0.02],
            },
        }
        score_state["state_sha256"] = canonical_sha256(score_state)
        if noncanonical_model_state:
            score_state["component_reference"]["ood"][0] = float("nan")
        if include_model_v2:
            model["uncertainty_v2"] = {
                "schema_version": 2,
                "status": "scored_no_threshold",
                "definition_sha256": selected_definition_sha256,
                "threshold": None,
                "abstention_policy": {
                    "status": "not_selected",
                    "threshold": None,
                    "quality_band": None,
                },
                "grid_validation": {
                    "status": "complete",
                    "definition_sha256": selected_definition_sha256,
                    "sample_count": 65,
                },
                "oof_evidence": {
                    "definition_sha256": selected_definition_sha256,
                    "coverage_grid": list(UNCERTAINTY_V2_COVERAGE_LEVELS),
                    "threshold_selected": False,
                    "threshold": None,
                    "fresh_matched_contract_capture_required": True,
                },
                "final_score_state": score_state,
            }
        model_path.write_text(
            json.dumps(model, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        model_file_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

        participant = json.loads(participant_path.read_text(encoding="utf-8"))
        collection = participant["general_collection"]
        measurement_contract_path = (
            Path(__file__).resolve().parents[1]
            / "core"
            / "gaze_core"
            / "participant_gaze_measurement_contract_v1.json"
        )
        measurement_contract = json.loads(
            measurement_contract_path.read_text(encoding="utf-8")
        )
        if target_x_shift_px:
            for target in measurement_contract["target_independence"][
                "selected_validation_targets"
            ]:
                target["target_x_viewport_fraction"] += target_x_shift_px / 1000.0
                target["target_x_norm"] += target_x_shift_px / 500.0
        targets = [
            (
                target["target_id"],
                float(
                    math.floor(
                        float(target["target_x_viewport_fraction"]) * 1000.0
                        + 0.5
                    )
                ),
                float(
                    math.floor(
                        float(target["target_y_viewport_fraction"]) * 800.0
                        + 0.5
                    )
                ),
            )
            for target in measurement_contract["target_independence"][
                "selected_validation_targets"
            ]
        ]
        measurement_sha256 = canonical_sha256(measurement_contract)
        collection["gaze_measurement_contract"] = {
            "contract_id": measurement_contract["contract_id"],
            "contract_version": measurement_contract["contract_version"],
            "sha256": measurement_sha256,
            "contract": measurement_contract,
        }
        collection["assessment_viewport"] = {
            "width_px": 1000,
            "height_px": 800,
        }
        collection["model_artifact_sha256"] = model_file_sha256
        collection["assessment_id"] = "assessment-1"
        participant["access_token_sha256"] = canonical_sha256(
            {"fixture": "authorization"}
        )
        participant["linked_data"]["assessment_id"] = "assessment-1"
        participant["linked_data"]["model_artifact_sha256"] = model_file_sha256
        capture_contract = {
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
            "jpeg_quality": 0.85,
            "mirror_applied": False,
            "facing_mode": "user",
        }
        participant["quality"]["calibration"] = {
            "capture_contract": capture_contract,
            "model_artifact_sha256": model_file_sha256,
        }

        def scored_observation(score: float) -> dict[str, object]:
            covariance_norm = [[0.0004, 0.0], [0.0, 0.0004]]
            return {
                "schema_version": 1,
                "status": "scored_no_threshold",
                "definition_sha256": frozen_definition_sha256,
                "score": score,
                "components": {
                    "ood": {"value": score, "percentile": score},
                    "leverage": {"value": 0.1, "percentile": 0.1},
                    "disagreement": {"value": 0.02, "percentile": 0.2},
                },
                "jackknife_disagreement_covariance_norm": covariance_norm,
                "jackknife_disagreement_covariance_px": [
                    [100.0, 0.0],
                    [0.0, 64.0],
                ],
                "abstention": {"status": "not_selected", "threshold": None},
            }

        validations: dict[str, dict[str, object]] = {}
        registry_records: dict[str, dict[str, object]] = {}
        for phase_index, phase in enumerate(("start", "end")):
            samples: list[dict[str, object]] = []
            observations: list[dict[str, object]] = []
            failures: list[dict[str, object]] = []
            record_sha256s: list[str] = []
            for target_index, (target_id, base_x, target_y) in enumerate(targets):
                target_x = base_x
                frozen_target = measurement_contract[
                    "target_independence"
                ]["selected_validation_targets"][target_index]
                for repeat_index in range(3):
                    ordinal = target_index * 3 + repeat_index
                    prediction_success = no_face != (phase, ordinal)
                    error_x = float(5 + ordinal * 2 + phase_index * 7)
                    sample: dict[str, object] = {
                        "target_id": target_id,
                        "target_x_px": target_x,
                        "target_y_px": target_y,
                        "target_x_norm": float(frozen_target["target_x_norm"]),
                        "target_y_norm": float(frozen_target["target_y_norm"]),
                        "prediction_success": prediction_success,
                        "predicted_x_px": (
                            target_x + error_x if prediction_success else None
                        ),
                        "predicted_y_px": (
                            target_y if prediction_success else None
                        ),
                        "spatial_error_px": (
                            abs(error_x) if prediction_success else None
                        ),
                    }
                    score = 0.25 + ordinal * 0.02 + phase_index * 0.001
                    if not prediction_success:
                        uncertainty = unavailable_uncertainty(
                            "unavailable_sensor_failure",
                            "no face was detected, so no sensor observation was scored",
                        )
                    elif successful_unavailable == (phase, ordinal):
                        uncertainty = unavailable_uncertainty(
                            "unavailable_receipt_missing",
                            "runtime uncertainty evidence was unavailable",
                        )
                    else:
                        uncertainty = scored_observation(score)
                    receipt_id_sha256 = canonical_sha256(
                        {
                            "fixture_receipt": "study-1",
                            "phase": phase,
                            "ordinal": ordinal,
                        }
                    )
                    prediction = (
                        {
                            "success": True,
                            "screen_xy_px": [target_x + error_x, target_y],
                            "screen_xy_norm": [
                                (target_x + error_x) / 500.0 - 1.0,
                                target_y / 400.0 - 1.0,
                            ],
                            "http_status": 200,
                            "failure_stage": None,
                            "failure_code": None,
                            "error": None,
                            "uncertainty_schema_version": 1,
                            "uncertainty": uncertainty,
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
                            "uncertainty_schema_version": 1,
                            "uncertainty": uncertainty,
                        }
                    )
                    issued = {
                        "schema_version": 1,
                        "receipt_id_sha256": receipt_id_sha256,
                        "issued_at_utc": "2026-08-10T00:10:00+00:00",
                        "study_session_id": "study-1",
                        "authorization_fingerprint_sha256": participant[
                            "access_token_sha256"
                        ],
                        "assessment_id": "assessment-1",
                        "model_name": model["name"],
                        "model_artifact_sha256": model_file_sha256,
                        "capture_session_id": "calibration-1",
                        "phase": phase,
                        "receipt_ordinal": ordinal,
                        "target_repeat_index": repeat_index,
                        "target": {
                            "target_id": target_id,
                            "target_x_viewport_fraction": float(
                                frozen_target["target_x_viewport_fraction"]
                            ),
                            "target_y_viewport_fraction": float(
                                frozen_target["target_y_viewport_fraction"]
                            ),
                            "target_x_norm": float(frozen_target["target_x_norm"]),
                            "target_y_norm": float(frozen_target["target_y_norm"]),
                            "target_x_px": target_x,
                            "target_y_px": target_y,
                        },
                        "viewport": {"width_px": 1000, "height_px": 800},
                        "measurement_contract_sha256": measurement_sha256,
                        "capture_contract": capture_contract,
                        "capture_contract_check": {
                            "status": "compatible",
                            "compatible": True,
                            "reasons": [],
                            "warnings": [],
                        },
                        "prediction": prediction,
                    }
                    record_sha256 = canonical_sha256(issued)
                    registry_records[receipt_id_sha256] = {
                        "issued": issued,
                        "issued_record_sha256": record_sha256,
                        "consumed_at_utc": "2026-08-10T00:20:00+00:00",
                        "consumed_validation_phase": phase,
                    }
                    record_sha256s.append(record_sha256)
                    samples.append(sample)
                    if not prediction_success:
                        failures.append(
                            {
                                "receipt_record_sha256": record_sha256,
                                "failure_stage": "attributable_sensor_failure",
                                "failure_code": "no_face_detected",
                                "http_status": 400,
                            }
                        )
                    observations.append(
                        {
                            "schema_version": 1,
                            "receipt_record_sha256": record_sha256,
                            "phase": phase,
                            "receipt_ordinal": ordinal,
                            "target_id": target_id,
                            "target_repeat_index": repeat_index,
                            "prediction_success": prediction_success,
                            "uncertainty": uncertainty,
                        }
                    )
            bundle_core = {
                "schema_version": 1,
                "status": "verified",
                "phase": phase,
                "count": 15,
                "receipt_record_sha256s": record_sha256s,
            }
            bundle = {
                **bundle_core,
                "bundle_sha256": canonical_sha256(bundle_core),
            }
            observation_sha256s = [
                canonical_sha256(observation) for observation in observations
            ]
            uncertainty_summary = {
                "schema_version": 1,
                "status": "verified",
                "count": 15,
                "scored_count": sum(
                    observation["uncertainty"]["status"]
                    == "scored_no_threshold"
                    for observation in observations
                ),
                "unavailable_count": sum(
                    observation["uncertainty"]["status"]
                    != "scored_no_threshold"
                    for observation in observations
                ),
                "observation_sha256s": observation_sha256s,
                "observations_sha256": canonical_sha256(observations),
            }
            summary: dict[str, object] = {
                "prediction_receipt_status": "verified",
                "prediction_receipts_verified": True,
                "samples": samples,
                "samples_sha256": canonical_sha256(samples),
                "capture_contract": capture_contract,
                "capture_contract_check": {
                    "status": "compatible",
                    "compatible": True,
                    "reasons": [],
                    "warnings": [],
                },
                "prediction_failures": failures,
                "prediction_receipt_bundle": bundle,
                "uncertainty_observations": observations,
                "uncertainty_summary": uncertainty_summary,
                "model_artifact_sha256": model_file_sha256,
                "gaze_measurement_contract_sha256": measurement_sha256,
                "gaze_measurement_contract": collection[
                    "gaze_measurement_contract"
                ],
                "assessment_viewport": {"width_px": 1000, "height_px": 800},
            }
            summary["validation_payload_sha256"] = canonical_sha256(
                {
                    "samples": samples,
                    "capture_contract": capture_contract,
                    "prediction_receipt_bundle": bundle,
                    "uncertainty_observations": observations,
                    "uncertainty_summary": uncertainty_summary,
                    "prediction_receipt_status": "verified",
                    "prediction_receipts_verified": True,
                    "model_artifact_sha256": model_file_sha256,
                    "gaze_measurement_contract_sha256": measurement_sha256,
                    "assessment_viewport": {
                        "width_px": 1000,
                        "height_px": 800,
                    },
                }
            )
            validations[phase] = summary
        collection["validations"] = validations
        collection["prediction_receipts"] = {
            "schema_version": 1,
            "records": registry_records,
        }
        participant_path.write_text(
            json.dumps(participant, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _rehash_v2_validation(validation: dict[str, object]) -> None:
        observations = validation["uncertainty_observations"]
        summary = validation["uncertainty_summary"]
        summary["observation_sha256s"] = [
            canonical_sha256(observation) for observation in observations
        ]
        summary["observations_sha256"] = canonical_sha256(observations)
        validation["validation_payload_sha256"] = canonical_sha256(
            {
                "samples": validation["samples"],
                "capture_contract": validation["capture_contract"],
                "prediction_receipt_bundle": validation[
                    "prediction_receipt_bundle"
                ],
                "uncertainty_observations": observations,
                "uncertainty_summary": summary,
                "prediction_receipt_status": validation[
                    "prediction_receipt_status"
                ],
                "prediction_receipts_verified": validation[
                    "prediction_receipts_verified"
                ],
                "model_artifact_sha256": validation["model_artifact_sha256"],
                "gaze_measurement_contract_sha256": validation[
                    "gaze_measurement_contract_sha256"
                ],
                "assessment_viewport": validation["assessment_viewport"],
            }
        )

    def _build(
        self,
        participant_path: Path,
        session_metadata_path: Path,
        manifest_path: Path,
        model_path: Path,
    ) -> dict:
        return build_measurement_ceiling_result(
            participant_session_path=participant_path,
            calibration_session_metadata_path=session_metadata_path,
            calibration_manifest_path=manifest_path,
            model_artifact_path=model_path,
            line_gap_px=DEFAULT_LINE_GAP_PX,
            median_word_width_px=DEFAULT_MEDIAN_WORD_WIDTH_PX,
            bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        )

    def test_build_is_deterministic_and_respects_truth_boundaries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-") as name:
            paths = self._write_fixture(Path(name))
            first = self._build(*paths)
            second = self._build(*paths)

        self.assertEqual(deterministic_json(first), deterministic_json(second))
        self.assertEqual(first["provenance"]["bindings"]["status"], "passed")
        self.assertEqual(
            first["provenance"]["capture_contract"]["status"],
            "passed",
        )
        self.assertEqual(
            first["provenance"]["calibration_viewport_contract"]["status"],
            "passed",
        )
        self.assertEqual(
            first["provenance"]["cross_phase_camera_geometry"]["status"],
            "passed",
        )
        self.assertEqual(first["status"], "diagnostic_only_unverified")
        self.assertEqual(
            first["measurement_status"]["geometry"],
            "diagnostic_only_unverified",
        )
        self.assertEqual(
            first["fixed_target_receipt_integrity"]["status"],
            "diagnostic_only_unverified",
        )
        self.assertEqual(first["decision"]["eligible_claim"], "none")
        self.assertEqual(first["target_independence"]["status"], "passed")
        self.assertEqual(first["target_independence"]["overlap_count"], 0)
        self.assertEqual(
            first["target_independence"]["signed_normalized_tolerance"],
            DEFAULT_TARGET_OVERLAP_TOLERANCE_SIGNED,
        )
        self.assertEqual(
            first["target_independence"][
                "viewport_fraction_tolerance_equivalent"
            ],
            0.1,
        )
        self.assertEqual(
            first["provenance"]["model"]["validation_metric_consistency"][
                "status"
            ],
            "passed",
        )
        self.assertEqual(
            first["provenance"]["model"]["hyperparameter_cv_px_error"],
            22.0,
        )
        self.assertTrue(
            all(
                value is not False
                for value in first["provenance"]["model"][
                    "validation_metric_consistency"
                ]["checks"].values()
            )
        )
        self.assertEqual(
            first["raw_validation"]["start"]["coarse_region"]["accuracy"],
            1.0,
        )
        self.assertAlmostEqual(
            first["raw_validation"]["start"]["x_error_px"]["signed_median"],
            -10.0,
        )
        self.assertAlmostEqual(
            first["raw_validation"]["start"]["y_error_px"]["absolute_median"],
            20.0,
        )
        self.assertAlmostEqual(
            first["raw_validation"]["start"]["target_macro"][
                "mean_bias_magnitude_px"
            ],
            (10.0**2 + 20.0**2) ** 0.5,
        )
        self.assertAlmostEqual(
            first["layout_normalized_resolution"]["start"][
                "median_error_in_line_gaps"
            ],
            (10.0**2 + 20.0**2) ** 0.5 / DEFAULT_LINE_GAP_PX,
        )
        self.assertAlmostEqual(
            first["temporal_correction"]["translation_x_px"],
            10.0,
        )
        self.assertAlmostEqual(
            first["temporal_correction"]["translation_y_px"],
            -20.0,
        )
        self.assertAlmostEqual(
            first["temporal_correction"]["corrected_end"][
                "median_spatial_error_px"
            ],
            0.0,
        )
        self.assertFalse(
            first["analysis_contract"][
                "natural_reading_nearest_word_index_used_as_ground_truth"
            ]
        )
        self.assertNotIn('"nearest_word_index":', deterministic_json(first))
        bootstrap = first["temporal_correction"]["bootstrap"]
        self.assertTrue(bootstrap["paired_raw_and_corrected"])
        self.assertEqual(bootstrap["cluster_unit"], "evaluation_target_id")
        self.assertIn("sha256", bootstrap["deterministic_sampler"])
        self.assertFalse(bootstrap["inferential_claim_authorized"])
        for metric in first["not_evaluable"].values():
            self.assertEqual(metric["status"], "not_evaluable")
        requirements = first["future_uncertainty_v2_data_requirements"]
        self.assertTrue(
            requirements["protocol_requirement"][
                "freeze_before_new_untouched_capture"
            ]
        )
        self.assertIn(
            "oof_residual_x_px",
            requirements["required_per_oof_sample_fields"],
        )
        self.assertIn(
            "uncertainty_definition_sha256",
            requirements["required_definition_binding"],
        )
        self.assertFalse(
            requirements["current_evidence_inventory"][
                "predictive_uncertainty_curve_constructable_from_current_artifacts"
            ]
        )

        markdown = render_measurement_ceiling_markdown(first)
        self.assertIn("does not imply line- or word-level resolution", markdown)
        self.assertIn("no independent word-level", markdown)
        self.assertIn("confusion matrices", markdown)
        self.assertIn("Signed mean px", markdown)
        self.assertIn("0.20", markdown)
        self.assertIn("viewport-fraction coordinates", markdown)
        self.assertIn("proxy_not_predictive_uncertainty", markdown)
        self.assertIn("20/40/60/80/100%", markdown)
        self.assertIn("oof_residual_x_px", markdown)
        self.assertIn("uncertainty_definition_sha256", markdown)
        self.assertIn("new untouched capture", markdown)
        self.assertIn("Legacy receipt boundary", markdown)
        self.assertIn("does **not** claim", markdown)

    def test_fresh_v2_receipts_build_fixed_cluster_coverage_risk(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-v2-") as name:
            paths = self._write_fixture(Path(name))
            self._add_v2_receipt_evidence(paths)
            first = self._build(*paths)
            second = self._build(*paths)

        self.assertEqual(deterministic_json(first), deterministic_json(second))
        self.assertEqual(first["status"], "completed")
        self.assertEqual(
            first["measurement_status"]["geometry"],
            "completed_receipt_verified",
        )
        self.assertEqual(
            first["fixed_target_receipt_integrity"]["status"], "verified"
        )
        self.assertEqual(
            first["fixed_target_receipt_integrity"]["record_count"], 30
        )
        self.assertEqual(
            first["decision"]["eligible_claim"],
            "coarse fixed-target development evidence only",
        )
        self.assertEqual(first["analysis_config"]["bootstrap_resamples"], 20_000)
        self.assertEqual(
            first["analysis_protocol"]["protocol_id"],
            "participant-gaze-integrity-preflight-v1",
        )
        self.assertFalse(
            first["analysis_protocol"][
                "full_193_sample_measurement_ceiling_protocol_executed"
            ]
        )
        self.assertRegex(
            first["freshness"]["analysis_source"]["sha256"],
            r"^[0-9a-f]{64}$",
        )
        evidence = first["heldout_uncertainty_coverage_risk"]
        self.assertEqual(evidence["status"], "evaluable_descriptive_heldout")
        self.assertEqual(evidence["integrity_status"], "passed")
        self.assertEqual(
            evidence["fixed_requested_coverages"],
            list(UNCERTAINTY_V2_COVERAGE_LEVELS),
        )
        self.assertFalse(evidence["threshold_selected"])
        self.assertIsNone(evidence["threshold"])
        self.assertFalse(evidence["quality_band_change_authorized"])
        self.assertFalse(evidence["abstention_policy_authorized"])
        for phase in ("start", "end"):
            phase_result = evidence["phases"][phase]
            self.assertEqual(
                phase_result["status"], "evaluable_descriptive_heldout"
            )
            self.assertEqual(phase_result["attempted_capture_count"], 15)
            self.assertEqual(phase_result["successful_prediction_count"], 15)
            self.assertEqual(phase_result["independent_target_cluster_count"], 5)
            self.assertFalse(phase_result["sample_rows_are_independent_units"])
            self.assertEqual(
                [
                    point["requested_score_coverage"]
                    for point in phase_result["coverage_risk_curve"]
                ],
                list(UNCERTAINTY_V2_COVERAGE_LEVELS),
            )
            full, _, _, _, lowest = phase_result["coverage_risk_curve"]
            self.assertEqual(full["target_clusters_with_zero_coverage"], [])
            self.assertIsNotNone(
                full["target_cluster_macro_all_clusters"][
                    "mean_spatial_error_px"
                ]
            )
            self.assertEqual(
                set(lowest["target_clusters_with_zero_coverage"]),
                {
                    "heldout_top_right",
                    "heldout_center_upper_left",
                    "heldout_bottom_left",
                    "heldout_bottom_right",
                },
            )
            self.assertIsNone(
                lowest["target_cluster_macro_all_clusters"][
                    "mean_spatial_error_px"
                ]
            )
            self.assertIsNone(
                lowest["worst_target_cluster_mean_spatial_error_px"]
            )
            receipt_integrity = phase_result["receipt_integrity"]
            self.assertRegex(
                receipt_integrity["uncertainty_observations_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertRegex(
                receipt_integrity["validation_payload_sha256"],
                r"^[0-9a-f]{64}$",
            )
        combined = evidence["combined"]
        self.assertEqual(combined["attempted_capture_count"], 30)
        self.assertEqual(combined["successful_prediction_count"], 30)
        self.assertEqual(combined["independent_target_cluster_count"], 5)
        self.assertFalse(combined["sample_rows_are_independent_units"])
        self.assertNotIn(
            "per_sample_uncertainty_calibration",
            first["not_evaluable"],
        )
        requirements = first["future_uncertainty_v2_data_requirements"]
        self.assertTrue(
            requirements["current_evidence_inventory"][
                "predictive_uncertainty_curve_constructable_from_current_artifacts"
            ]
        )
        markdown = render_measurement_ceiling_markdown(first)
        self.assertIn("100/80/60/40/20%", markdown)
        self.assertIn("Only five target clusters", markdown)
        self.assertIn("not independent units", markdown)

    def test_valid_receipts_complete_geometry_when_model_v2_is_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-no-v2-") as name:
            paths = self._write_fixture(Path(name))
            self._add_v2_receipt_evidence(paths, include_model_v2=False)
            result = self._build(*paths)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            result["fixed_target_receipt_integrity"]["integrity_status"],
            "passed",
        )
        uncertainty = result["heldout_uncertainty_coverage_risk"]
        self.assertEqual(
            uncertainty["status"], "not_evaluable_model_unavailable"
        )
        self.assertEqual(uncertainty["receipt_integrity_status"], "passed")
        self.assertEqual(uncertainty["integrity_status"], "not_applicable")
        self.assertEqual(
            result["decision"]["eligible_claim"],
            "coarse fixed-target development evidence only",
        )

    def test_rehashed_registry_target_repeat_tamper_fails_integrity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-registry-") as name:
            paths = self._write_fixture(Path(name))
            self._add_v2_receipt_evidence(paths)
            participant_path = paths[0]
            participant = json.loads(participant_path.read_text(encoding="utf-8"))
            collection = participant["general_collection"]
            records = collection["prediction_receipts"]["records"]
            record = next(
                value
                for value in records.values()
                if value["issued"]["phase"] == "start"
                and value["issued"]["receipt_ordinal"] == 0
            )
            old_sha256 = record["issued_record_sha256"]
            record["issued"]["target_repeat_index"] = 1
            new_sha256 = canonical_sha256(record["issued"])
            record["issued_record_sha256"] = new_sha256
            start = collection["validations"]["start"]
            bundle = start["prediction_receipt_bundle"]
            bundle["receipt_record_sha256s"][0] = new_sha256
            bundle_core = {
                "schema_version": 1,
                "status": "verified",
                "phase": "start",
                "count": 15,
                "receipt_record_sha256s": bundle["receipt_record_sha256s"],
            }
            bundle["bundle_sha256"] = canonical_sha256(bundle_core)
            self.assertEqual(
                start["uncertainty_observations"][0]["receipt_record_sha256"],
                old_sha256,
            )
            start["uncertainty_observations"][0][
                "receipt_record_sha256"
            ] = new_sha256
            self._rehash_v2_validation(start)
            participant_path.write_text(
                json.dumps(participant, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            result = self._build(*paths)

        self.assertEqual(result["status"], "failed_integrity_gate")
        self.assertEqual(
            result["fixed_target_receipt_integrity"]["status"],
            "failed_integrity",
        )
        self.assertIn(
            "frozen target/ordinal/repeat mismatch",
            result["fixed_target_receipt_integrity"]["reason"],
        )
        self.assertEqual(result["decision"]["eligible_claim"], "none")

    def test_receipt_model_hash_must_match_all_session_layers(self) -> None:
        mutations = (
            ("linked_data", "linked model artifact hash"),
            ("calibration_quality", "calibration quality model artifact hash"),
        )
        for mutation, reason in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix=f"lexigaze-ceiling-model-binding-{mutation}-"
            ) as name:
                paths = self._write_fixture(Path(name))
                self._add_v2_receipt_evidence(paths)
                participant_path = paths[0]
                participant = json.loads(
                    participant_path.read_text(encoding="utf-8")
                )
                if mutation == "linked_data":
                    participant["linked_data"]["model_artifact_sha256"] = "0" * 64
                else:
                    participant["quality"]["calibration"][
                        "model_artifact_sha256"
                    ] = "0" * 64
                participant_path.write_text(
                    json.dumps(participant, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                result = self._build(*paths)

            self.assertEqual(result["status"], "failed_integrity_gate")
            self.assertIn(
                reason,
                result["fixed_target_receipt_integrity"]["reason"],
            )
            self.assertEqual(result["decision"]["eligible_claim"], "none")

    def test_analysis_arguments_must_match_frozen_preflight(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-config-") as name:
            paths = self._write_fixture(Path(name))
            with self.assertRaisesRegex(
                MeasurementCeilingError,
                "do not match the frozen preflight config",
            ):
                build_measurement_ceiling_result(
                    participant_session_path=paths[0],
                    calibration_session_metadata_path=paths[1],
                    calibration_manifest_path=paths[2],
                    model_artifact_path=paths[3],
                    line_gap_px=DEFAULT_LINE_GAP_PX + 0.1,
                    median_word_width_px=DEFAULT_MEDIAN_WORD_WIDTH_PX,
                    bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
                    analysis_protocol_path=DEFAULT_PREFLIGHT_PROTOCOL_PATH,
                )

    def test_no_face_is_capture_coverage_not_missing_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-no-face-") as name:
            paths = self._write_fixture(Path(name))
            self._add_v2_receipt_evidence(paths, no_face=("start", 0))
            result = self._build(*paths)

        evidence = result["heldout_uncertainty_coverage_risk"]
        self.assertEqual(evidence["status"], "evaluable_descriptive_heldout")
        start = evidence["phases"]["start"]
        self.assertEqual(start["attempted_capture_count"], 15)
        self.assertEqual(start["successful_prediction_count"], 14)
        self.assertEqual(start["no_face_count"], 1)
        self.assertAlmostEqual(start["capture_success_fraction"], 14 / 15)
        self.assertEqual(evidence["combined"]["no_face_count"], 1)
        failed_row = next(
            row for row in start["rows"] if row["prediction_success"] is False
        )
        self.assertTrue(
            all(
                value is None
                for value in failed_row["would_abstain_at_fixed_coverage"].values()
            )
        )

    def test_successful_unavailable_uncertainty_fails_closed_without_band_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-unavailable-") as name:
            paths = self._write_fixture(Path(name))
            self._add_v2_receipt_evidence(
                paths,
                successful_unavailable=("start", 0),
            )
            result = self._build(*paths)

        evidence = result["heldout_uncertainty_coverage_risk"]
        self.assertEqual(
            evidence["status"], "not_evaluable_uncertainty_unavailable"
        )
        self.assertEqual(evidence["integrity_status"], "passed")
        self.assertEqual(
            evidence["phases"]["start"]["status"],
            "not_evaluable_successful_uncertainty_unavailable",
        )
        self.assertEqual(
            evidence["phases"]["start"][
                "successful_uncertainty_unavailable_count"
            ],
            1,
        )
        self.assertFalse(evidence["threshold_selected"])
        self.assertFalse(evidence["quality_band_change_authorized"])
        self.assertEqual(result["status"], "completed")
        self.assertIn(
            "per_sample_uncertainty_calibration",
            result["not_evaluable"],
        )

    def test_uncertainty_receipt_hash_and_payload_tampering_fail_integrity(
        self,
    ) -> None:
        mutations = ("observation", "list_hash", "payload")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix=f"lexigaze-ceiling-tamper-{mutation}-"
            ) as name:
                paths = self._write_fixture(Path(name))
                self._add_v2_receipt_evidence(paths)
                participant_path = paths[0]
                participant = json.loads(
                    participant_path.read_text(encoding="utf-8")
                )
                start = participant["general_collection"]["validations"]["start"]
                if mutation == "observation":
                    start["uncertainty_observations"][0]["uncertainty"][
                        "score"
                    ] = 0.99
                elif mutation == "list_hash":
                    start["uncertainty_summary"]["observations_sha256"] = "0" * 64
                else:
                    start["validation_payload_sha256"] = "0" * 64
                participant_path.write_text(
                    json.dumps(participant, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                result = self._build(*paths)

            evidence = result["heldout_uncertainty_coverage_risk"]
            self.assertEqual(
                evidence["status"], "not_evaluable_integrity_failure"
            )
            self.assertEqual(evidence["integrity_status"], "failed")
            self.assertEqual(result["status"], "failed_integrity_gate")

    def test_rehashed_boolean_receipt_ordinals_fail_strict_integer_schema(
        self,
    ) -> None:
        mutations = (
            (0, "schema_version"),
            (1, "receipt_ordinal"),
            (1, "target_repeat_index"),
        )
        for ordinal, field in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                prefix=f"lexigaze-ceiling-bool-{field}-"
            ) as name:
                paths = self._write_fixture(Path(name))
                self._add_v2_receipt_evidence(paths)
                participant_path = paths[0]
                participant = json.loads(
                    participant_path.read_text(encoding="utf-8")
                )
                start = participant["general_collection"]["validations"]["start"]
                start["uncertainty_observations"][ordinal][field] = True
                self._rehash_v2_validation(start)
                participant_path.write_text(
                    json.dumps(participant, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                result = self._build(*paths)

            evidence = result["heldout_uncertainty_coverage_risk"]
            self.assertEqual(
                evidence["status"], "not_evaluable_integrity_failure"
            )
            self.assertEqual(evidence["integrity_status"], "failed")
            self.assertEqual(result["status"], "failed_integrity_gate")

    def test_model_definition_mismatch_fails_before_receipt_risk(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-def-") as name:
            paths = self._write_fixture(Path(name))
            self._add_v2_receipt_evidence(
                paths,
                definition_sha256="0" * 64,
            )
            result = self._build(*paths)

        evidence = result["heldout_uncertainty_coverage_risk"]
        self.assertEqual(evidence["status"], "not_evaluable_integrity_failure")
        self.assertEqual(evidence["integrity_status"], "failed")
        self.assertFalse(
            evidence["model_binding"]["checks"][
                "bundle_definition_matches_frozen"
            ]
        )
        self.assertEqual(result["status"], "failed_integrity_gate")

    def test_noncanonical_model_score_state_fails_closed_without_cli_crash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-state-") as name:
            paths = self._write_fixture(Path(name))
            self._add_v2_receipt_evidence(
                paths,
                noncanonical_model_state=True,
            )
            result = self._build(*paths)

        evidence = result["heldout_uncertainty_coverage_risk"]
        self.assertEqual(evidence["status"], "not_evaluable_integrity_failure")
        self.assertEqual(evidence["integrity_status"], "failed")
        self.assertIn("not canonical JSON", evidence["reason"])
        self.assertEqual(result["status"], "failed_integrity_gate")

    def test_frozen_heldout_target_mutation_fails_preflight_integrity(self) -> None:
        results: list[dict] = []
        for shift in (0.0, 30.0):
            with tempfile.TemporaryDirectory(
                prefix="lexigaze-ceiling-target-mutation-"
            ) as name:
                paths = self._write_fixture(Path(name))
                self._add_v2_receipt_evidence(
                    paths,
                    target_x_shift_px=shift,
                )
                results.append(self._build(*paths))

        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(results[1]["status"], "failed_integrity_gate")
        self.assertEqual(
            results[1]["fixed_target_receipt_integrity"]["status"],
            "failed_integrity",
        )
        self.assertIn(
            "not the frozen preflight contract",
            results[1]["fixed_target_receipt_integrity"]["reason"],
        )
        self.assertEqual(results[1]["decision"]["eligible_claim"], "none")

    def test_coverage_helper_never_uses_heldout_error_for_ordering(self) -> None:
        rows: list[dict[str, object]] = []
        target_order = [f"target-{index}" for index in range(5)]
        for ordinal in range(15):
            rows.append(
                {
                    "sample_id": f"sample-{ordinal:02d}",
                    "phase": "start",
                    "receipt_ordinal": ordinal,
                    "receipt_record_sha256": f"{ordinal:064x}",
                    "target_id": target_order[ordinal // 3],
                    "prediction_success": True,
                    "uncertainty_score": ordinal / 20.0,
                    "spatial_error_px": float(ordinal + 1),
                }
            )
        baseline = _fixed_uncertainty_coverage_scope(
            rows,
            scope="start",
            target_order=target_order,
        )
        mutated_rows = [dict(row) for row in rows]
        for row in mutated_rows:
            row["spatial_error_px"] = 10_000.0 - float(
                row["spatial_error_px"]
            )
        mutated = _fixed_uncertainty_coverage_scope(
            mutated_rows,
            scope="start",
            target_order=target_order,
        )
        self.assertEqual(
            baseline["ordered_sample_ids_low_to_high_training_only_score"],
            mutated["ordered_sample_ids_low_to_high_training_only_score"],
        )
        self.assertNotEqual(
            baseline["coverage_risk_curve"], mutated["coverage_risk_curve"]
        )

    def test_p90_uses_the_collection_nearest_rank_contract(self) -> None:
        self.assertEqual(_nearest_rank(list(range(1, 16)), 0.90), 14.0)

    def test_target_overlap_is_reported_as_failed_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-overlap-") as name:
            paths = self._write_fixture(
                Path(name),
                overlapping_calibration_target=True,
            )
            result = self._build(*paths)

        self.assertEqual(result["target_independence"]["status"], "failed")
        self.assertEqual(result["target_independence"]["overlap_count"], 1)
        self.assertEqual(
            result["target_independence"]["overlapping_evaluation_target_ids"],
            ["center"],
        )
        self.assertEqual(
            result["fixed_target_receipt_integrity"]["status"],
            "diagnostic_only_unverified",
        )
        self.assertEqual(result["status"], "failed_integrity_gate")
        self.assertEqual(result["decision"]["eligible_claim"], "none")

    def test_target_at_frozen_separation_boundary_is_independent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-boundary-") as name:
            paths = self._write_fixture(
                Path(name),
                boundary_calibration_target=True,
            )
            result = self._build(*paths)

        independence = result["target_independence"]
        self.assertEqual(independence["status"], "passed")
        self.assertAlmostEqual(
            independence["evaluation_target_minimum_distances"]["center"][
                "signed_normalized_euclidean"
            ],
            0.2,
        )

    def test_out_of_range_calibration_coordinate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-range-") as name:
            paths = self._write_fixture(Path(name))
            manifest_path = paths[2]
            records = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["target_x_norm"] = 1.01
            manifest_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MeasurementCeilingError, "within"):
                self._build(*paths)

    def test_model_outer_inner_metric_mismatch_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-metric-") as name:
            paths = self._write_fixture(Path(name))
            model_path = paths[3]
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["validation_px_error"] = 22.0
            model_path.write_text(json.dumps(model), encoding="utf-8")
            result = self._build(*paths)

        consistency = result["provenance"]["model"][
            "validation_metric_consistency"
        ]
        self.assertEqual(consistency["status"], "failed")
        self.assertFalse(
            consistency["checks"]["top_level_matches_selected_outer_or_stage"]
        )
        self.assertTrue(
            consistency["checks"]["selected_stage_matches_selected_outer"]
        )
        self.assertEqual(result["status"], "failed_integrity_gate")
        self.assertEqual(
            result["measurement_status"]["geometry_provenance"],
            "failed_integrity_gate",
        )
        self.assertEqual(result["decision"]["eligible_claim"], "none")

    def test_capture_source_conflict_is_a_hard_integrity_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-source-") as name:
            paths = self._write_fixture(
                Path(name),
                conflicting_capture_source=True,
            )
            result = self._build(*paths)

        capture = result["provenance"]["capture_contract"]
        self.assertEqual(capture["status"], "failed")
        self.assertFalse(
            capture["checks"][
                "manifest_capture_source_matches_session_metadata"
            ]
        )
        self.assertEqual(result["status"], "failed_integrity_gate")
        self.assertEqual(
            result["measurement_status"]["geometry_provenance"],
            "failed_integrity_gate",
        )
        self.assertFalse(result["decision"]["integrity_gate_passed"])

    def test_camera_aspect_mismatch_is_a_hard_integrity_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-camera-") as name:
            paths = self._write_fixture(
                Path(name),
                participant_camera_width=640,
                participant_camera_height=480,
                participant_camera_fps_band="15_23",
            )
            result = self._build(*paths)

        geometry = result["provenance"]["cross_phase_camera_geometry"]
        self.assertEqual(geometry["status"], "failed")
        self.assertFalse(
            geometry["checks"]["aspect_ratio_matches_within_tolerance"]
        )
        self.assertAlmostEqual(
            geometry["maximum_observed_absolute_aspect_ratio_difference"],
            16 / 9 - 4 / 3,
        )
        self.assertIn(
            "cross_phase_camera_aspect_ratio_mismatch",
            geometry["hard_failures"],
        )
        self.assertIn(
            "absolute_camera_resolution_changed_diagnostic_only",
            geometry["warnings"],
        )
        self.assertIn(
            "calibration_frame_rate_outside_participant_estimated_band_diagnostic_only",
            geometry["warnings"],
        )
        self.assertEqual(result["status"], "failed_integrity_gate")
        self.assertEqual(
            result["measurement_status"]["geometry_provenance"],
            "failed_integrity_gate",
        )

    def test_resolution_and_frame_rate_warnings_do_not_fail_matching_aspect(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-camera-") as name:
            paths = self._write_fixture(
                Path(name),
                participant_camera_width=640,
                participant_camera_height=360,
                participant_camera_fps_band="15_23",
            )
            result = self._build(*paths)

        geometry = result["provenance"]["cross_phase_camera_geometry"]
        self.assertEqual(geometry["status"], "passed")
        self.assertTrue(
            geometry["checks"]["aspect_ratio_matches_within_tolerance"]
        )
        self.assertFalse(
            geometry["checks"]["exact_absolute_resolution_matches_diagnostic"]
        )
        self.assertFalse(
            geometry["checks"][
                "frame_rate_matches_participant_estimated_band_diagnostic"
            ]
        )
        self.assertEqual(result["status"], "diagnostic_only_unverified")
        markdown = render_measurement_ceiling_markdown(result)
        self.assertIn("diagnostic warnings only", markdown)

    def test_cli_writes_stable_json_and_markdown_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-ceiling-cli-") as name:
            root = Path(name)
            (
                participant_path,
                session_metadata_path,
                manifest_path,
                model_path,
            ) = self._write_fixture(root)
            json_output = root / "result.json"
            markdown_output = root / "report.md"
            args = [
                "--participant-session",
                str(participant_path),
                "--calibration-session-metadata",
                str(session_metadata_path),
                "--calibration-manifest",
                str(manifest_path),
                "--model-artifact",
                str(model_path),
                "--line-gap-px",
                str(DEFAULT_LINE_GAP_PX),
                "--median-word-width-px",
                str(DEFAULT_MEDIAN_WORD_WIDTH_PX),
                "--bootstrap-resamples",
                str(DEFAULT_BOOTSTRAP_RESAMPLES),
                "--bootstrap-seed",
                str(DEFAULT_BOOTSTRAP_SEED),
                "--json-output",
                str(json_output),
                "--markdown-output",
                str(markdown_output),
            ]

            self.assertEqual(main(args), 0)
            first_json = json_output.read_bytes()
            first_markdown = markdown_output.read_bytes()
            self.assertEqual(main(args), 0)

            self.assertEqual(first_json, json_output.read_bytes())
            self.assertEqual(first_markdown, markdown_output.read_bytes())
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "-1")
            self.assertEqual(
                json.loads(first_json)["analysis_id"],
                "webcam-gaze-measurement-ceiling-v1",
            )

    def test_cli_help_documents_the_frozen_coordinate_conversion(self) -> None:
        stream = StringIO()
        with redirect_stdout(stream), self.assertRaises(SystemExit) as raised:
            _parse_args(["--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = " ".join(stream.getvalue().split())
        self.assertIn("signed normalized [-1, 1]", help_text)
        self.assertIn("0.2 equals 0.1", help_text)
        self.assertIn("not the separate 193-sample", help_text)


if __name__ == "__main__":
    unittest.main()
