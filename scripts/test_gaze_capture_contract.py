"""Focused tests for webcam capture and pre-reading geometry contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.gaze_core.capture_contract import (
    build_fit_target_contract,
    compare_capture_contracts,
    load_participant_gaze_measurement_contract,
    normalize_capture_contract,
    representative_capture_contract,
    validate_transport_frame,
)
from core.participant_study import ParticipantStudyStore
from core.participant_study.general_collection import (
    classify_provisional_geometry_quality,
    evaluate_validation_target_independence,
    summarize_validation_samples,
)
from scripts.test_general_collection import (
    _assessment_viewport,
    _consent_payload,
    _profile,
    _record_prediction_receipt_validation,
    _rehearsal_settings,
    _system_profile,
    _validation_samples,
)


def _capture_contract(
    *,
    source_width_px: int = 1280,
    source_height_px: int = 720,
    source_frame_rate_hz: float = 30.0,
    transport_width_px: int = 640,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "intent_width_px": 1280,
        "intent_height_px": 720,
        "intent_frame_rate_hz": 30.0,
        "source_width_px": source_width_px,
        "source_height_px": source_height_px,
        "source_frame_rate_hz": source_frame_rate_hz,
        "transport_width_px": transport_width_px,
        "transport_height_px": round(
            transport_width_px * source_height_px / source_width_px
        ),
        "resize_policy": "fit_width_preserve_aspect",
        "mime_type": "image/jpeg",
        "jpeg_quality": 0.8,
        "mirror_applied": False,
        "facing_mode": "user",
    }


def _passed_independence() -> dict[str, object]:
    return {"status": "passed", "independent": True}


def _verified_geometry_summary(**metrics: object) -> dict[str, object]:
    return {
        "prediction_receipt_status": "verified",
        "prediction_receipts_verified": True,
        **metrics,
    }


def _reference_fit_target_contract() -> dict[str, object]:
    measurement = load_participant_gaze_measurement_contract()
    reference = measurement["target_independence"][
        "selection_reference_calibration_targets"
    ]
    signed = [(x * 2.0 - 1.0, y * 2.0 - 1.0) for x, y in reference]
    return build_fit_target_contract(signed)


class CaptureContractTests(unittest.TestCase):
    def test_normalization_requires_aspect_preserving_transport(self) -> None:
        contract = _capture_contract()
        normalized = normalize_capture_contract(contract)
        self.assertEqual(normalized["transport_height_px"], 360)

        stretched = {**contract, "transport_height_px": 480}
        with self.assertRaisesRegex(ValueError, "preserve source aspect ratio"):
            normalize_capture_contract(stretched)

    def test_decoded_frame_must_match_declared_transport(self) -> None:
        contract = _capture_contract()
        validated = validate_transport_frame(
            contract,
            frame_width_px=640,
            frame_height_px=360,
        )
        self.assertEqual(validated["source_width_px"], 1280)
        with self.assertRaisesRegex(ValueError, "decoded frame dimensions"):
            validate_transport_frame(
                contract,
                frame_width_px=640,
                frame_height_px=480,
            )

    def test_resolution_change_is_compatible_but_aspect_change_is_not(self) -> None:
        reference = _capture_contract()
        same_aspect = _capture_contract(
            source_width_px=1920,
            source_height_px=1080,
            source_frame_rate_hz=24.0,
        )
        comparison = compare_capture_contracts(reference, same_aspect)
        self.assertTrue(comparison["compatible"])
        self.assertIn("source_resolution_changed", comparison["warnings"])

        smaller_transport = _capture_contract(transport_width_px=320)
        comparison = compare_capture_contracts(reference, smaller_transport)
        self.assertTrue(comparison["compatible"])
        self.assertIn("transport_resolution_changed", comparison["warnings"])

        four_by_three = _capture_contract(
            source_width_px=640,
            source_height_px=480,
        )
        comparison = compare_capture_contracts(reference, four_by_three)
        self.assertFalse(comparison["compatible"])
        self.assertIn("source_aspect_ratio_mismatch", comparison["reasons"])

        rear_camera = {**reference, "facing_mode": "environment"}
        comparison = compare_capture_contracts(reference, rear_camera)
        self.assertFalse(comparison["compatible"])
        self.assertIn("facing_mode_mismatch", comparison["reasons"])

    def test_training_contract_is_representative_and_legacy_safe(self) -> None:
        first = _capture_contract(source_frame_rate_hz=30.0)
        second = _capture_contract(source_frame_rate_hz=28.0)
        representative = representative_capture_contract(
            [{"capture_contract": first}, {"capture_contract": second}]
        )
        self.assertEqual(representative["source_frame_rate_hz"], 29.0)
        self.assertIsNone(representative_capture_contract([{"legacy": True}]))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            representative_capture_contract(
                [{"capture_contract": first}, {"legacy": True}]
            )

        fit_targets = build_fit_target_contract([(-0.5, -0.5), (0.5, 0.5)])
        self.assertEqual(fit_targets["status"], "available")
        self.assertEqual(fit_targets["target_count"], 2)
        inherited_unknown = build_fit_target_contract(
            [(0.0, 0.0)],
            inherited_targets_required=True,
        )
        self.assertEqual(inherited_unknown["status"], "unavailable")
        self.assertFalse(inherited_unknown["complete"])


class TargetIndependenceContractTests(unittest.TestCase):
    def _summary(self) -> dict[str, object]:
        return summarize_validation_samples(
            _validation_samples(),
            viewport_width_px=1280,
            viewport_height_px=800,
            prediction_receipt_status="verified",
        )

    def test_frozen_targets_pass_against_reference_calibration_targets(self) -> None:
        check = evaluate_validation_target_independence(
            self._summary(),
            _reference_fit_target_contract(),
        )
        self.assertEqual(check["status"], "passed")
        self.assertTrue(check["independent"])
        self.assertGreaterEqual(check["minimum_signed_target_distance"], 0.2)

    def test_overlap_and_unavailable_fit_targets_fail_closed(self) -> None:
        summary = self._summary()
        overlap = build_fit_target_contract([(-0.64, -0.60)])
        failed = evaluate_validation_target_independence(summary, overlap)
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["independent"])
        self.assertIn(
            "heldout_top_left",
            failed["overlapping_validation_target_ids"],
        )

        unavailable = evaluate_validation_target_independence(summary, None)
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIsNone(unavailable["independent"])

    def test_exact_distance_boundary_is_independent(self) -> None:
        boundary = build_fit_target_contract([(-0.84, -0.60)])
        check = evaluate_validation_target_independence(self._summary(), boundary)
        self.assertEqual(check["status"], "passed")
        self.assertAlmostEqual(check["minimum_signed_target_distance"], 0.2)

    def test_client_cannot_forge_or_omit_frozen_target_coordinates(self) -> None:
        forged = _validation_samples()
        forged[0]["target_x_norm"] = 0.0
        with self.assertRaisesRegex(ValueError, "signed target coordinates"):
            summarize_validation_samples(
                forged,
                viewport_width_px=1280,
                viewport_height_px=800,
            )

        missing = _validation_samples()
        missing[0].pop("target_y_norm")
        with self.assertRaisesRegex(ValueError, "target_y_norm"):
            summarize_validation_samples(
                missing,
                viewport_width_px=1280,
                viewport_height_px=800,
            )

        forged_pixels = _validation_samples()
        forged_pixels[0]["target_x_px"] += 1
        with self.assertRaisesRegex(ValueError, "pixel target coordinates"):
            summarize_validation_samples(
                forged_pixels,
                viewport_width_px=1280,
                viewport_height_px=800,
            )

        forged_id = _validation_samples()
        forged_id[0]["target_id"] = "client_selected_center"
        with self.assertRaisesRegex(ValueError, "frozen held-out set"):
            summarize_validation_samples(
                forged_id,
                viewport_width_px=1280,
                viewport_height_px=800,
            )


class ProvisionalGeometryQualityTests(unittest.TestCase):
    def test_classifier_uses_only_sensor_geometry_and_marks_final_pending(self) -> None:
        quality = classify_provisional_geometry_quality(
            _verified_geometry_summary(
                median_spatial_error_px=40.0,
                p90_spatial_error_px=90.0,
                prediction_success_fraction=0.85,
            ),
            capture_contract_check={"status": "compatible", "compatible": True},
            target_independence_check=_passed_independence(),
        )
        self.assertEqual(quality["status"], "provisional_sensor_geometry_only")
        self.assertEqual(quality["recommended_gaze_mode"], "word_level_candidate")
        self.assertFalse(quality["effective_sampling_hz_evaluated"])
        self.assertTrue(quality["final_quality_pending"])

    def test_contract_mismatch_downgrades_without_changing_spatial_band(self) -> None:
        quality = classify_provisional_geometry_quality(
            {
                "prediction_receipt_status": "verified",
                "prediction_receipts_verified": True,
                "median_spatial_error_px": 40.0,
                "p90_spatial_error_px": 90.0,
                "prediction_success_fraction": 0.85,
            },
            capture_contract_check={"status": "mismatch", "compatible": False},
            target_independence_check=_passed_independence(),
        )
        self.assertEqual(quality["spatial_band"], "word_level_candidate")
        self.assertEqual(quality["recommended_gaze_mode"], "behavioral_only")
        self.assertIn("capture_contract_mismatch", quality["reasons"])

    def test_missing_contract_fails_closed_but_keeps_spatial_description(self) -> None:
        quality = classify_provisional_geometry_quality(
            {
                "prediction_receipt_status": "verified",
                "prediction_receipts_verified": True,
                "median_spatial_error_px": 40.0,
                "p90_spatial_error_px": 90.0,
                "prediction_success_fraction": 0.85,
            },
            capture_contract_check={
                "status": "unavailable",
                "compatible": None,
            },
            target_independence_check=_passed_independence(),
        )
        self.assertEqual(quality["spatial_band"], "word_level_candidate")
        self.assertEqual(quality["recommended_gaze_mode"], "behavioral_only")
        self.assertIn("capture_contract_unavailable", quality["reasons"])

    def test_target_independence_is_required_for_any_gaze_recommendation(self) -> None:
        summary = {
            "prediction_receipt_status": "verified",
            "prediction_receipts_verified": True,
            "median_spatial_error_px": 40.0,
            "p90_spatial_error_px": 90.0,
            "prediction_success_fraction": 0.85,
        }
        for check, reason in (
            (
                {"status": "failed", "independent": False},
                "validation_target_independence_failed",
            ),
            (
                {"status": "unavailable", "independent": None},
                "validation_target_independence_unavailable",
            ),
        ):
            with self.subTest(status=check["status"]):
                quality = classify_provisional_geometry_quality(
                    summary,
                    capture_contract_check={
                        "status": "compatible",
                        "compatible": True,
                    },
                    target_independence_check=check,
                )
                self.assertEqual(quality["spatial_band"], "word_level_candidate")
                self.assertEqual(
                    quality["recommended_gaze_mode"],
                    "behavioral_only",
                )
                self.assertIn(reason, quality["reasons"])

    def test_existing_spatial_success_bands_remain_descriptive(self) -> None:
        for summary, expected in (
            (
                {
                    "prediction_receipt_status": "verified",
                    "prediction_receipts_verified": True,
                    "median_spatial_error_px": 90.0,
                    "p90_spatial_error_px": 170.0,
                    "prediction_success_fraction": 0.6,
                },
                "passage_level_only",
            ),
            (
                {
                    "prediction_receipt_status": "verified",
                    "prediction_receipts_verified": True,
                    "median_spatial_error_px": 250.0,
                    "p90_spatial_error_px": 400.0,
                    "prediction_success_fraction": 0.2,
                },
                "behavioral_only",
            ),
        ):
            with self.subTest(expected=expected):
                quality = classify_provisional_geometry_quality(
                    summary,
                    capture_contract_check={
                        "status": "compatible",
                        "compatible": True,
                    },
                    target_independence_check=_passed_independence(),
                )
                self.assertEqual(quality["recommended_gaze_mode"], expected)
                self.assertEqual(
                    quality["threshold_status"],
                    "rehearsal_descriptive_not_promotion_thresholds",
                )


class ProvisionalGeometryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-provisional-")
        self.addCleanup(self.temp_dir.cleanup)
        self.store = ParticipantStudyStore(
            Path(self.temp_dir.name),
            settings=_rehearsal_settings(),
        )
        pair = self.store.create_collection_invite_pairs(1)[0]
        enrolled = self.store.enroll(
            _consent_payload(pair["visits"][0]["invite_code"])
        )
        self.session_id = enrolled["study_session_id"]
        self.token = enrolled["access_token"]

    def _prepare(self, calibration_contract: dict[str, object]) -> None:
        self.store.record_general_profile(self.session_id, self.token, _profile())
        self.store.record_general_system_check(
            self.session_id,
            self.token,
            _system_profile(),
        )
        self.store.start_calibration(self.session_id, self.token, "GAZE-CONTRACT")
        self.store.complete_calibration(
            self.session_id,
            self.token,
            {
                "passed": True,
                "test_fixture": True,
                "capture_contract": calibration_contract,
                "fit_target_contract": _reference_fit_target_contract(),
            },
            model_name="contract-test-model",
        )
        self.store.start_general_collection(
            self.session_id,
            self.token,
            assessment_viewport=_assessment_viewport(),
        )

    def test_public_start_quality_is_immediate_and_behavior_can_continue(self) -> None:
        calibration_contract = _capture_contract()
        self._prepare(calibration_contract)
        public = self.store.record_general_validation(
            self.session_id,
            self.token,
            phase="start",
            samples=_validation_samples(),
            capture_contract=_capture_contract(
                source_width_px=640,
                source_height_px=480,
            ),
        )

        collection = public["general_collection"]
        provisional = collection["provisional_geometry_quality"]
        self.assertEqual(collection["phase"], "reading_ready")
        self.assertEqual(provisional["recommended_gaze_mode"], "behavioral_only")
        self.assertEqual(provisional["prediction_receipt_status"], "unavailable")
        self.assertFalse(provisional["prediction_receipts_verified"])
        self.assertEqual(
            collection["validations"]["start"]["sample_count"],
            0,
        )

        first_round = self.store.begin_general_round(self.session_id, self.token)
        self.assertEqual(first_round["phase"], "reading_active")
        self.assertIn("passage", first_round)

    def test_matching_capture_and_independent_targets_remain_provisional(
        self,
    ) -> None:
        calibration_contract = _capture_contract()
        self._prepare(calibration_contract)
        public = _record_prediction_receipt_validation(
            self.store,
            self.session_id,
            self.token,
            phase="start",
        )
        collection = public["general_collection"]
        provisional = collection["provisional_geometry_quality"]
        self.assertEqual(provisional["recommended_gaze_mode"], "word_level_candidate")
        self.assertTrue(provisional["validation_targets_independent"])
        self.assertEqual(
            collection["validations"]["start"]["target_independence_check"][
                "status"
            ],
            "passed",
        )
        self.assertRegex(
            collection["gaze_measurement_contract"]["sha256"],
            r"^[0-9a-f]{64}$",
        )
        validation = collection["validations"]["start"]
        self.assertEqual(
            validation["gaze_measurement_contract_sha256"],
            collection["gaze_measurement_contract"]["sha256"],
        )
        self.assertRegex(validation["validation_payload_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
