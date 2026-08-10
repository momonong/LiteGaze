"""Contract tests for the frozen Webcam gaze measurement-ceiling protocol."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-10-webcam-gaze-measurement-ceiling-v1.json"
)
PREFLIGHT_PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-10-participant-gaze-integrity-preflight-v1.json"
)


class WebcamGazeMeasurementProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.preflight_wrapper = json.loads(
            PREFLIGHT_PROTOCOL_PATH.read_text(encoding="utf-8")
        )
        cls.preflight = cls.preflight_wrapper["protocol"]

    def test_participant_preflight_is_hash_bound_and_not_the_193_capture(
        self,
    ) -> None:
        canonical = json.dumps(
            self.preflight,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            self.preflight_wrapper["canonical_sha256"],
        )
        self.assertEqual(
            self.preflight["protocol_id"],
            "participant-gaze-integrity-preflight-v1",
        )
        self.assertFalse(
            self.preflight["scope"][
                "full_193_sample_measurement_ceiling_protocol_executed"
            ]
        )
        design = self.preflight["design"]
        self.assertEqual(design["calibration_context"]["rows"], 65)
        self.assertEqual(
            design["fixed_target_validation"]["receipts_per_phase"], 15
        )
        self.assertEqual(design["fixed_target_validation"]["total_receipts"], 30)
        self.assertEqual(design["participant_flow_total_observations"], 95)
        self.assertEqual(
            self.preflight["analysis_config"]["bootstrap_resamples"], 20_000
        )

    def test_protocol_is_frozen_before_new_capture(self) -> None:
        self.assertEqual(
            self.protocol["protocol_id"],
            "webcam-gaze-measurement-ceiling-v1",
        )
        self.assertEqual(self.protocol["status"], "frozen_before_new_capture")
        self.assertEqual(
            self.protocol["branch"],
            "experiment/webcam-gaze-measurement-ceiling-v1",
        )

    def test_calibration_and_evaluation_targets_are_separated(self) -> None:
        targets = self.protocol["targets"]
        calibration = targets["calibration"]["points"]
        evaluation = targets["evaluation"]["points"]
        minimum_required = targets["evaluation"][
            "minimum_normalized_distance_from_any_calibration_point"
        ]

        distances = [
            math.dist(calibration_point, evaluation_point)
            for calibration_point in calibration
            for evaluation_point in evaluation
        ]
        self.assertEqual(len(calibration), 13)
        self.assertEqual(len(evaluation), 16)
        self.assertGreaterEqual(min(distances), minimum_required)
        self.assertFalse(
            self.protocol["splits"][
                "calibration_and_evaluation_target_overlap_allowed"
            ]
        )
        self.assertFalse(targets["evaluation"]["fit_allowed"])

    def test_block_counts_and_order_match_the_frozen_design(self) -> None:
        blocks = self.protocol["blocks"]
        calibration_points = len(
            self.protocol["targets"]["calibration"]["points"]
        )
        evaluation_points = len(
            self.protocol["targets"]["evaluation"]["points"]
        )
        calibration_samples = sum(
            calibration_points * int(block["repeats"])
            for block in blocks
            if block["role"] == "calibration"
        )
        evaluation_samples = sum(
            evaluation_points * int(block["repeats"])
            for block in blocks
            if block["role"] == "evaluation"
        )
        block_ids = [block["id"] for block in blocks]

        self.assertEqual(calibration_samples, 65)
        self.assertEqual(evaluation_samples, 128)
        self.assertEqual(calibration_samples + evaluation_samples, 193)
        self.assertLess(
            block_ids.index("calibration_far"),
            block_ids.index("neutral_start"),
        )
        self.assertLess(
            block_ids.index("neutral_start"),
            block_ids.index("neutral_end"),
        )

    def test_sensor_gate_is_independent_and_cpu_only(self) -> None:
        forbidden = set(
            self.protocol["sample_contract"]["forbidden_as_sensor_quality_inputs"]
        )
        self.assertTrue(
            {
                "cursor position",
                "word identity",
                "text difficulty",
                "cognitive profile",
                "question or answer correctness",
                "reading outcome",
            }.issubset(forbidden)
        )
        self.assertTrue(
            self.protocol["quality_and_abstention"][
                "sensor_gate_must_run_before_layout_or_text_decoder"
            ]
        )
        self.assertFalse(
            self.protocol["quality_and_abstention"][
                "text_or_cognitive_prior_may_rescue_sensor_abstention"
            ]
        )
        self.assertEqual(self.protocol["compute"]["analysis_device"], "cpu")
        self.assertFalse(self.protocol["compute"]["gpu_allowed"])

    def test_capture_contract_forbids_aspect_ratio_distortion(self) -> None:
        capture = self.protocol["capture_contract"]
        accepted_aspect = capture["accepted_actual_aspect_ratio"]
        must_match = set(
            capture["calibration_validation_and_reading_must_match"]
        )
        self.assertTrue(
            capture["inference_frame"]["preserve_actual_aspect_ratio"]
        )
        self.assertFalse(
            capture["inference_frame"]["canvas_stretching_allowed"]
        )
        self.assertEqual(
            capture["mismatch_action"],
            "abstain_and_require_recalibration_or_continue_as_behavioral_only",
        )
        self.assertLessEqual(accepted_aspect["minimum"], 4 / 3)
        self.assertGreaterEqual(accepted_aspect["maximum"], 16 / 9)
        self.assertEqual(
            accepted_aspect["maximum_cross_phase_absolute_difference"],
            0.02,
        )
        self.assertEqual(
            must_match,
            {
                "source_aspect_ratio_within_0.02",
                "transport_aspect_ratio_within_0.02",
                "resize_policy",
                "mirror_policy",
                "facing_mode",
            },
        )
        self.assertFalse(capture["exact_source_resolution_must_match"])
        self.assertTrue(
            capture[
                "source_resolution_and_frame_rate_changes_are_recorded_as_warnings"
            ]
        )
        frame_rate = capture["actual_frame_rate_diagnostic"]
        self.assertEqual(frame_rate["expected_intent_fps"], 30)
        self.assertEqual(
            frame_rate["expected_operating_band_fps"],
            {"minimum_inclusive": 24, "maximum_exclusive": 31},
        )
        self.assertEqual(frame_rate["eligibility_effect"], "recorded_warning_only")
        self.assertFalse(frame_rate["hard_integrity_gate"])
        self.assertNotIn("accepted_actual_frame_rate_fps", capture)


if __name__ == "__main__":
    unittest.main()
