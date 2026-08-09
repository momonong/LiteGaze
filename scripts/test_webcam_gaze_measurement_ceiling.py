"""Tests for the CPU-only webcam gaze measurement-ceiling audit."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from core.gaze_core.measurement_ceiling import (
    DEFAULT_TARGET_OVERLAP_TOLERANCE_SIGNED,
    MeasurementCeilingError,
    _nearest_rank,
    build_measurement_ceiling_result,
    deterministic_json,
    render_measurement_ceiling_markdown,
)
from scripts.audit_webcam_gaze_measurement_ceiling import _parse_args, main


class WebcamGazeMeasurementCeilingTests(unittest.TestCase):
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
            line_gap_px=25.0,
            median_word_width_px=50.0,
            bootstrap_resamples=200,
            bootstrap_seed=17,
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
        self.assertEqual(first["status"], "completed")
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
            (10.0**2 + 20.0**2) ** 0.5 / 25.0,
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

        markdown = render_measurement_ceiling_markdown(first)
        self.assertIn("does not imply line- or word-level resolution", markdown)
        self.assertIn("no independent word-level", markdown)
        self.assertIn("confusion matrices", markdown)
        self.assertIn("Signed mean px", markdown)
        self.assertIn("0.20", markdown)
        self.assertIn("viewport-fraction coordinates", markdown)

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
        self.assertEqual(result["status"], "completed")
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
                "25",
                "--median-word-width-px",
                "50",
                "--bootstrap-resamples",
                "200",
                "--bootstrap-seed",
                "17",
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


if __name__ == "__main__":
    unittest.main()
