"""Focused tests for leakage-safe analysis of the frozen 193-row run."""

from __future__ import annotations

import contextlib
import io
import math
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from core.gaze_core.measurement_run_analysis import (
    EXPECTED_ANALYSIS_DEFINITION_SHA256,
    _analyze_reverified_live_evidence,
    analyze_measurement_run,
    load_frozen_analysis_definition,
    render_measurement_run_markdown,
)
from core.gaze_core.measurement_schedule import (
    MeasurementScheduleError,
    build_capture_artifact,
    build_run_manifest,
    canonical_sha256,
    verify_capture_artifact,
)
from scripts.analyze_webcam_gaze_measurement_run import main as cli_main


RUN_ID = "WGMC-20260810-analysis-contract"


def _samples(manifest: dict, *, uncertainty: bool = False) -> list[dict]:
    samples: list[dict] = []
    viewport_width = 1600
    viewport_height = 900
    for row in manifest["rows"]:
        sequence_index = int(row["sequence_index"])
        target_x = float(
            math.floor(row["target_x_viewport_fraction"] * viewport_width + 0.5)
        )
        target_y = float(
            math.floor(row["target_y_viewport_fraction"] * viewport_height + 0.5)
        )
        sample = {
            "capture_run_id": row["capture_run_id"],
            "capture_source": "synthetic-dry-run",
            "block_id": row["block_id"],
            "block_role": row["block_role"],
            "target_id": row["target_id"],
            "target_x_norm": row["target_x_norm"],
            "target_y_norm": row["target_y_norm"],
            "target_x_px": target_x,
            "target_y_px": target_y,
            "repeat_index": row["repeat_index"],
            "sequence_index": sequence_index,
            "frame_capture_monotonic_ms": float(sequence_index * 100),
            "inference_completed_monotonic_ms": float(sequence_index * 100 + 12.5),
            "inference_latency_ms": 12.5,
            "model_id": (
                "synthetic-base-encoder"
                if row["block_role"] == "calibration"
                else "synthetic-personal-calibrator"
            ),
            "model_sha256": (
                "a" * 64 if row["block_role"] == "calibration" else "b" * 64
            ),
            "prediction_success": True,
            "raw_gaze_pitch_yaw": [0.01, -0.02],
            "predicted_x_px": target_x + 8.0,
            "predicted_y_px": target_y - 5.0,
            "head_pose_pitch_yaw": [0.02, -0.01],
            "normalized_face_bbox": [0.2, 0.1, 0.8, 0.9],
            "camera_width": 1280,
            "camera_height": 720,
            "camera_frame_rate": 30.0,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "device_pixel_ratio": 1.0,
        }
        if uncertainty and row["block_role"] == "evaluation":
            sample["sensor_uncertainty_score"] = sequence_index / 1000.0
        samples.append(sample)
    return samples


def _artifact(*, uncertainty: bool = False) -> dict:
    manifest = build_run_manifest(RUN_ID)
    return build_capture_artifact(
        manifest,
        _samples(manifest, uncertainty=uncertainty),
        evidence_class="dry_run_synthetic",
    )


def _rehash(artifact: dict) -> None:
    artifact["samples_sha256"] = canonical_sha256(artifact["samples"])
    core = deepcopy(artifact)
    core.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = canonical_sha256(core)


def _live_evidence() -> dict:
    manifest = build_run_manifest(f"{RUN_ID}-live")
    artifact = build_capture_artifact(
        manifest,
        _samples(manifest, uncertainty=True),
        evidence_class="physical_self_development",
    )
    verified = verify_capture_artifact(artifact)
    timing = {"schema_version": 1, "server_receive_to_predict_ms": 12.5}
    entries = [
        {
            "sequence_index": index,
            "ledger_role": "calibration" if index < 65 else "evaluation",
            "prediction_success": sample["prediction_success"],
            "failure_code": None,
            "sample_sha256": canonical_sha256(sample),
            "ledger_record_sha256": f"{index + 1:064x}",
            "capture_contract_evidence_sha256": f"{index + 194:064x}",
            "server_timing_evidence": timing,
            "server_timing_evidence_sha256": canonical_sha256(timing),
            "frame_sha256": f"{index + 387:064x}",
        }
        for index, sample in enumerate(artifact["samples"])
    ]
    sidecar = {
        "schema_version": 1,
        "sidecar_type": "webcam_gaze_measurement_ceiling_attempt_sidecar_v1",
        "capture_run_id": verified["capture_run_id"],
        "protocol_sha256": verified["protocol_sha256"],
        "manifest_sha256": verified["run_manifest_sha256"],
        "capture_artifact_sha256": verified["artifact_sha256"],
        "entries": entries,
        "measurement_claim_authorized": False,
        "physical_capture_claim_authorized": False,
    }
    sidecar["entries_sha256"] = canonical_sha256(entries)
    sidecar["sidecar_sha256"] = canonical_sha256(sidecar)
    calibration_rows = [
        row for row in manifest["rows"] if row["block_role"] == "calibration"
    ]
    evaluation_rows = [
        row for row in manifest["rows"] if row["block_role"] == "evaluation"
    ]
    calibration_targets = sorted({row["target_id"] for row in calibration_rows})
    evaluation_targets = sorted({row["target_id"] for row in evaluation_rows})
    calibration_ledger_sha = "c" * 64
    provenance = {
        "schema_version": 1,
        "provenance_type": "webcam_gaze_measurement_ceiling_training_v1",
        "capture_run_id": verified["capture_run_id"],
        "protocol_sha256": verified["protocol_sha256"],
        "manifest_sha256": verified["run_manifest_sha256"],
        "calibration_ledger_sha256": calibration_ledger_sha,
        "calibration_ordinals": list(range(65)),
        "training_role": "calibration_only",
        "calibration_schedule_rows_sha256": canonical_sha256(calibration_rows),
        "evaluation_schedule_rows_sha256": canonical_sha256(evaluation_rows),
        "calibration_target_ids_sha256": canonical_sha256(calibration_targets),
        "evaluation_target_ids_sha256": canonical_sha256(evaluation_targets),
        "train_samples": 65,
        "allow_cuda": False,
        "training_device_required": "cpu",
        "evaluation_labels_used": False,
        "evaluation_rows_used": 0,
        "evaluation_targets_excluded": True,
        "calibration_evaluation_target_intersection_count": 0,
        "text_cursor_cognitive_inputs_used": False,
        "measurement_claim_authorized": False,
    }
    provenance_sha = canonical_sha256(provenance)
    model_sha = "b" * 64
    purge = {
        "status": "verified",
        "postcondition_verified": True,
        "removed_directories": ["crop", "normalized_face", "raw"],
    }
    status = {
        "ok": True,
        "capture_run_id": verified["capture_run_id"],
        "phase": "artifact_verified",
        "protocol_sha256": verified["protocol_sha256"],
        "manifest_sha256": verified["run_manifest_sha256"],
        "progress": {
            "next_sequence_index": 193,
            "calibration_count": 65,
            "evaluation_count": 128,
        },
        "model_binding": {
            "model_id": "synthetic-personal-calibrator",
            "model_sha256": model_sha,
        },
        "runner": {
            "capture_contract_proof_count": 193,
            "trained_artifact": {
                "model_sha256": model_sha,
                "training_provenance_sha256": provenance_sha,
                "calibration_ledger_sha256": calibration_ledger_sha,
            },
            "calibration_image_purge": purge,
        },
        "capture_artifact": {
            "artifact_sha256": verified["artifact_sha256"],
            "sample_count": 193,
            "attempt_sidecar_sha256": sidecar["sidecar_sha256"],
            "attempt_sidecar_entries_sha256": sidecar["entries_sha256"],
        },
        "acquisition_artifact_verified": True,
        "capture_contract_binding_verified": True,
        "measurement_claim_authorized": False,
        "physical_capture_claim_authorized": False,
    }
    verification = {
        **verified,
        "phase": "artifact_verified",
        "acquisition_artifact_verified": True,
        "capture_contract_binding_verified": True,
    }
    evidence = {
        "schema_version": 1,
        "evidence_type": (
            "webcam_gaze_measurement_ceiling_verified_analysis_evidence_v1"
        ),
        "capture_run_id": verified["capture_run_id"],
        "verification": verification,
        "status": status,
        "capture_artifact": artifact,
        "attempt_sidecar": sidecar,
        "training_provenance": provenance,
        "model_sha256": model_sha,
        "calibration_image_purge": purge,
        "spool_absence_verified": True,
        "raw_frames_included": False,
        "run_token_included": False,
        "measurement_claim_authorized": False,
        "physical_capture_claim_authorized": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


class WebcamGazeMeasurementRunAnalysisTests(unittest.TestCase):
    def test_analysis_definition_is_frozen_before_capture(self) -> None:
        definition, digest = load_frozen_analysis_definition()
        self.assertEqual(digest, EXPECTED_ANALYSIS_DEFINITION_SHA256)
        self.assertEqual(definition["status"], "frozen_before_new_capture")
        self.assertFalse(
            definition["claim_boundary"]["post_result_metric_or_decoder_tuning_allowed"]
        )
        changed = deepcopy(definition)
        changed["uncertainty"]["fixed_conditional_coverage_grid"] = [1.0, 0.5]
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/changed.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(changed, handle)
            with self.assertRaisesRegex(
                MeasurementScheduleError, "canonical SHA-256 mismatch"
            ):
                load_frozen_analysis_definition(path)

    def test_exact_geometry_metrics_and_claim_boundary(self) -> None:
        result = analyze_measurement_run(
            _artifact(), bootstrap_resamples=100, bootstrap_seed=20260810
        )
        expected_error = math.hypot(8.0, -5.0)
        selected = result["evaluation"]["selected_personal_model"]
        self.assertEqual(result["status"], "synthetic_diagnostic_only")
        self.assertAlmostEqual(selected["mean_spatial_error_px"], expected_error)
        self.assertAlmostEqual(selected["median_spatial_error_px"], expected_error)
        self.assertAlmostEqual(selected["p90_spatial_error_px"], expected_error)
        self.assertAlmostEqual(selected["signed_mean_error_x_px"], 8.0)
        self.assertAlmostEqual(selected["signed_mean_error_y_px"], -5.0)
        self.assertEqual(selected["attempted_count"], 128)
        self.assertEqual(selected["successful_count"], 128)
        self.assertEqual(selected["covered_target_cluster_count"], 96)
        self.assertFalse(result["claim_boundary"]["measurement_claim_authorized"])
        self.assertFalse(result["claim_boundary"]["quality_band_change_authorized"])
        self.assertFalse(result["claim_boundary"]["threshold_selected"])
        self.assertIsNone(result["decision"]["finest_supported_resolution_band"])
        self.assertGreater(
            result["evaluation"]["viewport_center_baseline_b0"][
                "mean_spatial_error_px"
            ],
            selected["mean_spatial_error_px"],
        )

    def test_region_drift_timing_and_negative_control_are_fixed(self) -> None:
        result = analyze_measurement_run(
            _artifact(), bootstrap_resamples=20, bootstrap_seed=7
        )
        region = result["evaluation"]["target_region_4x4"]
        self.assertEqual(region["decoder"], "nearest_axis_on_frozen_4x4_target_grid_v1")
        self.assertEqual(region["classified_successful_count"], 128)
        self.assertEqual(region["accuracy"], 1.0)
        self.assertEqual(
            region["target_label_cyclic_permutation_negative_control"]["accuracy"],
            0.0,
        )
        drift = result["evaluation"]["neutral_start_to_end_drift"]
        self.assertEqual(drift["available_target_count"], 16)
        self.assertAlmostEqual(drift["target_macro_mean_drift_magnitude_px"], 0.0)
        self.assertAlmostEqual(
            result["timing"]["within_block_inference_start_interval_ms"]["p50"],
            100.0,
        )
        self.assertAlmostEqual(
            result["timing"]["effective_within_block_inference_start_rate_hz"],
            10.0,
        )
        self.assertEqual(
            result["timing"]["camera_capture_jitter"]["status"],
            "not_evaluable_without_trusted_exposure_timestamp",
        )

    def test_uncertainty_uses_fixed_coverage_grid_without_selecting_threshold(self) -> None:
        result = analyze_measurement_run(
            _artifact(uncertainty=True),
            bootstrap_resamples=20,
            bootstrap_seed=7,
        )
        uncertainty = result["uncertainty"]
        self.assertEqual(uncertainty["status"], "scored_no_threshold_descriptive")
        self.assertEqual(
            [row["requested_coverage"] for row in uncertainty["coverage_risk"]],
            [1.0, 0.8, 0.6, 0.4, 0.2],
        )
        self.assertEqual(
            [row["retained_count"] for row in uncertainty["coverage_risk"]],
            [128, 103, 77, 52, 26],
        )
        self.assertFalse(uncertainty["threshold_selected"])
        self.assertFalse(uncertainty["abstention_policy_changed"])
        self.assertEqual(len(uncertainty["deciles"]), 10)
        evaluation_rows = [
            row for row in result["derived_rows"] if row["block_role"] == "evaluation"
        ]
        self.assertIn("1.0", evaluation_rows[0]["uncertainty_hypothetical_retained"])
        self.assertIn("0.2", evaluation_rows[0]["uncertainty_hypothetical_retained"])

    def test_no_face_attempt_is_retained_and_does_not_block_conditional_score(self) -> None:
        artifact = _artifact(uncertainty=True)
        failed = next(
            sample
            for sample in artifact["samples"]
            if sample["block_role"] == "evaluation"
        )
        failed["prediction_success"] = False
        failed["raw_gaze_pitch_yaw"] = None
        failed["predicted_x_px"] = None
        failed["predicted_y_px"] = None
        failed["head_pose_pitch_yaw"] = None
        failed["normalized_face_bbox"] = None
        failed.pop("sensor_uncertainty_score")
        _rehash(artifact)
        result = analyze_measurement_run(
            artifact, bootstrap_resamples=20, bootstrap_seed=7
        )
        selected = result["evaluation"]["selected_personal_model"]
        self.assertEqual(selected["attempted_count"], 128)
        self.assertEqual(selected["successful_count"], 127)
        self.assertAlmostEqual(selected["prediction_success_fraction"], 127 / 128)
        self.assertEqual(
            result["uncertainty"]["status"], "scored_no_threshold_descriptive"
        )
        failed_row = next(
            row
            for row in result["derived_rows"]
            if row["sequence_index"] == failed["sequence_index"]
        )
        self.assertFalse(failed_row["prediction_success"])
        self.assertIsNone(failed_row["spatial_error_px"])
        self.assertTrue(
            all(
                retained is False
                for retained in failed_row["uncertainty_hypothetical_retained"].values()
            )
        )

    def test_missing_successful_uncertainty_fails_closed_without_changing_geometry(self) -> None:
        artifact = _artifact(uncertainty=True)
        scored = next(
            sample
            for sample in artifact["samples"]
            if sample["block_role"] == "evaluation"
        )
        scored.pop("sensor_uncertainty_score")
        _rehash(artifact)
        result = analyze_measurement_run(
            artifact, bootstrap_resamples=20, bootstrap_seed=7
        )
        self.assertEqual(
            result["uncertainty"]["status"],
            "not_evaluable_incomplete_successful_prediction_scores",
        )
        self.assertEqual(
            result["evaluation"]["selected_personal_model"]["successful_count"],
            128,
        )
        self.assertFalse(result["uncertainty"]["threshold_selected"])

    def test_artifact_tamper_is_rejected_before_metrics(self) -> None:
        artifact = _artifact()
        artifact["samples"][65]["target_x_px"] += 1.0
        _rehash(artifact)
        with self.assertRaisesRegex(MeasurementScheduleError, "target pixels"):
            analyze_measurement_run(
                artifact, bootstrap_resamples=20, bootstrap_seed=7
            )

    def test_output_is_deterministic_and_api_has_no_prior_surface(self) -> None:
        artifact = _artifact(uncertainty=True)
        first = analyze_measurement_run(
            artifact, bootstrap_resamples=50, bootstrap_seed=123
        )
        second = analyze_measurement_run(
            artifact, bootstrap_resamples=50, bootstrap_seed=123
        )
        self.assertEqual(first, second)
        self.assertRegex(first["analysis_sha256"], r"^[0-9a-f]{64}$")
        with self.assertRaises(TypeError):
            analyze_measurement_run(  # type: ignore[call-arg]
                artifact,
                bootstrap_resamples=20,
                text_prior={"word": 1.0},
            )
        with self.assertRaises(TypeError):
            analyze_measurement_run(  # type: ignore[call-arg]
                artifact,
                bootstrap_resamples=20,
                cognitive_profile={"score": 1.0},
            )

    def test_physical_label_alone_cannot_promote_without_runner_provenance(self) -> None:
        manifest = build_run_manifest(f"{RUN_ID}-physical-label")
        artifact = build_capture_artifact(
            manifest,
            _samples(manifest, uncertainty=True),
            evidence_class="physical_self_development",
        )
        result = analyze_measurement_run(
            artifact, bootstrap_resamples=20, bootstrap_seed=7
        )
        self.assertEqual(
            result["status"],
            "descriptive_metrics_only_pending_runner_provenance",
        )
        self.assertFalse(
            result["claim_boundary"][
                "runner_capture_and_training_provenance_verified_here"
            ]
        )
        self.assertFalse(result["claim_boundary"]["measurement_claim_authorized"])

    def test_reverified_live_evidence_promotes_integrity_scope_not_accuracy(self) -> None:
        evidence = _live_evidence()
        result = _analyze_reverified_live_evidence(
            evidence,
            bootstrap_resamples=20,
            bootstrap_seed=7,
        )
        self.assertEqual(
            result["status"], "integrity_verified_descriptive_live_runner"
        )
        self.assertTrue(
            result["claim_boundary"][
                "runner_capture_and_training_provenance_verified_here"
            ]
        )
        self.assertFalse(result["claim_boundary"]["measurement_claim_authorized"])
        self.assertFalse(result["claim_boundary"]["threshold_selected"])
        self.assertFalse(result["claim_boundary"]["quality_band_change_authorized"])
        self.assertEqual(
            result["negative_controls"]["evaluation_target_fitting"],
            "verified_excluded_by_authenticated_live_runner_training_provenance",
        )
        self.assertEqual(
            result["live_runner_provenance"]["no_face_attempt_count"], 0
        )
        self.assertFalse(
            result["live_runner_provenance"][
                "persisted_bundle_self_attestation_accepted"
            ]
        )

    def test_rehashed_live_bundle_cannot_override_training_semantics(self) -> None:
        evidence = _live_evidence()
        evidence["training_provenance"]["evaluation_targets_excluded"] = False
        evidence["evidence_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key != "evidence_sha256"
            }
        )
        with self.assertRaisesRegex(
            MeasurementScheduleError, "training provenance changed"
        ):
            _analyze_reverified_live_evidence(
                evidence,
                bootstrap_resamples=20,
                bootstrap_seed=7,
            )

    def test_bootstrap_configuration_is_strict(self) -> None:
        artifact = _artifact()
        with self.assertRaisesRegex(
            MeasurementScheduleError, "positive integer"
        ):
            analyze_measurement_run(
                artifact,
                bootstrap_resamples=20.5,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(MeasurementScheduleError, "integer"):
            analyze_measurement_run(
                artifact,
                bootstrap_resamples=20,
                bootstrap_seed=True,  # type: ignore[arg-type]
            )

    def test_markdown_is_compact_deterministic_and_preserves_claim_boundary(self) -> None:
        result = analyze_measurement_run(
            _artifact(uncertainty=True), bootstrap_resamples=20, bootstrap_seed=7
        )
        first = render_measurement_run_markdown(result)
        second = render_measurement_run_markdown(result)
        self.assertEqual(first, second)
        self.assertIn("193-row analysis", first)
        self.assertIn("No finest production resolution band", first)
        self.assertIn("Camera exposure/capture jitter: `not evaluable`", first)
        self.assertNotIn("raw_gaze_pitch_yaw", first)
        self.assertNotIn("normalized_face_bbox", first)

    def test_cli_writes_distinct_outputs_and_rejects_input_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "capture.json"
            json_output = root / "analysis.json"
            markdown_output = root / "analysis.md"
            source.write_text(
                json.dumps(_artifact(uncertainty=True)), encoding="utf-8"
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_main(
                    [
                        "--capture-artifact",
                        str(source),
                        "--json-output",
                        str(json_output),
                        "--markdown-output",
                        str(markdown_output),
                        "--bootstrap-resamples",
                        "20",
                        "--bootstrap-seed",
                        "7",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(json_output.is_file())
            self.assertTrue(markdown_output.is_file())
            result = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertFalse(
                result["analysis_contract"]["matches_frozen_default_configuration"]
            )
            self.assertIn("measurement_claim_authorized", stdout.getvalue())
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rejected = cli_main(
                    [
                        "--capture-artifact",
                        str(source),
                        "--json-output",
                        str(source),
                        "--markdown-output",
                        str(markdown_output),
                    ]
                )
            self.assertEqual(rejected, 2)
            self.assertIn("must be distinct", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
