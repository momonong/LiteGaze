"""End-to-end contracts for the frozen 193-row gaze acquisition schedule."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from core.gaze_core.measurement_schedule import (
    EXPECTED_CALIBRATION_SAMPLE_COUNT,
    EXPECTED_EVALUATION_SAMPLE_COUNT,
    EXPECTED_PROTOCOL_CANONICAL_SHA256,
    EXPECTED_SAMPLE_COUNT,
    MeasurementScheduleError,
    build_capture_artifact,
    build_run_manifest,
    canonical_sha256,
    load_frozen_protocol,
    verify_capture_artifact,
    verify_run_manifest,
)
from scripts.run_webcam_gaze_measurement_acquisition_dry_run import main as cli_main


RUN_ID = "WGMC-20260810-synthetic-contract"


def _rehash_manifest(manifest: dict) -> None:
    manifest["rows_sha256"] = canonical_sha256(manifest["rows"])
    core = deepcopy(manifest)
    core.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(core)


def _rehash_artifact(artifact: dict) -> None:
    artifact["samples_sha256"] = canonical_sha256(artifact["samples"])
    core = deepcopy(artifact)
    core.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = canonical_sha256(core)


def _synthetic_samples(manifest: dict) -> list[dict]:
    samples: list[dict] = []
    viewport_width = 1600
    viewport_height = 900
    for row in manifest["rows"]:
        sequence_index = int(row["sequence_index"])
        captured_ms = float(sequence_index * 100)
        latency_ms = 12.5
        target_x_px = float(
            math.floor(row["target_x_viewport_fraction"] * viewport_width + 0.5)
        )
        target_y_px = float(
            math.floor(row["target_y_viewport_fraction"] * viewport_height + 0.5)
        )
        samples.append(
            {
                "capture_run_id": row["capture_run_id"],
                "capture_source": "synthetic-dry-run",
                "block_id": row["block_id"],
                "block_role": row["block_role"],
                "target_id": row["target_id"],
                "target_x_norm": row["target_x_norm"],
                "target_y_norm": row["target_y_norm"],
                "target_x_px": target_x_px,
                "target_y_px": target_y_px,
                "repeat_index": row["repeat_index"],
                "sequence_index": sequence_index,
                "frame_capture_monotonic_ms": captured_ms,
                "inference_completed_monotonic_ms": captured_ms + latency_ms,
                "inference_latency_ms": latency_ms,
                "model_id": (
                    "synthetic-base-encoder"
                    if row["block_role"] == "calibration"
                    else "synthetic-personal-calibrator"
                ),
                "model_sha256": "a" * 64,
                "prediction_success": True,
                "raw_gaze_pitch_yaw": [0.01, -0.02],
                "predicted_x_px": target_x_px + 8.0,
                "predicted_y_px": target_y_px - 5.0,
                "head_pose_pitch_yaw": [0.02, -0.01],
                "normalized_face_bbox": [0.2, 0.1, 0.8, 0.9],
                "camera_width": 1280,
                "camera_height": 720,
                "camera_frame_rate": 30.0,
                "viewport_width": viewport_width,
                "viewport_height": viewport_height,
                "device_pixel_ratio": 1.0,
            }
        )
    return samples


class WebcamGazeMeasurementScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_run_manifest(RUN_ID)

    def test_frozen_protocol_hash_and_exact_counts(self) -> None:
        protocol, protocol_sha256 = load_frozen_protocol()
        self.assertEqual(protocol_sha256, EXPECTED_PROTOCOL_CANONICAL_SHA256)
        self.assertEqual(protocol["protocol_id"], "webcam-gaze-measurement-ceiling-v1")
        summary = verify_run_manifest(self.manifest)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["block_count"], 11)
        self.assertEqual(
            summary["calibration_sample_count"],
            EXPECTED_CALIBRATION_SAMPLE_COUNT,
        )
        self.assertEqual(
            summary["evaluation_sample_count"],
            EXPECTED_EVALUATION_SAMPLE_COUNT,
        )
        self.assertEqual(summary["total_sample_count"], EXPECTED_SAMPLE_COUNT)
        self.assertFalse(summary["measurement_claim_authorized"])

    def test_schedule_is_deterministic_seeded_and_run_specific(self) -> None:
        repeated = build_run_manifest(RUN_ID)
        different = build_run_manifest(f"{RUN_ID}-other")
        self.assertEqual(self.manifest, repeated)
        self.assertNotEqual(
            [row["target_id"] for row in self.manifest["rows"]],
            [row["target_id"] for row in different["rows"]],
        )
        self.assertNotEqual(
            self.manifest["manifest_sha256"], different["manifest_sha256"]
        )
        all_seeds: list[str] = []
        for block in self.manifest["blocks"]:
            self.assertEqual(
                len(block["repeat_seed_sha256"]), block["repeat_count"]
            )
            all_seeds.extend(block["repeat_seed_sha256"])
        self.assertEqual(len(all_seeds), 13)
        self.assertEqual(len(set(all_seeds)), 13)
        self.assertEqual(
            self.manifest["blocks"][0]["repeat_seed_sha256"][0],
            hashlib.sha256(
                f"{RUN_ID}calibration_neutral0".encode("utf-8")
            ).hexdigest(),
        )

    def test_every_block_repeat_has_one_complete_shuffled_target_set(self) -> None:
        by_repeat: dict[tuple[str, int], list[dict]] = {}
        for row in self.manifest["rows"]:
            by_repeat.setdefault(
                (row["block_id"], row["repeat_index"]), []
            ).append(row)
        self.assertEqual(len(by_repeat), 13)
        for rows in by_repeat.values():
            role = rows[0]["block_role"]
            expected_count = 13 if role == "calibration" else 16
            expected_ids = {
                f"{role}_{index:02d}" for index in range(expected_count)
            }
            self.assertEqual(len(rows), expected_count)
            self.assertEqual({row["target_id"] for row in rows}, expected_ids)
            self.assertEqual(
                [row["block_sequence_index"] for row in rows],
                list(range(expected_count)),
            )
            self.assertEqual(len({row["target_order_seed_sha256"] for row in rows}), 1)

    def test_manifest_fails_closed_on_hash_missing_duplicate_order_and_field(self) -> None:
        mutations: list[tuple[str, dict, str, bool]] = []

        bad_hash = deepcopy(self.manifest)
        bad_hash["rows"][0]["target_x_norm"] += 0.01
        mutations.append(("hash", bad_hash, "SHA-256 mismatch", False))

        missing = deepcopy(self.manifest)
        del missing["rows"][0]["target_id"]
        mutations.append(("missing", missing, "missing fields", True))

        duplicate = deepcopy(self.manifest)
        duplicate["rows"][1] = deepcopy(duplicate["rows"][0])
        mutations.append(("duplicate", duplicate, "row order|duplicate", True))

        reordered = deepcopy(self.manifest)
        reordered["rows"][0], reordered["rows"][1] = (
            reordered["rows"][1],
            reordered["rows"][0],
        )
        mutations.append(("order", reordered, "row order", True))

        field = deepcopy(self.manifest)
        field["rows"][0]["posture"] = "forged"
        mutations.append(("field", field, "differs from frozen", True))

        for name, payload, message, rehash in mutations:
            with self.subTest(name=name):
                if rehash:
                    _rehash_manifest(payload)
                with self.assertRaisesRegex(MeasurementScheduleError, message):
                    verify_run_manifest(payload)

    def test_any_protocol_mutation_requires_a_new_version(self) -> None:
        protocol, _ = load_frozen_protocol()
        changed = deepcopy(protocol)
        changed["blocks"][5]["repeats"] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed-protocol.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(
                MeasurementScheduleError, "canonical SHA-256 mismatch"
            ):
                load_frozen_protocol(path)

    def test_synthetic_schedule_to_artifact_to_verify_end_to_end(self) -> None:
        artifact = build_capture_artifact(
            self.manifest,
            _synthetic_samples(self.manifest),
            evidence_class="dry_run_synthetic",
        )
        summary = verify_capture_artifact(artifact)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["sample_count"], EXPECTED_SAMPLE_COUNT)
        self.assertEqual(summary["successful_sample_count"], EXPECTED_SAMPLE_COUNT)
        self.assertEqual(summary["evidence_class"], "dry_run_synthetic")
        self.assertFalse(summary["measurement_claim_authorized"])
        self.assertEqual(
            artifact["measurement_evidence_status"], "synthetic_not_evidence"
        )

    def test_artifact_fails_closed_on_rehashed_semantic_tampering(self) -> None:
        artifact = build_capture_artifact(
            self.manifest,
            _synthetic_samples(self.manifest),
            evidence_class="dry_run_synthetic",
        )
        mutations: list[tuple[str, dict, str, bool]] = []

        hash_only = deepcopy(artifact)
        hash_only["samples"][0]["predicted_x_px"] += 1.0
        mutations.append(("hash", hash_only, "artifact SHA-256 mismatch", False))

        missing = deepcopy(artifact)
        del missing["samples"][0]["model_sha256"]
        mutations.append(("missing", missing, "missing required fields", True))

        order = deepcopy(artifact)
        order["samples"][0], order["samples"][1] = (
            order["samples"][1],
            order["samples"][0],
        )
        mutations.append(("order", order, "differs from schedule", True))

        target = deepcopy(artifact)
        target["samples"][0]["target_x_px"] += 1
        mutations.append(("target", target, "target pixels", True))

        forbidden = deepcopy(artifact)
        forbidden["samples"][0]["cognitive_profile"] = {"theta": 1.0}
        mutations.append(("forbidden", forbidden, "forbidden sensor input", True))

        raw_media = deepcopy(artifact)
        raw_media["samples"][0]["image_data"] = "data:image/jpeg;base64,fixture"
        mutations.append(("raw-media", raw_media, "forbidden sensor input", True))

        nested_prior = deepcopy(artifact)
        nested_prior["samples"][0]["sensor_uncertainty_score"] = {
            "cognitive_profile": {"theta": 1.0}
        }
        mutations.append(("nested-prior", nested_prior, "forbidden sensor input", True))

        malformed_uncertainty = deepcopy(artifact)
        malformed_uncertainty["samples"][0]["sensor_uncertainty_score"] = {
            "geometry_only": 0.5
        }
        mutations.append(
            (
                "malformed-uncertainty",
                malformed_uncertainty,
                "sensor_uncertainty_score must be numeric",
                True,
            )
        )

        camera = deepcopy(artifact)
        camera["samples"][0]["camera_width"] = 640
        camera["samples"][0]["camera_height"] = 480
        mutations.append(("camera", camera, "aspect ratio changed", True))

        timing = deepcopy(artifact)
        timing["samples"][0]["frame_capture_monotonic_ms"] = 1000.0
        timing["samples"][0]["inference_completed_monotonic_ms"] = 1012.5
        mutations.append(("timing", timing, "monotonic frame order", True))

        model = deepcopy(artifact)
        model["samples"][0]["model_sha256"] = "b" * 64
        mutations.append(("model", model, "calibration model binding changed", True))

        for name, payload, message, rehash in mutations:
            with self.subTest(name=name):
                if rehash:
                    _rehash_artifact(payload)
                with self.assertRaisesRegex(MeasurementScheduleError, message):
                    verify_capture_artifact(payload)

    def test_failed_prediction_keeps_required_fields_but_may_use_null_sensor_values(
        self,
    ) -> None:
        samples = _synthetic_samples(self.manifest)
        samples[7].update(
            {
                "prediction_success": False,
                "raw_gaze_pitch_yaw": None,
                "predicted_x_px": None,
                "predicted_y_px": None,
                "head_pose_pitch_yaw": None,
                "normalized_face_bbox": None,
            }
        )
        artifact = build_capture_artifact(
            self.manifest,
            samples,
            evidence_class="physical_self_development",
        )
        summary = verify_capture_artifact(artifact)
        self.assertEqual(summary["successful_sample_count"], 192)
        self.assertEqual(
            artifact["measurement_evidence_status"],
            "contract_complete_pending_quality_analysis",
        )
        self.assertFalse(summary["measurement_claim_authorized"])

    def test_dry_run_cli_creates_and_reverifies_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "run-manifest.json"
            summary_path = root / "summary.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = cli_main(
                    [
                        "--create-manifest",
                        str(manifest_path),
                        "--capture-run-id",
                        RUN_ID,
                        "--summary-output",
                        str(summary_path),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertIn("status=passed", stdout.getvalue())
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted, self.manifest)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["total_sample_count"], 193)
            self.assertEqual(summary["cuda_visible_devices"], "-1")
            self.assertFalse(summary["torch_imported"])
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                verify_result = cli_main(
                    ["--verify-manifest", str(manifest_path)]
                )
            self.assertEqual(verify_result, 0)
            self.assertIn("mode=verify_manifest", stdout.getvalue())

            artifact = build_capture_artifact(
                persisted,
                _synthetic_samples(persisted),
                evidence_class="dry_run_synthetic",
            )
            artifact_path = root / "synthetic-capture-artifact.json"
            artifact_path.write_text(
                json.dumps(artifact, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                artifact_result = cli_main(
                    ["--verify-capture-artifact", str(artifact_path)]
                )
            self.assertEqual(artifact_result, 0)
            self.assertIn("mode=verify_capture_artifact", stdout.getvalue())
            self.assertIn("measurement_claim_authorized=false", stdout.getvalue())

    def test_core_and_cli_have_no_gpu_or_network_dependencies(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sources = "\n".join(
            (root / path).read_text(encoding="utf-8")
            for path in (
                "core/gaze_core/measurement_schedule.py",
                "scripts/run_webcam_gaze_measurement_acquisition_dry_run.py",
            )
        )
        for forbidden in (
            "import cv2",
            "import numpy",
            "import requests",
            "import socket",
            "import torch",
            "urllib.request",
        ):
            self.assertNotIn(forbidden, sources)


if __name__ == "__main__":
    unittest.main()
