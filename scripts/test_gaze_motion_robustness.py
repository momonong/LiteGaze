"""CPU-only regression tests for motion-diverse calibration audits."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from core.gaze_core.motion_robustness import (
    audit_payload,
    capture_metadata,
    grouped_validation_folds,
    load_motion_samples,
)


def _record(
    *,
    block: str | None,
    point: int,
    repeat: int,
    yaw_deg: float,
) -> dict[str, object]:
    if block == "near":
        face_width, face_height = 0.65, 0.78
    elif block == "far":
        face_width, face_height = 0.4, 0.48
    else:
        face_width, face_height = 0.5, 0.6
    record: dict[str, object] = {
        "phase": "calibration",
        "point_index": point,
        "repeat_index": repeat,
        "head_pose_pitch_yaw": [math.radians(20.0), math.radians(yaw_deg)],
        "face_bbox": {"w_norm": face_width, "h_norm": face_height},
    }
    if block is not None:
        record.update(
            {
                "camera_id": "integrated-webcam",
                "capture_burst_id": f"{block}:r{repeat}",
                "capture_run_id": "capture-test-run",
                "capture_source": "direct-frame",
                "collect_mode": "motion_robust",
                "collection_protocol": "motion-diverse-v1",
                "device_class": "laptop",
                "distance_condition": (
                    block if block in {"near", "far"} else "nominal"
                ),
                "lighting_condition": "room-light",
                "motion_block_id": block,
                "posture_condition": (
                    block if block in {"left", "right"} else "neutral"
                ),
            }
        )
    return record


def _write_session(
    root: Path,
    *,
    session_id: str,
    participant_id: str,
    records: list[dict[str, object] | str],
) -> Path:
    session_dir = root / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(
        json.dumps({"participant_id": participant_id}),
        encoding="utf-8",
    )
    lines = [record if isinstance(record, str) else json.dumps(record) for record in records]
    manifest = session_dir / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


class MotionCoverageAuditTests(unittest.TestCase):
    def test_capture_metadata_uses_bounded_allow_list(self) -> None:
        metadata = capture_metadata(
            {
                "camera_id": "camera-a",
                "capture_run_id": "capture-a",
                "capture_source": "direct-frame",
                "camera_width": "1280",
                "posture_condition": "left",
                "unknown_private_field": "must-not-be-stored",
                "motion_block_id": "x" * 200,
            }
        )

        self.assertEqual(metadata["camera_id"], "camera-a")
        self.assertEqual(metadata["capture_run_id"], "capture-a")
        self.assertEqual(metadata["capture_source"], "direct-frame")
        self.assertEqual(metadata["camera_width"], 1280.0)
        self.assertEqual(metadata["posture_condition"], "left")
        self.assertEqual(len(metadata["motion_block_id"]), 128)
        self.assertNotIn("unknown_private_field", metadata)

    def test_current_style_data_fails_frozen_motion_gates_without_identity_leak(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-motion-audit-") as name:
            root = Path(name)
            _write_session(
                root,
                session_id="session-a",
                participant_id="private-participant-label",
                records=[
                    _record(block=None, point=0, repeat=0, yaw_deg=2.0),
                    _record(block=None, point=1, repeat=0, yaw_deg=4.0),
                    "{malformed-json",
                ],
            )

            samples, diagnostics = load_motion_samples(root)
            payload = audit_payload(samples, diagnostics)

        self.assertEqual(payload["status"], "not_ready")
        self.assertEqual(payload["summary"]["samples"], 2)
        self.assertEqual(payload["summary"]["malformed_lines"], 1)
        self.assertNotIn("private-participant-label", json.dumps(payload))
        issue_codes = {issue["code"] for issue in payload["issues"]}
        self.assertIn("INSUFFICIENT_USABLE_SAMPLES", issue_codes)
        self.assertIn("INSUFFICIENT_MOTION_BLOCKS", issue_codes)
        self.assertIn("NO_CROSS_CONDITION_TARGET_REPLICATION", issue_codes)
        self.assertIn("NARROW_HEAD_YAW_COVERAGE", issue_codes)
        self.assertIn("INCOMPLETE_MOTION_BLOCK_ID", issue_codes)
        self.assertIn("MALFORMED_MANIFEST_LINES", issue_codes)

    def test_diverse_replicated_protocol_passes_and_hash_is_stable(self) -> None:
        yaw_by_block = {
            "far": 5.0,
            "left": -15.0,
            "near": -5.0,
            "neutral": 0.0,
            "right": 15.0,
        }
        records = [
            _record(block=block, point=point, repeat=repeat, yaw_deg=yaw)
            for block, yaw in yaw_by_block.items()
            for point in range(9)
            for repeat in range(2)
        ]

        with tempfile.TemporaryDirectory(prefix="lexigaze-motion-audit-") as name:
            root = Path(name)
            manifest = _write_session(
                root,
                session_id="session-b",
                participant_id="participant-b",
                records=records,
            )
            samples, diagnostics = load_motion_samples(root)
            payload = audit_payload(samples, diagnostics)
            _, second_diagnostics = load_motion_samples(root)
            stable_hash = diagnostics.source_sha256
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            _, changed_diagnostics = load_motion_samples(root)

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["issues"], [])
        self.assertEqual(payload["summary"]["samples"], 90)
        self.assertEqual(payload["summary"]["unique_motion_blocks"], 5)
        self.assertEqual(payload["summary"]["replicated_session_targets"], 9)
        self.assertEqual(
            payload["summary"]["collection_protocol_distribution"],
            {"motion-diverse-v1": 90},
        )
        self.assertEqual(stable_hash, second_diagnostics.source_sha256)
        self.assertNotEqual(stable_hash, changed_diagnostics.source_sha256)

    def test_labels_alone_cannot_fake_physical_motion_coverage(self) -> None:
        records = []
        for block in ("neutral", "left", "right", "near", "far"):
            for point in range(9):
                for repeat in range(2):
                    record = _record(
                        block=block,
                        point=point,
                        repeat=repeat,
                        yaw_deg=0.0,
                    )
                    record["face_bbox"] = {"w_norm": 0.5, "h_norm": 0.6}
                    records.append(record)

        with tempfile.TemporaryDirectory(prefix="lexigaze-motion-labels-") as name:
            root = Path(name)
            _write_session(
                root,
                session_id="session-labels-only",
                participant_id="participant-labels-only",
                records=records,
            )
            samples, diagnostics = load_motion_samples(root)
            payload = audit_payload(samples, diagnostics)

        issue_codes = {issue["code"] for issue in payload["issues"]}
        self.assertIn("NARROW_HEAD_YAW_COVERAGE", issue_codes)
        self.assertIn("INSUFFICIENT_LATERAL_POSE_SEPARATION", issue_codes)
        self.assertIn("INSUFFICIENT_DISTANCE_SEPARATION", issue_codes)

    def test_grouped_folds_keep_motion_blocks_disjoint(self) -> None:
        samples = []
        with tempfile.TemporaryDirectory(prefix="lexigaze-motion-folds-") as name:
            root = Path(name)
            records = [
                _record(block=block, point=point, repeat=0, yaw_deg=yaw)
                for block, yaw in (("left", -15.0), ("right", 15.0))
                for point in range(3)
            ]
            _write_session(
                root,
                session_id="session-c",
                participant_id="participant-c",
                records=records,
            )
            samples, _ = load_motion_samples(root)

        folds = grouped_validation_folds(samples, group_field="motion_block_id")
        self.assertEqual([fold.group_name for fold in folds], ["left", "right"])
        for fold in folds:
            train_groups = {samples[index].motion_block_id for index in fold.train_indices}
            validation_groups = {
                samples[index].motion_block_id for index in fold.validation_indices
            }
            self.assertTrue(train_groups.isdisjoint(validation_groups))
            self.assertEqual(
                set(fold.train_indices) | set(fold.validation_indices),
                set(range(len(samples))),
            )

    def test_grouped_folds_reject_missing_group_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-motion-folds-") as name:
            root = Path(name)
            _write_session(
                root,
                session_id="session-d",
                participant_id="participant-d",
                records=[_record(block=None, point=0, repeat=0, yaw_deg=0.0)],
            )
            samples, _ = load_motion_samples(root)

        with self.assertRaisesRegex(ValueError, "missing motion_block_id"):
            grouped_validation_folds(samples, group_field="motion_block_id")

    def test_loader_can_isolate_a_frozen_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-motion-filter-") as name:
            root = Path(name)
            for session_id in ("keep", "exclude"):
                _write_session(
                    root,
                    session_id=session_id,
                    participant_id=f"participant-{session_id}",
                    records=[
                        _record(
                            block="neutral",
                            point=0,
                            repeat=0,
                            yaw_deg=0.0,
                        )
                    ],
                )

            samples, diagnostics = load_motion_samples(
                root,
                session_ids=("keep",),
            )

        self.assertEqual(diagnostics.sessions_seen, 1)
        self.assertEqual(diagnostics.manifests_seen, 1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].session_id, "keep")


if __name__ == "__main__":
    unittest.main()
