"""Contract tests for the fixed-form generalizable collection v1."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.participant_study import ParticipantStudyStore
from core.participant_study.general_collection import (
    WORD_PATTERN,
    assignment_for_cell,
    classify_gaze_quality,
    load_general_protocol,
    normalize_telemetry_batch,
    passage_by_id,
    probe_order,
    validate_general_design,
    validate_profile,
    validate_round_payload,
    validate_system_profile,
    williams_order,
)
from core.participant_study.protocol import load_protocol
from scripts.export_general_collection_dataset import export_bundle
from scripts.run_general_collection_rehearsal import resolve_data_location


def _rehearsal_settings() -> dict[str, object]:
    return {
        "LEXIGAZE_STUDY_MODE": "rehearsal",
        "LEXIGAZE_STUDY_REHEARSAL_MODE": "1",
        "LEXIGAZE_REHEARSAL_ACKNOWLEDGED_DEVELOPMENT_ONLY": "1",
        "LEXIGAZE_REHEARSAL_INVITES_ONLY": "1",
        "LEXIGAZE_REQUEST_BODY_LOGGING_DISABLED": "1",
        "LEXIGAZE_STORAGE_ENCRYPTED": "1",
        "LEXIGAZE_DATA_LOCATION": "encrypted-test-volume",
        "LEXIGAZE_PUBLIC_BASE_URL": "http://127.0.0.1:8080",
        "LEXIGAZE_DATA_RETENTION_DAYS": "30",
        "LEXIGAZE_RAW_FRAME_RETENTION_HOURS": "1",
    }


def _consent_payload(invite_code: str) -> dict[str, object]:
    protocol = load_protocol()
    return {
        "mode": "rehearsal",
        "invite_code": invite_code,
        "adult_confirmed": True,
        "private_space_confirmed": True,
        "consent_statements": {
            item["id"]: True for item in protocol["required_consent_statements"]
        },
        "comprehension_answers": {
            item["id"]: item["correct"] for item in protocol["comprehension_checks"]
        },
        "optional_scopes": {},
    }


def _profile() -> dict[str, str]:
    protocol = load_general_protocol()
    return {
        field: values[0]
        for field, values in protocol["profile_schema"]["required"].items()
    }


def _system_profile() -> dict[str, object]:
    protocol = load_general_protocol()
    return {
        "checks": {name: True for name in protocol["system_check"]["required"]},
        "device": {
            "device_class": "desktop",
            "browser_family": "chromium",
            "viewport_width": 1280,
            "viewport_height": 800,
            "device_pixel_ratio_bucket": "1_2",
            "camera_width": 640,
            "camera_height": 480,
            "estimated_camera_fps_band": "20_30",
        },
    }


def _validation_samples(offset: float = 10.0) -> list[dict[str, object]]:
    targets = [
        ("tl", 160.0, 120.0),
        ("tr", 1120.0, 120.0),
        ("c", 640.0, 400.0),
        ("bl", 160.0, 680.0),
        ("br", 1120.0, 680.0),
    ]
    return [
        {
            "target_id": target_id,
            "target_x_px": x,
            "target_y_px": y,
            "prediction_success": True,
            "predicted_x_px": x + offset + repeat,
            "predicted_y_px": y + offset - repeat,
        }
        for target_id, x, y in targets
        for repeat in range(3)
    ]


class GeneralCollectionDesignTests(unittest.TestCase):
    def test_frozen_design_and_bank_are_internally_consistent(self) -> None:
        audit = validate_general_design()
        self.assertEqual(audit["passage_count"], 12)
        self.assertEqual(audit["passage_family_count"], 12)
        self.assertEqual(audit["probe_count"], 96)
        self.assertRegex(audit["protocol_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(audit["bank_sha256"], r"^[0-9a-f]{64}$")

    def test_williams_rows_balance_first_order_carryover(self) -> None:
        rows = [williams_order(6, row) for row in range(6)]
        self.assertTrue(all(sorted(row) == list(range(6)) for row in rows))
        carryover = [
            (row[index], row[index + 1])
            for row in rows
            for index in range(len(row) - 1)
        ]
        self.assertEqual(len(carryover), 30)
        self.assertEqual(len(set(carryover)), 30)

    def test_twelve_schedule_cells_counterbalance_forms_and_orders(self) -> None:
        assignments = [assignment_for_cell(cell) for cell in range(12)]
        self.assertEqual(
            [item["sequence"] for item in assignments].count("A_then_B"), 6
        )
        self.assertEqual(
            [item["sequence"] for item in assignments].count("B_then_A"), 6
        )
        for assignment in assignments:
            self.assertEqual(len(assignment["visits"]), 2)
            self.assertEqual(len(assignment["visits"][0]["passage_order"]), 6)
            self.assertNotEqual(
                assignment["visits"][0]["form_id"],
                assignment["visits"][1]["form_id"],
            )


class GeneralCollectionRunnerTests(unittest.TestCase):
    def test_declared_data_location_must_match_actual_storage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-general-runner-") as name:
            root = Path(name)
            self.assertEqual(
                resolve_data_location(root, root / "data"),
                (root / "data").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "must exactly match"):
                resolve_data_location(root, root / "claimed-encrypted-location")


class GeneralCollectionInputTests(unittest.TestCase):
    def setUp(self) -> None:
        protocol = load_general_protocol()
        self.profile = {
            field: values[0]
            for field, values in protocol["profile_schema"]["required"].items()
        }

    def test_profile_is_strict_and_rejects_direct_identifiers(self) -> None:
        self.assertEqual(validate_profile(self.profile), self.profile)
        with self.assertRaisesRegex(ValueError, "direct participant identifiers"):
            validate_profile({**self.profile, "email": "person@example.invalid"})
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            validate_profile({**self.profile, "free_text": "anything"})

    def test_system_profile_stores_coarse_fields_without_user_agent(self) -> None:
        protocol = load_general_protocol()
        payload = {
            "checks": {name: True for name in protocol["system_check"]["required"]},
            "device": {
                "device_class": "desktop",
                "browser_family": "chromium",
                "viewport_width": 1280,
                "viewport_height": 800,
                "device_pixel_ratio_bucket": "1_2",
                "camera_width": 640,
                "camera_height": 480,
                "estimated_camera_fps_band": "20_30",
            },
        }
        normalized = validate_system_profile(payload)
        self.assertNotIn("user_agent", normalized["device"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_system_profile(
                {
                    **payload,
                    "device": {**payload["device"], "user_agent": "fingerprint"},
                }
            )

    def test_telemetry_accepts_derived_values_and_rejects_raw_media(self) -> None:
        payload = {
            "batch_id": "B-12345678",
            "passage_id": "foundation-a-seed-library",
            "viewport": {"width_px": 1280, "height_px": 800},
            "samples": [
                {
                    "monotonic_elapsed_ms": 250,
                    "prediction_success": True,
                    "screen_xy_norm": [0.0, 0.1],
                    "screen_xy_px": [640, 440],
                    "gaze_pitch_yaw": [0.01, -0.02],
                    "head_pose_pitch_yaw": [0.02, 0.03],
                    "normalized_face_bbox": [0.2, 0.1, 0.8, 0.9],
                    "nearest_word_index": 2,
                },
                {
                    "monotonic_elapsed_ms": 500,
                    "prediction_success": False,
                    "coarse_failure_code": "no_face",
                },
            ],
        }
        normalized = normalize_telemetry_batch(payload, maximum_word_index=100)
        self.assertEqual(len(normalized["samples"]), 2)
        self.assertNotIn("image_data", str(normalized))
        with self.assertRaisesRegex(ValueError, "raw media"):
            normalize_telemetry_batch(
                {**payload, "image_data": "data:image/jpeg;base64,raw"},
                maximum_word_index=100,
            )

    def test_quality_bands_abstain_instead_of_deleting_behavior(self) -> None:
        self.assertEqual(
            classify_gaze_quality(
                {
                    "median_spatial_error_px": 40,
                    "p90_spatial_error_px": 90,
                    "prediction_success_fraction": 0.85,
                    "effective_sampling_hz": 3.5,
                }
            ),
            "word_level_candidate",
        )
        self.assertEqual(
            classify_gaze_quality(
                {
                    "median_spatial_error_px": 90,
                    "p90_spatial_error_px": 170,
                    "prediction_success_fraction": 0.6,
                    "effective_sampling_hz": 1.5,
                }
            ),
            "passage_level_only",
        )
        self.assertEqual(
            classify_gaze_quality(
                {
                    "median_spatial_error_px": 250,
                    "p90_spatial_error_px": 400,
                    "prediction_success_fraction": 0.2,
                    "effective_sampling_hz": 0.5,
                }
            ),
            "behavioral_only",
        )

    def test_round_requires_every_fixed_probe_and_word_rectangle(self) -> None:
        passage_id = "foundation-a-seed-library"
        passage = passage_by_id(passage_id)
        participant_id = "GP-TEST"
        probes = probe_order(passage_id, participant_id, 1)
        word_count = len(WORD_PATTERN.findall(passage["text"]))
        payload = {
            "reading_elapsed_ms": 30_000,
            "scroll_occurred": False,
            "zoom_ratio": 1.0,
            "word_reviews": {item["probe_id"]: "unsure" for item in probes},
            "passage_self_report": {
                "understanding": 3,
                "mental_effort": 3,
                "read_complete": True,
                "interrupted": False,
            },
            "word_layout": [
                {
                    "word_index": index,
                    "left_px": index * 4,
                    "top_px": 10,
                    "right_px": index * 4 + 3,
                    "bottom_px": 28,
                }
                for index in range(word_count)
            ],
        }
        normalized = validate_round_payload(
            payload,
            passage_id=passage_id,
            participant_id=participant_id,
            visit_index=1,
        )
        self.assertEqual(len(normalized["word_reviews"]), 8)
        self.assertRegex(normalized["word_layout_sha256"], r"^[0-9a-f]{64}$")
        with self.assertRaisesRegex(ValueError, "word review IDs"):
            validate_round_payload(
                {**payload, "word_reviews": {}},
                passage_id=passage_id,
                participant_id=participant_id,
                visit_index=1,
            )


class GeneralCollectionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-general-store-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.store = ParticipantStudyStore(
            self.root,
            settings=_rehearsal_settings(),
        )
        self.assertTrue(self.store.activation["rehearsal_ready"])
        self.pair = self.store.create_collection_invite_pairs(1)[0]
        self.enrolled = self.store.enroll(
            _consent_payload(self.pair["visits"][0]["invite_code"])
        )
        self.session_id = self.enrolled["study_session_id"]
        self.token = self.enrolled["access_token"]

    def _prepare_collection(self) -> None:
        self.store.record_general_profile(
            self.session_id,
            self.token,
            _profile(),
        )
        self.store.record_general_system_check(
            self.session_id,
            self.token,
            _system_profile(),
        )
        self.store.start_calibration(self.session_id, self.token, "GAZE-TEST")
        self.store.complete_calibration(
            self.session_id,
            self.token,
            {"passed": True, "test_fixture": True},
            model_name="general-test-model",
        )
        self.store.start_general_collection(self.session_id, self.token)
        self.store.record_general_validation(
            self.session_id,
            self.token,
            phase="start",
            samples=_validation_samples(),
        )

    def _round_payload(self, passage_id: str) -> dict[str, object]:
        passage = passage_by_id(passage_id)
        probes = probe_order(
            passage_id,
            self.enrolled["participant_id"],
            1,
        )
        word_count = len(WORD_PATTERN.findall(passage["text"]))
        return {
            "reading_elapsed_ms": 30_000,
            "scroll_occurred": False,
            "zoom_ratio": 1.0,
            "word_reviews": {
                item["probe_id"]: "review_needed" for item in probes
            },
            "passage_self_report": {
                "understanding": 4,
                "mental_effort": 2,
                "read_complete": True,
                "interrupted": False,
            },
            "word_layout": [
                {
                    "word_index": index,
                    "left_px": 100 + index * 2,
                    "top_px": 100,
                    "right_px": 101 + index * 2,
                    "bottom_px": 120,
                }
                for index in range(word_count)
            ],
        }

    def test_invite_pair_freezes_same_participant_and_alternate_forms(self) -> None:
        self.assertEqual(self.enrolled["mode"], "rehearsal")
        status = self.store.get_session(self.session_id, self.token)
        assignment = status["collection_assignment"]
        self.assertEqual(assignment["form_id"], self.pair["visits"][0]["form_id"])
        self.assertRegex(assignment["protocol_sha256"], r"^[0-9a-f]{64}$")
        with self.assertRaisesRegex(Exception, "already-used"):
            self.store.enroll(
                _consent_payload(self.pair["visits"][0]["invite_code"])
            )
        with self.assertRaisesRegex(Exception, "visit 1 must be completed"):
            self.store.enroll(
                _consent_payload(self.pair["visits"][1]["invite_code"])
            )

    def test_full_six_round_rehearsal_keeps_behavior_and_classifies_gaze(self) -> None:
        self._prepare_collection()
        first_round = self.store.begin_general_round(self.session_id, self.token)
        first_passage_id = first_round["passage"]["passage_id"]
        telemetry = {
            "batch_id": "B-12345678",
            "passage_id": first_passage_id,
            "viewport": {"width_px": 1280, "height_px": 800},
            "samples": [
                {
                    "monotonic_elapsed_ms": index * 250,
                    "prediction_success": True,
                    "screen_xy_norm": [0.0, 0.0],
                    "screen_xy_px": [640, 400],
                    "gaze_pitch_yaw": [0.01, -0.01],
                    "head_pose_pitch_yaw": [index * 0.01, -index * 0.01],
                    "normalized_face_bbox": [0.2, 0.1, 0.8, 0.9],
                    "nearest_word_index": min(index, 4),
                }
                for index in range(8)
            ],
        }
        stored = self.store.record_general_telemetry_batch(
            self.session_id,
            self.token,
            telemetry,
        )
        self.assertFalse(stored["idempotent"])
        duplicate = self.store.record_general_telemetry_batch(
            self.session_id,
            self.token,
            telemetry,
        )
        self.assertTrue(duplicate["idempotent"])

        current = first_round
        for round_index in range(6):
            passage_id = current["passage"]["passage_id"]
            probes = self.store.open_general_word_reviews(
                self.session_id,
                self.token,
                passage_id=passage_id,
            )
            self.assertEqual(len(probes["probes"]), 8)
            status = self.store.record_general_round(
                self.session_id,
                self.token,
                passage_id=passage_id,
                payload=self._round_payload(passage_id),
            )
            if round_index < 5:
                current = self.store.begin_general_round(
                    self.session_id,
                    self.token,
                )

        self.assertEqual(
            status["general_collection"]["phase"],
            "end_validation_required",
        )
        completed = self.store.record_general_validation(
            self.session_id,
            self.token,
            phase="end",
            samples=_validation_samples(offset=14.0),
        )
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(
            completed["quality"]["general_collection"]["gaze_quality_band"],
            "behavioral_only",
        )
        self.assertTrue(
            completed["quality"]["general_collection"][
                "behavioral_labels_retained"
            ]
        )
        session_path = next(self.root.rglob("session.json"))
        self.assertEqual(list(session_path.parent.rglob("*.jpg")), [])
        self.assertEqual(len(list(session_path.parent.rglob("R*.json"))), 6)
        export_path = self.root / "private-export"
        manifest = export_bundle(self.root, export_path)
        self.assertEqual(manifest["participant_count"], 1)
        self.assertEqual(manifest["session_count"], 1)
        self.assertEqual(manifest["files"]["word_reviews.csv"]["row_count"], 48)
        self.assertFalse(manifest["formal_promotion_allowed"])
        exported_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in export_path.glob("*.csv")
        )
        self.assertNotIn("@", exported_text)

    def test_telemetry_batch_id_cannot_be_reused_with_new_content(self) -> None:
        self._prepare_collection()
        current = self.store.begin_general_round(self.session_id, self.token)
        passage_id = current["passage"]["passage_id"]
        payload = {
            "batch_id": "B-ABCDEFGH",
            "passage_id": passage_id,
            "viewport": {"width_px": 1280, "height_px": 800},
            "samples": [
                {
                    "monotonic_elapsed_ms": 250,
                    "prediction_success": False,
                    "coarse_failure_code": "timeout",
                }
            ],
        }
        self.store.record_general_telemetry_batch(
            self.session_id,
            self.token,
            payload,
        )
        with self.assertRaisesRegex(Exception, "reused with new content"):
            self.store.record_general_telemetry_batch(
                self.session_id,
                self.token,
                {
                    **payload,
                    "samples": [
                        {
                            "monotonic_elapsed_ms": 500,
                            "prediction_success": False,
                            "coarse_failure_code": "timeout",
                        }
                    ],
                },
            )


if __name__ == "__main__":
    unittest.main()
