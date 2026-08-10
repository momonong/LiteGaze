"""Contract tests for the fixed-form generalizable collection v1."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from core.participant_study import (
    ParticipantStudyStore,
    READING_VIDEO_SCOPE,
    StudyValidationError,
)
from core.participant_study.general_collection import (
    WORD_PATTERN,
    assignment_for_cell,
    canonical_sha256,
    classify_gaze_quality,
    load_general_protocol,
    normalize_telemetry_batch,
    passage_by_id,
    probe_order,
    validate_general_design,
    validate_profile,
    validate_round_payload,
    validate_system_profile,
    validation_target_definitions,
    williams_order,
)
from core.participant_study.protocol import (
    activation_status,
    load_protocol,
    public_protocol,
)
from scripts.export_general_collection_dataset import export_bundle
from scripts.run_general_collection_rehearsal import resolve_data_location
from web import create_app


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


def _unencrypted_self_settings() -> dict[str, object]:
    return {
        **_rehearsal_settings(),
        "LEXIGAZE_STORAGE_ENCRYPTED": "0",
        "LEXIGAZE_UNENCRYPTED_SELF_DEVELOPMENT": "1",
        "LEXIGAZE_DATA_RETENTION_DAYS": "0",
        "LEXIGAZE_DATA_RETENTION_POLICY": "manual_until_researcher_deletes",
    }


def _consent_payload(
    invite_code: str,
    *,
    retain_reading_video: bool = False,
) -> dict[str, object]:
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
        "optional_scopes": {
            READING_VIDEO_SCOPE: retain_reading_video,
        },
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


def _assessment_viewport() -> dict[str, int]:
    return {"width_px": 1280, "height_px": 800}


TEST_MODEL_ARTIFACT_SHA256 = "a" * 64


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
        "jpeg_quality": 0.8,
        "mirror_applied": False,
        "facing_mode": "user",
    }


def _validation_samples(offset: float = 10.0) -> list[dict[str, object]]:
    targets = validation_target_definitions()
    return [
        {
            "target_id": target["target_id"],
            "target_x_px": x,
            "target_y_px": y,
            "target_x_norm": target["target_x_norm"],
            "target_y_norm": target["target_y_norm"],
            "prediction_success": True,
            "predicted_x_px": x + offset + repeat,
            "predicted_y_px": y + offset - repeat,
        }
        for target in targets
        for x, y in [
            (
                float(
                    int(target["target_x_viewport_fraction"] * 1280 + 0.5)
                ),
                float(
                    int(target["target_y_viewport_fraction"] * 800 + 0.5)
                ),
            )
        ]
        for repeat in range(3)
    ]


def _issue_prediction_receipts(
    store: ParticipantStudyStore,
    session_id: str,
    access_token: str,
    *,
    phase: str,
    samples: list[dict[str, object]] | None = None,
    model_artifact_sha256: str = TEST_MODEL_ARTIFACT_SHA256,
) -> list[str]:
    issued: list[str] = []
    public = store.get_session(session_id, access_token)
    model_name = str(public["linked_data"]["model_name"])
    for sample in samples or _validation_samples():
        challenge = store.prepare_general_prediction_receipt(
            session_id,
            access_token,
            phase=phase,
            target_id=str(sample["target_id"]),
            model_name=model_name,
            model_artifact_sha256=model_artifact_sha256,
            viewport=_assessment_viewport(),
        )
        predicted_x = float(sample.get("predicted_x_px", 0.0))
        predicted_y = float(sample.get("predicted_y_px", 0.0))
        success = sample.get("prediction_success") is True
        response = {
            "ok": success,
            "capture_contract_check": {
                "status": "compatible",
                "compatible": True,
                "reasons": [],
                "warnings": [],
            },
        }
        if success:
            response.update(
                {
                    "screen_xy_px": [predicted_x, predicted_y],
                    "screen_xy_norm": [
                        predicted_x / 1280.0 * 2.0 - 1.0,
                        predicted_y / 800.0 * 2.0 - 1.0,
                    ],
                }
            )
        else:
            response.update(
                {
                    "failure_stage": "attributable_sensor_failure",
                    "failure_code": "no_face_detected",
                    "error": "no face detected in frame",
                }
            )
        receipt = store.issue_general_prediction_receipt(
            session_id,
            access_token,
            challenge=challenge,
            model_artifact_sha256_after=model_artifact_sha256,
            capture_contract=_capture_contract(),
            prediction_response=response,
            prediction_status=200 if response["ok"] is True else 400,
        )
        issued.append(receipt["token"])
    return issued


def _record_prediction_receipt_validation(
    store: ParticipantStudyStore,
    session_id: str,
    access_token: str,
    *,
    phase: str,
    samples: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    receipts = _issue_prediction_receipts(
        store,
        session_id,
        access_token,
        phase=phase,
        samples=samples,
    )
    return store.record_general_validation(
        session_id,
        access_token,
        phase=phase,
        prediction_receipts=receipts,
        model_artifact_sha256=TEST_MODEL_ARTIFACT_SHA256,
    )


class GeneralCollectionDesignTests(unittest.TestCase):
    def test_frozen_design_and_bank_are_internally_consistent(self) -> None:
        audit = validate_general_design()
        self.assertEqual(audit["passage_count"], 12)
        self.assertEqual(audit["passage_family_count"], 12)
        self.assertEqual(audit["probe_count"], 96)
        self.assertEqual(
            audit["protocol_sha256"],
            "7c4b25bb306b68bb2a2ee5f34217a67aace0de6778fb3f1ed9b462741a0a26b9",
        )
        self.assertRegex(audit["bank_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(audit["validation_target_count"], 5)
        self.assertRegex(
            audit["gaze_measurement_contract_sha256"],
            r"^[0-9a-f]{64}$",
        )

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

    def test_unencrypted_self_development_is_explicit_and_never_promotable(self) -> None:
        settings = _unencrypted_self_settings()
        status = activation_status(settings)
        self.assertTrue(status["rehearsal_ready"])
        self.assertTrue(status["rehearsal_self_only"])
        self.assertEqual(
            status["storage_security"],
            "unencrypted_self_development",
        )
        self.assertEqual(
            status["retention_policy"],
            "manual_until_researcher_deletes",
        )
        self.assertIsNone(status["retention_days"])
        self.assertFalse(status["formal_promotion_allowed"])
        self.assertIn(
            READING_VIDEO_SCOPE,
            {item["id"] for item in public_protocol(settings)["optional_scopes"]},
        )
        self.assertNotIn(
            READING_VIDEO_SCOPE,
            {
                item["id"]
                for item in public_protocol(_rehearsal_settings())["optional_scopes"]
            },
        )
        self.assertFalse(status["pilot_ready"])

        unacknowledged = dict(settings)
        unacknowledged["LEXIGAZE_UNENCRYPTED_SELF_DEVELOPMENT"] = "0"
        rejected = activation_status(unacknowledged)
        self.assertFalse(rejected["rehearsal_ready"])
        self.assertIn(
            "rehearsal_storage_policy_not_acknowledged",
            rejected["rehearsal_missing_requirements"],
        )

    def test_unencrypted_self_development_allows_only_one_pair(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-self-dev-") as name:
            store = ParticipantStudyStore(
                Path(name),
                settings=_unencrypted_self_settings(),
            )
            pair = store.create_collection_invite_pairs(1)[0]
            with self.assertRaisesRegex(Exception, "exactly one"):
                store.create_collection_invite_pairs(1)
            enrolled = store.enroll(
                _consent_payload(pair["visits"][0]["invite_code"])
            )
            session = store.get_session(
                enrolled["study_session_id"],
                enrolled["access_token"],
            )
            self.assertEqual(
                session["data_governance"]["storage_security"],
                "unencrypted_self_development",
            )
            self.assertTrue(session["data_governance"]["self_only"])
            self.assertFalse(
                session["data_governance"]["formal_promotion_allowed"]
            )

    def test_unused_invite_can_rotate_without_creating_a_second_pair(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-invite-rotation-") as name:
            root = Path(name)
            store = ParticipantStudyStore(
                root,
                settings=_unencrypted_self_settings(),
            )
            pair = store.create_collection_invite_pairs(1)[0]
            first_code = pair["visits"][0]["invite_code"]
            second_code = pair["visits"][1]["invite_code"]

            rotated_visit_two = store.rotate_unused_collection_invite(
                pair["pair_id"], 2
            )
            self.assertNotEqual(rotated_visit_two["invite_code"], second_code)
            self.assertEqual(rotated_visit_two["code_rotation_count"], 1)
            with self.assertRaisesRegex(Exception, "invalid or already-used"):
                store.enroll(_consent_payload(second_code))

            rotated_visit_one = store.rotate_unused_collection_invite(
                pair["pair_id"], 1
            )
            self.assertNotEqual(rotated_visit_one["invite_code"], first_code)
            with self.assertRaisesRegex(Exception, "invalid or already-used"):
                store.enroll(_consent_payload(first_code))
            visit_one = store.enroll(
                _consent_payload(rotated_visit_one["invite_code"])
            )
            with self.assertRaisesRegex(Exception, "used invitation"):
                store.rotate_unused_collection_invite(pair["pair_id"], 1)
            registry_path = (
                root
                / "data"
                / "participant_studies"
                / "lexigaze-reader-pilot"
                / "rehearsals"
                / "collection_invites.json"
            )
            registry_text = registry_path.read_text(encoding="utf-8")
            self.assertNotIn(second_code, registry_text)
            self.assertNotIn(rotated_visit_two["invite_code"], registry_text)
            self.assertNotIn(first_code, registry_text)
            self.assertNotIn(rotated_visit_one["invite_code"], registry_text)
            registry = json.loads(registry_text)
            self.assertEqual(len(registry["invites"]), 2)
            rotated_record = next(
                item
                for item in registry["invites"]
                if item["visit_index"] == 2
            )
            self.assertEqual(rotated_record["code_rotation_count"], 1)
            self.assertEqual(len(rotated_record["code_rotation_history"]), 1)
            first_record = next(
                item
                for item in registry["invites"]
                if item["visit_index"] == 1
            )
            self.assertEqual(first_record["code_rotation_count"], 1)
            self.assertTrue(visit_one["study_session_id"].startswith("ST-"))


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
    def rehearsal_settings(self) -> dict[str, object]:
        return _rehearsal_settings()

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-general-store-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.store = ParticipantStudyStore(
            self.root,
            settings=self.rehearsal_settings(),
        )
        self.assertTrue(self.store.activation["rehearsal_ready"])
        self.pair = self.store.create_collection_invite_pairs(1)[0]
        self.enrolled = self.store.enroll(
            _consent_payload(self.pair["visits"][0]["invite_code"])
        )
        self.session_id = self.enrolled["study_session_id"]
        self.token = self.enrolled["access_token"]

    def _set_visit_one_completion(
        self,
        *,
        used_at: datetime,
        completed_at: datetime | None,
        duplicate_completion_event: bool = False,
    ) -> None:
        session_path = self.store._session_path(  # noqa: SLF001 - fixture setup
            self.session_id,
            "rehearsal",
        )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["state"] = "completed"
        session["events"] = [
            event
            for event in session.get("events", [])
            if event.get("event") != "general_collection_completed"
        ]
        if completed_at is not None:
            completion = {
                "at_utc": completed_at.astimezone(UTC).isoformat(),
                "event": "general_collection_completed",
            }
            session["events"].append(completion)
            if duplicate_completion_event:
                session["events"].append(dict(completion))
        self.store._write(session_path, session)  # noqa: SLF001 - fixture setup

        registry_path = (
            self.store._study_root("rehearsal")  # noqa: SLF001 - fixture setup
            / "collection_invites.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        first = next(
            item for item in registry["invites"] if item["visit_index"] == 1
        )
        first["used_at_utc"] = used_at.astimezone(UTC).isoformat()
        registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _enroll_visit_two(self) -> dict[str, object]:
        return self.store.enroll(
            _consent_payload(self.pair["visits"][1]["invite_code"])
        )

    def _prepare_collection(self, *, record_start_validation: bool = True) -> None:
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
            {
                "passed": True,
                "test_fixture": True,
                "model_artifact_sha256": "a" * 64,
            },
            model_name="general-test-model",
            model_artifact_sha256="a" * 64,
        )
        self.store.start_general_collection(
            self.session_id,
            self.token,
            assessment_viewport=_assessment_viewport(),
            model_artifact_sha256="a" * 64,
        )
        if record_start_validation:
            self._record_receipt_validation(phase="start")

    def _record_receipt_validation(
        self,
        *,
        phase: str,
        samples: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return _record_prediction_receipt_validation(
            self.store,
            self.session_id,
            self.token,
            phase=phase,
            samples=samples,
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

    def test_visit_two_interval_is_anchored_to_completion_event(self) -> None:
        now = datetime.now(UTC)
        self._set_visit_one_completion(
            used_at=now - timedelta(hours=20),
            completed_at=now - timedelta(hours=17),
        )
        with self.assertRaisesRegex(Exception, "earlier than"):
            self._enroll_visit_two()

    def test_visit_two_accepts_completion_inside_window_even_if_invite_is_old(self) -> None:
        now = datetime.now(UTC)
        self._set_visit_one_completion(
            used_at=now - timedelta(hours=80),
            completed_at=now - timedelta(hours=71),
        )
        enrolled = self._enroll_visit_two()
        self.assertEqual(enrolled["mode"], "rehearsal")

    def test_visit_two_fails_closed_without_one_completion_event(self) -> None:
        now = datetime.now(UTC)
        for completed_at, duplicate in ((None, False), (now - timedelta(hours=20), True)):
            with self.subTest(completed_at=completed_at, duplicate=duplicate):
                self._set_visit_one_completion(
                    used_at=now - timedelta(hours=20),
                    completed_at=completed_at,
                    duplicate_completion_event=duplicate,
                )
                with self.assertRaisesRegex(Exception, "completion timestamp"):
                    self._enroll_visit_two()

    def test_visit_two_system_check_requires_same_coarse_device_policy(self) -> None:
        self.store.record_general_profile(self.session_id, self.token, _profile())
        self.store.record_general_system_check(
            self.session_id,
            self.token,
            _system_profile(),
        )
        now = datetime.now(UTC)
        self._set_visit_one_completion(
            used_at=now - timedelta(hours=20),
            completed_at=now - timedelta(hours=20),
        )
        visit_two = self._enroll_visit_two()
        visit_two_id = str(visit_two["study_session_id"])
        visit_two_token = str(visit_two["access_token"])
        self.store.record_general_profile(visit_two_id, visit_two_token, _profile())
        mismatched = _system_profile()
        mismatched["device"] = {
            **dict(mismatched["device"]),
            "browser_family": "firefox",
        }
        with self.assertRaisesRegex(Exception, "same device class and browser family"):
            self.store.record_general_system_check(
                visit_two_id,
                visit_two_token,
                mismatched,
            )

    def test_visit_two_geometry_differences_are_diagnostic_only(self) -> None:
        self.store.record_general_profile(self.session_id, self.token, _profile())
        self.store.record_general_system_check(
            self.session_id,
            self.token,
            _system_profile(),
        )
        now = datetime.now(UTC)
        self._set_visit_one_completion(
            used_at=now - timedelta(hours=20),
            completed_at=now - timedelta(hours=20),
        )
        visit_two = self._enroll_visit_two()
        visit_two_id = str(visit_two["study_session_id"])
        visit_two_token = str(visit_two["access_token"])
        self.store.record_general_profile(visit_two_id, visit_two_token, _profile())
        changed_geometry = _system_profile()
        changed_geometry["device"] = {
            **dict(changed_geometry["device"]),
            "viewport_width": 1440,
            "device_pixel_ratio_bucket": "1_5",
            "camera_width": 1280,
            "estimated_camera_fps_band": "30_60",
        }
        recorded = self.store.record_general_system_check(
            visit_two_id,
            visit_two_token,
            changed_geometry,
        )
        comparison = recorded["quality"]["general_system_check"][
            "paired_visit_device_comparison"
        ]
        self.assertTrue(comparison["policy_match"])
        self.assertFalse(comparison["diagnostics"]["viewport_size"])
        self.assertFalse(comparison["diagnostics"]["camera_resolution"])

    def test_legacy_invite_assignment_needs_no_additive_gaze_contract_field(self) -> None:
        status = self.store.get_session(self.session_id, self.token)
        self.assertNotIn(
            "gaze_measurement_contract",
            status["collection_assignment"],
        )
        self._prepare_collection()
        started = self.store.get_session(self.session_id, self.token)
        self.assertEqual(
            started["general_collection"]["phase"],
            "reading_ready",
        )
        self.assertRegex(
            started["general_collection"]["gaze_measurement_contract"]["sha256"],
            r"^[0-9a-f]{64}$",
        )
        frozen = started["general_collection"]["gaze_measurement_contract"]
        self.assertEqual(
            frozen["sha256"],
            canonical_sha256(frozen["contract"]),
        )
        self.assertEqual(
            started["general_collection"]["assessment_viewport"],
            _assessment_viewport(),
        )

    def test_validation_uses_session_contract_when_current_file_drifts(self) -> None:
        self._prepare_collection(record_start_validation=False)
        before = self.store.get_session(self.session_id, self.token)
        frozen = before["general_collection"]["gaze_measurement_contract"]
        frozen_contract_samples = _validation_samples()
        with (
            patch(
                "core.participant_study.store.load_participant_gaze_measurement_contract",
                side_effect=AssertionError("current contract must not be reloaded"),
            ),
            patch(
                "core.participant_study.general_collection."
                "load_participant_gaze_measurement_contract",
                side_effect=AssertionError("current contract must not be reloaded"),
            ),
        ):
            resumed = self.store.start_general_collection(
                self.session_id,
                self.token,
                assessment_viewport=_assessment_viewport(),
                model_artifact_sha256="a" * 64,
            )
            validated = self._record_receipt_validation(
                phase="start",
                samples=frozen_contract_samples,
            )
        self.assertEqual(
            resumed["general_collection"]["gaze_measurement_contract"],
            frozen,
        )
        self.assertEqual(
            validated["general_collection"]["phase"],
            "reading_ready",
        )
        self.assertEqual(
            validated["general_collection"]["gaze_measurement_contract"],
            frozen,
        )
        self.assertEqual(
            validated["general_collection"]["validations"]["start"][
                "gaze_measurement_contract_sha256"
            ],
            frozen["sha256"],
        )

    def test_start_rejects_viewport_drift_from_system_check(self) -> None:
        self.store.record_general_profile(self.session_id, self.token, _profile())
        self.store.record_general_system_check(
            self.session_id,
            self.token,
            _system_profile(),
        )
        self.store.start_calibration(self.session_id, self.token, "GAZE-DRIFT")
        self.store.complete_calibration(
            self.session_id,
            self.token,
            {
                "passed": True,
                "test_fixture": True,
                "model_artifact_sha256": "a" * 64,
            },
            model_name="viewport-test-model",
            model_artifact_sha256="a" * 64,
        )
        with self.assertRaisesRegex(Exception, "viewport changed since system check"):
            self.store.start_general_collection(
                self.session_id,
                self.token,
                assessment_viewport={"width_px": 1279, "height_px": 800},
                model_artifact_sha256="a" * 64,
            )

    def test_collection_start_rejects_post_training_model_artifact_replacement(
        self,
    ) -> None:
        self.store.record_general_profile(self.session_id, self.token, _profile())
        self.store.record_general_system_check(
            self.session_id,
            self.token,
            _system_profile(),
        )
        self.store.start_calibration(
            self.session_id,
            self.token,
            "GAZE-ARTIFACT-FREEZE",
        )
        self.store.complete_calibration(
            self.session_id,
            self.token,
            {
                "passed": True,
                "test_fixture": True,
                "model_artifact_sha256": "a" * 64,
            },
            model_name="artifact-freeze-test-model",
            model_artifact_sha256="a" * 64,
        )

        with self.assertRaisesRegex(Exception, "artifact changed"):
            self.store.start_general_collection(
                self.session_id,
                self.token,
                assessment_viewport=_assessment_viewport(),
                model_artifact_sha256="b" * 64,
            )

        public = self.store.get_session(self.session_id, self.token)
        self.assertEqual(public["state"], "calibration_complete")
        self.assertIsNone(
            public.get("general_collection", {}).get("assessment_id")
        )

    def test_validation_rejects_tampered_session_contract_hash(self) -> None:
        self._prepare_collection(record_start_validation=False)
        session_path = next(self.root.rglob("session.json"))
        stored = json.loads(session_path.read_text(encoding="utf-8"))
        stored["general_collection"]["gaze_measurement_contract"]["contract"][
            "status"
        ] = "tampered-after-start"
        session_path.write_text(
            json.dumps(stored, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(Exception, "contract hash mismatch"):
            self.store.record_general_validation(
                self.session_id,
                self.token,
                phase="start",
                samples=_validation_samples(),
            )

    def test_in_progress_legacy_contract_summary_fails_closed(self) -> None:
        self._prepare_collection(record_start_validation=False)
        session_path = next(self.root.rglob("session.json"))
        stored = json.loads(session_path.read_text(encoding="utf-8"))
        stored["general_collection"]["gaze_measurement_contract"].pop("contract")
        session_path.write_text(
            json.dumps(stored, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(Exception, "contract is incomplete"):
            self.store.record_general_validation(
                self.session_id,
                self.token,
                phase="start",
                samples=_validation_samples(),
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
        completed = self._record_receipt_validation(
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
        self.assertEqual(
            manifest["storage_governance"]["security_modes"],
            [self.store.activation["storage_security"]],
        )
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

    def test_telemetry_rejects_non_frozen_viewport(self) -> None:
        self._prepare_collection()
        current = self.store.begin_general_round(self.session_id, self.token)
        with self.assertRaisesRegex(Exception, "frozen assessment viewport"):
            self.store.record_general_telemetry_batch(
                self.session_id,
                self.token,
                {
                    "batch_id": "B-VIEWPORT1",
                    "passage_id": current["passage"]["passage_id"],
                    "viewport": {"width_px": 800, "height_px": 1280},
                    "samples": [
                        {
                            "monotonic_elapsed_ms": 0,
                            "prediction_success": False,
                            "coarse_failure_code": "prediction_failed",
                        }
                    ],
                },
            )

    def test_viewport_failure_sample_persists_gaze_integrity_failure(self) -> None:
        self._prepare_collection()
        current = self.store.begin_general_round(self.session_id, self.token)
        self.store.record_general_telemetry_batch(
            self.session_id,
            self.token,
            {
                "batch_id": "B-VIEWPORT2",
                "passage_id": current["passage"]["passage_id"],
                "viewport": _assessment_viewport(),
                "samples": [
                    {
                        "monotonic_elapsed_ms": 0,
                        "prediction_success": False,
                        "coarse_failure_code": "viewport_contract_mismatch",
                    }
                ],
            },
        )
        status = self.store.get_session(self.session_id, self.token)
        integrity = status["general_collection"]["gaze_integrity"]
        self.assertFalse(integrity["eligible"])
        self.assertIn(
            "assessment_viewport_changed_during_reading",
            integrity["reasons"],
        )
        for round_index in range(6):
            passage_id = current["passage"]["passage_id"]
            self.store.open_general_word_reviews(
                self.session_id,
                self.token,
                passage_id=passage_id,
            )
            self.store.record_general_round(
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
        completed = self._record_receipt_validation(
            phase="end",
            samples=_validation_samples(offset=14.0),
        )
        quality = completed["quality"]["general_collection"]
        self.assertEqual(quality["gaze_quality_band"], "behavioral_only")
        self.assertFalse(quality["gaze_integrity_eligible"])

    def test_reading_active_refresh_invalidates_contiguous_sampling(self) -> None:
        self._prepare_collection()
        current = self.store.begin_general_round(self.session_id, self.token)
        resumed = self.store.begin_general_round(self.session_id, self.token)
        self.assertEqual(resumed["round_number"], current["round_number"])
        for round_index in range(6):
            passage_id = current["passage"]["passage_id"]
            self.store.open_general_word_reviews(
                self.session_id,
                self.token,
                passage_id=passage_id,
            )
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
        completed = self._record_receipt_validation(
            phase="end",
            samples=_validation_samples(offset=14.0),
        )
        quality = completed["quality"]["general_collection"]
        self.assertEqual(quality["gaze_quality_band"], "behavioral_only")
        self.assertFalse(quality["gaze_integrity_eligible"])
        self.assertFalse(quality["telemetry_segments_contiguous"])
        self.assertIsNone(quality["effective_sampling_hz"])
        self.assertEqual(quality["raw_effective_sampling_hz"], 0.0)


class UnencryptedGeneralCollectionStoreTests(GeneralCollectionStoreTests):
    def rehearsal_settings(self) -> dict[str, object]:
        return _unencrypted_self_settings()


class UnencryptedReadingVideoStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-reading-video-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.store = ParticipantStudyStore(
            self.root,
            settings=_unencrypted_self_settings(),
        )
        self.pair = self.store.create_collection_invite_pairs(1)[0]
        self.enrolled = self.store.enroll(
            _consent_payload(
                self.pair["visits"][0]["invite_code"],
                retain_reading_video=True,
            )
        )
        self.session_id = self.enrolled["study_session_id"]
        self.token = self.enrolled["access_token"]
        self.store.record_general_profile(self.session_id, self.token, _profile())
        self.store.record_general_system_check(
            self.session_id,
            self.token,
            _system_profile(),
        )
        self.store.start_calibration(self.session_id, self.token, "GAZE-VIDEO")
        self.store.complete_calibration(
            self.session_id,
            self.token,
            {
                "passed": True,
                "test_fixture": True,
                "model_artifact_sha256": "a" * 64,
            },
            model_name="reading-video-test-model",
            model_artifact_sha256="a" * 64,
        )
        self.store.start_general_collection(
            self.session_id,
            self.token,
            assessment_viewport=_assessment_viewport(),
            model_artifact_sha256="a" * 64,
        )
        _record_prediction_receipt_validation(
            self.store,
            self.session_id,
            self.token,
            phase="start",
        )

    def test_video_is_immutable_required_before_probes_and_index_only_exported(self) -> None:
        current = self.store.begin_general_round(self.session_id, self.token)
        passage_id = current["passage"]["passage_id"]
        with self.assertRaisesRegex(Exception, "must be stored"):
            self.store.open_general_word_reviews(
                self.session_id,
                self.token,
                passage_id=passage_id,
            )

        video = b"\x1aE\xdf\xa3lexigaze-self-development-video"
        kwargs = {
            "recording_id": "VID-0123456789ABCDEF01234567",
            "passage_id": passage_id,
            "round_number": current["round_number"],
            "duration_ms": 30_000,
            "mime_type": "video/webm;codecs=vp8",
            "payload": video,
        }
        stored = self.store.record_general_reading_video(
            self.session_id,
            self.token,
            **kwargs,
        )
        self.assertFalse(stored["idempotent"])
        self.assertEqual(stored["reading_video"]["audio_track_count"], 0)
        self.assertEqual(
            stored["reading_video"]["dataset_role"],
            "self_development_only_not_confirmation",
        )
        duplicate = self.store.record_general_reading_video(
            self.session_id,
            self.token,
            **kwargs,
        )
        self.assertTrue(duplicate["idempotent"])
        with self.assertRaisesRegex(Exception, "reused with new content"):
            self.store.record_general_reading_video(
                self.session_id,
                self.token,
                **{**kwargs, "payload": video + b"changed"},
            )
        with self.assertRaisesRegex(Exception, "already has a reading video"):
            self.store.record_general_reading_video(
                self.session_id,
                self.token,
                **{**kwargs, "recording_id": "VID-AAAAAAAAAAAAAAAAAAAAAAAA"},
            )

        session_path = next(self.root.rglob("session.json"))
        interrupted_session = json.loads(session_path.read_text(encoding="utf-8"))
        interrupted_session["general_collection"]["reading_videos"] = []
        session_path.write_text(
            json.dumps(interrupted_session, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        probes = self.store.open_general_word_reviews(
            self.session_id,
            self.token,
            passage_id=passage_id,
        )
        self.assertEqual(len(probes["probes"]), 8)
        recovered = self.store.get_session(self.session_id, self.token)
        self.assertEqual(len(recovered["general_collection"]["reading_videos"]), 1)
        media_paths = list(
            self.root.rglob("collection/reading_video/R01.webm")
        )
        self.assertEqual(len(media_paths), 1)
        metadata = list(self.root.rglob("collection/reading_video/R01.json"))
        self.assertEqual(len(metadata), 1)

        export_path = self.root / "private-export"
        manifest = export_bundle(
            self.root,
            export_path,
            include_incomplete=True,
        )
        self.assertEqual(
            manifest["files"]["reading_video_index.csv"]["row_count"],
            1,
        )
        self.assertEqual(manifest["source_reading_videos"]["count"], 1)
        self.assertFalse(
            manifest["source_reading_videos"]["raw_media_files_exported"]
        )
        self.assertEqual(list(export_path.glob("*.webm")), [])

    def test_video_bounds_and_mime_are_enforced(self) -> None:
        current = self.store.begin_general_round(self.session_id, self.token)
        passage_id = current["passage"]["passage_id"]
        base = {
            "recording_id": "VID-0123456789ABCDEF01234567",
            "passage_id": passage_id,
            "round_number": current["round_number"],
            "duration_ms": 30_000,
            "mime_type": "video/webm",
            "payload": b"video",
        }
        with self.assertRaisesRegex(Exception, "duration"):
            self.store.record_general_reading_video(
                self.session_id,
                self.token,
                **{**base, "duration_ms": 19_999},
            )
        with self.assertRaisesRegex(Exception, "MIME"):
            self.store.record_general_reading_video(
                self.session_id,
                self.token,
                **{**base, "mime_type": "video/quicktime"},
            )

    def test_multipart_route_preserves_video_and_rejects_mime_mismatch(self) -> None:
        current = self.store.begin_general_round(self.session_id, self.token)
        metadata = {
            "recording_id": "VID-ABCDEF0123456789ABCDEF01",
            "passage_id": current["passage"]["passage_id"],
            "round_number": current["round_number"],
            "duration_ms": 30_000,
            "mime_type": "video/webm;codecs=vp8",
        }
        app = create_app(
            {
                "TESTING": True,
                "LEXIGAZE_BLUEPRINTS": ("study",),
                "LEXIGAZE_STUDY_ROOT": str(self.root),
                "LEXIGAZE_PUBLIC_STUDY_MODE": "1",
                **_unencrypted_self_settings(),
            }
        )
        client = app.test_client()
        url = (
            f"/api/study/sessions/{self.session_id}/general/reading-video"
        )
        headers = {"Authorization": f"Bearer {self.token}"}
        mismatch = client.post(
            url,
            headers=headers,
            data={
                "metadata": json.dumps(metadata),
                "reading_video": (
                    io.BytesIO(b"video"),
                    "R01.mp4",
                    "video/mp4",
                ),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(mismatch.status_code, 400)
        stored = client.post(
            url,
            headers=headers,
            data={
                "metadata": json.dumps(metadata),
                "reading_video": (
                    io.BytesIO(b"\x1aE\xdf\xa3route-video"),
                    "R01.webm",
                    "video/webm",
                ),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(stored.status_code, 200, stored.get_data(as_text=True))
        self.assertFalse(stored.get_json()["idempotent"])


class ReadingVideoScopeBoundaryTests(unittest.TestCase):
    def test_encrypted_general_rehearsal_cannot_accept_self_video_scope(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-video-scope-") as name:
            store = ParticipantStudyStore(Path(name), settings=_rehearsal_settings())
            pair = store.create_collection_invite_pairs(1)[0]
            with self.assertRaises(StudyValidationError):
                store.enroll(
                    _consent_payload(
                        pair["visits"][0]["invite_code"],
                        retain_reading_video=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
