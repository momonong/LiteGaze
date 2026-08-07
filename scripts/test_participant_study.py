"""Participant-study consent, lifecycle, privacy, and public-surface tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from core.cognitive_inspector.adaptive import PASSAGE_BY_ID
from core.gaze_core.sample_store import create_session, safe_session_dir
from core.participant_study import ParticipantStudyStore
from core.participant_study.protocol import activation_status, load_protocol
from web import create_app


def _consent_payload(protocol: dict, *, mode: str = "dry_run") -> dict:
    return {
        "mode": mode,
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


def _pilot_settings() -> dict[str, object]:
    return {
        "LEXIGAZE_STUDY_MODE": "pilot",
        "LEXIGAZE_PUBLIC_STUDY_MODE": "1",
        "LEXIGAZE_ETHICS_STATUS": "exempt_determination",
        "LEXIGAZE_ETHICS_REFERENCE": "TEST-ETHICS-001",
        "LEXIGAZE_INVESTIGATOR_NAME": "Test Investigator",
        "LEXIGAZE_INVESTIGATOR_EMAIL": "investigator@example.invalid",
        "LEXIGAZE_PARTICIPANT_RIGHTS_CONTACT": "rights@example.invalid",
        "LEXIGAZE_RESEARCHER_API_KEY": "researcher-test-key",
        "LEXIGAZE_ADAPTIVE_SIGNING_KEY": "s" * 48,
        "LEXIGAZE_EXTERNAL_ANCHOR_ID": "test-anchor",
        "LEXIGAZE_EXTERNAL_ANCHOR_AUTHORIZED": "1",
        "LEXIGAZE_NETWORK_PROCESSOR": "test-https-reverse-proxy",
        "LEXIGAZE_NETWORK_PROCESSOR_APPROVED": "1",
        "LEXIGAZE_REQUEST_BODY_LOGGING_DISABLED": "1",
        "LEXIGAZE_STORAGE_ENCRYPTED": "1",
        "LEXIGAZE_PUBLIC_BASE_URL": "https://study.example.invalid",
        "LEXIGAZE_DATA_LOCATION": "encrypted-test-volume",
        "LEXIGAZE_DATA_RETENTION_DAYS": "30",
        "LEXIGAZE_RAW_FRAME_RETENTION_HOURS": "1",
    }


def _approved_protocol() -> dict:
    protocol = load_protocol()
    protocol["collection_status"] = "approved_for_pilot"
    return protocol


class ParticipantProtocolTests(unittest.TestCase):
    def test_default_protocol_is_dry_run_only_and_signing_key_is_a_gate(self) -> None:
        settings = _pilot_settings()
        settings["LEXIGAZE_ADAPTIVE_SIGNING_KEY"] = "short"
        status = activation_status(settings, protocol=_approved_protocol())
        self.assertFalse(status["pilot_ready"])
        self.assertIn(
            "adaptive_signing_key_missing_or_short",
            status["missing_requirements"],
        )
        self.assertFalse(activation_status({})["pilot_ready"])
        self.assertTrue(activation_status({})["dry_run_ready"])

    def test_complete_pilot_configuration_can_issue_one_time_invite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-study-") as temp_name:
            store = ParticipantStudyStore(
                Path(temp_name),
                settings=_pilot_settings(),
                protocol=_approved_protocol(),
            )
            self.assertTrue(store.activation["pilot_ready"])
            invite = store.create_invites(1)[0]
            payload = _consent_payload(store.protocol, mode="pilot")
            payload["invite_code"] = invite
            first = store.enroll(payload)
            self.assertEqual(first["mode"], "pilot")
            with self.assertRaisesRegex(Exception, "already-used"):
                store.enroll(payload)


class ParticipantStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-study-store-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.protocol = load_protocol()
        self.store = ParticipantStudyStore(self.root, settings={})

    def test_consent_requires_every_statement_and_correct_comprehension(self) -> None:
        payload = _consent_payload(self.protocol)
        missing = self.protocol["required_consent_statements"][0]["id"]
        payload["consent_statements"][missing] = False
        with self.assertRaisesRegex(Exception, "missing"):
            self.store.enroll(payload)

        payload = _consent_payload(self.protocol)
        check = self.protocol["comprehension_checks"][0]
        payload["comprehension_answers"][check["id"]] = "wrong"
        with self.assertRaisesRegex(Exception, "failed"):
            self.store.enroll(payload)

    def test_direct_identifiers_are_ignored_and_dry_run_captures_no_media(self) -> None:
        payload = _consent_payload(self.protocol)
        payload.update({"name": "Alice Example", "email": "alice@example.invalid"})
        enrolled = self.store.enroll(payload)
        session_path = next(self.root.rglob("session.json"))
        stored_text = session_path.read_text(encoding="utf-8")
        self.assertNotIn("Alice", stored_text)
        self.assertNotIn("alice@example", stored_text)
        self.assertRegex(enrolled["participant_id"], r"^P-[A-F0-9]{12}$")

        session_id = enrolled["study_session_id"]
        token = enrolled["access_token"]
        for action in (
            "system_check",
            "calibration_start",
            "calibration_complete",
            "assessment_start",
            "assessment_complete",
        ):
            session = self.store.advance_dry_run(session_id, token, action)
        self.assertEqual(session["state"], "completed")
        self.assertEqual(list(self.root.rglob("*.jpg")), [])
        self.assertEqual(list(self.root.rglob("*.webm")), [])

    def test_withdrawal_replaces_entire_session_directory_with_tombstone(self) -> None:
        enrolled = self.store.enroll(_consent_payload(self.protocol))
        session_path = next(self.root.rglob("session.json"))
        extra = session_path.parent / "nested" / "private-observation.json"
        extra.parent.mkdir()
        extra.write_text('{"private": true}', encoding="utf-8")

        receipt = self.store.withdraw(
            enrolled["study_session_id"],
            withdrawal_code=enrolled["withdrawal_code"],
        )
        remaining = [
            path.relative_to(session_path.parent)
            for path in session_path.parent.rglob("*")
        ]
        self.assertEqual(remaining, [Path("session.json")])
        tombstone = json.loads(session_path.read_text(encoding="utf-8"))
        self.assertEqual(tombstone["state"], "withdrawn")
        self.assertEqual(
            receipt["withdrawal_receipt_id"], tombstone["withdrawal_receipt_id"]
        )


class ParticipantWebSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-study-web-")
        self.addCleanup(self.temp_dir.cleanup)
        self.app = create_app(
            {
                "TESTING": True,
                "LEXIGAZE_BLUEPRINTS": ("study", "inspector"),
                "LEXIGAZE_STUDY_ROOT": self.temp_dir.name,
                "LEXIGAZE_PUBLIC_STUDY_MODE": "1",
                "LEXIGAZE_ADAPTIVE_SIGNING_KEY": "t" * 48,
            }
        )
        self.client = self.app.test_client()

    def test_public_mode_allows_participant_pages_and_blocks_admin_apis(self) -> None:
        self.assertEqual(self.client.get("/study").status_code, 200)
        self.assertEqual(self.client.get("/study/assessment").status_code, 200)
        root = self.client.get("/")
        self.assertEqual(root.status_code, 302)
        self.assertTrue(root.headers["Location"].endswith("/study"))
        self.assertEqual(self.client.get("/api/sessions").status_code, 403)
        self.assertEqual(self.client.get("/api/inspector/reports").status_code, 403)
        self.assertEqual(
            self.client.post("/api/inspector/adaptive/start", json={}).status_code,
            403,
        )

    def test_study_responses_are_not_cacheable(self) -> None:
        response = self.client.get("/api/study/protocol")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")


class ParticipantCalibrationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-study-cal-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.settings = _pilot_settings()
        self.protocol = _approved_protocol()
        self.store = ParticipantStudyStore(
            self.root,
            settings=self.settings,
            protocol=self.protocol,
        )
        invite = self.store.create_invites(1)[0]
        payload = _consent_payload(self.protocol, mode="pilot")
        payload["invite_code"] = invite
        self.enrolled = self.store.enroll(payload)
        self.store.record_system_check(
            self.enrolled["study_session_id"],
            self.enrolled["access_token"],
            {
                "camera_api": True,
                "secure_context": True,
                "screen_size": True,
                "network": True,
            },
        )
        gaze = create_session(
            self.root,
            self.enrolled["participant_id"],
            study_metadata={
                "study_session_id": self.enrolled["study_session_id"],
            },
        )
        self.gaze_session_id = gaze["session_id"]
        self.store.start_calibration(
            self.enrolled["study_session_id"],
            self.enrolled["access_token"],
            self.gaze_session_id,
        )
        self.app = create_app(
            {
                "TESTING": True,
                "LEXIGAZE_BLUEPRINTS": ("study",),
                "LEXIGAZE_STUDY_ROOT": self.root,
                "LEXIGAZE_GAZE_ROOT": self.root,
            }
        )
        self.client = self.app.test_client()

    def _complete(self):
        return self.client.post(
            f"/api/study/sessions/{self.enrolled['study_session_id']}/calibration/complete",
            headers={"Authorization": f"Bearer {self.enrolled['access_token']}"},
            json={"gaze_session_id": self.gaze_session_id},
        )

    def test_quality_failure_deletes_the_entire_temporary_dataset(self) -> None:
        with (
            patch("web.routes.study.ParticipantStudyStore", return_value=self.store),
            patch(
                "web.routes.study.audit_participant_calibration",
                return_value={"passed": False, "reasons": ["coverage_failed"]},
            ),
            patch("web.routes.study.train_placeholder") as train,
        ):
            response = self._complete()
        self.assertEqual(response.status_code, 422)
        self.assertFalse(safe_session_dir(self.root, self.gaze_session_id).exists())
        self.assertFalse(train.called)
        status = self.store.get_session(
            self.enrolled["study_session_id"], self.enrolled["access_token"]
        )
        self.assertEqual(status["state"], "system_check_passed")
        self.assertNotIn("gaze_session_id", status["linked_data"])

    def test_interrupted_calibration_expires_and_returns_to_retry_state(self) -> None:
        future = datetime.now(UTC) + timedelta(hours=2)
        purged = self.store.enforce_expired_calibration_retention(now=future)
        self.assertEqual(purged, [self.enrolled["study_session_id"]])
        self.assertFalse(safe_session_dir(self.root, self.gaze_session_id).exists())
        status = self.store.get_session(
            self.enrolled["study_session_id"], self.enrolled["access_token"]
        )
        self.assertEqual(status["state"], "system_check_passed")
        self.assertTrue(status["quality"]["calibration_expiry"]["raw_data_purged"])

    def test_successful_personalization_is_cpu_only_and_purges_images(self) -> None:
        quality = {"passed": True, "reasons": [], "sample_count": 65}
        training_result = {
            "ok": True,
            "model_name": "participant-pilot-model",
            "training_device": "cpu",
            "best_val_px_error": 30.0,
            "validation_scheme": "participant_holdout",
        }
        with (
            patch("web.routes.study.ParticipantStudyStore", return_value=self.store),
            patch(
                "web.routes.study.audit_participant_calibration",
                return_value=quality,
            ),
            patch(
                "web.routes.study.train_placeholder",
                return_value=(training_result, 200),
            ) as train,
        ):
            response = self._complete()
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        training_payload = train.call_args.args[1]
        self.assertIs(training_payload["allow_cuda"], False)
        session_dir = safe_session_dir(
            self.root, self.gaze_session_id, require_exists=True
        )
        self.assertFalse((session_dir / "raw").exists())
        self.assertFalse((session_dir / "crop").exists())
        self.assertFalse((session_dir / "normalized_face").exists())


class ParticipantAdaptiveIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-study-adaptive-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.settings = _pilot_settings()
        self.protocol = _approved_protocol()
        self.store = ParticipantStudyStore(
            self.root,
            settings=self.settings,
            protocol=self.protocol,
        )
        invite = self.store.create_invites(1)[0]
        payload = _consent_payload(self.protocol, mode="pilot")
        payload["invite_code"] = invite
        self.enrolled = self.store.enroll(payload)
        session_id = self.enrolled["study_session_id"]
        token = self.enrolled["access_token"]
        self.store.record_system_check(
            session_id,
            token,
            {
                "camera_api": True,
                "secure_context": True,
                "screen_size": True,
                "network": True,
            },
        )
        self.store.start_calibration(session_id, token, "TEST-GAZE-SESSION")
        self.store.complete_calibration(
            session_id,
            token,
            {"passed": True, "simulated_test_fixture": True},
            model_name="test-participant-model",
        )
        config = {
            "TESTING": True,
            "LEXIGAZE_BLUEPRINTS": ("study", "inspector"),
            "LEXIGAZE_STUDY_ROOT": self.root,
            **self.settings,
        }
        self.app = create_app(config)
        self.client = self.app.test_client()
        self.headers = {"Authorization": f"Bearer {token}"}

    def _body(self, **extra):
        return {
            "study_session_id": self.enrolled["study_session_id"],
            "study_access_token": self.enrolled["access_token"],
            **extra,
        }

    def _answers(self, passage_id: str) -> dict[str, str]:
        return {
            item["question_id"]: item["answer"]
            for item in PASSAGE_BY_ID[passage_id]["questions"]
        }

    def test_signed_study_assessment_resumes_and_completes(self) -> None:
        with patch(
            "web.routes.participant_adaptive.ParticipantStudyStore",
            return_value=self.store,
        ):
            start = self.client.post(
                "/api/inspector/adaptive/start",
                headers=self.headers,
                json=self._body(),
            ).get_json()
            history = []
            duplicate_checked = False
            while True:
                score_payload = self._body(
                    assessment_id=start["assessment_id"],
                    round=start["round"],
                    round_token=start["round_token"],
                    passage_id=start["passage_id"],
                    responses=self._answers(start["passage_id"]),
                    metrics={
                        "wpm": 175.0,
                        "regression_rate": 0.05,
                        "data_quality_status": "good",
                    },
                )
                score_response = self.client.post(
                    "/api/inspector/adaptive/score",
                    headers=self.headers,
                    json=score_payload,
                )
                self.assertEqual(
                    score_response.status_code,
                    200,
                    score_response.get_data(as_text=True),
                )
                scored = score_response.get_json()
                self.assertNotIn("item_results", scored["round_result"])
                history.append(
                    {
                        "round": start["round"],
                        "passage_id": start["passage_id"],
                        "result_token": scored["result_token"],
                    }
                )
                if not duplicate_checked:
                    duplicate = self.client.post(
                        "/api/inspector/adaptive/score",
                        headers=self.headers,
                        json=score_payload,
                    )
                    self.assertEqual(duplicate.status_code, 409)
                    resumed = self.client.post(
                        "/api/inspector/adaptive/start",
                        headers=self.headers,
                        json=self._body(),
                    ).get_json()
                    self.assertEqual(len(resumed["resume_history"]), 1)
                    self.assertEqual(resumed["round"], 2)
                    start = resumed
                    duplicate_checked = True
                    continue
                next_response = self.client.post(
                    "/api/inspector/adaptive/next",
                    headers=self.headers,
                    json=self._body(
                        assessment_id=start["assessment_id"],
                        history=history,
                    ),
                )
                self.assertEqual(next_response.status_code, 200)
                next_round = next_response.get_json()
                if next_round.get("is_finished"):
                    break
                start = next_round

            report = self.client.post(
                "/api/inspector/adaptive/report",
                headers=self.headers,
                json=self._body(
                    assessment_id=start["assessment_id"],
                    history=history,
                    persist=True,
                ),
            )
        self.assertEqual(report.status_code, 200, report.get_data(as_text=True))
        payload = report.get_json()
        self.assertTrue(payload["study_completed"])
        self.assertIsNone(payload["report_path"])
        status = self.store.get_session(
            self.enrolled["study_session_id"], self.enrolled["access_token"]
        )
        self.assertEqual(status["state"], "completed")


if __name__ == "__main__":
    unittest.main()
