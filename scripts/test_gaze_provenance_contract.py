"""Regression tests for server-authoritative gaze capture provenance."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.gaze_core.model_registry import model_path
from core.gaze_core.sample_store import create_session
from core.participant_study import ParticipantStudyStore
from scripts.test_general_collection import (
    _assessment_viewport,
    _capture_contract,
    _consent_payload,
    _profile,
    _rehearsal_settings,
    _system_profile,
)
from web import create_app


class GazeProvenanceContractTests(unittest.TestCase):
    def test_linked_study_sample_uses_session_capture_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-gaze-provenance-") as name:
            root = Path(name)
            created = create_session(
                root,
                "participant-test",
                capture_run_id="capture-authoritative",
                capture_source="study-direct-frame",
                study_metadata={"study_session_id": "study-test"},
            )
            app = create_app(
                {
                    "TESTING": True,
                    "LEXIGAZE_BLUEPRINTS": ("gaze",),
                    "LEXIGAZE_GAZE_ROOT": root,
                    "LEXIGAZE_STUDY_ROOT": root,
                }
            )
            captured: dict = {}

            def fake_save_sample(_root: Path, payload: dict) -> tuple[dict, int]:
                captured.update(payload)
                return {"ok": True}, 200

            participant = {
                "state": "calibration_in_progress",
                "linked_data": {"gaze_session_id": created["session_id"]},
            }
            with (
                patch(
                    "web.routes.gaze._participant_session",
                    return_value=(object(), participant),
                ),
                patch(
                    "web.routes.gaze.save_sample",
                    side_effect=fake_save_sample,
                ),
            ):
                response = app.test_client().post(
                    "/api/gaze/sample",
                    json={
                        "session_id": created["session_id"],
                        "study_session_id": "study-test",
                        "phase": "forged-phase",
                        "point_index": 3,
                        "repeat_index": 0,
                        "target_x": 321.0,
                        "target_y": 456.0,
                        "target_x_norm": 0.99,
                        "target_y_norm": -0.99,
                        "collect_mode": "standard",
                        "collection_protocol": "forged-protocol",
                        "motion_block_id": "left",
                        "posture_condition": "right",
                        "distance_condition": "far",
                        "lighting_condition": "forged-lighting",
                        "capture_burst_id": "forged-burst",
                        "capture_run_id": "capture-forged",
                        "capture_source": "direct-frame",
                        "source_session_id": "forged-parent",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["capture_run_id"], "capture-authoritative")
        self.assertEqual(captured["capture_source"], "study-direct-frame")
        self.assertNotIn("source_session_id", captured)
        self.assertEqual(captured["phase"], "calibration")
        self.assertEqual(captured["point_index"], 3)
        self.assertEqual(captured["repeat_index"], 0)
        self.assertAlmostEqual(captured["target_x_norm"], -0.84)
        self.assertAlmostEqual(captured["target_y_norm"], 0.0)
        self.assertEqual(captured["target_x"], 321.0)
        self.assertEqual(captured["target_y"], 456.0)
        self.assertEqual(captured["collect_mode"], "motion_robust")
        self.assertEqual(captured["collection_protocol"], "motion-diverse-v1")
        self.assertEqual(captured["motion_block_id"], "left")
        self.assertEqual(captured["posture_condition"], "left")
        self.assertEqual(captured["distance_condition"], "nominal")
        self.assertEqual(captured["lighting_condition"], "ambient")
        self.assertEqual(
            captured["capture_burst_id"],
            f"{created['session_id']}:left:r0",
        )
        self.assertEqual(
            captured["calibration_label_authority"],
            "server_frozen_participant_motion_calibration_v1",
        )
        self.assertEqual(
            captured["target_pixel_role"],
            "client_reported_diagnostic_only",
        )

    def test_linked_study_rejects_targets_outside_frozen_design(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-gaze-labels-") as name:
            root = Path(name)
            created = create_session(
                root,
                "participant-test",
                study_metadata={"study_session_id": "study-test"},
            )
            app = create_app(
                {
                    "TESTING": True,
                    "LEXIGAZE_BLUEPRINTS": ("gaze",),
                    "LEXIGAZE_GAZE_ROOT": root,
                    "LEXIGAZE_STUDY_ROOT": root,
                }
            )
            participant = {
                "state": "calibration_in_progress",
                "linked_data": {"gaze_session_id": created["session_id"]},
            }
            with (
                patch(
                    "web.routes.gaze._participant_session",
                    return_value=(object(), participant),
                ),
                patch("web.routes.gaze.save_sample") as save,
            ):
                for payload, error_fragment in (
                    (
                        {"point_index": 13, "repeat_index": 0, "motion_block_id": "left"},
                        "point_index",
                    ),
                    (
                        {"point_index": 0, "repeat_index": 0, "motion_block_id": "diagonal"},
                        "motion_block_id",
                    ),
                    (
                        {"point_index": 0, "repeat_index": 1, "motion_block_id": "neutral"},
                        "repeat_index",
                    ),
                ):
                    with self.subTest(error=error_fragment):
                        response = app.test_client().post(
                            "/api/gaze/sample",
                            json={
                                "session_id": created["session_id"],
                                "study_session_id": "study-test",
                                **payload,
                            },
                        )
                        self.assertEqual(response.status_code, 400)
                        self.assertIn(
                            error_fragment,
                            response.get_json()["error"],
                        )
            self.assertFalse(save.called)

    def test_non_study_labels_remain_legacy_compatible(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-gaze-legacy-") as name:
            root = Path(name)
            created = create_session(root, "legacy-researcher")
            app = create_app(
                {
                    "TESTING": True,
                    "LEXIGAZE_BLUEPRINTS": ("gaze",),
                    "LEXIGAZE_GAZE_ROOT": root,
                    "LEXIGAZE_STUDY_ROOT": root,
                }
            )
            captured: dict = {}

            def fake_save_sample(_root: Path, payload: dict) -> tuple[dict, int]:
                captured.update(payload)
                return {"ok": True}, 200

            with patch(
                "web.routes.gaze.save_sample",
                side_effect=fake_save_sample,
            ):
                response = app.test_client().post(
                    "/api/gaze/sample",
                    json={
                        "session_id": created["session_id"],
                        "point_index": 99,
                        "repeat_index": 7,
                        "target_x_norm": 0.123,
                        "target_y_norm": -0.456,
                        "motion_block_id": "researcher-defined",
                        "collection_protocol": "legacy-custom-v7",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["point_index"], 99)
        self.assertEqual(captured["repeat_index"], 7)
        self.assertEqual(captured["target_x_norm"], 0.123)
        self.assertEqual(captured["target_y_norm"], -0.456)
        self.assertEqual(captured["motion_block_id"], "researcher-defined")
        self.assertEqual(captured["collection_protocol"], "legacy-custom-v7")


class PredictionReceiptRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-receipt-route-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.config = {
            **_rehearsal_settings(),
            "TESTING": True,
            "LEXIGAZE_PUBLIC_STUDY_MODE": "1",
            "LEXIGAZE_STUDY_ROOT": str(self.root),
            "LEXIGAZE_GAZE_ROOT": str(self.root),
        }
        self.store = ParticipantStudyStore(self.root, settings=self.config)
        pair = self.store.create_collection_invite_pairs(1)[0]
        enrolled = self.store.enroll(
            _consent_payload(pair["visits"][0]["invite_code"])
        )
        self.session_id = enrolled["study_session_id"]
        self.token = enrolled["access_token"]
        self.model_name = "receipt-route-model"
        self.store.record_general_profile(self.session_id, self.token, _profile())
        self.store.record_general_system_check(
            self.session_id, self.token, _system_profile()
        )
        self.store.start_calibration(self.session_id, self.token, "GAZE-ROUTE")
        artifact = model_path(self.root, self.model_name)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{"fixture":true}\n', encoding="utf-8")
        self.artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        self.store.complete_calibration(
            self.session_id,
            self.token,
            {
                "passed": True,
                "model_artifact_sha256": self.artifact_sha256,
                "capture_contract": _capture_contract(),
            },
            model_name=self.model_name,
            model_artifact_sha256=self.artifact_sha256,
        )
        self.store.start_general_collection(
            self.session_id,
            self.token,
            assessment_viewport=_assessment_viewport(),
            model_artifact_sha256=self.artifact_sha256,
        )
        self.client = create_app(self.config).test_client()
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Lexigaze-Study-Session": self.session_id,
        }
        self.payload = {
            "image_data": "data:image/jpeg;base64,fixture",
            "capture_contract": _capture_contract(),
            "model_name": self.model_name,
            "viewport_width": 1280,
            "viewport_height": 800,
            "study_session_id": self.session_id,
            "allow_cuda": False,
            "validation_phase": "start",
            "validation_target_id": "heldout_top_left",
        }

    def test_success_and_explicit_no_face_return_receipts_but_hard_errors_do_not(
        self,
    ) -> None:
        success = {
            "ok": True,
            "screen_xy_px": [640.0, 400.0],
            "screen_xy_norm": [0.0, 0.0],
            "capture_contract_check": {
                "status": "compatible",
                "compatible": True,
            },
        }
        with patch("web.routes.gaze.predict", return_value=(success, 200)):
            response = self.client.post(
                "/api/gaze/predict",
                json=self.payload,
                headers=self.headers,
            )
        self.assertEqual(response.status_code, 200)
        first = response.get_json()["prediction_receipt"]["token"]
        self.assertRegex(first, r"^PR-[A-F0-9]{48}$")

        no_face = {
            "ok": False,
            "error": "no face detected in frame",
            "failure_code": "no_face_detected",
            "failure_stage": "attributable_sensor_failure",
            "capture_contract_check": {
                "status": "compatible",
                "compatible": True,
            },
        }
        with patch("web.routes.gaze.predict", return_value=(no_face, 400)):
            response = self.client.post(
                "/api/gaze/predict",
                json=self.payload,
                headers=self.headers,
            )
        self.assertEqual(response.status_code, 400)
        second = response.get_json()["prediction_receipt"]["token"]
        self.assertRegex(second, r"^PR-[A-F0-9]{48}$")

        capture_hard = {
            "ok": False,
            "error": "capture contract mismatch",
            "failure_code": "capture_contract_mismatch",
            "failure_stage": "capture_hard_error",
            "capture_contract_check": {
                "status": "mismatch",
                "compatible": False,
            },
        }
        with patch("web.routes.gaze.predict", return_value=(capture_hard, 409)):
            response = self.client.post(
                "/api/gaze/predict",
                json=self.payload,
                headers=self.headers,
            )
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("prediction_receipt", response.get_json())

        public = self.store.get_session(self.session_id, self.token)
        self.assertNotIn("prediction_receipts", public["general_collection"])
        self.assertNotIn(first, json.dumps(public))
        self.assertNotIn(second, json.dumps(public))

    def test_validation_labels_never_cross_the_sensor_inference_boundary(self) -> None:
        success = {
            "ok": True,
            "screen_xy_px": [640.0, 400.0],
            "screen_xy_norm": [0.0, 0.0],
            "capture_contract_check": {
                "status": "compatible",
                "compatible": True,
            },
        }
        with patch("web.routes.gaze.predict", return_value=(success, 200)) as predict:
            response = self.client.post(
                "/api/gaze/predict",
                json={
                    **self.payload,
                    "validation_phase": "start",
                    "validation_target_id": "heldout_top_left",
                    "target_x_norm": -0.75,
                    "target_y_norm": -0.75,
                    "text_context": "must never reach inference",
                    "cognitive_profile": {"must": "never reach inference"},
                },
                headers=self.headers,
            )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        inference_payload = predict.call_args.args[1]
        self.assertEqual(
            set(inference_payload),
            {
                "image_data",
                "capture_contract",
                "model_name",
                "viewport_width",
                "viewport_height",
                "allow_cuda",
            },
        )
        for forbidden in (
            "validation_phase",
            "validation_target_id",
            "target_x_norm",
            "target_y_norm",
            "text_context",
            "cognitive_profile",
            "study_session_id",
        ):
            self.assertNotIn(forbidden, inference_payload)

    def test_receipt_routes_reject_non_object_json_without_predicting(self) -> None:
        endpoints = (
            "/api/gaze/predict",
            "/api/predict",
            f"/api/study/sessions/{self.session_id}/general/validation",
        )
        with patch("web.routes.gaze.predict") as predict:
            for endpoint in endpoints:
                with self.subTest(endpoint=endpoint):
                    response = self.client.post(
                        endpoint,
                        json=[{}],
                        headers=self.headers,
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(
                        response.get_json(),
                        {
                            "ok": False,
                            "error": "request JSON body must be an object",
                        },
                    )
        self.assertFalse(predict.called)
        session = self.store.get_session(self.session_id, self.token)
        self.assertEqual(
            session["general_collection"]["phase"],
            "start_validation_required",
        )

    def test_model_artifact_io_failures_are_stable_and_fail_closed(self) -> None:
        with patch(
            "web.routes.study.model_artifact_sha256",
            side_effect=PermissionError("artifact unreadable"),
        ):
            start = self.client.post(
                f"/api/study/sessions/{self.session_id}/general/start",
                headers=self.headers,
                json={"assessment_viewport": _assessment_viewport()},
            )
            validation = self.client.post(
                f"/api/study/sessions/{self.session_id}/general/validation",
                headers=self.headers,
                json={
                    "phase": "start",
                    "prediction_receipts": ["PR-" + "A" * 48],
                },
            )
        for response in (start, validation):
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.get_json()["error"], "artifact unreadable")
            self.assertEqual(response.get_json()["error_type"], "StudyStateError")

        success = {
            "ok": True,
            "screen_xy_px": [640.0, 400.0],
            "screen_xy_norm": [0.0, 0.0],
            "capture_contract_check": {
                "status": "compatible",
                "compatible": True,
            },
        }
        with patch(
            "web.routes.gaze.model_artifact_sha256",
            side_effect=PermissionError("artifact unreadable"),
        ), patch("web.routes.gaze.predict") as predict:
            before = self.client.post(
                "/api/gaze/predict",
                json=self.payload,
                headers=self.headers,
            )
        self.assertEqual(before.status_code, 409)
        self.assertEqual(before.get_json()["failure_stage"], "model_hard_error")
        self.assertFalse(predict.called)

        with patch(
            "web.routes.gaze.model_artifact_sha256",
            side_effect=[self.artifact_sha256, PermissionError("artifact unreadable")],
        ), patch("web.routes.gaze.predict", return_value=(success, 200)) as predict:
            after = self.client.post(
                "/api/gaze/predict",
                json=self.payload,
                headers=self.headers,
            )
        self.assertEqual(after.status_code, 409)
        self.assertEqual(after.get_json()["failure_stage"], "model_hard_error")
        self.assertTrue(predict.called)


if __name__ == "__main__":
    unittest.main()
