"""Route and security tests for the dedicated measurement browser surface."""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.gaze_core.measurement_preflight import (
    MeasurementPreflightAuthenticationError,
)
from web import create_app
from web.measurement_surface_security import MAX_MEASUREMENT_CONTENT_LENGTH
from web.routes.measurement import (
    CHALLENGE_TOKEN_HEADER,
    CREATE_REQUEST_ID_HEADER,
    PREFLIGHT_TOKEN_HEADER,
    RUN_ID_HEADER,
    RUN_TOKEN_HEADER,
)


RUN_ID = "WGMC-route-test"
RUN_TOKEN = "wgmr_client_" + "a" * 64
CREATE_REQUEST_ID = "WGMCREQ-" + "b" * 32
CHALLENGE_TOKEN = "wgmc_challenge_test_secret"
PREFLIGHT_TOKEN = "wgmc_preflight_test_secret"
BASE_URL = "http://127.0.0.1:8099"


def capture_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "intent_width_px": 1280,
        "intent_height_px": 720,
        "intent_frame_rate_hz": 30.0,
        "source_width_px": 1280,
        "source_height_px": 720,
        "source_frame_rate_hz": 30.0,
        "transport_width_px": 1280,
        "transport_height_px": 720,
    }


def viewport() -> dict[str, float]:
    return {"width": 1000.0, "height": 800.0, "device_pixel_ratio": 1.0}


def schedule_row() -> dict[str, Any]:
    return {
        "sequence_index": 0,
        "block_id": "calibration_center",
        "block_role": "calibration",
        "posture": "center",
        "distance": "normal",
        "target_x_viewport_fraction": 0.5,
        "target_y_viewport_fraction": 0.5,
    }


def browser_gate() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "implementation_id": "browser-visible-focus-dwell-render-v1",
        "sequence_index": 0,
        "visibility_state": "visible",
        "document_focused": True,
        "viewport_width": 1000.0,
        "viewport_height": 800.0,
        "device_pixel_ratio": 1.0,
        "rendered_target_center_x_px": 500.0,
        "rendered_target_center_y_px": 400.0,
        "minimum_dwell_ms": 900.0,
        "observed_stable_dwell_ms": 925.0,
        "stable_render_frame_count": 2,
        "structural_browser_gate_only": True,
        "visual_attestation_claimed": False,
    }


class FakePreflight:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.expired = False

    def start(self, *, capture_contract, viewport):
        self.events.append("preflight.start")
        return {
            "ok": True,
            "preflight_token": PREFLIGHT_TOKEN,
            "target_free": True,
            "persistent_storage_used": False,
            "measurement_claim_authorized": False,
        }

    def submit_frame(self, token, *, image_data, capture_contract):
        self.events.append("preflight.frame")
        if token != PREFLIGHT_TOKEN:
            raise MeasurementPreflightAuthenticationError("invalid token")
        return {
            "ok": True,
            "classification": "ready",
            "ready": True,
            "consecutive_successes": 3,
            "measurement_claim_authorized": False,
        }

    def consume(self, token, *, capture_contract, viewport):
        self.events.append("preflight.consume")
        if self.expired or token != PREFLIGHT_TOKEN:
            raise MeasurementPreflightAuthenticationError("expired")
        return {
            "receipt_type": "target_free_camera_readiness_receipt_v1",
            "receipt_sha256": "c" * 64,
        }


class FakeRunner:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.exists = False
        self.initialized = True
        self.inspect_mode = "active"
        self.spool_available = False
        self.prepared = False
        self.submit_result: dict[str, Any] = {
            "ok": True,
            "classification": "success",
            "consumed": True,
            "receipt": {"progress": {"next_sequence_index": 1}},
            "measurement_claim_authorized": False,
        }
        self.submit_kwargs: dict[str, Any] | None = None
        self.current_status = self._status()

    def _status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "capture_run_id": RUN_ID,
            "phase": "scheduled",
            "progress": {"next_sequence_index": 0},
            "challenge_outstanding": False,
            "runner": {
                "runtime_binding": {
                    "capture_contract": capture_contract(),
                    "viewport": viewport(),
                },
                "prepared_observation_pending": False,
            },
            "measurement_claim_authorized": False,
        }

    def lookup_create_request(self, *, create_request_id, run_token):
        self.events.append("runner.lookup")
        assert create_request_id == CREATE_REQUEST_ID
        assert run_token == RUN_TOKEN
        if not self.exists:
            return {"ok": True, "exists": False, "idempotent": False}
        result = self._status()
        result.update({"exists": True, "idempotent": True})
        if not self.initialized:
            result["runner"]["runtime_binding"] = None
        return result

    def create_run(self, **kwargs):
        self.events.append("runner.create")
        return {
            **self._status(),
            "created_new": not self.exists,
            "idempotent": self.exists,
            "run_token": RUN_TOKEN,
            "create_request_id": CREATE_REQUEST_ID,
            "received_readiness": kwargs["readiness_preflight"],
        }

    def get_status(self, capture_run_id, run_token):
        self.events.append("runner.status")
        assert (capture_run_id, run_token) == (RUN_ID, RUN_TOKEN)
        return deepcopy(self.current_status)

    def inspect_challenge(self, capture_run_id, run_token, challenge_token):
        self.events.append("runner.inspect")
        assert (capture_run_id, run_token, challenge_token) == (
            RUN_ID,
            RUN_TOKEN,
            CHALLENGE_TOKEN,
        )
        if self.inspect_mode == "committed":
            return {
                "ok": True,
                "status": "committed",
                "receipt": {"progress": {"next_sequence_index": 1}},
            }
        return {
            "ok": True,
            "status": "active",
            "challenge_id": "challenge-1",
            "ordinal": 0,
            "block_role": "calibration",
            "schedule_row": schedule_row(),
            "viewport": viewport(),
            "prepared_observation_pending": self.prepared,
            "server_spool_available": self.spool_available,
            "measurement_claim_authorized": False,
        }

    def issue_next_challenge(self, capture_run_id, run_token):
        self.events.append("runner.issue")
        return {
            "ok": True,
            "capture_run_id": RUN_ID,
            "challenge_id": "challenge-1",
            "challenge_token": CHALLENGE_TOKEN,
            "schedule_row": schedule_row(),
            "measurement_claim_authorized": False,
        }

    def rotate_unconsumed_challenge(self, capture_run_id, run_token):
        self.events.append("runner.rotate")
        return self.issue_next_challenge(capture_run_id, run_token)

    def submit_frame(self, capture_run_id, run_token, challenge_token, **kwargs):
        self.events.append("runner.submit")
        self.submit_kwargs = deepcopy(kwargs)
        return deepcopy(self.submit_result)

    def finalize_calibration(self, capture_run_id, run_token):
        self.events.append("runner.finalize")
        return {
            "ok": True,
            "phase": "model_bound",
            "measurement_claim_authorized": False,
        }

    def verify_artifact(self, capture_run_id, run_token):
        self.events.append("runner.verify")
        return {
            "ok": True,
            "phase": "artifact_verified",
            "measurement_claim_authorized": False,
        }

    def analyze_verified_run(self, capture_run_id, run_token):
        self.events.append("runner.analysis")
        assert (capture_run_id, run_token) == (RUN_ID, RUN_TOKEN)
        return {
            "schema_version": 1,
            "status": "integrity_verified_descriptive_live_runner",
            "analysis_sha256": "a" * 64,
            "claim_boundary": {
                "measurement_claim_authorized": False,
                "threshold_selected": False,
                "quality_band_change_authorized": False,
            },
            "evaluation": {
                "selected_personal_model": {
                    "successful_count": 128,
                    "median_spatial_error_px": 42.0,
                }
            },
        }

    def abort_and_cleanup(self, capture_run_id, run_token, *, reason):
        self.events.append("runner.abort")
        return {
            "ok": True,
            "phase": "aborted",
            "classification": "aborted_cleanup_verified",
            "cleanup_verified": True,
            "measurement_claim_authorized": False,
        }


class MeasurementSurfaceRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = FakeRunner()
        self.preflight = FakePreflight()
        self.app = create_app(
            {
                "TESTING": True,
                "LEXIGAZE_BLUEPRINTS": ("measurement",),
                "LEXIGAZE_MEASUREMENT_CEILING_MODE": True,
                "LEXIGAZE_MEASUREMENT_AUTHORITY": "127.0.0.1:8099",
                "LEXIGAZE_MEASUREMENT_RUNNER": self.runner,
                "LEXIGAZE_MEASUREMENT_PREFLIGHT": self.preflight,
            }
        )
        self.client = self.app.test_client()

    def request(self, method: str, path: str, **kwargs):
        return self.client.open(path, method=method, base_url=BASE_URL, **kwargs)

    def auth_headers(self, *, challenge: bool = False) -> dict[str, str]:
        headers = {RUN_ID_HEADER: RUN_ID, RUN_TOKEN_HEADER: RUN_TOKEN}
        if challenge:
            headers[CHALLENGE_TOKEN_HEADER] = CHALLENGE_TOKEN
        return headers

    def test_page_and_static_asset_have_strict_security_headers(self) -> None:
        response = self.request("GET", "/measurement-ceiling")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertNotIn("unsafe-inline", response.headers["Content-Security-Policy"])
        self.assertEqual(
            response.headers["Permissions-Policy"],
            "camera=(self), microphone=(), geolocation=()",
        )
        self.assertRegex(response.headers["Server-Timing"], r"^measurement;dur=")
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)
        static = self.request("GET", "/static/measurement_ceiling.js")
        self.assertEqual(static.status_code, 200)
        static.close()

    def test_dns_rebinding_remote_query_options_and_unscoped_routes_fail(self) -> None:
        wrong_host = self.client.get(
            "/measurement-ceiling", base_url="http://localhost:8099"
        )
        self.assertEqual(wrong_host.status_code, 421)
        wrong_port = self.client.get(
            "/measurement-ceiling", base_url="http://127.0.0.1:8100"
        )
        self.assertEqual(wrong_port.status_code, 421)
        remote = self.client.get(
            "/measurement-ceiling",
            base_url=BASE_URL,
            environ_overrides={"REMOTE_ADDR": "192.0.2.4"},
        )
        self.assertEqual(remote.status_code, 403)
        self.assertEqual(
            self.request("GET", "/measurement-ceiling?token=forbidden").status_code,
            400,
        )
        self.assertEqual(
            self.request("OPTIONS", "/api/measurement-ceiling/preflight").status_code,
            405,
        )
        self.assertEqual(self.request("GET", "/").status_code, 404)
        self.assertEqual(self.request("GET", "/api/sessions").status_code, 404)
        self.assertEqual(self.request("GET", "/static/app.js").status_code, 404)

    def test_cross_origin_and_cross_site_requests_fail(self) -> None:
        origin = self.request(
            "GET",
            "/measurement-ceiling",
            headers={"Origin": "http://evil.example"},
        )
        self.assertEqual(origin.status_code, 403)
        cross_site = self.request(
            "GET",
            "/measurement-ceiling",
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        self.assertEqual(cross_site.status_code, 403)

    def test_json_endpoints_require_application_json_object_and_exact_fields(self) -> None:
        path = "/api/measurement-ceiling/preflight"
        self.assertEqual(self.request("POST", path, data="{}").status_code, 415)
        self.assertEqual(self.request("POST", path, json=[]).status_code, 400)
        response = self.request(
            "POST",
            path,
            json={
                "capture_contract": capture_contract(),
                "viewport": viewport(),
                "target": [0.5, 0.5],
            },
        )
        self.assertEqual(response.status_code, 400)
        oversized = self.request(
            "POST",
            path,
            data=b"{}",
            content_type="application/json",
            environ_overrides={
                "CONTENT_LENGTH": str(MAX_MEASUREMENT_CONTENT_LENGTH + 1)
            },
        )
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(
            oversized.get_json()["classification"],
            "measurement_payload_too_large",
        )

    def test_measurement_mode_is_isolated_and_content_limited(self) -> None:
        self.assertEqual(
            self.app.config["MAX_CONTENT_LENGTH"],
            MAX_MEASUREMENT_CONTENT_LENGTH,
        )
        with self.assertRaises(ValueError):
            create_app(
                {
                    "LEXIGAZE_BLUEPRINTS": ("measurement", "study"),
                    "LEXIGAZE_MEASUREMENT_CEILING_MODE": True,
                }
            )
        with self.assertRaises(ValueError):
            create_app(
                {
                    "LEXIGAZE_BLUEPRINTS": ("measurement",),
                    "LEXIGAZE_MEASUREMENT_CEILING_MODE": True,
                    "LEXIGAZE_PUBLIC_STUDY_MODE": True,
                }
            )

    def test_preflight_secret_is_header_only_and_frames_are_target_free(self) -> None:
        started = self.request(
            "POST",
            "/api/measurement-ceiling/preflight",
            json={"capture_contract": capture_contract(), "viewport": viewport()},
        )
        self.assertEqual(started.status_code, 200)
        self.assertEqual(started.headers[PREFLIGHT_TOKEN_HEADER], PREFLIGHT_TOKEN)
        self.assertNotIn(PREFLIGHT_TOKEN, started.get_data(as_text=True))
        frame = self.request(
            "POST",
            "/api/measurement-ceiling/preflight/frames",
            headers={PREFLIGHT_TOKEN_HEADER: PREFLIGHT_TOKEN},
            json={"image_data": "data:image/jpeg;base64,AA==", "capture_contract": capture_contract()},
        )
        self.assertEqual(frame.status_code, 200)
        self.assertEqual(frame.get_json()["consecutive_successes"], 3)

    def test_create_lookup_precedes_consume_and_secrets_are_header_only(self) -> None:
        response = self.request(
            "POST",
            "/api/measurement-ceiling/runs",
            headers={
                CREATE_REQUEST_ID_HEADER: CREATE_REQUEST_ID,
                RUN_TOKEN_HEADER: RUN_TOKEN,
                PREFLIGHT_TOKEN_HEADER: PREFLIGHT_TOKEN,
            },
            json={"capture_contract": capture_contract(), "viewport": viewport()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers[RUN_ID_HEADER], RUN_ID)
        self.assertNotIn(RUN_TOKEN_HEADER, response.headers)
        rendered = response.get_data(as_text=True)
        self.assertNotIn(RUN_TOKEN, rendered)
        self.assertNotIn(CREATE_REQUEST_ID, rendered)
        self.assertEqual(self.runner.events[:2], ["runner.lookup", "runner.create"])
        self.assertEqual(self.preflight.events, ["preflight.consume"])

    def test_initialized_create_replay_never_consumes_preflight(self) -> None:
        self.runner.exists = True
        response = self.request(
            "POST",
            "/api/measurement-ceiling/runs",
            headers={CREATE_REQUEST_ID_HEADER: CREATE_REQUEST_ID, RUN_TOKEN_HEADER: RUN_TOKEN},
            json={"capture_contract": capture_contract(), "viewport": viewport()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.preflight.events, [])
        self.assertIsNone(response.get_json()["received_readiness"])

    def test_uninitialized_reservation_requires_fresh_preflight(self) -> None:
        self.runner.exists = True
        self.runner.initialized = False
        response = self.request(
            "POST",
            "/api/measurement-ceiling/runs",
            headers={
                CREATE_REQUEST_ID_HEADER: CREATE_REQUEST_ID,
                RUN_TOKEN_HEADER: RUN_TOKEN,
                PREFLIGHT_TOKEN_HEADER: PREFLIGHT_TOKEN,
            },
            json={"capture_contract": capture_contract(), "viewport": viewport()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.preflight.events, ["preflight.consume"])

    def test_expired_preflight_preserves_pending_create_authority(self) -> None:
        self.preflight.expired = True
        response = self.request(
            "POST",
            "/api/measurement-ceiling/runs",
            headers={
                CREATE_REQUEST_ID_HEADER: CREATE_REQUEST_ID,
                RUN_TOKEN_HEADER: RUN_TOKEN,
                PREFLIGHT_TOKEN_HEADER: PREFLIGHT_TOKEN,
            },
            json={"capture_contract": capture_contract(), "viewport": viewport()},
        )
        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertFalse(payload["existing_run"])
        self.assertTrue(payload["authority_retained"])
        self.assertTrue(payload["replace_preflight_allowed"])
        self.assertNotIn(RUN_TOKEN, response.get_data(as_text=True))

    def test_status_uses_challenge_outstanding_and_authenticates_recovery_token(self) -> None:
        self.runner.current_status["challenge_outstanding"] = True
        response = self.request(
            "GET",
            "/api/measurement-ceiling/status",
            headers=self.auth_headers(challenge=True),
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["challenge_outstanding"])
        self.assertEqual(payload["challenge_recovery"]["status"], "active")
        self.assertFalse(payload["challenge_recovery"]["server_spool_available"])
        self.assertNotIn(CHALLENGE_TOKEN, response.get_data(as_text=True))

    def test_challenge_secret_is_header_only_and_rotation_is_public_runner_api(self) -> None:
        issued = self.request(
            "POST",
            "/api/measurement-ceiling/challenges",
            headers=self.auth_headers(),
            json={},
        )
        self.assertEqual(issued.status_code, 200)
        self.assertEqual(issued.headers[CHALLENGE_TOKEN_HEADER], CHALLENGE_TOKEN)
        self.assertNotIn(CHALLENGE_TOKEN, issued.get_data(as_text=True))
        rotated = self.request(
            "POST",
            "/api/measurement-ceiling/challenges/rotate",
            headers=self.auth_headers(),
            json={},
        )
        self.assertEqual(rotated.status_code, 200)
        self.assertIn("runner.rotate", self.runner.events)

    def test_capture_gate_is_validated_then_discarded_before_runner(self) -> None:
        response = self.request(
            "POST",
            "/api/measurement-ceiling/captures",
            headers=self.auth_headers(challenge=True),
            json={
                "image_data": "data:image/jpeg;base64,AA==",
                "capture_contract": capture_contract(),
                "client_gate": browser_gate(),
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["classification"], "success")
        self.assertIsNotNone(self.runner.submit_kwargs)
        self.assertNotIn("client_gate", self.runner.submit_kwargs)
        self.assertNotIn("target", self.runner.submit_kwargs)

    def test_capture_rejects_gate_target_or_cognitive_extras(self) -> None:
        body = {
            "image_data": "data:image/jpeg;base64,AA==",
            "capture_contract": capture_contract(),
            "client_gate": browser_gate(),
            "cursor": [500, 400],
        }
        response = self.request(
            "POST",
            "/api/measurement-ceiling/captures",
            headers=self.auth_headers(challenge=True),
            json=body,
        )
        self.assertEqual(response.status_code, 400)
        gate = browser_gate()
        gate["cognitive_profile"] = "forbidden"
        response = self.request(
            "POST",
            "/api/measurement-ceiling/captures",
            headers=self.auth_headers(challenge=True),
            json={
                "image_data": "data:image/jpeg;base64,AA==",
                "capture_contract": capture_contract(),
                "client_gate": gate,
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_server_spool_resume_never_sends_a_new_frame(self) -> None:
        self.runner.spool_available = True
        response = self.request(
            "POST",
            "/api/measurement-ceiling/captures",
            headers=self.auth_headers(challenge=True),
            json={"resume_server_spool": True},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["resumed_from_server_spool"])
        self.assertIsNone(self.runner.submit_kwargs["image_data"])
        self.assertEqual(
            self.runner.submit_kwargs["observed_capture_contract"],
            capture_contract(),
        )

    def test_committed_challenge_can_recover_without_spool_or_new_inference(self) -> None:
        self.runner.inspect_mode = "committed"
        self.runner.submit_result = {
            "ok": True,
            "classification": "exact_retry",
            "consumed": True,
            "inference_replayed": False,
            "receipt": {"progress": {"next_sequence_index": 1}},
        }
        response = self.request(
            "POST",
            "/api/measurement-ceiling/captures",
            headers=self.auth_headers(challenge=True),
            json={"resume_server_spool": True},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["inference_replayed"])
        self.assertIsNone(self.runner.submit_kwargs["image_data"])

    def test_no_face_consumes_attempt_and_exact_retry_flags_are_preserved(self) -> None:
        self.runner.submit_result = {
            "ok": True,
            "classification": "attributable_sensor_failure",
            "consumed": True,
            "prediction_success": False,
            "receipt": {"progress": {"next_sequence_index": 1}},
        }
        no_face = self.request(
            "POST",
            "/api/measurement-ceiling/captures",
            headers=self.auth_headers(challenge=True),
            json={
                "image_data": "data:image/jpeg;base64,AA==",
                "capture_contract": capture_contract(),
                "client_gate": browser_gate(),
            },
        )
        self.assertEqual(no_face.status_code, 200)
        self.assertTrue(no_face.get_json()["consumed"])

        self.runner.submit_result = {
            "ok": False,
            "classification": "calibration_save_failed",
            "consumed": False,
            "retryable": True,
            "exact_frame_retry_required": True,
            "server_spool_retry_available": True,
            "new_frame_retry_allowed": False,
        }
        retry = self.request(
            "POST",
            "/api/measurement-ceiling/captures",
            headers=self.auth_headers(challenge=True),
            json={
                "image_data": "data:image/jpeg;base64,AA==",
                "capture_contract": capture_contract(),
                "client_gate": browser_gate(),
            },
        )
        self.assertEqual(retry.status_code, 409)
        self.assertTrue(retry.get_json()["exact_frame_retry_required"])

    def test_calibration_negative_result_is_terminal_and_normalized(self) -> None:
        self.runner.finalize_calibration = lambda *_args: {
            "ok": False,
            "classification": "negative_calibration_result",
            "calibration_attempt_count": 65,
            "usable_calibration_count": 63,
            "cleanup_verified": True,
            "images_purged": True,
            "measurement_claim_authorized": False,
        }
        response = self.request(
            "POST",
            "/api/measurement-ceiling/calibration/finalize",
            headers=self.auth_headers(),
            json={},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(
            payload["classification"],
            "calibration_unusable_negative_result",
        )
        self.assertTrue(payload["terminal"])
        self.assertTrue(payload["cleanup_verified"])
        self.assertEqual(payload["calibration_usable_count"], 63)
        self.assertTrue(payload["purge_verified"])

        self.runner.current_status.update(
            {
                "phase": "failed_integrity",
                "runner": {
                    **self.runner.current_status["runner"],
                    "calibration_usable_manifest_count": 63,
                },
                "failure": {
                    "code": "calibration_usable_count_below_65",
                    "cleanup": {
                        "cleanup_verified": True,
                        "image_directories_absent": True,
                    },
                },
            }
        )
        status = self.request(
            "GET",
            "/api/measurement-ceiling/status",
            headers=self.auth_headers(),
        ).get_json()
        self.assertEqual(status["calibration_usable_count"], 63)
        self.assertTrue(status["purge_verified"])

    def test_authenticated_analysis_preserves_nonpromotion_boundary(self) -> None:
        response = self.request(
            "POST",
            "/api/measurement-ceiling/analysis",
            headers=self.auth_headers(),
            json={},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(
            payload["classification"],
            "integrity_verified_descriptive_analysis",
        )
        self.assertEqual(
            payload["status"], "integrity_verified_descriptive_live_runner"
        )
        self.assertFalse(payload["measurement_claim_authorized"])
        self.assertFalse(payload["physical_capture_claim_authorized"])
        self.assertFalse(payload["claim_boundary"]["threshold_selected"])
        self.assertEqual(self.runner.events[-1], "runner.analysis")

    def test_abort_credentials_survive_until_cleanup_is_confirmed(self) -> None:
        response = self.request(
            "POST",
            "/api/measurement-ceiling/abort",
            headers=self.auth_headers(),
            json={"reason": "operator_aborted_browser_measurement"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["cleanup_verified"])
        self.runner.abort_and_cleanup = lambda *_args, **_kwargs: {
            "ok": False,
            "classification": "spool_cleanup_failed",
            "cleanup_verified": False,
            "retryable": True,
        }
        failed = self.request(
            "POST",
            "/api/measurement-ceiling/abort",
            headers=self.auth_headers(),
            json={"reason": "operator_aborted_browser_measurement"},
        )
        self.assertEqual(failed.status_code, 409)
        self.assertFalse(failed.get_json()["cleanup_verified"])

    def test_route_source_has_no_participant_store_or_secret_logging(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "web"
            / "routes"
            / "measurement.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("participant_study", source)
        self.assertNotIn("current_app.logger", source)
        self.assertNotIn("request.args", source)
        self.assertIn("_without_body_secrets", source)
        self.assertIn("validate_measurement_browser_gate", source)


if __name__ == "__main__":
    unittest.main()
