"""CPU-only integration tests for public-tunnel security guardrails."""

from __future__ import annotations

import io
import os
import re
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from web import LOCAL_MAX_CONTENT_LENGTH, create_app
from web.security import (
    AUTH_COOKIE_NAME,
    LOGIN_PATH,
    SESSION_PATH,
    build_tunnel_bootstrap_url,
)

TOKEN = "lexigaze-test-token-" + ("x" * 32)
BASE_URL = "https://lexigaze.example.test"


class TunnelSecurityTests(unittest.TestCase):
    def make_app(self, **overrides):
        config = {
            "TESTING": True,
            "TUNNEL_MAX_CONTENT_LENGTH": 1024 * 1024,
            "TUNNEL_MUTATION_LIMIT": 100,
            "TUNNEL_REALTIME_LIMIT": 100,
            "TUNNEL_EXPENSIVE_LIMIT": 100,
            "TUNNEL_AUTH_ATTEMPT_LIMIT": 20,
            "TUNNEL_RATE_WINDOW_SECONDS": 60,
            "TUNNEL_EXPENSIVE_CONCURRENCY": 1,
            "TUNNEL_SESSION_TTL_SECONDS": 300,
        }
        config.update(overrides)
        return create_app(config, tunnel_mode=True, tunnel_token=TOKEN)

    @staticmethod
    def auth_headers(token=TOKEN):
        return {"Authorization": f"Bearer {token}"}

    def test_tunnel_mode_refuses_missing_or_weak_token(self):
        with (
            patch.dict(os.environ, {"LEXIGAZE_TUNNEL_TOKEN": ""}),
            self.assertRaisesRegex(RuntimeError, "requires a security token"),
        ):
            create_app({"TESTING": True}, tunnel_mode=True)
        with self.assertRaisesRegex(RuntimeError, "at least 32"):
            create_app({"TESTING": True}, tunnel_mode=True, tunnel_token="too-short")

    def test_local_mode_remains_open_and_keeps_original_upload_limit(self):
        with patch.dict(os.environ, {"LEXIGAZE_TUNNEL_MAX_UPLOAD_MB": "invalid"}):
            app = create_app({"TESTING": True})
        self.assertFalse(app.config["TUNNEL_MODE"])
        self.assertEqual(app.config["MAX_CONTENT_LENGTH"], LOCAL_MAX_CONTENT_LENGTH)
        self.assertEqual(app.test_client().get("/api/ping").status_code, 200)

    def test_every_application_route_requires_authentication(self):
        app = self.make_app()
        client = app.test_client()

        for path in ("/", "/gaze", "/api/ping", "/api/gaze/health"):
            with self.subTest(path=path):
                response = client.get(path, base_url=BASE_URL)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.get_json()["error"], "authentication_required")

        response = client.get(
            "/api/ping",
            base_url=BASE_URL,
            headers=self.auth_headers(),
        )
        self.assertEqual(response.status_code, 200)

    def test_query_parameter_token_is_rejected(self):
        response = self.make_app().test_client().get(
            f"/api/ping?access_token={TOKEN}",
            base_url=BASE_URL,
        )
        self.assertEqual(response.status_code, 401)

    def test_fragment_bootstrap_creates_hardened_cookie(self):
        app = self.make_app()
        client = app.test_client()
        bootstrap_url = build_tunnel_bootstrap_url(
            "https://public.example",
            "/gaze",
            TOKEN,
        )
        parts = urlsplit(bootstrap_url)
        fragment = parse_qs(parts.fragment)
        self.assertEqual(parts.query, "")
        self.assertEqual(parts.path, LOGIN_PATH)
        self.assertEqual(fragment["token"], [TOKEN])
        self.assertEqual(fragment["next"], ["/gaze"])

        login_page = client.get(LOGIN_PATH, base_url=BASE_URL)
        self.assertEqual(login_page.status_code, 200)
        self.assertNotIn(TOKEN, login_page.get_data(as_text=True))
        self.assertIn("location.hash", login_page.get_data(as_text=True))
        self.assertNotIn(
            "unsafe-inline",
            login_page.headers["Content-Security-Policy"],
        )
        nonce_match = re.search(
            r"script-src 'nonce-([^']+)'",
            login_page.headers["Content-Security-Policy"],
        )
        self.assertIsNotNone(nonce_match)
        self.assertIn(
            f'<script nonce="{nonce_match.group(1)}">',
            login_page.get_data(as_text=True),
        )

        response = client.post(
            SESSION_PATH,
            base_url=BASE_URL,
            json={"token": TOKEN, "next": "/gaze"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["next"], "/gaze")
        cookie = response.headers["Set-Cookie"]
        self.assertIn(AUTH_COOKIE_NAME, cookie)
        self.assertIn("Secure", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertNotIn(TOKEN, cookie)

        protected = client.get("/api/ping", base_url=BASE_URL)
        self.assertEqual(protected.status_code, 200)

        logout = client.delete(SESSION_PATH, base_url=BASE_URL)
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(client.get("/api/ping", base_url=BASE_URL).status_code, 401)

    def test_bootstrap_rejects_invalid_token_and_open_redirect(self):
        app = self.make_app()
        client = app.test_client()
        invalid = client.post(
            SESSION_PATH,
            base_url=BASE_URL,
            json={"token": "wrong", "next": "/"},
        )
        self.assertEqual(invalid.status_code, 401)
        self.assertNotIn("Set-Cookie", invalid.headers)

        valid = client.post(
            SESSION_PATH,
            base_url=BASE_URL,
            json={"token": TOKEN, "next": "https://attacker.example/steal"},
        )
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(valid.get_json()["next"], "/")

        backslash = client.post(
            SESSION_PATH,
            base_url=BASE_URL,
            json={"token": TOKEN, "next": "/\\attacker.example/steal"},
        )
        self.assertEqual(backslash.status_code, 200)
        self.assertEqual(backslash.get_json()["next"], "/")

    def test_invalid_login_rate_limit_cannot_lock_out_valid_token(self):
        app = self.make_app(TUNNEL_AUTH_ATTEMPT_LIMIT=1)
        client = app.test_client()

        first_invalid = client.post(
            SESSION_PATH,
            base_url=BASE_URL,
            json={"token": "invalid"},
        )
        second_invalid = client.post(
            SESSION_PATH,
            base_url=BASE_URL,
            json={"token": "still-invalid"},
        )
        valid = client.post(
            SESSION_PATH,
            base_url=BASE_URL,
            json={"token": TOKEN},
        )

        self.assertEqual(first_invalid.status_code, 401)
        self.assertEqual(second_invalid.status_code, 429)
        self.assertEqual(second_invalid.get_json()["error"], "too_many_auth_attempts")
        self.assertEqual(valid.status_code, 200)

    def test_request_size_and_mutation_rate_limits(self):
        app = self.make_app(
            TUNNEL_MAX_CONTENT_LENGTH=64,
            TUNNEL_MUTATION_LIMIT=2,
        )

        @app.post("/api/security/mutation-probe")
        def mutation_probe():
            return {"ok": True}

        client = app.test_client()
        oversized = client.post(
            "/api/security/mutation-probe",
            base_url=BASE_URL,
            headers=self.auth_headers(),
            data=b"x" * 65,
            content_type="application/octet-stream",
        )
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(oversized.get_json()["max_bytes"], 64)

        first = client.post(
            "/api/security/mutation-probe",
            base_url=BASE_URL,
            headers=self.auth_headers(),
        )
        second = client.post(
            "/api/security/mutation-probe",
            base_url=BASE_URL,
            headers=self.auth_headers(),
        )
        third = client.post(
            "/api/security/mutation-probe",
            base_url=BASE_URL,
            headers=self.auth_headers(),
        )
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertEqual(third.status_code, 429)
        self.assertEqual(third.get_json()["error"], "mutation_rate_limit_exceeded")
        self.assertIn("Retry-After", third.headers)

    def test_realtime_prediction_has_a_separate_rate_budget(self):
        app = self.make_app(
            TUNNEL_MUTATION_LIMIT=1,
            TUNNEL_REALTIME_LIMIT=2,
        )
        app.view_functions["gaze.predict_gaze"] = lambda: {"ok": True}
        client = app.test_client()

        responses = [
            client.post(
                "/api/gaze/predict",
                base_url=BASE_URL,
                headers=self.auth_headers(),
            )
            for _ in range(3)
        ]
        self.assertEqual([response.status_code for response in responses], [200, 200, 429])
        self.assertEqual(responses[-1].get_json()["error"], "realtime_rate_limit_exceeded")

    def test_expensive_operations_use_nonblocking_global_backpressure(self):
        app = self.make_app(TUNNEL_EXPENSIVE_CONCURRENCY=1)
        started = threading.Event()
        release = threading.Event()
        first_status = []

        @app.post("/api/cognitive/analyze/security-probe")
        def expensive_probe():
            started.set()
            release.wait(timeout=3)
            return {"ok": True}

        def run_first_request():
            with app.test_client() as client:
                response = client.post(
                    "/api/cognitive/analyze/security-probe",
                    base_url=BASE_URL,
                    headers=self.auth_headers(),
                )
                first_status.append(response.status_code)

        worker = threading.Thread(target=run_first_request, daemon=True)
        worker.start()
        self.assertTrue(started.wait(timeout=2), "first expensive request did not start")
        try:
            second = app.test_client().post(
                "/api/cognitive/analyze/security-probe",
                base_url=BASE_URL,
                headers=self.auth_headers(),
            )
            self.assertEqual(second.status_code, 429)
            self.assertEqual(second.get_json()["error"], "expensive_operation_in_progress")
        finally:
            release.set()
            worker.join(timeout=3)

        self.assertEqual(first_status, [200])

    def test_security_headers_are_present_on_denied_responses(self):
        response = self.make_app().test_client().get("/api/ping", base_url=BASE_URL)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_ngrok_failure_never_starts_a_local_fallback(self):
        import run

        fake_app = Mock()
        fake_server = Mock()
        stop_server = threading.Event()
        fake_server.serve_forever.side_effect = lambda: stop_server.wait(timeout=2)
        fake_server.shutdown.side_effect = stop_server.set
        with (
            patch.object(run, "_resolve_tunnel_token", return_value=(TOKEN, True)),
            patch.object(run, "create_app", return_value=fake_app),
            patch.object(run, "_create_tunnel_server", return_value=fake_server),
            patch("scripts.setup_remote_collection.download_ngrok", return_value=False),
        ):
            exit_code = run.main(["--tunnel"])

        self.assertEqual(exit_code, 1)
        fake_app.run.assert_not_called()
        fake_server.shutdown.assert_called_once()
        fake_server.server_close.assert_called_once()

    def test_protected_server_must_bind_before_ngrok_is_called(self):
        import run

        fake_app = Mock()
        with (
            patch.object(run, "_resolve_tunnel_token", return_value=(TOKEN, True)),
            patch.object(run, "create_app", return_value=fake_app),
            patch.object(run, "_create_tunnel_server", side_effect=SystemExit(1)),
            patch("scripts.setup_remote_collection.download_ngrok") as download,
            patch("scripts.setup_remote_collection.start_tunnel") as start,
        ):
            exit_code = run.main(["--tunnel"])

        self.assertEqual(exit_code, 1)
        download.assert_not_called()
        start.assert_not_called()

    def test_standalone_tunnel_entrypoint_is_disabled(self):
        from scripts import setup_remote_collection

        with (
            patch.object(setup_remote_collection, "download_ngrok") as download,
            patch.object(setup_remote_collection, "start_tunnel") as start,
        ):
            self.assertEqual(setup_remote_collection.main(), 2)
        download.assert_not_called()
        start.assert_not_called()

    def test_ngrok_archive_extraction_ignores_archive_paths(self):
        from scripts.setup_remote_collection import _extract_ngrok_binary

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "ngrok.zip"
            destination = root / "bin" / "ngrok.exe"
            escaped_path = root / "escaped.txt"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../../ngrok.exe", b"expected-binary")
                archive.writestr("../../escaped.txt", b"must-not-be-extracted")

            _extract_ngrok_binary(archive_path, destination)

            self.assertEqual(destination.read_bytes(), b"expected-binary")
            self.assertFalse(escaped_path.exists())

    def test_ngrok_url_is_read_from_the_started_process_only(self):
        from scripts.setup_remote_collection import _NgrokLogMonitor

        stream = io.StringIO(
            '{"msg":"started tunnel","addr":"http://localhost:9999",'
            '"url":"https://wrong.ngrok.app"}\n'
            '{"msg":"started tunnel","addr":"http://localhost:8080",'
            '"url":"http://insecure.ngrok.app"}\n'
            '{"msg":"started tunnel","addr":"http://localhost:8080",'
            '"url":"https://correct.ngrok.app"}\n'
        )
        monitor = _NgrokLogMonitor(stream, 8080)
        self.assertTrue(monitor.ready.wait(timeout=1))
        self.assertEqual(monitor.public_url, "https://correct.ngrok.app")
        monitor.join()


if __name__ == "__main__":
    unittest.main()
