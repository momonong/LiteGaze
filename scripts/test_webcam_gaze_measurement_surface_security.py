"""Pure-standard-library tests for the dedicated HTTP security boundary."""

from __future__ import annotations

import unittest

from web.measurement_surface_security import (
    DEFAULT_MEASUREMENT_AUTHORITY,
    MAX_MEASUREMENT_CONTENT_LENGTH,
    MEASUREMENT_ALLOWED_PATHS,
    MeasurementSurfaceSecurityError,
    json_object_rejection,
    measurement_security_headers,
    normalize_measurement_authority,
    request_policy_rejection,
)


def _accepted(**updates: object) -> tuple[int, str] | None:
    values: dict[str, object] = {
        "authority": DEFAULT_MEASUREMENT_AUTHORITY,
        "remote_addr": "127.0.0.1",
        "host": DEFAULT_MEASUREMENT_AUTHORITY,
        "path": "/measurement-ceiling",
        "method": "GET",
        "query_string": b"",
        "origin": None,
        "sec_fetch_site": "none",
    }
    values.update(updates)
    return request_policy_rejection(**values)


class MeasurementSurfaceSecurityTests(unittest.TestCase):
    def test_exact_authority_and_loopback_request_are_accepted(self) -> None:
        self.assertIsNone(_accepted())
        self.assertIsNone(
            _accepted(
                path="/api/measurement-ceiling/captures",
                method="POST",
                origin="http://127.0.0.1:8099",
                sec_fetch_site="same-origin",
            )
        )
        self.assertEqual(
            normalize_measurement_authority(DEFAULT_MEASUREMENT_AUTHORITY),
            DEFAULT_MEASUREMENT_AUTHORITY,
        )

    def test_dns_rebinding_and_authority_aliases_fail_closed(self) -> None:
        for host in (
            "localhost:8099",
            "evil.example:8099",
            "127.0.0.1",
            "127.0.0.1:8080",
            "user@127.0.0.1:8099",
            "127.0.0.1:8099.evil.example",
            "127.0.0.1:8099 ",
        ):
            with self.subTest(host=host):
                self.assertEqual(_accepted(host=host)[0], 421)
        for configured in (
            "localhost:8099",
            "0.0.0.0:8099",
            "[::1]:8099",
            "127.0.0.1:0",
            "127.0.0.1:65536",
            "http://127.0.0.1:8099",
        ):
            with self.subTest(configured=configured):
                with self.assertRaises(MeasurementSurfaceSecurityError):
                    normalize_measurement_authority(configured)

    def test_non_loopback_query_options_origin_and_cross_site_are_rejected(self) -> None:
        self.assertEqual(_accepted(remote_addr="192.168.1.8")[0], 403)
        self.assertEqual(_accepted(query_string=b"token=secret")[0], 400)
        self.assertEqual(_accepted(method="OPTIONS")[0], 405)
        self.assertEqual(
            _accepted(origin="http://evil.example", sec_fetch_site="cross-site")[0],
            403,
        )
        self.assertEqual(_accepted(sec_fetch_site="same-site")[0], 403)

    def test_only_measurement_page_api_and_exact_static_assets_are_allowed(self) -> None:
        self.assertIn("/measurement-ceiling", MEASUREMENT_ALLOWED_PATHS)
        self.assertIn(
            "/static/gaze_capture_contract.js",
            MEASUREMENT_ALLOWED_PATHS,
        )
        for path in (
            "/",
            "/gaze",
            "/study",
            "/api/gaze/predict",
            "/api/sessions",
            "/static/participant_study.js",
            "/favicon.ico",
        ):
            with self.subTest(path=path):
                self.assertEqual(_accepted(path=path)[0], 404)

    def test_json_mutations_require_application_json_object(self) -> None:
        self.assertIsNone(
            json_object_rejection(
                mimetype="application/json",
                parsed_json={},
            )
        )
        self.assertEqual(
            json_object_rejection(mimetype="text/plain", parsed_json={})[0],
            415,
        )
        for value in (None, [], "object", 1, True):
            with self.subTest(value=value):
                self.assertEqual(
                    json_object_rejection(
                        mimetype="application/json",
                        parsed_json=value,
                    )[0],
                    400,
                )
        self.assertEqual(MAX_MEASUREMENT_CONTENT_LENGTH, 12 * 1024 * 1024)

    def test_headers_are_no_store_camera_only_csp_without_cors(self) -> None:
        headers = measurement_security_headers()
        self.assertIn("no-store", headers["Cache-Control"])
        self.assertEqual(
            headers["Permissions-Policy"],
            "camera=(self), microphone=(), geolocation=()",
        )
        csp = headers["Content-Security-Policy"]
        self.assertIn("default-src 'self'", csp)
        self.assertIn("script-src 'self'", csp)
        self.assertIn("style-src 'self'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        self.assertNotIn("'unsafe-inline'", csp)
        self.assertFalse(any(key.lower().startswith("access-control-") for key in headers))


if __name__ == "__main__":
    unittest.main()
