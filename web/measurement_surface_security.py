"""Pure request policy for the dedicated local measurement surface."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Any


DEFAULT_MEASUREMENT_AUTHORITY = "127.0.0.1:8099"
MAX_MEASUREMENT_CONTENT_LENGTH = 12 * 1024 * 1024
AUTHORITY_PATTERN = re.compile(r"^127\.0\.0\.1:([1-9][0-9]{0,4})$")

MEASUREMENT_PAGE_PATH = "/measurement-ceiling"
MEASUREMENT_API_PATHS = frozenset(
    {
        "/api/measurement-ceiling/health",
        "/api/measurement-ceiling/preflight",
        "/api/measurement-ceiling/preflight/frames",
        "/api/measurement-ceiling/runs",
        "/api/measurement-ceiling/status",
        "/api/measurement-ceiling/challenges",
        "/api/measurement-ceiling/challenges/rotate",
        "/api/measurement-ceiling/captures",
        "/api/measurement-ceiling/calibration/finalize",
        "/api/measurement-ceiling/artifact/verify",
        "/api/measurement-ceiling/analysis",
        "/api/measurement-ceiling/abort",
    }
)
MEASUREMENT_STATIC_PATHS = frozenset(
    {
        "/static/gaze_capture_contract.js",
        "/static/measurement_ceiling.css",
        "/static/measurement_ceiling.js",
        "/static/measurement_ceiling_gate.js",
        "/static/measurement_ceiling_client_policy.js",
    }
)
MEASUREMENT_ALLOWED_PATHS = frozenset(
    {
        MEASUREMENT_PAGE_PATH,
        *MEASUREMENT_API_PATHS,
        *MEASUREMENT_STATIC_PATHS,
    }
)

MEASUREMENT_SECURITY_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-DNS-Prefetch-Control": "off",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; connect-src 'self'; "
        "font-src 'self'; form-action 'none'; frame-ancestors 'none'; "
        "img-src 'self' data:; media-src 'self' blob:; object-src 'none'; "
        "script-src 'self'; style-src 'self'; worker-src 'none'"
    ),
}


class MeasurementSurfaceSecurityError(ValueError):
    """Raised when dedicated-surface configuration is unsafe."""


def normalize_measurement_authority(value: object) -> str:
    authority = str(value or "")
    match = AUTHORITY_PATTERN.fullmatch(authority)
    if match is None:
        raise MeasurementSurfaceSecurityError(
            "measurement authority must be exact 127.0.0.1:<port>"
        )
    port = int(match.group(1))
    if port > 65535:
        raise MeasurementSurfaceSecurityError("measurement authority port is invalid")
    return authority


def measurement_origin(authority: str) -> str:
    return f"http://{normalize_measurement_authority(authority)}"


def is_loopback_remote(value: object) -> bool:
    try:
        return ipaddress.ip_address(str(value or "")).is_loopback
    except ValueError:
        return False


def request_policy_rejection(
    *,
    authority: str,
    remote_addr: object,
    host: object,
    path: object,
    method: object,
    query_string: bytes | bytearray | memoryview | str | None,
    origin: object = None,
    sec_fetch_site: object = None,
) -> tuple[int, str] | None:
    """Return one fail-closed HTTP rejection, or ``None`` when accepted."""

    expected_authority = normalize_measurement_authority(authority)
    if not is_loopback_remote(remote_addr):
        return 403, "measurement surface is loopback-only"
    if str(host or "") != expected_authority:
        return 421, "measurement Host authority is invalid"
    request_path = str(path or "")
    if request_path not in MEASUREMENT_ALLOWED_PATHS:
        return 404, "endpoint is unavailable on the measurement surface"
    request_method = str(method or "").upper()
    if request_method == "OPTIONS":
        return 405, "CORS preflight is not supported"
    if query_string not in (None, b"", ""):
        return 400, "query parameters are forbidden on the measurement surface"
    supplied_origin = str(origin or "")
    if supplied_origin and supplied_origin != measurement_origin(expected_authority):
        return 403, "cross-origin request is forbidden"
    fetch_site = str(sec_fetch_site or "").lower()
    if fetch_site and fetch_site not in {"same-origin", "none"}:
        return 403, "cross-site request is forbidden"
    return None


def json_object_rejection(
    *,
    mimetype: object,
    parsed_json: Any,
) -> tuple[int, str] | None:
    if str(mimetype or "").lower() != "application/json":
        return 415, "Content-Type must be application/json"
    if not isinstance(parsed_json, Mapping):
        return 400, "request JSON body must be an object"
    return None


def measurement_security_headers() -> dict[str, str]:
    return dict(MEASUREMENT_SECURITY_HEADERS)
