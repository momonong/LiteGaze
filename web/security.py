"""Security guardrails for explicitly public LexiGaze tunnel sessions.

Local development remains unchanged.  These controls are installed only when
``create_app(tunnel_mode=True)`` is used by ``run.py --tunnel``.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit, urlunsplit

from flask import Flask, Response, g, jsonify, request

LOGIN_PATH = "/_tunnel/login"
SESSION_PATH = "/_tunnel/session"
AUTH_COOKIE_NAME = "__Host-lexigaze_tunnel"
MIN_TOKEN_BYTES = 32
MAX_TOKEN_BYTES = 512

_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_EXPENSIVE_EXACT_PATHS = frozenset(
    {
        "/api/gaze/train",
        "/api/train",
        "/api/demo/upload_video",
        "/api/inspector/analyze",
        "/api/inspector/quiz",
        "/api/inspector/adaptive/next",
        "/api/inspector/adaptive/report",
    }
)
_REALTIME_EXACT_PATHS = frozenset({"/api/gaze/predict", "/api/predict"})
_EXPENSIVE_PREFIXES = (
    "/api/cognitive/analyze/",
    "/api/cognitive/warmup",
    "/api/gaze/analyze_reading_video",
)


def validate_tunnel_token(token: str | None) -> str:
    """Return a valid token or raise before any public tunnel is opened."""
    if not isinstance(token, str) or not token:
        raise RuntimeError(
            "Tunnel mode requires a security token. Set LEXIGAZE_TUNNEL_TOKEN "
            "or let run.py generate an ephemeral token."
        )
    if token != token.strip():
        raise RuntimeError("LEXIGAZE_TUNNEL_TOKEN must not start or end with whitespace.")
    token_size = len(token.encode("utf-8"))
    if token_size < MIN_TOKEN_BYTES:
        raise RuntimeError(
            f"LEXIGAZE_TUNNEL_TOKEN must be at least {MIN_TOKEN_BYTES} UTF-8 bytes."
        )
    if token_size > MAX_TOKEN_BYTES:
        raise RuntimeError(
            f"LEXIGAZE_TUNNEL_TOKEN must not exceed {MAX_TOKEN_BYTES} UTF-8 bytes."
        )
    return token


def build_tunnel_bootstrap_url(public_url: str, next_path: str, token: str) -> str:
    """Build a fragment-based login URL so credentials never enter HTTP logs."""
    validated_token = validate_tunnel_token(token)
    safe_next = _safe_next_path(next_path)
    base = public_url.rstrip("/")
    fragment = urlencode({"token": validated_token, "next": safe_next})
    return f"{base}{LOGIN_PATH}#{fragment}"


def _safe_next_path(candidate: object) -> str:
    """Restrict post-login navigation to a same-origin absolute path."""
    if (
        not isinstance(candidate, str)
        or not candidate.startswith("/")
        or candidate.startswith("//")
        or "\\" in candidate
        or any(ord(character) < 32 for character in candidate)
    ):
        return "/"
    parts = urlsplit(candidate)
    if parts.scheme or parts.netloc or not parts.path.startswith("/"):
        return "/"
    return urlunsplit(("", "", parts.path, parts.query, ""))


class _SlidingWindowLimiter:
    """Small process-local limiter; intentionally global to resist IP spoofing."""

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("rate limit and window must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def consume(self) -> tuple[bool, int]:
        now = self._clock()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._events and self._events[0] <= cutoff:
                self._events.popleft()
            if len(self._events) >= self.limit:
                retry_after = max(1, math.ceil(self._events[0] + self.window_seconds - now))
                return False, retry_after
            self._events.append(now)
        return True, 0


@dataclass(frozen=True)
class TunnelSecurityConfig:
    max_content_length: int
    mutation_limit: int
    realtime_limit: int
    expensive_limit: int
    auth_attempt_limit: int
    rate_window_seconds: int
    expensive_concurrency: int
    session_ttl_seconds: int


class TunnelSecurity:
    """Authenticate and budget requests arriving through a public tunnel."""

    _SLOT_FLAG = "_lexigaze_tunnel_expensive_slot"

    def __init__(self, token: str, config: TunnelSecurityConfig) -> None:
        token_bytes = validate_tunnel_token(token).encode("utf-8")
        self._token_digest = hashlib.sha256(token_bytes).digest()
        self._cookie_signing_key = hmac.new(
            token_bytes,
            b"lexigaze-tunnel-cookie-key-v1",
            hashlib.sha256,
        ).digest()
        self.config = config
        self._mutation_limiter = _SlidingWindowLimiter(
            config.mutation_limit, config.rate_window_seconds
        )
        self._realtime_limiter = _SlidingWindowLimiter(
            config.realtime_limit, config.rate_window_seconds
        )
        self._expensive_limiter = _SlidingWindowLimiter(
            config.expensive_limit, config.rate_window_seconds
        )
        self._auth_limiter = _SlidingWindowLimiter(
            config.auth_attempt_limit, config.rate_window_seconds
        )
        self._expensive_slots = threading.BoundedSemaphore(config.expensive_concurrency)

    def install(self, app: Flask) -> None:
        app.extensions["lexigaze_tunnel_security"] = self
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_request(self.teardown_request)

        app.add_url_rule(LOGIN_PATH, "tunnel_login", self.login_page, methods=["GET"])
        app.add_url_rule(SESSION_PATH, "tunnel_session", self.create_session, methods=["POST"])
        app.add_url_rule(SESSION_PATH, "tunnel_logout", self.delete_session, methods=["DELETE"])

    def before_request(self) -> Response | tuple[Response, int] | None:
        if request.content_length is not None and request.content_length > self.config.max_content_length:
            return self._json_error(
                "request_too_large",
                413,
                max_bytes=self.config.max_content_length,
            )

        if request.path == LOGIN_PATH and request.method in {"GET", "HEAD"}:
            return None

        if request.path == SESSION_PATH and request.method == "POST":
            return None

        if not self._is_authenticated():
            return self._json_error("authentication_required", 401)

        is_realtime = self._is_realtime_request()
        if request.method in _MUTATING_METHODS and not is_realtime:
            allowed, retry_after = self._mutation_limiter.consume()
            if not allowed:
                return self._rate_limited("mutation_rate_limit_exceeded", retry_after)

        if is_realtime:
            allowed, retry_after = self._realtime_limiter.consume()
            limit_error = "realtime_rate_limit_exceeded"
        elif self._is_expensive_request():
            allowed, retry_after = self._expensive_limiter.consume()
            limit_error = "expensive_rate_limit_exceeded"
        else:
            return None

        if not allowed:
            return self._rate_limited(limit_error, retry_after)
        if not self._expensive_slots.acquire(blocking=False):
            return self._rate_limited("expensive_operation_in_progress", 1)
        setattr(g, self._SLOT_FLAG, True)

        return None

    def after_request(self, response: Response) -> Response:
        self._release_expensive_slot()
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(self), microphone=(), geolocation=()",
        )
        if request.path.startswith("/api/") or request.path.startswith("/_tunnel/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def teardown_request(self, _error: BaseException | None) -> None:
        self._release_expensive_slot()

    def login_page(self) -> Response:
        nonce = secrets.token_urlsafe(18)
        page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LexiGaze secure tunnel</title>
  <style nonce="__CSP_NONCE__">
    body{font:16px system-ui,sans-serif;max-width:42rem;margin:12vh auto;padding:0 1.25rem;color:#172033}
    main{border:1px solid #ccd3df;border-radius:12px;padding:1.5rem} .error{color:#a11}
  </style>
</head>
<body><main><h1>LexiGaze</h1><p id="status">Authenticating secure tunnel…</p></main>
<script nonce="__CSP_NONCE__">
(async () => {
  const status = document.getElementById('status');
  const values = new URLSearchParams(location.hash.slice(1));
  const token = values.get('token');
  const next = values.get('next') || '/';
  history.replaceState(null, '', location.pathname);
  if (!token) { status.className='error'; status.textContent='This access link is incomplete.'; return; }
  try {
    const response = await fetch('/_tunnel/session', {
      method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token, next})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Authentication failed');
    location.replace(result.next || '/');
  } catch (error) {
    status.className='error'; status.textContent=error.message || 'Authentication failed';
  }
})();
</script></body></html>""".replace("__CSP_NONCE__", nonce)
        response = Response(page, content_type="text/html; charset=utf-8")
        response.headers["Content-Security-Policy"] = (
            f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'"
        )
        return response

    def create_session(self) -> tuple[Response, int] | Response:
        body = request.get_json(silent=True)
        candidate = body.get("token") if isinstance(body, dict) else None
        if not self._valid_raw_token(candidate):
            allowed, retry_after = self._auth_limiter.consume()
            if not allowed:
                return self._rate_limited("too_many_auth_attempts", retry_after)
            return self._json_error("invalid_credentials", 401)

        next_path = _safe_next_path(body.get("next", "/"))
        response = jsonify({"ok": True, "next": next_path})
        response.set_cookie(
            AUTH_COOKIE_NAME,
            self._make_session_cookie(),
            max_age=self.config.session_ttl_seconds,
            secure=True,
            httponly=True,
            samesite="Strict",
            path="/",
        )
        return response

    def delete_session(self) -> Response:
        response = jsonify({"ok": True})
        response.delete_cookie(
            AUTH_COOKIE_NAME,
            secure=True,
            httponly=True,
            samesite="Strict",
            path="/",
        )
        return response

    def _is_authenticated(self) -> bool:
        authorization = request.headers.get("Authorization")
        if authorization is not None:
            scheme, separator, value = authorization.partition(" ")
            return bool(separator and scheme.lower() == "bearer" and self._valid_raw_token(value))

        header_token = request.headers.get("X-LexiGaze-Token")
        if header_token is not None:
            return self._valid_raw_token(header_token)

        return self._valid_session_cookie(request.cookies.get(AUTH_COOKIE_NAME))

    def _valid_raw_token(self, candidate: object) -> bool:
        if not isinstance(candidate, str):
            return False
        candidate_bytes = candidate.encode("utf-8")
        if len(candidate_bytes) > MAX_TOKEN_BYTES:
            return False
        return hmac.compare_digest(hashlib.sha256(candidate_bytes).digest(), self._token_digest)

    def _make_session_cookie(self) -> str:
        expires_at = int(time.time()) + self.config.session_ttl_seconds
        payload = str(expires_at)
        signature = hmac.new(
            self._cookie_signing_key,
            f"session:{payload}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload}.{signature}"

    def _valid_session_cookie(self, candidate: object) -> bool:
        if not isinstance(candidate, str) or len(candidate) > 128:
            return False
        try:
            expires_text, signature = candidate.split(".", 1)
            expires_at = int(expires_text)
        except (TypeError, ValueError):
            return False
        now = int(time.time())
        if expires_at < now or expires_at > now + self.config.session_ttl_seconds + 60:
            return False
        expected = hmac.new(
            self._cookie_signing_key,
            f"session:{expires_text}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _is_expensive_request(self) -> bool:
        path = request.path.rstrip("/") or "/"
        if path in _EXPENSIVE_EXACT_PATHS or any(path.startswith(prefix) for prefix in _EXPENSIVE_PREFIXES):
            return True
        return path.startswith("/api/gaze/datasets/") and path.endswith("/reprocess")

    def _is_realtime_request(self) -> bool:
        return (request.path.rstrip("/") or "/") in _REALTIME_EXACT_PATHS

    def _release_expensive_slot(self) -> None:
        if getattr(g, self._SLOT_FLAG, False):
            setattr(g, self._SLOT_FLAG, False)
            self._expensive_slots.release()

    @staticmethod
    def _json_error(code: str, status: int, **details: object) -> tuple[Response, int]:
        payload: dict[str, object] = {"ok": False, "error": code}
        payload.update(details)
        return jsonify(payload), status

    def _rate_limited(self, code: str, retry_after: int) -> tuple[Response, int]:
        response, status = self._json_error(
            code,
            429,
            retry_after_seconds=retry_after,
        )
        response.headers["Retry-After"] = str(retry_after)
        return response, status


def install_tunnel_security(app: Flask, token: str, config: TunnelSecurityConfig) -> TunnelSecurity:
    security = TunnelSecurity(token, config)
    security.install(app)
    return security
