"""Ephemeral, target-free camera readiness gate for measurement capture.

The gate proves only that three distinct, consecutive frames can reach the
fixed CPU baseline and produce a face-backed sensor prediction.  It never
creates a schedule, selects a model, accepts target labels, or writes media to
disk.  A completed proof is single-use and is consumed before a measurement
run can be created.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import math
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .capture_contract import (
    compare_capture_contracts,
    normalize_capture_contract,
    validate_transport_frame,
)
from .measurement_schedule import canonical_sha256


PREFLIGHT_SCHEMA_VERSION = 1
PREFLIGHT_IMPLEMENTATION_ID = "target-free-camera-readiness-v1"
REQUIRED_CONSECUTIVE_SUCCESSES = 3
DEFAULT_TTL_SECONDS = 15 * 60
DEFAULT_MAX_ACTIVE = 64
MAX_PREFLIGHT_IMAGE_BYTES = 4 * 1024 * 1024
LOWER_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class MeasurementPreflightError(RuntimeError):
    """Base error for the in-memory readiness registry."""


class MeasurementPreflightAuthenticationError(MeasurementPreflightError):
    """Raised when a preflight token is absent, invalid, expired, or consumed."""


class MeasurementPreflightValidationError(MeasurementPreflightError, ValueError):
    """Raised when capture identity or viewport input is malformed."""


class MeasurementPreflightStateError(MeasurementPreflightError):
    """Raised when a readiness proof is incomplete or the registry is full."""


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalized_token(value: object) -> str:
    token = str(value or "")
    if not token or token != token.strip() or len(token) > 256:
        raise MeasurementPreflightAuthenticationError("preflight token is invalid")
    return token


def _positive_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise MeasurementPreflightValidationError(f"{field} must be positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementPreflightValidationError(
            f"{field} must be positive"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise MeasurementPreflightValidationError(f"{field} must be positive")
    return number


def normalize_preflight_viewport(viewport: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(viewport, Mapping):
        raise MeasurementPreflightValidationError("viewport must be an object")
    if set(viewport) != {"width", "height", "device_pixel_ratio"}:
        raise MeasurementPreflightValidationError("viewport fields are invalid")
    return {
        "width": _positive_finite(viewport.get("width"), field="viewport width"),
        "height": _positive_finite(viewport.get("height"), field="viewport height"),
        "device_pixel_ratio": _positive_finite(
            viewport.get("device_pixel_ratio"),
            field="device pixel ratio",
        ),
    }


def normalize_base_bundle_identity(bundle: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(bundle, Mapping):
        raise MeasurementPreflightValidationError(
            "base inference bundle must be an object"
        )
    identity: dict[str, str] = {}
    for field in (
        "model_id",
        "model_name",
        "model_sha256",
        "bundle_sha256",
        "checkpoint_sha256",
    ):
        value = str(bundle.get(field) or "")
        if field.endswith("sha256"):
            if not LOWER_SHA256_PATTERN.fullmatch(value):
                raise MeasurementPreflightValidationError(
                    f"base inference {field} is invalid"
                )
        elif not value or len(value) > 256:
            raise MeasurementPreflightValidationError(
                f"base inference {field} is invalid"
            )
        identity[field] = value
    if identity["model_sha256"] != identity["bundle_sha256"]:
        raise MeasurementPreflightValidationError(
            "base inference model and bundle SHA-256 differ"
        )
    return identity


def _default_frame_validator(
    image_data: str,
    capture_contract: Mapping[str, Any],
) -> str:
    """Decode one transport frame without retaining the bytes."""

    if not isinstance(image_data, str) or not image_data:
        raise MeasurementPreflightValidationError("image_data is required")
    encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise MeasurementPreflightValidationError("image_data is invalid") from exc
    if not raw or len(raw) > MAX_PREFLIGHT_IMAGE_BYTES:
        raise MeasurementPreflightValidationError("image payload size is invalid")

    # Keep heavy CV imports lazy so registry/source tests remain CPU-light.
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise MeasurementPreflightValidationError("image_data cannot be decoded")
    try:
        validate_transport_frame(
            capture_contract,
            frame_width_px=int(frame.shape[1]),
            frame_height_px=int(frame.shape[0]),
        )
    except ValueError as exc:
        raise MeasurementPreflightValidationError(str(exc)) from exc
    return hashlib.sha256(raw).hexdigest()


def _default_infer(root: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    from .inference import predict

    return predict(root, payload)


class MeasurementPreflightRegistry:
    """Bounded process-memory registry for single-use readiness proofs."""

    def __init__(
        self,
        data_root: Path,
        *,
        base_inference_bundle: Mapping[str, Any],
        infer: Callable[[Path, dict[str, Any]], tuple[dict[str, Any], int]] | None = None,
        frame_validator: Callable[[str, Mapping[str, Any]], str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_active: int = DEFAULT_MAX_ACTIVE,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.base_identity = normalize_base_bundle_identity(
            base_inference_bundle
        )
        self._infer = infer or _default_infer
        self._frame_validator = frame_validator or _default_frame_validator
        self._monotonic = monotonic
        self.ttl_seconds = _positive_finite(ttl_seconds, field="preflight TTL")
        if isinstance(max_active, bool) or not isinstance(max_active, int) or max_active < 1:
            raise MeasurementPreflightValidationError("max_active must be positive")
        self.max_active = max_active
        self._lock = threading.RLock()
        self._entries: dict[str, dict[str, Any]] = {}

    def start(
        self,
        *,
        capture_contract: Mapping[str, Any],
        viewport: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Start a target-free readiness check; return its secret exactly once."""

        try:
            normalized_contract = normalize_capture_contract(capture_contract)
        except ValueError as exc:
            raise MeasurementPreflightValidationError(str(exc)) from exc
        normalized_viewport = normalize_preflight_viewport(viewport)
        now = self._monotonic()
        with self._lock:
            self._sweep_unlocked(now)
            if len(self._entries) >= self.max_active:
                raise MeasurementPreflightStateError(
                    "too many active camera preflights; wait for expiry"
                )
            plaintext = f"wgmc_pf_{secrets.token_urlsafe(32)}"
            digest = _token_sha256(plaintext)
            preflight_id = f"PF-{secrets.token_hex(12)}"
            self._entries[digest] = {
                "preflight_id": preflight_id,
                "created_monotonic": now,
                "expires_monotonic": now + self.ttl_seconds,
                "capture_contract": normalized_contract,
                "capture_contract_sha256": canonical_sha256(normalized_contract),
                "viewport": normalized_viewport,
                "viewport_sha256": canonical_sha256(normalized_viewport),
                "base_inference_bundle": deepcopy(self.base_identity),
                "consecutive_successes": 0,
                "distinct_frame_sha256s": [],
                "last_frame_sha256": None,
                "last_result": None,
                "capture_contract_warnings": [],
                "ready": False,
            }
        return {
            "ok": True,
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "implementation_id": PREFLIGHT_IMPLEMENTATION_ID,
            "preflight_id": preflight_id,
            "preflight_token": plaintext,
            "preflight_token_returned_once": True,
            "required_consecutive_successes": REQUIRED_CONSECUTIVE_SUCCESSES,
            "target_free": True,
            "base_bundle_sha256": self.base_identity["bundle_sha256"],
            "persistent_storage_used": False,
            "measurement_claim_authorized": False,
        }

    def submit_frame(
        self,
        preflight_token: str,
        *,
        image_data: str,
        capture_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Check one distinct frame against the fixed CPU baseline."""

        digest = _token_sha256(_normalized_token(preflight_token))
        with self._lock:
            entry = self._authenticated_entry_unlocked(digest)
            try:
                observed_contract = normalize_capture_contract(capture_contract)
            except ValueError as exc:
                entry["consecutive_successes"] = 0
                raise MeasurementPreflightValidationError(str(exc)) from exc
            try:
                comparison = compare_capture_contracts(
                    entry["capture_contract"],
                    observed_contract,
                )
            except ValueError as exc:
                entry["consecutive_successes"] = 0
                raise MeasurementPreflightValidationError(str(exc)) from exc
            if comparison["compatible"] is not True:
                entry["consecutive_successes"] = 0
                raise MeasurementPreflightValidationError(
                    "capture contract changed during camera readiness preflight"
                )
            entry["capture_contract_warnings"] = sorted(
                {
                    *entry["capture_contract_warnings"],
                    *comparison.get("warnings", []),
                }
            )
            if entry["ready"] is True:
                return self._public_status(entry, idempotent=True)
            try:
                frame_sha256 = self._frame_validator(
                    image_data,
                    observed_contract,
                )
            except Exception:
                entry["consecutive_successes"] = 0
                raise
            if not isinstance(frame_sha256, str) or len(frame_sha256) != 64:
                raise MeasurementPreflightValidationError(
                    "frame validator did not return SHA-256"
                )
            if frame_sha256 == entry["last_frame_sha256"]:
                previous = entry.get("last_result")
                if not isinstance(previous, Mapping):
                    raise MeasurementPreflightStateError(
                        "preflight replay state is unavailable"
                    )
                return {**deepcopy(dict(previous)), "idempotent": True}
            if frame_sha256 in entry["distinct_frame_sha256s"]:
                entry["consecutive_successes"] = 0
                return {
                    **self._public_status(entry),
                    "ok": False,
                    "retryable": True,
                    "classification": "nonconsecutive_frame_replay",
                    "error": "preflight frame was already observed out of sequence",
                }

            inference_payload = {
                "image_data": image_data,
                "capture_contract": observed_contract,
                "model_name": "before",
                "viewport_width": entry["viewport"]["width"],
                "viewport_height": entry["viewport"]["height"],
                "allow_cuda": False,
            }
            try:
                response, status = self._infer(self.data_root, inference_payload)
                response = dict(response) if isinstance(response, Mapping) else {}
            except Exception:
                response = {
                    "ok": False,
                    "failure_stage": "preflight_inference_hard_error",
                    "error": "camera readiness inference failed",
                }
                status = 500
            entry["distinct_frame_sha256s"].append(frame_sha256)
            entry["last_frame_sha256"] = frame_sha256

            valid_success = (
                response.get("ok") is True
                and status == 200
                and response.get("model_name") == "before"
            )
            if valid_success:
                entry["consecutive_successes"] += 1
                entry["ready"] = (
                    entry["consecutive_successes"]
                    == REQUIRED_CONSECUTIVE_SUCCESSES
                )
                result = {
                    **self._public_status(entry),
                    "ok": True,
                    "retryable": False,
                    "classification": "ready" if entry["ready"] else "frame_passed",
                    "prediction_status": int(status),
                    "idempotent": False,
                }
            else:
                entry["consecutive_successes"] = 0
                classification = str(response.get("failure_stage") or "")
                if response.get("ok") is True:
                    classification = "preflight_inference_contract_error"
                    error = (
                        "camera readiness success response did not prove the "
                        "fixed baseline contract"
                    )
                elif classification == "attributable_sensor_failure":
                    error = "no face was detected in the readiness frame"
                else:
                    classification = classification or "preflight_inference_hard_error"
                    error = str(response.get("error") or "camera readiness inference failed")
                result = {
                    **self._public_status(entry),
                    "ok": False,
                    "retryable": True,
                    "classification": classification,
                    "prediction_status": int(status),
                    "error": error,
                    "idempotent": False,
                }
            entry["last_result"] = deepcopy(result)
            return result

    def consume(
        self,
        preflight_token: str,
        *,
        capture_contract: Mapping[str, Any],
        viewport: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Consume one complete proof before creating a measurement run."""

        try:
            normalized_contract = normalize_capture_contract(capture_contract)
        except ValueError as exc:
            raise MeasurementPreflightValidationError(str(exc)) from exc
        normalized_viewport = normalize_preflight_viewport(viewport)
        digest = _token_sha256(_normalized_token(preflight_token))
        with self._lock:
            entry = self._authenticated_entry_unlocked(digest)
            if (
                entry["ready"] is not True
                or entry["consecutive_successes"]
                != REQUIRED_CONSECUTIVE_SUCCESSES
            ):
                raise MeasurementPreflightStateError(
                    "three consecutive successful readiness frames are required"
                )
            try:
                comparison = compare_capture_contracts(
                    entry["capture_contract"],
                    normalized_contract,
                )
            except ValueError as exc:
                raise MeasurementPreflightValidationError(str(exc)) from exc
            if comparison["compatible"] is not True:
                raise MeasurementPreflightValidationError(
                    "capture contract differs from the completed preflight"
                )
            if not hmac.compare_digest(
                canonical_sha256(normalized_viewport),
                entry["viewport_sha256"],
            ):
                raise MeasurementPreflightValidationError(
                    "viewport differs from the completed preflight"
                )
            proof = {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "implementation_id": PREFLIGHT_IMPLEMENTATION_ID,
                "preflight_id": entry["preflight_id"],
                "capture_contract_sha256": entry["capture_contract_sha256"],
                "consumed_capture_contract_sha256": canonical_sha256(
                    normalized_contract
                ),
                "capture_contract_warnings": sorted(
                    {
                        *entry["capture_contract_warnings"],
                        *comparison.get("warnings", []),
                    }
                ),
                "viewport_sha256": entry["viewport_sha256"],
                "consecutive_successes": entry["consecutive_successes"],
                "distinct_frame_count": len(entry["distinct_frame_sha256s"]),
                "target_free": True,
                "persistent_storage_used": False,
                "fixed_model_name": "before",
                "base_inference_bundle": deepcopy(
                    entry["base_inference_bundle"]
                ),
                "allow_cuda": False,
                "measurement_claim_authorized": False,
            }
            receipt_core = deepcopy(proof)
            receipt_core["receipt_type"] = (
                "target_free_camera_readiness_receipt_v1"
            )
            receipt_core["distinct_frame_set_sha256"] = canonical_sha256(
                entry["distinct_frame_sha256s"]
            )
            proof = receipt_core
            proof["receipt_sha256"] = canonical_sha256(receipt_core)
            del self._entries[digest]
            return proof

    def _authenticated_entry_unlocked(self, digest: str) -> dict[str, Any]:
        now = self._monotonic()
        self._sweep_unlocked(now)
        entry = self._entries.get(digest)
        if entry is None:
            raise MeasurementPreflightAuthenticationError(
                "preflight token is invalid, expired, or already consumed"
            )
        return entry

    def _sweep_unlocked(self, now: float) -> None:
        expired = [
            digest
            for digest, entry in self._entries.items()
            if float(entry["expires_monotonic"]) <= now
        ]
        for digest in expired:
            del self._entries[digest]

    @staticmethod
    def _public_status(
        entry: Mapping[str, Any],
        *,
        idempotent: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "implementation_id": PREFLIGHT_IMPLEMENTATION_ID,
            "preflight_id": entry["preflight_id"],
            "consecutive_successes": int(entry["consecutive_successes"]),
            "required_consecutive_successes": REQUIRED_CONSECUTIVE_SUCCESSES,
            "distinct_frame_count": len(entry["distinct_frame_sha256s"]),
            "ready": entry["ready"] is True,
            "capture_contract_warnings": list(
                entry["capture_contract_warnings"]
            ),
            "target_free": True,
            "base_bundle_sha256": entry["base_inference_bundle"][
                "bundle_sha256"
            ],
            "persistent_storage_used": False,
            "measurement_claim_authorized": False,
            "idempotent": idempotent,
        }
