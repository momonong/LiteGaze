"""Dedicated HTTP surface for the frozen webcam measurement-ceiling run.

This blueprint is intentionally separate from the participant-study routes.
It accepts only server-scheduled measurement attempts and delegates all
persistent state to ``MeasurementRunner``.  Browser-only gate evidence is
validated here, then discarded before inference and ledger construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, TypeVar

from flask import Blueprint, Response, current_app, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from core.gaze_core.measurement_browser_gate import (
    MeasurementBrowserGateError,
    validate_measurement_browser_gate,
)
from core.gaze_core.measurement_preflight import (
    MeasurementPreflightAuthenticationError,
    MeasurementPreflightStateError,
    MeasurementPreflightValidationError,
)
from core.gaze_core.measurement_run_store import (
    MeasurementRunAuthenticationError,
    MeasurementRunChallengeError,
    MeasurementRunIntegrityError,
    MeasurementRunStateError,
    MeasurementRunValidationError,
)
from core.gaze_core.measurement_runner import MeasurementRunnerError
from web.measurement_surface_security import json_object_rejection


measurement_bp = Blueprint("measurement", __name__)

API_PREFIX = "/api/measurement-ceiling"
RUN_ID_HEADER = "X-Lexigaze-Measurement-Run-Id"
RUN_TOKEN_HEADER = "X-Lexigaze-Measurement-Run-Token"
CHALLENGE_TOKEN_HEADER = "X-Lexigaze-Measurement-Challenge-Token"
PREFLIGHT_TOKEN_HEADER = "X-Lexigaze-Measurement-Preflight-Token"
CREATE_REQUEST_ID_HEADER = "X-Lexigaze-Measurement-Create-Request-Id"

RUNNER_EXTENSION = "lexigaze_measurement_runner"
PREFLIGHT_EXTENSION = "lexigaze_measurement_preflight"

_BODY_SECRET_FIELDS = frozenset(
    {
        "challenge_token",
        "create_request_id",
        "plaintext_token",
        "preflight_token",
        "run_token",
    }
)


class MeasurementHttpValidationError(ValueError):
    """Raised for a malformed request at the dedicated HTTP boundary."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


Endpoint = TypeVar("Endpoint", bound=Callable[..., Response])


def _runner() -> Any:
    runner = current_app.extensions.get(RUNNER_EXTENSION)
    if runner is None:
        raise RuntimeError("measurement runner is unavailable")
    return runner


def _preflight_registry() -> Any:
    registry = current_app.extensions.get(PREFLIGHT_EXTENSION)
    if registry is None:
        raise RuntimeError("measurement preflight registry is unavailable")
    return registry


def _required_header(name: str) -> str:
    value = request.headers.get(name, "")
    if (
        not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise MeasurementHttpValidationError(f"{name} is required")
    return value


def _optional_header(name: str) -> str | None:
    value = request.headers.get(name)
    if value is None:
        return None
    return _required_header(name)


def _json_object(*, exact_fields: frozenset[str]) -> dict[str, Any]:
    parsed = request.get_json(silent=True)
    rejection = json_object_rejection(
        mimetype=request.mimetype,
        parsed_json=parsed,
    )
    if rejection is not None:
        status, detail = rejection
        raise MeasurementHttpValidationError(detail, status=status)
    body = dict(parsed)
    if set(body) != exact_fields:
        raise MeasurementHttpValidationError("request JSON fields are invalid")
    return body


def _run_authority() -> tuple[str, str]:
    return _required_header(RUN_ID_HEADER), _required_header(RUN_TOKEN_HEADER)


def _without_body_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_body_secrets(nested)
            for key, nested in value.items()
            if str(key) not in _BODY_SECRET_FIELDS
        }
    if isinstance(value, list):
        return [_without_body_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_without_body_secrets(item) for item in value]
    return deepcopy(value)


def _normalized_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_without_body_secrets(payload))
    detail = result.get("detail")
    if isinstance(detail, str) and detail and "error" not in result:
        result["error"] = detail

    failure = result.get("failure")
    failure_code = failure.get("code") if isinstance(failure, Mapping) else None
    if (
        result.get("classification") == "negative_calibration_result"
        or failure_code == "calibration_usable_count_below_65"
    ):
        failure_cleanup = (
            failure.get("cleanup") if isinstance(failure, Mapping) else None
        )
        runner_status = result.get("runner")
        result["classification"] = "calibration_unusable_negative_result"
        result.setdefault("calibration_attempt_count", 65)
        if "calibration_usable_count" not in result:
            result["calibration_usable_count"] = result.get(
                "usable_calibration_count",
                runner_status.get("calibration_usable_manifest_count", 0)
                if isinstance(runner_status, Mapping)
                else 0,
            )
        if "cleanup_verified" not in result and isinstance(
            failure_cleanup, Mapping
        ):
            result["cleanup_verified"] = bool(
                failure_cleanup.get("cleanup_verified") is True
            )
        if "images_purged" not in result and isinstance(failure_cleanup, Mapping):
            result["images_purged"] = bool(
                failure_cleanup.get("image_directories_absent") is True
            )
        if "purge_verified" not in result:
            result["purge_verified"] = bool(
                result.get("images_purged") is True
                and result.get("cleanup_verified") is True
            )
        result["terminal"] = True
        result["measurement_claim_authorized"] = False
    elif result.get("phase") == "failed_integrity":
        result.setdefault("classification", "abort_required")
        result["abort_required"] = True
        result["terminal"] = True
        result["measurement_claim_authorized"] = False
    return result


def _response(
    payload: Mapping[str, Any],
    *,
    status: int = 200,
    run_id: str | None = None,
    run_token: str | None = None,
    challenge_token: str | None = None,
    preflight_token: str | None = None,
) -> Response:
    response = jsonify(_normalized_payload(payload))
    response.status_code = status
    if run_id is not None:
        response.headers[RUN_ID_HEADER] = run_id
    if run_token is not None:
        response.headers[RUN_TOKEN_HEADER] = run_token
    if challenge_token is not None:
        response.headers[CHALLENGE_TOKEN_HEADER] = challenge_token
    if preflight_token is not None:
        response.headers[PREFLIGHT_TOKEN_HEADER] = preflight_token
    return response


def _error_response(
    *,
    status: int,
    classification: str,
    error: str,
    **fields: Any,
) -> Response:
    return _response(
        {
            "ok": False,
            "classification": classification,
            "error": error,
            "measurement_claim_authorized": False,
            **fields,
        },
        status=status,
    )


def _api_endpoint(function: Endpoint) -> Endpoint:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Response:
        try:
            return function(*args, **kwargs)
        except MeasurementHttpValidationError as exc:
            return _error_response(
                status=exc.status,
                classification="http_validation_error",
                error=str(exc),
            )
        except (
            MeasurementRunAuthenticationError,
            MeasurementPreflightAuthenticationError,
        ):
            return _error_response(
                status=401,
                classification="authentication_failed",
                error="measurement authority is invalid or expired",
            )
        except MeasurementRunChallengeError as exc:
            return _error_response(
                status=409,
                classification="challenge_conflict",
                error=str(exc),
            )
        except (MeasurementRunStateError, MeasurementPreflightStateError) as exc:
            return _error_response(
                status=409,
                classification="measurement_state_conflict",
                error=str(exc),
            )
        except MeasurementRunIntegrityError:
            return _error_response(
                status=409,
                classification="integrity_failure",
                error="persistent measurement integrity verification failed",
                abort_required=True,
                terminal=True,
            )
        except (
            MeasurementRunValidationError,
            MeasurementPreflightValidationError,
            MeasurementBrowserGateError,
        ) as exc:
            return _error_response(
                status=400,
                classification="measurement_validation_error",
                error=str(exc),
            )
        except MeasurementRunnerError as exc:
            return _error_response(
                status=409,
                classification="runner_contract_error",
                error=str(exc),
            )
        except HTTPException:
            raise
        except Exception:
            if current_app.testing:
                raise
            return _error_response(
                status=500,
                classification="internal_hard_error",
                error="measurement service failed closed",
            )

    return wrapped  # type: ignore[return-value]


def _viewport_arguments(viewport: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(viewport, Mapping):
        raise MeasurementHttpValidationError("viewport must be an object")
    if set(viewport) != {"width", "height", "device_pixel_ratio"}:
        raise MeasurementHttpValidationError("viewport fields are invalid")
    return {
        "viewport_width": viewport["width"],
        "viewport_height": viewport["height"],
        "device_pixel_ratio": viewport["device_pixel_ratio"],
    }


def _challenge_response(payload: Mapping[str, Any]) -> Response:
    token = str(payload.get("challenge_token") or "")
    if not token:
        raise MeasurementRunIntegrityError("challenge token is unavailable")
    return _response(payload, challenge_token=token)


def _capture_next_action(payload: Mapping[str, Any]) -> str | None:
    if payload.get("consumed") is not True:
        return None
    receipt = payload.get("receipt")
    progress = receipt.get("progress") if isinstance(receipt, Mapping) else None
    ordinal = (
        progress.get("next_sequence_index")
        if isinstance(progress, Mapping)
        else None
    )
    if ordinal == 65:
        return "finalize_calibration"
    if ordinal == 193:
        return "verify_artifact"
    return None


@measurement_bp.get("/measurement-ceiling")
def measurement_page() -> str:
    return render_template("measurement_ceiling.html")


@measurement_bp.get(f"{API_PREFIX}/health")
@_api_endpoint
def health() -> Response:
    return _response(
        {
            "ok": True,
            "service": "webcam-gaze-measurement-ceiling-v1",
            "loopback_only": True,
            "cpu_only": True,
            "offline": True,
            "measurement_claim_authorized": False,
        }
    )


@measurement_bp.post(f"{API_PREFIX}/preflight")
@_api_endpoint
def start_preflight() -> Response:
    body = _json_object(exact_fields=frozenset({"capture_contract", "viewport"}))
    result = _preflight_registry().start(
        capture_contract=body["capture_contract"],
        viewport=body["viewport"],
    )
    token = str(result.get("preflight_token") or "")
    if not token:
        raise RuntimeError("preflight token is unavailable")
    return _response(result, preflight_token=token)


@measurement_bp.post(f"{API_PREFIX}/preflight/frames")
@_api_endpoint
def submit_preflight_frame() -> Response:
    body = _json_object(
        exact_fields=frozenset({"image_data", "capture_contract"})
    )
    result = _preflight_registry().submit_frame(
        _required_header(PREFLIGHT_TOKEN_HEADER),
        image_data=body["image_data"],
        capture_contract=body["capture_contract"],
    )
    return _response(result, status=200 if result.get("ok") is True else 409)


@measurement_bp.post(f"{API_PREFIX}/runs")
@_api_endpoint
def create_run() -> Response:
    body = _json_object(exact_fields=frozenset({"capture_contract", "viewport"}))
    create_request_id = _required_header(CREATE_REQUEST_ID_HEADER)
    run_token = _required_header(RUN_TOKEN_HEADER)
    runner = _runner()
    lookup = getattr(runner, "lookup_create_request", None)
    if not callable(lookup):
        return _error_response(
            status=503,
            classification="runner_lookup_unavailable",
            error="measurement create recovery is unavailable",
        )

    existing = lookup(
        create_request_id=create_request_id,
        run_token=run_token,
    )
    readiness_preflight: Mapping[str, Any] | None = None
    existing_runtime = (
        existing.get("runner", {}).get("runtime_binding")
        if isinstance(existing.get("runner"), Mapping)
        else None
    )
    needs_preflight = existing.get("exists") is not True or not isinstance(
        existing_runtime, Mapping
    )
    if needs_preflight:
        preflight_token = _optional_header(PREFLIGHT_TOKEN_HEADER)
        if preflight_token is None:
            return _error_response(
                status=409,
                classification="pending_create_preflight_required",
                error="camera readiness preflight must be replaced",
                existing_run=False,
                authority_retained=True,
                replace_preflight_allowed=True,
            )
        try:
            readiness_preflight = _preflight_registry().consume(
                preflight_token,
                capture_contract=body["capture_contract"],
                viewport=body["viewport"],
            )
        except (
            MeasurementPreflightAuthenticationError,
            MeasurementPreflightStateError,
        ):
            # Close the race where a duplicate request created the run while
            # this request was checking the ephemeral preflight registry.
            existing = lookup(
                create_request_id=create_request_id,
                run_token=run_token,
            )
            recovered_runner = existing.get("runner")
            recovered_runtime = (
                recovered_runner.get("runtime_binding")
                if isinstance(recovered_runner, Mapping)
                else None
            )
            if existing.get("exists") is not True or not isinstance(
                recovered_runtime, Mapping
            ):
                return _error_response(
                    status=409,
                    classification="pending_create_preflight_required",
                    error="camera readiness preflight must be replaced",
                    existing_run=False,
                    authority_retained=True,
                    replace_preflight_allowed=True,
                )

    result = runner.create_run(
        create_request_id=create_request_id,
        run_token=run_token,
        capture_contract=body["capture_contract"],
        readiness_preflight=readiness_preflight,
        **_viewport_arguments(body["viewport"]),
    )
    run_id = str(result.get("capture_run_id") or "")
    if not run_id:
        raise MeasurementRunIntegrityError("created run identifier is unavailable")
    return _response(result, run_id=run_id)


@measurement_bp.get(f"{API_PREFIX}/status")
@_api_endpoint
def status() -> Response:
    run_id, run_token = _run_authority()
    runner = _runner()
    result = dict(runner.get_status(run_id, run_token))
    challenge_token = _optional_header(CHALLENGE_TOKEN_HEADER)
    if challenge_token is not None:
        inspected = runner.inspect_challenge(run_id, run_token, challenge_token)
        result["challenge_recovery"] = inspected
    return _response(result)


@measurement_bp.post(f"{API_PREFIX}/challenges")
@_api_endpoint
def issue_challenge() -> Response:
    _json_object(exact_fields=frozenset())
    run_id, run_token = _run_authority()
    runner = _runner()
    current = runner.get_status(run_id, run_token)
    phase = current.get("phase")
    if phase == "calibration_sealed":
        return _error_response(
            status=409,
            classification="calibration_finalize_required",
            error="calibration must be finalized before evaluation",
        )
    if phase in {"capture_sealed", "artifact_verified"}:
        return _error_response(
            status=409,
            classification="capture_complete",
            error="all 193 scheduled attempts are complete",
        )
    if current.get("challenge_outstanding") is True:
        return _error_response(
            status=409,
            classification="challenge_recovery_required",
            error="an authenticated outstanding challenge must be resumed or rotated",
        )
    return _challenge_response(runner.issue_next_challenge(run_id, run_token))


@measurement_bp.post(f"{API_PREFIX}/challenges/rotate")
@_api_endpoint
def rotate_challenge() -> Response:
    _json_object(exact_fields=frozenset())
    run_id, run_token = _run_authority()
    rotate = getattr(_runner(), "rotate_unconsumed_challenge", None)
    if not callable(rotate):
        return _error_response(
            status=503,
            classification="challenge_rotation_unavailable",
            error="challenge recovery is unavailable",
        )
    return _challenge_response(rotate(run_id, run_token))


@measurement_bp.post(f"{API_PREFIX}/captures")
@_api_endpoint
def submit_capture() -> Response:
    parsed = request.get_json(silent=True)
    rejection = json_object_rejection(
        mimetype=request.mimetype,
        parsed_json=parsed,
    )
    if rejection is not None:
        raise MeasurementHttpValidationError(rejection[1], status=rejection[0])
    body = dict(parsed)
    normal_fields = {"image_data", "capture_contract", "client_gate"}
    resume_fields = {"resume_server_spool"}
    body_fields = frozenset(body)
    if body_fields not in {frozenset(normal_fields), frozenset(resume_fields)}:
        raise MeasurementHttpValidationError("request JSON fields are invalid")

    run_id, run_token = _run_authority()
    challenge_token = _required_header(CHALLENGE_TOKEN_HEADER)
    runner = _runner()
    inspected = runner.inspect_challenge(run_id, run_token, challenge_token)
    resumed = body_fields == frozenset(resume_fields)

    if resumed:
        if body["resume_server_spool"] is not True:
            raise MeasurementHttpValidationError(
                "resume_server_spool must be true"
            )
        recoverable = (
            inspected.get("status") == "committed"
            or inspected.get("server_spool_available") is True
            or inspected.get("prepared_observation_pending") is True
        )
        if not recoverable:
            return _error_response(
                status=409,
                classification="server_spool_unavailable",
                error="no committed or server-spooled frame can be resumed",
                consumed=False,
                retryable=True,
                new_frame_retry_allowed=True,
            )
        current = runner.get_status(run_id, run_token)
        runtime = current.get("runner", {}).get("runtime_binding")
        if not isinstance(runtime, Mapping):
            raise MeasurementRunIntegrityError("runtime binding is unavailable")
        observed_contract = runtime.get("capture_contract")
        observed_viewport = runtime.get("viewport")
        if not isinstance(observed_contract, Mapping) or not isinstance(
            observed_viewport, Mapping
        ):
            raise MeasurementRunIntegrityError("runtime capture context is unavailable")
        image_data = None
    else:
        observed_contract = body["capture_contract"]
        if inspected.get("status") == "active":
            validate_measurement_browser_gate(
                body["client_gate"],
                schedule_row=inspected["schedule_row"],
                runtime_viewport=inspected["viewport"],
            )
            observed_viewport = inspected["viewport"]
        else:
            # A committed exact retry does not rerun inference and therefore
            # does not need to recover the already-discarded browser gate.
            current = runner.get_status(run_id, run_token)
            runtime = current.get("runner", {}).get("runtime_binding")
            if not isinstance(runtime, Mapping):
                raise MeasurementRunIntegrityError("runtime binding is unavailable")
            observed_viewport = runtime.get("viewport")
        if not isinstance(observed_viewport, Mapping):
            raise MeasurementRunIntegrityError("runtime viewport is unavailable")
        image_data = body["image_data"]

    result = dict(
        runner.submit_frame(
            run_id,
            run_token,
            challenge_token,
            image_data=image_data,
            observed_capture_contract=observed_contract,
            observed_viewport_width=observed_viewport["width"],
            observed_viewport_height=observed_viewport["height"],
            observed_device_pixel_ratio=observed_viewport["device_pixel_ratio"],
        )
    )
    if resumed:
        result["resumed_from_server_spool"] = True
    next_action = _capture_next_action(result)
    if next_action is not None:
        result["next_action"] = next_action
    http_status = 200 if result.get("ok") is True or result.get("consumed") is True else 409
    return _response(result, status=http_status)


@measurement_bp.post(f"{API_PREFIX}/calibration/finalize")
@_api_endpoint
def finalize_calibration() -> Response:
    _json_object(exact_fields=frozenset())
    run_id, run_token = _run_authority()
    result = _normalized_payload(
        _runner().finalize_calibration(run_id, run_token)
    )
    expected_negative = (
        result.get("classification") == "calibration_unusable_negative_result"
    )
    return _response(
        result,
        status=200 if result.get("ok") is True or expected_negative else 409,
    )


@measurement_bp.post(f"{API_PREFIX}/artifact/verify")
@_api_endpoint
def verify_artifact() -> Response:
    _json_object(exact_fields=frozenset())
    run_id, run_token = _run_authority()
    result = _runner().verify_artifact(run_id, run_token)
    return _response(result, status=200 if result.get("ok") is True else 409)


@measurement_bp.post(f"{API_PREFIX}/analysis")
@_api_endpoint
def analyze_verified_run() -> Response:
    _json_object(exact_fields=frozenset())
    run_id, run_token = _run_authority()
    result = dict(_runner().analyze_verified_run(run_id, run_token))
    result.update(
        {
            "ok": True,
            "classification": "integrity_verified_descriptive_analysis",
            "measurement_claim_authorized": False,
            "physical_capture_claim_authorized": False,
        }
    )
    return _response(result)


@measurement_bp.post(f"{API_PREFIX}/abort")
@_api_endpoint
def abort_and_cleanup() -> Response:
    body = _json_object(exact_fields=frozenset({"reason"}))
    reason = body["reason"]
    if not isinstance(reason, str) or not reason or len(reason) > 128:
        raise MeasurementHttpValidationError("abort reason is invalid")
    run_id, run_token = _run_authority()
    result = _runner().abort_and_cleanup(run_id, run_token, reason=reason)
    confirmed = result.get("cleanup_verified") is True
    return _response(result, status=200 if confirmed else 409)


__all__ = [
    "API_PREFIX",
    "CHALLENGE_TOKEN_HEADER",
    "CREATE_REQUEST_ID_HEADER",
    "PREFLIGHT_EXTENSION",
    "PREFLIGHT_TOKEN_HEADER",
    "RUNNER_EXTENSION",
    "RUN_ID_HEADER",
    "RUN_TOKEN_HEADER",
    "measurement_bp",
]
