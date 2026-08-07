from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

from core.gaze_core.model_registry import delete_model
from core.gaze_core.sample_store import delete_dataset, purge_session_images
from core.gaze_core.training import train_placeholder
from core.participant_study import (
    ParticipantStudyStore,
    StudyAuthorizationError,
    StudyNotReadyError,
    StudyStateError,
    StudyValidationError,
    audit_participant_calibration,
    public_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
study_bp = Blueprint("study", __name__)


def _store() -> ParticipantStudyStore:
    root = Path(current_app.config.get("LEXIGAZE_STUDY_ROOT") or ROOT)
    store = ParticipantStudyStore(root, settings=current_app.config)
    store.enforce_expired_calibration_retention()
    return store


def _access_token(body: dict | None = None) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return str((body or {}).get("access_token") or "").strip()


def _error_response(exc: Exception):
    if isinstance(exc, StudyAuthorizationError):
        status = 403
    elif isinstance(exc, (StudyNotReadyError, StudyStateError)):
        status = 409
    else:
        status = 400
    return jsonify(
        {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
    ), status


def _failed_calibration(
    store: ParticipantStudyStore,
    session_id: str,
    access_token: str,
    gaze_root: Path,
    gaze_session_id: str,
    quality: dict,
    *,
    model_name: str = "",
) -> dict:
    """Remove every derived artifact before reopening calibration for a retry."""

    quality["passed"] = False
    dataset_result = delete_dataset(gaze_root, gaze_session_id)
    quality["failed_dataset_deleted"] = dataset_result.get("ok") is True
    if model_name:
        model_result = delete_model(gaze_root, model_name)
        quality["failed_model_deleted"] = model_result.get("ok") is True
    return store.complete_calibration(session_id, access_token, quality)


@study_bp.after_request
def _study_no_store(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "camera=(self), microphone=(), geolocation=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'none'; connect-src 'self'; "
        "font-src 'self'; form-action 'self'; frame-ancestors 'none'; "
        "img-src 'self' data:; media-src 'self' blob:; "
        "script-src 'self'; style-src 'self' 'unsafe-inline'"
    )
    return response


@study_bp.get("/study")
def participant_study_page():
    return render_template("participant_study.html")


@study_bp.get("/study/assessment")
def participant_assessment_page():
    return render_template("participant_assessment.html")


@study_bp.get("/api/study/protocol")
def get_study_protocol():
    return jsonify({"ok": True, "protocol": public_protocol(current_app.config)})


@study_bp.post("/api/study/enroll")
def enroll_participant():
    body = request.get_json(force=True) or {}
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400
    try:
        return jsonify(_store().enroll(body)), 201
    except (
        StudyAuthorizationError,
        StudyNotReadyError,
        StudyStateError,
        StudyValidationError,
    ) as exc:
        return _error_response(exc)


@study_bp.get("/api/study/sessions/<session_id>")
def participant_session_status(session_id: str):
    try:
        session = _store().get_session(session_id, _access_token())
        return jsonify({"ok": True, "session": session})
    except (StudyAuthorizationError, StudyValidationError) as exc:
        return _error_response(exc)


@study_bp.get("/api/study/sessions/<session_id>/consent-receipt")
def participant_consent_receipt(session_id: str):
    try:
        receipt = _store().get_receipt(session_id, _access_token())
        return jsonify({"ok": True, "consent_receipt": receipt})
    except (StudyAuthorizationError, StudyValidationError) as exc:
        return _error_response(exc)


@study_bp.post("/api/study/sessions/<session_id>/system-check")
def participant_system_check(session_id: str):
    body = request.get_json(force=True) or {}
    try:
        session = _store().record_system_check(
            session_id,
            _access_token(body),
            body.get("checks", {}),
        )
        return jsonify({"ok": True, "session": session})
    except (
        StudyAuthorizationError,
        StudyStateError,
        StudyValidationError,
    ) as exc:
        return _error_response(exc)


@study_bp.post("/api/study/sessions/<session_id>/dry-run")
def advance_participant_dry_run(session_id: str):
    body = request.get_json(force=True) or {}
    try:
        session = _store().advance_dry_run(
            session_id,
            _access_token(body),
            str(body.get("action") or ""),
        )
        return jsonify({"ok": True, "session": session})
    except (
        StudyAuthorizationError,
        StudyStateError,
        StudyValidationError,
    ) as exc:
        return _error_response(exc)


@study_bp.post("/api/study/sessions/<session_id>/calibration/complete")
def complete_participant_calibration(session_id: str):
    body = request.get_json(force=True) or {}
    access_token = _access_token(body)
    gaze_session_id = str(body.get("gaze_session_id") or "").strip()
    store = _store()
    try:
        participant = store.get_session(session_id, access_token)
        if participant["mode"] != "pilot":
            raise StudyStateError("dry runs cannot process calibration data")
        if participant["state"] != "calibration_in_progress":
            raise StudyStateError("calibration is not in progress")
        if participant.get("linked_data", {}).get("gaze_session_id") != gaze_session_id:
            raise StudyStateError("gaze session is not linked to this participant")
        gaze_root = Path(current_app.config.get("LEXIGAZE_GAZE_ROOT") or ROOT)
        try:
            quality = audit_participant_calibration(
                gaze_root,
                gaze_session_id,
                expected_study_session_id=session_id,
            )
        except Exception as exc:
            quality = {
                "passed": False,
                "reasons": ["calibration_audit_error"],
                "failure_type": type(exc).__name__,
            }
            session = _failed_calibration(
                store,
                session_id,
                access_token,
                gaze_root,
                gaze_session_id,
                quality,
            )
            return jsonify(
                {
                    "ok": False,
                    "error": "calibration audit failed; temporary data was removed",
                    "quality": quality,
                    "session": session,
                }
            ), 422
        if not quality["passed"]:
            session = _failed_calibration(
                store,
                session_id,
                access_token,
                gaze_root,
                gaze_session_id,
                quality,
            )
            return jsonify(
                {
                    "ok": False,
                    "error": (
                        "calibration quality gate failed; temporary data was removed"
                    ),
                    "quality": quality,
                    "session": session,
                }
            ), 422

        model_name = f"{participant['participant_id'].lower()}_pilot_v1"
        try:
            training, training_status = train_placeholder(
                gaze_root,
                {
                    "data_session_id": gaze_session_id,
                    "base_model_name": "0",
                    "output_model_name": model_name,
                    "allow_cuda": False,
                },
            )
        except Exception as exc:
            quality["reasons"].append("personalization_training_error")
            quality["training"] = {
                "ok": False,
                "device": "cpu",
                "failure_type": type(exc).__name__,
            }
            session = _failed_calibration(
                store,
                session_id,
                access_token,
                gaze_root,
                gaze_session_id,
                quality,
                model_name=model_name,
            )
            return jsonify(
                {
                    "ok": False,
                    "error": "personalization failed; temporary data was removed",
                    "quality": quality,
                    "session": session,
                }
            ), 422
        quality["training"] = {
            "ok": training.get("ok") is True,
            "device": training.get("training_device"),
            "validation_px_error": training.get("best_val_px_error"),
            "validation_scheme": training.get("validation_scheme"),
        }
        if training_status != 200 or training.get("ok") is not True:
            quality["reasons"].append("personalization_training_failed")
            session = _failed_calibration(
                store,
                session_id,
                access_token,
                gaze_root,
                gaze_session_id,
                quality,
                model_name=model_name,
            )
            return jsonify(
                {
                    "ok": False,
                    "error": training.get("error", "training failed"),
                    "quality": quality,
                    "session": session,
                }
            ), 422

        try:
            purge = purge_session_images(gaze_root, gaze_session_id)
        except Exception as exc:
            quality["reasons"].append("calibration_image_purge_error")
            quality["purge_failure_type"] = type(exc).__name__
            session = _failed_calibration(
                store,
                session_id,
                access_token,
                gaze_root,
                gaze_session_id,
                quality,
                model_name=model_name,
            )
            return jsonify(
                {
                    "ok": False,
                    "error": (
                        "image purge failed; the full temporary dataset was removed"
                    ),
                    "quality": quality,
                    "session": session,
                }
            ), 422
        quality["calibration_images_purged"] = purge["ok"] is True
        quality["removed_image_directories"] = purge["removed_directories"]
        session = store.complete_calibration(
            session_id,
            access_token,
            quality,
            model_name=training["model_name"],
        )
        return jsonify({"ok": True, "quality": quality, "session": session})
    except (
        FileNotFoundError,
        StudyAuthorizationError,
        StudyStateError,
        StudyValidationError,
        ValueError,
    ) as exc:
        return _error_response(exc)


@study_bp.post("/api/study/withdraw")
def withdraw_participant():
    body = request.get_json(force=True) or {}
    try:
        result = _store().withdraw(
            str(body.get("study_session_id") or ""),
            access_token=_access_token(body),
            withdrawal_code=str(body.get("withdrawal_code") or "").strip(),
        )
        return jsonify(result)
    except (
        StudyAuthorizationError,
        StudyStateError,
        StudyValidationError,
    ) as exc:
        return _error_response(exc)
