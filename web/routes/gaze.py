from __future__ import annotations

import json
import os
import secrets
import time
import traceback
from pathlib import Path

import cv2
from flask import Blueprint, current_app, jsonify, request

from core.gaze_core.inference import predict
from core.gaze_core.model_registry import (
    delete_model,
    ensure_runs_dir,
    list_models,
    model_artifact_sha256,
    rename_model,
)
from core.gaze_core.motion_robustness import audit_payload, load_motion_samples
from core.gaze_core.capture_contract import (
    authoritative_participant_calibration_labels,
)
from core.gaze_core.sample_store import (
    create_session,
    delete_dataset,
    ensure_sessions_dir,
    get_preprocessor,
    list_datasets,
    read_session_metadata,
    rename_dataset,
    safe_session_dir,
    save_sample,
)
from core.gaze_core.training import train_placeholder
from core.participant_study import (
    ParticipantStudyStore,
    StudyAuthorizationError,
    StudyError,
    StudyStateError,
    StudyValidationError,
)

ROOT = Path(__file__).resolve().parents[2]
gaze_bp = Blueprint("gaze", __name__, url_prefix="/api/gaze")
gaze_api_bp = Blueprint("gaze_api", __name__, url_prefix="/api")


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _setting(name: str, default: str = "") -> str:
    if name in current_app.config:
        return str(current_app.config.get(name) or default).strip()
    return str(os.environ.get(name, default)).strip()


def _public_study_mode() -> bool:
    return _truthy(_setting("LEXIGAZE_PUBLIC_STUDY_MODE"))


def _study_store() -> ParticipantStudyStore:
    root = Path(current_app.config.get("LEXIGAZE_STUDY_ROOT") or ROOT)
    store = ParticipantStudyStore(root, settings=current_app.config)
    store.enforce_expired_calibration_retention()
    return store


def _gaze_root() -> Path:
    return Path(current_app.config.get("LEXIGAZE_GAZE_ROOT") or ROOT).resolve()


def _study_access_token(body: dict | None = None) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return str((body or {}).get("study_access_token") or "").strip()


def _study_session_id(body: dict | None = None) -> str:
    return str(
        (body or {}).get("study_session_id")
        or request.headers.get("X-Lexigaze-Study-Session")
        or ""
    ).strip()


def _researcher_authorized() -> bool:
    expected = _setting("LEXIGAZE_RESEARCHER_API_KEY")
    supplied = request.headers.get("X-Lexigaze-Researcher-Key", "")
    return bool(expected) and secrets.compare_digest(supplied, expected)


def _admin_blocked_response():
    return jsonify(
        {
            "ok": False,
            "error": "researcher operation is disabled on the public study surface",
        }
    ), 403


def _participant_session(
    body: dict | None = None,
) -> tuple[ParticipantStudyStore, dict]:
    study_session_id = _study_session_id(body)
    access_token = _study_access_token(body)
    if not study_session_id or not access_token:
        raise StudyError("study session credential is required")
    store = _study_store()
    return store, store.get_session(study_session_id, access_token)


def _authoritative_capture_payload(body: dict, metadata: dict) -> dict:
    """Bind sample provenance to the server-created capture session.

    Capture identity is a property of the dataset session, not of an
    individual browser request. Removing client values before copying the
    linked metadata prevents stale or forged sample payloads from creating a
    session/manifest provenance conflict.
    """

    resolved = dict(body)
    for field in ("capture_run_id", "capture_source", "source_session_id"):
        resolved.pop(field, None)
        value = str(metadata.get(field) or "").strip()
        if value:
            resolved[field] = value
    return resolved


def _create_gaze_session(body: dict) -> tuple[dict, int]:
    study_session_id = _study_session_id(body)
    if not study_session_id:
        if _public_study_mode():
            return {
                "ok": False,
                "error": "public collection requires a study session",
            }, 403
        return (
            create_session(
                _gaze_root(),
                body.get("participant_id", "anonymous"),
                capture_run_id=body.get("capture_run_id"),
                capture_source=body.get("capture_source", "direct-frame"),
                source_session_id=body.get("source_session_id"),
            ),
            200,
        )
    try:
        store, participant = _participant_session(body)
        if participant["mode"] not in {"pilot", "rehearsal"}:
            raise StudyError("dry runs cannot create gaze datasets")
        if participant["state"] != "system_check_passed":
            raise StudyError("study system check must pass before calibration")
        result = create_session(
            _gaze_root(),
            participant["participant_id"],
            capture_run_id=body.get("capture_run_id"),
            capture_source="study-direct-frame",
            study_metadata={
                "study_session_id": participant["study_session_id"],
                "study_protocol_id": participant["protocol_id"],
                "study_protocol_version": participant["protocol_version"],
                "study_consent_version": store.protocol["consent_version"],
                "study_consent_digest_sha256": store.public_protocol[
                    "consent_digest_sha256"
                ],
                "study_mode": participant["mode"],
            },
        )
        try:
            store.start_calibration(
                participant["study_session_id"],
                _study_access_token(body),
                result["session_id"],
            )
        except StudyError:
            delete_dataset(_gaze_root(), result["session_id"])
            raise
        result.update(
            {
                "participant_id": participant["participant_id"],
                "study_session_id": participant["study_session_id"],
                "study_mode": True,
            }
        )
        return result, 200
    except StudyError as exc:
        return {"ok": False, "error": str(exc)}, 403


def _save_gaze_sample(body: dict) -> tuple[dict, int]:
    session_id = body.get("session_id", "")
    try:
        metadata = read_session_metadata(_gaze_root(), session_id)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {"ok": False, "error": "session not found"}, 404
    linked_study_id = str(metadata.get("study_session_id") or "")
    resolved_body = _authoritative_capture_payload(body, metadata)
    if linked_study_id:
        if _study_session_id(body) != linked_study_id:
            return {"ok": False, "error": "study session does not match dataset"}, 403
        try:
            _, participant = _participant_session(body)
        except StudyError as exc:
            return {"ok": False, "error": str(exc)}, 403
        if participant["state"] != "calibration_in_progress":
            return {"ok": False, "error": "calibration is not active"}, 409
        if participant.get("linked_data", {}).get("gaze_session_id") != session_id:
            return {
                "ok": False,
                "error": "gaze session is not linked to participant",
            }, 403
        try:
            resolved_body.update(
                authoritative_participant_calibration_labels(
                    body,
                    session_id=session_id,
                )
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}, 400
    elif _public_study_mode():
        return {"ok": False, "error": "non-study samples are disabled"}, 403
    return save_sample(
        _gaze_root(),
        resolved_body,
    )


def _participant_model_list() -> tuple[dict, int]:
    try:
        _, participant = _participant_session()
    except StudyError as exc:
        return {"ok": False, "error": str(exc)}, 403
    model_name = str(participant.get("linked_data", {}).get("model_name") or "")
    visible = [
        model
        for model in list_models(_gaze_root())
        if model.get("name") == model_name
    ]
    return {"ok": True, "models": visible}, 200


def _predict_response(body: dict) -> tuple[dict, int]:
    receipt_challenge = None
    store = None
    access_token = ""
    study_session_id = ""
    artifact_sha256_before = ""
    if _public_study_mode():
        body = dict(body)
        body["allow_cuda"] = False
        try:
            store, participant = _participant_session(body)
        except StudyError as exc:
            return {"ok": False, "error": str(exc)}, 403
        access_token = _study_access_token(body)
        study_session_id = _study_session_id(body)
        allowed_model = str(participant.get("linked_data", {}).get("model_name") or "")
        if not allowed_model or body.get("model_name") != allowed_model:
            return {
                "ok": False,
                "error": "model is not linked to this study session",
            }, 403
        if participant["state"] not in {
            "calibration_complete",
            "assessment_in_progress",
        }:
            return {
                "ok": False,
                "error": "study session is not ready for prediction",
            }, 409
        validation_phase = str(body.get("validation_phase") or "").strip()
        validation_target_id = str(body.get("validation_target_id") or "").strip()
        if bool(validation_phase) != bool(validation_target_id):
            return {
                "ok": False,
                "error": "validation phase and target ID must be supplied together",
            }, 400
        if validation_phase:
            try:
                artifact_sha256_before = model_artifact_sha256(
                    _gaze_root(), allowed_model
                )
                receipt_challenge = store.prepare_general_prediction_receipt(
                    study_session_id,
                    access_token,
                    phase=validation_phase,
                    target_id=validation_target_id,
                    model_name=allowed_model,
                    model_artifact_sha256=artifact_sha256_before,
                    viewport={
                        "width_px": body.get("viewport_width"),
                        "height_px": body.get("viewport_height"),
                    },
                )
            except FileNotFoundError as exc:
                return {
                    "ok": False,
                    "error": str(exc),
                    "failure_stage": "model_hard_error",
                }, 409
            except StudyAuthorizationError as exc:
                return {"ok": False, "error": str(exc)}, 403
            except StudyStateError as exc:
                return {"ok": False, "error": str(exc)}, 409
            except StudyValidationError as exc:
                return {"ok": False, "error": str(exc)}, 400
    response, status = predict(_gaze_root(), body)
    if receipt_challenge is None:
        return response, status
    receipt_eligible = response.get("ok") is True or response.get(
        "failure_stage"
    ) == "attributable_sensor_failure"
    if not receipt_eligible:
        return response, status
    try:
        artifact_sha256_after = model_artifact_sha256(
            _gaze_root(), str(receipt_challenge["model_name"])
        )
        if artifact_sha256_after != artifact_sha256_before:
            raise StudyStateError("model artifact changed during prediction")
        receipt = store.issue_general_prediction_receipt(
            study_session_id,
            access_token,
            challenge=receipt_challenge,
            model_artifact_sha256_after=artifact_sha256_after,
            capture_contract=body.get("capture_contract"),
            prediction_response=response,
            prediction_status=status,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "failure_stage": "model_hard_error",
        }, 409
    except StudyAuthorizationError as exc:
        return {"ok": False, "error": str(exc)}, 403
    except (StudyStateError, StudyValidationError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "failure_stage": "prediction_receipt_hard_error",
        }, 409
    return {**response, "prediction_receipt": receipt}, status


@gaze_bp.get("/health")
def health():
    if _public_study_mode() and not _researcher_authorized():
        return jsonify({"ok": True})
    ensure_runs_dir(_gaze_root())
    return jsonify({"ok": True, "backend": "chenghao-gaze", "mode": "http-polling"})


@gaze_bp.get("/models")
def models():
    if _public_study_mode() and not _researcher_authorized():
        response, status = _participant_model_list()
        return jsonify(response), status
    return jsonify({"ok": True, "models": list_models(_gaze_root())})


@gaze_bp.get("/datasets")
def datasets():
    if _public_study_mode() and not _researcher_authorized():
        return _admin_blocked_response()
    return jsonify({"ok": True, "datasets": list_datasets(_gaze_root())})


@gaze_bp.post("/session")
def session():
    body = request.get_json(force=True) or {}
    response, status = _create_gaze_session(body)
    return jsonify(response), status


@gaze_bp.post("/sample")
def sample():
    body = request.get_json(force=True) or {}
    response, status = _save_gaze_sample(body)
    return jsonify(response), status


@gaze_bp.post("/train")
def train():
    if _public_study_mode() and not _researcher_authorized():
        return _admin_blocked_response()
    body = request.get_json(force=True) or {}
    response, status = train_placeholder(_gaze_root(), body)
    return jsonify(response), status


@gaze_bp.post("/predict")
def predict_gaze():
    body = request.get_json(force=True)
    if not isinstance(body, dict):
        return jsonify(
            {"ok": False, "error": "request JSON body must be an object"}
        ), 400
    response, status = _predict_response(body)
    return jsonify(response), status


@gaze_bp.route("/datasets/<session_id>", methods=["DELETE", "PUT"])
def dataset_ops(session_id):
    if _public_study_mode() and not _researcher_authorized():
        return _admin_blocked_response()
    if request.method == "DELETE":
        return jsonify(delete_dataset(_gaze_root(), session_id))
    body = request.get_json(force=True) or {}
    new_name = body.get("new_name", "")
    if not new_name:
        return jsonify({"ok": False, "error": "new_name required"}), 400
    return jsonify(rename_dataset(_gaze_root(), session_id, new_name))


@gaze_bp.post("/datasets/<session_id>/reprocess")
def reprocess_dataset(session_id):
    """Re-run MediaPipe on raw frames for sessions that had face detection failures."""
    if _public_study_mode() and not _researcher_authorized():
        return _admin_blocked_response()
    try:
        session_dir = safe_session_dir(
            _gaze_root(), session_id, require_exists=True
        )
    except (FileNotFoundError, ValueError):
        return jsonify({"ok": False, "error": "session not found"}), 404
    manifest_path = session_dir / "manifest.jsonl"
    if not session_dir.exists() or not manifest_path.exists():
        return jsonify({"ok": False, "error": "session not found"}), 404

    try:
        preprocessor = get_preprocessor()
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        fixed = 0
        still_failed = 0
        updated = []
        for rec in records:
            if rec.get("normalized_face_path"):  # already processed
                updated.append(rec)
                continue
            raw_p = session_dir / rec.get("raw_path", "")
            img = cv2.imread(str(raw_p), cv2.IMREAD_COLOR)
            if img is None:
                still_failed += 1
                updated.append(rec)
                continue
            try:
                processed = preprocessor.process(img)
                stem = Path(rec["raw_path"]).stem
                crop_path = session_dir / "crop" / f"{stem}.jpg"
                norm_path = session_dir / "normalized_face" / f"{stem}.jpg"
                cv2.imwrite(str(crop_path), processed.crop_bgr)
                cv2.imwrite(str(norm_path), processed.image_bgr)
                rec["crop_path"] = crop_path.relative_to(session_dir).as_posix()
                rec["normalized_face_path"] = norm_path.relative_to(
                    session_dir
                ).as_posix()
                rec["head_pose_pitch_yaw"] = processed.head_pose_pitch_yaw.tolist()
                rec["face_bbox"] = processed.face_bbox
                rec.pop("warning", None)
                rec["face_detected"] = True
                rec["ok"] = True
                fixed += 1
            except Exception as exc:
                rec["warning"] = str(exc)
                still_failed += 1
            updated.append(rec)
        manifest_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in updated)
            + "\n",
            encoding="utf-8",
        )
        return jsonify(
            {
                "ok": True,
                "fixed": fixed,
                "still_failed": still_failed,
                "total": len(updated),
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500


@gaze_bp.get("/datasets/<session_id>/motion-audit")
def motion_audit(session_id):
    if _public_study_mode() and not _researcher_authorized():
        return _admin_blocked_response()
    sessions_dir = ensure_sessions_dir(_gaze_root())
    session_dir = (sessions_dir / session_id).resolve()
    if session_dir.parent != sessions_dir.resolve() or not session_dir.is_dir():
        return jsonify({"ok": False, "error": "session not found"}), 404
    samples, diagnostics = load_motion_samples(
        sessions_dir,
        session_ids=(session_id,),
    )
    return jsonify({"ok": True, **audit_payload(samples, diagnostics)})


@gaze_bp.route("/models/<model_name>", methods=["DELETE", "PUT"])
def model_ops(model_name):
    if _public_study_mode() and not _researcher_authorized():
        return _admin_blocked_response()
    if request.method == "DELETE":
        return jsonify(delete_model(_gaze_root(), model_name))
    body = request.get_json(force=True) or {}
    new_name = body.get("new_name", "")
    if not new_name:
        return jsonify({"ok": False, "error": "new_name required"}), 400
    return jsonify(rename_model(_gaze_root(), model_name, new_name))


@gaze_api_bp.get("/health")
def api_health():
    ensure_runs_dir(_gaze_root())
    return jsonify({"ok": True, "backend": "chenghao-gaze", "mode": "http-polling"})


@gaze_api_bp.get("/list_models")
def api_list_models():
    if _public_study_mode() and not _researcher_authorized():
        response, status = _participant_model_list()
        return jsonify(response), status
    models_data = []
    for model in list_models(_gaze_root()):
        models_data.append(
            {
                **model,
                "num_stages": model.get("num_stages", 1),
                "noise_level": model.get("noise_level", 0.0),
            }
        )
    return jsonify({"ok": True, "models": models_data})


@gaze_api_bp.get("/list_datasets")
def api_list_datasets():
    if _public_study_mode() and not _researcher_authorized():
        return _admin_blocked_response()
    return jsonify({"ok": True, "datasets": list_datasets(_gaze_root())})


@gaze_api_bp.post("/session")
def api_session():
    body = request.get_json(force=True) or {}
    response, status = _create_gaze_session(body)
    return jsonify(response), status


@gaze_api_bp.post("/sample")
def api_sample():
    body = request.get_json(force=True) or {}
    response, status = _save_gaze_sample(body)
    return jsonify(response), status


@gaze_api_bp.post("/train")
def api_train():
    if _public_study_mode() and not _researcher_authorized():
        return _admin_blocked_response()
    try:
        body = request.get_json(force=True) or {}
        response, status = train_placeholder(_gaze_root(), body)
        return jsonify(response), status
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Train failed: {exc}"}), 500


@gaze_api_bp.post("/predict")
def api_predict():
    try:
        body = request.get_json(force=True)
        if not isinstance(body, dict):
            return jsonify(
                {"ok": False, "error": "request JSON body must be an object"}
            ), 400
        response, status = _predict_response(body)
        return jsonify(response), status
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Predict failed: {exc}"}), 500


@gaze_bp.post("/save_pairs")
def save_pairs():
    if _public_study_mode() and not _researcher_authorized():
        return _admin_blocked_response()
    body = request.get_json(force=True) or {}
    session_id = body.get("session_id", "")
    pairs = body.get("pairs", [])
    if not session_id or not pairs:
        return jsonify({"ok": False, "error": "session_id and pairs are required"}), 400

    gaze_root = _gaze_root()
    gt_dir = gaze_root / "data" / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    file_path = gt_dir / f"{session_id}.json"
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "timestamp_created": time.time(),
                    "viewport_width": body.get("viewport_width", 1920),
                    "viewport_height": body.get("viewport_height", 1080),
                    "pairs": pairs,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return jsonify(
            {"ok": True, "saved_to": str(file_path.relative_to(gaze_root))}
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
