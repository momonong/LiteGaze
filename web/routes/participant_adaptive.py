"""Integrity-bound adaptive assessment handlers for public study sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import secrets
import time
from pathlib import Path

from flask import current_app, jsonify, request

from core.cognitive_inspector.adaptive import (
    CALIBRATION_STATUS,
    ITEM_BANK_VERSION,
    MAX_ROUNDS,
    MIN_ROUNDS,
    PROTOCOL_VERSION,
    adaptive_analysis,
    generate_adaptive_report,
    initial_passage,
    public_passage,
    score_passage,
    select_next_passage,
    should_stop,
)
from core.participant_study import (
    ParticipantStudyStore,
    StudyAuthorizationError,
    StudyError,
    StudyStateError,
    StudyValidationError,
)

ROOT = Path(__file__).resolve().parents[2]
_FALLBACK_SIGNING_SECRET = secrets.token_bytes(32)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _public_study_mode() -> bool:
    value = current_app.config.get(
        "LEXIGAZE_PUBLIC_STUDY_MODE",
        os.environ.get("LEXIGAZE_PUBLIC_STUDY_MODE", ""),
    )
    return _truthy(value)


def _study_access_token(body: dict) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return str(body.get("study_access_token") or "").strip()


def _study_context(body: dict) -> tuple[ParticipantStudyStore, dict, str] | None:
    session_id = str(body.get("study_session_id") or "").strip()
    if not session_id:
        if _public_study_mode():
            raise StudyError("study session credential is required")
        return None
    access_token = _study_access_token(body)
    root = Path(current_app.config.get("LEXIGAZE_STUDY_ROOT") or ROOT)
    store = ParticipantStudyStore(root, settings=current_app.config)
    store.enforce_expired_calibration_retention()
    participant = store.get_session(session_id, access_token)
    if participant["mode"] != "pilot":
        raise StudyError("dry runs cannot start the participant assessment")
    return store, participant, access_token


def _study_error(exc: StudyError):
    if isinstance(exc, StudyAuthorizationError):
        status = 403
    elif isinstance(exc, StudyStateError):
        status = 409
    elif isinstance(exc, StudyValidationError):
        status = 400
    else:
        status = 403
    return jsonify({"ok": False, "error": str(exc)}), status


def _signing_secret() -> bytes:
    configured = current_app.config.get(
        "LEXIGAZE_ADAPTIVE_SIGNING_KEY",
        os.environ.get("LEXIGAZE_ADAPTIVE_SIGNING_KEY", ""),
    )
    if configured:
        return str(configured).encode("utf-8")
    return _FALLBACK_SIGNING_SECRET


def _encode_token(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(_signing_secret(), raw, hashlib.sha256).digest()
    encoded_payload = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{encoded_payload}.{encoded_signature}"


def _decode_token(token: str) -> dict:
    if not isinstance(token, str) or not token or len(token) > 8192:
        raise ValueError("invalid adaptive result token")
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        raw = base64.urlsafe_b64decode(
            encoded_payload + "=" * (-len(encoded_payload) % 4)
        )
        signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("invalid adaptive result token") from None
    canonical_payload = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    canonical_signature = (
        base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    )
    if (
        encoded_payload != canonical_payload
        or encoded_signature != canonical_signature
    ):
        raise ValueError("adaptive result token is not canonically encoded")
    expected = hmac.new(_signing_secret(), raw, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("adaptive result token verification failed")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("invalid adaptive result payload") from None
    if not isinstance(payload, dict):
        raise TypeError("invalid adaptive result payload")
    return payload


def _validated_metrics(raw_metrics: object) -> dict:
    if raw_metrics is None:
        raw_metrics = {}
    if not isinstance(raw_metrics, dict):
        raise TypeError("metrics must be an object")
    validated: dict[str, object] = {}
    for key, lower, upper in (
        ("wpm", 1.0, 1000.0),
        ("regression_rate", 0.0, 1.0),
        ("avg_fixation_duration_ms", 0.0, 5000.0),
    ):
        if key not in raw_metrics or raw_metrics[key] is None:
            continue
        value = float(raw_metrics[key])
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"invalid adaptive metric: {key}")
        validated[key] = value
    quality_status = str(raw_metrics.get("data_quality_status") or "unknown")
    if quality_status not in {
        "good",
        "limited",
        "insufficient",
        "unknown",
        "behavioral_only",
    }:
        raise ValueError("invalid adaptive data quality status")
    validated["data_quality_status"] = quality_status
    return validated


def _verified_history(
    raw_history: object,
    *,
    expected_assessment_id: str,
    expected_study_session_id: str,
) -> list[dict]:
    if not isinstance(raw_history, list):
        raise TypeError("history must be an array")
    if len(raw_history) > MAX_ROUNDS:
        raise ValueError("history exceeds the assessment round limit")
    verified: list[dict] = []
    seen_passages: set[str] = set()
    for position, record in enumerate(raw_history, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"history item {position} must be an object")
        payload = _decode_token(str(record.get("result_token", "")))
        expected = {
            "token_type": "adaptive_result_v2",
            "assessment_id": expected_assessment_id,
            "study_session_id": expected_study_session_id,
            "round": position,
            "protocol_version": PROTOCOL_VERSION,
            "item_bank_version": ITEM_BANK_VERSION,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"history item {position} context does not match")
        if payload.get("passage_id") != record.get("passage_id"):
            raise ValueError(f"history item {position} passage does not match its token")
        passage_id = str(payload["passage_id"])
        if passage_id in seen_passages:
            raise ValueError(f"history item {position} repeats a passage")
        seen_passages.add(passage_id)
        merged = dict(payload)
        merged.update(_validated_metrics(payload.get("metrics", {})))
        merged["result_token"] = record.get("result_token")
        merged["quiz_score"] = payload.get("correct")
        merged["quiz_total"] = payload.get("total")
        verified.append(merged)
    return verified


def _round_response(
    passage: dict,
    *,
    assessment_id: str,
    round_number: int,
    study_session_id: str = "",
) -> dict:
    assignment = {
        "token_type": "adaptive_assignment_v2",
        "assessment_id": assessment_id,
        "study_session_id": study_session_id,
        "round": round_number,
        "passage_id": passage["passage_id"],
        "protocol_version": PROTOCOL_VERSION,
        "item_bank_version": ITEM_BANK_VERSION,
    }
    return {
        "ok": True,
        "assessment_id": assessment_id,
        "study_session_id": study_session_id or None,
        "round": round_number,
        "round_token": _encode_token(assignment),
        "min_rounds": MIN_ROUNDS,
        "max_rounds": MAX_ROUNDS,
        "protocol_version": PROTOCOL_VERSION,
        "item_bank_version": ITEM_BANK_VERSION,
        "calibration_status": CALIBRATION_STATUS,
        **public_passage(passage),
    }


def start_response():
    body = request.get_json(force=True) or {}
    resume_history: list[dict] = []
    try:
        study = _study_context(body)
    except StudyError as exc:
        return _study_error(exc)
    study_session_id = ""
    if study:
        store, participant, access_token = study
        if participant["state"] == "assessment_in_progress":
            assessment_id = str(participant["linked_data"]["assessment_id"])
        elif participant["state"] == "calibration_complete":
            assessment_id = secrets.token_hex(12)
        else:
            return jsonify(
                {"ok": False, "error": "completed calibration is required"}
            ), 409
        study_session_id = participant["study_session_id"]
        try:
            participant = store.start_assessment(
                study_session_id,
                access_token,
                assessment_id,
            )
            stored_rounds = participant.get("quality", {}).get(
                "assessment_rounds", []
            )
            resume_history = [
                {
                    "round": item.get("round"),
                    "passage_id": item.get("passage_id"),
                    "result_token": item.get("result_token"),
                }
                for item in stored_rounds
            ]
            verified_resume = _verified_history(
                resume_history,
                expected_assessment_id=assessment_id,
                expected_study_session_id=study_session_id,
            )
        except StudyError as exc:
            return _study_error(exc)
        except (TypeError, ValueError) as exc:
            return jsonify({"ok": False, "error": f"resume failed: {exc}"}), 409
    else:
        assessment_id = str(body.get("assessment_id") or secrets.token_hex(12))[:128]
        verified_resume = []
    if verified_resume and should_stop(verified_resume):
        return jsonify(
            {
                "ok": True,
                "assessment_id": assessment_id,
                "study_session_id": study_session_id or None,
                "is_finished": True,
                "resume_history": resume_history,
                "rounds_completed": len(verified_resume),
            }
        )
    passage = (
        select_next_passage(verified_resume, assessment_id)
        if verified_resume
        else initial_passage(assessment_id)
    )
    if passage is None:
        return jsonify(
            {
                "ok": True,
                "assessment_id": assessment_id,
                "study_session_id": study_session_id or None,
                "is_finished": True,
                "resume_history": resume_history,
                "rounds_completed": len(verified_resume),
            }
        )
    response = _round_response(
        passage,
        assessment_id=assessment_id,
        round_number=len(verified_resume) + 1,
        study_session_id=study_session_id,
    )
    response["resume_history"] = resume_history
    return jsonify(response)


def score_response():
    body = request.get_json(force=True) or {}
    try:
        study = _study_context(body)
        assessment_id = str(body.get("assessment_id") or "")[:128]
        round_number = int(body.get("round") or 0)
        assignment = _decode_token(str(body.get("round_token") or ""))
        expected_study_id = study[1]["study_session_id"] if study else ""
        expected_assignment = {
            "token_type": "adaptive_assignment_v2",
            "assessment_id": assessment_id,
            "study_session_id": expected_study_id,
            "round": round_number,
            "passage_id": str(body.get("passage_id") or ""),
            "protocol_version": PROTOCOL_VERSION,
            "item_bank_version": ITEM_BANK_VERSION,
        }
        if any(
            assignment.get(key) != value
            for key, value in expected_assignment.items()
        ):
            raise ValueError("adaptive round assignment does not match the request")
        if not 1 <= round_number <= MAX_ROUNDS:
            raise ValueError("invalid adaptive assessment round")
        metrics = _validated_metrics(body.get("metrics", {}))
        scored = score_passage(
            str(body.get("passage_id", "")), body.get("responses", {})
        )
    except StudyError as exc:
        return _study_error(exc)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    signed_payload = {
        "token_type": "adaptive_result_v2",
        "assessment_id": assessment_id,
        "study_session_id": expected_study_id,
        "round": round_number,
        "protocol_version": PROTOCOL_VERSION,
        "item_bank_version": ITEM_BANK_VERSION,
        "metrics": metrics,
        "passage_id": scored["passage_id"],
        "correct": scored["correct"],
        "total": scored["total"],
        "item_results": [
            {
                "question_id": result["question_id"],
                "correct": result["correct"],
            }
            for result in scored["item_results"]
        ],
    }
    result_token = _encode_token(signed_payload)
    if study:
        store, participant, access_token = study
        try:
            store.record_assessment_round(
                participant["study_session_id"],
                access_token,
                assessment_id=assessment_id,
                round_number=round_number,
                passage_id=scored["passage_id"],
                correct=scored["correct"],
                total=scored["total"],
                result_token=result_token,
                metrics=metrics,
            )
        except StudyError as exc:
            return _study_error(exc)
        public_result = {
            "passage_id": scored["passage_id"],
            "correct": scored["correct"],
            "total": scored["total"],
        }
    else:
        public_result = scored
    return jsonify(
        {
            "ok": True,
            "round_result": public_result,
            "result_token": result_token,
        }
    )


def next_response():
    body = request.get_json(force=True) or {}
    try:
        study = _study_context(body)
        assessment_id = str(body.get("assessment_id") or "")[:128]
        if not assessment_id:
            raise ValueError("assessment_id is required")
        study_session_id = study[1]["study_session_id"] if study else ""
        if study and study[1]["linked_data"].get("assessment_id") != assessment_id:
            raise StudyError("assessment does not match the study session")
        history = _verified_history(
            body.get("history", []),
            expected_assessment_id=assessment_id,
            expected_study_session_id=study_session_id,
        )
    except StudyError as exc:
        return _study_error(exc)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if should_stop(history):
        return jsonify(
            {
                "ok": True,
                "is_finished": True,
                "rounds_completed": len(history),
                "analysis": adaptive_analysis(history),
            }
        )
    passage = select_next_passage(history, assessment_id)
    if passage is None:
        return jsonify(
            {"ok": True, "is_finished": True, "rounds_completed": len(history)}
        )
    return jsonify(
        _round_response(
            passage,
            assessment_id=assessment_id,
            round_number=len(history) + 1,
            study_session_id=study_session_id,
        )
    )


def report_response(reports_dir: Path):
    body = request.get_json(force=True) or {}
    try:
        study = _study_context(body)
        assessment_id = str(body.get("assessment_id") or "")[:128]
        if not assessment_id:
            raise ValueError("assessment_id is required")
        study_session_id = study[1]["study_session_id"] if study else ""
        if study and study[1]["linked_data"].get("assessment_id") != assessment_id:
            raise StudyError("assessment does not match the study session")
        history = _verified_history(
            body.get("history", []),
            expected_assessment_id=assessment_id,
            expected_study_session_id=study_session_id,
        )
    except StudyError as exc:
        return _study_error(exc)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if not should_stop(history):
        return jsonify(
            {"ok": False, "error": "assessment has not reached its stop rule"}
        ), 409
    if study:
        participant_id = study[1]["participant_id"]
        persist = False
    else:
        participant_raw = body.get("participant_id", "anonymous")
        participant_id = str(participant_raw).strip() or "anonymous"
        persist = bool(body.get("persist", False))
    analysis = adaptive_analysis(history)
    report_md = generate_adaptive_report(analysis, participant_id, history)
    report_path = None
    if persist:
        timestamp = int(time.time())
        safe_id = "".join(
            character
            for character in participant_id
            if character.isalnum() or character in ("-", "_")
        ).strip() or "anonymous"
        filename = f"adaptive_v2_{safe_id}_{timestamp}.md"
        out_path = reports_dir / filename
        out_path.write_text(report_md, encoding="utf-8")
        report_path = f"docs/cognitive_reports/{filename}"
    observations = analysis["observations"]
    model = analysis["experimental_model"]
    comprehension_rate = (
        round((observations["correct"] / observations["total"]) * 100.0, 1)
        if observations["total"]
        else None
    )
    summary = {
        "comprehension_rate": comprehension_rate,
        "experimental_theta": model["theta"],
        "posterior_sd": model["posterior_sd"],
        "claim_status": analysis["claims"]["english_proficiency"]["status"],
        "optimal_font_size": None,
        "optimal_line_width": None,
        "optimal_line_height": None,
        "typography_status": "not_estimated",
    }
    if study:
        try:
            study[0].complete_assessment(
                study_session_id,
                study[2],
                {
                    "assessment_id": assessment_id,
                    "passage_count": analysis["data_quality"]["passage_count"],
                    "item_count": analysis["data_quality"]["item_count"],
                    "construct_count": analysis["data_quality"]["construct_count"],
                    "data_quality_status": analysis["data_quality"]["status"],
                    "claim_status": summary["claim_status"],
                },
            )
        except StudyError as exc:
            return _study_error(exc)
    return jsonify(
        {
            "ok": True,
            "report_md": report_md,
            "analysis": analysis,
            "report_path": report_path,
            "summary": summary,
            "study_completed": bool(study),
            "debrief_url": "/study" if study else None,
        }
    )
