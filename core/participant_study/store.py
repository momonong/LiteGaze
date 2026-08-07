"""Privacy-preserving participant-study state and consent receipts."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .protocol import activation_status, load_protocol, public_protocol


class StudyError(RuntimeError):
    """Base class for participant-study errors."""


class StudyValidationError(StudyError):
    """Raised for malformed enrollment or transition input."""


class StudyAuthorizationError(StudyError):
    """Raised for invalid participant credentials."""


class StudyStateError(StudyError):
    """Raised when a lifecycle transition is not allowed."""


class StudyNotReadyError(StudyError):
    """Raised when real collection is attempted before activation gates pass."""


ACTIVE_STATES = {
    "consented",
    "system_check_passed",
    "calibration_in_progress",
    "calibration_complete",
    "assessment_in_progress",
    "completed",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class ParticipantStudyStore:
    """Store consented study sessions without direct participant identifiers."""

    _lock = threading.RLock()

    def __init__(
        self,
        root: Path,
        *,
        settings: Mapping[str, object] | None = None,
        protocol: Mapping[str, object] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.settings = settings
        self.protocol = dict(protocol or load_protocol())
        self.public_protocol = public_protocol(settings, protocol=self.protocol)
        self.activation = activation_status(settings, protocol=self.protocol)

    def _secret_hash(self, value: str) -> str:
        material = f"{self.protocol['protocol_id']}:{value}".encode()
        return hashlib.sha256(material).hexdigest()

    def _study_root(self, mode: str) -> Path:
        bucket = "dry_runs" if mode == "dry_run" else "pilot"
        return (
            self.root
            / "data"
            / "participant_studies"
            / self.protocol["protocol_id"]
            / bucket
        )

    def _session_path(self, session_id: str, mode: str) -> Path:
        if not session_id.startswith("ST-") or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            for character in session_id
        ):
            raise StudyValidationError("invalid study session ID")
        base = self._study_root(mode).resolve()
        candidate = (base / session_id / "session.json").resolve()
        if candidate.parent.parent != base:
            raise StudyValidationError("invalid study session path")
        return candidate

    def _find_session_path(self, session_id: str) -> Path:
        for mode in ("dry_run", "pilot"):
            path = self._session_path(session_id, mode)
            if path.exists():
                return path
        raise StudyValidationError("study session not found")

    def _read(self, session_id: str) -> tuple[Path, dict[str, Any]]:
        path = self._find_session_path(session_id)
        return path, json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path: Path, session: Mapping[str, object]) -> None:
        _atomic_json(path, session)

    def _authorize(self, session: Mapping[str, object], access_token: str) -> None:
        supplied = self._secret_hash(str(access_token or ""))
        expected = str(session.get("access_token_sha256") or "")
        if not expected or not secrets.compare_digest(supplied, expected):
            raise StudyAuthorizationError("invalid study session credential")

    def _event(self, session: dict[str, Any], event: str, **details: object) -> None:
        session.setdefault("events", []).append(
            {"at_utc": _utc_now(), "event": event, **details}
        )
        session["updated_at_utc"] = _utc_now()

    def _validate_enrollment(self, payload: Mapping[str, object]) -> dict[str, Any]:
        if payload.get("adult_confirmed") is not True:
            raise StudyValidationError("adult confirmation is required")
        if payload.get("private_space_confirmed") is not True:
            raise StudyValidationError("private-space confirmation is required")

        supplied_statements = payload.get("consent_statements")
        if not isinstance(supplied_statements, dict):
            raise StudyValidationError("consent statements must be an object")
        required_ids = {
            item["id"] for item in self.protocol["required_consent_statements"]
        }
        missing_statements = sorted(
            statement_id
            for statement_id in required_ids
            if supplied_statements.get(statement_id) is not True
        )
        if missing_statements:
            raise StudyValidationError(
                f"required consent statements missing: {missing_statements}"
            )

        supplied_checks = payload.get("comprehension_answers")
        if not isinstance(supplied_checks, dict):
            raise StudyValidationError("comprehension answers must be an object")
        incorrect = sorted(
            item["id"]
            for item in self.protocol["comprehension_checks"]
            if supplied_checks.get(item["id"]) != item["correct"]
        )
        if incorrect:
            raise StudyValidationError(
                f"consent comprehension check failed: {incorrect}"
            )

        optional_payload = payload.get("optional_scopes", {})
        if not isinstance(optional_payload, dict):
            raise StudyValidationError("optional scopes must be an object")
        optional_ids = {item["id"] for item in self.protocol["optional_scopes"]}
        optional_scopes = {
            scope_id: optional_payload.get(scope_id) is True
            for scope_id in sorted(optional_ids)
        }
        return {
            "required_statement_ids": sorted(required_ids),
            "comprehension_answers": {
                item["id"]: supplied_checks[item["id"]]
                for item in self.protocol["comprehension_checks"]
            },
            "optional_scopes": optional_scopes,
        }

    def _consume_invite(self, invite_code: str, session_id: str) -> None:
        invite_path = self._study_root("pilot") / "invites.json"
        if not invite_path.exists():
            raise StudyNotReadyError("no pilot invitation registry is available")
        registry = json.loads(invite_path.read_text(encoding="utf-8"))
        supplied = self._secret_hash(invite_code)
        matched = None
        for invite in registry.get("invites", []):
            if secrets.compare_digest(str(invite.get("code_sha256", "")), supplied):
                matched = invite
                break
        if matched is None or matched.get("used_at_utc"):
            raise StudyAuthorizationError("invalid or already-used invitation code")
        matched["used_at_utc"] = _utc_now()
        matched["study_session_id"] = session_id
        _atomic_json(invite_path, registry)

    def create_invites(self, count: int) -> list[str]:
        if not self.activation["pilot_ready"]:
            raise StudyNotReadyError(
                "pilot invitation creation is locked until all activation gates pass"
            )
        if not 1 <= count <= 100:
            raise StudyValidationError("invite count must be between 1 and 100")
        path = self._study_root("pilot") / "invites.json"
        with self._lock:
            if path.exists():
                registry = json.loads(path.read_text(encoding="utf-8"))
            else:
                registry = {
                    "schema_version": 1,
                    "protocol_id": self.protocol["protocol_id"],
                    "created_at_utc": _utc_now(),
                    "invites": [],
                }
            codes = [f"LG-{secrets.token_hex(5).upper()}" for _ in range(count)]
            registry["invites"].extend(
                {
                    "code_sha256": self._secret_hash(code),
                    "created_at_utc": _utc_now(),
                    "used_at_utc": None,
                    "study_session_id": None,
                }
                for code in codes
            )
            _atomic_json(path, registry)
        return codes

    def enforce_expired_calibration_retention(
        self,
        *,
        now: datetime | None = None,
    ) -> list[str]:
        """Purge interrupted pilot calibrations after the approved raw-frame TTL."""

        retention_hours = self.activation.get("raw_frame_retention_hours")
        if not retention_hours:
            return []
        current = now or datetime.now(UTC)
        cutoff = current - timedelta(hours=int(retention_hours))
        purged: list[str] = []
        pilot_root = self._study_root("pilot")
        with self._lock:
            for path in pilot_root.glob("ST-*/session.json"):
                try:
                    session = json.loads(path.read_text(encoding="utf-8"))
                    if session.get("state") != "calibration_in_progress":
                        continue
                    started_text = next(
                        (
                            str(event.get("at_utc"))
                            for event in reversed(session.get("events", []))
                            if event.get("event") == "calibration_started"
                        ),
                        "",
                    )
                    started = datetime.fromisoformat(started_text)
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=UTC)
                    if started > cutoff:
                        continue
                    gaze_session_id = str(
                        session.get("linked_data", {}).get("gaze_session_id") or ""
                    )
                    sessions_root = (self.root / "data" / "sessions").resolve()
                    gaze_path = (sessions_root / gaze_session_id).resolve()
                    if (
                        gaze_session_id
                        and gaze_path.parent == sessions_root
                        and gaze_path.is_dir()
                    ):
                        shutil.rmtree(gaze_path)
                    session["linked_data"].pop("gaze_session_id", None)
                    session["linked_data"].pop("model_name", None)
                    session["state"] = "system_check_passed"
                    session["quality"]["calibration_expiry"] = {
                        "raw_data_purged": True,
                        "retention_hours": int(retention_hours),
                    }
                    self._event(session, "expired_calibration_data_purged")
                    self._write(path, session)
                    purged.append(str(session["study_session_id"]))
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return purged

    def enroll(self, payload: Mapping[str, object]) -> dict[str, Any]:
        mode = str(payload.get("mode") or "dry_run").strip().lower()
        if mode not in {"dry_run", "pilot"}:
            raise StudyValidationError("mode must be dry_run or pilot")
        if mode == "pilot" and not self.activation["pilot_ready"]:
            raise StudyNotReadyError(
                "real participant collection is locked; run the readiness audit"
            )
        enrollment = self._validate_enrollment(payload)
        session_id = "ST-" + secrets.token_hex(10).upper()
        participant_id = "P-" + secrets.token_hex(6).upper()
        access_token = secrets.token_urlsafe(32)
        withdrawal_code = "WD-" + secrets.token_hex(8).upper()
        timestamp = _utc_now()
        path = self._session_path(session_id, mode)
        session = {
            "schema_version": 1,
            "protocol_id": self.protocol["protocol_id"],
            "protocol_version": self.protocol["protocol_version"],
            "protocol_digest_sha256": self.public_protocol["protocol_digest_sha256"],
            "consent_version": self.protocol["consent_version"],
            "consent_digest_sha256": self.public_protocol["consent_digest_sha256"],
            "study_session_id": session_id,
            "participant_id": participant_id,
            "mode": mode,
            "state": "consented",
            "created_at_utc": timestamp,
            "updated_at_utc": timestamp,
            "access_token_sha256": self._secret_hash(access_token),
            "withdrawal_code_sha256": self._secret_hash(withdrawal_code),
            "eligibility": {
                "adult_confirmed": True,
                "private_space_confirmed": True,
                "exact_age_collected": False,
            },
            "consent": {
                "accepted_at_utc": timestamp,
                "required_statement_ids": enrollment["required_statement_ids"],
                "comprehension_answers": enrollment["comprehension_answers"],
                "comprehension_passed": True,
                "optional_scopes": enrollment["optional_scopes"],
            },
            "linked_data": {},
            "quality": {},
            "events": [{"at_utc": timestamp, "event": "consent_recorded"}],
        }
        with self._lock:
            if mode == "pilot":
                invite_code = str(payload.get("invite_code") or "").strip()
                if not invite_code:
                    raise StudyAuthorizationError("pilot invitation code is required")
                self._consume_invite(invite_code, session_id)
            self._write(path, session)
        return {
            "ok": True,
            "study_session_id": session_id,
            "participant_id": participant_id,
            "mode": mode,
            "state": "consented",
            "access_token": access_token,
            "withdrawal_code": withdrawal_code,
            "consent_receipt": self.consent_receipt(session),
        }

    def consent_receipt(self, session: Mapping[str, object]) -> dict[str, Any]:
        consent = dict(session.get("consent") or {})
        return {
            "schema_version": 1,
            "protocol_id": session["protocol_id"],
            "protocol_version": session["protocol_version"],
            "protocol_digest_sha256": session["protocol_digest_sha256"],
            "consent_version": session["consent_version"],
            "consent_digest_sha256": session["consent_digest_sha256"],
            "study_session_id": session["study_session_id"],
            "participant_id": session["participant_id"],
            "accepted_at_utc": consent.get("accepted_at_utc"),
            "required_statement_ids": consent.get("required_statement_ids", []),
            "optional_scopes": consent.get("optional_scopes", {}),
            "comprehension_passed": consent.get("comprehension_passed") is True,
            "mode": session["mode"],
        }

    def _public_session(self, session: Mapping[str, object]) -> dict[str, Any]:
        return {
            "study_session_id": session["study_session_id"],
            "participant_id": session["participant_id"],
            "protocol_id": session["protocol_id"],
            "protocol_version": session["protocol_version"],
            "mode": session["mode"],
            "state": session["state"],
            "created_at_utc": session["created_at_utc"],
            "updated_at_utc": session["updated_at_utc"],
            "optional_scopes": dict(session.get("consent", {})).get(
                "optional_scopes", {}
            ),
            "quality": session.get("quality", {}),
            "linked_data": session.get("linked_data", {}),
        }

    def get_session(self, session_id: str, access_token: str) -> dict[str, Any]:
        with self._lock:
            _, session = self._read(session_id)
            self._authorize(session, access_token)
            return self._public_session(session)

    def get_receipt(self, session_id: str, access_token: str) -> dict[str, Any]:
        with self._lock:
            _, session = self._read(session_id)
            self._authorize(session, access_token)
            return self.consent_receipt(session)

    def record_system_check(
        self,
        session_id: str,
        access_token: str,
        checks: Mapping[str, object],
    ) -> dict[str, Any]:
        required = ("camera_api", "secure_context", "screen_size", "network")
        if any(checks.get(name) is not True for name in required):
            raise StudyValidationError("all system checks must pass")
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session["state"] != "consented":
                raise StudyStateError("system check requires consented state")
            session["state"] = "system_check_passed"
            session["quality"]["system_check"] = {name: True for name in required}
            self._event(session, "system_check_passed")
            self._write(path, session)
            return self._public_session(session)

    def start_calibration(
        self,
        session_id: str,
        access_token: str,
        gaze_session_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session["state"] != "system_check_passed":
                raise StudyStateError("calibration requires a passed system check")
            session["state"] = "calibration_in_progress"
            session["linked_data"]["gaze_session_id"] = str(gaze_session_id)
            self._event(session, "calibration_started")
            self._write(path, session)
            return self._public_session(session)

    def complete_calibration(
        self,
        session_id: str,
        access_token: str,
        quality: Mapping[str, object],
        *,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        passed = quality.get("passed") is True
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session["state"] != "calibration_in_progress":
                raise StudyStateError("calibration is not in progress")
            session["quality"]["calibration"] = dict(quality)
            if passed:
                session["state"] = "calibration_complete"
                if model_name:
                    session["linked_data"]["model_name"] = str(model_name)
                self._event(session, "calibration_completed")
            else:
                session["state"] = "system_check_passed"
                session["linked_data"].pop("gaze_session_id", None)
                session["linked_data"].pop("model_name", None)
                self._event(session, "calibration_failed_quality_gate")
            self._write(path, session)
            return self._public_session(session)

    def start_assessment(
        self,
        session_id: str,
        access_token: str,
        assessment_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session["state"] not in {
                "calibration_complete",
                "assessment_in_progress",
            }:
                raise StudyStateError("assessment requires completed calibration")
            existing = session["linked_data"].get("assessment_id")
            if existing and existing != assessment_id:
                raise StudyStateError("a different assessment is already linked")
            session["state"] = "assessment_in_progress"
            session["linked_data"]["assessment_id"] = str(assessment_id)
            self._event(session, "assessment_started")
            self._write(path, session)
            return self._public_session(session)

    def record_assessment_round(
        self,
        session_id: str,
        access_token: str,
        *,
        assessment_id: str,
        round_number: int,
        passage_id: str,
        correct: int,
        total: int,
        result_token: str,
        metrics: Mapping[str, object],
    ) -> dict[str, Any]:
        """Record one immutable scored round and reject retries or reordering."""

        if not passage_id or len(passage_id) > 128:
            raise StudyValidationError("invalid assessment passage ID")
        if not result_token or len(result_token) > 8192:
            raise StudyValidationError("invalid assessment result token")
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session["state"] != "assessment_in_progress":
                raise StudyStateError("assessment is not in progress")
            if assessment_id != session["linked_data"].get("assessment_id"):
                raise StudyStateError("assessment ID does not match the study session")
            rounds = session["quality"].setdefault("assessment_rounds", [])
            expected_round = len(rounds) + 1
            if round_number != expected_round:
                raise StudyStateError(
                    f"assessment round {expected_round} must be scored next"
                )
            if any(item.get("passage_id") == passage_id for item in rounds):
                raise StudyStateError("assessment passage was already scored")
            rounds.append(
                {
                    "round": round_number,
                    "passage_id": passage_id,
                    "correct": int(correct),
                    "total": int(total),
                    "result_token": result_token,
                    "wpm": metrics.get("wpm"),
                    "regression_rate": metrics.get("regression_rate"),
                    "avg_fixation_duration_ms": metrics.get("avg_fixation_duration_ms"),
                    "data_quality_status": metrics.get("data_quality_status"),
                }
            )
            self._event(
                session,
                "assessment_round_scored",
                round=round_number,
                passage_id=passage_id,
            )
            self._write(path, session)
            return self._public_session(session)

    def complete_assessment(
        self,
        session_id: str,
        access_token: str,
        summary: Mapping[str, object],
    ) -> dict[str, Any]:
        allowed_keys = {
            "assessment_id",
            "passage_count",
            "item_count",
            "construct_count",
            "data_quality_status",
            "claim_status",
        }
        safe_summary = {key: summary.get(key) for key in sorted(allowed_keys)}
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session["state"] != "assessment_in_progress":
                raise StudyStateError("assessment is not in progress")
            if safe_summary.get("assessment_id") != session["linked_data"].get(
                "assessment_id"
            ):
                raise StudyStateError("assessment ID does not match the study session")
            recorded_rounds = session["quality"].get("assessment_rounds", [])
            expected_passages = len(recorded_rounds)
            expected_items = sum(int(item.get("total", 0)) for item in recorded_rounds)
            if session["mode"] == "pilot" and (
                safe_summary.get("passage_count") != expected_passages
                or safe_summary.get("item_count") != expected_items
            ):
                raise StudyStateError("assessment summary does not match scored rounds")
            session["quality"]["assessment"] = safe_summary
            session["state"] = "completed"
            self._event(session, "study_completed")
            self._write(path, session)
            return self._public_session(session)

    def advance_dry_run(
        self,
        session_id: str,
        access_token: str,
        action: str,
    ) -> dict[str, Any]:
        _, session = self._read(session_id)
        self._authorize(session, access_token)
        if session["mode"] != "dry_run":
            raise StudyStateError("simulation actions are dry-run only")
        if action == "system_check":
            return self.record_system_check(
                session_id,
                access_token,
                {
                    "camera_api": True,
                    "secure_context": True,
                    "screen_size": True,
                    "network": True,
                },
            )
        if action == "calibration_start":
            return self.start_calibration(
                session_id, access_token, "DRY-RUN-NO-CAPTURE"
            )
        if action == "calibration_complete":
            return self.complete_calibration(
                session_id,
                access_token,
                {
                    "passed": True,
                    "simulated": True,
                    "raw_images_saved": 0,
                    "note": "Dry run only; no camera data was captured.",
                },
                model_name="DRY-RUN-NO-MODEL",
            )
        if action == "assessment_start":
            return self.start_assessment(session_id, access_token, "DRY-RUN-ASSESSMENT")
        if action == "assessment_complete":
            return self.complete_assessment(
                session_id,
                access_token,
                {
                    "assessment_id": "DRY-RUN-ASSESSMENT",
                    "passage_count": 0,
                    "item_count": 0,
                    "construct_count": 0,
                    "data_quality_status": "dry_run_no_data",
                    "claim_status": "not_estimated",
                },
            )
        raise StudyValidationError("unknown dry-run action")

    def withdraw(
        self,
        session_id: str,
        *,
        access_token: str = "",
        withdrawal_code: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            path, session = self._read(session_id)
            access_ok = False
            if access_token:
                try:
                    self._authorize(session, access_token)
                    access_ok = True
                except StudyAuthorizationError:
                    access_ok = False
            withdrawal_ok = bool(withdrawal_code) and secrets.compare_digest(
                self._secret_hash(withdrawal_code),
                str(session.get("withdrawal_code_sha256") or ""),
            )
            if not access_ok and not withdrawal_ok:
                raise StudyAuthorizationError("invalid withdrawal credential")
            if session["state"] == "withdrawn":
                raise StudyStateError("study session is already withdrawn")

            deleted: list[str] = []
            gaze_session_id = str(
                dict(session.get("linked_data") or {}).get("gaze_session_id") or ""
            )
            sessions_root = (self.root / "data" / "sessions").resolve()
            if gaze_session_id:
                gaze_path = (sessions_root / gaze_session_id).resolve()
                if gaze_path.parent == sessions_root and gaze_path.is_dir():
                    shutil.rmtree(gaze_path)
                    deleted.append(f"data/sessions/{gaze_session_id}")

            model_name = str(
                dict(session.get("linked_data") or {}).get("model_name") or ""
            )
            if model_name:
                from core.gaze_core.model_registry import delete_model

                result = delete_model(self.root, model_name)
                if result.get("ok"):
                    deleted.append(f"examples/models/{model_name}.json")

            receipt_id = "WR-" + secrets.token_hex(8).upper()
            tombstone = {
                "schema_version": 1,
                "protocol_id": session["protocol_id"],
                "study_session_id": session_id,
                "participant_id": session["participant_id"],
                "mode": session["mode"],
                "state": "withdrawn",
                "withdrawn_at_utc": _utc_now(),
                "withdrawal_receipt_id": receipt_id,
                "deleted_scopes": deleted,
                "note": (
                    "Identifiable session payload was replaced by this minimal "
                    "tombstone."
                ),
            }
            session_dir = path.parent
            if session_dir.is_dir():
                shutil.rmtree(session_dir)
            self._write(path, tombstone)
            return {"ok": True, **tombstone}
