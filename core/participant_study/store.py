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

from .general_collection import (
    assignment_for_cell,
    canonical_sha256,
    classify_gaze_quality,
    load_general_bank,
    load_general_protocol,
    normalize_telemetry_batch,
    probe_order,
    public_passage,
    summarize_validation_samples,
    validate_general_design,
    validate_profile,
    validate_round_payload,
    validate_system_profile,
)
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
        buckets = {
            "dry_run": "dry_runs",
            "pilot": "pilot",
            "rehearsal": "rehearsals",
        }
        if mode not in buckets:
            raise StudyValidationError("invalid study mode")
        bucket = buckets[mode]
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
        for mode in ("dry_run", "pilot", "rehearsal"):
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

    def _consume_invite(
        self,
        invite_code: str,
        session_id: str,
        *,
        mode: str = "pilot",
    ) -> dict[str, Any]:
        filename = "collection_invites.json" if mode == "rehearsal" else "invites.json"
        invite_path = self._study_root(mode) / filename
        if not invite_path.exists():
            raise StudyNotReadyError(f"no {mode} invitation registry is available")
        registry = json.loads(invite_path.read_text(encoding="utf-8"))
        supplied = self._secret_hash(invite_code)
        matched = None
        for invite in registry.get("invites", []):
            if secrets.compare_digest(str(invite.get("code_sha256", "")), supplied):
                matched = invite
                break
        if matched is None or matched.get("used_at_utc"):
            raise StudyAuthorizationError("invalid or already-used invitation code")
        if int(matched.get("visit_index", 1)) == 2:
            pair_id = str(matched.get("pair_id") or "")
            first = next(
                (
                    item
                    for item in registry.get("invites", [])
                    if item.get("pair_id") == pair_id
                    and int(item.get("visit_index", 0)) == 1
                ),
                None,
            )
            if not first or not first.get("used_at_utc"):
                raise StudyStateError("visit 1 must be completed before visit 2")
            first_session_id = str(first.get("study_session_id") or "")
            try:
                _, first_session = self._read(first_session_id)
            except StudyError as exc:
                raise StudyStateError("visit 1 session is unavailable") from exc
            if first_session.get("state") != "completed":
                raise StudyStateError("visit 1 must be completed before visit 2")
            general_protocol = load_general_protocol()
            elapsed = datetime.now(UTC) - datetime.fromisoformat(
                str(first["used_at_utc"])
            )
            minimum = timedelta(
                hours=int(general_protocol["sessions"]["minimum_interval_hours"])
            )
            maximum = timedelta(
                hours=int(general_protocol["sessions"]["maximum_interval_hours"])
            )
            if elapsed < minimum:
                raise StudyStateError("visit 2 is earlier than the frozen retest interval")
            if elapsed > maximum:
                raise StudyStateError("visit 2 is later than the frozen retest interval")
        matched["used_at_utc"] = _utc_now()
        matched["study_session_id"] = session_id
        _atomic_json(invite_path, registry)
        return dict(matched)

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

    def create_collection_invite_pairs(self, count: int) -> list[dict[str, Any]]:
        """Create local development-only A/B retest invitation pairs."""

        if not self.activation.get("rehearsal_ready"):
            raise StudyNotReadyError(
                "rehearsal invitations are locked until local privacy gates pass"
            )
        if not 1 <= count <= 100:
            raise StudyValidationError("invite pair count must be between 1 and 100")
        general_protocol = load_general_protocol()
        bank = load_general_bank()
        design = validate_general_design(general_protocol, bank)
        path = self._study_root("rehearsal") / "collection_invites.json"
        created: list[dict[str, Any]] = []
        with self._lock:
            if path.exists():
                registry = json.loads(path.read_text(encoding="utf-8"))
            else:
                registry = {
                    "schema_version": 1,
                    "protocol_id": general_protocol["protocol_id"],
                    "protocol_version": general_protocol["protocol_version"],
                    "protocol_sha256": design["protocol_sha256"],
                    "bank_id": bank["bank_id"],
                    "bank_version": bank["bank_version"],
                    "bank_sha256": design["bank_sha256"],
                    "dataset_role": general_protocol["dataset_role"],
                    "created_at_utc": _utc_now(),
                    "invites": [],
                }
            existing_pair_ids = {
                str(item.get("pair_id")) for item in registry.get("invites", [])
            }
            base_pair_count = len(existing_pair_ids)
            for offset in range(count):
                schedule_cell = (base_pair_count + offset) % 12
                assignment = assignment_for_cell(schedule_cell, bank=bank)
                pair_id = "PAIR-" + secrets.token_hex(8).upper()
                participant_id = "GP-" + secrets.token_hex(6).upper()
                visit_results: list[dict[str, Any]] = []
                for visit in assignment["visits"]:
                    code = f"LGR-{secrets.token_hex(6).upper()}"
                    registry["invites"].append(
                        {
                            "code_sha256": self._secret_hash(code),
                            "created_at_utc": _utc_now(),
                            "used_at_utc": None,
                            "study_session_id": None,
                            "pair_id": pair_id,
                            "participant_id": participant_id,
                            "schedule_cell": schedule_cell,
                            "sequence": assignment["sequence"],
                            "order_cell": assignment["order_cell"],
                            "visit_index": visit["visit_index"],
                            "form_id": visit["form_id"],
                            "passage_order": visit["passage_order"],
                            "protocol_sha256": design["protocol_sha256"],
                            "bank_sha256": design["bank_sha256"],
                        }
                    )
                    visit_results.append(
                        {
                            "visit_index": visit["visit_index"],
                            "form_id": visit["form_id"],
                            "invite_code": code,
                        }
                    )
                created.append(
                    {
                        "pair_id": pair_id,
                        "participant_id": participant_id,
                        "schedule_cell": schedule_cell,
                        "sequence": assignment["sequence"],
                        "visits": visit_results,
                    }
                )
                existing_pair_ids.add(pair_id)
            _atomic_json(path, registry)
        return created

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
        with self._lock:
            session_paths = list(
                self._study_root("pilot").glob("ST-*/session.json")
            ) + list(self._study_root("rehearsal").glob("ST-*/session.json"))
            for path in session_paths:
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
        if mode not in {"dry_run", "pilot", "rehearsal"}:
            raise StudyValidationError("mode must be dry_run, rehearsal, or pilot")
        if mode == "pilot" and not self.activation["pilot_ready"]:
            raise StudyNotReadyError(
                "real participant collection is locked; run the readiness audit"
            )
        if mode == "rehearsal" and not self.activation.get("rehearsal_ready"):
            raise StudyNotReadyError(
                "local development rehearsal is locked; run the readiness audit"
            )
        enrollment = self._validate_enrollment(payload)
        session_id = "ST-" + secrets.token_hex(10).upper()
        participant_id = "P-" + secrets.token_hex(6).upper()
        access_token = secrets.token_urlsafe(32)
        withdrawal_code = "WD-" + secrets.token_hex(8).upper()
        timestamp = _utc_now()
        path = self._session_path(session_id, mode)
        invite_metadata: dict[str, Any] = {}
        with self._lock:
            if mode in {"pilot", "rehearsal"}:
                invite_code = str(payload.get("invite_code") or "").strip()
                if not invite_code:
                    raise StudyAuthorizationError(
                        f"{mode} invitation code is required"
                    )
                invite_metadata = self._consume_invite(
                    invite_code,
                    session_id,
                    mode=mode,
                )
                if mode == "rehearsal":
                    participant_id = str(invite_metadata["participant_id"])
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
        if mode == "rehearsal":
            session["collection_assignment"] = {
                key: invite_metadata[key]
                for key in (
                    "pair_id",
                    "schedule_cell",
                    "sequence",
                    "order_cell",
                    "visit_index",
                    "form_id",
                    "passage_order",
                    "protocol_sha256",
                    "bank_sha256",
                )
            }
            session["dataset_role"] = (
                "workflow_quality_and_development_exploration_only"
            )
            session["events"].append(
                {
                    "at_utc": timestamp,
                    "event": "general_collection_assignment_consumed",
                    "visit_index": invite_metadata["visit_index"],
                    "form_id": invite_metadata["form_id"],
                }
            )
        with self._lock:
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
        public = {
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
        if session.get("collection_assignment"):
            public["collection_assignment"] = session["collection_assignment"]
        collection = dict(session.get("general_collection") or {})
        if collection:
            public["general_collection"] = {
                "assessment_id": collection.get("assessment_id"),
                "phase": collection.get("phase"),
                "completed_rounds": len(collection.get("rounds", [])),
                "required_rounds": len(
                    dict(session.get("collection_assignment") or {}).get(
                        "passage_order", []
                    )
                ),
                "current_round": collection.get("current_round"),
                "validations": {
                    key: {
                        metric: value
                        for metric, value in dict(summary).items()
                        if metric != "samples"
                    }
                    for key, summary in dict(collection.get("validations") or {}).items()
                },
                "gaze_quality_band": collection.get("gaze_quality_band"),
            }
        return public

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

    def record_general_profile(
        self,
        session_id: str,
        access_token: str,
        profile: Mapping[str, object],
    ) -> dict[str, Any]:
        """Store only the frozen categorical profile; direct identifiers fail."""

        try:
            normalized = validate_profile(profile)
        except ValueError as exc:
            raise StudyValidationError(str(exc)) from exc
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session.get("mode") != "rehearsal":
                raise StudyStateError("general profile is rehearsal-only in v1")
            if session.get("state") != "consented":
                raise StudyStateError("profile requires a consented session")
            collection = session.setdefault("general_collection", {})
            existing = collection.get("profile")
            if existing and existing != normalized:
                raise StudyStateError("the frozen participant profile cannot be changed")
            collection["profile"] = normalized
            collection["phase"] = "profile_recorded"
            self._event(session, "general_profile_recorded")
            self._write(path, session)
            return self._public_session(session)

    def record_general_system_check(
        self,
        session_id: str,
        access_token: str,
        payload: Mapping[str, object],
    ) -> dict[str, Any]:
        try:
            normalized = validate_system_profile(payload)
        except ValueError as exc:
            raise StudyValidationError(str(exc)) from exc
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            collection = session.get("general_collection")
            if session.get("mode") != "rehearsal" or not isinstance(collection, dict):
                raise StudyStateError("general system check requires a rehearsal profile")
            if session.get("state") != "consented":
                raise StudyStateError("system check requires a consented session")
            if not collection.get("profile"):
                raise StudyStateError("participant profile must be recorded first")
            session["state"] = "system_check_passed"
            session["quality"]["general_system_check"] = normalized
            collection["phase"] = "system_check_passed"
            self._event(session, "general_system_check_passed")
            self._write(path, session)
            return self._public_session(session)

    def start_general_collection(
        self,
        session_id: str,
        access_token: str,
    ) -> dict[str, Any]:
        general_protocol = load_general_protocol()
        bank = load_general_bank()
        design = validate_general_design(general_protocol, bank)
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session.get("mode") != "rehearsal":
                raise StudyStateError("general collection is rehearsal-only in v1")
            if session.get("state") not in {
                "calibration_complete",
                "assessment_in_progress",
            }:
                raise StudyStateError("general collection requires completed calibration")
            assignment = dict(session.get("collection_assignment") or {})
            if assignment.get("protocol_sha256") != design["protocol_sha256"]:
                raise StudyStateError("assigned protocol no longer matches the frozen design")
            if assignment.get("bank_sha256") != design["bank_sha256"]:
                raise StudyStateError("assigned bank no longer matches the frozen design")
            collection = session.setdefault("general_collection", {})
            if not collection.get("profile"):
                raise StudyStateError("participant profile is missing")
            assessment_id = collection.get("assessment_id")
            if not assessment_id:
                assessment_id = "GC-" + secrets.token_hex(10).upper()
                collection.update(
                    {
                        "assessment_id": assessment_id,
                        "protocol_id": general_protocol["protocol_id"],
                        "protocol_version": general_protocol["protocol_version"],
                        "protocol_sha256": design["protocol_sha256"],
                        "bank_id": bank["bank_id"],
                        "bank_version": bank["bank_version"],
                        "bank_sha256": design["bank_sha256"],
                        "dataset_role": general_protocol["dataset_role"],
                        "phase": "start_validation_required",
                        "validations": {},
                        "rounds": [],
                        "current_round": None,
                        "telemetry_stats": {
                            "batch_count": 0,
                            "attempt_count": 0,
                            "successful_count": 0,
                            "head_pose_min": [None, None],
                            "head_pose_max": [None, None],
                            "face_scale_min": None,
                            "face_scale_max": None,
                        },
                    }
                )
                self._event(session, "general_collection_started")
            session["state"] = "assessment_in_progress"
            session["linked_data"]["assessment_id"] = str(assessment_id)
            self._write(path, session)
            return self._public_session(session)

    def record_general_validation(
        self,
        session_id: str,
        access_token: str,
        *,
        phase: str,
        samples: list[Mapping[str, object]],
    ) -> dict[str, Any]:
        if phase not in {"start", "end"}:
            raise StudyValidationError("validation phase must be start or end")
        try:
            summary = summarize_validation_samples(samples)
        except ValueError as exc:
            raise StudyValidationError(str(exc)) from exc
        summary["samples_sha256"] = canonical_sha256(summary["samples"])
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session.get("state") != "assessment_in_progress":
                raise StudyStateError("general collection is not in progress")
            collection = dict(session.get("general_collection") or {})
            expected_phase = (
                "start_validation_required" if phase == "start" else "end_validation_required"
            )
            if collection.get("phase") != expected_phase:
                existing = dict(collection.get("validations") or {}).get(phase)
                if existing and existing.get("samples_sha256") == summary["samples_sha256"]:
                    return self._public_session(session)
                raise StudyStateError(f"{phase} validation is not expected now")
            validations = collection.setdefault("validations", {})
            validations[phase] = summary
            if phase == "start":
                collection["phase"] = "reading_ready"
                self._event(session, "general_start_validation_recorded")
            else:
                collection["phase"] = "completed"
                telemetry = dict(collection.get("telemetry_stats") or {})
                rounds = list(collection.get("rounds") or [])
                reading_seconds = sum(
                    float(item.get("reading_elapsed_ms", 0)) for item in rounds
                ) / 1000.0
                attempts = int(telemetry.get("attempt_count", 0))
                successes = int(telemetry.get("successful_count", 0))
                validation_summaries = [
                    dict(validations.get(key) or {}) for key in ("start", "end")
                ]
                medians = [
                    float(item["median_spatial_error_px"])
                    for item in validation_summaries
                    if item.get("median_spatial_error_px") is not None
                ]
                p90_values = [
                    float(item["p90_spatial_error_px"])
                    for item in validation_summaries
                    if item.get("p90_spatial_error_px") is not None
                ]
                metrics = {
                    "median_spatial_error_px": max(medians) if medians else None,
                    "p90_spatial_error_px": max(p90_values) if p90_values else None,
                    "precision_rms_px": max(
                        (
                            float(item["precision_rms_px"])
                            for item in validation_summaries
                            if item.get("precision_rms_px") is not None
                        ),
                        default=None,
                    ),
                    "prediction_success_fraction": successes / attempts if attempts else 0.0,
                    "effective_sampling_hz": successes / reading_seconds
                    if reading_seconds > 0
                    else 0.0,
                    "head_pose_range": [
                        (
                            float(telemetry["head_pose_max"][index])
                            - float(telemetry["head_pose_min"][index])
                            if telemetry.get("head_pose_min", [None, None])[index]
                            is not None
                            else None
                        )
                        for index in range(2)
                    ],
                    "face_scale_range": (
                        float(telemetry["face_scale_max"])
                        - float(telemetry["face_scale_min"])
                        if telemetry.get("face_scale_min") is not None
                        else None
                    ),
                    "drift_change_px": (
                        float(validations["end"]["median_spatial_error_px"])
                        - float(validations["start"]["median_spatial_error_px"])
                        if validations["end"].get("median_spatial_error_px") is not None
                        and validations["start"].get("median_spatial_error_px") is not None
                        else None
                    ),
                }
                if metrics["median_spatial_error_px"] is None or metrics[
                    "p90_spatial_error_px"
                ] is None:
                    quality_band = "behavioral_only"
                else:
                    quality_band = classify_gaze_quality(metrics)
                collection["gaze_quality_metrics"] = metrics
                collection["gaze_quality_band"] = quality_band
                session["quality"]["general_collection"] = {
                    **metrics,
                    "gaze_quality_band": quality_band,
                    "behavioral_labels_retained": True,
                    "threshold_status": "rehearsal_descriptive_not_promotion_thresholds",
                }
                session["state"] = "completed"
                self._event(
                    session,
                    "general_collection_completed",
                    gaze_quality_band=quality_band,
                )
            session["general_collection"] = collection
            self._write(path, session)
            return self._public_session(session)

    def begin_general_round(
        self,
        session_id: str,
        access_token: str,
    ) -> dict[str, Any]:
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            if session.get("state") != "assessment_in_progress":
                raise StudyStateError("general collection is not in progress")
            collection = dict(session.get("general_collection") or {})
            assignment = dict(session.get("collection_assignment") or {})
            if collection.get("phase") in {"reading_active", "probes_open"}:
                current = dict(collection.get("current_round") or {})
            elif collection.get("phase") == "reading_ready":
                completed = len(collection.get("rounds", []))
                order = list(assignment.get("passage_order") or [])
                if completed >= len(order):
                    collection["phase"] = "end_validation_required"
                    session["general_collection"] = collection
                    self._write(path, session)
                    return {"ok": True, "is_finished": True, "phase": collection["phase"]}
                current = {
                    "round_number": completed + 1,
                    "passage_id": order[completed],
                    "started_at_utc": _utc_now(),
                }
                collection["current_round"] = current
                collection["phase"] = "reading_active"
                session["general_collection"] = collection
                self._event(
                    session,
                    "general_round_started",
                    round_number=current["round_number"],
                    passage_id=current["passage_id"],
                )
                self._write(path, session)
            else:
                raise StudyStateError("a reading round is not expected now")
            return {
                "ok": True,
                "is_finished": False,
                "phase": collection["phase"],
                "round_number": current["round_number"],
                "round_count": len(assignment.get("passage_order") or []),
                "passage": public_passage(str(current["passage_id"])),
            }

    def open_general_word_reviews(
        self,
        session_id: str,
        access_token: str,
        *,
        passage_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            collection = dict(session.get("general_collection") or {})
            current = dict(collection.get("current_round") or {})
            if collection.get("phase") not in {"reading_active", "probes_open"}:
                raise StudyStateError("word reviews are not expected now")
            if passage_id != current.get("passage_id"):
                raise StudyStateError("word reviews do not match the current passage")
            if collection.get("phase") == "reading_active":
                collection["phase"] = "probes_open"
                session["general_collection"] = collection
                self._event(
                    session,
                    "general_word_reviews_opened",
                    passage_id=passage_id,
                )
                self._write(path, session)
            assignment = dict(session.get("collection_assignment") or {})
            probes = probe_order(
                passage_id,
                str(session["participant_id"]),
                int(assignment["visit_index"]),
            )
            return {
                "ok": True,
                "passage_id": passage_id,
                "probes": [
                    {"probe_id": item["probe_id"], "surface": item["surface"]}
                    for item in probes
                ],
            }

    def record_general_telemetry_batch(
        self,
        session_id: str,
        access_token: str,
        payload: Mapping[str, object],
    ) -> dict[str, Any]:
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            collection = dict(session.get("general_collection") or {})
            current = dict(collection.get("current_round") or {})
            if collection.get("phase") not in {"reading_active", "probes_open"}:
                raise StudyStateError("telemetry is not expected now")
            passage_id = str(current.get("passage_id") or "")
            word_count = int(public_passage(passage_id)["word_count"])
            try:
                normalized = normalize_telemetry_batch(
                    payload,
                    maximum_word_index=word_count,
                )
            except ValueError as exc:
                raise StudyValidationError(str(exc)) from exc
            if normalized["passage_id"] != passage_id:
                raise StudyStateError("telemetry does not match the current passage")
            batch_path = (
                path.parent
                / "collection"
                / "telemetry"
                / passage_id
                / f"{normalized['batch_id']}.json"
            )
            payload_digest = canonical_sha256(normalized)
            if batch_path.exists():
                existing = json.loads(batch_path.read_text(encoding="utf-8"))
                if existing.get("payload_sha256") != payload_digest:
                    raise StudyStateError("telemetry batch ID was reused with new content")
                return {"ok": True, "idempotent": True, "batch_id": normalized["batch_id"]}
            observation = {
                "schema_version": 1,
                "participant_id": session["participant_id"],
                "study_session_id": session["study_session_id"],
                "visit_index": session["collection_assignment"]["visit_index"],
                "capture_session_id": session.get("linked_data", {}).get(
                    "gaze_session_id"
                ),
                "passage_id": passage_id,
                "received_at_utc": _utc_now(),
                "payload_sha256": payload_digest,
                **normalized,
            }
            _atomic_json(batch_path, observation)
            stats = collection.setdefault("telemetry_stats", {})
            stats["batch_count"] = int(stats.get("batch_count", 0)) + 1
            stats["attempt_count"] = int(stats.get("attempt_count", 0)) + len(
                normalized["samples"]
            )
            successful = [
                item for item in normalized["samples"] if item["prediction_success"]
            ]
            stats["successful_count"] = int(stats.get("successful_count", 0)) + len(
                successful
            )
            pose_min = list(stats.get("head_pose_min") or [None, None])
            pose_max = list(stats.get("head_pose_max") or [None, None])
            scales: list[float] = []
            for item in successful:
                pose = item["head_pose_pitch_yaw"]
                for index in range(2):
                    pose_min[index] = (
                        pose[index]
                        if pose_min[index] is None
                        else min(float(pose_min[index]), pose[index])
                    )
                    pose_max[index] = (
                        pose[index]
                        if pose_max[index] is None
                        else max(float(pose_max[index]), pose[index])
                    )
                bbox = item["normalized_face_bbox"]
                scales.append(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))
            stats["head_pose_min"] = pose_min
            stats["head_pose_max"] = pose_max
            if scales:
                old_min = stats.get("face_scale_min")
                old_max = stats.get("face_scale_max")
                stats["face_scale_min"] = (
                    min(scales) if old_min is None else min(float(old_min), *scales)
                )
                stats["face_scale_max"] = (
                    max(scales) if old_max is None else max(float(old_max), *scales)
                )
            session["general_collection"] = collection
            self._write(path, session)
            return {"ok": True, "idempotent": False, "batch_id": normalized["batch_id"]}

    def record_general_round(
        self,
        session_id: str,
        access_token: str,
        *,
        passage_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, Any]:
        with self._lock:
            path, session = self._read(session_id)
            self._authorize(session, access_token)
            collection = dict(session.get("general_collection") or {})
            current = dict(collection.get("current_round") or {})
            if collection.get("phase") != "probes_open":
                raise StudyStateError("the current word reviews have not been opened")
            if passage_id != current.get("passage_id"):
                raise StudyStateError("round does not match the current passage")
            assignment = dict(session.get("collection_assignment") or {})
            try:
                normalized = validate_round_payload(
                    payload,
                    passage_id=passage_id,
                    participant_id=str(session["participant_id"]),
                    visit_index=int(assignment["visit_index"]),
                )
            except ValueError as exc:
                raise StudyValidationError(str(exc)) from exc
            round_number = int(current["round_number"])
            round_path = (
                path.parent / "collection" / "rounds" / f"R{round_number:02d}.json"
            )
            if round_path.exists():
                raise StudyStateError("round was already recorded")
            observation = {
                "schema_version": 1,
                "participant_id": session["participant_id"],
                "study_session_id": session["study_session_id"],
                "visit_index": assignment["visit_index"],
                "form_id": assignment["form_id"],
                "round_number": round_number,
                "recorded_at_utc": _utc_now(),
                **normalized,
            }
            _atomic_json(round_path, observation)
            labels: dict[str, int] = {}
            for item in normalized["word_reviews"]:
                label = str(item["label"])
                labels[label] = labels.get(label, 0) + 1
            collection.setdefault("rounds", []).append(
                {
                    "round_number": round_number,
                    "passage_id": passage_id,
                    "passage_family_id": normalized["passage_family_id"],
                    "reading_elapsed_ms": normalized["reading_elapsed_ms"],
                    "label_counts": labels,
                    "word_layout_sha256": normalized["word_layout_sha256"],
                    "probe_order_sha256": normalized["probe_order_sha256"],
                    "round_payload_sha256": canonical_sha256(observation),
                }
            )
            collection["current_round"] = None
            required_rounds = len(assignment.get("passage_order") or [])
            collection["phase"] = (
                "end_validation_required"
                if len(collection["rounds"]) == required_rounds
                else "reading_ready"
            )
            session["general_collection"] = collection
            self._event(
                session,
                "general_round_recorded",
                round_number=round_number,
                passage_id=passage_id,
            )
            self._write(path, session)
            return self._public_session(session)

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
