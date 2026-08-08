"""Versioned participant-study protocol and activation gates."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

PROTOCOL_PATH = Path(__file__).with_name("protocol_v1.json")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _setting(
    settings: Mapping[str, object] | None, name: str, default: str = ""
) -> str:
    if settings is not None and name in settings:
        return str(settings.get(name) or default).strip()
    return str(os.environ.get(name, default)).strip()


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    protocol_path = path or PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "protocol_version",
        "consent_version",
        "collection_status",
        "required_consent_statements",
        "comprehension_checks",
        "data_categories",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"participant study protocol missing fields: {missing}")
    statement_ids = [item["id"] for item in protocol["required_consent_statements"]]
    check_ids = [item["id"] for item in protocol["comprehension_checks"]]
    if len(statement_ids) != len(set(statement_ids)):
        raise ValueError("participant study consent statement IDs must be unique")
    if len(check_ids) != len(set(check_ids)):
        raise ValueError("participant study comprehension check IDs must be unique")
    return protocol


def protocol_digest(protocol: Mapping[str, object] | None = None) -> str:
    return hashlib.sha256(_canonical_json(protocol or load_protocol())).hexdigest()


def activation_status(
    settings: Mapping[str, object] | None = None,
    *,
    protocol: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    active_protocol = dict(protocol or load_protocol())
    mode = _setting(settings, "LEXIGAZE_STUDY_MODE", "dry_run").lower()
    ethics_status = _setting(settings, "LEXIGAZE_ETHICS_STATUS").lower()
    base_url = _setting(settings, "LEXIGAZE_PUBLIC_BASE_URL")
    retention_text = _setting(settings, "LEXIGAZE_DATA_RETENTION_DAYS")
    raw_retention_text = _setting(settings, "LEXIGAZE_RAW_FRAME_RETENTION_HOURS")

    missing: list[str] = []

    def require(condition: bool, code: str) -> None:
        if not condition:
            missing.append(code)

    require(
        active_protocol.get("collection_status") == "approved_for_pilot",
        "protocol_not_approved_for_pilot",
    )
    require(mode == "pilot", "study_mode_not_pilot")
    require(
        _truthy(_setting(settings, "LEXIGAZE_PUBLIC_STUDY_MODE")),
        "public_study_mode_disabled",
    )
    require(
        ethics_status in {"approved", "exempt_determination"},
        "ethics_determination_missing",
    )
    for name, code in (
        ("LEXIGAZE_ETHICS_REFERENCE", "ethics_reference_missing"),
        ("LEXIGAZE_INVESTIGATOR_NAME", "investigator_name_missing"),
        ("LEXIGAZE_INVESTIGATOR_EMAIL", "investigator_email_missing"),
        (
            "LEXIGAZE_PARTICIPANT_RIGHTS_CONTACT",
            "participant_rights_contact_missing",
        ),
        ("LEXIGAZE_RESEARCHER_API_KEY", "researcher_api_key_missing"),
        ("LEXIGAZE_EXTERNAL_ANCHOR_ID", "external_anchor_missing"),
        ("LEXIGAZE_NETWORK_PROCESSOR", "network_processor_missing"),
        ("LEXIGAZE_DATA_LOCATION", "data_location_missing"),
    ):
        require(bool(_setting(settings, name)), code)
    require(
        len(_setting(settings, "LEXIGAZE_ADAPTIVE_SIGNING_KEY")) >= 32,
        "adaptive_signing_key_missing_or_short",
    )
    require(
        _truthy(_setting(settings, "LEXIGAZE_EXTERNAL_ANCHOR_AUTHORIZED")),
        "external_anchor_authorization_missing",
    )
    require(
        _truthy(_setting(settings, "LEXIGAZE_NETWORK_PROCESSOR_APPROVED")),
        "network_processor_approval_missing",
    )
    require(
        _truthy(_setting(settings, "LEXIGAZE_REQUEST_BODY_LOGGING_DISABLED")),
        "request_body_logging_not_disabled",
    )
    require(
        _truthy(_setting(settings, "LEXIGAZE_STORAGE_ENCRYPTED")),
        "encrypted_storage_not_confirmed",
    )
    require(base_url.lower().startswith("https://"), "public_https_url_missing")

    try:
        retention_days = int(retention_text)
    except (TypeError, ValueError):
        retention_days = 0
    require(1 <= retention_days <= 3650, "data_retention_days_invalid")

    try:
        raw_retention_hours = int(raw_retention_text)
    except (TypeError, ValueError):
        raw_retention_hours = 0
    require(1 <= raw_retention_hours <= 24, "raw_frame_retention_hours_invalid")

    rehearsal_missing: list[str] = []

    def require_rehearsal(condition: bool, code: str) -> None:
        if not condition:
            rehearsal_missing.append(code)

    rehearsal_enabled = mode == "rehearsal" or _truthy(
        _setting(settings, "LEXIGAZE_STUDY_REHEARSAL_MODE")
    )
    local_base_url = base_url.lower().rstrip("/")
    require_rehearsal(rehearsal_enabled, "rehearsal_mode_disabled")
    require_rehearsal(
        local_base_url.startswith("http://127.0.0.1")
        or local_base_url.startswith("http://localhost"),
        "rehearsal_must_be_localhost_only",
    )
    require_rehearsal(
        _truthy(
            _setting(
                settings,
                "LEXIGAZE_REHEARSAL_ACKNOWLEDGED_DEVELOPMENT_ONLY",
            )
        ),
        "rehearsal_development_only_acknowledgement_missing",
    )
    require_rehearsal(
        _truthy(_setting(settings, "LEXIGAZE_REHEARSAL_INVITES_ONLY")),
        "rehearsal_invites_only_not_confirmed",
    )
    require_rehearsal(
        _truthy(_setting(settings, "LEXIGAZE_REQUEST_BODY_LOGGING_DISABLED")),
        "request_body_logging_not_disabled",
    )
    require_rehearsal(
        _truthy(_setting(settings, "LEXIGAZE_STORAGE_ENCRYPTED")),
        "encrypted_storage_not_confirmed",
    )
    require_rehearsal(
        bool(_setting(settings, "LEXIGAZE_DATA_LOCATION")),
        "data_location_missing",
    )
    require_rehearsal(
        1 <= retention_days <= 30,
        "rehearsal_data_retention_days_invalid",
    )
    require_rehearsal(
        1 <= raw_retention_hours <= 24,
        "raw_frame_retention_hours_invalid",
    )

    return {
        "protocol_id": active_protocol["protocol_id"],
        "protocol_version": active_protocol["protocol_version"],
        "protocol_digest_sha256": protocol_digest(active_protocol),
        "configured_mode": mode,
        "dry_run_ready": True,
        "pilot_ready": not missing,
        "missing_requirements": missing,
        "rehearsal_ready": not rehearsal_missing,
        "rehearsal_missing_requirements": rehearsal_missing,
        "rehearsal_scope": "local_invited_development_only",
        "ethics_status": ethics_status or "not_provided",
        "ethics_reference": _setting(settings, "LEXIGAZE_ETHICS_REFERENCE") or None,
        "external_anchor_id": _setting(settings, "LEXIGAZE_EXTERNAL_ANCHOR_ID") or None,
        "retention_days": retention_days or None,
        "raw_frame_retention_hours": raw_retention_hours or None,
    }


def public_protocol(
    settings: Mapping[str, object] | None = None,
    *,
    protocol: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    active_protocol = deepcopy(dict(protocol or load_protocol()))
    activation = activation_status(settings, protocol=active_protocol)
    active_protocol.pop("activation_requirements", None)
    active_protocol["protocol_digest_sha256"] = activation["protocol_digest_sha256"]
    active_protocol["activation"] = activation
    active_protocol["research_contacts"] = {
        "investigator": _setting(settings, "LEXIGAZE_INVESTIGATOR_NAME") or None,
        "investigator_email": _setting(settings, "LEXIGAZE_INVESTIGATOR_EMAIL") or None,
        "participant_rights": _setting(settings, "LEXIGAZE_PARTICIPANT_RIGHTS_CONTACT")
        or None,
    }
    active_protocol["data_governance"] = {
        "location": _setting(settings, "LEXIGAZE_DATA_LOCATION") or None,
        "retention_days": activation["retention_days"],
        "raw_frame_retention_hours": activation["raw_frame_retention_hours"],
        "encrypted_storage_confirmed": _truthy(
            _setting(settings, "LEXIGAZE_STORAGE_ENCRYPTED")
        ),
        "public_base_url": _setting(settings, "LEXIGAZE_PUBLIC_BASE_URL") or None,
        "network_processor": _setting(settings, "LEXIGAZE_NETWORK_PROCESSOR") or None,
        "network_processor_approved": _truthy(
            _setting(settings, "LEXIGAZE_NETWORK_PROCESSOR_APPROVED")
        ),
        "request_body_logging_disabled": _truthy(
            _setting(settings, "LEXIGAZE_REQUEST_BODY_LOGGING_DISABLED")
        ),
    }
    active_protocol["consent_digest_sha256"] = hashlib.sha256(
        _canonical_json(
            {
                "protocol": active_protocol,
                "contacts": active_protocol["research_contacts"],
                "governance": active_protocol["data_governance"],
            }
        )
    ).hexdigest()
    return active_protocol
