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

SELF_ONLY_READING_VIDEO_SCOPE_ID = "retain_reading_video_self_development"
SELF_ONLY_READING_VIDEO_CATEGORY_ID = "optional_self_development_reading_video"
FORMAL_VIDEO_SCOPE_BLOCKER = "optional_video_scope_present_in_formal_protocol"
FROZEN_SELF_ONLY_READING_VIDEO_SCOPE = {
    "id": SELF_ONLY_READING_VIDEO_SCOPE_ID,
    "self_development_only": True,
    "text_zh": (
        "我另行同意保存每篇文章閱讀期間的無音訊 webcam 影片，作為我本人的 "
        "development data；我了解它存於未加密 D 槽、不設自動刪除期限、不得作為"
        "正式或 confirmation 證據。"
    ),
}
FROZEN_TEMPORARY_CALIBRATION_CATEGORY = {
    "id": "temporary_calibration_images",
    "required": True,
    "description_zh": (
        "校正時擷取的臉部影格與處理後影像；只用於校正與品質檢查，正式研究最長"
        "保存時間由啟動設定揭露。"
    ),
}
FROZEN_TRANSIENT_READING_CATEGORY = {
    "id": "transient_reading_frames",
    "required": True,
    "description_zh": (
        "閱讀時逐次送出的視線推論影格只在記憶體中處理，不把這些請求影格逐張寫入"
        "研究儲存空間；系統保留衍生摘要。另行同意的 self-only 無聲閱讀影片屬於"
        "下一個獨立資料類別。"
    ),
}
FORMAL_TRANSIENT_READING_CATEGORY = {
    "id": "transient_reading_frames",
    "required": True,
    "description_zh": (
        "閱讀時逐次送出的視線推論影格只在記憶體中處理，不把這些請求影格逐張寫入"
        "研究儲存空間；系統保留衍生摘要。"
    ),
}
FROZEN_SELF_ONLY_READING_VIDEO_CATEGORY = {
    "id": SELF_ONLY_READING_VIDEO_CATEGORY_ID,
    "required": False,
    "description_zh": (
        "只有研究者本人在未加密 self-only development 模式另行勾選時，才保存"
        "每篇文章閱讀期間的無音訊 webcam 影片；不錄製同意、背景表單或單字回顧畫面。"
    ),
}
FROZEN_RAW_VIDEO_CHECK = {
    "id": "raw_video_not_collected",
    "question_zh": "v1 會錄製或保存完整校正影片嗎？",
    "options": {
        "not_collected": "不會；未來若要收集必須另行同意與審查",
        "collected": "會，而且會無限期保存",
    },
    "correct": "not_collected",
}
FROZEN_SELF_ONLY_READING_VIDEO_CHECK = {
    "id": "reading_video_optional_self_only",
    "question_zh": "什麼情況下系統才會保存閱讀期間的 webcam 影片？",
    "options": {
        "explicit_self_development_scope": (
            "只有研究者本人使用 self-only development 模式並另行勾選時"
        ),
        "always_for_every_participant": "所有受試者都會自動保存",
    },
    "correct": "explicit_self_development_scope",
}
MEDIA_MARKERS = ("video", "audio", "webcam", "影片", "影格", "音訊")
SELF_ONLY_DATA_CATEGORIES_SHA256 = (
    "375f8a25c028fec857ad74fb3a0884e2a8162b548549688735d2bc1571064567"
)
SELF_ONLY_COMPREHENSION_CHECKS_SHA256 = (
    "8a0b59d08a55b0a095a841d00ea9a9f9666fa2e349eafd003f41c2a2c599eca2"
)
FORMAL_DATA_CATEGORIES_SHA256 = (
    "9026a44f2f5c6cfa74f32d399e135ceaac03e4fcd802caf06a95a963075bc898"
)
FORMAL_COMPREHENSION_CHECKS_SHA256 = (
    "cdee2bcca3aa7043dad549a49ffc0a760c6366f6f383aaca3ff68403940f9b4f"
)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _objects_by_id(
    value: object, label: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        errors.append(f"{label}_invalid")
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            errors.append(f"{label}_item_invalid")
            continue
        item_id = item["id"]
        if item_id in indexed:
            errors.append(f"{label}_duplicate_id")
            continue
        indexed[item_id] = item
    return indexed


def _contains_media_marker(value: object) -> bool:
    if isinstance(value, dict):
        text = " ".join(str(item) for item in value.values())
    else:
        text = str(value or "")
    lowered = text.casefold()
    return any(marker.casefold() in lowered for marker in MEDIA_MARKERS)


def optional_video_scope_boundary(
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Accept only no-video formal or the exact frozen self-only exception."""

    errors: list[str] = []
    optional_scopes = protocol.get("optional_scopes")
    if optional_scopes == []:
        mode = "formal_video_disabled"
    elif optional_scopes == [FROZEN_SELF_ONLY_READING_VIDEO_SCOPE]:
        mode = "self_only_reading_video"
    else:
        mode = "invalid"
        errors.append("optional_scopes_invalid")

    categories = _objects_by_id(
        protocol.get("data_categories"), "data_categories", errors
    )
    expected_categories_sha256 = (
        SELF_ONLY_DATA_CATEGORIES_SHA256
        if mode == "self_only_reading_video"
        else FORMAL_DATA_CATEGORIES_SHA256
    )
    if _canonical_sha256(protocol.get("data_categories")) != (
        expected_categories_sha256
    ):
        errors.append("data_categories_not_frozen_for_video_mode")
    expected_transient = (
        FROZEN_TRANSIENT_READING_CATEGORY
        if mode == "self_only_reading_video"
        else FORMAL_TRANSIENT_READING_CATEGORY
    )
    if categories.get("temporary_calibration_images") != (
        FROZEN_TEMPORARY_CALIBRATION_CATEGORY
    ):
        errors.append("temporary_calibration_image_boundary_changed")
    if categories.get("transient_reading_frames") != expected_transient:
        errors.append("transient_reading_frame_boundary_changed")

    expected_media_category_ids = {
        "temporary_calibration_images",
        "transient_reading_frames",
    }
    optional_category = categories.get(SELF_ONLY_READING_VIDEO_CATEGORY_ID)
    if mode == "self_only_reading_video":
        expected_media_category_ids.add(SELF_ONLY_READING_VIDEO_CATEGORY_ID)
        if optional_category != FROZEN_SELF_ONLY_READING_VIDEO_CATEGORY:
            errors.append("optional_reading_video_category_changed")
    elif optional_category is not None:
        errors.append("formal_protocol_retains_optional_reading_video_category")
    actual_media_category_ids = {
        item_id
        for item_id, item in categories.items()
        if _contains_media_marker(item)
    }
    if actual_media_category_ids != expected_media_category_ids:
        errors.append("unexpected_media_data_category")

    checks = _objects_by_id(
        protocol.get("comprehension_checks"), "comprehension_checks", errors
    )
    expected_checks_sha256 = (
        SELF_ONLY_COMPREHENSION_CHECKS_SHA256
        if mode == "self_only_reading_video"
        else FORMAL_COMPREHENSION_CHECKS_SHA256
    )
    if _canonical_sha256(protocol.get("comprehension_checks")) != (
        expected_checks_sha256
    ):
        errors.append("comprehension_checks_not_frozen_for_video_mode")
    if checks.get("raw_video_not_collected") != FROZEN_RAW_VIDEO_CHECK:
        errors.append("full_calibration_video_comprehension_changed")
    optional_check = checks.get("reading_video_optional_self_only")
    if mode == "self_only_reading_video":
        if optional_check != FROZEN_SELF_ONLY_READING_VIDEO_CHECK:
            errors.append("self_only_reading_video_comprehension_changed")
    elif optional_check is not None:
        errors.append("formal_protocol_retains_self_only_video_comprehension")
    expected_media_check_ids = {"raw_video_not_collected"}
    if mode == "self_only_reading_video":
        expected_media_check_ids.add("reading_video_optional_self_only")
    actual_media_check_ids = {
        item_id
        for item_id, item in checks.items()
        if _contains_media_marker(item)
    }
    if actual_media_check_ids != expected_media_check_ids:
        errors.append("unexpected_media_comprehension_check")

    bounded = not errors
    return {
        "status": (
            "bounded_self_only_development_reading_video"
            if bounded and mode == "self_only_reading_video"
            else "video_collection_disabled"
            if bounded and mode == "formal_video_disabled"
            else "unsafe_or_unrecognized_optional_video_scope"
        ),
        "dry_run_allowed": bounded,
        "formal_collection_allowed": bounded and mode == "formal_video_disabled",
        "full_video_collection_disabled": bounded,
        "errors": errors,
    }


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
    video_scope = optional_video_scope_boundary(active_protocol)
    mode = _setting(settings, "LEXIGAZE_STUDY_MODE", "dry_run").lower()
    ethics_status = _setting(settings, "LEXIGAZE_ETHICS_STATUS").lower()
    base_url = _setting(settings, "LEXIGAZE_PUBLIC_BASE_URL")
    retention_text = _setting(settings, "LEXIGAZE_DATA_RETENTION_DAYS")
    raw_retention_text = _setting(settings, "LEXIGAZE_RAW_FRAME_RETENTION_HOURS")
    storage_encrypted = _truthy(_setting(settings, "LEXIGAZE_STORAGE_ENCRYPTED"))
    unencrypted_self_development = _truthy(
        _setting(settings, "LEXIGAZE_UNENCRYPTED_SELF_DEVELOPMENT")
    )
    retention_policy = _setting(
        settings,
        "LEXIGAZE_DATA_RETENTION_POLICY",
        "fixed_days",
    ).lower()

    missing: list[str] = []

    def require(condition: bool, code: str) -> None:
        if not condition:
            missing.append(code)

    require(
        active_protocol.get("collection_status") == "approved_for_pilot",
        "protocol_not_approved_for_pilot",
    )
    require(
        video_scope["formal_collection_allowed"] is True,
        FORMAL_VIDEO_SCOPE_BLOCKER,
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
    require(storage_encrypted, "encrypted_storage_not_confirmed")
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
    self_development_exception = (
        unencrypted_self_development and not storage_encrypted
    )
    require_rehearsal(
        storage_encrypted or self_development_exception,
        "rehearsal_storage_policy_not_acknowledged",
    )
    require_rehearsal(
        bool(_setting(settings, "LEXIGAZE_DATA_LOCATION")),
        "data_location_missing",
    )
    if self_development_exception:
        require_rehearsal(
            retention_policy == "manual_until_researcher_deletes",
            "self_development_retention_policy_not_acknowledged",
        )
        require_rehearsal(
            retention_days == 0,
            "self_development_retention_days_must_be_zero",
        )
    else:
        require_rehearsal(
            retention_policy == "fixed_days" and 1 <= retention_days <= 30,
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
        "rehearsal_scope": (
            "local_invited_self_development_unencrypted"
            if self_development_exception
            else "local_invited_development_only"
        ),
        "rehearsal_self_only": self_development_exception,
        "storage_encrypted": storage_encrypted,
        "storage_security": (
            "unencrypted_self_development"
            if self_development_exception
            else "encrypted"
            if storage_encrypted
            else "unconfirmed"
        ),
        "retention_policy": (
            "manual_until_researcher_deletes"
            if self_development_exception
            else retention_policy
        ),
        "formal_promotion_allowed": False,
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
    if not activation["rehearsal_self_only"]:
        active_protocol["optional_scopes"] = [
            scope
            for scope in active_protocol.get("optional_scopes", [])
            if not scope.get("self_development_only")
        ]
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
        "encrypted_storage_confirmed": activation["storage_encrypted"],
        "storage_security": activation["storage_security"],
        "retention_policy": activation["retention_policy"],
        "self_only": activation["rehearsal_self_only"],
        "formal_promotion_allowed": activation["formal_promotion_allowed"],
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
