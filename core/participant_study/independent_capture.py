"""Outcome-blind validation for independent participant capture plans.

The module reads planning metadata only. It never opens participant media,
imports a model runtime, or returns slot/binding identifiers in its audit.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

ANALYSIS_ROLES = ("development", "validation", "confirmation")
PLAN_STATUSES = ("template_only", "frozen_before_collection")
BINDING_STATUSES = ("unbound", "bound", "withdrawn")
AUDIT_TARGETS = ("template", "collection", "evidence")
DEVICE_ANALYSIS_ROLES = (*ANALYSIS_ROLES, "shared")
DEVICE_CLASSES = ("desktop", "phone", "tablet", "external-camera")
DEVICE_ROLES = (
    "laptop-primary",
    "phone-secondary",
    "external-primary",
    "external-secondary",
)
SOURCE_ROLES = DEVICE_ROLES
MISSING_VIEW_POLICIES = ("not_applicable", "primary_only_fallback", "abstain")
CLOCK_STRATEGIES = (
    "hardware-trigger",
    "monotonic-offset",
    "posthoc-timestamp-alignment",
)

TOP_LEVEL_KEYS = {
    "schema_version",
    "plan_id",
    "plan_version",
    "status",
    "created_at_utc",
    "frozen_at_utc",
    "protocol",
    "requirements",
    "claims",
    "participant_slots",
    "session_slots",
    "device_slots",
    "article_slots",
    "capture_run_slots",
}
PROTOCOL_KEYS = {
    "protocol_id",
    "protocol_version",
    "protocol_commit",
    "protocol_digest_sha256",
    "consent_version",
    "consent_digest_sha256",
}
REQUIREMENT_KEYS = {
    "declared_target_participants",
    "minimum_bound_participants_per_role",
    "minimum_sessions_per_participant",
    "minimum_session_gap_hours",
    "required_condition_ids",
}
CLAIM_KEYS = {
    "participant_generalization",
    "session_generalization",
    "article_generalization",
    "device_generalization",
    "multiview_incremental_value",
    "missing_view_policy",
}
PARTICIPANT_KEYS = {
    "slot_id",
    "analysis_role",
    "binding_status",
    "binding_sha256",
}
SESSION_KEYS = {
    "slot_id",
    "participant_slot_id",
    "analysis_role",
    "planned_offset_hours",
    "binding_status",
    "binding_sha256",
}
DEVICE_KEYS = {
    "slot_id",
    "analysis_role",
    "device_role",
    "device_class",
    "binding_status",
    "binding_sha256",
}
ARTICLE_KEYS = {
    "slot_id",
    "family_slot_id",
    "analysis_role",
    "content_sha256",
    "authorization_id",
}
CAPTURE_RUN_KEYS = {
    "slot_id",
    "participant_slot_id",
    "session_slot_id",
    "analysis_role",
    "article_slot_ids",
    "condition_ids",
    "source_slots",
    "synchronization",
}
SOURCE_KEYS = {"slot_id", "device_slot_id", "source_role", "required"}
SYNCHRONIZATION_KEYS = {
    "required",
    "max_offset_ms",
    "clock_strategy",
    "relative_camera_calibration_slot_id",
    "relative_camera_calibration_sha256",
}

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
SLOT_PATTERNS = {
    "participant": re.compile(r"^PSLOT-[0-9]{3,6}$"),
    "session": re.compile(r"^SSLOT-[0-9]{3,6}-[0-9]{2,3}$"),
    "device": re.compile(r"^DSLOT-[0-9]{3,6}$"),
    "article": re.compile(r"^ASLOT-[0-9]{3,6}$"),
    "article_family": re.compile(r"^AFAMILY-[0-9]{3,6}$"),
    "capture_run": re.compile(r"^CRUN-[0-9]{3,6}-[0-9]{2,3}$"),
    "source": re.compile(r"^SRC-[0-9]{3,6}-[0-9]{2,3}-[0-9]{2,3}$"),
    "calibration": re.compile(r"^XCAL-[0-9]{3,6}$"),
}


@dataclass(frozen=True)
class CapturePlanIssue:
    code: str
    severity: str
    message: str


def canonical_plan_sha256(plan: Mapping[str, Any]) -> str:
    """Hash the exact logical JSON payload independently of formatting."""

    encoded = json.dumps(
        plan,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_capture_plan(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capture plan must be a JSON object")
    return payload


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _records(value: object) -> list[Mapping[str, Any]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if not all(isinstance(item, Mapping) for item in value):
        return None
    return list(value)


def _bounded_slug(value: object) -> bool:
    return isinstance(value, str) and bool(SLUG_PATTERN.fullmatch(value))


def _bounded_label(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return 1 <= len(value) <= 64 and bool(re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value))


def _valid_sha256(value: object, *, nullable: bool = False) -> bool:
    if value is None:
        return nullable
    return isinstance(value, str) and bool(SHA256_PATTERN.fullmatch(value))


def _valid_utc(value: object, *, nullable: bool = False) -> bool:
    if value is None:
        return nullable
    if not isinstance(value, str) or not value.endswith(("Z", "+00:00")):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def _finite_number(value: object, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) and number >= minimum else None


def _add(
    issues: list[CapturePlanIssue],
    code: str,
    message: str,
    *,
    severity: str = "error",
) -> None:
    issues.append(CapturePlanIssue(code, severity, message))


def _check_exact_keys(
    value: Mapping[str, Any] | None,
    expected: set[str],
    issues: list[CapturePlanIssue],
    code: str,
) -> bool:
    if value is None:
        _add(issues, code, "A required object has the wrong JSON type.")
        return False
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        _add(issues, code, f"A required object is missing {len(missing)} field(s).")
    if unknown:
        _add(
            issues,
            "UNKNOWN_OR_OUTCOME_FIELD",
            f"A strict-schema object contains {len(unknown)} unexpected field(s).",
        )
    return not missing and not unknown


def _valid_slot(value: object, kind: str) -> bool:
    return isinstance(value, str) and bool(SLOT_PATTERNS[kind].fullmatch(value))


def _index_records(
    records: list[Mapping[str, Any]],
    *,
    kind: str,
    issues: list[CapturePlanIssue],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    invalid = 0
    duplicates = 0
    for record in records:
        slot_id = record.get("slot_id")
        if not _valid_slot(slot_id, kind):
            invalid += 1
            continue
        if slot_id in indexed:
            duplicates += 1
            continue
        indexed[str(slot_id)] = record
    if invalid:
        _add(
            issues,
            "INVALID_SLOT_IDENTIFIER",
            f"{invalid} {kind} slot identifier(s) violate the opaque slot format.",
        )
    if duplicates:
        _add(
            issues,
            "DUPLICATE_SLOT_IDENTIFIER",
            f"{duplicates} duplicate {kind} slot identifier(s) were found.",
        )
    return indexed


def _binding_checks(
    records: list[Mapping[str, Any]],
    issues: list[CapturePlanIssue],
    *,
    kind: str,
) -> None:
    populated: list[str] = []
    invalid_pairs = 0
    for record in records:
        status = record.get("binding_status")
        digest = record.get("binding_sha256")
        if status not in BINDING_STATUSES:
            invalid_pairs += 1
            continue
        if status == "unbound" and digest is None:
            continue
        if status == "bound" and _valid_sha256(digest):
            populated.append(str(digest))
            continue
        if status == "withdrawn" and digest is None:
            continue
        invalid_pairs += 1
    if invalid_pairs:
        _add(
            issues,
            "INVALID_BINDING_STATE",
            f"{invalid_pairs} {kind} binding state/digest pair(s) are inconsistent.",
        )
    duplicates = sum(count - 1 for count in Counter(populated).values() if count > 1)
    if duplicates:
        _add(
            issues,
            "DUPLICATE_BINDING",
            f"{duplicates} duplicate {kind} binding digest(s) were found.",
        )


def audit_independent_capture_plan(
    plan: Mapping[str, Any],
    *,
    target: str = "template",
) -> dict[str, Any]:
    """Return an identifier-free structural and readiness audit."""

    if target not in AUDIT_TARGETS:
        raise ValueError(f"target must be one of {AUDIT_TARGETS}")
    issues: list[CapturePlanIssue] = []
    top = _mapping(plan)
    if top is None:
        top = {}
        _add(issues, "INVALID_PLAN_OBJECT", "The capture plan is not a JSON object.")
    _check_exact_keys(top, TOP_LEVEL_KEYS, issues, "INVALID_TOP_LEVEL_SCHEMA")

    if top.get("schema_version") != 1:
        _add(issues, "UNSUPPORTED_SCHEMA_VERSION", "The schema version must equal 1.")
    if not _bounded_slug(top.get("plan_id")) or not _bounded_slug(
        top.get("plan_version")
    ):
        _add(issues, "INVALID_PLAN_METADATA", "Plan identifiers must be bounded slugs.")
    status = top.get("status")
    if status not in PLAN_STATUSES:
        _add(issues, "INVALID_PLAN_STATUS", "The plan status is not recognized.")
    if not _valid_utc(top.get("created_at_utc")):
        _add(
            issues, "INVALID_PLAN_TIMESTAMP", "created_at_utc must be a UTC timestamp."
        )
    frozen_at = top.get("frozen_at_utc")
    if not _valid_utc(frozen_at, nullable=True):
        _add(issues, "INVALID_PLAN_TIMESTAMP", "frozen_at_utc must be null or UTC.")
    if status == "template_only" and frozen_at is not None:
        _add(
            issues, "TEMPLATE_HAS_FREEZE_TIME", "A template cannot have a freeze time."
        )
    if status == "frozen_before_collection" and frozen_at is None:
        _add(
            issues, "FROZEN_PLAN_MISSING_TIME", "A frozen plan requires a freeze time."
        )
    if _valid_utc(top.get("created_at_utc")) and _valid_utc(frozen_at, nullable=True):
        created = datetime.fromisoformat(
            str(top.get("created_at_utc")).replace("Z", "+00:00")
        )
        frozen = (
            datetime.fromisoformat(str(frozen_at).replace("Z", "+00:00"))
            if frozen_at is not None
            else None
        )
        if frozen is not None and frozen < created:
            _add(
                issues,
                "FREEZE_PRECEDES_PLAN_CREATION",
                "The freeze timestamp cannot precede plan creation.",
            )

    protocol = _mapping(top.get("protocol"))
    if _check_exact_keys(protocol, PROTOCOL_KEYS, issues, "INVALID_PROTOCOL_SCHEMA"):
        assert protocol is not None
        if not _bounded_slug(protocol.get("protocol_id")) or not _bounded_slug(
            protocol.get("protocol_version")
        ):
            _add(issues, "INVALID_PROTOCOL_REFERENCE", "Protocol IDs must be slugs.")
        if not isinstance(
            protocol.get("protocol_commit"), str
        ) or not COMMIT_PATTERN.fullmatch(str(protocol.get("protocol_commit"))):
            _add(issues, "INVALID_PROTOCOL_REFERENCE", "Protocol commit is invalid.")
        for key in ("protocol_digest_sha256", "consent_digest_sha256"):
            if not _valid_sha256(protocol.get(key)):
                _add(
                    issues,
                    "INVALID_PROTOCOL_REFERENCE",
                    "A protocol digest is invalid.",
                )
        if not _bounded_slug(protocol.get("consent_version")):
            _add(issues, "INVALID_PROTOCOL_REFERENCE", "Consent version is invalid.")

    requirements = _mapping(top.get("requirements"))
    role_minimums: Mapping[str, Any] = {}
    min_sessions = 0
    min_gap = 0.0
    required_conditions: set[str] = set()
    if _check_exact_keys(
        requirements,
        REQUIREMENT_KEYS,
        issues,
        "INVALID_REQUIREMENTS_SCHEMA",
    ):
        assert requirements is not None
        declared = requirements.get("declared_target_participants")
        if not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0:
            _add(
                issues, "INVALID_REQUIREMENT", "Declared participants must be positive."
            )
        role_minimums_value = _mapping(
            requirements.get("minimum_bound_participants_per_role")
        )
        if role_minimums_value is None or set(role_minimums_value) != set(
            ANALYSIS_ROLES
        ):
            _add(
                issues, "INVALID_REQUIREMENT", "Every analysis role needs one minimum."
            )
        else:
            role_minimums = role_minimums_value
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in role_minimums.values()
            ):
                _add(issues, "INVALID_REQUIREMENT", "Role minimums must be positive.")
        raw_min_sessions = requirements.get("minimum_sessions_per_participant")
        if (
            not isinstance(raw_min_sessions, int)
            or isinstance(raw_min_sessions, bool)
            or raw_min_sessions < 2
        ):
            _add(issues, "INVALID_REQUIREMENT", "At least two sessions are required.")
        else:
            min_sessions = raw_min_sessions
        parsed_gap = _finite_number(
            requirements.get("minimum_session_gap_hours"), minimum=0.000001
        )
        if parsed_gap is None:
            _add(
                issues,
                "INVALID_REQUIREMENT",
                "Session gap must be positive and finite.",
            )
        else:
            min_gap = parsed_gap
        conditions = requirements.get("required_condition_ids")
        if (
            not isinstance(conditions, Sequence)
            or isinstance(conditions, (str, bytes))
            or not conditions
            or any(not _bounded_label(value) for value in conditions)
            or len(set(conditions)) != len(conditions)
        ):
            _add(issues, "INVALID_REQUIREMENT", "Required conditions are invalid.")
        else:
            required_conditions = {str(value) for value in conditions}

    claims = _mapping(top.get("claims"))
    device_generalization = False
    multiview_claim = False
    missing_view_policy = None
    if _check_exact_keys(claims, CLAIM_KEYS, issues, "INVALID_CLAIMS_SCHEMA"):
        assert claims is not None
        for key in (
            "participant_generalization",
            "session_generalization",
            "article_generalization",
            "device_generalization",
            "multiview_incremental_value",
        ):
            if not isinstance(claims.get(key), bool):
                _add(issues, "INVALID_CLAIM", "Every claim flag must be boolean.")
        if any(
            claims.get(key) is not True
            for key in (
                "participant_generalization",
                "session_generalization",
                "article_generalization",
            )
        ):
            _add(
                issues,
                "REQUIRED_HOLDOUT_DISABLED",
                "Participant, session, and article holdout claims must remain enabled.",
            )
        device_generalization = claims.get("device_generalization") is True
        multiview_claim = claims.get("multiview_incremental_value") is True
        missing_view_policy = claims.get("missing_view_policy")
        if missing_view_policy not in MISSING_VIEW_POLICIES:
            _add(issues, "INVALID_CLAIM", "Missing-view policy is not recognized.")
        if multiview_claim and missing_view_policy == "not_applicable":
            _add(
                issues,
                "MULTIVIEW_MISSING_FALLBACK",
                "A multi-view claim requires a frozen missing-view policy.",
            )

    list_specs = {
        "participant_slots": (PARTICIPANT_KEYS, "participant"),
        "session_slots": (SESSION_KEYS, "session"),
        "device_slots": (DEVICE_KEYS, "device"),
        "article_slots": (ARTICLE_KEYS, "article"),
        "capture_run_slots": (CAPTURE_RUN_KEYS, "capture_run"),
    }
    lists: dict[str, list[Mapping[str, Any]]] = {}
    for key, (expected_keys, kind) in list_specs.items():
        records = _records(top.get(key))
        if records is None or not records:
            _add(issues, "INVALID_RECORD_LIST", f"The {kind} list must be non-empty.")
            records = []
        for record in records:
            _check_exact_keys(
                record,
                expected_keys,
                issues,
                f"INVALID_{kind.upper()}_SCHEMA",
            )
        lists[key] = records

    participants = lists["participant_slots"]
    sessions = lists["session_slots"]
    devices = lists["device_slots"]
    articles = lists["article_slots"]
    capture_runs = lists["capture_run_slots"]
    participant_index = _index_records(participants, kind="participant", issues=issues)
    session_index = _index_records(sessions, kind="session", issues=issues)
    device_index = _index_records(devices, kind="device", issues=issues)
    article_index = _index_records(articles, kind="article", issues=issues)
    capture_index = _index_records(capture_runs, kind="capture_run", issues=issues)
    _binding_checks(participants, issues, kind="participant")
    _binding_checks(sessions, issues, kind="session")
    _binding_checks(devices, issues, kind="device")

    participant_role_counts = Counter()
    for participant in participants:
        role = participant.get("analysis_role")
        if role not in ANALYSIS_ROLES:
            _add(issues, "INVALID_ANALYSIS_ROLE", "A participant role is invalid.")
        else:
            participant_role_counts[str(role)] += 1
    declared = (
        requirements.get("declared_target_participants") if requirements else None
    )
    if isinstance(declared, int) and declared != len(participants):
        _add(
            issues,
            "DECLARED_PARTICIPANT_COUNT_MISMATCH",
            "Declared and enumerated participant slot counts differ.",
        )
    for role in ANALYSIS_ROLES:
        minimum = role_minimums.get(role) if role_minimums else None
        if isinstance(minimum, int) and participant_role_counts[role] < minimum:
            _add(
                issues,
                "INSUFFICIENT_PLANNED_PARTICIPANTS",
                "A role has fewer planned participant slots than its minimum.",
            )

    session_offsets: dict[str, list[float]] = defaultdict(list)
    sessions_by_participant: Counter[str] = Counter()
    for session in sessions:
        role = session.get("analysis_role")
        participant_slot = session.get("participant_slot_id")
        offset = _finite_number(session.get("planned_offset_hours"), minimum=0.0)
        participant = participant_index.get(str(participant_slot))
        if role not in ANALYSIS_ROLES:
            _add(issues, "INVALID_ANALYSIS_ROLE", "A session role is invalid.")
        if participant is None:
            _add(issues, "BROKEN_REFERENCE", "A session has no participant slot.")
        elif participant.get("analysis_role") != role:
            _add(
                issues,
                "CROSS_ROLE_REFERENCE",
                "A session and its participant use different analysis roles.",
            )
        if offset is None:
            _add(issues, "INVALID_SESSION_OFFSET", "A session offset is invalid.")
        elif isinstance(participant_slot, str):
            session_offsets[participant_slot].append(offset)
        if isinstance(participant_slot, str):
            sessions_by_participant[participant_slot] += 1
    if min_sessions:
        short = sum(
            sessions_by_participant[slot_id] < min_sessions
            for slot_id in participant_index
        )
        if short:
            _add(
                issues,
                "INSUFFICIENT_SESSION_SLOTS",
                f"{short} participant slot(s) lack the planned repeated sessions.",
            )
    if min_gap:
        close_pairs = 0
        duplicate_offsets = 0
        for offsets in session_offsets.values():
            ordered = sorted(offsets)
            duplicate_offsets += len(ordered) - len(set(ordered))
            close_pairs += sum(
                right - left < min_gap
                for left, right in zip(ordered, ordered[1:], strict=False)
            )
        if duplicate_offsets or close_pairs:
            _add(
                issues,
                "SESSION_GAP_TOO_SMALL",
                "One or more repeated-session gaps violate the frozen minimum.",
            )

    device_binding_roles: dict[str, set[str]] = defaultdict(set)
    for device in devices:
        role = device.get("analysis_role")
        if role not in DEVICE_ANALYSIS_ROLES:
            _add(issues, "INVALID_ANALYSIS_ROLE", "A device role is invalid.")
        if device.get("device_role") not in DEVICE_ROLES:
            _add(issues, "INVALID_DEVICE_METADATA", "A device role label is invalid.")
        if device.get("device_class") not in DEVICE_CLASSES:
            _add(issues, "INVALID_DEVICE_METADATA", "A device class is invalid.")
        digest = device.get("binding_sha256")
        if (
            _valid_sha256(digest)
            and isinstance(role, str)
            and role in DEVICE_ANALYSIS_ROLES
        ):
            device_binding_roles[str(digest)].add(role)
    if device_generalization:
        if any(device.get("analysis_role") == "shared" for device in devices):
            _add(
                issues,
                "DEVICE_HOLDOUT_REUSES_SHARED_SLOT",
                "A device-generalization plan cannot use shared device slots.",
            )
        if any(len(roles) > 1 for roles in device_binding_roles.values()):
            _add(
                issues,
                "DEVICE_BINDING_CROSSES_ROLES",
                "A physical-device binding occurs in more than one analysis role.",
            )

    article_family_roles: dict[str, set[str]] = defaultdict(set)
    article_content_roles: dict[str, set[str]] = defaultdict(set)
    for article in articles:
        family = article.get("family_slot_id")
        role = article.get("analysis_role")
        if not _valid_slot(family, "article_family"):
            _add(
                issues, "INVALID_SLOT_IDENTIFIER", "An article family slot is invalid."
            )
        elif role in ANALYSIS_ROLES:
            article_family_roles[str(family)].add(str(role))
        else:
            _add(issues, "INVALID_ANALYSIS_ROLE", "An article role is invalid.")
        if not _valid_sha256(article.get("content_sha256"), nullable=True):
            _add(issues, "INVALID_ARTICLE_METADATA", "An article hash is invalid.")
        elif article.get("content_sha256") is not None and role in ANALYSIS_ROLES:
            article_content_roles[str(article["content_sha256"])].add(str(role))
        authorization = article.get("authorization_id")
        if authorization is not None and not _bounded_label(authorization):
            _add(
                issues,
                "INVALID_ARTICLE_METADATA",
                "An article authorization reference is invalid.",
            )
    if any(len(roles) > 1 for roles in article_family_roles.values()):
        _add(
            issues,
            "ARTICLE_FAMILY_CROSSES_ROLES",
            "An article or near-duplicate family crosses analysis roles.",
        )
    if any(len(roles) > 1 for roles in article_content_roles.values()):
        _add(
            issues,
            "ARTICLE_CONTENT_CROSSES_ROLES",
            "Identical article content appears in more than one analysis role.",
        )

    captures_by_session: Counter[str] = Counter()
    source_slot_ids: set[str] = set()
    duplicate_source_slots = 0
    multiview_runs = 0
    synchronized_multiview_runs = 0
    for capture in capture_runs:
        role = capture.get("analysis_role")
        participant_slot = capture.get("participant_slot_id")
        session_slot = capture.get("session_slot_id")
        participant = participant_index.get(str(participant_slot))
        session = session_index.get(str(session_slot))
        if role not in ANALYSIS_ROLES:
            _add(issues, "INVALID_ANALYSIS_ROLE", "A capture-run role is invalid.")
        if participant is None or session is None:
            _add(
                issues, "BROKEN_REFERENCE", "A capture run has a broken core reference."
            )
        else:
            if session.get("participant_slot_id") != participant_slot:
                _add(
                    issues,
                    "SESSION_CAPTURE_PARTICIPANT_MISMATCH",
                    "A capture run links a session to a different participant.",
                )
            if (
                participant.get("analysis_role") != role
                or session.get("analysis_role") != role
            ):
                _add(
                    issues,
                    "CROSS_ROLE_REFERENCE",
                    "A capture run crosses participant or session roles.",
                )
        if isinstance(session_slot, str):
            captures_by_session[session_slot] += 1

        article_ids = capture.get("article_slot_ids")
        if (
            not isinstance(article_ids, Sequence)
            or isinstance(article_ids, (str, bytes))
            or not article_ids
            or any(not _valid_slot(value, "article") for value in article_ids)
            or len(set(article_ids)) != len(article_ids)
        ):
            _add(
                issues, "INVALID_ARTICLE_ASSIGNMENT", "Article assignments are invalid."
            )
        else:
            for article_id in article_ids:
                article = article_index.get(str(article_id))
                if article is None:
                    _add(
                        issues,
                        "BROKEN_REFERENCE",
                        "A capture run has an unknown article.",
                    )
                elif article.get("analysis_role") != role:
                    _add(
                        issues,
                        "ARTICLE_FAMILY_CROSSES_ROLES",
                        "A capture run uses an article from another analysis role.",
                    )

        conditions = capture.get("condition_ids")
        if (
            not isinstance(conditions, Sequence)
            or isinstance(conditions, (str, bytes))
            or not conditions
            or any(not _bounded_label(value) for value in conditions)
            or len(set(conditions)) != len(conditions)
        ):
            _add(
                issues, "INVALID_CAPTURE_CONDITIONS", "Capture conditions are invalid."
            )
        elif not required_conditions.issubset(set(conditions)):
            _add(
                issues,
                "MISSING_REQUIRED_CONDITION",
                "A capture run omits one or more frozen conditions.",
            )

        sources = _records(capture.get("source_slots"))
        if not sources:
            _add(issues, "INVALID_SOURCE_LIST", "A capture run needs a sensor source.")
            sources = []
        source_roles: set[str] = set()
        for source in sources:
            _check_exact_keys(source, SOURCE_KEYS, issues, "INVALID_SOURCE_SCHEMA")
            source_slot = source.get("slot_id")
            if not _valid_slot(source_slot, "source"):
                _add(issues, "INVALID_SLOT_IDENTIFIER", "A source slot is invalid.")
            elif str(source_slot) in source_slot_ids:
                duplicate_source_slots += 1
            else:
                source_slot_ids.add(str(source_slot))
            device = device_index.get(str(source.get("device_slot_id")))
            if device is None:
                _add(issues, "BROKEN_REFERENCE", "A source has no device slot.")
            elif device.get("analysis_role") not in (role, "shared"):
                _add(
                    issues,
                    "CROSS_ROLE_REFERENCE",
                    "A sensor source uses a device from another analysis role.",
                )
            source_role = source.get("source_role")
            if source_role not in SOURCE_ROLES:
                _add(issues, "INVALID_SOURCE_METADATA", "A source role is invalid.")
            elif str(source_role) in source_roles:
                _add(
                    issues,
                    "DUPLICATE_SOURCE_ROLE",
                    "A capture run repeats the same sensor-source role.",
                )
            else:
                source_roles.add(str(source_role))
            if not isinstance(source.get("required"), bool):
                _add(
                    issues,
                    "INVALID_SOURCE_METADATA",
                    "Source required must be boolean.",
                )

        synchronization = capture.get("synchronization")
        if len(sources) >= 2:
            multiview_runs += 1
            sync = _mapping(synchronization)
            if not _check_exact_keys(
                sync,
                SYNCHRONIZATION_KEYS,
                issues,
                "MULTIVIEW_SYNC_MISSING",
            ):
                continue
            assert sync is not None
            max_offset = _finite_number(sync.get("max_offset_ms"), minimum=0.000001)
            calibration = sync.get("relative_camera_calibration_slot_id")
            calibration_sha256 = sync.get("relative_camera_calibration_sha256")
            if (
                sync.get("required") is not True
                or max_offset is None
                or max_offset > 1000.0
                or sync.get("clock_strategy") not in CLOCK_STRATEGIES
                or not _valid_slot(calibration, "calibration")
                or not _valid_sha256(calibration_sha256, nullable=True)
            ):
                _add(
                    issues,
                    "MULTIVIEW_SYNC_INVALID",
                    "Multi-view synchronization or relative-camera calibration is invalid.",
                )
            else:
                synchronized_multiview_runs += 1
        elif synchronization is not None:
            _add(
                issues,
                "SINGLE_VIEW_HAS_SYNC_CONTRACT",
                "A single-view run must not carry a multi-view synchronization object.",
            )
    if duplicate_source_slots:
        _add(
            issues,
            "DUPLICATE_SLOT_IDENTIFIER",
            f"{duplicate_source_slots} duplicate source slot identifier(s) were found.",
        )
    if min_sessions:
        empty_sessions = sum(
            captures_by_session[slot_id] == 0 for slot_id in session_index
        )
        if empty_sessions:
            _add(
                issues,
                "SESSION_WITHOUT_CAPTURE_RUN",
                f"{empty_sessions} session slot(s) have no planned capture run.",
            )
    if multiview_claim and multiview_runs != len(capture_index):
        _add(
            issues,
            "MULTIVIEW_CLAIM_HAS_SINGLE_VIEW_RUN",
            "Every capture run must be paired for a multi-view incremental claim.",
        )

    if target in {"collection", "evidence"} and status != "frozen_before_collection":
        _add(
            issues,
            "PLAN_NOT_FROZEN",
            "Collection or evidence auditing requires a pre-collection frozen plan.",
        )
    if target in {"collection", "evidence"} and any(
        not _valid_sha256(article.get("content_sha256"))
        or not _bounded_label(article.get("authorization_id"))
        for article in articles
    ):
        _add(
            issues,
            "ARTICLE_ASSET_NOT_FROZEN",
            "Every article must be content-hashed and authorized before collection.",
        )
    if target == "template" and status == "template_only":
        _add(
            issues,
            "TEMPLATE_NOT_AUTHORIZED",
            "The structure is a template and is not authorization to collect data.",
            severity="warning",
        )

    bound_participant_role_counts = Counter(
        str(record.get("analysis_role"))
        for record in participants
        if record.get("binding_status") == "bound"
    )
    bound_session_counts: Counter[str] = Counter(
        str(record.get("participant_slot_id"))
        for record in sessions
        if record.get("binding_status") == "bound"
    )
    if target == "evidence":
        for role in ANALYSIS_ROLES:
            minimum = role_minimums.get(role) if role_minimums else None
            if (
                isinstance(minimum, int)
                and bound_participant_role_counts[role] < minimum
            ):
                _add(
                    issues,
                    "INSUFFICIENT_BOUND_PARTICIPANTS",
                    "A role has fewer bound, non-withdrawn participants than required.",
                )
        bound_without_sessions = sum(
            participant.get("binding_status") == "bound"
            and bound_session_counts[str(participant.get("slot_id"))] < min_sessions
            for participant in participants
        )
        if bound_without_sessions:
            _add(
                issues,
                "INSUFFICIENT_BOUND_SESSIONS",
                "One or more bound participants lack repeated bound sessions.",
            )
        inconsistent_withdrawals = 0
        orphan_session_bindings = 0
        participant_status = {
            str(record.get("slot_id")): record.get("binding_status")
            for record in participants
        }
        for session in sessions:
            owner_status = participant_status.get(
                str(session.get("participant_slot_id"))
            )
            if (
                owner_status == "withdrawn"
                and session.get("binding_status") != "withdrawn"
            ):
                inconsistent_withdrawals += 1
            if owner_status != "bound" and session.get("binding_status") == "bound":
                orphan_session_bindings += 1
        if inconsistent_withdrawals:
            _add(
                issues,
                "WITHDRAWAL_NOT_PROPAGATED",
                "A withdrawn participant still has a non-withdrawn session binding.",
            )
        if orphan_session_bindings:
            _add(
                issues,
                "ORPHAN_SESSION_BINDING",
                "A bound session belongs to a participant who is not bound.",
            )
        active_device_ids: set[str] = set()
        active_article_ids: set[str] = set()
        active_multiview_without_calibration = 0
        bound_sessions = {
            str(record.get("slot_id"))
            for record in sessions
            if record.get("binding_status") == "bound"
        }
        for capture in capture_runs:
            if str(capture.get("session_slot_id")) not in bound_sessions:
                continue
            sources = _records(capture.get("source_slots")) or []
            for source in sources:
                active_device_ids.add(str(source.get("device_slot_id")))
            if len(sources) >= 2:
                synchronization = _mapping(capture.get("synchronization")) or {}
                if not _valid_sha256(
                    synchronization.get("relative_camera_calibration_sha256")
                ):
                    active_multiview_without_calibration += 1
            article_ids = capture.get("article_slot_ids")
            if isinstance(article_ids, Sequence) and not isinstance(
                article_ids, (str, bytes)
            ):
                active_article_ids.update(map(str, article_ids))
        if any(
            device_index.get(slot_id, {}).get("binding_status") != "bound"
            for slot_id in active_device_ids
        ):
            _add(
                issues,
                "UNBOUND_ACTIVE_DEVICE",
                "A bound session uses a device without a bound instance digest.",
            )
        if any(
            not _valid_sha256(article_index.get(slot_id, {}).get("content_sha256"))
            or not _bounded_label(
                article_index.get(slot_id, {}).get("authorization_id")
            )
            for slot_id in active_article_ids
        ):
            _add(
                issues,
                "UNBOUND_ACTIVE_ARTICLE",
                "A bound session uses an unhashed or unauthorized article slot.",
            )
        if active_multiview_without_calibration:
            _add(
                issues,
                "UNBOUND_MULTIVIEW_CALIBRATION",
                "A bound multi-view session lacks a hashed relative-camera calibration.",
            )

    error_count = sum(issue.severity == "error" for issue in issues)
    ready_status = {
        "template": "template_valid",
        "collection": "collection_ready",
        "evidence": "evidence_ready",
    }[target]
    summary = {
        "participant_slots": len(participants),
        "session_slots": len(sessions),
        "device_slots": len(devices),
        "article_slots": len(articles),
        "article_family_slots": len(article_family_roles),
        "capture_run_slots": len(capture_runs),
        "source_slots": len(source_slot_ids),
        "multi_view_capture_runs": multiview_runs,
        "synchronized_multi_view_capture_runs": synchronized_multiview_runs,
        "shared_device_slots": sum(
            record.get("analysis_role") == "shared" for record in devices
        ),
        "bound_participants": sum(
            record.get("binding_status") == "bound" for record in participants
        ),
        "withdrawn_participants": sum(
            record.get("binding_status") == "withdrawn" for record in participants
        ),
        "unbound_participants": sum(
            record.get("binding_status") == "unbound" for record in participants
        ),
        "bound_sessions": sum(
            record.get("binding_status") == "bound" for record in sessions
        ),
        "planned_participants_by_role": {
            role: participant_role_counts[role] for role in ANALYSIS_ROLES
        },
        "bound_participants_by_role": {
            role: bound_participant_role_counts[role] for role in ANALYSIS_ROLES
        },
    }
    try:
        digest = canonical_plan_sha256(top)
    except (TypeError, ValueError):
        digest = None
        _add(
            issues,
            "NON_CANONICAL_JSON_VALUE",
            "The plan contains a value that cannot be canonically hashed.",
        )
        error_count += 1
    return {
        "schema_version": 1,
        "target": target,
        "status": ready_status if error_count == 0 else "not_ready",
        "plan_sha256": digest,
        "summary": summary,
        "issues": [asdict(issue) for issue in issues],
        "warning": (
            "Engineering readiness is not ethics approval, recruitment authorization, "
            "or evidence of model effectiveness."
        ),
    }
