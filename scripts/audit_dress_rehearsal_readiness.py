"""Audit participant-facing dress-rehearsal readiness without participant data.

The explicit ``materials`` target verifies that the additive invitation,
moderator runbook, and debrief templates carry the frozen two-visit boundary.
The default ``external-rehearsal`` target fails closed until the canonical
pilot activation, final participant materials, independent passage review, and
manual browser QA are all evidenced. This script does not use the network,
camera, GPU, study store, or participant outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.participant_study.protocol import activation_status, public_protocol  # noqa: E402


INVITATION = "docs/participant_study/PARTICIPANT_INVITATION_V1.md"
RUNBOOK = "docs/participant_study/DRESS_REHEARSAL_RUNBOOK_V1.md"
DEBRIEF = "docs/participant_study/PARTICIPANT_DEBRIEF_V1.md"
CONSENT_DRAFT = "docs/participant_study/CONSENT_DRAFT.md"
FINAL_CONSENT = "docs/participant_study/CONSENT_FINAL_V1.md"
COMPENSATION_POLICY = "docs/participant_study/COMPENSATION_POLICY_FINAL_V1.md"
GENERAL_COLLECTION_DOC = "docs/participant_study/GENERALIZABLE_COLLECTION_V1.md"
CANONICAL_PROTOCOL = "core/participant_study/protocol_v1.json"
GENERAL_COLLECTION_PROTOCOL = "core/participant_study/general_collection_v1.json"
GENERAL_COLLECTION_BANK = "core/participant_study/general_collection_bank_v1.json"
REHEARSAL_PROTOCOL = (
    "docs/CHI/protocols/2026-08-17-dress-rehearsal-process-measurement-v1.json"
)
FROZEN_REHEARSAL_PROTOCOL_SHA256 = (
    "6f6264a58e820e47c414f1e86fd499dccb4930a587258cabc694ba80e7c610bd"
)
STUDY_TEMPLATE = "web/templates/participant_study.html"
COLLECTION_TEMPLATE = "web/templates/participant_collection.html"
COLLECTION_CLIENT = "web/static/participant_collection.js"
VISUAL_QA_EVIDENCE = (
    "docs/participant_study/evidence/dress_rehearsal_visual_qa_v1.json"
)
PASSAGE_REVIEW_EVIDENCE = (
    "docs/participant_study/evidence/dress_rehearsal_passage_review_v1.json"
)
MATERIAL_APPROVAL_EVIDENCE = (
    "docs/participant_study/evidence/dress_rehearsal_material_approval_v1.json"
)

READ_PATHS = (
    INVITATION,
    RUNBOOK,
    DEBRIEF,
    CONSENT_DRAFT,
    GENERAL_COLLECTION_DOC,
    CANONICAL_PROTOCOL,
    GENERAL_COLLECTION_PROTOCOL,
    GENERAL_COLLECTION_BANK,
    REHEARSAL_PROTOCOL,
    STUDY_TEMPLATE,
    COLLECTION_TEMPLATE,
    COLLECTION_CLIENT,
)

PLACEHOLDER_RE = re.compile(
    r"\[(?:使用前填入[^\]]*|研究聯絡|獨立權益聯絡|reference|核定政策摘要|"
    r"服務商名稱|N [^\]]*|加密位置|授權角色|研究者姓名／電子郵件|"
    r"獨立權益聯絡方式|approved [^\]]*|姓名|電子郵件|獨立聯絡|名稱／核准依據)\]"
    r"|<[^>\r\n]{1,100}>|\b(?:TBD|TODO)\b"
)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _read_text(root: Path, relative_path: str) -> str:
    return (root / relative_path).read_text(encoding="utf-8")


def _has_all(text: str, markers: Sequence[str]) -> bool:
    return all(marker in text for marker in markers)


def _find_placeholders(*texts: str) -> list[str]:
    return sorted({match.group(0) for text in texts for match in PLACEHOLDER_RE.finditer(text)})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_resolved_value(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value == value.strip()
        and not PLACEHOLDER_RE.search(value)
        and "使用前填入" not in value
    )


def _rehearsal_contract_status(protocol: Mapping[str, object], sha256: str) -> dict:
    """Validate the exact outcome-blind 5--8-person process-only boundary."""

    errors: list[str] = []
    if sha256 != FROZEN_REHEARSAL_PROTOCOL_SHA256:
        errors.append("rehearsal_protocol_sha256_not_frozen_revision")
    participant_scope = protocol.get("participant_scope")
    if not isinstance(participant_scope, dict):
        participant_scope = {}
        errors.append("rehearsal_participant_scope_missing")
    expected_fields = {
        "minimum_started_participants": (5, "rehearsal_minimum_started_participants_not_five"),
        "maximum_started_participants": (8, "rehearsal_maximum_started_participants_not_eight"),
        "extension_to_chase_a_gate_allowed": (
            False,
            "rehearsal_extension_to_chase_gate_not_forbidden",
        ),
        "analysis_role": (
            "process_and_measurement_only",
            "rehearsal_analysis_role_not_process_only",
        ),
        "rehearsal_rows_may_enter_formal_confirmation": (
            False,
            "rehearsal_rows_not_excluded_from_confirmation",
        ),
    }
    for field, (expected, error) in expected_fields.items():
        actual = participant_scope.get(field)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(error)

    slot_plan = participant_scope.get("outcome_blind_slot_plan")
    if not isinstance(slot_plan, dict):
        slot_plan = {}
        errors.append("rehearsal_outcome_blind_slot_plan_missing")
    expected_slot_ids = [f"DR{index:02d}" for index in range(1, 9)]
    if slot_plan.get("slot_ids") != expected_slot_ids:
        errors.append("rehearsal_slot_ids_not_exactly_DR01_through_DR08")
    if slot_plan.get("maximum_activated_slots") != 8:
        errors.append("rehearsal_maximum_activated_slots_not_eight")
    if slot_plan.get("pseudonymous_slots_only") is not True:
        errors.append("rehearsal_pseudonymous_slot_boundary_not_confirmed")
    if slot_plan.get("frozen_before_target_outcomes") is not True:
        errors.append("rehearsal_slot_plan_not_frozen_before_outcomes")
    if slot_plan.get("replacement_or_extension_based_on_outcomes_allowed") is not False:
        errors.append("rehearsal_outcome_based_slot_replacement_or_extension_not_forbidden")

    return {
        "valid": not errors,
        "path": REHEARSAL_PROTOCOL,
        "sha256": sha256,
        "frozen_sha256": FROZEN_REHEARSAL_PROTOCOL_SHA256,
        "participant_count_boundary": {
            "minimum": participant_scope.get("minimum_started_participants"),
            "maximum": participant_scope.get("maximum_started_participants"),
        },
        "slot_ids": slot_plan.get("slot_ids"),
        "extension_to_chase_a_gate_allowed": participant_scope.get(
            "extension_to_chase_a_gate_allowed"
        ),
        "analysis_role": participant_scope.get("analysis_role"),
        "rehearsal_rows_may_enter_formal_confirmation": participant_scope.get(
            "rehearsal_rows_may_enter_formal_confirmation"
        ),
        "slot_plan_boundary": {
            "maximum_activated_slots": slot_plan.get("maximum_activated_slots"),
            "pseudonymous_slots_only": slot_plan.get("pseudonymous_slots_only"),
            "frozen_before_target_outcomes": slot_plan.get(
                "frozen_before_target_outcomes"
            ),
            "replacement_or_extension_based_on_outcomes_allowed": slot_plan.get(
                "replacement_or_extension_based_on_outcomes_allowed"
            ),
        },
        "errors": sorted(set(errors)),
    }


def _material_approval_status(
    root: Path,
    environment: Mapping[str, str],
    *,
    canonical_protocol: Mapping[str, object],
    rehearsal_protocol_sha256: str,
    runtime_consent_digest_sha256: str,
) -> dict:
    """Bind approved public materials to their exact runtime and policy sources."""

    evidence_path = root / MATERIAL_APPROVAL_EVIDENCE
    if not evidence_path.is_file():
        return {
            "present": False,
            "valid": False,
            "path": MATERIAL_APPROVAL_EVIDENCE,
            "files_read": [],
            "errors": ["approved_material_binding_evidence_missing"],
        }
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "present": True,
            "valid": False,
            "path": MATERIAL_APPROVAL_EVIDENCE,
            "files_read": [MATERIAL_APPROVAL_EVIDENCE],
            "errors": [f"approved_material_binding_invalid_json:{type(error).__name__}"],
        }
    if not isinstance(evidence, dict):
        return {
            "present": True,
            "valid": False,
            "path": MATERIAL_APPROVAL_EVIDENCE,
            "files_read": [MATERIAL_APPROVAL_EVIDENCE],
            "errors": ["approved_material_binding_not_an_object"],
        }

    errors: list[str] = []
    files_read = [MATERIAL_APPROVAL_EVIDENCE]
    if evidence.get("schema_version") != 1:
        errors.append("approved_material_binding_schema_version_mismatch")
    if evidence.get("status") != "approved_for_external_rehearsal":
        errors.append("approved_material_binding_status_not_approved")
    if evidence.get("canonical_protocol_id") != canonical_protocol.get("protocol_id"):
        errors.append("approved_material_binding_protocol_id_mismatch")
    if evidence.get("canonical_protocol_version") != canonical_protocol.get("protocol_version"):
        errors.append("approved_material_binding_protocol_version_mismatch")
    if evidence.get("rehearsal_protocol_sha256") != rehearsal_protocol_sha256:
        errors.append("approved_material_binding_rehearsal_protocol_sha256_mismatch")
    if evidence.get("runtime_consent_digest_sha256") != runtime_consent_digest_sha256:
        errors.append("approved_material_binding_runtime_consent_digest_mismatch")

    bindings = evidence.get("approved_bindings")
    if not isinstance(bindings, dict):
        bindings = {}
        errors.append("approved_material_bindings_missing")
    exact_bindings = {
        "investigator_email": "LEXIGAZE_INVESTIGATOR_EMAIL",
        "participant_rights_contact": "LEXIGAZE_PARTICIPANT_RIGHTS_CONTACT",
        "ethics_reference": "LEXIGAZE_ETHICS_REFERENCE",
    }
    for field, environment_key in exact_bindings.items():
        approved_value = bindings.get(field)
        if not _is_resolved_value(approved_value):
            errors.append(f"approved_material_{field}_missing")
        elif field == "investigator_email" and EMAIL_RE.fullmatch(approved_value) is None:
            errors.append("approved_material_investigator_email_invalid")
        elif approved_value != environment.get(environment_key):
            errors.append(f"approved_material_{field}_runtime_mismatch")

    compensation = bindings.get("compensation_policy")
    if not isinstance(compensation, dict):
        compensation = {}
        errors.append("approved_compensation_policy_binding_missing")
    compensation_text = compensation.get("participant_facing_text_zh")
    if (
        compensation.get("artifact_path") != COMPENSATION_POLICY
        or not _is_sha256(compensation.get("artifact_sha256"))
        or not _is_resolved_value(compensation_text)
    ):
        errors.append("approved_compensation_policy_binding_invalid")
    compensation_path = root / COMPENSATION_POLICY
    if not compensation_path.is_file():
        errors.append("approved_compensation_policy_artifact_missing")
    else:
        files_read.append(COMPENSATION_POLICY)
        compensation_body = compensation_path.read_text(encoding="utf-8")
        if _sha256_file(compensation_path) != compensation.get("artifact_sha256"):
            errors.append("approved_compensation_policy_sha256_mismatch")
        if isinstance(compensation_text, str) and compensation_text not in compensation_body:
            errors.append("approved_compensation_policy_text_mismatch")

    material_contract = {
        "invitation": INVITATION,
        "debrief": DEBRIEF,
        "consent": FINAL_CONSENT,
    }
    materials = evidence.get("materials")
    if not isinstance(materials, dict):
        materials = {}
        errors.append("approved_material_hash_manifest_missing")
    for label, relative_path in material_contract.items():
        entry = materials.get(label)
        if not isinstance(entry, dict):
            entry = {}
            errors.append(f"approved_material_{label}_manifest_missing")
        if entry.get("path") != relative_path:
            errors.append(f"approved_material_{label}_path_mismatch")
        if not _is_sha256(entry.get("sha256")):
            errors.append(f"approved_material_{label}_sha256_invalid")
        material_path = root / relative_path
        if not material_path.is_file():
            errors.append(f"approved_material_{label}_file_missing")
            continue
        files_read.append(relative_path)
        material_text = material_path.read_text(encoding="utf-8")
        if _sha256_file(material_path) != entry.get("sha256"):
            errors.append(f"approved_material_{label}_sha256_mismatch")
        for field in exact_bindings:
            approved_value = bindings.get(field)
            if isinstance(approved_value, str) and approved_value not in material_text:
                errors.append(f"approved_material_{label}_{field}_mismatch")
        if isinstance(compensation_text, str) and compensation_text not in material_text:
            errors.append(f"approved_material_{label}_compensation_text_mismatch")
        if _find_placeholders(material_text):
            errors.append(f"approved_material_{label}_has_unresolved_placeholders")
        if label == "consent" and not _has_all(
            material_text,
            ("兩次 Visit", "50 分鐘", "18–72 小時", "6 篇", "8 個單字"),
        ):
            errors.append("approved_material_consent_flow_mismatch")

    return {
        "present": True,
        "valid": not errors,
        "path": MATERIAL_APPROVAL_EVIDENCE,
        "files_read": sorted(set(files_read)),
        "errors": sorted(set(errors)),
    }


def _visual_qa_status(root: Path, protocol: Mapping[str, object]) -> dict:
    path = root / VISUAL_QA_EVIDENCE
    if not path.is_file():
        return {
            "present": False,
            "valid": False,
            "path": VISUAL_QA_EVIDENCE,
            "errors": ["manual_browser_visual_qa_evidence_missing"],
        }
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "present": True,
            "valid": False,
            "path": VISUAL_QA_EVIDENCE,
            "errors": [f"visual_qa_evidence_invalid_json:{type(error).__name__}"],
        }
    required_checks = (
        "consent_readable",
        "keyboard_order_checked",
        "camera_preview_checked",
        "calibration_stop_path_checked",
        "error_messages_checked",
        "withdrawal_path_checked",
        "completion_and_visit2_debrief_checked",
        "minimum_viewport_checked",
    )
    checks = evidence.get("checks") if isinstance(evidence, dict) else None
    errors = []
    if evidence.get("status") != "passed":
        errors.append("visual_qa_status_not_passed")
    if evidence.get("protocol_version") != protocol.get("protocol_version"):
        errors.append("visual_qa_protocol_version_mismatch")
    if not evidence.get("browser_family"):
        errors.append("visual_qa_browser_family_missing")
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in required_checks):
        errors.append("visual_qa_required_checks_incomplete")
    return {
        "present": True,
        "valid": not errors,
        "path": VISUAL_QA_EVIDENCE,
        "errors": errors,
    }


def _passage_review_status(root: Path, bank_sha256: str) -> dict:
    path = root / PASSAGE_REVIEW_EVIDENCE
    if not path.is_file():
        return {
            "present": False,
            "valid": False,
            "path": PASSAGE_REVIEW_EVIDENCE,
            "bank_sha256": bank_sha256,
            "errors": ["independent_passage_review_evidence_missing"],
        }
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "present": True,
            "valid": False,
            "path": PASSAGE_REVIEW_EVIDENCE,
            "bank_sha256": bank_sha256,
            "errors": [f"passage_review_evidence_invalid_json:{type(error).__name__}"],
        }
    required_checks = (
        "factual_accuracy",
        "naturalness",
        "accessibility",
        "sensitive_content",
        "difficulty_and_probe_clarity",
    )
    checks = evidence.get("checks") if isinstance(evidence, dict) else None
    errors = []
    if evidence.get("status") != "passed":
        errors.append("passage_review_status_not_passed")
    if evidence.get("bank_sha256") != bank_sha256:
        errors.append("passage_review_bank_sha256_mismatch")
    if evidence.get("reviewers_independent") is not True:
        errors.append("passage_review_independence_not_confirmed")
    reviewer_count = evidence.get("reviewer_count")
    if not isinstance(reviewer_count, int) or reviewer_count < 2:
        errors.append("passage_review_requires_two_reviewers")
    if not isinstance(checks, dict) or any(
        checks.get(key) is not True for key in required_checks
    ):
        errors.append("passage_review_required_checks_incomplete")
    return {
        "present": True,
        "valid": not errors,
        "path": PASSAGE_REVIEW_EVIDENCE,
        "bank_sha256": bank_sha256,
        "errors": errors,
    }


def audit(
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] | None = None,
) -> dict:
    root = root.resolve()
    env = os.environ if environment is None else environment
    missing_files = [name for name in READ_PATHS if not (root / name).is_file()]
    if missing_files:
        return {
            "schema_version": 1,
            "audit_id": "lexigaze-dress-rehearsal-readiness-v1",
            "documentation_ready": False,
            "external_rehearsal_ready": False,
            "missing_files": missing_files,
            "files_read": [],
            "blockers": ["required_material_or_contract_file_missing"],
            "warning": "No external participant may be invited from this result.",
        }

    texts = {name: _read_text(root, name) for name in READ_PATHS}
    protocol = json.loads(texts[CANONICAL_PROTOCOL])
    general = json.loads(texts[GENERAL_COLLECTION_PROTOCOL])
    rehearsal_protocol = json.loads(texts[REHEARSAL_PROTOCOL])
    if not all(isinstance(item, dict) for item in (protocol, general, rehearsal_protocol)):
        raise ValueError("canonical protocols must be JSON objects")

    invitation = texts[INVITATION]
    runbook = texts[RUNBOOK]
    debrief = texts[DEBRIEF]
    consent = texts[CONSENT_DRAFT]
    general_doc = texts[GENERAL_COLLECTION_DOC]
    study_html = texts[STUDY_TEMPLATE]
    collection_html = texts[COLLECTION_TEMPLATE]
    collection_js = texts[COLLECTION_CLIENT]
    rehearsal_protocol_sha256 = _sha256_file(root / REHEARSAL_PROTOCOL)
    rehearsal_contract = _rehearsal_contract_status(
        rehearsal_protocol,
        rehearsal_protocol_sha256,
    )

    sessions = dict(general.get("sessions") or {})
    canonical_flow = {
        "estimated_minutes_per_visit": protocol.get("estimated_duration_minutes"),
        "required_visits": sessions.get("required_per_participant"),
        "minimum_interval_hours": sessions.get("minimum_interval_hours"),
        "maximum_interval_hours": sessions.get("maximum_interval_hours"),
        "passages_per_visit": sessions.get("passages_per_session"),
        "word_probes_per_passage": sessions.get("word_probes_per_passage"),
        "minimum_age": dict(protocol.get("population") or {}).get("minimum_age"),
    }
    contract_values_correct = canonical_flow == {
        "estimated_minutes_per_visit": 50,
        "required_visits": 2,
        "minimum_interval_hours": 18,
        "maximum_interval_hours": 72,
        "passages_per_visit": 6,
        "word_probes_per_passage": 8,
        "minimum_age": 18,
    }

    automated_checks = {
        "required_files_present": True,
        "rehearsal_contract_is_process_only_and_bounded_5_to_8": rehearsal_contract[
            "valid"
        ],
        "canonical_two_visit_contract_is_50min_18to72h_6x8": contract_values_correct,
        "invitation_discloses_two_visits_duration_interval_and_device": _has_all(
            invitation,
            ("兩次 Visit", "50 分鐘", "18–72 小時", "相同裝置類別", "瀏覽器 family"),
        ),
        "invitation_discloses_camera_break_withdrawal_and_compensation": _has_all(
            invitation,
            ("webcam", "休息", "退出", "補償", "不會勸你繼續"),
        ),
        "invitation_excludes_ability_and_effect_claims": _has_all(
            invitation,
            ("非效果性", "CEFR", "認知", "不用來證明模型成效"),
        ),
        "runbook_covers_camera_discomfort_retry_break_and_withdrawal": _has_all(
            runbook,
            (
                "相機舒適度 script",
                "mid-passage pause",
                "最多做一次標準 retry",
                "安全點",
                "只停止與撤回",
            ),
        ),
        "runbook_forbids_answer_coaching_and_unsupported_claims": _has_all(
            runbook,
            ("不看 participant 的 word-review", "不得用 5–8 人資料估計效果", "CEFR"),
        ),
        "debrief_separates_gaze_quality_from_participant_performance": _has_all(
            debrief,
            ("gaze quality", "不代表你沒有專心", "不代表閱讀表現不好"),
        ),
        "debrief_covers_visit2_discomfort_and_withdrawal": _has_all(
            debrief,
            ("18–72 小時", "相同裝置類別", "不舒服", "撤回碼"),
        ),
        "study_home_has_decline_and_withdrawal_controls": _has_all(
            study_html,
            ('id="declineBtn"', 'id="withdrawBtn"', 'id="downloadReceiptBtn"'),
        ),
        "collection_has_camera_preview_and_word_review_controls": _has_all(
            collection_html,
            ('id="cameraPreview"', 'id="startReadingBtn"', 'id="reviewForm"'),
        ),
        "general_collection_doc_keeps_rehearsal_development_only": _has_all(
            general_doc,
            (
                "formal participant collection and model",
                "development-only",
                "two independent human reviews",
            ),
        ),
        "participant_data_paths_not_read": all(not name.startswith("data/") for name in READ_PATHS),
    }
    documentation_ready = all(automated_checks.values())

    consent_alignment = {
        "duration_50_minutes": "50 分鐘" in consent,
        "six_passages": "6 篇" in consent,
        "eight_word_reviews": "8 個單字" in consent or "8 個" in consent,
        "two_visits_18_to_72_hours": "18–72 小時" in consent and "Visit 2" in consent,
    }
    participant_placeholders = _find_placeholders(invitation, debrief)
    visual_qa = _visual_qa_status(root, protocol)
    bank_sha256 = hashlib.sha256((root / GENERAL_COLLECTION_BANK).read_bytes()).hexdigest()
    passage_review = _passage_review_status(root, bank_sha256)
    activation = activation_status(env, protocol=protocol)
    runtime_protocol = public_protocol(env, protocol=protocol)
    material_approval = _material_approval_status(
        root,
        env,
        canonical_protocol=protocol,
        rehearsal_protocol_sha256=rehearsal_protocol_sha256,
        runtime_consent_digest_sha256=runtime_protocol["consent_digest_sha256"],
    )
    self_only_video_scope_present = any(
        item.get("self_development_only") is True
        for item in protocol.get("optional_scopes", [])
        if isinstance(item, dict)
    )
    ui_constraints = {
        "mid_passage_pause_control_present": (
            'id="pauseReadingBtn"' in collection_html or "pauseReadingBtn" in collection_js
        ),
        "collection_page_withdraw_control_present": 'id="withdrawBtn"' in collection_html,
        "completion_page_discloses_visit2_window": "18–72 小時" in collection_html,
        "participant_facing_8min_auto_stop_present": bool(
            re.search(r"elapsed\s*>=\s*480_000[\s\S]{0,240}finishReading", collection_js)
        ),
        "camera_stop_control_present_on_setup": 'id="stopCameraBtn"' in collection_html,
        "moderated_mitigation_documented": _has_all(
            runbook,
            ("已知介面限制與標準處置", "mid_passage_stop", "接近 8 分鐘"),
        ),
    }

    blockers: list[str] = []
    if protocol.get("collection_status") != "approved_for_pilot":
        blockers.append("canonical_protocol_not_approved_for_pilot")
    blockers.extend(f"activation:{item}" for item in activation.get("missing_requirements", []))
    if not all(consent_alignment.values()) and not material_approval["valid"]:
        blockers.append("consent_draft_not_aligned_to_current_two_visit_50min_6x8_flow")
    if participant_placeholders:
        blockers.append("participant_invitation_or_debrief_has_unresolved_fields")
    if self_only_video_scope_present:
        blockers.append("formal_external_protocol_still_contains_self_only_video_scope")
    blockers.extend(passage_review["errors"])
    blockers.extend(visual_qa["errors"])
    blockers.extend(material_approval["errors"])
    blockers.extend(rehearsal_contract["errors"])
    if not documentation_ready:
        blockers.append("participant_material_template_contract_failed")

    external_rehearsal_ready = not blockers
    return {
        "schema_version": 1,
        "audit_id": "lexigaze-dress-rehearsal-readiness-v1",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_version": protocol.get("protocol_version"),
        "collection_status": protocol.get("collection_status"),
        "canonical_flow": canonical_flow,
        "documentation_ready": documentation_ready,
        "external_rehearsal_ready": external_rehearsal_ready,
        "current_decision": (
            "GO_external_moderated_rehearsal" if external_rehearsal_ready else "NO_GO_external_participants"
        ),
        "automated_checks": automated_checks,
        "consent_alignment": consent_alignment,
        "participant_material_placeholders": participant_placeholders,
        "approved_material_binding": material_approval,
        "rehearsal_contract": rehearsal_contract,
        "visual_qa": visual_qa,
        "passage_review": passage_review,
        "ui_constraints": ui_constraints,
        "activation_missing_requirements": activation.get("missing_requirements", []),
        "blockers": sorted(set(blockers)),
        "missing_files": [],
        "files_read": sorted(
            set(
                list(READ_PATHS)
                + ([VISUAL_QA_EVIDENCE] if visual_qa["present"] else [])
                + ([PASSAGE_REVIEW_EVIDENCE] if passage_review["present"] else [])
                + material_approval["files_read"]
            )
        ),
        "claim_boundary": {
            "allowed": [
                "participant burden and flow clarity",
                "retry break stop withdrawal and Visit 2 operations",
                "camera discomfort and process incidents",
            ],
            "prohibited": [
                "model effect or promotion",
                "webcam gaze accuracy",
                "English reading cognitive attention fatigue CEFR or clinical ability claims",
            ],
        },
        "warning": (
            "A documentation-ready result is not ethics approval or recruitment authorization. "
            "The external target must remain fail-closed until every blocker is cleared."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("materials", "external-rehearsal"),
        default="external-rehearsal",
        help="Readiness level whose status determines the exit code.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    result = audit()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        output = args.json_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, output)
    ready = (
        result["external_rehearsal_ready"]
        if args.target == "external-rehearsal"
        else result["documentation_ready"]
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
