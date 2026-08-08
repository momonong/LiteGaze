"""Audit the frozen Reader Assessment v3 design without GPU or network."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "docs" / "reader_assessment" / "reader_assessment_validity_v3.json"
)

LATENT_CLAIMS = (
    "lexical_knowledge",
    "english_reading_proficiency",
    "cefr",
    "working_memory",
    "general_cognitive_ability",
    "attention",
    "fatigue",
    "typography_preference",
)
PROHIBITED_TARGETS = {
    "question_answer_correctness",
    "gaze_derived_duration",
    "text_model_score",
    "calibration_target_coordinates",
    "synthetic_fusion_label",
}
REQUIRED_HOLDOUT_AXES = {
    "participant_id",
    "passage_family_id",
    "item_family_id",
    "capture_session_id",
    "device_class",
}
REQUIRED_MODEL_IDS = {
    "b0_prevalence",
    "b1_lexical_controls",
    "b2_frozen_text_artifact",
    "b3_person_only",
    "b4_gaze_only",
    "f1_text_person",
    "f2_quality_aware_fusion",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("protocol root must be an object")
    return payload


def audit_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return explicit safeguards and blockers for a v3 protocol payload."""

    claims = protocol.get("claim_registry", {})
    stimulus = protocol.get("stimulus_design", {})
    split = protocol.get("split_policy", {})
    outcomes = protocol.get("outcomes", {})
    compute = protocol.get("compute_policy", {})
    modules = protocol.get("modules", [])
    models = protocol.get("model_ladder", [])
    decision = protocol.get("decision_rule", {})
    calibration = protocol.get("calibration_and_validation", {})
    tracks = protocol.get("study_tracks", {})
    legacy = protocol.get("legacy_data_policy", {})

    module_by_id = {
        str(item.get("id")): item for item in modules if isinstance(item, dict)
    }
    model_by_id = {
        str(item.get("id")): item for item in models if isinstance(item, dict)
    }
    fixed_typography = stimulus.get("fixed_typography", {})
    checks = {
        "design_is_not_live_collection": (
            protocol.get("status") == "design_frozen_not_collection_ready"
            and protocol.get("live_collection_enabled") is False
            and protocol.get("replaces_live_protocol") is False
        ),
        "latent_claims_abstain": all(
            isinstance(claims.get(name), dict)
            and claims[name].get("status") == "not_estimated"
            for name in LATENT_CLAIMS
        ),
        "primary_target_is_independent_word_audit": (
            outcomes.get("primary", {}).get("id")
            == "post_reading_word_review_need"
            and outcomes.get("primary", {}).get("collection_timing")
            == "after_reading_before_feedback"
            and module_by_id.get("post_reading_word_audit", {}).get(
                "required_for_primary"
            )
            is True
        ),
        "qa_and_circular_targets_prohibited": PROHIBITED_TARGETS.issubset(
            set(outcomes.get("prohibited_training_or_selection_targets", []))
        ),
        "adaptive_routing_disabled_before_calibration": (
            stimulus.get("adaptive_routing_enabled") is False
            and calibration.get("three_parameter_logistic_allowed_initially") is False
        ),
        "fixed_typography_during_measurement": fixed_typography
        == {"font_size_px": 16, "line_width_px": 650, "line_height": 1.7},
        "participant_and_content_holdouts_frozen": (
            split.get("freeze_before_outcome_access") is True
            and REQUIRED_HOLDOUT_AXES.issubset(set(split.get("holdout_axes", [])))
            and split.get("same_participant_same_partition") is True
            and split.get("confirmation_access_once") is True
        ),
        "all_model_ablation_rungs_present": REQUIRED_MODEL_IDS.issubset(model_by_id),
        "text_artifact_remains_frozen": (
            model_by_id.get("b2_frozen_text_artifact", {}).get(
                "fine_tuning_allowed"
            )
            is False
        ),
        "fusion_can_abstain_from_gaze": (
            model_by_id.get("f2_quality_aware_fusion", {}).get(
                "must_support_gaze_abstention"
            )
            is True
        ),
        "cognition_is_separate_and_non_composite": (
            module_by_id.get("cognitive_add_on", {}).get("session")
            == "separate_optional"
            and "no composite score"
            in str(module_by_id.get("cognitive_add_on", {}).get("implementation"))
        ),
        "measurement_and_fusion_tracks_are_independent": (
            tracks.get("reading_measurement_calibration", {}).get(
                "participant_overlap_with_fusion"
            )
            is False
            and tracks.get("reading_measurement_calibration", {}).get(
                "webcam_gaze_required"
            )
            is False
            and tracks.get("personalized_word_fusion", {}).get(
                "participant_overlap_with_measurement_calibration"
            )
            is False
        ),
        "live_v2_bank_is_not_reused_for_measurement": (
            tracks.get("reading_measurement_calibration", {}).get(
                "v2_live_bank_role"
            )
            == "software_dry_run_only"
            and tracks.get("reading_measurement_calibration", {})
            .get("new_bank_target", {})
            .get("passage_family_count")
            == 36
            and tracks.get("reading_measurement_calibration", {})
            .get("new_bank_target", {})
            .get("passage_split_counts", {})
            .get("confirmation")
            == 9
        ),
        "fusion_confirmation_has_content_clusters": (
            stimulus.get("passage_family_count") == 48
            and stimulus.get("passage_split_counts", {}).get("confirmation") == 12
            and tracks.get("personalized_word_fusion", {})
            .get("stimulus_pool_target", {})
            .get("passage_family_count")
            == 48
        ),
        "preprotocol_data_cannot_enter_formal_evidence": (
            legacy.get("role") == "workflow_quality_and_exploratory_evidence_only"
            and legacy.get("eligible_for_item_or_scale_calibration") is False
            and legacy.get("eligible_for_model_or_threshold_selection") is False
            and legacy.get("eligible_for_validation_or_confirmation") is False
            and legacy.get("must_remain_separately_provenanced") is True
        ),
        "external_anchor_authorization_required": (
            "authorization"
            in str(
                module_by_id.get("external_lexical_anchor", {}).get(
                    "implementation", ""
                )
            )
        ),
        "confirmation_cannot_retune_or_use_qa": (
            decision.get("confirmation_threshold_retuning_allowed") is False
            and decision.get("question_answer_outcomes_can_change_decision") is False
        ),
        "current_design_audit_is_cpu_only": (
            compute.get("current_phase_device") == "cpu"
            and compute.get("gpu_allowed_current_phase") is False
            and compute.get("network_allowed_for_design_audit") is False
        ),
        "unresolved_governance_is_explicit": len(
            protocol.get("readiness_blockers", [])
        )
        >= 5,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_version": protocol.get("protocol_version"),
        "design_contract_valid": not failed,
        "collection_ready": not failed
        and protocol.get("live_collection_enabled") is True
        and not protocol.get("readiness_blockers"),
        "checks": checks,
        "failed_checks": failed,
        "readiness_blockers": list(protocol.get("readiness_blockers", [])),
        "runtime_guards": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "network_required": False,
            "gpu_required": False,
        },
        "interpretation": (
            "A valid design contract freezes safeguards; it does not establish "
            "measurement validity or authorize participant collection."
        ),
    }


def audit_file(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    return audit_protocol(_load(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = audit_file(args.protocol.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        output = args.json_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, output)
    return 0 if result["design_contract_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
