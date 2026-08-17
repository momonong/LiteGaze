"""Validate the additive CHI dress-rehearsal and practical-effect contracts.

The default worksheet is intentionally unfilled.  It can be structurally
valid while remaining fail-closed for threshold freeze.  This validator reads
only the two machine contracts; it never reads participant or model outcomes,
fits a model, uses a GPU, or authorizes participant contact or confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT
    / "docs"
    / "CHI"
    / "protocols"
    / "2026-08-17-dress-rehearsal-process-measurement-v1.json"
)
DEFAULT_WORKSHEET = (
    ROOT
    / "docs"
    / "CHI"
    / "protocols"
    / "2026-08-17-practical-effect-cost-elicitation-v1.json"
)
LABELS = ("no_review", "unsure", "review_needed")
COMPARISONS = ("F2_minus_F1", "F2_minus_F0")
GROUP_KEYS = (
    "participant_id",
    "passage_family_id",
    "passage_probe_id",
    "capture_session_id",
)
BUDGET_KEYS = (
    "maximum_added_active_seconds_per_visit",
    "maximum_added_compute_ms_per_passage",
    "maximum_added_review_prompts_per_passage",
    "minimum_system_gaze_eligible_coverage",
    "maximum_absolute_calibration_slope_change",
)
OPERATIONAL_GATE_ESTIMANDS = {
    "maximum_added_active_seconds_per_visit": {
        "estimand": (
            "mean paired F2-enabled minus F1-only foreground active seconds "
            "per scheduled visit block with every unfinished block assigned "
            "the frozen 3600-second active-time cap"
        ),
        "reference": (
            "counterbalanced within-participant F1-only block on the same "
            "frozen task bank and device"
        ),
        "denominator": (
            "all scheduled paired visit blocks; any unmatched F2 or F1 block "
            "fails the co-gate and no completed-only deletion is allowed"
        ),
        "uncertainty": (
            "participant-cluster bootstrap upper one-sided 95 percent bound; "
            "fewer than eight independent participant clusters withholds the "
            "bound and fails closed"
        ),
        "pass_rule": (
            "upper_one_sided_95_percent_bound_at_or_below_maximum_added_"
            "active_seconds_per_visit"
        ),
    },
    "maximum_added_compute_ms_per_passage": {
        "estimand": (
            "mean paired F2 minus F1 end-to-end CPU elapsed milliseconds per "
            "passage replay"
        ),
        "reference": (
            "identical frozen input replay on the same declared CPU device "
            "and software revision with F1 gaze disabled"
        ),
        "denominator": (
            "all scheduled passage replays including timeout and error replays "
            "assigned the predeclared timeout cap"
        ),
        "uncertainty": (
            "passage-family-cluster bootstrap upper one-sided 95 percent bound; "
            "fewer than eight independent passage families withholds the bound "
            "and fails closed"
        ),
        "pass_rule": (
            "upper_one_sided_95_percent_bound_at_or_below_maximum_added_"
            "compute_ms_per_passage"
        ),
    },
    "maximum_added_review_prompts_per_passage": {
        "estimand": (
            "mean paired F2-enabled minus F1-only emitted review prompts per "
            "scheduled passage"
        ),
        "reference": (
            "counterbalanced within-participant F1-only passage on the same "
            "frozen task bank"
        ),
        "denominator": (
            "all scheduled paired passages including incomplete passages with "
            "every prompt emitted before the terminal or last event"
        ),
        "uncertainty": (
            "participant-by-passage-family crossed-cluster bootstrap upper "
            "one-sided 95 percent bound; fewer than eight independent participant "
            "clusters or eight independent passage-family clusters withholds the "
            "bound and fails closed"
        ),
        "pass_rule": (
            "upper_one_sided_95_percent_bound_at_or_below_maximum_added_"
            "review_prompts_per_passage"
        ),
    },
    "minimum_system_gaze_eligible_coverage": {
        "estimand": (
            "fraction of all sampled confirmation probe rows that are "
            "predeclared gaze-eligible before labels and selected for F2; "
            "missing, ineligible, and unselected rows remain in the denominator "
            "and use exact F1 fallback"
        ),
        "reference": (
            "no comparator subtraction; the frozen minimum is an absolute "
            "system-hybrid coverage floor"
        ),
        "denominator": (
            "all sampled probe rows on the single frozen joint confirmation "
            "partition with no completed-only or eligible-only deletion"
        ),
        "uncertainty": (
            "participant-by-passage-family crossed-cluster bootstrap lower "
            "one-sided 95 percent bound; fewer than eight independent participant "
            "clusters or eight independent passage-family clusters withholds the "
            "bound and fails closed"
        ),
        "pass_rule": (
            "lower_one_sided_95_percent_bound_at_or_above_minimum_system_gaze_"
            "eligible_coverage"
        ),
    },
    "maximum_absolute_calibration_slope_change": {
        "estimand": (
            "maximum across the three one-vs-rest classes of the absolute F2 "
            "minus F1 logistic recalibration-slope difference on all sampled "
            "confirmation rows"
        ),
        "reference": (
            "F1 probabilities on the identical rows and weights; F2 uses exact "
            "F1 probabilities on missing, ineligible, and unselected rows"
        ),
        "denominator": (
            "all sampled probe-label rows on the single frozen joint "
            "confirmation partition using the same pre-outcome sampling weights "
            "as primary NLL"
        ),
        "uncertainty": (
            "participant-by-passage-family crossed-cluster bootstrap upper "
            "one-sided 95 percent bound of the maximum absolute classwise "
            "difference; fewer than eight independent participant clusters, "
            "fewer than eight independent passage-family clusters, any "
            "non-estimable class, or any withheld bound fails closed"
        ),
        "pass_rule": (
            "upper_one_sided_95_percent_bound_at_or_below_maximum_absolute_"
            "calibration_slope_change"
        ),
    },
}
NLL_REDUCTION_GRID = (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
COST_CONTEXT_IDS = (
    "participant_burden_boundary",
    "prompt_error_boundary",
    "camera_missingness_boundary",
)
COST_CONTEXT_DEFINITIONS = {
    "participant_burden_boundary": (
        "synthetic scenario fixes added active time and compute at their "
        "operational maxima while review-prompt and gaze-coverage co-gates "
        "remain exactly at budget"
    ),
    "prompt_error_boundary": (
        "synthetic scenario fixes added review prompts at their operational "
        "maximum and exposes no participant data while other co-gates remain "
        "exactly at budget"
    ),
    "camera_missingness_boundary": (
        "synthetic scenario fixes system gaze eligibility at its operational "
        "minimum and calibration-slope change at its maximum while every "
        "unselected or ineligible row uses exact F1 fallback"
    ),
}
ELICITATION_INSTRUCTIONS = [
    (
        "Use only synthetic probability examples and product-process costs; "
        "do not inspect target rehearsal or future confirmation cohort labels, "
        "model predictions, losses, or comparison results."
    ),
    (
        "For every role, comparator, cost context, and frozen NLL-grid value, "
        "record worthwhile or not_worthwhile while holding all operational "
        "co-gates at their budget boundaries."
    ),
    (
        "Ratings must be monotonic within each role-comparator-context series; "
        "the threshold is the smallest positive grid value unanimously "
        "worthwhile across every role and cost context, with no interpolation "
        "or manual choice inside a bracket."
    ),
    (
        "Record at least two signoff roles disjoint from every elicitor plus "
        "actual file hashes for the synthetic packet, unopened confirmation "
        "manifest, and outcome-access log."
    ),
]
RATING_VALUES = ("not_worthwhile", "worthwhile")
ATTESTATION_KEYS = (
    "target_cohort_word_review_outcomes_accessed_before_freeze",
    "target_cohort_F0_F1_F2_predictions_or_losses_accessed_before_freeze",
    "confirmation_partition_opened_before_freeze",
    "thresholds_selected_from_observed_effects",
    "rehearsal_or_confirmation_sample_size_changed_to_chase_threshold",
)
AUTHORIZATION_KEYS = (
    "participant_collection_authorized",
    "confirmation_outcome_access_authorized",
    "model_training_or_promotion_authorized",
    "effect_accuracy_ability_or_learning_claim_authorized",
)
PENDING_STATUS = "template_pending_preoutcome_elicitation"
FROZEN_STATUS = "frozen_preoutcome_practical_thresholds"
SHA256_LENGTH = 64
FROZEN_CONTRACT_CANONICAL_SHA256 = (
    "e447ac27ae1dd4475612efaa4f1d049e4ccd0dacadcad09838a513028fccdf0c"
)
FROZEN_CONTRACT_SECTION_SHA256 = {
    "participant_scope": (
        "49114842fdaee4371ff7b17df4d4dd6afca473c1fe64c07df963233ae5cf577c"
    ),
    "timeline_and_blinding": (
        "b5c4167c35a9997f6c4e8f9582f0c3372a137fb4da8e0042451f35b7c31ee729"
    ),
    "claim_boundary": (
        "3dcd2c20253df7871af09e55cc1f30763c304eee3800ae9fefe20c872f328d65"
    ),
    "completion_definitions": (
        "d03c5c4348d3d16833c34858732bf5f7712798ad0541af2370d38b24e974aade"
    ),
    "burden_definitions": (
        "0148d4d37cfc06ac64fbf0b38719a2599c9945d0fb86dc889999a8ee0fef8345"
    ),
    "gaze_eligibility_definition": (
        "7b45375f1b3dc7cd5d6d73e084bd119e4d710914b62f22af3211f13d55734370"
    ),
    "class_prevalence_input": (
        "7721e5bbf6379e9649db850cac9a910db76c4f2632504e11c53258d8b2f9ef84"
    ),
    "icc_planning_inputs": (
        "d7f4ce7cdc33791741f2f7ea63ef84f3143b9c64dc7f8b3483c18a29f216089d"
    ),
    "stop_go_criteria": (
        "8c566b827af57d91299c1dec1ee9d1b982bb0c7a2f44b94fc2f0daa02b4a3493"
    ),
    "aggregate_result_contract": (
        "d84728d7e5507e78af09520ffb3eeb3023a1e776c198d472a5b83cf1a045340c"
    ),
    "compute": (
        "18a63ef83f34dd58500b578f473c2ec7b6fedfe88156da11749dca2fd9938b65"
    ),
    "decision": (
        "736637ab8a16cbceb6e053e670c499546a2423d534ffe4064d3136eeaa396711"
    ),
}


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object and reject other top-level values."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not _is_utc_timestamp(value):
        return None
    return datetime.fromisoformat(str(value)[:-1] + "+00:00")


def _unique_nonempty_strings(value: Any, minimum: int) -> bool:
    if not isinstance(value, list):
        return False
    normalized = [item.strip() for item in value if isinstance(item, str)]
    return (
        len(normalized) == len(value)
        and len(normalized) >= minimum
        and all(normalized)
        and len(set(normalized)) == len(normalized)
    )


def _authorization_failures(value: Any, path: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if (
                isinstance(key, str)
                and key.endswith("authorized")
                and child is not False
            ):
                failures.append(f"authorization_must_remain_false:{child_path}")
            failures.extend(_authorization_failures(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_authorization_failures(child, f"{path}[{index}]"))
    return failures


def validate_rehearsal_contract(contract: Mapping[str, Any]) -> list[str]:
    """Return deterministic structural and claim-boundary failure codes."""

    failures: list[str] = []
    if _canonical_sha256(contract) != FROZEN_CONTRACT_CANONICAL_SHA256:
        failures.append("frozen_whole_contract_changed")
    if contract.get("schema_version") != 1:
        failures.append("contract_schema_version")
    if contract.get("protocol_id") != (
        "lexigaze-chi-dress-rehearsal-process-measurement-v1"
    ):
        failures.append("contract_protocol_id")
    if contract.get("status") != (
        "additive_process_measurement_rehearsal_only_no_formal_"
        "collection_authorization"
    ):
        failures.append("contract_process_measurement_only_status")
    for section, expected_sha256 in FROZEN_CONTRACT_SECTION_SHA256.items():
        if _canonical_sha256(contract.get(section)) != expected_sha256:
            failures.append(f"frozen_contract_section_changed:{section}")

    participant = _mapping(contract.get("participant_scope"))
    if (
        participant.get("minimum_started_participants") != 5
        or participant.get("maximum_started_participants") != 8
        or participant.get("target_started_participants") not in range(5, 9)
        or participant.get("extension_to_chase_a_gate_allowed") is not False
        or participant.get("required_visits_per_participant") != 2
        or participant.get("analysis_role") != "process_and_measurement_only"
        or participant.get("rehearsal_rows_may_enter_formal_confirmation")
        is not False
    ):
        failures.append("frozen_5_to_8_process_only_participant_scope")
    interval = _mapping(participant.get("visit_interval_hours"))
    if interval != {"minimum": 18, "maximum": 72}:
        failures.append("frozen_18_to_72_hour_visit_interval")
    slots = _mapping(participant.get("outcome_blind_slot_plan"))
    if slots != {
        "slot_ids": [f"DR{index:02d}" for index in range(1, 9)],
        "maximum_activated_slots": 8,
        "pseudonymous_slots_only": True,
        "frozen_before_target_outcomes": True,
        "replacement_or_extension_based_on_outcomes_allowed": False,
    }:
        failures.append("frozen_outcome_blind_DR01_to_DR08_slot_plan")

    timeline = contract.get("timeline_and_blinding")
    if not isinstance(timeline, list) or len(timeline) != 5:
        failures.append("preoutcome_timeline_and_blinding")
    else:
        timeline_text = " ".join(str(item) for item in timeline)
        for required in (
            "before opening any target rehearsal or future confirmation cohort word-review outcome",
            "no F0, F1, or F2 predictions, losses, or comparison results",
            "without adding participants beyond eight",
            "separate frozen confirmation cohort",
        ):
            if required not in timeline_text:
                failures.append("preoutcome_timeline_and_blinding")
                break

    claim = _mapping(contract.get("claim_boundary"))
    prohibited = set(claim.get("prohibited") or [])
    required_prohibitions = {
        "human effect or model superiority",
        "webcam or word-level gaze accuracy",
        "cognitive ability, attention, fatigue, reading ability, English proficiency, or CEFR",
        "learning improvement or educational benefit",
        "formal sample-size authorization from rehearsal point estimates",
    }
    if not required_prohibitions.issubset(prohibited):
        failures.append("effect_accuracy_ability_learning_claims_prohibited")
    if (
        claim.get("formal_collection_authorized") is not False
        or claim.get("model_training_or_promotion_authorized") is not False
    ):
        failures.append("claim_authorization_must_remain_false")

    completion = _mapping(contract.get("completion_definitions"))
    completion_keys = {
        "visit1_consented_attempt",
        "visit1_started",
        "visit1_completed",
        "visit2_started",
        "visit2_completed",
        "paired_completed",
        "rate_denominators",
        "required_report",
    }
    denominators = _mapping(completion.get("rate_denominators"))
    if not completion_keys.issubset(completion) or set(denominators) != {
        "session_creation_fraction",
        "visit1_completion_fraction",
        "visit2_retention_fraction",
        "paired_completion_fraction",
    }:
        failures.append("completion_definitions_and_denominators")

    burden = _mapping(contract.get("burden_definitions"))
    if not {
        "active_visit_seconds",
        "wall_clock_attempt_seconds",
        "censored_unfinished_attempt_seconds",
        "calibration_validation_seconds",
        "reading_review_seconds",
        "researcher_intervention_count",
        "forced_restart_count",
        "required_summaries",
    }.issubset(burden):
        failures.append("active_time_and_intervention_burden")
    if burden.get("free_text_used_for_gate_decisions") is not False:
        failures.append("burden_gate_must_use_structured_fields")

    gaze = _mapping(contract.get("gaze_eligibility_definition"))
    if (
        gaze.get("unit") != "scheduled_capture_opportunity"
        or gaze.get("behavioral_labels_retained_when_gaze_ineligible") is not True
        or gaze.get("threshold_refitting_in_rehearsal_allowed") is not False
        or not isinstance(gaze.get("required_inputs"), list)
        or len(gaze.get("required_inputs")) < 10
    ):
        failures.append("gaze_eligibility_denominator_and_abstention")

    prevalence = _mapping(contract.get("class_prevalence_input"))
    if (
        tuple(prevalence.get("levels_in_report_order") or ()) != LABELS
        or tuple(prevalence.get("required_keys") or ()) != GROUP_KEYS
        or prevalence.get("sampling_probability_required") is not True
        or prevalence.get(
            "model_predictions_or_losses_in_blinded_aggregate_allowed"
        )
        is not False
        or "bare probe_id is prohibited"
        not in str(prevalence.get("passage_probe_id_definition", ""))
    ):
        failures.append("three_class_prevalence_blinded_input")

    icc = _mapping(contract.get("icc_planning_inputs"))
    if (
        icc.get("minimum_unique_participants_for_description") != 5
        or icc.get("minimum_unique_passage_families_for_description") != 8
        or icc.get("point_estimate_alone_may_replace_planning_range") is not False
        or icc.get("participant_or_passage_icc_estimable_under_this_rehearsal")
        is not False
        or icc.get("narrower_range_than_prior_pessimistic_allowed") is not False
        or icc.get("boundary_diagnostic_behavior")
        != (
            "retain_or_widen_prior_pessimistic_ranges_and_report_ICC_not_estimable"
        )
        or icc.get("sum_of_participant_and_passage_icc_must_be_below") != 1.0
    ):
        failures.append("crossed_cluster_icc_planning_boundary")

    stop_go = _mapping(contract.get("stop_go_criteria"))
    if set(stop_go) != {
        "STOP_AND_INVESTIGATE",
        "REPAIR_AND_REPEAT_REHEARSAL",
        "HOLD_NUISANCE_UPDATE",
        "GO_TO_BLINDED_FEASIBILITY_PLANNING_ONLY",
        "go_does_not_authorize",
    }:
        failures.append("frozen_stop_go_categories")
    elif any(
        not isinstance(stop_go[key], list) or not stop_go[key]
        for key in stop_go
    ):
        failures.append("frozen_stop_go_categories")

    aggregate = _mapping(contract.get("aggregate_result_contract"))
    if (
        aggregate.get("participant_count_is_bounded_5_to_8") is not True
        or aggregate.get("direct_identifiers_or_raw_media_allowed") is not False
        or aggregate.get("F0_F1_F2_predictions_losses_or_comparisons_allowed")
        is not False
        or aggregate.get("negative_and_missing_results_must_be_retained")
        is not True
        or aggregate.get(
            "five_to_eight_person_diagnostics_may_narrow_prior_pessimistic_ranges"
        )
        is not False
    ):
        failures.append("privacy_minimal_blinded_aggregate_contract")

    compute = _mapping(contract.get("compute"))
    if compute != {
        "device": "cpu",
        "gpu_allowed": False,
        "network_required": False,
        "deterministic_validator": True,
    }:
        failures.append("cpu_only_deterministic_contract")
    decision = _mapping(contract.get("decision"))
    if any(
        decision.get(key) is not False
        for key in (
            "participant_contact_authorized_by_this_contract",
            "formal_collection_authorized",
            "confirmation_outcome_access_authorized",
            "model_promotion_authorized",
        )
    ):
        failures.append("decision_must_remain_fail_closed")
    failures.extend(_authorization_failures(contract, "contract"))
    return sorted(set(failures))


def validate_worksheet_structure(worksheet: Mapping[str, Any]) -> list[str]:
    """Validate invariant worksheet structure while allowing null template cells."""

    failures: list[str] = []
    expected_top_level_keys = {
        "schema_version",
        "worksheet_id",
        "created_on",
        "status",
        "applies_to_protocol",
        "required_freeze_order",
        "interpretation_boundary",
        "comparison_contract",
        "elicitation_method",
        "practical_thresholds",
        "synthetic_anchor_packet",
        "external_evidence_contract",
        "attestations",
        "freeze_record",
        "authorization",
        "compute",
    }
    if set(worksheet) != expected_top_level_keys:
        failures.append("exact_worksheet_top_level_schema")
    if worksheet.get("schema_version") != 1:
        failures.append("worksheet_schema_version")
    if worksheet.get("worksheet_id") != (
        "lexigaze-chi-practical-effect-cost-elicitation-v1"
    ):
        failures.append("worksheet_id")
    if worksheet.get("status") not in {PENDING_STATUS, FROZEN_STATUS}:
        failures.append("worksheet_status")
    if worksheet.get("applies_to_protocol") != (
        "lexigaze-chi-reliability-aware-selective-fusion-v1"
    ):
        failures.append("worksheet_parent_protocol")
    if worksheet.get("required_freeze_order") != (
        "freeze_before_any_target_rehearsal_or_future_confirmation_cohort_"
        "word_review_outcome_or_F0_F1_F2_prediction_loss_or_comparison_is_"
        "opened"
    ):
        failures.append("worksheet_preoutcome_freeze_order")
    if worksheet.get("interpretation_boundary") != (
        "This worksheet defines a product-relevance threshold; it does not "
        "estimate an effect, accuracy, ability, proficiency, learning benefit, "
        "or required sample size."
    ):
        failures.append("worksheet_claim_boundary")

    comparison = _mapping(worksheet.get("comparison_contract"))
    expected_comparison = {
        "metric": "mean_multiclass_negative_log_likelihood",
        "loss_difference_convention": "F2_minus_reference_negative_is_better",
        "references": ["F1", "F0"],
        "estimand_population": (
            "all sampled probe-label rows in the single frozen joint "
            "participant-passage-capture-device confirmation partition"
        ),
        "F2_system_definition": (
            "use F2 only on predeclared selected gaze-eligible rows and exact "
            "F1 probabilities on missing, ineligible, and unselected rows; "
            "never drop a row"
        ),
        "F1_reference_definition": (
            "score F1 text-plus-person probabilities on the identical all-row "
            "confirmation population"
        ),
        "F0_reference_definition": (
            "score always-on F0 text-person-gaze probabilities on the identical "
            "all-row confirmation population using its development-frozen "
            "missing-gaze branch"
        ),
        "sampling_weight_rule": (
            "equal row weights only when all pre-outcome probe sampling "
            "probabilities are identical; otherwise use Hajek-normalized "
            "inverse-probability weights frozen before labels"
        ),
        "uncertainty_rule": (
            "the same participant-by-passage-family crossed-cluster method and "
            "the same rows and weights for F2_minus_F1 and F2_minus_F0"
        ),
        "joint_confirmatory_rule": (
            "on_the_same_frozen_confirmation_partition_both_upper_95_percent_"
            "bounds_must_be_at_or_below_their_negative_practical_thresholds"
        ),
        "separate_component_claims": (
            "descriptive_unless_separately_multiplicity_controlled"
        ),
        "conditional_selected_eligible_risk_may_replace_primary_estimand": False,
    }
    if comparison != expected_comparison:
        failures.append("exact_all_row_F2_vs_F1_F0_practical_estimand")

    elicitation = _mapping(worksheet.get("elicitation_method"))
    expected_elicitation_keys = {
        "name",
        "instructions",
        "nll_reduction_grid",
        "cost_context_ids",
        "cost_context_definitions",
        "rating_values",
        "role_ratings",
        "selected_threshold_rule",
        "operational_co_gates_are_separate_from_NLL",
        "operational_budget",
        "operational_gate_estimands",
        "confirmation_operational_rule",
    }
    if set(elicitation) != expected_elicitation_keys:
        failures.append("exact_elicitation_cells")
    if elicitation.get("name") != (
        "deterministic_unanimous_synthetic_grid_with_separate_operational_co_gates"
    ):
        failures.append("deterministic_synthetic_cost_elicitation_method")
    if elicitation.get("instructions") != ELICITATION_INSTRUCTIONS:
        failures.append("preoutcome_elicitation_instructions")
    if tuple(elicitation.get("nll_reduction_grid") or ()) != NLL_REDUCTION_GRID:
        failures.append("frozen_NLL_reduction_grid")
    if tuple(elicitation.get("cost_context_ids") or ()) != COST_CONTEXT_IDS:
        failures.append("frozen_cost_contexts")
    if _mapping(elicitation.get("cost_context_definitions")) != (
        COST_CONTEXT_DEFINITIONS
    ):
        failures.append("frozen_cost_context_definitions")
    if tuple(elicitation.get("rating_values") or ()) != RATING_VALUES:
        failures.append("frozen_rating_values")
    if not isinstance(elicitation.get("role_ratings"), list):
        failures.append("role_ratings_must_be_a_list")
    if elicitation.get("selected_threshold_rule") != (
        "minimum_positive_grid_value_unanimously_worthwhile_across_all_"
        "signoff_roles_and_cost_contexts"
    ):
        failures.append("deterministic_threshold_selection_rule")
    if elicitation.get("operational_co_gates_are_separate_from_NLL") is not True:
        failures.append("operational_co_gates_must_be_separate")
    if elicitation.get("confirmation_operational_rule") != (
        "all_five_operational_co_gates_must_pass_separately_with_no_weighting_"
        "tradeoff_or_compensation"
    ):
        failures.append("no_operational_cost_compensation")
    if set(_mapping(elicitation.get("operational_budget"))) != set(BUDGET_KEYS):
        failures.append("complete_operational_budget_cells")
    if _mapping(elicitation.get("operational_gate_estimands")) != (
        OPERATIONAL_GATE_ESTIMANDS
    ):
        failures.append("frozen_operational_gate_estimands")

    thresholds = _mapping(worksheet.get("practical_thresholds"))
    if set(thresholds) != set(COMPARISONS):
        failures.append("F2_vs_F1_F0_threshold_cells")
    else:
        required_threshold_keys = {
            "selected_minimum_reduction_nll",
            "derivation_rule",
            "required_observed_difference",
            "required_interval_bound",
        }
        for comparison_id in COMPARISONS:
            cells = _mapping(thresholds[comparison_id])
            reference_id = comparison_id.removeprefix("F2_minus_")
            if set(cells) != required_threshold_keys:
                failures.append("F2_vs_F1_F0_threshold_cells")
            if cells.get("derivation_rule") != (
                "minimum_positive_grid_value_unanimously_worthwhile_across_all_"
                "signoff_roles_and_cost_contexts"
            ):
                failures.append("deterministic_threshold_selection_rule")
            if cells.get("required_observed_difference") != (
                f"mean_F2_minus_{reference_id}_NLL_at_or_below_negative_"
                "selected_threshold"
            ) or cells.get("required_interval_bound") != (
                "upper_95_percent_bound_at_or_below_negative_selected_threshold"
            ):
                failures.append("F2_vs_F1_F0_practical_success_rules")

    packet = _mapping(worksheet.get("synthetic_anchor_packet"))
    if set(packet) != {
        "packet_id",
        "packet_sha256",
        "contains_participant_data",
        "contains_observed_model_effects",
        "scenario_cell_count",
        "nll_reduction_grid_matches_worksheet",
        "cost_context_ids_match_worksheet",
        "operational_profile_held_constant",
    } or packet.get("scenario_cell_count") != 36:
        failures.append("synthetic_anchor_packet_contract")

    external_contract = _mapping(worksheet.get("external_evidence_contract"))
    if external_contract != {
        "synthetic_packet": (
            "actual SHA256 match plus synthetic_only, no participant data, no "
            "observed effects, exact NLL grid, exact cost contexts, and 36 "
            "scenario cells"
        ),
        "confirmation_manifest": (
            "actual SHA256 match plus assignment frozen, target outcomes and "
            "model predictions absent, and confirmation partition unopened"
        ),
        "outcome_access_log": (
            "actual SHA256 match plus an as-of time at or after worksheet "
            "freeze and no target outcome, prediction, loss, or comparison access"
        ),
        "manual_review_still_required": True,
    }:
        failures.append("external_evidence_and_manual_review_contract")

    if set(_mapping(worksheet.get("attestations"))) != set(ATTESTATION_KEYS):
        failures.append("complete_preoutcome_attestations")
    freeze = _mapping(worksheet.get("freeze_record"))
    if set(freeze) != {
        "frozen_at_utc",
        "elicitor_roles",
        "independent_signoff_roles",
        "unopened_confirmation_assignment_manifest_sha256",
        "outcome_access_log_sha256",
        "change_policy",
    } or freeze.get("change_policy") != (
        "any_postfreeze_change_creates_a_new_version_and_requires_new_"
        "unopened_confirmation_data"
    ):
        failures.append("postfreeze_change_requires_new_confirmation")

    authorization = _mapping(worksheet.get("authorization"))
    if set(authorization) != set(AUTHORIZATION_KEYS) or any(
        authorization.get(key) is not False for key in AUTHORIZATION_KEYS
    ):
        failures.append("worksheet_authorization_must_remain_false")
    compute = _mapping(worksheet.get("compute"))
    if compute != {
        "validator_device": "cpu",
        "gpu_required": False,
        "network_required": False,
    }:
        failures.append("worksheet_cpu_only_offline")
    failures.extend(_authorization_failures(worksheet, "worksheet"))
    return sorted(set(failures))


def threshold_freeze_failures(worksheet: Mapping[str, Any]) -> list[str]:
    """Return why fields are incomplete for independent manual freeze review."""

    failures: list[str] = []
    if worksheet.get("status") != FROZEN_STATUS:
        failures.append("worksheet_not_frozen_preoutcome")

    elicitation = _mapping(worksheet.get("elicitation_method"))
    budget = _mapping(elicitation.get("operational_budget"))
    budget_values = {key: budget.get(key) for key in BUDGET_KEYS}
    if not all(_is_finite_number(value) for value in budget_values.values()):
        failures.append("operational_budget_must_be_finite")
    else:
        if not 0.0 < float(
            budget_values["maximum_added_active_seconds_per_visit"]
        ) <= 3600.0:
            failures.append("active_time_budget_out_of_range")
        if not 0.0 <= float(
            budget_values["maximum_added_compute_ms_per_passage"]
        ) <= 60_000.0:
            failures.append("compute_budget_out_of_range")
        if not 0.0 <= float(
            budget_values["maximum_added_review_prompts_per_passage"]
        ) <= 8.0:
            failures.append("review_prompt_budget_out_of_range")
        if not 0.0 < float(
            budget_values["minimum_system_gaze_eligible_coverage"]
        ) <= 1.0:
            failures.append("gaze_coverage_budget_out_of_range")
        if not 0.0 <= float(
            budget_values["maximum_absolute_calibration_slope_change"]
        ) <= 1.0:
            failures.append("calibration_budget_out_of_range")

    packet = _mapping(worksheet.get("synthetic_anchor_packet"))
    if not isinstance(packet.get("packet_id"), str) or not packet.get(
        "packet_id", ""
    ).strip():
        failures.append("synthetic_packet_id_required")
    if not _is_sha256(packet.get("packet_sha256")):
        failures.append("synthetic_packet_sha256_required")
    if packet.get("contains_participant_data") is not False:
        failures.append("synthetic_packet_must_exclude_participant_data")
    if packet.get("contains_observed_model_effects") is not False:
        failures.append("synthetic_packet_must_exclude_observed_effects")
    if packet.get("scenario_cell_count") != 36:
        failures.append("exact_36_synthetic_scenario_cells_required")
    if packet.get("nll_reduction_grid_matches_worksheet") is not True:
        failures.append("synthetic_packet_NLL_grid_must_match")
    if packet.get("cost_context_ids_match_worksheet") is not True:
        failures.append("synthetic_packet_cost_contexts_must_match")
    if packet.get("operational_profile_held_constant") is not True:
        failures.append("synthetic_threshold_profile_must_be_constant")

    attestations = _mapping(worksheet.get("attestations"))
    for key in ATTESTATION_KEYS:
        if attestations.get(key) is not False:
            failures.append(f"preoutcome_attestation_must_be_false:{key}")

    freeze = _mapping(worksheet.get("freeze_record"))
    if not _is_utc_timestamp(freeze.get("frozen_at_utc")):
        failures.append("valid_freeze_utc_required")
    elicitors = freeze.get("elicitor_roles")
    signoffs = freeze.get("independent_signoff_roles")
    if not _unique_nonempty_strings(elicitors, 1):
        failures.append("elicitor_role_required")
    if not _unique_nonempty_strings(signoffs, 2):
        failures.append("two_independent_signoff_roles_required")
    if (
        _unique_nonempty_strings(elicitors, 1)
        and _unique_nonempty_strings(signoffs, 2)
        and not set(signoffs).isdisjoint(elicitors)
    ):
        failures.append("signoff_roles_must_be_disjoint_from_elicitors")
    for key in (
        "unopened_confirmation_assignment_manifest_sha256",
        "outcome_access_log_sha256",
    ):
        if not _is_sha256(freeze.get(key)):
            failures.append(f"{key}_required")
    rating_failures, derived_thresholds = _rating_grid_diagnostics(
        worksheet, signoffs if isinstance(signoffs, list) else []
    )
    failures.extend(rating_failures)
    thresholds = _mapping(worksheet.get("practical_thresholds"))
    for comparison_id in COMPARISONS:
        selected = _mapping(thresholds.get(comparison_id)).get(
            "selected_minimum_reduction_nll"
        )
        derived = derived_thresholds.get(comparison_id)
        if not _is_finite_number(selected):
            failures.append(f"{comparison_id}_selected_threshold_must_be_finite")
        elif derived is None or not math.isclose(
            float(selected), derived, rel_tol=0.0, abs_tol=1e-12
        ):
            failures.append(f"{comparison_id}_threshold_must_equal_grid_consensus")
    return sorted(set(failures))


def _rating_grid_diagnostics(
    worksheet: Mapping[str, Any], signoff_roles: list[Any]
) -> tuple[list[str], dict[str, float]]:
    failures: list[str] = []
    if not _unique_nonempty_strings(signoff_roles, 2):
        return ["rating_grid_requires_valid_signoff_roles"], {}
    ratings = _mapping(worksheet.get("elicitation_method")).get("role_ratings")
    if not isinstance(ratings, list):
        return ["role_ratings_must_be_a_list"], {}
    observed: dict[tuple[str, str, str, float], str] = {}
    required_keys = {
        "role",
        "comparison_id",
        "cost_context_id",
        "nll_reduction",
        "judgment",
    }
    for index, raw_record in enumerate(ratings):
        record = _mapping(raw_record)
        if set(record) != required_keys:
            failures.append(f"rating_record_schema:{index}")
            continue
        role = record.get("role")
        comparison = record.get("comparison_id")
        context = record.get("cost_context_id")
        reduction = record.get("nll_reduction")
        judgment = record.get("judgment")
        if (
            role not in signoff_roles
            or comparison not in COMPARISONS
            or context not in COST_CONTEXT_IDS
            or not _is_finite_number(reduction)
            or float(reduction) not in NLL_REDUCTION_GRID
            or judgment not in RATING_VALUES
        ):
            failures.append(f"rating_record_value:{index}")
            continue
        key = (str(role), str(comparison), str(context), float(reduction))
        if key in observed:
            failures.append(f"duplicate_rating_record:{index}")
        observed[key] = str(judgment)

    expected = {
        (role, comparison, context, reduction)
        for role in signoff_roles
        for comparison in COMPARISONS
        for context in COST_CONTEXT_IDS
        for reduction in NLL_REDUCTION_GRID
    }
    if set(observed) != expected:
        failures.append("rating_grid_must_cover_exact_cartesian_product")

    for role in signoff_roles:
        for comparison in COMPARISONS:
            for context in COST_CONTEXT_IDS:
                judgments = [
                    observed.get((role, comparison, context, reduction))
                    for reduction in NLL_REDUCTION_GRID
                ]
                if "worthwhile" in judgments:
                    first_yes = judgments.index("worthwhile")
                    if any(
                        value != "worthwhile" for value in judgments[first_yes:]
                    ):
                        failures.append(
                            "rating_series_must_be_monotonic:"
                            f"{role}:{comparison}:{context}"
                        )

    derived: dict[str, float] = {}
    for comparison in COMPARISONS:
        for reduction in NLL_REDUCTION_GRID:
            if all(
                observed.get((role, comparison, context, reduction))
                == "worthwhile"
                for role in signoff_roles
                for context in COST_CONTEXT_IDS
            ):
                derived[comparison] = reduction
                break
        if comparison not in derived:
            failures.append(f"no_unanimous_threshold_on_grid:{comparison}")
    return sorted(set(failures)), derived


def _valid_probability_vector(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(LABELS)
        and all(_is_finite_number(item) for item in value)
        and all(0.0 <= float(item) <= 1.0 for item in value)
        and math.isclose(
            sum(float(item) for item in value),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _synthetic_scenario_cell_failures(
    packet: Mapping[str, Any], worksheet: Mapping[str, Any]
) -> list[str]:
    """Validate the exact synthetic, budget-boundary elicitation grid."""

    failures: list[str] = []
    cells = packet.get("scenario_cells")
    if not isinstance(cells, list):
        return ["synthetic_scenario_cells_must_be_a_list"]
    expected_cell_keys = {
        "comparison_id",
        "cost_context_id",
        "cost_context_definition",
        "nll_reduction",
        "operational_budget",
        "probability_examples",
    }
    expected_example_keys = {
        "true_class",
        "reference_probabilities",
        "F2_probabilities",
    }
    frozen_budget = _mapping(
        _mapping(worksheet.get("elicitation_method")).get("operational_budget")
    )
    observed: set[tuple[str, str, float]] = set()
    for cell_index, raw_cell in enumerate(cells):
        cell = _mapping(raw_cell)
        if set(cell) != expected_cell_keys:
            failures.append(f"synthetic_scenario_cell_schema:{cell_index}")
            continue
        comparison = cell.get("comparison_id")
        context = cell.get("cost_context_id")
        reduction = cell.get("nll_reduction")
        if (
            comparison not in COMPARISONS
            or context not in COST_CONTEXT_IDS
            or not _is_finite_number(reduction)
            or float(reduction) not in NLL_REDUCTION_GRID
        ):
            failures.append(f"synthetic_scenario_cell_value:{cell_index}")
            continue
        if cell.get("cost_context_definition") != COST_CONTEXT_DEFINITIONS[context]:
            failures.append(f"synthetic_cost_context_definition:{cell_index}")
        if _mapping(cell.get("operational_budget")) != frozen_budget:
            failures.append(f"synthetic_operational_budget_boundary:{cell_index}")
        cell_key = (str(comparison), str(context), float(reduction))
        if cell_key in observed:
            failures.append(f"duplicate_synthetic_scenario_cell:{cell_index}")
        observed.add(cell_key)

        examples = cell.get("probability_examples")
        if not isinstance(examples, list) or len(examples) != len(LABELS):
            failures.append(f"three_synthetic_probability_examples:{cell_index}")
            continue
        seen_labels: set[str] = set()
        for example_index, raw_example in enumerate(examples):
            example = _mapping(raw_example)
            if set(example) != expected_example_keys:
                failures.append(
                    f"synthetic_probability_example_schema:{cell_index}:"
                    f"{example_index}"
                )
                continue
            true_class = example.get("true_class")
            reference = example.get("reference_probabilities")
            f2 = example.get("F2_probabilities")
            if (
                true_class not in LABELS
                or not _valid_probability_vector(reference)
                or not _valid_probability_vector(f2)
            ):
                failures.append(
                    f"synthetic_probability_example_value:{cell_index}:"
                    f"{example_index}"
                )
                continue
            seen_labels.add(str(true_class))
            class_index = LABELS.index(str(true_class))
            reference_true = float(reference[class_index])
            f2_true = float(f2[class_index])
            if reference_true <= 0.0 or f2_true <= 0.0 or not math.isclose(
                math.log(f2_true / reference_true),
                float(reduction),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                failures.append(
                    f"synthetic_probability_NLL_reduction:{cell_index}:"
                    f"{example_index}"
                )
        if seen_labels != set(LABELS):
            failures.append(f"synthetic_probability_examples_cover_classes:{cell_index}")

    expected = {
        (comparison, context, reduction)
        for comparison in COMPARISONS
        for context in COST_CONTEXT_IDS
        for reduction in NLL_REDUCTION_GRID
    }
    if observed != expected or len(cells) != len(expected):
        failures.append("synthetic_scenario_grid_must_cover_exact_cartesian_product")
    return sorted(set(failures))


def external_evidence_failures(
    worksheet: Mapping[str, Any],
    *,
    synthetic_packet_path: Path | None,
    confirmation_manifest_path: Path | None,
    outcome_access_log_path: Path | None,
) -> list[str]:
    """Verify file hashes and fail-closed pre-outcome metadata semantics."""

    paths = {
        "synthetic_packet": synthetic_packet_path,
        "confirmation_manifest": confirmation_manifest_path,
        "outcome_access_log": outcome_access_log_path,
    }
    failures: list[str] = []
    evidence: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if path is None:
            failures.append(f"external_evidence_path_required:{name}")
            continue
        resolved = path.resolve()
        if not resolved.is_file():
            failures.append(f"external_evidence_file_missing:{name}")
            continue
        try:
            evidence[name] = load_json_object(resolved)
        except (OSError, ValueError, json.JSONDecodeError):
            failures.append(f"external_evidence_invalid_json:{name}")

    packet_contract = _mapping(worksheet.get("synthetic_anchor_packet"))
    freeze = _mapping(worksheet.get("freeze_record"))
    expected_hashes = {
        "synthetic_packet": packet_contract.get("packet_sha256"),
        "confirmation_manifest": freeze.get(
            "unopened_confirmation_assignment_manifest_sha256"
        ),
        "outcome_access_log": freeze.get("outcome_access_log_sha256"),
    }
    for name, path in paths.items():
        if path is not None and path.resolve().is_file() and _is_sha256(
            expected_hashes[name]
        ):
            if sha256(path.resolve()) != str(expected_hashes[name]).lower():
                failures.append(f"external_evidence_sha256_mismatch:{name}")

    packet = evidence.get("synthetic_packet")
    packet_schema = {
        "evidence_type",
        "packet_id",
        "data_source",
        "contains_participant_data",
        "contains_observed_model_effects",
        "nll_reduction_grid",
        "cost_context_ids",
        "scenario_cell_count",
        "operational_profile_held_constant",
        "created_at_utc",
        "scenario_cells",
    }
    if packet is not None:
        if set(packet) != packet_schema or not (
            packet.get("evidence_type") == "chi_synthetic_cost_anchor_packet_v1"
            and packet.get("packet_id") == packet_contract.get("packet_id")
            and packet.get("data_source") == "synthetic_only"
            and packet.get("contains_participant_data") is False
            and packet.get("contains_observed_model_effects") is False
            and tuple(packet.get("nll_reduction_grid") or ()) == NLL_REDUCTION_GRID
            and tuple(packet.get("cost_context_ids") or ()) == COST_CONTEXT_IDS
            and packet.get("scenario_cell_count") == 36
            and packet.get("operational_profile_held_constant") is True
        ):
            failures.append("synthetic_packet_semantics")
        failures.extend(_synthetic_scenario_cell_failures(packet, worksheet))

    manifest = evidence.get("confirmation_manifest")
    manifest_schema = {
        "evidence_type",
        "assignment_frozen_before_outcomes",
        "target_outcomes_included",
        "model_predictions_losses_or_comparisons_included",
        "confirmation_partition_opened",
        "frozen_at_utc",
    }
    if manifest is not None and (
        set(manifest) != manifest_schema
        or not (
            manifest.get("evidence_type")
            == "chi_unopened_confirmation_manifest_v1"
            and manifest.get("assignment_frozen_before_outcomes") is True
            and manifest.get("target_outcomes_included") is False
            and manifest.get("model_predictions_losses_or_comparisons_included")
            is False
            and manifest.get("confirmation_partition_opened") is False
        )
    ):
        failures.append("unopened_confirmation_manifest_semantics")

    access_log = evidence.get("outcome_access_log")
    access_log_schema = {
        "evidence_type",
        "target_cohort_word_review_outcomes_accessed",
        "target_cohort_F0_F1_F2_predictions_losses_or_comparisons_accessed",
        "confirmation_partition_opened",
        "as_of_utc",
    }
    if access_log is not None and (
        set(access_log) != access_log_schema
        or not (
            access_log.get("evidence_type")
            == "chi_outcome_access_log_snapshot_v1"
            and access_log.get("target_cohort_word_review_outcomes_accessed")
            is False
            and access_log.get(
                "target_cohort_F0_F1_F2_predictions_losses_or_comparisons_accessed"
            )
            is False
            and access_log.get("confirmation_partition_opened") is False
        )
    ):
        failures.append("outcome_access_log_semantics")

    freeze_time = _parse_utc_timestamp(freeze.get("frozen_at_utc"))
    packet_time = _parse_utc_timestamp(
        packet.get("created_at_utc") if packet is not None else None
    )
    manifest_time = _parse_utc_timestamp(
        manifest.get("frozen_at_utc") if manifest is not None else None
    )
    access_time = _parse_utc_timestamp(
        access_log.get("as_of_utc") if access_log is not None else None
    )
    if freeze_time is None:
        failures.append("external_evidence_requires_valid_freeze_time")
    else:
        if packet is not None and (packet_time is None or packet_time > freeze_time):
            failures.append("synthetic_packet_must_predate_or_equal_freeze")
        if manifest is not None and (
            manifest_time is None or manifest_time > freeze_time
        ):
            failures.append("confirmation_manifest_must_predate_or_equal_freeze")
        if access_log is not None and (
            access_time is None or access_time < freeze_time
        ):
            failures.append("outcome_access_log_must_cover_freeze_time")
    return sorted(set(failures))


def validation_report(
    contract: Mapping[str, Any],
    worksheet: Mapping[str, Any],
    *,
    contract_sha256: str | None = None,
    worksheet_sha256: str | None = None,
    evidence_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, fail-closed report without external state."""

    contract_failures = validate_rehearsal_contract(contract)
    worksheet_failures = validate_worksheet_structure(worksheet)
    freeze_failures = threshold_freeze_failures(worksheet)
    fields_complete = not worksheet_failures and not freeze_failures
    external_failures = (
        list(evidence_failures)
        if evidence_failures is not None
        else ["external_evidence_files_not_supplied"]
    )
    machine_gate_passed = (
        not contract_failures and fields_complete and not external_failures
    )
    return {
        "schema_version": 1,
        "validation_id": "chi-dress-rehearsal-contract-validation-v1",
        "contract_id": contract.get("protocol_id"),
        "worksheet_id": worksheet.get("worksheet_id"),
        "contract_sha256": contract_sha256,
        "worksheet_sha256": worksheet_sha256,
        "status": (
            "machine_verified_preoutcome_threshold_contract_pending_manual_acceptance"
            if not contract_failures and machine_gate_passed
            else (
                "valid_contract_threshold_gate_not_ready"
                if not contract_failures and not worksheet_failures
                else "invalid_contract_or_worksheet"
            )
        ),
        "contract_failures": contract_failures,
        "worksheet_structure_failures": worksheet_failures,
        "threshold_freeze_failures": freeze_failures,
        "external_evidence_failures": external_failures,
        "rehearsal_contract_structurally_valid": not contract_failures,
        "worksheet_structurally_valid": not worksheet_failures,
        "worksheet_fields_complete_for_manual_freeze_review": fields_complete,
        "external_preoutcome_evidence_machine_verified": not external_failures,
        "practical_threshold_contract_machine_gate_passed": machine_gate_passed,
        "manual_reviewer_must_confirm_role_identity_and_evidence_provenance": True,
        "preoutcome_order_machine_check_scope": (
            "contract_fields_ratings_actual_file_hashes_and_metadata_semantics"
        ),
        "hash_bound_snapshot_metadata_verified": not external_failures,
        "external_independently_timestamped_log_verified": False,
        "reads_participant_or_model_outcomes": False,
        "cpu_only": True,
        "gpu_used": False,
        "network_used": False,
        "participant_contact_authorized": False,
        "formal_collection_authorized": False,
        "confirmation_outcome_access_authorized": False,
        "model_training_or_promotion_authorized": False,
        "effect_accuracy_ability_or_learning_claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    parser.add_argument("--synthetic-packet-evidence", type=Path)
    parser.add_argument("--confirmation-manifest-evidence", type=Path)
    parser.add_argument("--outcome-access-log-evidence", type=Path)
    parser.add_argument(
        "--allow-pending-template",
        action="store_true",
        help=(
            "Return success for a structurally valid but unfilled template. "
            "Without this explicit flag, pending thresholds fail closed."
        ),
    )
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    worksheet_path = args.worksheet.resolve()
    contract = load_json_object(contract_path)
    worksheet = load_json_object(worksheet_path)
    evidence_failures = external_evidence_failures(
        worksheet,
        synthetic_packet_path=args.synthetic_packet_evidence,
        confirmation_manifest_path=args.confirmation_manifest_evidence,
        outcome_access_log_path=args.outcome_access_log_evidence,
    )
    report = validation_report(
        contract,
        worksheet,
        contract_sha256=sha256(contract_path),
        worksheet_sha256=sha256(worksheet_path),
        evidence_failures=evidence_failures,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["contract_failures"] or report["worksheet_structure_failures"]:
        return 2
    if (
        not args.allow_pending_template
        and not report["practical_threshold_contract_machine_gate_passed"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
