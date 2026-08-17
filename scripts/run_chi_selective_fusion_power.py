"""Run the frozen CPU-only CHI selective-fusion planning simulation.

This runner generates synthetic crossed participant/passage observations. It
does not read participant, question-answer, cognitive-profile, gaze, or text
model outcome data and cannot authorize recruitment or model promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT
    / "docs"
    / "CHI"
    / "protocols"
    / "2026-08-17-reliability-aware-selective-fusion-v1.json"
)
DEFAULT_JSON_OUTPUT = (
    ROOT
    / "docs"
    / "CHI"
    / "experiments"
    / "2026-08-17-reliability-aware-selective-fusion-power-v1.json"
)
DEFAULT_MARKDOWN_OUTPUT = DEFAULT_JSON_OUTPUT.with_suffix(".md")
LABELS = ("no_review", "unsure", "review_needed")
COMPARATOR_IDS = ("B0", "B1", "B2", "B3", "G1", "F0", "F1", "F2")
REQUIRED_GROUP_AXES = {
    "participant_id",
    "passage_family_id",
    "passage_probe_id",
    "capture_session_id",
    "device_group_id",
}


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(protocol: dict[str, Any]) -> list[str]:
    """Return protocol validation failures without touching any outcome data."""

    failures: list[str] = []
    if protocol.get("status") != (
        "planning_only_no_collection_or_model_promotion_authorization"
    ):
        failures.append("planning_only_status")

    outcome = dict(protocol.get("outcome") or {})
    if tuple(outcome.get("levels_in_report_order") or ()) != LABELS:
        failures.append("three_class_word_review_outcome")
    if outcome.get("primary_model_family") != (
        "regularized_multinomial_probability_model"
    ):
        failures.append("multinomial_primary_model")

    comparator_ids = tuple(
        str(item.get("id")) for item in protocol.get("comparators", [])
    )
    if comparator_ids != COMPARATOR_IDS:
        failures.append("frozen_B0_B1_B2_B3_G1_F0_F1_F2_comparators")
    f0 = next(
        (
            item
            for item in protocol.get("comparators", [])
            if item.get("id") == "F0"
        ),
        {},
    )
    if (
        f0.get("name") != "always_on_text_person_gaze_fusion"
        or f0.get("gaze_behavior")
        != "uses_observed_gaze_without_quality_abstention"
        or f0.get("true_missing_gaze_behavior")
        != (
            "development_only_deterministic_median_imputation_plus_"
            "missingness_indicator_without_F1_fallback"
        )
    ):
        failures.append("explicit_always_on_F0_comparator")
    f1 = next(
        (
            item
            for item in protocol.get("comparators", [])
            if item.get("id") == "F1"
        ),
        {},
    )
    if f1.get("name") != "text_plus_person_fallback":
        failures.append("explicit_text_person_F1_comparator")
    f2 = next(
        (
            item
            for item in protocol.get("comparators", [])
            if item.get("id") == "F2"
        ),
        {},
    )
    if f2.get("ineligible_gaze_behavior") != "exact_F1_probability_fallback":
        failures.append("exact_F1_fallback")

    split = dict(protocol.get("split_policy") or {})
    axes = set(split.get("required_group_axes") or [])
    if not REQUIRED_GROUP_AXES.issubset(axes):
        failures.append("joint_participant_passage_capture_device_holdout")
    key_definitions = dict(split.get("group_key_definitions") or {})
    if (
        "probe_id" in axes
        or split.get("bare_probe_id_split_key_allowed") is not False
        or "canonical_passage_id_double_colon_probe_id_composite"
        not in str(key_definitions.get("passage_probe_id", ""))
    ):
        failures.append("passage_probe_composite_split_key")
    partition_policy = split.get("partition_assignment_and_discard_policy")
    if not isinstance(partition_policy, list) or len(partition_policy) < 5:
        failures.append("operational_joint_split_and_discard_policy")
    else:
        joined_policy = " ".join(str(item) for item in partition_policy)
        required_policy_terms = (
            "Before outcome access",
            "inherits",
            "same partition",
            "discard it before",
            "Never reassign",
        )
        if not all(term in joined_policy for term in required_policy_terms):
            failures.append("operational_joint_split_and_discard_policy")
    if split.get("confirmation_access") != (
        "single_open_no_retuning_or_threshold_changes"
    ):
        failures.append("single_open_confirmation")
    if split.get("primary_device_group_scope") != (
        "participant_device_instance_configuration_group_nested_under_"
        "participant; holding out a participant also holds out that device "
        "instance but does not establish unseen device-class transfer"
    ):
        failures.append("primary_nested_device_group_scope")

    metrics = dict(protocol.get("metrics") or {})
    if metrics.get("primary") != "mean_multiclass_negative_log_likelihood":
        failures.append("primary_NLL")
    if metrics.get("coverage_grid") != [1.0, 0.8, 0.6, 0.4, 0.2]:
        failures.append("frozen_coverage_grid")
    if (
        metrics.get("coverage_denominator")
        != "gaze_eligible_observations_only"
        or metrics.get("coverage_selection_count_rounding")
        != "ceil_product_minus_1e-12_with_minimum_one_when_any_eligible"
        or metrics.get("conditional_accepted_risk")
        != "score_only_the_selected_eligible_observations"
        or metrics.get("system_hybrid_risk")
        != (
            "use_F2_on_selected_eligible_observations_and_F1_on_every_other_observation"
        )
    ):
        failures.append("selective_coverage_semantics")
    if metrics.get("ranked_probability_score_definition") != (
        "mean_squared_cumulative_probability_error_over_K_minus_1"
    ):
        failures.append("normalized_RPS_definition")
    if metrics.get(
        "minimum_unique_confirmation_device_groups_for_structural_diagnostic"
    ) != 8:
        failures.append("minimum_confirmation_device_groups")
    secondary = set(metrics.get("secondary") or [])
    if not {
        "multiclass_brier_score",
        "ranked_probability_score",
        "risk_coverage_curve",
    }.issubset(secondary):
        failures.append("secondary_probability_metrics")

    simulation = dict(protocol.get("simulation") or {})
    if simulation.get("data_source") != (
        "deterministic_synthetic_crossed_cluster_generator"
    ):
        failures.append("synthetic_only_data_source")
    for key in (
        "human_outcomes_used",
        "question_answer_data_used",
        "cognitive_profile_data_used",
        "truly_missing_gaze_simulated",
    ):
        if simulation.get(key) is not False:
            failures.append(f"{key}_must_be_false")
    if simulation.get("F2_vs_F0_simulation_scope") != (
        "observed_noisy_synthetic_gaze_only; excludes_true_missing_gaze_cases_"
        "and_is_not_informative_about_F0_imputation_behavior"
    ):
        failures.append("F2_vs_F0_true_missing_scope_boundary")
    if simulation.get("device_group_generator") != (
        "one_nested_synthetic_device_instance_group_per_paired_completed_"
        "confirmation_participant; development_validation_confirmation_"
        "assignment_is_not_exercised"
    ):
        failures.append("synthetic_nested_device_group_contract")
    if simulation.get("candidate_enrollments") != [20, 40, 144, 300, 600, 900]:
        failures.append("frozen_enrollment_grid")

    effects = dict(simulation.get("effect_scenarios") or {})
    if float(
        dict(effects.get("no_added_gaze_signal") or {}).get(
            "reliability_blend", math.nan
        )
    ) != 0.0:
        failures.append("exact_null_sentinel")
    if len(effects) < 3:
        failures.append("effect_sensitivity_breadth")

    power_test = dict(simulation.get("power_test") or {})
    estimands = list(power_test.get("estimands") or [])
    if (
        [item.get("id") for item in estimands if isinstance(item, dict)]
        != ["F2_minus_F1", "F2_minus_F0"]
        or power_test.get("joint_success")
        != "both_F2_minus_F1_and_F2_minus_F0_components_succeed"
    ):
        failures.append("F2_vs_F1_and_F0_joint_confirmatory_power_contract")

    scenarios = dict(simulation.get("scenario_bundles") or {})
    if set(scenarios) != {"optimistic", "base", "pessimistic"}:
        failures.append("quality_prevalence_ICC_scenario_breadth")
    for scenario_id, raw_scenario in scenarios.items():
        scenario = dict(raw_scenario)
        rates = (
            scenario.get("visit1_completion_probability"),
            scenario.get("visit2_retention_given_visit1"),
            scenario.get("word_level_gaze_session_eligibility"),
        )
        if any(value is None or not 0.0 <= float(value) <= 1.0 for value in rates):
            failures.append(f"{scenario_id}_valid_yield_rates")
        prevalence = [float(value) for value in scenario.get("class_prevalence", [])]
        if len(prevalence) != 3 or not math.isclose(sum(prevalence), 1.0):
            failures.append(f"{scenario_id}_valid_class_prevalence")
        participant_icc = float(scenario.get("latent_participant_icc", math.nan))
        passage_icc = float(scenario.get("latent_passage_icc", math.nan))
        if (
            not 0.0 <= participant_icc < 1.0
            or not 0.0 <= passage_icc < 1.0
            or participant_icc + passage_icc >= 1.0
        ):
            failures.append(f"{scenario_id}_valid_crossed_ICC")

    prohibited = set(
        dict(protocol.get("claim_boundary") or {}).get(
            "prohibited_training_selection_or_claim_sources", []
        )
    )
    required_prohibitions = {
        "question_answer_correctness",
        "comprehension_score",
        "cognitive_profile",
        "attention_label",
        "fatigue_label",
        "English_proficiency",
        "CEFR",
    }
    if not required_prohibitions.issubset(prohibited):
        failures.append("QA_cognitive_fatigue_proficiency_claims_prohibited")

    compute = dict(protocol.get("compute") or {})
    if (
        compute.get("device") != "cpu"
        or compute.get("gpu_allowed") is not False
        or compute.get("torch_required") is not False
        or compute.get("network_allowed") is not False
    ):
        failures.append("CPU_only_offline_compute")
    decision = dict(protocol.get("decision_policy") or {})
    if (
        decision.get("formal_recruitment_authorized") is not False
        or decision.get("model_promotion_authorized") is not False
        or decision.get("negative_and_non_monotonic_results_must_be_retained")
        is not True
    ):
        failures.append("planning_decision_boundary")
    return sorted(set(failures))


def _class_probabilities(
    latent_difficulty: np.ndarray, prevalence: np.ndarray
) -> np.ndarray:
    slopes = np.array([-0.85, -0.05, 0.85], dtype=np.float64)
    logits = np.log(prevalence)[None, :] + latent_difficulty[:, None] * slopes
    logits -= np.max(logits, axis=1, keepdims=True)
    unnormalized = np.exp(logits)
    return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)


def _draw_labels(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    draws = rng.random(probabilities.shape[0])
    cumulative = np.cumsum(probabilities, axis=1)
    return np.sum(draws[:, None] > cumulative, axis=1).astype(np.int64)


def loss_vectors(
    probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, np.ndarray]:
    """Return per-observation proper scoring-rule losses."""

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    if labels.shape != (probabilities.shape[0],):
        raise ValueError("labels must have shape (n,)")
    if probabilities.shape[0] == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-12):
        raise ValueError("each probability row must sum to one")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if np.any(labels < 0) or np.any(labels >= 3):
        raise ValueError("labels must be integer class indices in [0, 2]")

    row_index = np.arange(probabilities.shape[0])
    clipped = np.clip(probabilities[row_index, labels], 1e-15, 1.0)
    nll = -np.log(clipped)
    one_hot = np.eye(3, dtype=np.float64)[labels]
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    predicted_cumulative = np.cumsum(probabilities, axis=1)[:, :-1]
    observed_cumulative = np.cumsum(one_hot, axis=1)[:, :-1]
    rps = np.mean((predicted_cumulative - observed_cumulative) ** 2, axis=1)
    return {"nll": nll, "brier": brier, "rps": rps}


def _cluster_variance_component(values: np.ndarray, groups: np.ndarray) -> float:
    unique, inverse = np.unique(groups, return_inverse=True)
    if unique.size < 2:
        return math.nan
    centered = values - float(np.mean(values))
    cluster_sums = np.bincount(inverse, weights=centered)
    correction = unique.size / (unique.size - 1.0)
    return float(correction * np.sum(cluster_sums**2) / values.size**2)


def conservative_two_way_cluster_se(
    values: np.ndarray,
    participant_ids: np.ndarray,
    passage_ids: np.ndarray,
) -> float:
    """Return a conservative participant-by-passage cluster standard error.

    The standard two-way sandwich combines the participant and passage
    components and subtracts their intersection. For planning, this function
    conservatively takes the largest of that estimate and each one-way
    component so a negative finite-sample intersection adjustment cannot make
    the interval spuriously narrow.
    """

    if not (values.shape == participant_ids.shape == passage_ids.shape):
        raise ValueError("values and cluster identifiers must have equal shape")
    participant_variance = _cluster_variance_component(values, participant_ids)
    passage_variance = _cluster_variance_component(values, passage_ids)
    intersection = np.column_stack((participant_ids, passage_ids))
    _, intersection_ids = np.unique(intersection, axis=0, return_inverse=True)
    intersection_variance = _cluster_variance_component(values, intersection_ids)
    if not all(
        math.isfinite(item)
        for item in (participant_variance, passage_variance, intersection_variance)
    ):
        return math.nan
    two_way = participant_variance + passage_variance - intersection_variance
    return math.sqrt(max(0.0, participant_variance, passage_variance, two_way))


def _comparison_summary(
    differences: np.ndarray,
    participant_ids: np.ndarray,
    passage_ids: np.ndarray,
    *,
    structurally_evaluable: bool,
    critical_value: float,
) -> dict[str, float | bool | None]:
    mean_difference = float(np.mean(differences))
    standard_error = conservative_two_way_cluster_se(
        differences, participant_ids, passage_ids
    )
    ci_low = (
        mean_difference - critical_value * standard_error
        if math.isfinite(standard_error)
        else None
    )
    ci_high = (
        mean_difference + critical_value * standard_error
        if math.isfinite(standard_error)
        else None
    )
    return {
        "mean_difference": mean_difference,
        "conservative_two_way_cluster_se": (
            standard_error if math.isfinite(standard_error) else None
        ),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "significant_improvement": bool(
            structurally_evaluable
            and ci_high is not None
            and mean_difference < 0.0
            and ci_high < 0.0
        ),
    }


def _risk_coverage_rows(
    coverage_grid: list[float],
    reliability: np.ndarray,
    f0_losses: dict[str, np.ndarray],
    f1_losses: dict[str, np.ndarray],
    f2_losses: dict[str, np.ndarray],
    gaze_eligible: np.ndarray,
) -> list[dict[str, Any]]:
    eligible_ids = np.flatnonzero(gaze_eligible)
    eligible_order = eligible_ids[
        np.lexsort((eligible_ids, -reliability[eligible_ids]))
    ]
    eligible_count = int(eligible_ids.size)
    rows: list[dict[str, Any]] = []
    for coverage in coverage_grid:
        selected_count = (
            max(1, math.ceil(float(coverage) * eligible_count - 1e-12))
            if eligible_count
            else 0
        )
        selected = eligible_order[:selected_count]
        row: dict[str, Any] = {
            "target_eligible_coverage": float(coverage),
            "eligible_observations": eligible_count,
            "selected_eligible_observations": selected_count,
            "realized_eligible_coverage": (
                selected_count / eligible_count if eligible_count else 0.0
            ),
            "conditional_accepted_risk": {},
            "system_hybrid_risk": {},
        }
        for metric in ("nll", "brier", "rps"):
            conditional: dict[str, float | None]
            if selected_count:
                conditional_f0 = float(np.mean(f0_losses[metric][selected]))
                conditional_f1 = float(np.mean(f1_losses[metric][selected]))
                conditional_f2 = float(np.mean(f2_losses[metric][selected]))
                conditional = {
                    "F0": conditional_f0,
                    "F1": conditional_f1,
                    "F2": conditional_f2,
                    "F2_minus_F1": conditional_f2 - conditional_f1,
                    "F2_minus_F0": conditional_f2 - conditional_f0,
                }
            else:
                conditional = {
                    "F0": None,
                    "F1": None,
                    "F2": None,
                    "F2_minus_F1": None,
                    "F2_minus_F0": None,
                }
            hybrid_losses = f1_losses[metric].copy()
            hybrid_losses[selected] = f2_losses[metric][selected]
            system_f0 = float(np.mean(f0_losses[metric]))
            system_f1 = float(np.mean(f1_losses[metric]))
            system_hybrid = float(np.mean(hybrid_losses))
            row["conditional_accepted_risk"][metric] = conditional
            row["system_hybrid_risk"][metric] = {
                "F0_always_on": system_f0,
                "F1_text_person": system_f1,
                "selective_hybrid": system_hybrid,
                "selective_hybrid_minus_F1": system_hybrid - system_f1,
                "selective_hybrid_minus_F0": system_hybrid - system_f0,
            }
        rows.append(row)
    return rows


def simulate_replication(
    *,
    enrollment: int,
    protocol: dict[str, Any],
    scenario: dict[str, Any],
    reliability_blend: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    simulation = dict(protocol["simulation"])
    metrics = dict(protocol["metrics"])
    confirmation_fraction = float(simulation["confirmation_participant_fraction"])
    planned_confirmation = max(1, round(enrollment * confirmation_fraction))
    visit1_complete = rng.random(planned_confirmation) < float(
        scenario["visit1_completion_probability"]
    )
    visit2_complete = visit1_complete & (
        rng.random(planned_confirmation)
        < float(scenario["visit2_retention_given_visit1"])
    )
    completed_original_ids = np.flatnonzero(visit2_complete)
    completed_count = int(completed_original_ids.size)
    passage_count = int(simulation["confirmation_passage_families"])
    passages_per_participant = int(
        simulation["confirmation_passages_per_paired_participant"]
    )
    probes_per_passage = int(simulation["word_probes_per_passage"])

    if completed_count == 0:
        return {
            "planned_confirmation_participants": planned_confirmation,
            "paired_completed_participants": 0,
            "joint_confirmation_observations": 0,
            "unique_confirmation_passages": 0,
            "unique_confirmation_device_groups": 0,
            "gaze_eligible_fraction": 0.0,
            "class_counts": [0, 0, 0],
            "structurally_evaluable": False,
            "fallback_exact": True,
            "metrics": None,
            "risk_coverage": [],
        }

    participant_indices: list[int] = []
    passage_indices: list[int] = []
    capture_indices: list[int] = []
    for compact_participant, original_participant in enumerate(
        completed_original_ids.tolist()
    ):
        selected_passages = rng.choice(
            passage_count, size=passages_per_participant, replace=False
        )
        for slot, passage_id in enumerate(selected_passages.tolist()):
            capture_id = original_participant * 2 + min(slot, 1)
            participant_indices.extend([compact_participant] * probes_per_passage)
            passage_indices.extend([passage_id] * probes_per_passage)
            capture_indices.extend([capture_id] * probes_per_passage)

    participant_ids = np.asarray(participant_indices, dtype=np.int64)
    passage_ids = np.asarray(passage_indices, dtype=np.int64)
    capture_ids = np.asarray(capture_indices, dtype=np.int64)
    device_group_ids = participant_ids.copy()
    participant_icc = float(scenario["latent_participant_icc"])
    passage_icc = float(scenario["latent_passage_icc"])
    residual_fraction = 1.0 - participant_icc - passage_icc
    participant_effects = rng.normal(size=completed_count)
    passage_effects = rng.normal(size=passage_count)
    participant_component = (
        math.sqrt(participant_icc) * participant_effects[participant_ids]
    )
    passage_component = math.sqrt(passage_icc) * passage_effects[passage_ids]
    residual_component = math.sqrt(residual_fraction) * rng.normal(
        size=participant_ids.size
    )
    true_latent = participant_component + passage_component + residual_component

    f1_latent = (
        float(scenario["person_anchor_fraction"]) * participant_component
        + float(scenario["text_signal_fraction"])
        * (passage_component + residual_component)
        + rng.normal(
            scale=float(scenario["f1_noise_sd"]), size=participant_ids.size
        )
    )
    prevalence = np.asarray(scenario["class_prevalence"], dtype=np.float64)
    true_probabilities = _class_probabilities(true_latent, prevalence)
    labels = _draw_labels(true_probabilities, rng)
    f1_probabilities = _class_probabilities(f1_latent, prevalence)

    unique_capture_ids, capture_inverse = np.unique(capture_ids, return_inverse=True)
    capture_eligible = rng.random(unique_capture_ids.size) < float(
        scenario["word_level_gaze_session_eligibility"]
    )
    raw_capture_reliability = rng.beta(
        float(scenario["reliability_beta_alpha"]),
        float(scenario["reliability_beta_beta"]),
        size=unique_capture_ids.size,
    )
    capture_reliability = np.where(capture_eligible, raw_capture_reliability, 0.0)
    gaze_eligible = capture_eligible[capture_inverse]
    reliability = capture_reliability[capture_inverse]
    gaze_noise_scale = float(scenario["gaze_noise_sd"]) / (0.25 + reliability)
    gaze_latent = true_latent + rng.normal(
        scale=gaze_noise_scale, size=true_latent.size
    )
    f0_latent = f1_latent + reliability_blend * (gaze_latent - f1_latent)
    f2_latent = f1_latent.copy()
    f2_latent[gaze_eligible] += (
        reliability_blend
        * reliability[gaze_eligible]
        * (gaze_latent[gaze_eligible] - f1_latent[gaze_eligible])
    )
    f0_probabilities = _class_probabilities(f0_latent, prevalence)
    f2_probabilities = _class_probabilities(f2_latent, prevalence)
    fallback_difference = (
        np.max(
            np.abs(
                f2_probabilities[~gaze_eligible]
                - f1_probabilities[~gaze_eligible]
            )
        )
        if np.any(~gaze_eligible)
        else 0.0
    )
    fallback_exact = bool(fallback_difference <= 1e-12)

    f0_losses = loss_vectors(f0_probabilities, labels)
    f1_losses = loss_vectors(f1_probabilities, labels)
    f2_losses = loss_vectors(f2_probabilities, labels)
    metric_summary: dict[str, Any] = {}
    minimum_participants = int(
        metrics["minimum_unique_confirmation_participants_for_power_diagnostic"]
    )
    minimum_passages = int(
        metrics["minimum_unique_confirmation_passage_families_for_power_diagnostic"]
    )
    minimum_device_groups = int(
        metrics["minimum_unique_confirmation_device_groups_for_structural_diagnostic"]
    )
    unique_passages = int(np.unique(passage_ids).size)
    unique_device_groups = int(np.unique(device_group_ids).size)
    structurally_evaluable = (
        completed_count >= minimum_participants
        and unique_passages >= minimum_passages
        and unique_device_groups >= minimum_device_groups
    )
    critical = NormalDist().inv_cdf(
        1.0 - float(metrics["alpha_two_sided"]) / 2.0
    )
    for metric in ("nll", "brier", "rps"):
        metric_summary[metric] = {
            "model_means": {
                "F0": float(np.mean(f0_losses[metric])),
                "F1": float(np.mean(f1_losses[metric])),
                "F2": float(np.mean(f2_losses[metric])),
            },
            "comparisons": {
                "F2_minus_F1": _comparison_summary(
                    f2_losses[metric] - f1_losses[metric],
                    participant_ids,
                    passage_ids,
                    structurally_evaluable=structurally_evaluable,
                    critical_value=critical,
                ),
                "F2_minus_F0": _comparison_summary(
                    f2_losses[metric] - f0_losses[metric],
                    participant_ids,
                    passage_ids,
                    structurally_evaluable=structurally_evaluable,
                    critical_value=critical,
                ),
                "F0_minus_F1": _comparison_summary(
                    f0_losses[metric] - f1_losses[metric],
                    participant_ids,
                    passage_ids,
                    structurally_evaluable=structurally_evaluable,
                    critical_value=critical,
                ),
            },
        }

    return {
        "planned_confirmation_participants": planned_confirmation,
        "paired_completed_participants": completed_count,
        "joint_confirmation_observations": int(labels.size),
        "unique_confirmation_passages": unique_passages,
        "unique_confirmation_device_groups": unique_device_groups,
        "gaze_eligible_fraction": float(np.mean(gaze_eligible)),
        "class_counts": np.bincount(labels, minlength=3).astype(int).tolist(),
        "structurally_evaluable": structurally_evaluable,
        "fallback_exact": fallback_exact,
        "metrics": metric_summary,
        "risk_coverage": _risk_coverage_rows(
            list(metrics["coverage_grid"]),
            reliability,
            f0_losses,
            f1_losses,
            f2_losses,
            gaze_eligible,
        ),
    }


def _mean(values: list[float | int]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _quantile(values: list[float | int], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _mean_optional(values: list[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return _mean(finite) if finite else None


def _aggregate_replications(
    replications: list[dict[str, Any]], coverage_grid: list[float]
) -> dict[str, Any]:
    evaluable = [item for item in replications if item["structurally_evaluable"]]
    metric_rows = [item for item in replications if item["metrics"] is not None]
    totals = np.sum(
        np.asarray([item["class_counts"] for item in replications], dtype=np.int64),
        axis=0,
    )
    total_labels = int(np.sum(totals))
    aggregate_metrics: dict[str, Any] = {}
    for metric in ("nll", "brier", "rps"):
        aggregate_metrics[metric] = {
            "model_means": {
                model_id: _mean(
                    [
                        item["metrics"][metric]["model_means"][model_id]
                        for item in metric_rows
                    ]
                )
                for model_id in ("F0", "F1", "F2")
            },
            "comparisons": {},
        }
        for comparison_id in ("F2_minus_F1", "F2_minus_F0", "F0_minus_F1"):
            differences = [
                float(
                    item["metrics"][metric]["comparisons"][comparison_id][
                        "mean_difference"
                    ]
                )
                for item in metric_rows
            ]
            standard_errors = [
                float(
                    item["metrics"][metric]["comparisons"][comparison_id][
                        "conservative_two_way_cluster_se"
                    ]
                )
                for item in evaluable
                if item["metrics"][metric]["comparisons"][comparison_id][
                    "conservative_two_way_cluster_se"
                ]
                is not None
            ]
            significant = [
                bool(
                    item["metrics"][metric]["comparisons"][comparison_id][
                        "significant_improvement"
                    ]
                )
                for item in evaluable
            ]
            aggregate_metrics[metric]["comparisons"][comparison_id] = {
                "mean_difference": _mean(differences) if differences else None,
                "replication_mean_difference_p05": (
                    _quantile(differences, 0.05) if differences else None
                ),
                "replication_mean_difference_p95": (
                    _quantile(differences, 0.95) if differences else None
                ),
                "mean_conservative_two_way_cluster_se": (
                    _mean(standard_errors) if standard_errors else None
                ),
                "diagnostic_power_among_structurally_evaluable_replications": (
                    _mean([int(value) for value in significant])
                    if significant
                    else None
                ),
            }

    joint_significant = [
        all(
            bool(
                item["metrics"]["nll"]["comparisons"][comparison_id][
                    "significant_improvement"
                ]
            )
            for comparison_id in ("F2_minus_F1", "F2_minus_F0")
        )
        for item in evaluable
    ]
    aggregate_metrics["nll"]["joint_H1_intersection_union"] = {
        "definition": (
            "same_replication_F2_minus_F1_and_F2_minus_F0_"
            "significant_improvement"
        ),
        "structurally_evaluable_replications": len(evaluable),
        "joint_diagnostic_power_among_structurally_evaluable_replications": (
            _mean([int(value) for value in joint_significant])
            if joint_significant
            else None
        ),
    }

    coverage_rows: list[dict[str, Any]] = []
    for coverage_index, coverage in enumerate(coverage_grid):
        available = [
            item["risk_coverage"][coverage_index]
            for item in metric_rows
            if item["risk_coverage"]
        ]
        row: dict[str, Any] = {
            "target_eligible_coverage": coverage,
            "mean_eligible_observations": _mean(
                [item["eligible_observations"] for item in available]
            ),
            "mean_selected_eligible_observations": _mean(
                [item["selected_eligible_observations"] for item in available]
            ),
            "mean_realized_eligible_coverage": _mean(
                [item["realized_eligible_coverage"] for item in available]
            ),
            "conditional_accepted_risk": {},
            "system_hybrid_risk": {},
        }
        for metric in ("nll", "brier", "rps"):
            row["conditional_accepted_risk"][metric] = {
                key: _mean_optional(
                    [item["conditional_accepted_risk"][metric][key] for item in available]
                )
                for key in ("F0", "F1", "F2", "F2_minus_F1", "F2_minus_F0")
            }
            row["system_hybrid_risk"][metric] = {
                key: _mean(
                    [item["system_hybrid_risk"][metric][key] for item in available]
                )
                for key in (
                    "F0_always_on",
                    "F1_text_person",
                    "selective_hybrid",
                    "selective_hybrid_minus_F1",
                    "selective_hybrid_minus_F0",
                )
            }
        coverage_rows.append(row)

    return {
        "mean_planned_confirmation_participants": _mean(
            [item["planned_confirmation_participants"] for item in replications]
        ),
        "mean_paired_completed_confirmation_participants": _mean(
            [item["paired_completed_participants"] for item in replications]
        ),
        "p05_paired_completed_confirmation_participants": _quantile(
            [item["paired_completed_participants"] for item in replications], 0.05
        ),
        "mean_joint_confirmation_observations": _mean(
            [item["joint_confirmation_observations"] for item in replications]
        ),
        "mean_unique_confirmation_passages": _mean(
            [item["unique_confirmation_passages"] for item in replications]
        ),
        "mean_unique_confirmation_device_groups": _mean(
            [item["unique_confirmation_device_groups"] for item in replications]
        ),
        "mean_gaze_eligible_fraction": _mean(
            [item["gaze_eligible_fraction"] for item in replications]
        ),
        "structurally_evaluable_probability": len(evaluable) / len(replications),
        "all_replications_exact_F1_fallback": all(
            item["fallback_exact"] for item in replications
        ),
        "aggregate_class_proportions": (
            (totals / total_labels).astype(float).tolist()
            if total_labels
            else [0.0, 0.0, 0.0]
        ),
        "metrics": aggregate_metrics,
        "risk_coverage": coverage_rows,
    }


def run_simulation(
    protocol: dict[str, Any],
    *,
    replicates_override: int | None = None,
    protocol_sha256: str | None = None,
    implementation_sha256: str | None = None,
    runtime_reference_seconds: float | None = None,
) -> dict[str, Any]:
    failures = validate_protocol(protocol)
    if failures:
        raise ValueError("invalid protocol: " + ", ".join(failures))
    if runtime_reference_seconds is not None and runtime_reference_seconds < 0.0:
        raise ValueError("runtime_reference_seconds must be nonnegative")
    simulation = dict(protocol["simulation"])
    metrics = dict(protocol["metrics"])
    replicates = int(
        replicates_override
        if replicates_override is not None
        else simulation["replicates_per_cell"]
    )
    if replicates < 1:
        raise ValueError("replicates must be positive")
    seed = int(simulation["seed"])
    rows: list[dict[str, Any]] = []
    scenario_items = list(dict(simulation["scenario_bundles"]).items())
    effect_items = list(dict(simulation["effect_scenarios"]).items())
    for scenario_index, (scenario_id, raw_scenario) in enumerate(scenario_items):
        scenario = dict(raw_scenario)
        for effect_index, (effect_id, raw_effect) in enumerate(effect_items):
            effect = dict(raw_effect)
            blend = float(effect["reliability_blend"])
            for enrollment in simulation["candidate_enrollments"]:
                replications = []
                for replicate in range(replicates):
                    child_seed = np.random.SeedSequence(
                        [
                            seed,
                            scenario_index,
                            effect_index,
                            int(enrollment),
                            replicate,
                        ]
                    )
                    replications.append(
                        simulate_replication(
                            enrollment=int(enrollment),
                            protocol=protocol,
                            scenario=scenario,
                            reliability_blend=blend,
                            rng=np.random.default_rng(child_seed),
                        )
                    )
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "effect_id": effect_id,
                        "reliability_blend": blend,
                        "enrolled": int(enrollment),
                        **_aggregate_replications(
                            replications, list(metrics["coverage_grid"])
                        ),
                    }
                )

    power_policy = dict(simulation["power_test"])
    power_threshold = float(power_policy["diagnostic_power_threshold"])
    evaluable_threshold = float(
        power_policy["diagnostic_evaluable_probability_threshold"]
    )
    first_n_diagnostics: list[dict[str, Any]] = []
    negative_findings: list[str] = []
    for scenario_id, _ in scenario_items:
        for effect_id, _ in effect_items:
            candidates = [
                row
                for row in rows
                if row["scenario_id"] == scenario_id
                and row["effect_id"] == effect_id
                and row["structurally_evaluable_probability"] >= evaluable_threshold
                and row["metrics"]["nll"]["joint_H1_intersection_union"][
                    "joint_diagnostic_power_among_structurally_evaluable_replications"
                ]
                is not None
                and row["metrics"]["nll"]["joint_H1_intersection_union"][
                    "joint_diagnostic_power_among_structurally_evaluable_replications"
                ]
                >= power_threshold
            ]
            first_n = (
                min(int(item["enrolled"]) for item in candidates)
                if candidates
                else None
            )
            first_n_diagnostics.append(
                {
                    "scenario_id": scenario_id,
                    "effect_id": effect_id,
                    "first_candidate_n_meeting_planning_diagnostics": first_n,
                    "is_recruitment_recommendation": False,
                }
            )
            if first_n is None and effect_id != "no_added_gaze_signal":
                negative_findings.append(
                    f"{scenario_id}/{effect_id}: no candidate enrollment met both "
                    "the 0.80 evaluability and 0.80 same-replication joint "
                    "diagnostic-power thresholds."
                )

    for enrollment in (20, 40):
        minimum_evaluable = min(
            float(row["structurally_evaluable_probability"])
            for row in rows
            if row["enrolled"] == enrollment
        )
        if minimum_evaluable < evaluable_threshold:
            negative_findings.append(
                f"N={enrollment}: at least one assumption bundle did not reach "
                "0.80 structural evaluability; this cell remains feasibility-only."
            )

    for scenario_id, _ in scenario_items:
        for enrollment in simulation["candidate_enrollments"]:
            weak = next(
                row
                for row in rows
                if row["scenario_id"] == scenario_id
                and row["effect_id"] == "weak_added_gaze_signal"
                and row["enrolled"] == enrollment
            )
            moderate = next(
                row
                for row in rows
                if row["scenario_id"] == scenario_id
                and row["effect_id"] == "moderate_added_gaze_signal"
                and row["enrolled"] == enrollment
            )
            weak_power = weak["metrics"]["nll"]["comparisons"]["F2_minus_F1"][
                "diagnostic_power_among_structurally_evaluable_replications"
            ]
            moderate_power = moderate["metrics"]["nll"]["comparisons"][
                "F2_minus_F1"
            ]["diagnostic_power_among_structurally_evaluable_replications"]
            if (
                weak_power is not None
                and moderate_power is not None
                and float(moderate_power) + 0.05 < float(weak_power)
            ):
                negative_findings.append(
                    f"{scenario_id}/N={enrollment}: the moderate blend had lower "
                    "F2-vs-F1 diagnostic power than the weak blend; more gaze weight "
                    "is not uniformly better under the frozen noise model."
                )

    null_rows = [
        row for row in rows if row["effect_id"] == "no_added_gaze_signal"
    ]
    null_fallback_passed = all(
        row["all_replications_exact_F1_fallback"]
        and all(
            abs(
                float(
                    row["metrics"][metric]["comparisons"][comparison_id][
                        "mean_difference"
                    ]
                    or 0.0
                )
            )
            <= 1e-12
            for metric in ("nll", "brier", "rps")
            for comparison_id in (
                "F2_minus_F1",
                "F2_minus_F0",
                "F0_minus_F1",
            )
        )
        for row in null_rows
    )
    if not null_fallback_passed:
        negative_findings.append(
            "The exact-F1 null sentinel failed and invalidates the planning run."
        )

    result = {
        "schema_version": 1,
        "experiment_id": "chi-reliability-aware-selective-fusion-power-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "status": (
            "completed_planning_only"
            if null_fallback_passed
            else "failed_integrity_gate"
        ),
        "seed": seed,
        "replicates_per_cell": replicates,
        "candidate_enrollments": simulation["candidate_enrollments"],
        "scenario_ids": [item[0] for item in scenario_items],
        "effect_ids": [item[0] for item in effect_items],
        "labels": list(LABELS),
        "primary_metric": metrics["primary"],
        "coverage_grid": metrics["coverage_grid"],
        "comparators": list(COMPARATOR_IDS),
        "holdout_contract": {
            "primary": protocol["split_policy"]["primary_holdout"],
            "required_group_axes": protocol["split_policy"]["required_group_axes"],
            "protocol_declared_joint_holdout": True,
            "partition_assignment_exercised_by_generator": False,
            "complete_capture_and_device_groups_held_out_by_construction": False,
            "generator_scope": "confirmation_only_fit_free_sensitivity_rows",
            "simulation_fits_or_selects_no_model": True,
        },
        "integrity": {
            "protocol_validation_failures": [],
            "synthetic_data_only": True,
            "human_outcomes_used": False,
            "question_answer_data_used": False,
            "cognitive_profile_data_used": False,
            "truly_missing_gaze_simulated": False,
            "F2_vs_F0_includes_true_missing_gaze_cases": False,
            "network_used": False,
            "gpu_used": False,
            "torch_imported": False,
            "all_cells_exact_F1_fallback": all(
                row["all_replications_exact_F1_fallback"] for row in rows
            ),
            "null_sentinel_passed": null_fallback_passed,
        },
        "rows": rows,
        "planning_diagnostics": first_n_diagnostics,
        "negative_findings": negative_findings,
        "decision": {
            "formal_recruitment_authorized": False,
            "model_promotion_authorized": False,
            "recommended_enrollment": None,
            "reason": (
                "The displayed first-N cells are assumption-sensitive diagnostics, "
                "not power based on human effect or nuisance estimates. Replace the "
                "yield, prevalence, quality, ICC, and practical-effect assumptions "
                "with blinded rehearsal estimates before freezing a formal sample size."
            ),
        },
        "limitations": [
            "No human outcome, gaze, question-answer, text-model, or cognitive-profile data are read.",
            "Effect blends are sensitivity anchors, not a smallest effect of interest.",
            "The H1 practical threshold is not yet frozen; this run diagnoses zero-bound interval exclusion only and therefore cannot test H1 as written.",
            "The no-added-signal cell is an exact F1 sentinel, not a calibrated type-I-error experiment.",
            "The generator supplies noisy gaze for every row and treats eligibility as a quality gate; F2-vs-F0 diagnostics exclude true-missing cases and are not informative about F0 imputation behavior.",
            "The conservative crossed-cluster interval is a planning approximation and is unstable with few independent participant or passage clusters.",
            "Synthetic device groups are nested one-per-participant, so they add a structural count but no independent variance axis; a separate device-class transfer study still requires enough independent device classes.",
            "The generator emits fit-free confirmation-only sensitivity rows and does not exercise the declared development-validation-confirmation partition assignment or discard implementation.",
            "NLL, Brier, RPS, and risk-coverage estimates are properties of the frozen synthetic generator, not evidence of human model benefit.",
        ],
        "execution": {
            "runtime_reference_seconds": runtime_reference_seconds,
            "runtime_reference_method": (
                "prior_identical_no_write_run"
                if runtime_reference_seconds is not None
                else "not_embedded"
            ),
        },
    }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    def format_optional(value: float | int | None, digits: int = 3) -> str:
        if value is None:
            return "NE"
        return f"{float(value):.{digits}f}"

    table_rows = []
    for row in result["rows"]:
        nll = row["metrics"]["nll"]
        versus_f1 = nll["comparisons"]["F2_minus_F1"]
        versus_f0 = nll["comparisons"]["F2_minus_F0"]
        joint_power = nll["joint_H1_intersection_union"][
            "joint_diagnostic_power_among_structurally_evaluable_replications"
        ]
        table_rows.append(
            "| {scenario_id} | {effect_id} | {enrolled} | "
            "{completed:.1f} | {observations:.1f} | {gaze:.3f} | "
            "{evaluable:.3f} | {delta_f1} | {power_f1} | {delta_f0} | "
            "{power_f0} | {joint_power} |".format(
                scenario_id=row["scenario_id"],
                effect_id=row["effect_id"],
                enrolled=row["enrolled"],
                completed=row["mean_paired_completed_confirmation_participants"],
                observations=row["mean_joint_confirmation_observations"],
                gaze=row["mean_gaze_eligible_fraction"],
                evaluable=row["structurally_evaluable_probability"],
                delta_f1=format_optional(versus_f1["mean_difference"], 5),
                power_f1=format_optional(
                    versus_f1[
                        "diagnostic_power_among_structurally_evaluable_replications"
                    ]
                ),
                delta_f0=format_optional(versus_f0["mean_difference"], 5),
                power_f0=format_optional(
                    versus_f0[
                        "diagnostic_power_among_structurally_evaluable_replications"
                    ]
                ),
                joint_power=format_optional(joint_power),
            )
        )

    diagnostics = []
    for item in result["planning_diagnostics"]:
        diagnostic_n = item["first_candidate_n_meeting_planning_diagnostics"]
        diagnostics.append(
            "| {scenario_id} | {effect_id} | {n} | no |".format(
                scenario_id=item["scenario_id"],
                effect_id=item["effect_id"],
                n=diagnostic_n if diagnostic_n is not None else "none tested",
            )
        )

    max_enrollment = max(result["candidate_enrollments"])
    coverage_source = next(
        row
        for row in result["rows"]
        if row["scenario_id"] == "base"
        and row["effect_id"] == "moderate_added_gaze_signal"
        and row["enrolled"] == max_enrollment
    )
    coverage_rows = []
    for row in coverage_source["risk_coverage"]:
        conditional = row["conditional_accepted_risk"]["nll"]
        system = row["system_hybrid_risk"]["nll"]
        coverage_rows.append(
            "| {coverage:.1f} | {realized:.3f} | {selected:.1f} | "
            "{conditional_delta} | {system_f1:.4f} | {hybrid:.4f} | "
            "{hybrid_f1:+.5f} | {hybrid_f0:+.5f} |".format(
                coverage=row["target_eligible_coverage"],
                realized=row["mean_realized_eligible_coverage"],
                selected=row["mean_selected_eligible_observations"],
                conditional_delta=format_optional(conditional["F2_minus_F1"], 5),
                system_f1=system["F1_text_person"],
                hybrid=system["selective_hybrid"],
                hybrid_f1=system["selective_hybrid_minus_F1"],
                hybrid_f0=system["selective_hybrid_minus_F0"],
            )
        )

    negative_findings = "\n".join(
        f"- {finding}" for finding in result["negative_findings"]
    ) or "- No additional threshold failure was generated; all limitations still apply."
    limitations = "\n".join(
        f"- {limitation}" for limitation in result["limitations"]
    )
    return f"""# Reliability-Aware Selective Fusion v1: Planning Simulation

- Status: **`{result['status']}`**
- Protocol: `{result['protocol_id']}`
- Seed: `{result['seed']}`; replications per cell: `{result['replicates_per_cell']}`
- Compute: CPU-only NumPy; GPU used: `{str(result['integrity']['gpu_used']).lower()}`; Torch imported: `{str(result['integrity']['torch_imported']).lower()}`
- Human, QA, and cognitive-profile outcomes used: **none**
- Runtime reference: `{result['execution']['runtime_reference_seconds']}` seconds (`{result['execution']['runtime_reference_method']}`)

## Interpretation boundary

This is a deterministic crossed-cluster **planning sensitivity analysis**. It
does not estimate a human effect, authorize recruitment, promote F2, establish
webcam accuracy, or support cognitive, attention, fatigue, English-proficiency,
CEFR, or learning-benefit claims. The first-N diagnostics below are not sample
size recommendations.

The simulated outcome is the independently collected three-class word-review
response (`no_review`, `unsure`, `review_needed`). F0 is always-on
text-person-gaze fusion, F1 is the text-plus-person fallback, and F2 is
reliability-aware selective fusion. The primary estimands are F2 minus F1 and
F2 minus F0 mean multiclass NLL on a joint held-out participant,
passage-family, capture-session, and device-group cell. Brier score, normalized
ranked probability score (mean over the K-1 cumulative thresholds), and every
frozen risk-coverage cell are secondary.

The F2-vs-F0 sensitivity diagnostic includes only rows with synthetic noisy
gaze. It contains no true-missing gaze cases and cannot evaluate F0's frozen
future imputation branch. The generator also emits confirmation-only rows and
does not exercise the declared partition-assignment implementation.

## Yield and primary NLL sensitivity

| Scenario | Added gaze signal | Enrolled | Mean paired confirmation participants | Mean joint labels | Mean gaze eligibility | Structural evaluability | Mean F2-F1 NLL | Power vs F1 | Mean F2-F0 NLL | Power vs F0 | Same-replication joint power |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_rows)}

`NE` means too few independent participant or passage clusters for the frozen
planning diagnostic. Marginal power is the fraction of structurally evaluable
synthetic replications whose conservative participant-by-passage 95% interval
excludes zero in the improvement direction. Joint power is the fraction where
both F2-vs-F1 and F2-vs-F0 succeed in the same replication; only joint power is
used for the first-N diagnostic.

## First tested N meeting the planning diagnostics

| Scenario | Added gaze signal | First tested N | Recruitment recommendation? |
| --- | --- | ---: | --- |
{chr(10).join(diagnostics)}

These cells use assumed completion, gaze quality, class prevalence, crossed
ICC, and gaze signal. They must be replaced with blinded rehearsal estimates
and a pre-outcome product cost threshold before a formal sample size is frozen.

## Frozen risk-coverage example

Base assumptions, moderate sensitivity anchor, N={max_enrollment}:

| Target eligible coverage | Realized eligible coverage | Mean selected eligible | Conditional accepted F2-F1 NLL | All-row F1 NLL | All-row selective hybrid NLL | Hybrid-F1 NLL | Hybrid-F0 NLL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(coverage_rows)}

Conditional accepted risk scores only the selected eligible rows. All-row
system risk uses F2 on those selected rows and exact F1 fallback on every
unselected eligible or ineligible row; no observation is dropped from system
risk. No threshold was selected from these results. All five predeclared
coverage cells are retained, including any non-monotonic outcome.

## Integrity and negative results

- Exact F1 fallback passed in every cell: `{str(result['integrity']['all_cells_exact_F1_fallback']).lower()}`.
- Exact no-added-gaze null sentinel passed: `{str(result['integrity']['null_sentinel_passed']).lower()}`.
- Formal recruitment authorized: **false**.
- Model promotion authorized: **false**.

{negative_findings}

## Limitations

{limitations}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT
    )
    parser.add_argument(
        "--runtime-reference-seconds",
        type=float,
        help=(
            "Fixed runtime from a prior identical no-write run. The current "
            "runtime is printed but never embedded, preserving byte identity."
        ),
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    implementation_path = Path(__file__).resolve()
    started = time.perf_counter()
    result = run_simulation(
        protocol,
        replicates_override=args.replicates,
        protocol_sha256=_sha256(protocol_path),
        implementation_sha256=_sha256(implementation_path),
        runtime_reference_seconds=args.runtime_reference_seconds,
    )
    actual_runtime_seconds = round(time.perf_counter() - started, 4)
    if not args.no_write:
        _atomic_write(
            args.json_output.resolve(),
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write(args.markdown_output.resolve(), render_markdown(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "replicates_per_cell": result["replicates_per_cell"],
                "cells": len(result["rows"]),
                "formal_recruitment_authorized": False,
                "model_promotion_authorized": False,
                "gpu_used": False,
                "actual_runtime_seconds": actual_runtime_seconds,
                "runtime_embedded": args.runtime_reference_seconds,
                "json_output": None if args.no_write else str(args.json_output.resolve()),
                "markdown_output": (
                    None if args.no_write else str(args.markdown_output.resolve())
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "completed_planning_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
