"""Tests for the additive CHI dress-rehearsal and cost-threshold contracts."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.validate_chi_dress_rehearsal import (
    COMPARISONS,
    COST_CONTEXT_DEFINITIONS,
    COST_CONTEXT_IDS,
    DEFAULT_CONTRACT,
    DEFAULT_WORKSHEET,
    FROZEN_STATUS,
    NLL_REDUCTION_GRID,
    external_evidence_failures,
    load_json_object,
    main as validator_main,
    sha256,
    threshold_freeze_failures,
    validate_rehearsal_contract,
    validate_worksheet_structure,
    validation_report,
)


SIGNOFF_ROLES = ("study_methodologist", "hci_researcher")


def completed_worksheet() -> dict:
    worksheet = copy.deepcopy(load_json_object(DEFAULT_WORKSHEET))
    worksheet["status"] = FROZEN_STATUS
    worksheet["elicitation_method"]["operational_budget"] = {
        "maximum_added_active_seconds_per_visit": 300.0,
        "maximum_added_compute_ms_per_passage": 500.0,
        "maximum_added_review_prompts_per_passage": 1.0,
        "minimum_system_gaze_eligible_coverage": 0.30,
        "maximum_absolute_calibration_slope_change": 0.05,
    }
    thresholds = {"F2_minus_F1": 0.01, "F2_minus_F0": 0.02}
    ratings = []
    for role in SIGNOFF_ROLES:
        for comparison_id in COMPARISONS:
            threshold = thresholds[comparison_id]
            for context_id in COST_CONTEXT_IDS:
                for reduction in NLL_REDUCTION_GRID:
                    ratings.append(
                        {
                            "role": role,
                            "comparison_id": comparison_id,
                            "cost_context_id": context_id,
                            "nll_reduction": reduction,
                            "judgment": (
                                "worthwhile"
                                if reduction >= threshold
                                else "not_worthwhile"
                            ),
                        }
                    )
    worksheet["elicitation_method"]["role_ratings"] = ratings
    for comparison_id, threshold in thresholds.items():
        worksheet["practical_thresholds"][comparison_id][
            "selected_minimum_reduction_nll"
        ] = threshold
    worksheet["synthetic_anchor_packet"].update(
        {
            "packet_id": "synthetic-cost-anchor-v1",
            "packet_sha256": "a" * 64,
            "contains_participant_data": False,
            "contains_observed_model_effects": False,
            "nll_reduction_grid_matches_worksheet": True,
            "cost_context_ids_match_worksheet": True,
            "operational_profile_held_constant": True,
        }
    )
    worksheet["attestations"] = {
        key: False for key in worksheet["attestations"]
    }
    worksheet["freeze_record"].update(
        {
            "frozen_at_utc": "2026-08-17T12:00:00Z",
            "elicitor_roles": ["product_owner"],
            "independent_signoff_roles": list(SIGNOFF_ROLES),
            "unopened_confirmation_assignment_manifest_sha256": "b" * 64,
            "outcome_access_log_sha256": "c" * 64,
        }
    )
    return worksheet


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_external_evidence(
    root: Path, worksheet: dict
) -> tuple[Path, Path, Path]:
    packet_path = root / "synthetic_packet.json"
    manifest_path = root / "confirmation_manifest.json"
    access_path = root / "outcome_access_log.json"
    scenario_cells = []
    operational_budget = worksheet["elicitation_method"]["operational_budget"]
    for comparison_id in COMPARISONS:
        for context_id in COST_CONTEXT_IDS:
            for reduction in NLL_REDUCTION_GRID:
                probability_examples = []
                for true_class_index, true_class in enumerate(
                    ("no_review", "unsure", "review_needed")
                ):
                    reference = [0.3, 0.3, 0.3]
                    reference[true_class_index] = 0.4
                    f2_true = 0.4 * math.exp(reduction)
                    f2 = [(1.0 - f2_true) / 2.0] * 3
                    f2[true_class_index] = f2_true
                    probability_examples.append(
                        {
                            "true_class": true_class,
                            "reference_probabilities": reference,
                            "F2_probabilities": f2,
                        }
                    )
                scenario_cells.append(
                    {
                        "comparison_id": comparison_id,
                        "cost_context_id": context_id,
                        "cost_context_definition": COST_CONTEXT_DEFINITIONS[
                            context_id
                        ],
                        "nll_reduction": reduction,
                        "operational_budget": copy.deepcopy(operational_budget),
                        "probability_examples": probability_examples,
                    }
                )
    _write_json(
        packet_path,
        {
            "evidence_type": "chi_synthetic_cost_anchor_packet_v1",
            "packet_id": worksheet["synthetic_anchor_packet"]["packet_id"],
            "data_source": "synthetic_only",
            "contains_participant_data": False,
            "contains_observed_model_effects": False,
            "nll_reduction_grid": list(NLL_REDUCTION_GRID),
            "cost_context_ids": list(COST_CONTEXT_IDS),
            "scenario_cell_count": 36,
            "operational_profile_held_constant": True,
            "created_at_utc": "2026-08-17T10:00:00Z",
            "scenario_cells": scenario_cells,
        },
    )
    _write_json(
        manifest_path,
        {
            "evidence_type": "chi_unopened_confirmation_manifest_v1",
            "assignment_frozen_before_outcomes": True,
            "target_outcomes_included": False,
            "model_predictions_losses_or_comparisons_included": False,
            "confirmation_partition_opened": False,
            "frozen_at_utc": "2026-08-17T11:00:00Z",
        },
    )
    _write_json(
        access_path,
        {
            "evidence_type": "chi_outcome_access_log_snapshot_v1",
            "target_cohort_word_review_outcomes_accessed": False,
            "target_cohort_F0_F1_F2_predictions_losses_or_comparisons_accessed": False,
            "confirmation_partition_opened": False,
            "as_of_utc": "2026-08-17T12:01:00Z",
        },
    )
    worksheet["synthetic_anchor_packet"]["packet_sha256"] = sha256(packet_path)
    worksheet["freeze_record"][
        "unopened_confirmation_assignment_manifest_sha256"
    ] = sha256(manifest_path)
    worksheet["freeze_record"]["outcome_access_log_sha256"] = sha256(
        access_path
    )
    return packet_path, manifest_path, access_path


class DressRehearsalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_json_object(DEFAULT_CONTRACT)
        cls.worksheet = load_json_object(DEFAULT_WORKSHEET)

    def test_default_contract_is_valid_but_template_fails_closed(self) -> None:
        report = validation_report(self.contract, self.worksheet)
        self.assertEqual(report["contract_failures"], [])
        self.assertEqual(report["worksheet_structure_failures"], [])
        self.assertTrue(report["rehearsal_contract_structurally_valid"])
        self.assertFalse(
            report["worksheet_fields_complete_for_manual_freeze_review"]
        )
        self.assertFalse(report["practical_threshold_contract_machine_gate_passed"])
        self.assertIn(
            "worksheet_not_frozen_preoutcome",
            report["threshold_freeze_failures"],
        )
        for key in (
            "participant_contact_authorized",
            "formal_collection_authorized",
            "confirmation_outcome_access_authorized",
            "model_training_or_promotion_authorized",
            "effect_accuracy_ability_or_learning_claim_authorized",
        ):
            self.assertFalse(report[key])

    def test_contract_freezes_slots_and_5_to_8_process_only_scope(self) -> None:
        participant = self.contract["participant_scope"]
        self.assertEqual(participant["minimum_started_participants"], 5)
        self.assertEqual(participant["maximum_started_participants"], 8)
        self.assertFalse(participant["extension_to_chase_a_gate_allowed"])
        self.assertEqual(participant["analysis_role"], "process_and_measurement_only")
        self.assertFalse(
            participant["rehearsal_rows_may_enter_formal_confirmation"]
        )
        self.assertEqual(
            participant["outcome_blind_slot_plan"]["slot_ids"],
            [f"DR{index:02d}" for index in range(1, 9)],
        )

    def test_contract_includes_failed_attempts_and_never_narrows_priors(self) -> None:
        completion = self.contract["completion_definitions"]
        self.assertIn("visit1_consented_attempt", completion)
        self.assertIn("session_creation_fraction", completion["rate_denominators"])
        self.assertIn(
            "censored_unfinished_attempt_seconds", self.contract["burden_definitions"]
        )
        self.assertEqual(
            self.contract["gaze_eligibility_definition"]["unit"],
            "scheduled_capture_opportunity",
        )
        self.assertFalse(
            self.contract["icc_planning_inputs"][
                "participant_or_passage_icc_estimable_under_this_rehearsal"
            ]
        )
        self.assertFalse(
            self.contract["aggregate_result_contract"][
                "five_to_eight_person_diagnostics_may_narrow_prior_pessimistic_ranges"
            ]
        )

    def test_expanded_sample_or_removed_claim_prohibition_is_rejected(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["participant_scope"]["maximum_started_participants"] = 12
        contract["claim_boundary"]["prohibited"].remove(
            "webcam or word-level gaze accuracy"
        )
        failures = validate_rehearsal_contract(contract)
        self.assertIn("frozen_5_to_8_process_only_participant_scope", failures)
        self.assertIn(
            "effect_accuracy_ability_learning_claims_prohibited", failures
        )

    def test_every_frozen_semantic_section_rejects_rewrites(self) -> None:
        mutations = {
            "participant_scope": lambda value: value.update(
                {"analysis_role": "formal_model_evaluation"}
            ),
            "claim_boundary": lambda value: value["allowed"].append(
                "human effect or model superiority"
            ),
            "completion_definitions": lambda value: value[
                "rate_denominators"
            ].update({"paired_completion_fraction": "always_one"}),
            "gaze_eligibility_definition": lambda value: value.update(
                {"numerator": value["denominator"], "denominator": value["numerator"]}
            ),
            "icc_planning_inputs": lambda value: value.update(
                {"frozen_estimator": "unclustered_mean"}
            ),
            "stop_go_criteria": lambda value: value.update(
                {key: ["always pass"] for key in value}
            ),
            "aggregate_result_contract": lambda value: value.update(
                {
                    "individual_process_values_must_use_rehearsal_pseudonyms": False,
                    "protocol_bank_and_export_sha256_required": False,
                }
            ),
        }
        for section, mutate in mutations.items():
            with self.subTest(section=section):
                contract = copy.deepcopy(self.contract)
                mutate(contract[section])
                self.assertIn(
                    f"frozen_contract_section_changed:{section}",
                    validate_rehearsal_contract(contract),
                )

    def test_whole_contract_hash_rejects_unsectioned_rewrite(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["purpose"] = "use rehearsal rows to claim model superiority"
        self.assertIn(
            "frozen_whole_contract_changed",
            validate_rehearsal_contract(contract),
        )

    def test_any_authorization_flip_is_rejected(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["decision"]["formal_collection_authorized"] = True
        failures = validate_rehearsal_contract(contract)
        self.assertIn("decision_must_remain_fail_closed", failures)
        self.assertTrue(
            any(value.startswith("authorization_must_remain_false:") for value in failures)
        )


class PracticalThresholdWorksheetTests(unittest.TestCase):
    def test_completed_fields_require_external_evidence(self) -> None:
        worksheet = completed_worksheet()
        self.assertEqual(validate_worksheet_structure(worksheet), [])
        self.assertEqual(threshold_freeze_failures(worksheet), [])
        report = validation_report(load_json_object(DEFAULT_CONTRACT), worksheet)
        self.assertTrue(report["worksheet_fields_complete_for_manual_freeze_review"])
        self.assertFalse(report["external_preoutcome_evidence_machine_verified"])
        self.assertFalse(report["practical_threshold_contract_machine_gate_passed"])
        self.assertEqual(report["status"], "valid_contract_threshold_gate_not_ready")

    def test_actual_external_evidence_can_pass_machine_gate_only(self) -> None:
        worksheet = completed_worksheet()
        with tempfile.TemporaryDirectory() as raw_root:
            paths = create_external_evidence(Path(raw_root), worksheet)
            failures = external_evidence_failures(
                worksheet,
                synthetic_packet_path=paths[0],
                confirmation_manifest_path=paths[1],
                outcome_access_log_path=paths[2],
            )
        self.assertEqual(failures, [])
        report = validation_report(
            load_json_object(DEFAULT_CONTRACT),
            worksheet,
            evidence_failures=failures,
        )
        self.assertTrue(report["external_preoutcome_evidence_machine_verified"])
        self.assertTrue(report["practical_threshold_contract_machine_gate_passed"])
        self.assertTrue(
            report["manual_reviewer_must_confirm_role_identity_and_evidence_provenance"]
        )
        self.assertFalse(report["participant_contact_authorized"])

    def test_invalid_contract_cannot_report_machine_gate_passed(self) -> None:
        worksheet = completed_worksheet()
        invalid_contract = load_json_object(DEFAULT_CONTRACT)
        invalid_contract["purpose"] = "formal model superiority claim"
        with tempfile.TemporaryDirectory() as raw_root:
            paths = create_external_evidence(Path(raw_root), worksheet)
            failures = external_evidence_failures(
                worksheet,
                synthetic_packet_path=paths[0],
                confirmation_manifest_path=paths[1],
                outcome_access_log_path=paths[2],
            )
        report = validation_report(
            invalid_contract,
            worksheet,
            evidence_failures=failures,
        )
        self.assertTrue(report["contract_failures"])
        self.assertFalse(report["practical_threshold_contract_machine_gate_passed"])

    def test_operational_gate_estimands_are_frozen(self) -> None:
        worksheet = completed_worksheet()
        worksheet["elicitation_method"]["operational_gate_estimands"][
            "maximum_absolute_calibration_slope_change"
        ]["reference"] = "unspecified"
        self.assertIn(
            "frozen_operational_gate_estimands",
            validate_worksheet_structure(worksheet),
        )

    def test_external_hash_mismatch_fails(self) -> None:
        worksheet = completed_worksheet()
        with tempfile.TemporaryDirectory() as raw_root:
            paths = create_external_evidence(Path(raw_root), worksheet)
            paths[0].write_text("{}\n", encoding="utf-8")
            failures = external_evidence_failures(
                worksheet,
                synthetic_packet_path=paths[0],
                confirmation_manifest_path=paths[1],
                outcome_access_log_path=paths[2],
            )
        self.assertIn(
            "external_evidence_sha256_mismatch:synthetic_packet", failures
        )

    def test_empty_external_objects_fail_even_with_matching_hashes(self) -> None:
        worksheet = completed_worksheet()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            paths = tuple(root / name for name in ("packet.json", "manifest.json", "log.json"))
            for path in paths:
                _write_json(path, {})
            worksheet["synthetic_anchor_packet"]["packet_sha256"] = sha256(paths[0])
            worksheet["freeze_record"][
                "unopened_confirmation_assignment_manifest_sha256"
            ] = sha256(paths[1])
            worksheet["freeze_record"]["outcome_access_log_sha256"] = sha256(
                paths[2]
            )
            failures = external_evidence_failures(
                worksheet,
                synthetic_packet_path=paths[0],
                confirmation_manifest_path=paths[1],
                outcome_access_log_path=paths[2],
            )
        self.assertIn("synthetic_packet_semantics", failures)
        self.assertIn("unopened_confirmation_manifest_semantics", failures)
        self.assertIn("outcome_access_log_semantics", failures)

    def test_synthetic_packet_must_cover_exact_validated_scenario_grid(self) -> None:
        worksheet = completed_worksheet()
        with tempfile.TemporaryDirectory() as raw_root:
            paths = create_external_evidence(Path(raw_root), worksheet)
            packet = load_json_object(paths[0])
            packet["scenario_cells"][0] = copy.deepcopy(
                packet["scenario_cells"][1]
            )
            _write_json(paths[0], packet)
            worksheet["synthetic_anchor_packet"]["packet_sha256"] = sha256(paths[0])
            failures = external_evidence_failures(
                worksheet,
                synthetic_packet_path=paths[0],
                confirmation_manifest_path=paths[1],
                outcome_access_log_path=paths[2],
            )
        self.assertIn(
            "synthetic_scenario_grid_must_cover_exact_cartesian_product",
            failures,
        )

    def test_opened_outcome_log_fails_even_when_hash_matches(self) -> None:
        worksheet = completed_worksheet()
        with tempfile.TemporaryDirectory() as raw_root:
            paths = create_external_evidence(Path(raw_root), worksheet)
            log = load_json_object(paths[2])
            log["target_cohort_word_review_outcomes_accessed"] = True
            _write_json(paths[2], log)
            worksheet["freeze_record"]["outcome_access_log_sha256"] = sha256(paths[2])
            failures = external_evidence_failures(
                worksheet,
                synthetic_packet_path=paths[0],
                confirmation_manifest_path=paths[1],
                outcome_access_log_path=paths[2],
            )
        self.assertIn("outcome_access_log_semantics", failures)

    def test_observed_outcome_or_model_access_attestation_fails(self) -> None:
        worksheet = completed_worksheet()
        for key in (
            "target_cohort_word_review_outcomes_accessed_before_freeze",
            "target_cohort_F0_F1_F2_predictions_or_losses_accessed_before_freeze",
        ):
            worksheet["attestations"][key] = True
        failures = threshold_freeze_failures(worksheet)
        self.assertIn(
            "preoutcome_attestation_must_be_false:"
            "target_cohort_word_review_outcomes_accessed_before_freeze",
            failures,
        )

    def test_primary_estimand_cannot_be_changed_to_selected_subset(self) -> None:
        worksheet = completed_worksheet()
        worksheet["comparison_contract"][
            "conditional_selected_eligible_risk_may_replace_primary_estimand"
        ] = True
        self.assertIn(
            "exact_all_row_F2_vs_F1_F0_practical_estimand",
            validate_worksheet_structure(worksheet),
        )

    def test_signoff_roles_must_be_disjoint_from_elicitor(self) -> None:
        worksheet = completed_worksheet()
        worksheet["freeze_record"]["independent_signoff_roles"] = [
            "product_owner",
            "study_methodologist",
        ]
        self.assertIn(
            "signoff_roles_must_be_disjoint_from_elicitors",
            threshold_freeze_failures(worksheet),
        )

    def test_selected_threshold_must_equal_unanimous_grid_boundary(self) -> None:
        worksheet = completed_worksheet()
        worksheet["practical_thresholds"]["F2_minus_F1"][
            "selected_minimum_reduction_nll"
        ] = 0.02
        self.assertIn(
            "F2_minus_F1_threshold_must_equal_grid_consensus",
            threshold_freeze_failures(worksheet),
        )

    def test_nonmonotonic_role_rating_fails(self) -> None:
        worksheet = completed_worksheet()
        record = next(
            item
            for item in worksheet["elicitation_method"]["role_ratings"]
            if item["role"] == SIGNOFF_ROLES[0]
            and item["comparison_id"] == "F2_minus_F1"
            and item["cost_context_id"] == COST_CONTEXT_IDS[0]
            and item["nll_reduction"] == 0.04
        )
        record["judgment"] = "not_worthwhile"
        self.assertTrue(
            any(
                failure.startswith("rating_series_must_be_monotonic:")
                for failure in threshold_freeze_failures(worksheet)
            )
        )

    def test_nonfinite_operational_budget_fails(self) -> None:
        for invalid in (float("nan"), float("inf"), -1.0):
            with self.subTest(invalid=invalid):
                worksheet = completed_worksheet()
                worksheet["elicitation_method"]["operational_budget"][
                    "maximum_added_active_seconds_per_visit"
                ] = invalid
                self.assertTrue(threshold_freeze_failures(worksheet))

    def test_report_is_deterministic_and_cpu_only(self) -> None:
        contract = load_json_object(DEFAULT_CONTRACT)
        worksheet = completed_worksheet()
        first = validation_report(
            contract,
            worksheet,
            contract_sha256="a" * 64,
            worksheet_sha256="b" * 64,
        )
        second = validation_report(
            copy.deepcopy(contract),
            copy.deepcopy(worksheet),
            contract_sha256="a" * 64,
            worksheet_sha256="b" * 64,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["cpu_only"])
        self.assertFalse(first["gpu_used"])
        self.assertFalse(first["network_used"])

    def test_default_cli_fails_closed_and_template_mode_is_explicit(self) -> None:
        def invoke(*arguments: str) -> tuple[int, dict]:
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    ["validate_chi_dress_rehearsal.py", *arguments],
                ),
                contextlib.redirect_stdout(stdout),
            ):
                return validator_main(), json.loads(stdout.getvalue())

        default_code, default_payload = invoke()
        self.assertEqual(default_code, 3)
        self.assertFalse(
            default_payload["practical_threshold_contract_machine_gate_passed"]
        )
        template_code, template_payload = invoke("--allow-pending-template")
        self.assertEqual(template_code, 0)
        self.assertFalse(
            template_payload["practical_threshold_contract_machine_gate_passed"]
        )


if __name__ == "__main__":
    unittest.main()
