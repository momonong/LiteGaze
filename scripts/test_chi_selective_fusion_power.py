"""Regression tests for the frozen CHI selective-fusion planning simulation."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from core.cognition.selective_fusion_evaluation import (
    F0_MODEL_ID,
    F1_MODEL_ID,
    F2_MODEL_ID,
    probability_metrics,
)
from scripts.run_chi_selective_fusion_power import (
    COMPARATOR_IDS,
    DEFAULT_PROTOCOL,
    _risk_coverage_rows,
    conservative_two_way_cluster_se,
    load_protocol,
    loss_vectors,
    run_simulation,
    validate_protocol,
)


class ChiSelectiveFusionContractTests(unittest.TestCase):
    def test_frozen_contract_has_complete_claim_and_holdout_boundaries(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        self.assertEqual(validate_protocol(protocol), [])
        self.assertEqual(
            tuple(item["id"] for item in protocol["comparators"]),
            COMPARATOR_IDS,
        )
        self.assertEqual(
            protocol["metrics"]["primary"],
            "mean_multiclass_negative_log_likelihood",
        )
        self.assertEqual(protocol["metrics"]["coverage_grid"], [1, 0.8, 0.6, 0.4, 0.2])
        self.assertEqual(
            protocol["split_policy"]["primary_holdout"],
            "joint_participant_passage_capture_device_group_confirmation",
        )
        self.assertIn(
            "passage_probe_id", protocol["split_policy"]["required_group_axes"]
        )
        self.assertNotIn("probe_id", protocol["split_policy"]["required_group_axes"])
        self.assertFalse(
            protocol["split_policy"]["bare_probe_id_split_key_allowed"]
        )
        self.assertEqual(
            [item["id"] for item in protocol["simulation"]["power_test"]["estimands"]],
            ["F2_minus_F1", "F2_minus_F0"],
        )
        self.assertEqual(
            (F0_MODEL_ID, F1_MODEL_ID, F2_MODEL_ID),
            (
                "F0_always_on_text_person_gaze",
                "F1_text_person",
                "F2_selective_exact_F1_fallback",
            ),
        )
        self.assertEqual(
            tuple(model_id.split("_", 1)[0] for model_id in (
                F0_MODEL_ID,
                F1_MODEL_ID,
                F2_MODEL_ID,
            )),
            tuple(item["id"] for item in protocol["comparators"][-3:]),
        )
        self.assertFalse(
            protocol["decision_policy"]["formal_recruitment_authorized"]
        )
        self.assertFalse(protocol["decision_policy"]["model_promotion_authorized"])

    def test_QA_cognitive_and_fatigue_inputs_are_rejected(self) -> None:
        protocol = copy.deepcopy(load_protocol(DEFAULT_PROTOCOL))
        protocol["simulation"]["question_answer_data_used"] = True
        protocol["claim_boundary"][
            "prohibited_training_selection_or_claim_sources"
        ].remove("cognitive_profile")
        failures = validate_protocol(protocol)
        self.assertIn("question_answer_data_used_must_be_false", failures)
        self.assertIn(
            "QA_cognitive_fatigue_proficiency_claims_prohibited", failures
        )

    def test_nonexact_fallback_contract_is_rejected(self) -> None:
        protocol = copy.deepcopy(load_protocol(DEFAULT_PROTOCOL))
        f2 = next(
            item for item in protocol["comparators"] if item["id"] == "F2"
        )
        f2["ineligible_gaze_behavior"] = "learned_imputation"
        self.assertIn("exact_F1_fallback", validate_protocol(protocol))


class ChiSelectiveFusionMetricTests(unittest.TestCase):
    def test_three_class_proper_scoring_rules_match_manual_values(self) -> None:
        probabilities = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype=np.float64
        )
        labels = np.array([0, 2], dtype=np.int64)
        losses = loss_vectors(probabilities, labels)
        np.testing.assert_allclose(losses["nll"], -np.log([0.8, 0.7]))
        np.testing.assert_allclose(losses["brier"], [0.06, 0.14])
        np.testing.assert_allclose(losses["rps"], [0.025, 0.05])

    def test_power_losses_match_core_evaluator_across_probability_shapes(
        self,
    ) -> None:
        probabilities = np.array(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.5, 0.3],
                [0.6, 0.1, 0.3],
            ],
            dtype=np.float64,
        )
        labels = np.array([2, 0, 1, 2, 1, 0], dtype=np.int64)
        power_losses = loss_vectors(probabilities, labels)
        evaluator = probability_metrics(labels, probabilities)
        expected_keys = {
            "nll": "negative_log_likelihood",
            "brier": "multiclass_brier_score",
            "rps": "ranked_probability_score",
        }
        for power_key, evaluator_key in expected_keys.items():
            self.assertAlmostEqual(
                float(np.mean(power_losses[power_key])), evaluator[evaluator_key]
            )

    def test_partial_coverage_uses_F1_fallback_for_all_unselected_rows(self) -> None:
        f0 = np.array([10.0, 20.0, 30.0, 40.0])
        f1 = np.array([1.0, 2.0, 3.0, 4.0])
        f2 = np.array([0.0, 0.0, 0.0, 4.0])
        losses = {
            metric: values
            for metric, values in (("nll", f0), ("brier", f0), ("rps", f0))
        }
        f1_losses = {metric: f1 for metric in losses}
        f2_losses = {metric: f2 for metric in losses}
        rows = _risk_coverage_rows(
            [0.4],
            np.array([0.9, 0.8, 0.7, 0.0]),
            losses,
            f1_losses,
            f2_losses,
            np.array([True, True, True, False]),
        )
        row = rows[0]
        self.assertEqual(row["selected_eligible_observations"], 2)
        self.assertAlmostEqual(row["realized_eligible_coverage"], 2 / 3)
        self.assertEqual(
            row["conditional_accepted_risk"]["nll"]["F2_minus_F1"], -1.5
        )
        system = row["system_hybrid_risk"]["nll"]
        self.assertEqual(system["F1_text_person"], 2.5)
        self.assertEqual(system["selective_hybrid"], 1.75)
        self.assertEqual(system["selective_hybrid_minus_F1"], -0.75)

    def test_crossed_cluster_standard_error_is_finite_and_nonnegative(self) -> None:
        values = np.array([-0.2, -0.1, 0.0, 0.1, -0.1, 0.2, 0.0, -0.2])
        participants = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        passages = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        standard_error = conservative_two_way_cluster_se(
            values, participants, passages
        )
        self.assertTrue(np.isfinite(standard_error))
        self.assertGreaterEqual(standard_error, 0.0)


class ChiSelectiveFusionSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(DEFAULT_PROTOCOL)
        cls.result = run_simulation(cls.protocol, replicates_override=2)

    def test_run_is_synthetic_CPU_only_and_never_authorizes_claims(self) -> None:
        result = self.result
        self.assertEqual(result["status"], "completed_planning_only")
        self.assertTrue(result["integrity"]["synthetic_data_only"])
        self.assertFalse(result["integrity"]["human_outcomes_used"])
        self.assertFalse(result["integrity"]["question_answer_data_used"])
        self.assertFalse(result["integrity"]["cognitive_profile_data_used"])
        self.assertFalse(result["integrity"]["gpu_used"])
        self.assertFalse(result["integrity"]["torch_imported"])
        self.assertFalse(result["integrity"]["truly_missing_gaze_simulated"])
        self.assertFalse(
            result["holdout_contract"]["partition_assignment_exercised_by_generator"]
        )
        self.assertFalse(
            result["holdout_contract"][
                "complete_capture_and_device_groups_held_out_by_construction"
            ]
        )
        self.assertFalse(result["decision"]["formal_recruitment_authorized"])
        self.assertFalse(result["decision"]["model_promotion_authorized"])
        self.assertIsNone(result["decision"]["recommended_enrollment"])

    def test_null_signal_is_exact_F1_and_all_fallback_cells_pass(self) -> None:
        self.assertTrue(self.result["integrity"]["all_cells_exact_F1_fallback"])
        self.assertTrue(self.result["integrity"]["null_sentinel_passed"])
        null_rows = [
            row
            for row in self.result["rows"]
            if row["effect_id"] == "no_added_gaze_signal"
        ]
        self.assertTrue(null_rows)
        for row in null_rows:
            for metric in ("nll", "brier", "rps"):
                for comparison in (
                    "F2_minus_F1",
                    "F2_minus_F0",
                    "F0_minus_F1",
                ):
                    self.assertEqual(
                        row["metrics"][metric]["comparisons"][comparison][
                            "mean_difference"
                        ],
                        0.0,
                    )

    def test_every_cell_retains_the_frozen_coverage_grid_and_secondary_metrics(
        self,
    ) -> None:
        expected_grid = self.protocol["metrics"]["coverage_grid"]
        for row in self.result["rows"]:
            self.assertEqual(
                [
                    item["target_eligible_coverage"]
                    for item in row["risk_coverage"]
                ],
                expected_grid,
            )
            self.assertEqual(
                set(row["metrics"]["nll"]["model_means"]), {"F0", "F1", "F2"}
            )
            for item in row["risk_coverage"]:
                self.assertIn("conditional_accepted_risk", item)
                self.assertIn("system_hybrid_risk", item)
                for metric in ("nll", "brier", "rps"):
                    self.assertIn(
                        "F2_minus_F1", item["conditional_accepted_risk"][metric]
                    )
                    self.assertIn(
                        "selective_hybrid_minus_F0",
                        item["system_hybrid_risk"][metric],
                    )

    def test_joint_power_requires_both_comparisons_in_same_replication(self) -> None:
        for row in self.result["rows"]:
            nll = row["metrics"]["nll"]
            joint = nll["joint_H1_intersection_union"][
                "joint_diagnostic_power_among_structurally_evaluable_replications"
            ]
            if joint is None:
                continue
            for comparison in ("F2_minus_F1", "F2_minus_F0"):
                marginal = nll["comparisons"][comparison][
                    "diagnostic_power_among_structurally_evaluable_replications"
                ]
                self.assertLessEqual(joint, marginal)

        threshold = self.protocol["simulation"]["power_test"][
            "diagnostic_power_threshold"
        ]
        evaluable_threshold = self.protocol["simulation"]["power_test"][
            "diagnostic_evaluable_probability_threshold"
        ]
        for diagnostic in self.result["planning_diagnostics"]:
            first_n = diagnostic[
                "first_candidate_n_meeting_planning_diagnostics"
            ]
            if first_n is None:
                continue
            row = next(
                item
                for item in self.result["rows"]
                if item["scenario_id"] == diagnostic["scenario_id"]
                and item["effect_id"] == diagnostic["effect_id"]
                and item["enrolled"] == first_n
            )
            self.assertGreaterEqual(
                row["structurally_evaluable_probability"], evaluable_threshold
            )
            self.assertGreaterEqual(
                row["metrics"]["nll"]["joint_H1_intersection_union"][
                    "joint_diagnostic_power_among_structurally_evaluable_replications"
                ],
                threshold,
            )

    def test_fixed_seed_reproduces_every_scientific_field(self) -> None:
        rerun = run_simulation(self.protocol, replicates_override=2)
        self.assertEqual(self.result, rerun)


if __name__ == "__main__":
    unittest.main()
