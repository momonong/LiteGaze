"""Focused CPU-only tests for CHI selective word-review fusion evaluation."""

from __future__ import annotations

import math
import sys
import unittest

import numpy as np

from core.cognition.selective_fusion_evaluation import (
    CLASS_LABELS,
    F0_MODEL_ID,
    F1_MODEL_ID,
    F2_MODEL_ID,
    FIXED_COVERAGES,
    MIN_INFERENTIAL_DIAGNOSTIC_BOOTSTRAP_RESAMPLES,
    MIN_INFERENTIAL_DIAGNOSTIC_PARTICIPANTS,
    MIN_INFERENTIAL_DIAGNOSTIC_PASSAGE_FAMILIES,
    STATISTICAL_REVIEW_BOUNDARY,
    check_exact_f1_fallback,
    crossed_cluster_multiplier_bootstrap_difference,
    crossed_cluster_multiplier_weights,
    deterministic_label_shuffle,
    evaluate_selective_fusion,
    fixed_coverage_risk_curve,
    label_shuffle_sentinel_metrics,
    probability_metrics,
    validate_class_probabilities,
)


def _probability_rows() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    labels = [
        "no_review",
        "unsure",
        "review_needed",
        "no_review",
        "unsure",
        "review_needed",
        "no_review",
        "unsure",
    ]
    f2 = np.asarray(
        [
            [0.88, 0.08, 0.04],
            [0.08, 0.85, 0.07],
            [0.03, 0.07, 0.90],
            [0.82, 0.12, 0.06],
            [0.10, 0.81, 0.09],
            [0.04, 0.10, 0.86],
            [0.80, 0.15, 0.05],
            [0.12, 0.78, 0.10],
        ],
        dtype=np.float64,
    )
    f1 = np.full((len(labels), 3), 1.0 / 3.0, dtype=np.float64)
    f0 = np.tile(np.asarray([0.55, 0.30, 0.15]), (len(labels), 1))
    return f2, f1, f0, labels


def _eight_by_eight_rows() -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[str],
]:
    f2_base, f1_base, _f0, labels_base = _probability_rows()
    return (
        np.tile(f2_base, (8, 1)),
        np.tile(f1_base, (8, 1)),
        labels_base * 8,
        [f"p{index}" for index in range(8) for _ in range(8)],
        [f"passage-{index}" for _ in range(8) for index in range(8)],
    )


class ProbabilityMetricTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_losses(self) -> None:
        probabilities = np.eye(3, dtype=np.float64)
        metrics = probability_metrics(CLASS_LABELS, probabilities)

        self.assertAlmostEqual(metrics["negative_log_likelihood"], 0.0)
        self.assertAlmostEqual(metrics["multiclass_brier_score"], 0.0)
        self.assertAlmostEqual(metrics["ranked_probability_score"], 0.0)
        self.assertEqual(metrics["class_order"], list(CLASS_LABELS))

    def test_uniform_probabilities_match_closed_form_scores(self) -> None:
        probabilities = np.full((3, 3), 1.0 / 3.0)
        metrics = probability_metrics(CLASS_LABELS, probabilities)

        self.assertAlmostEqual(metrics["negative_log_likelihood"], math.log(3.0))
        self.assertAlmostEqual(metrics["multiclass_brier_score"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["ranked_probability_score"], 2.0 / 9.0)
        self.assertEqual(
            metrics["ranked_probability_score_definition"],
            "mean_squared_cumulative_probability_error_over_K_minus_1",
        )

    def test_probability_validation_rejects_malformed_rows(self) -> None:
        invalid = (
            [[0.5, 0.5]],
            [[0.5, 0.6, -0.1]],
            [[0.2, 0.2, 0.2]],
            [[0.2, float("nan"), 0.8]],
            [[True, False, False]],
        )
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    validate_class_probabilities(probabilities)

    def test_unknown_or_mixed_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown classes"):
            probability_metrics(["easy"], [[1.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "all frozen class names"):
            probability_metrics(["no_review", 1], [[1, 0, 0], [0, 1, 0]])


class CoverageAndFallbackTests(unittest.TestCase):
    def test_fixed_coverage_uses_label_free_ranking_and_exact_grid(self) -> None:
        labels = ["no_review"] * 10
        f2 = np.asarray(
            [[0.95 - index * 0.04, 0.03 + index * 0.02, 0.02 + index * 0.02]
             for index in range(10)],
            dtype=np.float64,
        )
        f1 = np.full((10, 3), 1.0 / 3.0, dtype=np.float64)
        f0 = np.tile(np.asarray([0.45, 0.35, 0.20]), (10, 1))
        gaze_eligible = np.asarray([True] * 8 + [False] * 2)
        f2[~gaze_eligible] = f1[~gaze_eligible]
        uncertainty = np.asarray([0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        sample_ids = ["sample-b", "sample-a", *[f"sample-{i}" for i in range(2, 10)]]
        curve = fixed_coverage_risk_curve(
            labels,
            f2,
            uncertainty,
            sample_ids=sample_ids,
            gaze_eligible=gaze_eligible,
            f1_probabilities=f1,
            f0_probabilities=f0,
        )

        self.assertEqual(
            [item["requested_eligible_gaze_coverage"] for item in curve],
            list(FIXED_COVERAGES),
        )
        self.assertEqual(
            [item["selected_eligible_gaze_count"] for item in curve],
            [8, 7, 5, 4, 2],
        )
        self.assertEqual(
            [item["overall_evaluation_row_count"] for item in curve],
            [10] * len(FIXED_COVERAGES),
        )
        self.assertEqual(
            [item["total_rows_falling_back_to_f1"] for item in curve],
            [2, 3, 5, 6, 8],
        )
        self.assertEqual(
            curve[-1]["selected_eligible_sample_ids"], ["sample-a", "sample-b"]
        )
        self.assertEqual(
            curve[-1]["deployed_hybrid_all_row_metrics"][F2_MODEL_ID]["row_count"],
            10,
        )
        self.assertEqual(
            set(curve[-1]["deployed_hybrid_all_row_metric_differences"]),
            {"F2_minus_F1", "F2_minus_F0"},
        )
        expected_hybrid = f1.copy()
        expected_hybrid[[0, 1]] = f2[[0, 1]]
        self.assertAlmostEqual(
            curve[-1]["deployed_hybrid_all_row_metrics"][F2_MODEL_ID][
                "negative_log_likelihood"
            ],
            probability_metrics(labels, expected_hybrid)[
                "negative_log_likelihood"
            ],
        )
        self.assertLessEqual(
            curve[-1]["conditional_selected_eligible_metrics"][F2_MODEL_ID][
                "negative_log_likelihood"
            ],
            curve[0]["conditional_selected_eligible_metrics"][F2_MODEL_ID][
                "negative_log_likelihood"
            ],
        )

    def test_missing_or_ineligible_gaze_requires_bit_exact_f1_fallback(self) -> None:
        f2, f1, _f0, _labels = _probability_rows()
        available = np.asarray([False, True, True, True, True, True, True, True])
        eligible = np.asarray([False, False, True, True, True, True, True, True])
        f2[:2] = f1[:2]
        sample_ids = [f"sample-{index}" for index in range(len(f2))]

        result = check_exact_f1_fallback(
            f2,
            f1,
            gaze_available=available,
            gaze_eligible=eligible,
            sample_ids=sample_ids,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["fallback_row_count"], 2)

        f2[1, 0] += 1e-12
        f2[1, 1] -= 1e-12
        with self.assertRaisesRegex(ValueError, "exact F1"):
            check_exact_f1_fallback(
                f2,
                f1,
                gaze_available=available,
                gaze_eligible=eligible,
                sample_ids=sample_ids,
            )

    def test_unavailable_gaze_cannot_be_marked_eligible(self) -> None:
        baseline = np.asarray([[0.5, 0.3, 0.2]])
        with self.assertRaisesRegex(ValueError, "cannot be eligible"):
            check_exact_f1_fallback(
                baseline,
                baseline,
                gaze_available=[False],
                gaze_eligible=[True],
                sample_ids=["sample-1"],
            )


class CrossedClusterBootstrapTests(unittest.TestCase):
    def test_component_weights_are_shared_and_deterministic(self) -> None:
        participants = ["p1", "p1", "p2", "p2"]
        passages = ["a", "b", "a", "b"]
        first = crossed_cluster_multiplier_weights(
            participants, passages, seed=17, replicate=3
        )
        second = crossed_cluster_multiplier_weights(
            participants, passages, seed=17, replicate=3
        )

        np.testing.assert_array_equal(first.row_weights, second.row_weights)
        self.assertEqual(first.participant_weights, second.participant_weights)
        self.assertEqual(first.passage_family_weights, second.passage_family_weights)
        for index, (participant, passage) in enumerate(zip(participants, passages)):
            self.assertAlmostEqual(
                first.row_weights[index],
                first.participant_weights[participant]
                * first.passage_family_weights[passage],
            )

    def test_eight_by_eight_at_minimum_emits_deterministic_diagnostic(self) -> None:
        f2, f1, labels, participants, passages = _eight_by_eight_rows()
        kwargs = {
            "model_id": F2_MODEL_ID,
            "reference_id": F1_MODEL_ID,
            "participant_ids": participants,
            "passage_family_ids": passages,
            "resamples": MIN_INFERENTIAL_DIAGNOSTIC_BOOTSTRAP_RESAMPLES,
            "seed": 20260817,
        }
        first = crossed_cluster_multiplier_bootstrap_difference(
            labels, f2, f1, **kwargs
        )
        second = crossed_cluster_multiplier_bootstrap_difference(
            labels, f2, f1, **kwargs
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first["participant_cluster_count"],
            MIN_INFERENTIAL_DIAGNOSTIC_PARTICIPANTS,
        )
        self.assertEqual(
            first["passage_family_cluster_count"],
            MIN_INFERENTIAL_DIAGNOSTIC_PASSAGE_FAMILIES,
        )
        self.assertEqual(first["observed_crossed_cell_count"], 64)
        self.assertEqual(
            first["inference_evaluability"]["status"],
            "inferential_diagnostic_available",
        )
        self.assertTrue(
            first["inference_evaluability"][
                "inferential_diagnostic_available"
            ]
        )
        self.assertTrue(
            first["inference_evaluability"][
                "bootstrap_resample_minimum_met"
            ]
        )
        self.assertFalse(
            first["inference_evaluability"][
                "sample_size_sufficiency_established"
            ]
        )
        self.assertEqual(first["claim_boundary"], STATISTICAL_REVIEW_BOUNDARY)
        self.assertTrue(all(first["group_integrity"].values()))
        for metric in first["metrics"].values():
            self.assertLess(
                metric["point_difference_model_minus_reference"], 0.0
            )
            self.assertLess(metric["ci95_high"], 0.0)

    def test_eight_by_eight_with_one_resample_withholds_diagnostic(self) -> None:
        f2, f1, labels, participants, passages = _eight_by_eight_rows()
        result = crossed_cluster_multiplier_bootstrap_difference(
            labels,
            f2,
            f1,
            model_id=F2_MODEL_ID,
            reference_id=F1_MODEL_ID,
            participant_ids=participants,
            passage_family_ids=passages,
            resamples=1,
        )

        evaluability = result["inference_evaluability"]
        self.assertTrue(evaluability["cluster_structure_minima_met"])
        self.assertFalse(evaluability["bootstrap_resample_minimum_met"])
        self.assertFalse(evaluability["inferential_diagnostic_available"])
        self.assertEqual(
            evaluability["status"], "inferential_diagnostic_unavailable"
        )
        for metric in result["metrics"].values():
            self.assertIsNone(metric["ci95_low"])
            self.assertIsNone(metric["ci95_high"])
            self.assertIsNone(metric["bootstrap_probability_model_better"])

    def test_two_by_two_bootstrap_is_descriptive_and_withholds_intervals(self) -> None:
        f2, f1, _f0, labels = _probability_rows()
        result = crossed_cluster_multiplier_bootstrap_difference(
            labels,
            f2,
            f1,
            model_id=F2_MODEL_ID,
            reference_id=F1_MODEL_ID,
            participant_ids=["p1"] * 4 + ["p2"] * 4,
            passage_family_ids=["a", "a", "b", "b"] * 2,
            resamples=MIN_INFERENTIAL_DIAGNOSTIC_BOOTSTRAP_RESAMPLES,
        )

        evaluability = result["inference_evaluability"]
        self.assertFalse(evaluability["cluster_structure_minima_met"])
        self.assertTrue(evaluability["bootstrap_resample_minimum_met"])
        self.assertFalse(evaluability["inferential_diagnostic_available"])
        self.assertTrue(
            evaluability["intervals_withheld"]
        )
        for metric in result["metrics"].values():
            self.assertLess(
                metric["point_difference_model_minus_reference"], 0.0
            )
            self.assertIsNone(metric["ci95_low"])
            self.assertIsNone(metric["ci95_high"])
            self.assertIsNone(metric["bootstrap_probability_model_better"])

    def test_bootstrap_refuses_a_single_participant_or_passage_family(self) -> None:
        f2, f1, _f0, labels = _probability_rows()
        with self.assertRaisesRegex(ValueError, "two participants"):
            crossed_cluster_multiplier_bootstrap_difference(
                labels,
                f2,
                f1,
                model_id=F2_MODEL_ID,
                reference_id=F1_MODEL_ID,
                participant_ids=["p1"] * len(labels),
                passage_family_ids=["a", "a", "b", "b"] * 2,
                resamples=10,
            )
        with self.assertRaisesRegex(ValueError, "two passage families"):
            crossed_cluster_multiplier_bootstrap_difference(
                labels,
                f2,
                f1,
                model_id=F2_MODEL_ID,
                reference_id=F1_MODEL_ID,
                participant_ids=["p1"] * 4 + ["p2"] * 4,
                passage_family_ids=["a"] * len(labels),
                resamples=10,
            )


class SentinelAndSurfaceTests(unittest.TestCase):
    def test_label_shuffle_is_deterministic_and_count_preserving(self) -> None:
        labels = ["no_review", "no_review", "unsure", "review_needed"]
        sample_ids = ["s3", "s1", "s4", "s2"]
        first = deterministic_label_shuffle(labels, sample_ids=sample_ids, seed=9)
        second = deterministic_label_shuffle(labels, sample_ids=sample_ids, seed=9)
        reverse = deterministic_label_shuffle(
            list(reversed(labels)), sample_ids=list(reversed(sample_ids)), seed=9
        )

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), sorted(labels))
        self.assertEqual(
            dict(zip(sample_ids, first)),
            dict(zip(reversed(sample_ids), reverse)),
        )

    def test_sentinel_reports_metrics_without_fitting(self) -> None:
        f2, _f1, _f0, labels = _probability_rows()
        result = label_shuffle_sentinel_metrics(
            labels,
            f2,
            sample_ids=[f"s{index}" for index in range(len(labels))],
            seed=22,
        )

        self.assertEqual(sum(result["class_counts"].values()), len(labels))
        self.assertEqual(result["metrics"]["row_count"], len(labels))

    def test_complete_evaluation_surface_is_cpu_only_and_fit_free(self) -> None:
        torch_was_loaded = "torch" in sys.modules
        f2, f1, f0, labels = _probability_rows()
        # The first two rows exercise exact fallback; the remaining rows can use gaze.
        f2[:2] = f1[:2]
        result = evaluate_selective_fusion(
            labels,
            f0,
            f1,
            f2,
            uncertainty_scores=np.linspace(0.0, 0.7, len(labels)),
            sample_ids=[f"sample-{index}" for index in range(len(labels))],
            participant_ids=["p1"] * 4 + ["p2"] * 4,
            passage_family_ids=["a", "a", "b", "b"] * 2,
            gaze_available=[False, True, True, True, True, True, True, True],
            gaze_eligible=[False, False, True, True, True, True, True, True],
            bootstrap_resamples=50,
            bootstrap_seed=20260817,
        )

        self.assertEqual(result["compute"], {
            "device": "cpu",
            "gpu_used": False,
            "model_fit": False,
        })
        self.assertFalse(result["threshold_selected"])
        self.assertFalse(result["production_model_changed"])
        self.assertEqual(len(result["coverage_risk"]), len(FIXED_COVERAGES))
        self.assertEqual(
            set(result["crossed_cluster_comparisons"]),
            {"F2_minus_F1", "F2_minus_F0"},
        )
        self.assertEqual(
            set(result["metric_differences"]),
            {"F2_minus_F1", "F2_minus_F0"},
        )
        self.assertFalse(
            result["inference_evaluability"][
                "inferential_diagnostic_available"
            ]
        )
        self.assertEqual(
            result["inference_evaluability"]["status"],
            "inferential_diagnostic_unavailable",
        )
        self.assertFalse(
            result["inference_evaluability"][
                "sample_size_sufficiency_established"
            ]
        )
        self.assertEqual(result["claim_boundary"], STATISTICAL_REVIEW_BOUNDARY)
        self.assertTrue(
            all(
                item["overall_evaluation_row_count"] == len(labels)
                for item in result["coverage_risk"]
            )
        )
        self.assertEqual("torch" in sys.modules, torch_was_loaded)


if __name__ == "__main__":
    unittest.main()
