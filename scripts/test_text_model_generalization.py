"""CPU-only tests for grouped text-model evaluation utilities."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.cognition.generalization import (
    cross_fit_grouped_ridge,
    fit_standardized_ridge,
    fit_weighted_standardized_ridge,
    paired_bootstrap_mean_difference,
    predict_standardized_ridge,
    safe_spearman,
    stable_group_folds,
)


class TextModelGeneralizationTests(unittest.TestCase):
    def test_stable_group_folds_are_deterministic_and_balanced(self) -> None:
        groups = [f"text-{index}" for index in range(23)]
        first = stable_group_folds(groups, 5, seed=20260805)
        second = stable_group_folds(reversed(groups), 5, seed=20260805)

        self.assertEqual(first, second)
        counts = np.bincount(list(first.values()), minlength=5)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_ridge_uses_training_statistics_and_recovers_linear_signal(self) -> None:
        features = np.arange(30, dtype=float).reshape(-1, 1)
        target = 3.0 + 2.5 * features[:, 0]
        model = fit_standardized_ridge(features, target, alpha=0.0)
        predictions = predict_standardized_ridge(model, features)

        np.testing.assert_allclose(predictions, target, atol=1e-9)
        self.assertAlmostEqual(float(model.mean[0]), float(features.mean()))

    def test_weighted_ridge_balances_groups_without_duplicating_rows(self) -> None:
        features = np.array([[0.0], [1.0], [10.0]], dtype=float)
        target = np.array([0.0, 1.0, 10.0], dtype=float)
        weights = np.array([5.0, 5.0, 1.0], dtype=float)
        model = fit_weighted_standardized_ridge(
            features,
            target,
            weights,
            alpha=0.0,
        )
        predictions = predict_standardized_ridge(model, features)

        np.testing.assert_allclose(predictions, target, atol=1e-9)
        self.assertAlmostEqual(
            float(model.mean[0]),
            float(np.average(features[:, 0], weights=weights)),
        )

    def test_cross_fit_holds_out_complete_texts_and_shuffle_loses_signal(self) -> None:
        rng = np.random.default_rng(7)
        rows = []
        for text_index in range(25):
            for word_index in range(30):
                signal = rng.normal()
                rows.append(
                    {
                        "text": f"text-{text_index}",
                        "subject": f"subject-{word_index % 5}",
                        "signal": signal,
                        "noise": rng.normal(),
                        "target": 2.0 * signal + rng.normal(scale=0.1),
                    }
                )
        frame = pd.DataFrame(rows)

        predictions, diagnostics = cross_fit_grouped_ridge(
            frame,
            group_column="text",
            target_column="target",
            feature_sets={"baseline": ("noise",), "signal": ("noise", "signal")},
            n_folds=5,
            alpha=1.0,
            seed=20260805,
            shuffled_target_model="signal",
        )

        self.assertEqual(diagnostics["group_overlap"], 0)
        self.assertTrue(all(fold["group_overlap"] == 0 for fold in diagnostics["folds"]))
        self.assertFalse(predictions.filter(like="prediction_").isna().any().any())
        signal_rho = safe_spearman(predictions.prediction_signal, predictions.target)
        shuffled_rho = safe_spearman(
            predictions.prediction_target_shuffle_sentinel,
            predictions.target,
        )
        self.assertGreater(signal_rho, 0.95)
        self.assertLess(abs(shuffled_rho), 0.15)

    def test_paired_bootstrap_reports_direction(self) -> None:
        result = paired_bootstrap_mean_difference(
            [0.5, 0.6, 0.7, 0.8],
            [0.1, 0.2, 0.3, 0.4],
            samples=2_000,
            seed=4,
        )

        self.assertAlmostEqual(result["mean_difference"], 0.4)
        self.assertGreater(result["ci_95_low"], 0.0)
        self.assertEqual(result["n_pairs"], 4)


if __name__ == "__main__":
    unittest.main()
