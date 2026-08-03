import unittest

import numpy as np
import pandas as pd

from scripts.evaluate_geco_generalization import (
    add_population_priors,
    assign_balanced_folds,
    cross_fitted_double_holdout,
    fit_ridge,
    predict_ridge,
)


class TestGecoGeneralization(unittest.TestCase):
    def test_balanced_folds_are_deterministic_within_strata(self):
        values = [f"L1|pp{i:02d}" for i in range(1, 10)] + [
            f"L2|pp{i:02d}" for i in range(1, 11)
        ]
        strata = {value: value.split("|", 1)[0] for value in values}

        first = assign_balanced_folds(values, 5, strata=strata)
        second = assign_balanced_folds(reversed(values), 5, strata=strata)

        self.assertEqual(first, second)
        for lang in ("L1", "L2"):
            counts = [
                sum(first[value] == fold for value in values if value.startswith(lang))
                for fold in range(5)
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_ridge_uses_explicit_training_statistics(self):
        train_x = np.array([[0.0], [1.0], [2.0], [3.0]])
        train_y = np.array([1.0, 3.0, 5.0, 7.0])
        test_x = np.array([[100.0]])

        model = fit_ridge(train_x, train_y, alpha=0.0)
        prediction = predict_ridge(model, test_x)

        self.assertAlmostEqual(float(model["mean"][0]), 1.5)
        self.assertAlmostEqual(float(prediction[0]), 201.0, places=8)

    def test_population_priors_exclude_each_subjects_own_outcome(self):
        frame = pd.DataFrame(
            {
                "lang": ["L1", "L1", "L1"],
                "trial": ["trial_1"] * 3,
                "word_id": ["1"] * 3,
                "fixated": [True, True, False],
                "log_reading_time": [np.log1p(100), np.log1p(300), np.nan],
            }
        )

        result = add_population_priors(frame)

        self.assertAlmostEqual(result.loc[0, "population_log_duration"], np.log1p(300))
        self.assertAlmostEqual(result.loc[1, "population_log_duration"], np.log1p(100))
        self.assertAlmostEqual(result.loc[0, "population_fixation_rate"], 0.5)
        self.assertAlmostEqual(result.loc[2, "population_fixation_rate"], 1.0)

    def test_double_holdout_cross_fits_every_test_cell(self):
        rows = []
        for subject_fold in range(5):
            for trial_fold in range(5):
                for repetition in range(4):
                    surprisal = 1.0 + subject_fold + repetition * 0.1
                    attention = 2.0 + trial_fold + repetition * 0.2
                    word_length = 3.0 + repetition
                    is_l2 = float(subject_fold % 2)
                    log_target = (
                        0.7 * np.log1p(surprisal)
                        + 0.4 * np.log1p(attention)
                        + 0.05 * word_length
                        + 0.2 * is_l2
                    )
                    rows.append(
                        {
                            "fixated": True,
                            "subject_fold": subject_fold,
                            "trial_fold": trial_fold,
                            "log_surprisal": np.log1p(surprisal),
                            "log_attention": np.log1p(attention),
                            "word_length": word_length,
                            "is_l2": is_l2,
                            "log_reading_time": log_target,
                            "surprisal_score": surprisal,
                            "attention_score": attention,
                            "cognitive_mass": 0.5,
                        }
                    )

        predictions, fold_metrics = cross_fitted_double_holdout(
            pd.DataFrame(rows), alpha=0.0
        )

        self.assertEqual(len(fold_metrics), 25)
        self.assertTrue(np.isfinite(predictions["ridge_text_only"]).all())
        self.assertGreater(
            predictions["ridge_text_only"].corr(
                predictions["log_reading_time"], method="spearman"
            ),
            0.99,
        )


if __name__ == "__main__":
    unittest.main()
