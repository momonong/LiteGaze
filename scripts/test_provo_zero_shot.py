import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_provo import verify_provo_file
from scripts.evaluate_provo_zero_shot import (
    FEATURE_COLUMNS,
    add_frozen_features,
    fit_geco_models,
    fit_logistic,
    predict_logistic,
    score_provo,
    validate_provo_frame,
    word_features,
)


class TestProvoZeroShot(unittest.TestCase):
    def test_official_file_verifier_checks_size_and_hash(self):
        payload = b"official-provo-fixture\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "provo.csv"
            path.write_bytes(payload)
            fingerprint = verify_provo_file(
                path,
                expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(fingerprint["size_bytes"], len(payload))
            self.assertEqual(
                fingerprint["sha256"], hashlib.sha256(payload).hexdigest()
            )
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_provo_file(
                    path,
                    expected_size=len(payload) + 1,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                )

    def test_word_features_apply_frozen_unicode_rules(self):
        log_length, frequency, char_length, upper, punctuation, lookup = word_features(
            "  “Hello,”  "
        )

        self.assertAlmostEqual(log_length, np.log1p(5))
        self.assertGreater(frequency, 0)
        self.assertEqual(char_length, 5)
        self.assertEqual(upper, 1.0)
        self.assertEqual(punctuation, 1.0)
        self.assertEqual(lookup, "hello")

    def test_relative_position_is_dense_within_each_subject_trial(self):
        frame = pd.DataFrame(
            {
                "subject": ["s1", "s1", "s1", "s2", "s2"],
                "trial": ["t1", "t1", "t1", "t2", "t2"],
                "word_id": ["a", "b", "c", "a", "b"],
                "word": ["One", "two", "three.", "Four", "five"],
                "word_position": [2, 4, 9, 10, 20],
            }
        )

        featured = add_frozen_features(frame)

        np.testing.assert_allclose(
            featured["relative_position"], [0.0, 0.5, 1.0, 0.0, 1.0]
        )
        self.assertTrue(
            np.isfinite(featured.loc[:, FEATURE_COLUMNS].to_numpy()).all()
        )

    def test_provo_schema_uses_complete_interest_area_fields(self):
        raw = pd.DataFrame(
            {
                "Participant_ID": ["Sub01", "Sub01"],
                "Word_Unique_ID": ["NA", "QID2"],
                "Text_ID": [1, 1],
                "Word_Number": ["NA", 3],
                "Word": ["NA", "world"],
                "Word_Cleaned": ["NA", "world"],
                "Word_Length": ["NA", 5],
                "IA_ID": [1, 2],
                "IA_LABEL": ["Hello", "world"],
                "TRIAL_INDEX": [7, 7],
                "IA_DWELL_TIME": [120, 90],
                "IA_SKIP": [0, 1],
            }
        )

        validated = validate_provo_frame(raw)

        self.assertEqual(validated["word"].tolist(), ["Hello", "world"])
        self.assertEqual(validated["word_id"].tolist(), ["1", "2"])
        self.assertEqual(validated["fixated"].tolist(), [True, True])
        self.assertEqual(validated["first_pass_skipped"].tolist(), [False, True])

    def test_fixed_logistic_uses_train_statistics_and_converges(self):
        features = np.array(
            [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]], dtype=float
        )
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)

        model = fit_logistic(features, labels, alpha=1.0)
        probabilities = predict_logistic(model, np.array([[-10.0], [10.0]]))

        self.assertTrue(model["converged"])
        self.assertAlmostEqual(float(model["mean"][0]), 0.0)
        self.assertLess(probabilities[0], 0.5)
        self.assertGreater(probabilities[1], 0.5)

    def test_provo_outcomes_cannot_change_frozen_predictions(self):
        training = pd.DataFrame(
            {
                "log_char_length": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                "zipf_frequency": [7.0, 6.5, 6.0, 5.5, 5.0, 4.5],
                "relative_position": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "starts_upper": [1, 0, 0, 0, 0, 0],
                "ends_punctuation": [0, 0, 0, 0, 0, 1],
                "char_length": [1, 2, 3, 4, 5, 6],
                "fixated": [False, True, True, True, True, False],
                "log_reading_time": [np.nan, 4.0, 4.2, 4.5, 4.8, np.nan],
            }
        )
        models = fit_geco_models(training)
        test = training.copy()
        test["reading_time"] = [np.nan, 100, 120, 150, 180, np.nan]

        first = score_provo(test, models)
        test["reading_time"] = [np.nan, 10_000, 1, 99_999, 3, np.nan]
        test["log_reading_time"] = np.where(
            test["fixated"], np.log1p(test["reading_time"]), np.nan
        )
        second = score_provo(test, models)

        np.testing.assert_allclose(
            first["geco_lexical_ridge"], second["geco_lexical_ridge"]
        )
        np.testing.assert_allclose(
            first["geco_fixation_logistic"], second["geco_fixation_logistic"]
        )


if __name__ == "__main__":
    unittest.main()
