"""No-Torch tests for cognitive calibration and model-selection policy."""

from __future__ import annotations

import math
import sys
import unittest

from core.cognition.calibration import calibrate_reading_time_prediction
from core.cognition.model_policy import default_model_for_language


class CognitivePolicyTests(unittest.TestCase):
    def test_imports_remain_model_runtime_free(self) -> None:
        self.assertNotIn("core.cognition.pipeline", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_calibration_uses_frozen_training_bounds(self) -> None:
        self.assertEqual(
            calibrate_reading_time_prediction(math.log(50), log_space=True),
            0.0,
        )
        self.assertEqual(
            calibrate_reading_time_prediction(math.log(3000), log_space=True),
            1.0,
        )
        self.assertEqual(
            calibrate_reading_time_prediction(50, log_space=False),
            0.0,
        )
        self.assertEqual(
            calibrate_reading_time_prediction(3000, log_space=False),
            1.0,
        )

    def test_default_model_policy_matches_metric_evidence(self) -> None:
        self.assertEqual(default_model_for_language("en"), "gpt2")
        self.assertEqual(default_model_for_language("zh"), "bert")
        self.assertEqual(default_model_for_language(" NL "), "bert")
        with self.assertRaisesRegex(ValueError, "unsupported language"):
            default_model_for_language("fr")


if __name__ == "__main__":
    unittest.main()
