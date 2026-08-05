"""CPU-only tests for leakage-resistant gaze calibration regression."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.gaze_core.calibration_regression import (
    MOTION_FEATURE_NAMES,
    face_geometry_from_bbox,
    fit_best_stage,
    fit_standardized_ridge,
    motion_challenger_decision,
    motion_conditioned_features,
    standardized_design,
)
from core.gaze_core.model_registry import list_models


class CalibrationRegressionTests(unittest.TestCase):
    def test_model_registry_prefers_held_out_metric_but_keeps_legacy_compatibility(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lexigaze-model-registry-") as name:
            root = Path(name)
            models_dir = root / "examples" / "models"
            models_dir.mkdir(parents=True)
            (models_dir / "legacy.json").write_text(
                json.dumps({"mean_px_error": 12.0, "train_samples": 13}),
                encoding="utf-8",
            )
            (models_dir / "grouped.json").write_text(
                json.dumps(
                    {
                        "mean_px_error": 8.0,
                        "validation_px_error": 42.0,
                        "validation_scheme": "leave_one_motion_block_out",
                        "train_samples": 65,
                    }
                ),
                encoding="utf-8",
            )

            models = {model["name"]: model for model in list_models(root)}

        self.assertEqual(models["legacy"]["validation_px_error"], 12.0)
        self.assertEqual(models["legacy"]["validation_scheme"], "legacy_train_error")
        self.assertIn("legacy train 12.0 px", models["legacy"]["display_name"])
        self.assertEqual(models["grouped"]["validation_px_error"], 42.0)
        self.assertIn("held-out 42.0 px", models["grouped"]["display_name"])

    def test_motion_challenger_must_clear_both_promotion_margins(self) -> None:
        selected, required, observed = motion_challenger_decision(100.0, 94.0)
        self.assertTrue(selected)
        self.assertEqual(required, 5.0)
        self.assertEqual(observed, 6.0)

        selected, required, observed = motion_challenger_decision(200.0, 191.0)
        self.assertFalse(selected)
        self.assertEqual(required, 10.0)
        self.assertEqual(observed, 9.0)

        self.assertEqual(
            motion_challenger_decision(0.0, 0.0),
            (False, 5.0, 0.0),
        )

    def test_face_geometry_uses_normalized_box(self) -> None:
        geometry = face_geometry_from_bbox(
            {"x_norm": 0.2, "y_norm": 0.3, "w_norm": 0.4, "h_norm": 0.9}
        )
        self.assertEqual(geometry[:2], (0.4, 0.75))
        self.assertAlmostEqual(geometry[2], 0.6)

    def test_motion_conditioned_ridge_uses_frozen_schema(self) -> None:
        sample_index = np.arange(50, dtype=float)
        gaze = np.column_stack(
            [
                np.sin(sample_index * 0.21) * 0.2,
                np.cos(sample_index * 0.17) * 0.3,
            ]
        )
        head = np.column_stack(
            [
                np.sin(sample_index * 0.11) * 0.25,
                np.cos(sample_index * 0.13) * 0.35,
            ]
        )
        face = np.column_stack(
            [
                0.5 + np.sin(sample_index * 0.07) * 0.1,
                0.5 + np.cos(sample_index * 0.09) * 0.1,
                0.6 + np.sin(sample_index * 0.05) * 0.08,
            ]
        )
        features = motion_conditioned_features(gaze, head, face)
        targets = np.column_stack(
            [
                0.8 * features[:, 0] + 0.2 * features[:, 2],
                -0.7 * features[:, 1] + 0.15 * features[:, 3],
            ]
        )
        viewports = [[100.0, 100.0]] * len(features)
        groups = [f"block-{index % 5}" for index in range(len(features))]

        weights, mean, scale, alpha, validation_error = fit_standardized_ridge(
            features,
            targets,
            viewports,
            validation_groups=groups,
        )
        predictions = standardized_design(features, mean, scale) @ weights

        self.assertEqual(features.shape[1], len(MOTION_FEATURE_NAMES))
        self.assertEqual(weights.shape, (len(MOTION_FEATURE_NAMES) + 1, 2))
        self.assertIn(alpha, {1e-3, 1e-2, 0.1, 1.0, 10.0})
        self.assertLess(validation_error, 1.0)
        np.testing.assert_allclose(predictions, targets, atol=0.02)

    def test_linear_mapping_has_low_held_out_error(self) -> None:
        pitch = np.linspace(-0.3, 0.3, 8)
        yaw = np.linspace(-0.5, 0.5, 8)
        inputs = np.column_stack([pitch, yaw])
        targets = np.column_stack([1.2 * yaw + 0.1, -0.8 * pitch - 0.05])
        viewports = [[1920.0, 1080.0]] * len(inputs)

        weights, degree, alpha, validation_error = fit_best_stage(
            inputs,
            targets,
            viewports,
            unique_targets=2,
        )

        self.assertEqual(weights.shape, (3, 2))
        self.assertEqual(degree, 1)
        self.assertIn(alpha, {1e-4, 1e-3, 1e-2, 0.1})
        self.assertLess(validation_error, 1.0)

    def test_group_holdout_exposes_duplicate_frame_leakage(self) -> None:
        inputs = np.array(
            [
                [0.0, -1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        targets = np.array(
            [
                [-1.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        viewports = [[100.0, 100.0]] * len(inputs)

        *_, sample_error = fit_best_stage(
            inputs,
            targets,
            viewports,
            unique_targets=2,
        )
        *_, group_error = fit_best_stage(
            inputs,
            targets,
            viewports,
            unique_targets=2,
            validation_groups=["left", "left", "right", "right"],
        )

        self.assertLess(sample_error, 1.0)
        self.assertGreater(group_error, 80.0)

    def test_group_validation_rejects_misaligned_or_single_group(self) -> None:
        inputs = np.array([[0.0, -1.0], [0.0, 1.0]])
        targets = np.array([[-1.0, 0.0], [1.0, 0.0]])
        viewports = [[100.0, 100.0]] * 2

        with self.assertRaisesRegex(ValueError, "align"):
            fit_best_stage(
                inputs,
                targets,
                viewports,
                unique_targets=2,
                validation_groups=["left"],
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            fit_best_stage(
                inputs,
                targets,
                viewports,
                unique_targets=2,
                validation_groups=["same", "same"],
            )


if __name__ == "__main__":
    unittest.main()
