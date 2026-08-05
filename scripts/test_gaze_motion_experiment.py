"""CPU-only tests for nested motion-shift candidate evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from core.gaze_core.motion_experiment import (
    BASELINE_MODEL,
    CHALLENGER_MODEL,
    VALIDATION_SCHEME,
    evaluate_motion_candidates,
)
from scripts.run_gaze_motion_experiment import _optional_float

TARGETS = np.asarray(
    [
        (-0.84, -0.80),
        (0.00, -0.80),
        (0.84, -0.80),
        (-0.84, 0.00),
        (0.00, 0.00),
        (0.84, 0.00),
        (-0.84, 0.80),
        (0.00, 0.80),
        (0.84, 0.80),
        (-0.42, -0.40),
        (0.42, -0.40),
        (-0.42, 0.40),
        (0.42, 0.40),
    ],
    dtype=np.float64,
)

BLOCKS = {
    "neutral": (0.04, 0.00, 0.50, 0.50, 0.58),
    "left": (0.04, -0.30, 0.47, 0.50, 0.58),
    "right": (0.04, 0.30, 0.53, 0.50, 0.58),
    "near": (0.10, 0.00, 0.50, 0.53, 0.78),
    "far": (-0.08, 0.00, 0.50, 0.47, 0.40),
}


def _synthetic_inputs(*, shifted: bool) -> tuple[np.ndarray, ...]:
    gaze_rows: list[tuple[float, float]] = []
    head_rows: list[tuple[float, float]] = []
    face_rows: list[tuple[float, float, float]] = []
    target_rows: list[tuple[float, float]] = []
    groups: list[str] = []
    for block, (head_pitch, head_yaw, face_x, face_y, scale) in BLOCKS.items():
        for target_x, target_y in TARGETS:
            if shifted:
                gaze_yaw = (
                    target_x - 0.90 * head_yaw - 0.15 * face_x
                ) / (1.10 + 0.50 * scale)
                gaze_pitch = (
                    target_y - 0.70 * head_pitch - 0.12 * face_y
                ) / (1.05 + 0.45 * scale)
            else:
                gaze_yaw = target_x / 1.40
                gaze_pitch = target_y / 1.20
            gaze_rows.append((gaze_pitch, gaze_yaw))
            head_rows.append((head_pitch, head_yaw))
            face_rows.append((face_x, face_y, scale))
            target_rows.append((target_x, target_y))
            groups.append(block)
    viewports = np.tile(np.asarray([[1920.0, 1080.0]]), (len(groups), 1))
    return (
        np.asarray(gaze_rows),
        np.asarray(head_rows),
        np.asarray(face_rows),
        np.asarray(target_rows),
        viewports,
        groups,
    )


class MotionExperimentTests(unittest.TestCase):
    def test_windows_gpu_telemetry_accepts_not_available_memory(self) -> None:
        self.assertIsNone(_optional_float("[N/A]"))
        self.assertEqual(_optional_float("0"), 0.0)

    def test_nested_outer_folds_hold_out_complete_motion_blocks(self) -> None:
        result = evaluate_motion_candidates(*_synthetic_inputs(shifted=True))

        self.assertEqual(result["validation_scheme"], VALIDATION_SCHEME)
        self.assertEqual(result["sample_count"], 65)
        self.assertEqual(result["outer_motion_block_count"], 5)
        self.assertEqual(len(result["folds"]), 5)
        self.assertEqual(
            {fold["outer_motion_block"] for fold in result["folds"]},
            set(BLOCKS),
        )
        for fold in result["folds"]:
            self.assertEqual(fold["train_motion_block_count"], 4)
            self.assertEqual(fold["train_samples"], 52)
            self.assertEqual(fold["validation_samples"], 13)

    def test_predeclared_shift_promotes_motion_conditioned_candidate(self) -> None:
        result = evaluate_motion_candidates(*_synthetic_inputs(shifted=True))
        gate = result["promotion_gate"]

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["selected_model"], CHALLENGER_MODEL)
        self.assertGreaterEqual(
            gate["observed_improvement_px"],
            gate["required_improvement_px"],
        )
        self.assertLess(
            result["candidates"][CHALLENGER_MODEL]["macro_mean_px"],
            result["candidates"][BASELINE_MODEL]["macro_mean_px"],
        )

    def test_no_shift_negative_control_keeps_gaze_baseline(self) -> None:
        result = evaluate_motion_candidates(*_synthetic_inputs(shifted=False))

        self.assertFalse(result["promotion_gate"]["passed"])
        self.assertEqual(
            result["promotion_gate"]["selected_model"],
            BASELINE_MODEL,
        )

    def test_nested_validation_rejects_too_few_groups(self) -> None:
        arrays = list(_synthetic_inputs(shifted=True))
        arrays[-1] = ["one" if index % 2 else "two" for index in range(65)]

        with self.assertRaisesRegex(ValueError, "at least three"):
            evaluate_motion_candidates(*arrays)


if __name__ == "__main__":
    unittest.main()
