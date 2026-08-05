"""CPU-only tests for independent text/gaze fusion validation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.cognition.fusion_validation import (
    FusionValidationConfig,
    evaluate_fusion_candidate,
    prepare_fusion_evaluation_frame,
)


def _metadata() -> dict:
    return {
        "dataset_id": "synthetic-contract-test",
        "dataset_role": "independent_real_capture",
        "difficulty_target_source": "synthetic independent latent target",
        "question_answer_dataset_used": False,
        "difficulty_target_derived_from_gaze": False,
        "difficulty_target_derived_from_text_model": False,
        "fusion_parameters_frozen_before_outcomes": True,
    }


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260806)
    rows = []
    for capture_index in range(40):
        article_id = f"article-{capture_index % 20:02d}"
        for word_index in range(50):
            gaze_latent = rng.uniform(0.0, 1.0)
            text_latent = rng.uniform(0.0, 1.0)
            confidence = rng.uniform(0.65, 1.0)
            target = np.clip(
                0.50 * gaze_latent
                + 0.50 * text_latent
                + rng.normal(scale=0.015),
                0.0,
                1.0,
            )
            rows.append(
                {
                    "participant_id": f"participant-{capture_index:02d}",
                    "session_id": "session-01",
                    "device_id": f"device-{capture_index % 4}",
                    "article_id": article_id,
                    "word_id": str(word_index),
                    "difficulty_target": target,
                    "gaze_score": np.clip(
                        gaze_latent + rng.normal(scale=0.025), 0.0, 1.0
                    ),
                    "gaze_confidence": confidence,
                    "text_score": np.clip(
                        text_latent + rng.normal(scale=0.025), 0.0, 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


class ProvenanceFusionTests(unittest.TestCase):
    def test_combined_signal_passes_both_grouped_holdouts(self) -> None:
        summary, predictions = evaluate_fusion_candidate(
            _frame(),
            dataset_metadata=_metadata(),
            config=FusionValidationConfig(
                bootstrap_samples=2_000,
                minimum_groups=10,
            ),
        )

        self.assertTrue(summary["promotion"]["passed"])
        self.assertFalse(summary["promotion"]["production_model_changed"])
        self.assertEqual(set(predictions["holdout_axis"]), {"capture_group", "article"})
        for holdout in summary["holdouts"].values():
            self.assertEqual(holdout["cross_fit_diagnostics"]["group_overlap"], 0)
            self.assertTrue(holdout["gate"]["passed"])

    def test_qa_derived_contract_is_rejected(self) -> None:
        metadata = _metadata()
        metadata["question_answer_dataset_used"] = True
        with self.assertRaisesRegex(ValueError, "contract failed"):
            evaluate_fusion_candidate(
                _frame(),
                dataset_metadata=metadata,
                config=FusionValidationConfig(bootstrap_samples=100),
            )

    def test_duplicate_word_outcome_is_rejected(self) -> None:
        frame = _frame().iloc[:20].copy()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "one aggregated row"):
            prepare_fusion_evaluation_frame(frame)


if __name__ == "__main__":
    unittest.main()
