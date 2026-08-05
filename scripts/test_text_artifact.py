"""CPU-only tests for the provenance-complete text artifact contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.cognition.text_artifact import (
    ArtifactValidationError,
    TextDifficultyArtifact,
    load_text_difficulty_artifact,
    sha256_file,
)


def _payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "standardized_ridge_text_difficulty",
        "artifact_id": "lexigaze-en-text-difficulty-m1-v1",
        "language": "en",
        "feature_schema": ["log_char_length", "causal_surprisal"],
        "feature_policy": {
            "context_direction": "left_only",
            "language_model_frozen": True,
            "language_model_fine_tuned": False,
        },
        "model": {
            "family": "standardized_ridge",
            "feature_mean": [2.0, 5.0],
            "feature_scale": [1.0, 2.0],
            "coefficients": [0.2, 0.4],
            "intercept": 5.0,
            "alpha": 1.0,
        },
        "score_calibration": {
            "method": "fixed_training_prediction_quantile_clip",
            "raw_lower": 4.0,
            "raw_upper": 6.0,
            "request_local_normalization": False,
        },
        "training_distribution": {
            "feature_lower": [1.0, 1.0],
            "feature_upper": [4.0, 10.0],
        },
        "training": {
            "corpora": ["PROVO", "GECO_L2_English"],
            "excluded_corpora": ["OneStop_Eye_Movements"],
        },
        "leakage_controls": {
            "question_answer_dataset_used": False,
            "onestop_used_for_training_selection_or_thresholding": False,
            "language_model_fine_tuned": False,
            "request_local_normalization_used": False,
        },
        "promotion": {
            "status": "candidate",
            "independent_fusion_evaluation_required": True,
        },
    }


class TextArtifactTests(unittest.TestCase):
    def test_fixed_calibration_is_stable_when_an_extreme_row_is_appended(self) -> None:
        artifact = TextDifficultyArtifact.from_mapping(_payload())
        row = np.array([[2.0, 5.0]])
        alone = artifact.predict_matrix(row)
        appended = artifact.predict_matrix(np.vstack([row, [20.0, 80.0]]))

        self.assertAlmostEqual(float(alone.score[0]), float(appended.score[0]))
        self.assertTrue(bool(alone.in_distribution[0]))
        self.assertFalse(bool(appended.in_distribution[1]))
        self.assertEqual(int(appended.out_of_bounds_feature_count[1]), 2)

    def test_entropy_feature_is_rejected(self) -> None:
        payload = _payload()
        payload["feature_schema"].append("renyi_entropy")
        payload["model"]["feature_mean"].append(0.0)
        payload["model"]["feature_scale"].append(1.0)
        payload["model"]["coefficients"].append(0.0)
        payload["training_distribution"]["feature_lower"].append(0.0)
        payload["training_distribution"]["feature_upper"].append(1.0)

        with self.assertRaisesRegex(ArtifactValidationError, "entropy"):
            TextDifficultyArtifact.from_mapping(payload)

    def test_manifest_detects_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "artifact.json"
            manifest_path = root / "artifact.manifest.json"
            artifact_path.write_text(
                json.dumps(_payload()),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_id": "lexigaze-en-text-difficulty-m1-v1",
                        "artifact": {
                            "file": artifact_path.name,
                            "sha256": sha256_file(artifact_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_text_difficulty_artifact(
                artifact_path,
                manifest_path=manifest_path,
            )
            self.assertEqual(loaded.artifact_id, _payload()["artifact_id"])

            artifact_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ArtifactValidationError):
                load_text_difficulty_artifact(
                    artifact_path,
                    manifest_path=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
