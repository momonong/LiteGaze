"""Offline regressions for stable cognitive-score calibration and policy."""

from __future__ import annotations

import copy
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.cognition.pipeline import CognitiveLoadPipeline


def _word(surprisal: float, *, pos_score: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        surprisal=surprisal,
        entropy=2.0,
        renyi_entropy=3.0,
        aoa_score=0.4,
        word_length=6,
        zipf_score=4.5,
        pos_score=pos_score,
        dependency_load=0.1,
        load_score=0.0,
    )


class _FakeDMatrix:
    def __init__(self, values: np.ndarray, *, feature_names: list[str]):
        self.values = np.asarray(values, dtype=float)
        self.feature_names = feature_names


class _FakeBooster:
    def predict(self, matrix: _FakeDMatrix) -> np.ndarray:
        # Deterministic log-ms prediction based only on each row.
        return 5.2 + 0.03 * matrix.values[:, 0]


class CognitiveScoreStabilityTests(unittest.TestCase):
    def test_xgb_prefix_scores_do_not_depend_on_appended_extreme(self) -> None:
        pipeline = object.__new__(CognitiveLoadPipeline)
        pipeline._xgb_model = _FakeBooster()
        base = [_word(2.0), _word(6.0), _word(9.0)]
        extended = copy.deepcopy(base) + [_word(100.0)]
        fake_xgboost = types.ModuleType("xgboost")
        fake_xgboost.DMatrix = _FakeDMatrix

        with patch.dict(sys.modules, {"xgboost": fake_xgboost}):
            pipeline._apply_xgb(base)
            pipeline._apply_xgb(extended)

        self.assertEqual(
            [word.load_score for word in base],
            [word.load_score for word in extended[: len(base)]],
        )

    def test_ridge_prefix_scores_do_not_depend_on_appended_extreme(self) -> None:
        pipeline = object.__new__(CognitiveLoadPipeline)
        pipeline._ridge_mu = np.zeros(6)
        pipeline._ridge_std = np.ones(6)
        pipeline._ridge_coef = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pipeline._ridge_icept = 100.0
        base = [_word(2.0), _word(6.0), _word(9.0)]
        extended = copy.deepcopy(base) + [_word(100.0)]

        pipeline._apply_ridge(base)
        pipeline._apply_ridge(extended)

        self.assertEqual(
            [word.load_score for word in base],
            [word.load_score for word in extended[: len(base)]],
        )

if __name__ == "__main__":
    unittest.main()
