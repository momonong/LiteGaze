"""CPU-only contracts for personalized text-backbone matching."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from core.cognition.generalization import safe_spearman
from scripts.run_personalized_backbone_matching import (
    build_nested_masks,
    grouped_spearman,
    load_protocol,
    make_decision,
    select_with_abstention,
    stable_calibration_text_order,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "docs/experiments/protocols/2026-08-08-personalized-backbone-matching-feasibility-v1.json"
)


class PersonalizedBackboneMatchingTests(unittest.TestCase):
    def test_protocol_is_exploratory_cpu_only_and_source_restricted(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)

        self.assertEqual(
            protocol["claim_scope"]["study_role"],
            "exploratory_feasibility_only",
        )
        self.assertEqual(protocol["compute"]["feasibility_device"], "cpu")
        self.assertEqual(protocol["compute"]["feasibility_gpu_hours"], 0.0)
        self.assertFalse(protocol["dataset"]["question_answer_dataset_used"])
        model_ids = [
            item["model_id"]
            for item in (
                *protocol["frozen_backbones"],
                *protocol["conditional_expansion_bank"],
            )
        ]
        excluded = tuple(
            prefix.lower()
            for prefix in protocol["source_policy"]["excluded_model_id_prefixes"]
        )
        self.assertTrue(
            all(not model_id.lower().startswith(excluded) for model_id in model_ids)
        )

    def test_nested_masks_hold_out_target_participant_and_evaluation_texts(self) -> None:
        participants = np.repeat(["p0", "p1", "p2"], 10)
        texts = np.tile(np.repeat(["t0", "t1", "t2", "t3", "t4"], 2), 3)
        folds = np.tile(np.repeat(np.arange(5), 2), 3)

        masks, diagnostics = build_nested_masks(
            participants,
            texts,
            folds,
            participant="p0",
            outer_fold=2,
        )

        self.assertTrue(diagnostics["target_participant_rows_in_fit"] == 0)
        self.assertTrue(diagnostics["evaluation_texts_in_fit"] == 0)
        self.assertTrue(diagnostics["evaluation_texts_in_calibration"] == 0)
        self.assertTrue(np.all(participants[masks["fit"]] != "p0"))
        self.assertEqual(set(texts[masks["evaluation"]]), {"t2"})

    def test_grouped_spearman_matches_reference_with_ties(self) -> None:
        groups = np.repeat(["a", "b"], 6)
        prediction = np.array([1, 2, 2, 4, 5, 6, 6, 5, 4, 3, 2, 2], dtype=float)
        target = np.array([1, 1, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], dtype=float)

        observed = grouped_spearman(groups, prediction, target)

        self.assertAlmostEqual(
            observed["a"], safe_spearman(prediction[:6], target[:6])
        )
        self.assertAlmostEqual(
            observed["b"], safe_spearman(prediction[6:], target[6:])
        )

    def test_calibration_order_is_deterministic_and_identity_specific(self) -> None:
        texts = [f"text-{index}" for index in range(30)]
        first = stable_calibration_text_order(
            texts,
            participant="p0",
            outer_fold=1,
            seed=20260808,
        )
        repeated = stable_calibration_text_order(
            reversed(texts),
            participant="p0",
            outer_fold=1,
            seed=20260808,
        )
        another_participant = stable_calibration_text_order(
            texts,
            participant="p1",
            outer_fold=1,
            seed=20260808,
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, another_participant)
        self.assertEqual(set(first), set(texts))

    def test_selector_switches_only_with_positive_lower_bound(self) -> None:
        switched = select_with_abstention(
            [0.20, 0.18, 0.22, 0.19, 0.21],
            samples=2_000,
            seed=5,
        )
        uncertain = select_with_abstention(
            [-0.20, 0.20, -0.10, 0.10, 0.0],
            samples=2_000,
            seed=5,
        )

        self.assertEqual(switched["selected"], "pythia_410m_deduped_full")
        self.assertGreater(switched["ci_95_low"], 0.0)
        self.assertEqual(uncertain["selected"], "gpt2")
        self.assertLessEqual(uncertain["ci_95_low"], 0.0)

    def test_decision_requires_every_frozen_gate(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        passing_primary = {
            "selected_minus_gpt2_participant_bootstrap": {"ci_95_low": 0.001},
            "positive_outer_folds": 4,
            "fraction_participants_with_positive_selected_delta": 0.7,
            "challenger_selection": {"fraction": 0.1},
        }
        passing = make_decision(
            {"10": passing_primary}, {"passed": True}, protocol
        )
        failing = make_decision(
            {
                "10": {
                    **passing_primary,
                    "challenger_selection": {"fraction": 0.0},
                }
            },
            {"passed": True},
            protocol,
        )

        self.assertTrue(passing["conditional_model_bank_expansion_allowed"])
        self.assertFalse(failing["conditional_model_bank_expansion_allowed"])
        self.assertFalse(failing["production_model_changed"])


if __name__ == "__main__":
    unittest.main()
