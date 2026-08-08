"""CPU-only tests for the frozen GECO Pythia replication."""

from __future__ import annotations

import copy
import unittest

from core.cognition.causal_backbone import model_id_has_excluded_prefix
from scripts import run_geco_text_backbone_replication as replication


def _comparison(
    *,
    participant_mean: float,
    participant_low: float,
    text_mean: float,
    text_low: float,
    positive_folds: int,
) -> dict:
    return {
        "participant_bootstrap": {
            "mean_difference": participant_mean,
            "ci_95_low": participant_low,
            "ci_95_high": participant_mean + 0.01,
        },
        "text_bootstrap": {
            "mean_difference": text_mean,
            "ci_95_low": text_low,
            "ci_95_high": text_mean + 0.01,
        },
        "positive_outer_folds": positive_folds,
    }


class GecoTextBackboneReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol, cls.specs = replication.load_protocol(
            replication.DEFAULT_PROTOCOL_PATH
        )

    def test_protocol_contains_only_the_frozen_baseline_and_challenger(self) -> None:
        self.assertEqual(
            [spec.key for spec in self.specs],
            ["gpt2", "pythia_410m_deduped_full"],
        )
        excluded = self.protocol["source_policy"]["excluded_model_id_prefixes"]
        self.assertFalse(
            any(
                model_id_has_excluded_prefix(spec.model_id, excluded)
                for spec in self.specs
            )
        )
        self.assertEqual(
            self.protocol["feature_extraction"]["separator_only_token_policy"],
            "assign_to_following_word_if_exact_unicode_whitespace_gap",
        )
        self.assertFalse(
            self.protocol["feature_extraction"]["v1_feature_reuse_allowed"]
        )

    def test_label_free_loader_preserves_frozen_geco_identity(self) -> None:
        items, fingerprint = replication.load_label_free_geco_items(
            replication.DEFAULT_GECO_PATH,
            self.protocol,
        )

        self.assertEqual(len(items), 56411)
        self.assertEqual(fingerprint["participant_count"], 19)
        self.assertEqual(fingerprint["text_count"], 588)
        self.assertFalse(fingerprint["outcome_columns_read"])

    def test_strong_replication_requires_incremental_gate(self) -> None:
        outcome = {"incremental_gate_passed": False}
        comparison = _comparison(
            participant_mean=0.02,
            participant_low=0.01,
            text_mean=0.02,
            text_low=0.01,
            positive_folds=5,
        )

        decision = replication.make_decision(outcome, comparison, self.protocol)

        self.assertEqual(decision["label"], "does_not_replicate_over_gpt2")

    def test_strong_and_directional_tiers_are_distinct(self) -> None:
        outcome = {"incremental_gate_passed": True}
        strong = _comparison(
            participant_mean=0.02,
            participant_low=0.01,
            text_mean=0.02,
            text_low=0.01,
            positive_folds=4,
        )
        directional = copy.deepcopy(strong)
        directional["text_bootstrap"]["ci_95_low"] = -0.001

        strong_decision = replication.make_decision(
            outcome, strong, self.protocol
        )
        directional_decision = replication.make_decision(
            outcome, directional, self.protocol
        )

        self.assertEqual(strong_decision["label"], "strong_replication_over_gpt2")
        self.assertEqual(
            directional_decision["label"],
            "directional_replication_over_gpt2",
        )

    def test_secondary_direction_cannot_rescue_primary_fold_failure(self) -> None:
        outcome = {"incremental_gate_passed": True}
        comparison = _comparison(
            participant_mean=0.02,
            participant_low=-0.01,
            text_mean=0.02,
            text_low=-0.01,
            positive_folds=3,
        )

        decision = replication.make_decision(outcome, comparison, self.protocol)

        self.assertEqual(decision["label"], "does_not_replicate_over_gpt2")
        self.assertFalse(decision["production_model_changed"])
        self.assertFalse(decision["one_stop_retest_allowed"])


if __name__ == "__main__":
    unittest.main()
