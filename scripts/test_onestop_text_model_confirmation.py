"""Offline contract tests for the preregistered OneStop confirmation runner."""

from __future__ import annotations

import sys
import unittest

import pandas as pd

from scripts.prepare_onestop_confirmation import ANALYSIS_READ_COLUMNS
from scripts.run_onestop_text_model_confirmation import (
    FEATURE_SETS,
    IDENTITY_COLUMNS,
    _assert_protocol_contract,
    _scope_mask,
    make_decision,
    prepare_identity,
)


def _row(**updates: str) -> dict[str, str]:
    row = {column: "." for column in IDENTITY_COLUMNS}
    row.update(
        {
            "participant_id": "l1_001",
            "list_number": "1",
            "question_preview": "False",
            "article_batch": "1",
            "trial_index": "2",
            "practice_trial": "False",
            "article_id": "3",
            "paragraph_id": "1",
            "difficulty_level": "Adv",
            "repeated_reading_trial": "False",
            "IA_ID": "1",
            "IA_LABEL": " The ",
        }
    )
    row.update(updates)
    return row


def _bootstrap(low: float) -> dict[str, float]:
    return {
        "mean_difference": low + 0.01,
        "ci_95_low": low,
        "ci_95_high": low + 0.02,
        "n_pairs": 30,
    }


class OneStopTextModelConfirmationTests(unittest.TestCase):
    def test_lightweight_runner_import_does_not_load_spacy_or_thinc(self) -> None:
        self.assertNotIn("spacy", sys.modules)
        self.assertNotIn("thinc", sys.modules)

    def test_runner_exactly_matches_machine_protocol(self) -> None:
        protocol = _assert_protocol_contract()
        self.assertEqual(protocol["protocol_id"], "onestop-ordinary-advanced-confirmation-v1")
        self.assertEqual(set(FEATURE_SETS), {
            "word_length_only",
            "m0_lexical",
            "m1_causal_surprisal",
        })

    def test_scope_excludes_qa_preview_repeats_simplified_and_practice(self) -> None:
        rows = [
            _row(),
            _row(question_preview="True"),
            _row(repeated_reading_trial="True"),
            _row(difficulty_level="Ele"),
            _row(practice_trial="True"),
            _row(article_id="0"),
        ]
        mask = _scope_mask(pd.DataFrame(rows))
        self.assertEqual(mask.tolist(), [True, False, False, False, False, False])

    def test_identity_hashes_display_variants_and_holds_article_identity(self) -> None:
        frame = pd.DataFrame(
            [
                _row(),
                _row(IA_ID="2", IA_LABEL="reader"),
                _row(participant_id="l2_002", list_number="2", trial_index="8"),
                _row(
                    participant_id="l2_002",
                    list_number="2",
                    trial_index="8",
                    IA_ID="2",
                    IA_LABEL="reader",
                ),
            ]
        )
        trial_map, items, identity = prepare_identity(
            frame, enforce_expected_articles=False
        )

        self.assertEqual(len(trial_map), 2)
        self.assertEqual(identity["article_count"], 1)
        self.assertEqual(identity["context_count"], 1)
        self.assertEqual(len(items), 2)
        self.assertEqual(items.iloc[0]["IA_LABEL"], "The")
        self.assertFalse(identity["outcome_columns_read"])

    def test_decision_requires_effect_and_failed_shuffle_sentinel(self) -> None:
        primary = {
            "comparisons": [
                {
                    "comparison": "m1_causal_surprisal_minus_m0_lexical",
                    "participant_bootstrap": _bootstrap(0.01),
                    "article_bootstrap": _bootstrap(0.02),
                    "positive_outer_folds": 4,
                }
            ],
            "shuffle_sentinel": {
                "participant_bootstrap_vs_zero": _bootstrap(-0.02),
                "article_bootstrap_vs_zero": _bootstrap(-0.01),
            },
        }
        decision = make_decision({"total_reading_time": primary})
        self.assertTrue(decision["confirmed"])

        primary["shuffle_sentinel"]["article_bootstrap_vs_zero"] = _bootstrap(0.01)
        decision = make_decision({"total_reading_time": primary})
        self.assertFalse(decision["confirmed"])

    def test_no_qa_or_precomputed_annotation_is_in_read_whitelist(self) -> None:
        forbidden = {
            "question",
            "selected_answer",
            "is_correct",
            "gpt2_surprisal",
            "wordfreq_frequency",
            "critical_span_indices",
        }
        self.assertTrue(forbidden.isdisjoint(ANALYSIS_READ_COLUMNS))


if __name__ == "__main__":
    unittest.main()
