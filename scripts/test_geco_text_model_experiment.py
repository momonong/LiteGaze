"""Offline schema tests for the frozen GECO L2 text-model replication."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_geco_l2_text_model_experiment import (
    SOURCE_COLUMNS,
    _prepare_geco_l2,
)


def _row(**updates: str) -> dict[str, str]:
    row = {column: "." for column in SOURCE_COLUMNS}
    row.update(
        {
            "PP_NR": "pp01",
            "GROUP": "bilingual",
            "LANGUAGE": "English",
            "PART": "3",
            "TRIAL": "5",
            "WORD_ID_WITHIN_TRIAL": "1",
            "WORD_ID": "3-5-1",
            "WORD": "  There\u00a0",
            "WORD_TOTAL_READING_TIME": "330",
            "WORD_GAZE_DURATION": "330",
            "WORD_FIRST_FIXATION_DURATION": "187",
        }
    )
    row.update(updates)
    return row


class GecoTextModelExperimentTests(unittest.TestCase):
    def test_preparation_normalizes_display_padding_and_builds_text_id(self) -> None:
        raw, items = _prepare_geco_l2(pd.DataFrame([_row()]))

        self.assertEqual(raw.loc[0, "IA_LABEL"], "There")
        self.assertEqual(raw.loc[0, "Text_ID"], "3:5")
        self.assertEqual(raw.loc[0, "IA_ID"], 1)
        self.assertEqual(items.to_dict("records"), [
            {"Text_ID": "3:5", "IA_ID": 1, "IA_LABEL": "There"}
        ])

    def test_preparation_rejects_non_english_trials(self) -> None:
        with self.assertRaisesRegex(ValueError, "only English"):
            _prepare_geco_l2(pd.DataFrame([_row(LANGUAGE="Dutch")]))

    def test_preparation_rejects_participant_item_duplicates(self) -> None:
        duplicate = pd.DataFrame([_row(), _row()])
        with self.assertRaisesRegex(ValueError, "duplicate participant"):
            _prepare_geco_l2(duplicate)


if __name__ == "__main__":
    unittest.main()
