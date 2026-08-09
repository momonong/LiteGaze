"""Offline contract tests for the shadow-only line-first gaze decoder."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECODER_PATH = ROOT / "web" / "static" / "gaze_line_decoder.js"
MAPPING_PATH = ROOT / "web" / "static" / "mapping.js"
INTEGRATION_PATH = ROOT / "web" / "static" / "gaze_integration.js"
TEMPLATE_PATH = ROOT / "web" / "templates" / "word_track.html"


def _function_to_end(source: str, name: str) -> str:
    match = re.search(
        rf"function {re.escape(name)}\b[\s\S]*?(?=\n\s*return Object\.freeze)",
        source,
    )
    if not match:
        raise AssertionError(f"could not locate JavaScript function {name}")
    return match.group(0)


class GazeLineDecoderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.decoder = DECODER_PATH.read_text(encoding="utf-8")
        cls.mapping = MAPPING_PATH.read_text(encoding="utf-8")
        cls.integration = INTEGRATION_PATH.read_text(encoding="utf-8")
        cls.template = TEMPLATE_PATH.read_text(encoding="utf-8")

    def test_decoder_is_loaded_but_not_wired_into_production_mapping(self) -> None:
        self.assertIn('/static/gaze_line_decoder.js', self.template)
        self.assertNotIn("LexiGazeLineDecoder", self.mapping)
        self.assertNotIn("LexiGazeLineDecoder", self.integration)
        self.assertIn('shadow_only: true', self.decoder)
        self.assertIn('geometry_only: true', self.decoder)

    def test_decoder_does_not_read_non_geometry_fields(self) -> None:
        for forbidden in (
            ".text",
            '["text"]',
            "['text']",
            "load_score",
            "cognitive",
            "profile",
            "prior",
        ):
            self.assertNotIn(forbidden, self.decoder.lower())

    def test_sensor_abstention_precedes_layout_decoding(self) -> None:
        body = _function_to_end(self.decoder, "decodeLineFirst")
        sensor_gate = body.index('if (!accepted)')
        token_normalization = body.index("normalizeTokens(tokens)")
        line_decoding = body.index("buildLines(")
        self.assertLess(sensor_gate, token_normalization)
        self.assertLess(sensor_gate, line_decoding)
        self.assertIn('abstention("sensor_abstained"', body)

    def test_occurrence_and_line_contract_is_explicit(self) -> None:
        for field in (
            "occurrence_id",
            "bbox",
            "line_id",
            "reading_order",
            "line_scores",
            "top_k_occurrences",
            "geometry_only_result",
        ):
            self.assertIn(field, self.decoder)
        self.assertIn('reason: "duplicate_occurrence_id"', self.decoder)

    def test_abstention_reasons_are_explicit(self) -> None:
        for reason in (
            "missing_sensor_observation",
            "sensor_abstained",
            "invalid_sensor_coordinates",
            "missing_layout_tokens",
            "invalid_layout_token",
            "duplicate_occurrence_id",
            "outside_line_geometry",
            "outside_token_geometry",
        ):
            self.assertIn(f'"{reason}"', self.decoder)


if __name__ == "__main__":
    unittest.main()
