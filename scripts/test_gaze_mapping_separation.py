"""Offline source-contract tests for geometry-only live gaze mapping."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = ROOT / "web" / "static" / "mapping.js"
INTEGRATION_PATH = ROOT / "web" / "static" / "gaze_integration.js"
TEMPLATE_PATH = ROOT / "web" / "templates" / "word_track.html"


def _function_body(source: str, name: str, next_name: str) -> str:
    match = re.search(
        rf"function {re.escape(name)}\b[\s\S]*?"
        rf"(?=\n\s*function {re.escape(next_name)}\b)",
        source,
    )
    if not match:
        raise AssertionError(f"could not locate JavaScript function {name}")
    return match.group(0)


class GazeMappingSeparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapping = MAPPING_PATH.read_text(encoding="utf-8")
        cls.integration = INTEGRATION_PATH.read_text(encoding="utf-8")
        cls.template = TEMPLATE_PATH.read_text(encoding="utf-8")

    def test_geometry_core_loads_before_live_mapping(self) -> None:
        core_index = self.template.index('/static/gaze_mapping_core.js')
        mapping_index = self.template.index('/static/mapping.js')
        self.assertLess(core_index, mapping_index)

    def test_geometry_mapping_does_not_read_text_or_cognitive_prior(self) -> None:
        body = _function_body(
            self.mapping,
            "findNearestExtractedWord",
            "drawHighlights",
        )
        self.assertIn("findNearestGeometryCandidate", body)
        for forbidden in (
            "lookupCognitive",
            "loadScore",
            "cognitiveMass",
            "effectiveDistance",
        ):
            self.assertNotIn(forbidden, body)

        process_body = _function_body(
            self.mapping,
            "processGazeOnExtractedData",
            "distanceToExtractedRect",
        )
        self.assertIn('mapping_mode: SENSOR_MAPPING_MODE', process_body)
        self.assertIn("sensor_accepted: true", process_body)
        self.assertNotIn("lookupCognitive", process_body)

    def test_sensor_buffer_rejects_non_geometry_or_prior_rescued_hits(self) -> None:
        body = _function_body(
            self.integration,
            "recordGazeHit",
            "flushGazeBuffer",
        )
        self.assertIn('mappingMode !== "geometry_only_v1"', body)
        self.assertIn("mappingContext.sensor_accepted !== true", body)
        self.assertIn("return false", body)

    def test_sensor_coverage_uses_geometry_counter_only(self) -> None:
        self.assertIn(
            "trackingStats.sensor_mapped_samples / trackingStats.inference_samples",
            self.integration,
        )
        quality_section = self.integration.split("quality_context:", maxsplit=1)[1]
        coverage_section = quality_section.split("tracking_coverage:", maxsplit=1)[1]
        coverage_section = coverage_section.split("...(", maxsplit=1)[0]
        self.assertNotIn("trackingStats.mapped_samples /", coverage_section)
        self.assertLess(
            quality_section.index("...(window.lexiGazeQualityContext || {})"),
            quality_section.index("tracking_coverage:"),
            "external quality context can override sensor-derived coverage",
        )

    def test_cognitive_mapping_preview_is_explicit_opt_in(self) -> None:
        draw_body = _function_body(
            self.mapping,
            "drawHighlights",
            "drawGazeWordFusionAttractor",
        )
        self.assertIn("window.lexiEnableCognitiveMappingPreview === true", draw_body)


if __name__ == "__main__":
    unittest.main()
