"""Offline source contracts for focused, independently runnable Node tests.

The offline quality-gate worker intentionally denies child-process creation, so
it cannot launch Node itself.  These checks keep each focused frontend test and
its pure module entrypoint present without weakening that process boundary.
"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORD_TRACK_PATH = ROOT / "web" / "templates" / "word_track.html"
GAZE_INTEGRATION_PATH = ROOT / "web" / "static" / "gaze_integration.js"
PARTICIPANT_ASSESSMENT_TEMPLATE_PATH = (
    ROOT / "web" / "templates" / "participant_assessment.html"
)
PARTICIPANT_ASSESSMENT_PATH = (
    ROOT / "web" / "static" / "participant_assessment.js"
)
EXPECTED_PREDICT_CALLSITES = {
    "web/static/gaze_integration.js",
    "web/static/gaze_page.js",
    "web/static/participant_assessment.js",
    "web/static/participant_collection.js",
}

# These are also the documented commands for running the behavior tests outside
# the process-denying offline worker.
NODE_TEST_CONTRACTS = (
    (
        "node scripts/test_gaze_capture_contract.js",
        "scripts/test_gaze_capture_contract.js",
        "web/static/gaze_capture_contract.js",
        'require("../web/static/gaze_capture_contract.js")',
        "gaze capture contract tests passed",
    ),
    (
        "node scripts/test_gaze_mapping_geometry.js",
        "scripts/test_gaze_mapping_geometry.js",
        "web/static/gaze_mapping_core.js",
        'require("../web/static/gaze_mapping_core.js")',
        "gaze mapping geometry separation: ok",
    ),
    (
        "node scripts/test_gaze_line_decoder.js",
        "scripts/test_gaze_line_decoder.js",
        "web/static/gaze_line_decoder.js",
        'require("../web/static/gaze_line_decoder.js")',
        "gaze line-first shadow decoder: ok",
    ),
)


class GazeFrontendNodeContractTests(unittest.TestCase):
    def test_focused_node_tests_have_standalone_commands_and_modules(self) -> None:
        for command, test_name, module_name, require_statement, success_marker in (
            NODE_TEST_CONTRACTS
        ):
            with self.subTest(command=command):
                self.assertEqual(command, f"node {test_name}")
                test_path = ROOT / test_name
                module_path = ROOT / module_name
                self.assertTrue(test_path.is_file(), f"missing {test_name}")
                self.assertTrue(module_path.is_file(), f"missing {module_name}")

                source = test_path.read_text(encoding="utf-8")
                self.assertIn('require("node:assert/strict")', source)
                self.assertIn(require_statement, source)
                self.assertIn(success_marker, source)

    def test_live_gaze_uses_shared_aspect_preserving_capture_contract(self) -> None:
        template = WORD_TRACK_PATH.read_text(encoding="utf-8")
        capture_index = template.index('/static/gaze_capture_contract.js')
        integration_index = template.index('/static/gaze_integration.js')
        self.assertLess(capture_index, integration_index)

        integration = GAZE_INTEGRATION_PATH.read_text(encoding="utf-8")
        self.assertIn("const gazeCapture = window.LexiGazeCapture", integration)
        self.assertIn("gazeCapture.mediaConstraints()", integration)
        self.assertIn(
            "gazeCapture.captureSnapshot(state.video, state.canvas)",
            integration,
        )
        self.assertIn("image_data: snapshot.image_data", integration)
        self.assertIn("capture_contract: snapshot.capture_contract", integration)
        self.assertNotIn("const width = 240", integration)
        self.assertNotIn('toDataURL("image/jpeg", 0.5)', integration)

        assessment_template = PARTICIPANT_ASSESSMENT_TEMPLATE_PATH.read_text(
            encoding="utf-8"
        )
        assessment_capture_index = assessment_template.index(
            "gaze_capture_contract.js"
        )
        assessment_script_index = assessment_template.index(
            "participant_assessment.js"
        )
        self.assertLess(assessment_capture_index, assessment_script_index)

        assessment = PARTICIPANT_ASSESSMENT_PATH.read_text(encoding="utf-8")
        self.assertIn("gazeCapture.mediaConstraints()", assessment)
        self.assertIn("gazeCapture.captureSnapshot(", assessment)
        self.assertIn("image_data: snapshot.image_data", assessment)
        self.assertIn("capture_contract: snapshot.capture_contract", assessment)

    def test_every_web_gaze_predict_callsite_sends_capture_contract(self) -> None:
        endpoint = "/api/gaze/predict"
        callsites: dict[str, str] = {}
        for pattern in ("*.js", "*.html"):
            for path in (ROOT / "web").rglob(pattern):
                source = path.read_text(encoding="utf-8")
                if endpoint in source:
                    callsites[path.relative_to(ROOT).as_posix()] = source

        self.assertEqual(set(callsites), EXPECTED_PREDICT_CALLSITES)
        for relative_path, source in callsites.items():
            positions = []
            start = 0
            while (position := source.find(endpoint, start)) >= 0:
                positions.append(position)
                start = position + len(endpoint)
            for position in positions:
                with self.subTest(path=relative_path, position=position):
                    request_source = source[position : position + 1200]
                    self.assertIn("capture_contract:", request_source)

    def test_node_contract_does_not_spawn_from_python_worker(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        process_api = "sub" + "process"
        self.assertNotIn(f"import {process_api}", source)
        self.assertNotIn(f"from {process_api}", source)


if __name__ == "__main__":
    unittest.main()
