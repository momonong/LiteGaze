"""Regression tests for the evidence-bounded reader inspector v2."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.cognitive_inspector.inspector import ASSESSMENT_VERSION, CognitiveInspector
from web import create_app


def _sequential_events(
    word_count: int = 30,
    *,
    samples_per_word: int = 2,
    tick_ms: float = 125.0,
    confidence: float | str = "high",
) -> list[dict]:
    events = []
    timestamp = 0.0
    for index in range(word_count):
        for _ in range(samples_per_word):
            events.append(
                {
                    "word": f"word{index}",
                    "index": index,
                    "timestamp_ms": timestamp,
                    "confidence": confidence,
                }
            )
            timestamp += tick_ms
    return events


class CognitiveInspectorV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.inspector = CognitiveInspector(sample_rate_hz=8)

    def test_empty_session_abstains_instead_of_scoring_zero_ability(self) -> None:
        result = self.inspector.analyze([])
        self.assertEqual(result["assessment_version"], ASSESSMENT_VERSION)
        self.assertEqual(result["data_quality"]["status"], "insufficient")
        self.assertIsNone(result["user_profile"]["reading_ability_score"])
        self.assertIsNone(result["user_profile"]["english_proficiency_score"])
        self.assertEqual(
            result["claims"]["cognitive_ability"]["status"], "not_estimated"
        )

    def test_one_gaze_trace_never_becomes_english_or_attention_score(self) -> None:
        indices = [0, 1, 2, 1, 3, 0, 4, 5, 4, 6, 7, 2]
        events = [
            {
                "word": "common" if index % 2 else "sesquipedalian",
                "index": index,
                "timestamp_ms": position * 125,
                "confidence": "high",
            }
            for position, index in enumerate(indices)
        ]
        result = self.inspector.analyze(events, lang="en")
        self.assertGreater(result["summary"]["regression_count"], 0)
        self.assertEqual(result["claims"]["attention"]["status"], "not_estimated")
        self.assertEqual(
            result["claims"]["english_proficiency"]["status"], "not_estimated"
        )
        self.assertIsNone(result["user_profile"]["attention_index"])

    def test_equivalent_sampling_rates_produce_equivalent_dwell(self) -> None:
        slow_samples = _sequential_events(25, samples_per_word=2, tick_ms=125.0)
        fast_samples = _sequential_events(25, samples_per_word=4, tick_ms=62.5)
        slow = CognitiveInspector(sample_rate_hz=8).analyze(slow_samples)
        fast = CognitiveInspector(sample_rate_hz=16).analyze(fast_samples)
        self.assertEqual(slow["summary"]["total_fixations"], 25)
        self.assertEqual(fast["summary"]["total_fixations"], 25)
        self.assertAlmostEqual(
            slow["summary"]["median_fixation_duration_ms"],
            fast["summary"]["median_fixation_duration_ms"],
            delta=0.1,
        )
        self.assertAlmostEqual(
            slow["summary"]["total_dwell_time_ms"],
            fast["summary"]["total_dwell_time_ms"],
            delta=1,
        )

    def test_tracking_confidence_changes_quality_not_raw_behavior(self) -> None:
        high = self.inspector.analyze(_sequential_events(confidence=0.95))
        low = self.inspector.analyze(_sequential_events(confidence=0.1))
        self.assertEqual(high["summary"], low["summary"])
        self.assertGreater(high["data_quality"]["score"], low["data_quality"]["score"])
        self.assertIn("low_mean_tracking_confidence", low["data_quality"]["reasons"])

    def test_full_text_wpm_requires_explicit_elapsed_and_completion(self) -> None:
        events = _sequential_events(30)
        no_context = self.inspector.analyze(events)
        incomplete = self.inspector.analyze(
            events,
            context={"text_word_count": 120, "elapsed_time_ms": 60_000},
        )
        complete = self.inspector.analyze(
            events,
            context={
                "text_word_count": 120,
                "elapsed_time_ms": 60_000,
                "completed_text": True,
                "comprehension": {"correct": 2, "total": 3},
            },
        )
        self.assertIsNone(no_context["summary"]["words_per_minute"])
        self.assertIsNone(incomplete["summary"]["words_per_minute"])
        self.assertEqual(complete["summary"]["words_per_minute"], 120.0)
        self.assertEqual(
            complete["claims"]["reading_fluency"]["status"],
            "provisional_session_estimate",
        )

    def test_observed_word_rate_is_not_labeled_as_full_text_wpm(self) -> None:
        result = self.inspector.analyze(_sequential_events(20))
        self.assertIsNotNone(result["summary"]["observed_word_rate_wpm"])
        self.assertIsNone(result["summary"]["words_per_minute"])
        self.assertEqual(result["summary"]["words_per_minute_basis"], "not_available")

    def test_non_monotonic_timestamps_reduce_quality(self) -> None:
        events = _sequential_events(20)
        events[8]["timestamp_ms"], events[9]["timestamp_ms"] = (
            events[9]["timestamp_ms"],
            events[8]["timestamp_ms"],
        )
        result = self.inspector.analyze(events)
        self.assertFalse(result["data_quality"]["checks"]["timestamps_monotonic"])
        self.assertIn(
            "timestamps_were_not_monotonic", result["data_quality"]["reasons"]
        )

    def test_sparse_video_sampling_is_flagged_not_silently_capped(self) -> None:
        events = _sequential_events(12, samples_per_word=1, tick_ms=800)
        result = self.inspector.analyze(events)
        self.assertIn(
            "sampling_too_sparse_for_fixation_timing", result["data_quality"]["reasons"]
        )
        self.assertGreaterEqual(result["summary"]["median_fixation_duration_ms"], 800)

    def test_rare_word_dwell_is_only_a_session_signal(self) -> None:
        words = [
            "the",
            "garden",
            "water",
            "people",
            "simple",
            "system",
            "common",
            "house",
            "sesquipedalian",
            "antediluvian",
            "perspicacious",
            "defenestration",
            "pulchritudinous",
            "circumlocution",
        ]
        events = []
        timestamp = 0
        for index, word in enumerate(words):
            repeats = 4 if index >= 8 else 2
            for _ in range(repeats):
                events.append(
                    {
                        "word": word,
                        "index": index,
                        "timestamp_ms": timestamp,
                        "confidence": "high",
                    }
                )
                timestamp += 125
        result = self.inspector.analyze(events)
        lexical = result["claims"]["lexical_processing_signal"]
        self.assertEqual(lexical["status"], "session_observation")
        self.assertIsInstance(lexical["rarity_dwell_association"], float)
        self.assertEqual(
            result["claims"]["english_proficiency"]["status"], "not_estimated"
        )


class CognitiveInspectorApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-inspector-v2-")
        self.addCleanup(self.temp_dir.cleanup)
        self.reports_dir = Path(self.temp_dir.name)
        patcher = patch("web.routes.inspector.REPORTS_DIR", self.reports_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = create_app({"TESTING": True, "LEXIGAZE_BLUEPRINTS": ("inspector",)})
        self.client = self.app.test_client()

    def test_analyze_endpoint_accepts_measurement_context(self) -> None:
        response = self.client.post(
            "/api/inspector/analyze",
            json={
                "gaze_history": _sequential_events(30),
                "lang": "en",
                "context": {
                    "text_word_count": 150,
                    "elapsed_time_ms": 60_000,
                    "completed_text": True,
                    "comprehension": {"correct": 3, "total": 3},
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        analysis = response.get_json()["analysis"]
        self.assertEqual(analysis["summary"]["words_per_minute"], 150.0)
        self.assertIn("claims", analysis)

    def test_analyze_rejects_non_object_context(self) -> None:
        response = self.client.post(
            "/api/inspector/analyze",
            json={"gaze_history": [], "context": "not-an-object"},
        )
        self.assertEqual(response.status_code, 400)

    def test_report_is_transparent_and_can_be_persisted(self) -> None:
        response = self.client.post(
            "/api/inspector/report",
            json={
                "gaze_history": _sequential_events(30),
                "participant_id": "v2-user",
                "lang": "en",
                "context": {
                    "text_word_count": 120,
                    "elapsed_time_ms": 60_000,
                    "completed_text": True,
                },
                "persist": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("不是智力", payload["report_md"])
        self.assertIn("CEFR", payload["report_md"])
        report_name = Path(payload["analysis"]["report_path"]).name
        persisted = self.reports_dir / report_name
        self.assertTrue(persisted.exists())
        self.assertIn(
            "abstain_without_validity_evidence", persisted.read_text(encoding="utf-8")
        )

    def test_report_lifecycle(self) -> None:
        created = self.client.post(
            "/api/inspector/report",
            json={
                "gaze_history": _sequential_events(10),
                "participant_id": "lifecycle-v2",
                "persist": True,
            },
        ).get_json()
        filename = Path(created["analysis"]["report_path"]).name
        listed = self.client.get("/api/inspector/reports").get_json()["reports"]
        self.assertIn(filename, [report["filename"] for report in listed])
        self.assertEqual(
            self.client.get(f"/api/inspector/reports/{filename}").status_code, 200
        )
        self.assertEqual(
            self.client.delete(f"/api/inspector/reports/{filename}").status_code, 200
        )
        self.assertFalse((self.reports_dir / filename).exists())

    def test_report_normalizes_non_string_and_unsafe_participant_ids(self) -> None:
        numeric = self.client.post(
            "/api/inspector/report",
            json={
                "gaze_history": _sequential_events(10),
                "participant_id": 42,
                "persist": True,
            },
        )
        self.assertEqual(numeric.status_code, 200)
        numeric_name = Path(numeric.get_json()["analysis"]["report_path"]).name
        self.assertTrue(numeric_name.startswith("42_"))

        unsafe = self.client.post(
            "/api/inspector/report",
            json={
                "gaze_history": _sequential_events(10),
                "participant_id": "../",
                "persist": True,
            },
        )
        self.assertEqual(unsafe.status_code, 200)
        unsafe_name = Path(unsafe.get_json()["analysis"]["report_path"]).name
        self.assertTrue(unsafe_name.startswith("anonymous_"))

    def test_invalid_gaze_payload_is_rejected(self) -> None:
        response = self.client.post(
            "/api/inspector/analyze", json={"gaze_history": "not-an-array"}
        )
        self.assertEqual(response.status_code, 400)


class ReaderAssessmentUiCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        template_path = (
            Path(__file__).resolve().parents[1]
            / "web"
            / "templates"
            / "word_track.html"
        )
        cls.template = template_path.read_text(encoding="utf-8")

    def test_active_inspector_copy_does_not_make_legacy_diagnostic_claims(self) -> None:
        forbidden_claims = (
            "基於眼動序列，分析使用者的閱讀理解力、英語熟練度與疲勞程度。",
            "開始診斷認知能力",
            "跨階段認知負載與排版分析",
            "適應性多輪認知診斷測驗",
            "認知負荷評估",
            "高認知負擔詞彙",
            "回答兩題理解選擇題",
            "認知與排版適應性評估完成",
            "眼動軌跡與認知回視重建",
        )
        for claim in forbidden_claims:
            with self.subTest(claim=claim):
                self.assertNotIn(claim, self.template)

    def test_active_inspector_copy_discloses_scope_and_limitations(self) -> None:
        self.assertIn("閱讀測量證據", self.template)
        self.assertIn("不推論智力、注意力、疲勞或英語程度", self.template)
        self.assertIn("實驗性多輪閱讀評量", self.template)
        self.assertIn("只有在全文完成且時間資訊完整時才顯示全文 WPM", self.template)
        self.assertIn("這是文本模型輸出，不是使用者認知負荷或能力測量", self.template)
        self.assertIn("回答三題理解選擇題", self.template)
        self.assertIn("在文件上標示處理需求", self.template)
        self.assertIn(
            "顏色只標示座標移動方向，不代表理解困難、注意力或認知狀態", self.template
        )


if __name__ == "__main__":
    unittest.main()
