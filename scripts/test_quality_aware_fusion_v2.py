"""CPU-only contract tests for occurrence-aware quality fusion v2."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from core.cognition.quality_fusion import (
    PROTOCOL_ID,
    QualityAwareFusionConfig,
    aggregate_quality,
    confidence_quality,
    fuse_quality_aware,
    stable_gaze_score,
)
from scripts.fusion.orchestrator import (
    QUALITY_AWARE_SHADOW_METHOD,
    aggregate_gaze_events,
    align_gaze_occurrences,
    compute_rds,
)
from scripts.run_quality_aware_fusion_v2_experiment import (
    _mean_gaze_weight_order_met,
)
from web import create_app

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "experiments"
    / "protocols"
    / "2026-08-07-quality-aware-text-fusion-v2.json"
)


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


class QualityFusionCoreTests(unittest.TestCase):
    def test_gate_pairs_every_adjacent_quality_condition(self) -> None:
        ordered = ["clean", "drift", "jitter", "dropout", "missing"]
        decreasing = {
            "clean": 0.9,
            "drift": 0.7,
            "jitter": 0.5,
            "dropout": 0.2,
            "missing": 0.0,
        }

        self.assertTrue(_mean_gaze_weight_order_met(ordered, decreasing))
        decreasing["jitter"] = decreasing["drift"]
        self.assertFalse(
            _mean_gaze_weight_order_met(ordered, decreasing)
        )

    def test_protocol_is_frozen_cpu_only_and_loads_exact_config(self) -> None:
        protocol = _protocol()
        config = QualityAwareFusionConfig.from_protocol(protocol)

        self.assertEqual(protocol["status"], "frozen_before_benchmark")
        self.assertEqual(protocol["protocol_id"], PROTOCOL_ID)
        self.assertFalse(protocol["compute"]["gpu_allowed"])
        self.assertFalse(protocol["compute"]["torch_allowed"])
        self.assertFalse(protocol["leakage_controls"]["question_answer_dataset_used"])
        self.assertEqual(config.gaze_weight_power, 1.5)
        self.assertEqual(config.medium_confidence, 0.55)

    def test_missing_gaze_returns_text_score_exactly(self) -> None:
        result = fuse_quality_aware(
            text_score=0.731,
            gaze_score=None,
            mapping_confidence=0.0,
            tracking_coverage=0.0,
            stability=0.0,
            calibration_quality=0.0,
        )

        self.assertEqual(result.fused_score, 0.731)
        self.assertEqual(result.gaze_weight, 0.0)
        self.assertEqual(result.reason, "missing_gaze_text_fallback")

    def test_gaze_weight_is_monotonic_in_every_quality_component(self) -> None:
        qualities = [
            aggregate_quality(
                mapping_confidence=value,
                tracking_coverage=value,
                stability=value,
                calibration_quality=value,
            )
            for value in (1.0, 0.8, 0.5, 0.2, 0.0)
        ]

        self.assertEqual(qualities, sorted(qualities, reverse=True))
        self.assertEqual(qualities[-1], 0.0)

    def test_confidence_uses_full_distribution_not_best_hit(self) -> None:
        mostly_low = confidence_quality({"high": 1, "medium": 0, "low": 9})
        all_high = confidence_quality({"high": 10, "medium": 0, "low": 0})

        self.assertAlmostEqual(mostly_low, 0.235)
        self.assertEqual(all_high, 1.0)

    def test_stable_gaze_score_is_independent_of_other_words(self) -> None:
        before = stable_gaze_score(720, 2)
        _unrelated_extreme = stable_gaze_score(3000, 20)
        after = stable_gaze_score(720, 2)

        self.assertEqual(before, after)
        self.assertGreater(before, 0)

    def test_low_quality_and_text_ood_abstains(self) -> None:
        result = fuse_quality_aware(
            text_score=0.7,
            gaze_score=0.2,
            mapping_confidence=0.2,
            tracking_coverage=0.3,
            stability=0.2,
            calibration_quality=0.2,
            text_in_distribution=False,
        )

        self.assertTrue(result.abstain)
        self.assertEqual(result.reason, "text_ood_and_low_gaze_quality")


class OccurrenceAlignmentTests(unittest.TestCase):
    @staticmethod
    def _analysis() -> list[dict]:
        return [
            {"word": "the", "position": 0, "load_score": 0.2},
            {"word": "reader", "position": 1, "load_score": 0.5},
            {"word": "the", "position": 2, "load_score": 0.8},
        ]

    @staticmethod
    def _events() -> list[dict]:
        return [
            {
                "occurrence_id": "page:1:word:0",
                "page_num": 1,
                "word_index": 0,
                "word": "the",
                "dwell_count": 1,
                "fixation_count": 1,
                "hit_count": 1,
                "confidence_counts": {"high": 1, "medium": 0, "low": 0},
            },
            {
                "occurrence_id": "page:1:word:1",
                "page_num": 1,
                "word_index": 1,
                "word": "reader",
                "dwell_count": 3,
                "fixation_count": 1,
                "hit_count": 3,
                "confidence_counts": {"high": 2, "medium": 1, "low": 0},
            },
            {
                "occurrence_id": "page:1:word:2",
                "page_num": 1,
                "word_index": 2,
                "word": "the",
                "dwell_count": 8,
                "fixation_count": 2,
                "hit_count": 8,
                "confidence_counts": {"high": 6, "medium": 2, "low": 0},
            },
        ]

    def test_repeated_spellings_remain_distinct_occurrences(self) -> None:
        aggregated = aggregate_gaze_events(self._events())
        aligned = align_gaze_occurrences(self._analysis(), aggregated)

        self.assertEqual(len(aggregated), 3)
        self.assertEqual(aligned[0]["occurrence_id"], "page:1:word:0")
        self.assertEqual(aligned[2]["occurrence_id"], "page:1:word:2")
        self.assertEqual(aligned[0]["dwell_ms"], 120)
        self.assertEqual(aligned[2]["dwell_ms"], 960)

    def test_quality_candidate_is_shadow_only_and_occurrence_aware(self) -> None:
        torch_was_loaded = "torch" in sys.modules
        results = compute_rds(
            self._events(),
            {"word_analysis": self._analysis()},
            method=QUALITY_AWARE_SHADOW_METHOD,
            quality_context={
                "tracking_coverage": 0.9,
                "stability": 0.8,
                "calibration_quality": 0.85,
            },
        )
        by_occurrence = {result["occurrence_id"]: result for result in results}

        self.assertEqual(len(by_occurrence), 3)
        self.assertEqual(
            by_occurrence["page:1:word:0"]["candidate_status"], "shadow_only"
        )
        self.assertNotEqual(
            by_occurrence["page:1:word:0"]["dwell_ms"],
            by_occurrence["page:1:word:2"]["dwell_ms"],
        )
        self.assertEqual("torch" in sys.modules, torch_was_loaded)

    def test_legacy_word_only_events_remain_backward_compatible(self) -> None:
        legacy = [{"word": "the", "dwell_count": 2, "fixation_count": 1}]
        aligned = align_gaze_occurrences(
            self._analysis(), aggregate_gaze_events(legacy)
        )

        self.assertEqual(aligned[0]["dwell_ms"], 240)
        self.assertEqual(aligned[2]["dwell_ms"], 240)
        self.assertEqual(aligned[2]["alignment_source"], "legacy_normalized_word")


class QualityFusionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        app = create_app(
            {"TESTING": True, "LEXIGAZE_BLUEPRINTS": ("fusion",)}
        )
        self.client = app.test_client()

    @staticmethod
    def _payload() -> dict:
        return {
            "session_id": "quality-v2-test",
            "persist": False,
            "method": QUALITY_AWARE_SHADOW_METHOD,
            "cognitive_result": {
                "word_analysis": [
                    {"word": "read", "position": 0, "load_score": 0.4}
                ]
            },
            "gaze_events": [
                {
                    "occurrence_id": "page:1:word:0",
                    "page_num": 1,
                    "word_index": 0,
                    "word": "read",
                    "dwell_count": 4,
                    "fixation_count": 1,
                    "hit_count": 4,
                    "confidence_counts": {"high": 3, "medium": 1, "low": 0},
                }
            ],
        }

    def test_route_requires_all_quality_context_fields(self) -> None:
        response = self.client.post("/api/fuse/", json=self._payload())

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_fusion_input")

    def test_route_returns_explicit_shadow_candidate(self) -> None:
        payload = self._payload()
        payload["quality_context"] = {
            "tracking_coverage": 0.9,
            "stability": 0.85,
            "calibration_quality": 0.8,
        }
        response = self.client.post("/api/fuse/", json=payload)
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["candidate_status"], "shadow_only")
        self.assertEqual(body["summary"]["fusion_method"], QUALITY_AWARE_SHADOW_METHOD)
        self.assertEqual(body["rds"][0]["quality_aware_v2"]["protocol_id"], PROTOCOL_ID)


if __name__ == "__main__":
    unittest.main()
