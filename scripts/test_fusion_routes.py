"""
scripts/test_fusion_routes.py
══════════════════════════════════════════════════════════════════════════════
Unit tests to verify that the newly added neural cross-attention and fatigue-adaptive
fusion methods work correctly in the Flask API.
"""

import json
import sys
import unittest

from web import create_app


class TestFusionRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app({
            "TESTING": True,
            "LEXIGAZE_BLUEPRINTS": ("fusion",),
        })
        self.client = self.app.test_client()

    @staticmethod
    def _request_payload(method):
        cognitive_result = {
            "model": "gpt2",
            "lang": "en",
            "domain": "auto",
            "word_analysis": [
                {"word": "the", "load_score": 0.1, "surprisal": 2.1},
                {"word": "neural", "load_score": 0.8, "surprisal": 9.4},
                {"word": "fusion", "load_score": 0.7, "surprisal": 8.1},
                {"word": "works", "load_score": 0.4, "surprisal": 4.5}
            ]
        }
        gaze_events = [
            {"word": "the", "confidence": "high", "dwell_count": 2, "fixation_count": 1, "timestamp_ms": 1000},
            {"word": "neural", "confidence": "high", "dwell_count": 6, "fixation_count": 2, "timestamp_ms": 1100},
            {"word": "fusion", "confidence": "medium", "dwell_count": 5, "fixation_count": 1, "timestamp_ms": 1220},
            {"word": "works", "confidence": "low", "dwell_count": 1, "fixation_count": 0, "timestamp_ms": 1400}
        ]
        return {
            "session_id": "test_cross_attn_session",
            "persist": False,
            "cognitive_result": cognitive_result,
            "gaze_events": gaze_events,
            "method": method,
        }

    def test_cross_attention_method(self):
        """Untrained cross-attention is rejected without importing Torch."""
        torch_was_loaded = "torch" in sys.modules
        res_cross = self.client.post(
            "/api/fuse/",
            json=self._request_payload("cross_attention"),
        )
        self.assertEqual(res_cross.status_code, 422)
        data_cross = json.loads(res_cross.data)
        self.assertFalse(data_cross["ok"])
        self.assertEqual(data_cross["error"], "production_ineligible_fusion_method")
        self.assertEqual("torch" in sys.modules, torch_was_loaded)

    def test_fatigue_adaptive_method(self):
        """Fast deterministic fusion regression used by the offline CPU gate."""
        payload = self._request_payload("fatigue_adaptive")
        payload["session_id"] = "test_fatigue_session"
        res_fatigue = self.client.post("/api/fuse/", json=payload)
        self.assertEqual(res_fatigue.status_code, 200)
        data_fatigue = json.loads(res_fatigue.data)
        self.assertTrue(data_fatigue["ok"])
        self.assertEqual(len(data_fatigue["rds"]), 4)

if __name__ == "__main__":
    unittest.main()
