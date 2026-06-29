"""
scripts/test_fusion_routes.py
══════════════════════════════════════════════════════════════════════════════
Unit tests to verify that the newly added neural cross-attention and fatigue-adaptive
fusion methods work correctly in the Flask API.
"""

import sys
import unittest
import json
from pathlib import Path

# Setup root path for import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web import create_app

class TestFusionRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        
    def test_fusion_methods(self):
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
        
        # Test cross_attention method
        res_cross = self.client.post("/api/fuse/", json={
            "session_id": "test_cross_attn_session",
            "persist": False,
            "cognitive_result": cognitive_result,
            "gaze_events": gaze_events,
            "method": "cross_attention"
        })
        self.assertEqual(res_cross.status_code, 200)
        data_cross = json.loads(res_cross.data)
        self.assertTrue(data_cross["ok"])
        self.assertEqual(len(data_cross["rds"]), 4)
        
        # Verify RDS scores are non-empty
        for item in data_cross["rds"]:
            self.assertIn("rds", item)
            self.assertIn("rds_level", item)
            
        # Test fatigue_adaptive method
        res_fatigue = self.client.post("/api/fuse/", json={
            "session_id": "test_fatigue_session",
            "persist": False,
            "cognitive_result": cognitive_result,
            "gaze_events": gaze_events,
            "method": "fatigue_adaptive"
        })
        self.assertEqual(res_fatigue.status_code, 200)
        data_fatigue = json.loads(res_fatigue.data)
        self.assertTrue(data_fatigue["ok"])
        self.assertEqual(len(data_fatigue["rds"]), 4)
        
if __name__ == "__main__":
    unittest.main()
