import sys
import unittest
import json
from pathlib import Path

# Setup root path for import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web import create_app
from web.routes.inspector import _clean_json_response

class TestAdaptiveStepper(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_clean_json_response_helper(self):
        """測試 _clean_json_response 能否安全剝離 <thought> 標籤與 markdown 語法。"""
        # Case 1: 正常 JSON 應保持原樣
        raw1 = '{"key": "value"}'
        self.assertEqual(_clean_json_response(raw1), '{"key": "value"}')

        # Case 2: 包含 markdown 包裝的 JSON
        raw2 = '```json\n{"key": "value"}\n```'
        self.assertEqual(_clean_json_response(raw2), '{"key": "value"}')

        # Case 3: 包含 <thought> 標籤的 JSON
        raw3 = '<thought>I am thinking here</thought>{"key": "value"}'
        self.assertEqual(_clean_json_response(raw3), '{"key": "value"}')

        # Case 4: 包含換行、大小寫 <Thought> 標籤與 markdown 的複雜 JSON
        raw4 = '<Thought>\nLet me think...\n</Thought>```json\n{"key": "value"}\n```'
        self.assertEqual(_clean_json_response(raw4), '{"key": "value"}')

    def test_adaptive_stepper_flow(self):
        """測試適應性測驗完整端到端流程：Start -> Next (Round 2) -> Next (Round 3) -> Next (Finish) -> Report。"""
        
        # 1. 啟動測試 (Round 1)
        res_start = self.client.post("/api/inspector/adaptive/start", 
                                     data=json.dumps({"lang": "en"}),
                                     content_type="application/json")
        self.assertEqual(res_start.status_code, 200)
        start_data = res_start.get_json()
        self.assertTrue(start_data["ok"])
        self.assertEqual(start_data["round"], 1)
        self.assertEqual(start_data["difficulty"], "easy")
        self.assertEqual(start_data["font_size"], 16)
        self.assertIn("quiz", start_data)

        # 模擬 Round 1 閱讀表現 (回答正確，低回看)
        history = [
            {
                "round": 1,
                "difficulty": "easy",
                "font_size": 16,
                "line_width": 650,
                "line_height": 1.6,
                "quiz_score": 2,
                "quiz_total": 2,
                "regression_rate": 0.05,
                "wpm": 240.0
            }
        ]

        # 2. 請求 Round 2 參數
        res_r2 = self.client.post("/api/inspector/adaptive/next",
                                  data=json.dumps({
                                      "lang": "en",
                                      "current_round": 1,
                                      "history": history
                                  }),
                                  content_type="application/json")
        self.assertEqual(res_r2.status_code, 200)
        r2_data = res_r2.get_json()
        self.assertTrue(r2_data["ok"])
        self.assertEqual(r2_data["round"], 2)
        # 由於 Round 1 答對且回看低，Round 2 難度應升為 medium
        self.assertEqual(r2_data["difficulty"], "medium")
        self.assertEqual(r2_data["font_size"], 14) # medium English font size is 14

        # 模擬 Round 2 閱讀表現 (回答正確，低回看)
        history.append({
            "round": 2,
            "difficulty": "medium",
            "font_size": 14,
            "line_width": 550,
            "line_height": 1.5,
            "quiz_score": 2,
            "quiz_total": 2,
            "regression_rate": 0.08,
            "wpm": 210.0
        })

        # 3. 請求 Round 3 參數
        res_r3 = self.client.post("/api/inspector/adaptive/next",
                                  data=json.dumps({
                                      "lang": "en",
                                      "current_round": 2,
                                      "history": history
                                  }),
                                  content_type="application/json")
        self.assertEqual(res_r3.status_code, 200)
        r3_data = res_r3.get_json()
        self.assertTrue(r3_data["ok"])
        self.assertEqual(r3_data["round"], 3)
        # 由於 Round 2 答對，Round 3 難度應升為 hard
        self.assertEqual(r3_data["difficulty"], "hard")
        self.assertEqual(r3_data["font_size"], 12) # hard English font size is 12

        # 模擬 Round 3 閱讀表現 (回答錯誤，高回看)
        history.append({
            "round": 3,
            "difficulty": "hard",
            "font_size": 12,
            "line_width": 450,
            "line_height": 1.4,
            "quiz_score": 0,
            "quiz_total": 2,
            "regression_rate": 0.35,
            "wpm": 90.0
        })

        # 4. 結束測驗判定
        res_fin = self.client.post("/api/inspector/adaptive/next",
                                   data=json.dumps({
                                       "lang": "en",
                                       "current_round": 3,
                                       "history": history
                                   }),
                                   content_type="application/json")
        self.assertEqual(res_fin.status_code, 200)
        fin_data = res_fin.get_json()
        self.assertTrue(fin_data["ok"])
        self.assertTrue(fin_data["is_finished"])

        # 5. 產生綜合報告與排版建議
        res_rep = self.client.post("/api/inspector/adaptive/report",
                                   data=json.dumps({
                                       "lang": "en",
                                       "history": history,
                                       "participant_id": "tester-stepper",
                                       "persist": False
                                   }),
                                   content_type="application/json")
        self.assertEqual(res_rep.status_code, 200)
        rep_data = res_rep.get_json()
        self.assertTrue(rep_data["ok"])
        self.assertIn("report_md", rep_data)
        
        # 驗證最佳版面排版建議：應取閱讀效率（WPM）最高、回看率低的 round (即 Round 1 或 2，而非 Round 3 艱深且高認知負荷的 hard 版面)
        summary = rep_data["summary"]
        self.assertEqual(summary["optimal_font_size"], 16) # Round 1 efficiency: 240 * (2/2) * 0.95 = 228; Round 2: 210 * 1 * 0.92 = 193.2; Round 1 is optimal
        self.assertEqual(summary["optimal_line_width"], 650)
        self.assertEqual(summary["optimal_line_height"], 1.6)

if __name__ == "__main__":
    unittest.main()
