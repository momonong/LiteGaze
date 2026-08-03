import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.cognitive_inspector.inspector import CognitiveInspector
from web import create_app


class TestCognitiveInspectorUnit(unittest.TestCase):
    def setUp(self):
        self.inspector = CognitiveInspector(sample_rate_hz=8) # 1 tick = 125ms

    def test_empty_gaze_history(self):
        """測試空眼動紀錄時的邊界行為，應回傳預設的零值結構。"""
        result = self.inspector.analyze([], lang="en")
        self.assertEqual(result["summary"]["total_fixations"], 0)
        self.assertEqual(result["user_profile"]["reading_ability_score"], 0)
        self.assertEqual(result["user_profile"]["english_proficiency_score"], 0)
        self.assertEqual(result["user_profile"]["fatigue_level"], "none")

    def test_single_fixation(self):
        """測試僅有單一注視點的情況。"""
        gaze_history = [
            {"word": "Hello", "index": 0, "confidence": "high", "timestamp_ms": 1000}
        ]
        result = self.inspector.analyze(gaze_history, lang="en")
        summary = result["summary"]
        profile = result["user_profile"]
        
        self.assertEqual(summary["total_fixations"], 1)
        self.assertEqual(summary["unique_words_read"], 1)
        self.assertEqual(summary["regression_count"], 0)
        self.assertEqual(summary["reread_count"], 0)
        self.assertEqual(profile["fatigue_level"], "low")

    def test_fluent_reading_flow(self):
        """測試流暢閱讀（無回看、無長注視、順序閱讀），應獲得高閱讀能力分數、低認知負荷與高注意力穩定度。"""
        gaze_history = []
        words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        # 順序產生 9 個單字的注視，每個單字停留 1 個 tick (125ms)
        for i, word in enumerate(words):
            gaze_history.append({
                "word": word,
                "index": i,
                "confidence": "high",
                "timestamp_ms": 1000 + i * 150
            })
            
        result = self.inspector.analyze(gaze_history, lang="en")
        summary = result["summary"]
        profile = result["user_profile"]

        self.assertEqual(summary["regression_count"], 0)
        self.assertEqual(summary["reread_count"], 0)
        self.assertEqual(summary["unique_words_read"], 9)
        self.assertGreaterEqual(profile["reading_ability_score"], 70)
        self.assertLess(profile["cognitive_load_index"], 40)
        self.assertEqual(profile["attention_index"], 100)

    def test_regressions_and_rereads(self):
        """測試包含回看（Regression）與重讀（Reread）的困難閱讀情境。"""
        # 軌跡順序: 0 -> 1 -> 2 -> 3 -> 1 (回看 1) -> 2 (重讀 2) -> 4
        gaze_history = [
            {"word": "word0", "index": 0, "confidence": "high", "timestamp_ms": 1000},
            {"word": "word1", "index": 1, "confidence": "high", "timestamp_ms": 1150},
            {"word": "word2", "index": 2, "confidence": "high", "timestamp_ms": 1300},
            {"word": "word3", "index": 3, "confidence": "high", "timestamp_ms": 1450},
            # 回看 word1
            {"word": "word1", "index": 1, "confidence": "high", "timestamp_ms": 1600},
            # 重讀 word2
            {"word": "word2", "index": 2, "confidence": "high", "timestamp_ms": 1750},
            {"word": "word4", "index": 4, "confidence": "high", "timestamp_ms": 1900},
        ]
        
        result = self.inspector.analyze(gaze_history, lang="en")
        summary = result["summary"]
        profile = result["user_profile"]

        # 軌跡中 3->1 是一次回看 (f.index < last_index)
        self.assertGreaterEqual(summary["regression_count"], 1)
        # word1 與 word2 被二次讀取
        self.assertGreaterEqual(summary["reread_count"], 1)
        # 注意力穩定指數應有所下降
        self.assertLess(profile["attention_index"], 100)

    def test_english_proficiency_rare_vs_common(self):
        """測試英語熟練度指標：卡在罕見單字代表英文水準高，卡在常見基礎字代表英文水準低。"""
        # Case A: 受阻（長注視）發生在罕見專業字 (如 "surprisal", zipf freq 很低)
        gaze_rare = [
            {"word": "surprisal", "index": 0, "confidence": "high", "timestamp_ms": 1000},
            {"word": "surprisal", "index": 0, "confidence": "high", "timestamp_ms": 1120},
            {"word": "surprisal", "index": 0, "confidence": "high", "timestamp_ms": 1240},
            {"word": "surprisal", "index": 0, "confidence": "high", "timestamp_ms": 1360}, # 4 ticks = 500ms > 350ms (長注視)
        ]
        result_rare = self.inspector.analyze(gaze_rare, lang="en")
        score_rare = result_rare["user_profile"]["english_proficiency_score"]

        # Case B: 受阻（長注視）發生在極常見基礎字 (如 "the", zipf freq 極高)
        gaze_common = [
            {"word": "the", "index": 0, "confidence": "high", "timestamp_ms": 1000},
            {"word": "the", "index": 0, "confidence": "high", "timestamp_ms": 1120},
            {"word": "the", "index": 0, "confidence": "high", "timestamp_ms": 1240},
            {"word": "the", "index": 0, "confidence": "high", "timestamp_ms": 1360}, # 4 ticks = 500ms (長注視)
        ]
        result_common = self.inspector.analyze(gaze_common, lang="en")
        score_common = result_common["user_profile"]["english_proficiency_score"]

        # 預期卡在罕見字得到的分數（代表使用者懂常見字，只是卡生字）應高於卡在常見字
        self.assertGreater(score_rare, score_common)

    def test_fatigue_level_stability(self):
        """測試疲勞度評估：前後半段注視時長比例變化應正確映射至低/中/高疲勞標籤。"""
        # Case A: 前後半段注視時長完全相同 -> 低疲勞
        gaze_low = []
        for i in range(10):
            gaze_low.append({"word": f"w{i}", "index": i, "confidence": "high", "timestamp_ms": 1000 + i * 150})
        res_low = self.inspector.analyze(gaze_low, lang="en")
        self.assertEqual(res_low["user_profile"]["fatigue_level"], "low")

        # Case B: 後半段注視點變多/拉長（模擬後半段疲勞加工速度變慢）-> 高疲勞
        # 前半段: 5個單字，每個停留 1 tick
        # 後半段: 5個單字，每個停留 4 ticks (大於 1.20 比例)
        gaze_high = []
        t = 1000
        for i in range(5):
            gaze_high.append({"word": f"w{i}", "index": i, "confidence": "high", "timestamp_ms": t})
            t += 150
        for i in range(5, 10):
            # 連續 4 次命中同一個單字，使該 fixation 停留時長拉長
            for tick in range(4):
                gaze_high.append({"word": f"w{i}", "index": i, "confidence": "high", "timestamp_ms": t})
                t += 120
            t += 150

        res_high = self.inspector.analyze(gaze_high, lang="en")
        self.assertEqual(res_high["user_profile"]["fatigue_level"], "high")

    def test_video_mode_tick_estimation(self):
        """測試低取樣率/影片離線模式 (約 800ms) 下的動態 Tick 與聚合判定。"""
        gaze_history = [
            {"word": "word0", "index": 0, "confidence": "high", "timestamp_ms": 0},
            {"word": "word0", "index": 0, "confidence": "high", "timestamp_ms": 800},
            {"word": "word1", "index": 1, "confidence": "high", "timestamp_ms": 1600},
            {"word": "word2", "index": 2, "confidence": "high", "timestamp_ms": 2400},
            {"word": "word3", "index": 3, "confidence": "high", "timestamp_ms": 3200},
            {"word": "word4", "index": 4, "confidence": "high", "timestamp_ms": 4000},
        ]
        result = self.inspector.analyze(gaze_history, lang="en")
        summary = result["summary"]
        
        # 總注視次數應為 5 (word0 的 2 個 hits 被聚合，因為 800ms < threshold = max(350, 800*1.5=1200ms))
        self.assertEqual(summary["total_fixations"], 5)
        # 總注視時間 (total_dwell_time_ms) 應為 6 * 200ms = 1200ms
        self.assertEqual(summary["total_dwell_time_ms"], 1200)
        # 平均注視時間應為 1200 / 5 = 240ms
        self.assertEqual(summary["avg_fixation_duration_ms"], 240.0)



class TestCognitiveInspectorIntegration(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory(prefix="lexigaze-inspector-")
        self.addCleanup(self._temp_dir.cleanup)
        self.reports_dir = Path(self._temp_dir.name)
        self._reports_patcher = patch(
            "web.routes.inspector.REPORTS_DIR",
            self.reports_dir,
        )
        self._reports_patcher.start()
        self.addCleanup(self._reports_patcher.stop)
        self.app = create_app({
            "TESTING": True,
            "LEXIGAZE_BLUEPRINTS": ("inspector",),
        })
        self.client = self.app.test_client()
        self.reports_to_clean = []

    def tearDown(self):
        # 清除測試時產生的 Markdown 報告檔案
        for p in self.reports_to_clean:
            path = Path(p)
            if path.exists():
                path.unlink()

    def test_flask_analyze_endpoint(self):
        """測試 POST /api/inspector/analyze 端點整合。"""
        payload = {
            "gaze_history": [
                {"word": "Neuro", "index": 0, "confidence": "high", "timestamp_ms": 1000},
                {"word": "Symbolic", "index": 1, "confidence": "high", "timestamp_ms": 1150}
            ],
            "lang": "en"
        }
        res = self.client.post("/api/inspector/analyze", 
                               data=json.dumps(payload),
                               content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertIn("analysis", data)
        self.assertIn("user_profile", data["analysis"])
        self.assertIn("summary", data["analysis"])

    def test_flask_report_endpoint_persisted(self):
        """測試 POST /api/inspector/report 生成報告端點，並驗證 Markdown 檔案是否成功寫入磁碟。"""
        payload = {
            "gaze_history": [
                {"word": "Neuro", "index": 0, "confidence": "high", "timestamp_ms": 1000},
                {"word": "Symbolic", "index": 1, "confidence": "high", "timestamp_ms": 1150}
            ],
            "participant_id": "test_integration_user",
            "lang": "en",
            "persist": True
        }
        res = self.client.post("/api/inspector/report", 
                               data=json.dumps(payload),
                               content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertIn("report_md", data)
        self.assertIn("report_path", data["analysis"])
        
        # 驗證實體檔案
        rel_path = data["analysis"]["report_path"]
        abs_path = self.reports_dir / Path(rel_path).name
        self.assertTrue(abs_path.exists())
        
        # 標記以供 tearDown 清理
        self.reports_to_clean.append(abs_path)

        # 讀取檔案，驗證包含受試者 ID
        content = abs_path.read_text(encoding="utf-8")
        self.assertIn("test_integration_user", content)
        self.assertIn("認知能力指標評估", content)

    def test_flask_reports_lifecycle_endpoints(self):
        """測試認知診斷報告完整的生命週期 API：建立 -> 列表 -> 讀取 -> 刪除。"""
        # 1. 建立並儲存報告
        payload = {
            "gaze_history": [
                {"word": "Test", "index": 0, "confidence": "high", "timestamp_ms": 1000}
            ],
            "participant_id": "lifecycle_test_user",
            "lang": "en",
            "persist": True
        }
        res = self.client.post("/api/inspector/report", 
                               data=json.dumps(payload),
                               content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        report_path = data["analysis"]["report_path"]
        filename = Path(report_path).name
        
        # 2. 列表報告，應包含剛剛建立的報告
        res_list = self.client.get("/api/inspector/reports")
        self.assertEqual(res_list.status_code, 200)
        list_data = res_list.get_json()
        self.assertTrue(list_data["ok"])
        reports = list_data["reports"]
        filenames = [r["filename"] for r in reports]
        self.assertIn(filename, filenames)
        
        # 3. 讀取報告內容
        res_get = self.client.get(f"/api/inspector/reports/{filename}")
        self.assertEqual(res_get.status_code, 200)
        get_data = res_get.get_json()
        self.assertTrue(get_data["ok"])
        self.assertIn("lifecycle_test_user", get_data["markdown"])
        
        # 4. 刪除報告
        res_del = self.client.delete(f"/api/inspector/reports/{filename}")
        self.assertEqual(res_del.status_code, 200)
        del_data = res_del.get_json()
        self.assertTrue(del_data["ok"])
        
        # 驗證實體檔案已被刪除
        abs_path = self.reports_dir / Path(report_path).name
        self.assertFalse(abs_path.exists())

    def test_flask_analyze_invalid_payload(self):
        """測試傳入無效 Gaze 格式時的 API 錯誤處理機制。"""
        res = self.client.post("/api/inspector/analyze", 
                               data=json.dumps({"gaze_history": "not_an_array"}),
                               content_type="application/json")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["ok"])
        self.assertIn("error", data)

if __name__ == "__main__":
    unittest.main()
