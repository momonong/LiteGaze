# 任務：實作「感知 ✕ 認知」雙模態數據融合管線 (Fusion Orchestrator)

我們正在建構 LexiGaze 系統的數據融合核心。請嚴格遵守以下架構規範，協助我實作一個能自動串接多模組數據、並進行「重力捕捉 (Gravity Snap)」計算的融合引擎。

## 1. 架構佈局 (Architecture Layout)
- **程式入口**：請在 `scripts/fusion/` 目錄下建立一個 `orchestrator.py` 協調腳本。
- **配置目錄**：將所有開發用的 Instructions 存放在 `prompts/fusion/` 目錄中。
- **文件輸出**：所有產出的報告、技術文檔請統一輸出至 `docs/` 資料夾下。

## 2. 跨模組資料調度 (Data Orchestration)
請 AI 掃描 `chenghao/`, `weichi/` 與 `shengwen/` 的最新代碼，並執行以下動作：
- **感知端數據抓取**：從 `data/` 目錄中讀取受試者的眼動 log (JSONL 格式)。請確保能解析 `timestamp`、`gaze_x`、`gaze_y` 等欄位。
- **認知端特徵對齊**：從 `weichi/` 或對應的文字建模模組讀取最新版的單字認知權重資料。請確保能對齊所有細粒度屬性，包含：
  - `surprisal` (BERT 驚奇度), `entropy` (預測不確定性), `dependency_load`, `zipf_score`, `word_length`, `aoa_score`, `renyi_entropy`。
- **雙模態對齊邏輯**：實作一個對齊函數，將眼動軌跡中的 (x, y) 座標映射至文字模型的單字邊界框（Bounding Box）。需考慮大小寫一致性 (`.toLowerCase()`) 與連字號詞彙（Hyphenated words）的模糊比對規則。

## 3. 核心融合演算法實作 (RDS: Reading Difficulty Score)
在 `scripts/fusion/orchestrator.py` 中實作 RDS 計算邏輯：
1. **正規化處理**：對單次會話（Session）中的 `dwell_ms` 與 `fixation_count` 進行 Min-Max Scaling，使其落在 [0, 1] 區間。
2. **加權融合公式**：
   - `RDS = 0.35 * dwell_norm + 0.25 * fix_norm + 0.40 * load_score`
   - (其中 `load_score` 來自文字模型已正規化的認知分數)
3. **分級輸出**：
   - `RDS >= 0.70` → `"difficulty"`
   - `RDS 0.40 - 0.69` → `"attention"`
   - `RDS < 0.40` → `"fluent"`
4. **報告產出**：將計算好的 RDS 與原始的所有細粒度 linguistic 特徵，合併為一個完整的 JSON 報告，輸出至 `docs/fusion_reports/<session_id>.json`。

---

## 4. 前端資料對齊 (Frontend Pipeline)
請檢查 `chenghao/` 下的前端 JS 檔案，確保在發送資料時：
- 眼動計數（dwell/fixation）已正確累計並打包。
- 文字分析分數（`load_score` 等特徵）在 POST 請求中完整隨附，確保後端融合引擎有足夠的「大腦特徵」可用。

請協助我編寫 `scripts/fusion/orchestrator.py` 的核心程式碼，並說明我該如何觸發這個管線來完成實驗數據的融合。