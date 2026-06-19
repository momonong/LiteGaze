# Skill Builder 知識轉譯工具使用指南

`skill_builder.py` 是一個強大的自動化腳本，它的核心任務是扮演「知識轉譯者」。它可以讀取生硬的學術文獻，透過呼叫 Gemini AI 的強大理解能力，將這些知識轉換為結構化、可直接輔助 Vibe Coding 的 Markdown "Skill" 文件。

透過這個工具，你可以輕鬆建立專案專屬的領域知識庫，讓後續的 AI 診斷代理人 (Diagnostic Agent) 或程式碼生成更加精準！

## 1. 準備知識庫 (Knowledge Base)

在執行腳本之前，請確保專案根目錄下有以下的目錄結構。腳本預設會讀取 `knowledge/` 目錄下的特定模組（如：`web_cam`, `text_model`, `calibration`）：

```text
專案根目錄/
├── default_template.md       <-- Skill 輸出的標準 Markdown 模板
└── knowledge/
    └── web_cam/              <-- 知識模組名稱
        ├── papers/           <-- 放入相關的文獻純文字檔或 Markdown
        │   └── paper1.md
        └── questions.yaml    <-- 設定要讓 AI 針對該模組回答的問題 (Section -> Query)
```

### `questions.yaml` 範例：

```yaml
overview: "請根據文獻簡述眼動軌跡如何反映認知負荷？"
parameters: "請從文獻中提取眼動追蹤的核心參數（輸出 JSON 格式）。"
```

## 2. 設定環境變數

此腳本會呼叫 Gemini API 進行文本分析，因此你必須先設定好 API 金鑰：

1. 確保專案根目錄下有 `.env` 檔案。
2. 填入你的金鑰：`GEMINI_API_KEY="你的金鑰"`。

> **提示**：如果還沒有金鑰，請參考 [取得你的專屬 API Key指南](API_KEY.md)。

## 3. 執行腳本

打開終端機，在專案根目錄執行以下指令：

**執行全部模組 (預設)**：
```bash
python tutorial/skill_builder.py
```

**只執行特定模組 (例如：只執行 web_cam 模組)**：
```bash
python tutorial/skill_builder.py --module web_cam
# 或使用縮寫
python tutorial/skill_builder.py -m web_cam
```

**執行流程說明**：

1. **載入模組**：腳本會自動遍歷 `knowledge/` 下的模組資料夾。
2. **防呆檢查**：如果某個模組缺少 `questions.yaml` 或 `papers/` 裡面沒檔案，會友善跳過，不會當機。
3. **平行處理**：腳本使用了 `asyncio` 非同步機制，會同時把 YAML 裡的所有問題送給 Gemini 處理，大幅節省時間。
4. **組合輸出**：AI 產生的回答會替換掉 `default_template.md` 裡的佔位符（如 `{overview}`）。

## 4. 查看產出結果

執行成功後，專案根目錄下會自動建立一個 `skills/` 資料夾，裡面會包含產出的 Markdown 文件：

```text
skills/
├── skill_web_cam.md
├── skill_text_model.md
└── skill_calibration.md
```

這些檔案已經整理成了包含具體參數、公式與邏輯的精確內容，可以直接餵給其他的 AI Agent 作為 Prompt 的背景知識庫！

## 常見問題

- **出現 `API 錯誤`**：請檢查 `.env` 金鑰是否正確，或檢查網路連線。
- **出現 `找不到 default_template.md`**：請確認該模板檔案確實存在於專案根目錄，腳本依賴它來排版。

