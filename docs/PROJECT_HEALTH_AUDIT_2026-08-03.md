# LexiGaze 專案健檢與推進路線圖（2026-08-03）

本文件依據目前 repository 的產品程式、研究腳本、測試與既有報告整理。目標是先修正會影響結果可信度與可重現性的問題，再安排不浪費 GPU 的推進順序。

## 結論摘要

LexiGaze 已具備完整的端到端雛形：Webcam gaze、個人化校正、語言模型認知特徵、序列校正、RDS 融合與閱讀者報告都已串起來。下一階段的主要瓶頸不是再增加演算法，而是：

1. 統一實驗版本與數字來源，確保每個結果都可由同一份設定重現。
2. 把 heuristic 的能力評分與真實受試者 ground truth 分開驗證。
3. 將公開 tunnel、模型推論及大型上傳加上安全與資源背壓。
4. 把產品 runtime、研究實驗與歷史 archive 的邊界再切清楚。

## 本輪已完成

- 新增共享的 PyTorch 裝置政策；`LEXIGAZE_DEVICE=cpu` 時，產品主路徑不探測或配置 CUDA。
- Gaze inference、個人化訓練、Cognition pipeline 與獨立 UniGaze server 共用相同裝置解析。
- Health API 回報設定的裝置政策，以及已載入 Cognition 模型實際所在裝置。
- 修正 Cognitive Inspector 將所有已讀單字誤判成 struggle 的邏輯偏差，並加入回歸測試。
- Adaptive 測試固定停用真實 Gemini key，避免單元測試意外呼叫外部 API 與消耗額度。
- 將 `uv.lock` 與必要的 MediaPipe `face_landmarker.task` 納入版本控制範圍。
- 補上程式直接使用但未宣告的 `requests` 依賴。
- 修正文檔中將主線 UniGaze 誤寫成 ONNX Runtime 的內容；現行主路徑是 PyTorch `unigaze`。
- 建立共用實驗 manifest，記錄 Git、執行腳本、資料集、設定、套件、硬體、指標與輸出檔 SHA-256。
- 將 fusion 與整體系統比較改為固定 seed、區域 RNG、絕對路徑與原子化 manifest 輸出。
- 修正 GECO pp01 / Trial 5 報告硬編碼為 157 筆的偏差；目前兩份輸入與 merge 結果均為 156 筆。
- 先 commit 鎖定 GECO 跨受試者 protocol，再以 37 人、5,892 participant-trials 完成不使用測試折調參的 CPU-only 雙重 holdout。
- 識別兩條關鍵洩漏風險：單一 trial fusion 的 gaze dwell 由評估目標 TRT 建構；`cognitive_mass` 的現行 extraction path 可包含以 GECO TRT 訓練的 XGBoost/Ridge。

## 本輪成果與驗證紀錄

執行分支：`codex/feat/runtime-resource-guardrails`

所有測試與 benchmark 均設定 `LEXIGAZE_DEVICE=cpu` 與空白 `CUDA_VISIBLE_DEVICES`。實驗腳本只透過 `nvidia-smi` 唯讀記錄 GPU 型號與驅動，沒有初始化 CUDA 或載入 GPU 模型。

| 項目 | 結果 | 可追溯產物 |
| --- | --- | --- |
| CPU-safe 單元與 API 測試 | 22/22 通過 | `scripts/test_device_policy.py`、`scripts/test_cognitive_inspector.py`、`scripts/test_fusion_routes.py`、`scripts/test_adaptive_stepper.py`、`scripts/test_experiment_manifest.py` |
| 靜態驗證 | `compileall` 通過；Ruff `E9,F63,F7,F82` 通過 | 本分支 Python 變更 |
| 依賴解析 | `uv lock --check` 通過；`uv sync --locked --offline --dry-run` 無變更 | `uv.lock` |
| Fusion benchmark | 156 筆；最佳 Spearman 為 RRF，$\rho=0.6569$；最佳 Pearson 為 Sigmoid，$r=0.7503$ | `output/fusion_experiment_manifest.json`、`output/fusion_experiment_report.md` |
| Joint system benchmark | 156 筆；Viterbi + EM gaze accuracy 96.79%；STOCK-T v3 + CogMass + Bayesian 的 RDS $\rho=0.4267$ | `output/demo_system_comparison_manifest.json`、`output/demo_system_comparison.csv` |
| New-reader + new-trial double holdout | 37 人、5,892 participant-trials；text-only Ridge macro $\rho=0.1216$，95% CI $[0.0926, 0.1513]$；未勝過 word length $\rho=0.1225$ | `output/geco_generalization_manifest.json`、`docs/GECO_GENERALIZATION_EXECUTION_LOG_2026-08-03.md` |
| New-reader + known-passage LOSO | other-reader duration prior $\rho=0.3105$；fixation-rate prior AUC $0.7766$ | `output/geco_generalization_summary.json` |

Manifest schema 版本為 1。每份 manifest 都包含 benchmark entry point 與 manifest helper 的 SHA-256、資料集與產物 SHA-256、seed、參數、套件版本、device policy、Git branch/HEAD/dirty 狀態，以及 tracked diff SHA-256。這讓尚未 commit 的研究執行也能回溯到實際程式內容。

## 優先風險與建議

| 優先級 | 面向 | 現況與證據 | 建議動作 | GPU 成本 |
| --- | --- | --- | --- | --- |
| P0 | 實驗可追溯性 | 新 benchmark 已有 manifest；但 `PROJECT_OVERVIEW.md` 的 78.21% 與 `docs/NeurIPS/RESULT.md` 的 92.31% 仍缺少原始版本資訊 | 替歷史結果補版本標籤；之後所有實驗沿用 manifest schema | 無 |
| P0 | 泛化效度 | 37 人雙重 holdout 已完成；text-only Ridge 僅 $\rho=0.1216$ 且未超越 word length，證明單一 trial 高相關不可代表泛化 | 凍結 GECO v1.1 測試；新特徵只在獨立 development data 開發，下一次 confirmatory test 改用預註冊的跨 corpus zero-shot | 低至中 |
| P0 | 公開 tunnel | `run.py --tunnel` 可公開 Flask API；目前包含刪除、訓練、報告及最高 500 MB 上傳端點，沒有一致的存取控制 | tunnel 模式要求 bearer token；對推論、訓練與上傳加入 rate limit、併發上限及可設定的大小限制 | 無 |
| P0 | 指標效度 | Inspector 的 WPM 以 fixation dwell 總和估算，疲勞以文章前後半 fixation 比值估算，容易混入 gaze loss 與後半內容難度 | 前端傳入 session wall-clock、coverage 與 lost-sample rate；疲勞實驗採 counterbalanced passages | 無至低 |
| P1 | GPU 背壓 | 模型採 lazy cache，但同時請求仍可能並行佔用 VRAM；研究腳本也有獨立的自動 CUDA 判斷 | 每種模型設 semaphore/queue；記錄 allocated/reserved VRAM、batch size、latency；研究腳本增加 `--device` | 低 |
| P1 | 測試架構 | 測試主要散在 `scripts/test_*.py`，包含 GPU、外部 API、整合與單元測試，無一致 marker 或預設安全集合 | 建立 `tests/`，區分 `unit`、`integration`、`network`、`gpu`；CI 預設只跑 CPU/unit | 無 |
| P1 | 模組邊界 | `web/routes/inspector.py` 約 1,096 行，`word_track.html` 約 331 KB；API、prompt、fallback data 與 schema 混在一起 | 拆成 service/schema/prompt/store；前端拆成 ES modules 與獨立樣式 | 無 |
| P1 | 全域副作用 | `web/routes/cognitive.py` 在 import 時替換 `sys.stdout`、`builtins.print` 與 `warnings.warn` | 改用標準 logging 與單一 console encoding 初始化，不修改全域 builtins | 無 |
| P1 | 歷史程式邊界 | 現行 Web 仍從 `archive/shengwen` 提供部分 static；獨立 FastAPI server 使用未宣告的 FastAPI/Uvicorn | 決定獨立 server 是正式 optional extra 或 archive；正式 runtime 不再依賴 `archive/` | 無 |
| P2 | 依賴維護 | `pyproject.toml` 與 `requirements.txt` 重複維護且最低版本不同 | 以 `pyproject.toml` + committed `uv.lock` 為唯一來源，requirements 由工具輸出 | 無 |

## 建議研究順序

### A. 完全不使用 GPU

1. 對齊 78.21%、88.63%、90.49%、92.31% 各自的資料範圍與演算法版本。
2. 建立與 GECO v1.1 隔離的 development split；禁止根據凍結 test 結果調權重。
3. 補齊 Inspector 的 gaze loss 與稀疏取樣邊界測試，並加入 coverage 指標。
4. 加上 tunnel authentication、request validation 與資源上限。

### B. 低 GPU 預算

1. 每個模型只跑一個固定短樣本，量測 cold/warm latency 與 peak VRAM。
2. 比較 `cpu`、`cuda` 與不同 batch size；先建立 Pareto frontier，不做全面 grid search。
3. 對 2 位受試者 × 2 trials 做 smoke benchmark，確認 pipeline 後再擴大。

### C. 需排程的高 GPU 工作

1. 預註冊跨 corpus zero-shot Cognition 評估；優先採用未參與訓練的 PROVO 或其他閱讀語料。
2. 真實 webcam drift 條件的多輪個人化校正，而非只使用 synthetic jitter。
3. 只有當誤差分析顯示模型容量是瓶頸時，才進行重新訓練或大型超參數搜尋。

## 建議的完成判準

- 任一論文或產品數字都能由 manifest 對應到 commit、輸入資料與輸出檔。
- CPU-only 測試不需要模型下載、外部 API 或 CUDA，且可在 CI 穩定完成。
- tunnel 模式未帶 token 時，無法觸發寫入、刪除、訓練或昂貴推論。
- health/benchmark 能同時回報 latency、實際 device 與 VRAM 峰值。
- Inspector 的 reading ability、proficiency、fatigue 至少有 held-out participant 的效度結果，不只依賴手工 threshold。
