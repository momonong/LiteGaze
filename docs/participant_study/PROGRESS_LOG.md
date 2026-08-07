# Participant Study Progress Log

## 2026-08-07 — 受試者 readiness milestone

### 問題

舊流程讓使用者自由輸入 participant name，沒有版本化同意、理解確認、退出 credential 或正式收案閘；公開 `/gaze` 同時帶有模型、資料集、訓練與刪除功能。校準上傳失敗仍可能繼續，完整影片可錄製，且 UI 可能顯示完成但資料不完整。閱讀測驗結果 token 只綁 passage，未綁 assessment、round、participant 或題庫版本。

### 決策

- 把 `dry_run` 與 `pilot` 分開；預設永遠是 dry run。
- v1 不保存完整影片；正式 pilot 的 calibration／inference 都 CPU-only。
- 使用伺服器 pseudonym、一次性 invite、版本化 consent receipt 與 withdrawal code。
- 專用 participant UI + public endpoint allowlist，不再把研究者 UI 當受試者入口。
- 固定排版做 ability-routing pilot；排版效應另立實驗，避免 construct confounding。
- 未校準題庫只回報 session evidence，所有 latent-trait claims abstain。
- 真實資料集與 export 延後到倫理、外部效標、schema 和 holdout 設計完成後。

### 完成的工程

- 新增 protocol/store/calibration audit、study routes、consent UI、assessment UI。
- calibration 品質／訓練／清除任一失敗即刪除 dataset；中斷資料 TTL cleanup。
- adaptive assignment/result 簽章、重試／換題／跳輪／跨 session 防護與 restart resume。
- withdrawal 以 minimal tombstone 取代整個 session directory。
- legacy public tunnel 停用；新 tunnel 必須通過 pilot audit，使用 participant-only surface、Waitress 與 ngrok `--inspect=false`。
- 新增 offline readiness audit、one-time invite generator、測試與治理文件。

### 當前結論

工程 dry run 可驗證；正式 pilot 故意保持 locked。不可由程式代填的 blocker 列於 `READINESS_CHECKLIST.md`。尚未收集或修改任何既有受試者 dataset，也未為這一里程碑使用 GPU。

### 驗證證據

- Participant/adaptive/app focused suite：21 tests passed。
- Final participant/adaptive rerun after browser verification：18 tests passed。
- Full offline quality gate：99 tests passed；0 failure、0 error、0 skip。
- Safeguards：network blocked、subprocess blocked、provider credentials cleared、artifact changes `[]`、Torch not imported、`CUDA_VISIBLE_DEVICES=-1`。
- GPU before/after：RTX 5090 utilization `0%`，memory `80/24463 MiB`，前後完全相同。
- Readiness audit：`dry_run_ready=true`、`pilot_ready=false`；未完成項目由 machine-readable audit 列出。
- In-app browser dry run：必要同意與理解確認預設皆未勾選；建立 pseudonymous session 後可跨重新整理續接，並依序通過 `consented → system_check_passed → calibration_complete → assessment_in_progress → completed`。
- Dry-run browser safeguards：未要求攝影機權限、未使用 local/session storage、console 0 errors；撤回後回傳 receipt 並清除 session，`deleted_scopes=[]` 符合不產生真人研究資料的設計。
- 未具有效 session 的 `/study/assessment` 會 redirect 回 `/study`，無法繞過受試者流程直接進入評量。
