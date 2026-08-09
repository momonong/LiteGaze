# Researcher Runbook

## 目前新增允許：研究團隊本人的 localhost rehearsal

這條路徑會保存真實但 development-only 的衍生資料，與不收資料的 `dry_run`
不同。只允許研究團隊本人做 moderated software rehearsal；在倫理／豁免判定、
正式聯絡資訊與兩位獨立文章 reviewer 完成前，不得把邀請碼交給朋友或外部受試者。

### A. 已加密儲存

以系統管理員 PowerShell 確認實際資料位置已加密，例如檢查磁碟的
`ProtectionStatus=On` 與完整加密狀態。若無法確認，不要使用
`--confirm-encrypted-storage`。在不影響既有 8080 server 的情況下，可使用 8098：

   ```powershell
   $studyRoot = (Resolve-Path ".").Path
   $dataLocation = Join-Path $studyRoot "data"
   .\.venv\Scripts\python.exe -X utf8 -m scripts.run_general_collection_rehearsal `
     --root $studyRoot `
     --port 8098 `
     --data-location $dataLocation `
     --retention-days 7 `
     --raw-frame-retention-hours 1 `
     --create-invite-pairs 1 `
     --acknowledge-development-only `
     --confirm-encrypted-storage
   ```

### B. 研究者本人、未加密、手動保存的 development data

這是明確接受的較低資料治理等級，不是假裝已加密。只允許資料擁有者本人，
不設自動刪除期限，由研究者手動保留或刪除；仍禁止外部受試者、Git/GitHub、
公開 tunnel 與任何 confirmation／正式結果升格。推論用的逐次閱讀影格不另外
落盤；若本人在同意頁另行勾選，可逐篇保存「按下開始」至「完成閱讀」之間的
無音訊 webcam 影片。成功或失敗的校正影像仍須立即清除，中斷校正最長保留一小時。

```powershell
$studyRoot = (Resolve-Path ".").Path
$dataLocation = Join-Path $studyRoot "data"
.\.venv\Scripts\python.exe -X utf8 -m scripts.run_general_collection_rehearsal `
  --root $studyRoot `
  --port 8098 `
  --data-location $dataLocation `
  --raw-frame-retention-hours 1 `
  --create-invite-pairs 1 `
  --acknowledge-development-only `
  --allow-unencrypted-self-development-data `
  --retain-until-manual-deletion
```

1. Plaintext invite 只會在 console 顯示一次；不要 commit、貼到 issue 或傳給
   非研究團隊人員。開啟 `http://127.0.0.1:8098/study`，本次只使用 visit 1 code。
   若要保存本人的閱讀影片，必須親自在 consent 頁勾選獨立 optional scope；它預設
   不勾選。瀏覽器只建立 video track，不請求或保存 audio。
2. Visit 2 必須在 visit 1 完成後 18–72 小時，以同一裝置類別與瀏覽器 family
   使用配對 code；server 會拒絕過早、過晚或 visit 1 未完成的 code。
3. 每次完成後執行：

   ```powershell
   .\.venv\Scripts\python.exe -X utf8 -m scripts.audit_general_collection_readiness --root $studyRoot
   .\.venv\Scripts\python.exe -X utf8 -m scripts.export_general_collection_dataset --root $studyRoot
   ```

   Export 是 private pseudonymous bundle，預設排除未完成／已撤回 session，且
   manifest 永遠標示 `formal_promotion_allowed=false`。Raw 影片不會複製進 bundle；
   `reading_video_index.csv` 只保存來源相對路徑、hash、文章／round 與同步資訊。
   不要 commit export 或 `data/` 下的影片。

4. Rehearsal 只回答 completion、missingness、quality、abstention、resume、export
   與 withdrawal 是否正常。影片可供眼動模組開發、動作魯棒性與同步除錯；不得用
   同一人／同一 12 篇文章反覆調參後，再把相同影片回報為模型成效。模型比較至少
   需要另一次未參與調參的 capture session，正式 claim 仍須獨立 confirmation cohort。

## 目前允許做的事：dry run

1. 安裝依賴：`uv sync`。
2. 執行工程測試：

   ```powershell
   .\.venv\Scripts\python.exe -X utf8 -m unittest scripts.test_participant_study scripts.test_adaptive_stepper -v
   .\.venv\Scripts\python.exe -X utf8 -m scripts.audit_participant_study_readiness --target dry-run
   ```

3. 本機啟動：`.\.venv\Scripts\python.exe -X utf8 run.py`，開啟 `http://127.0.0.1:8080/study`。
4. 完整走一次 dry run：同意、理解題、收據下載、系統檢查、模擬校準、模擬評量、退出。
5. 不使用朋友資料、不分享 tunnel、不把 dry run 當研究收案。

## 正式 pilot 前逐項解鎖

1. 由所屬機構／合格審查者取得 `approved` 或正式 `exempt_determination` 與 reference；不要自行判定。
2. 補齊研究主持人、研究聯絡、受試者權益獨立聯絡與最終同意文字。
3. 在招募前選定外部效標、確認使用授權並預註冊主要分析、participant split、item/text split、排除規則與停止條件。
4. 依 `INDEPENDENT_CAPTURE_PLAN.md` 建立正式 numeric-slot plan，在未查看任何
   participant outcome 前提交 freeze commit，並確認 `--target collection`
   exit code 0。Example template 不得直接用於招募。
5. 核准 HTTPS/network processor；關閉 local inspection 與 dashboard full-body capture，記錄第三方 metadata 留存。
6. 確認加密 storage、位置、存取角色、保留天數、備份與 raw-frame TTL（1–24 小時）。
7. 產生至少 32 字元的 adaptive signing key 及獨立 researcher API key；只放本機 secret store／`.env`。
8. 經審查後把 canonical JSON 的 `collection_status` 由 `dry_run_only` 改為 `approved_for_pilot`，更新版本與 digest，重新保存正式同意文件。
9. 執行：

   ```powershell
   .\.venv\Scripts\python.exe -X utf8 -m scripts.audit_participant_study_readiness --target pilot
   ```

   只有 exit code 0 才能繼續。

10. 建立一次性邀請碼：

   ```powershell
   .\.venv\Scripts\python.exe -X utf8 -m scripts.create_pilot_invites --count 1
   ```

11. 啟動 participant-only tunnel：

    ```powershell
    .\.venv\Scripts\python.exe -X utf8 run.py --study-tunnel
    ```

    legacy `--tunnel` 已停用。正式 tunnel 使用 Waitress、公開 API allowlist 與 ngrok local inspection disabled；仍須確認帳號層 full capture 關閉。

## 每位受試者前後

- 前：確認版本／digest、剩餘一次性 invite、磁碟加密、相機、CPU 餘裕、無殘留 raw directory、同意與退出聯絡可用。
- 中：不要看受試者答案或提示；如不適立即停止。不可為通過品質閘而手改資料。
- 後：確認狀態 `completed` 或 `withdrawn`、校準影像已清除、事件與失敗原因完整；關閉 tunnel。
- 任何 incident：停止新增 invite，保存不含敏感 body 的時間線，依 `INCIDENT_RESPONSE.md` 處理。

## 明確禁止

- 未通過 pilot audit 仍找朋友「先試幾個」。
- 以姓名、email 或社群帳號作 participant ID。
- 錄製／下載完整 calibration video。
- 反覆看同一批朋友或同一題組的結果後調參，再把它們當測試集。
- 對外宣稱 CEFR、英文／認知能力、注意力或診斷結果。
