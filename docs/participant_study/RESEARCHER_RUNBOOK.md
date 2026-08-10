# Researcher Runbook

## 目前新增允許：研究團隊本人的 localhost rehearsal

這條路徑會保存真實但 development-only 的衍生資料，與不收資料的 `dry_run`
不同。只允許研究團隊本人做 moderated software rehearsal；在倫理／豁免判定、
正式聯絡資訊與兩位獨立文章 reviewer 完成前，不得把邀請碼交給朋友或外部受試者。

### 本輪確認的隔離 worktree 啟動方式

本輪不用隔離 worktree 內的相對 `.venv` 啟動。程式碼從本分支的
worktree 執行，但依賴沿用主專案 `.venv` 的 site-packages，開發資料仍寫入
`D:\projects\lexigaze\data`。先在 PowerShell 設定：

```powershell
$codeRoot = 'D:\projects\lexigaze\.tmp\workspaces\webcam-gaze-measurement-ceiling-v1'
$studyRoot = 'D:\projects\lexigaze'
$dataLocation = Join-Path $studyRoot 'data'
$python = 'C:\Users\morris\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONPATH = 'D:\projects\lexigaze\.venv\Lib\site-packages'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Set-Location $codeRoot
```

啟動前與啟動後都要檢查 8098：

```powershell
netstat -ano | Select-String ':8098\s+.*LISTENING'
```

只能有一個已確認是本輪服務的 listener。若看到舊服務，先由研究者核對其
exact PID 和啟動來源後再停止；不要批次終止 Python 進程。本輪曾發現同一
`127.0.0.1:8098` 同時有前一日殘留服務與本輪服務兩個 LISTENING PID，
瀏覽器因而讀到舊的 data-location 與 raw-frame 設定。必須先切換為單一
current-branch listener，才能把後續 preflight 當作本輪結果。

### A. 已加密儲存

以系統管理員 PowerShell 確認實際資料位置已加密，例如檢查磁碟的
`ProtectionStatus=On` 與完整加密狀態。若無法確認，不要使用
`--confirm-encrypted-storage`。在不影響既有 8080 server 的情況下，可使用 8098：

   ```powershell
   & $python -X utf8 -m scripts.run_general_collection_rehearsal `
     --root $studyRoot `
     --port 8098 `
     --data-location $dataLocation `
     --retention-days 7 `
     --raw-frame-retention-hours 1 `
     --create-invite-pairs 0 `
     --acknowledge-development-only `
     --confirm-encrypted-storage
   ```

   `--create-invite-pairs 0` 是重啟既有 rehearsal 的預設。只有在確認邀請 registry
   尚不存在、且確實要建立第一組 pair 時，才能在那一次啟動改為 `1`。

### B. 研究者本人、未加密、手動保存的 development data

這是明確接受的較低資料治理等級，不是假裝已加密。只允許資料擁有者本人，
不設自動刪除期限，由研究者手動保留或刪除；仍禁止外部受試者、Git/GitHub、
公開 tunnel 與任何 confirmation／正式結果升格。推論用的逐次閱讀影格不另外
落盤；若本人在同意頁另行勾選，可逐篇保存「按下開始」至「完成閱讀」之間的
無音訊 webcam 影片。成功或失敗的校正影像仍須立即清除，中斷校正最長保留一小時。

```powershell
$env:LEXIGAZE_STUDY_MODE = 'rehearsal'
$env:LEXIGAZE_STUDY_REHEARSAL_MODE = '1'
$env:LEXIGAZE_REHEARSAL_ACKNOWLEDGED_DEVELOPMENT_ONLY = '1'
$env:LEXIGAZE_REHEARSAL_INVITES_ONLY = '1'
$env:LEXIGAZE_REQUEST_BODY_LOGGING_DISABLED = '1'
$env:LEXIGAZE_STORAGE_ENCRYPTED = '0'
$env:LEXIGAZE_UNENCRYPTED_SELF_DEVELOPMENT = '1'
$env:LEXIGAZE_DATA_LOCATION = $dataLocation
$env:LEXIGAZE_PUBLIC_BASE_URL = 'http://127.0.0.1:8098'
$env:LEXIGAZE_DATA_RETENTION_DAYS = '0'
$env:LEXIGAZE_DATA_RETENTION_POLICY = 'manual_until_researcher_deletes'
$env:LEXIGAZE_RAW_FRAME_RETENTION_HOURS = '1'
$env:LEXIGAZE_PUBLIC_STUDY_MODE = '1'
& $python -X utf8 -m scripts.run_general_collection_rehearsal `
  --root $studyRoot `
  --port 8098 `
  --data-location $dataLocation `
  --raw-frame-retention-hours 1 `
  --create-invite-pairs 0 `
  --acknowledge-development-only `
  --allow-unencrypted-self-development-data `
  --retain-until-manual-deletion
```

已有 pair 時一律保持 `--create-invite-pairs 0`。只有首次建立時才使用 `1`，
並立即把當次 console 顯示的 Visit 1/Visit 2 plaintext 碼收到不受 Git 追蹤的
安全位置。不要用「再建一組 pair」解決單一 invite 遺失，否則會改變
participant/pair 配對與 schedule cell。

1. Plaintext invite 只會在 console 顯示一次；registry 只保存 hash，事後不能
   還原 Visit 2 碼。不要 commit、貼到 issue 或傳給
   非研究團隊人員。開啟 `http://127.0.0.1:8098/study`，本次只使用 visit 1 code。
   若要保存本人的閱讀影片，必須親自在 consent 頁勾選獨立 optional scope；它預設
   不勾選。瀏覽器只建立 video track，不請求或保存 audio。
2. Visit 2 必須在 visit 1 完成後 18–72 小時，以同一裝置類別與瀏覽器 family
   使用配對 code；server 會拒絕過早、過晚或 visit 1 未完成的 code。
3. 每次完成後執行：

   ```powershell
   & $python -X utf8 -m scripts.audit_general_collection_readiness --root $studyRoot
   & $python -X utf8 -m scripts.export_general_collection_dataset --root $studyRoot
   ```

   Export 是 private pseudonymous bundle，預設排除未完成／已撤回 session，且
   manifest 永遠標示 `formal_promotion_allowed=false`。Raw 影片不會複製進 bundle；
   `reading_video_index.csv` 只保存來源相對路徑、hash、文章／round 與同步資訊。
   不要 commit export 或 `data/` 下的影片。

4. Rehearsal 只回答 completion、missingness、quality、abstention、resume、export
   與 withdrawal 是否正常。影片可供眼動模組開發、動作魯棒性與同步除錯；不得用
   同一人／同一 12 篇文章反覆調參後，再把相同影片回報為模型成效。模型比較至少
   需要另一次未參與調參的 capture session，正式 claim 仍須獨立 confirmation cohort。

### 邀請碼遺失與 Visit 切換

- 若遺失的是尚未使用的單一 invite，先停止 localhost rehearsal，再在上述
  B 模式的同一組 rehearsal/privacy environment 設定下替換它：

  ```powershell
  $registryPath = Join-Path $studyRoot 'data\participant_studies\lexigaze-reader-pilot\rehearsals\collection_invites.json'
  $registry = Get-Content -Raw -Encoding utf8 $registryPath | ConvertFrom-Json
  $registry.invites | Select-Object pair_id, visit_index, used_at_utc, code_rotation_count

  & $python -X utf8 -m scripts.rotate_unused_collection_invite `
    --root $studyRoot `
    --pair-id '<PAIR-ID-FROM-LOCAL-REGISTRY>' `
    --visit-index 2
  ```

  這些欄位只用來在本機確認目標；不要把 registry 或輸出貼到 issue。舊碼會
  立即失效，新碼也只顯示一次。不得旋轉已使用的 invite；也不得為了
  遺失碼而創建另一組 pair。安全收好新碼後，用 `--create-invite-pairs 0`
  重啟服務。
- Visit 2 仍必須在 Visit 1 完成後 18–72 小時開始，不是「拿到新碼就能
  提早開始」。
- 要在同一瀏覽器分頁輸入下一個 invite 時，先保存撤回碼與同意憑證，再按
  「開始另一個邀請 / Visit」。這只會清除該分頁的 session 連結，不會撤回、刪除
  或修改 server 資料；未完成的 Visit 清除後無法從該分頁續接。

### 服務與手動相機門檻

1. 在只有一個 current-branch listener 後，從另一個 PowerShell 的參與者
   公開面確認：

   ```powershell
   Invoke-RestMethod 'http://127.0.0.1:8098/api/gaze/health' | ConvertTo-Json -Compress
   Invoke-RestMethod 'http://127.0.0.1:8098/api/study/protocol' | ConvertTo-Json -Depth 8
   ```

   liveness 必須回傳只含 `{"ok":true}` 的 participant-safe 結果；不需 researcher
   key，也不應暴露 model/dataset 細節。protocol 回傳的 activation、data location、
   retention 與 self-only scope 必須與本次啟動相符。
2. 在 system check 先手動確認私密空間、均勻正面光線與舒適距離，再允許
   localhost 使用相機。預覽中額頭、雙眼、鼻子與下巴都要入鏡，不可有
   強烈背光或遮擋。相機至少 640×480，viewport 至少 1024×700。
3. 儲存 system check 後不要改變視窗大小、顯示縮放、方向、瀏覽器、相機、
   螢幕或裝置位置。若 viewport 已變更，不要硬續；回到 system check 重建凍結狀態。
4. 動作校正必須完成 neutral、left、right、near、far 五個區塊的 13 個目標；
   left/right 各轉頭約 15 度、眼睛仍看目標，near/far 各移動約 15–20 公分。
   不要略過區塊、切換分頁或在進行中搬動裝置。
5. 若顯示「偵測不到完整臉部」，先把臉移回預覽中央、改善光線後才繼續。
   反覆 no-face 會使品質門檻失敗。校正失敗時要閱讀畫面上的 reason code；可修正的
   動作／距離問題可在同頁重來，出現 model binding、影像清除無法驗證或 audit 錯誤時
   必須停止並聯絡研究者。失敗回傳不是測量成果。
6. 校正成功後才進入閱讀前五點驗證，每點三次，只看橘色圓點並保持相機／
   螢幕不動。若 gaze 降級，behavioral reading 可依畫面提示繼續；若本次必須取得
   gaze，則在開始閱讀前停止並聯絡研究者，不要在同一 session 自行反覆調整。

本輪尚未完成新的使用者實體相機 round trip；上述步驟是 capture-ready
preflight 與手動執行門檻，不是新的 measurement result。

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
