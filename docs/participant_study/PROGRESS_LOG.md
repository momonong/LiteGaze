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

## 2026-08-08 — independent capture-plan readiness

在未查看新 participant outcome、未下載外部資料且 GPU 使用量為零的情況下，
新增 strict capture-plan contract。它把 participant、repeated session、physical
capture run、article family、device 與 sensor source 在 development、validation、
confirmation 間的關係變成 machine-checkable，而不是只靠文件約定。

Optional laptop-plus-phone plan 必須提供 paired source、timestamp tolerance、
clock strategy、relative-camera calibration slot／artifact hash 與 missing-view
policy；同一次實體 capture 的兩個來源不能跨 split 或被算成獨立 evidence。

工具只接受 numeric slots 與 isolated binding digests，withdrawal 會清除 digest，
audit 輸出不回傳 slot 或 binding。Template、pre-collection freeze 與 post-binding
evidence 分成三個不同 target，避免把 schema-valid 誤寫成可招募或有效模型。

這項里程碑沒有解除 `dry_run_only`。倫理／豁免判定、正式聯絡、樣本數、外部
效標與文章授權、storage、HTTPS、retention、正式 frozen manifest 及 dress
rehearsal 仍是 pilot blockers。

## 2026-08-08 — generalizable fixed-form collection rehearsal v1

### 目標與決策

把研究團隊本人的資料收集流程做成未來每位使用者都能照同一規格執行的版本，
並明確切開 development rehearsal 與正式 confirmation。舊 adaptive QA 不再作為
這條 track 的 outcome；primary label 改為每篇閱讀後、模型輸出隱藏時取得的
8 個 `no_review / unsure / review_needed` 單字回顧標籤。

每位 participant 兩次 visit，間隔 18–72 小時，A/B alternate forms 與六種
Williams-style passage order 在看到 outcome 前由一次性 invite pair 指定。同一
participant、capture session、device、passage family 與 probe 都保留 group ID，
後續實驗必須做 participant/text/probe/capture/device holdout，不能 random-row split。

### 完成的工程

- 凍結 general collection protocol 與 12-family rehearsal bank；A/B 各 6 篇、
  每篇 8 probes，另有不寫入 observation 的 practice passage。
- 新增 strict categorical profile、coarse device schema、5-point start/end validation、
  fixed 16 px / 650 px / 1.7 layout、20 秒最短閱讀、no-scroll/no-zoom gate。
- 閱讀期間只傳 transient frame 做同源推論；server 僅接受 derived gaze、head pose、
  normalized face geometry、word index、failure code 與 monotonic time。Raw media field
  會被 schema 拒絕，batch ID 重送只能是 byte-equivalent payload。
- 新增 invite-pair、visit interval、immutable round order、48 labels/visit、resume、
  withdrawal、private pseudonymous export 與 manifest hash。低品質 gaze 只降級成
  `passage_level_only` 或 `behavioral_only`，不刪 behavioral labels。
- 新增 localhost-only runner；必須由操作人明確確認 encrypted storage、development
  only、invite-only、retention 與 raw-frame TTL。Gaze routes 現在一致遵守 configured
  storage root，避免 study metadata 與 calibration artifacts 分散到不同位置。
- 同意內容升版，明列分類式語言／閱讀／視力／教育背景，不收 direct identifiers、
  exact age、user-agent fingerprint 或自由文字。

### 文章與樣本規劃結果

- Automated bank screen：12 unique families、96 unique-in-passage probes；foundation /
  standard / advanced heuristic grades 8.74 / 13.23 / 14.75；A/B mean grade 12.21 /
  12.26；9 domains、5 genres、maximum five-word overlap 0.0。這只通過 automated
  screen，兩位 independent human reviews 仍是 0/2。
- 50,000-iteration CPU sensitivity simulation：在 joint targets（52 paired behavioral、
  30 word-gaze、25% subgroup 至少 20 人、paired d=0.40 power）下，conditional minimum
  enrollment 是 optimistic 128、base 144、pessimistic 208。這不是招募授權；正式
  design 必須用 blinded rehearsal rates 更新，任何 development tuning 之後需另收
  frozen confirmation cohort。

### 驗證證據

- Focused Python/JS validation：27 tests passed；三個 browser scripts syntax pass；
  六個 collection CLI 的 direct `--help` entry point 均可啟動。
- Full offline quality gate：171 tests passed；0 failure、0 error、0 skip。
- Safeguards：network/subprocess probes blocked、provider credentials cleared、artifact
  changes `[]`、Torch not imported、`CUDA_VISIBLE_DEVICES=-1`。
- GPU before/after：RTX 5090 utilization 均為 `0%`，memory 均為
  `76/24463 MiB`；worker 明確設為 CPU-only。
- In-app browser isolated QA：正確顯示 rehearsal boundary、versioned consent、one-time
  invite、server pseudonym 與 Visit 1 / Form A；1024×700 無 horizontal overflow；
  console 0 warnings/errors。沒有代替使用者接受 camera permission。
- Browser QA 使用的 synthetic session、plaintext QA invites、logs 與 localhost:8097
  process 已全部刪除／停止；沒有混入 project data。既有 localhost:8080 process
  未被停止或修改。

### 仍然 fail closed

正式朋友／外部受試者收案仍缺 revised ethics/exempt determination、正式聯絡與
rights contact、authorized external anchor、48-family bank + 2 independent reviews、
frozen dev/validation/confirmation manifest、rehearsal-rate sample-size update、
practical-utility threshold 與 moderated camera dress rehearsal。本里程碑沒有解除
正式 `pilot_ready=false`。

## 2026-08-09 — self-only reading-video development capture

### 使用者決策與資料角色

研究者本人明確選擇把資料保存在 repo 的 `data/`（實體位於未加密 D 槽），不設
自動刪除期限，並希望保留閱讀影片來改善系統。這項同意只涵蓋研究者自己的
development data；不是正式受試者收案，也不能升格為 validation／confirmation
結果。未來要比較模型時，同一批影片不能同時參與調參與成效回報，至少要使用另一次
未參與調參的 capture session；正式 claim 仍需獨立 frozen confirmation cohort。

### 工程決策與完成項目

- 新增明確的 `unencrypted_self_development` 啟動模式，只允許一組 Visit 1/2 invite
  pair、localhost、手動保留，並固定 `formal_promotion_allowed=false`；原本的正式
  pilot 仍強制加密，沒有被此例外解鎖。
- Consent 升至 `2026-08-09.v5`。閱讀影片是預設未勾選的獨立 scope，文字明列未
  加密 D 槽、無自動期限、self-development-only 與不得當 confirmation 證據。
- 瀏覽器使用 video-only `MediaStream` 與 `MediaRecorder`：640×480、目標 15 fps、
  750 kbps，只在每篇文章按下開始至完成閱讀期間錄製；不錄同意、背景、校正、
  validation、practice、單字回顧或音訊。
- 每篇影片以不可變 `Rxx.webm`／`Rxx.mp4` 保存，綁定 participant、session、visit、
  passage、round、duration、MIME、bytes 與 SHA-256；限制 64 MiB，MIME 不一致、
  重複 round 或同 ID 不同內容均 fail closed。若 media/metadata 已完成但 session
  summary 寫入中斷，開啟 word review 前會驗證 hash 並恢復索引。
- 分析 export 不複製 raw video，只輸出 `reading_video_index.csv` 與 manifest 的來源
  計數／development role。原始影片留在 ignored `data/`，不得進 Git/GitHub。
- 一般加密 rehearsal、正式 pilot 與未另行勾選者看不到或不能使用 self-only video
  scope；逐次 gaze inference snapshot 仍不寫入磁碟。

### 驗證與 GPU 紀錄

- Focused participant/general/app integration：36 tests passed；包含 multipart route、
  MIME mismatch、immutable round、opt-in boundary、crash recovery 與 index-only export。
- Full offline quality gate 重跑皆為 180 tests passed，0 failure、0 error、0 skip；
  network/subprocess blocked、credentials cleared、artifact changes `[]`、Torch not
  imported、worker `CUDA_VISIBLE_DEVICES=-1`。
- GPU supervisor snapshot 受桌面同時活動污染，故不宣稱整機 GPU 全程 0%。第一次為
  before `0%, 531 MiB`、after `25%, 541 MiB`；中間一次 console run 為 before
  `63%, 541 MiB`、after `0%, 166 MiB`。確認整機回到 idle 後的 final rerun 為
  before/after 均 `0%, 166 MiB`。所有 worker 均沒有 Torch/CUDA 路徑；前兩次整機
  瞬時值不能歸因於測試 worker。第一次與 final rerun 的 machine-readable 結果保存在
  `2026-08-09-unencrypted-self-development-offline-gate.json` 與
  `2026-08-09-unencrypted-self-development-offline-gate-rerun.json`。
- 本里程碑完成時尚未建立真人 session 或影片；下一步是由研究者本人在瀏覽器親自
  輸入一次性 Visit 1 code、閱讀同意內容、勾選 optional video scope 並授權相機。
