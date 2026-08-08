# Independent Capture Plan Contract

狀態：工程工具可用；`independent_capture_plan.example.json` 只是 synthetic
template，不是正式研究計畫、倫理核准、招募授權或樣本數決策。

## 為什麼需要另一層 manifest

目前 LexiGaze 已能把同一次瀏覽器錄製產生的 direct frames 與
video-derived frames 綁回同一個 `capture_run_id`，也能檢查 motion block
coverage。不過，這仍不足以證明模型能泛化到新的受試者、日期、文章或
裝置。

Independent Capture Plan 在第一次收集或查看 outcome 前，先固定五種單位：

| 單位 | 作用 | 不可跨越的邊界 |
| --- | --- | --- |
| Participant slot | 綁定一次性 pseudonymous participant | analysis role |
| Session slot | 同一人不同日期／時段的重複量測 | participant 與 role |
| Device slot | 研究者指定的裝置實例，不使用 browser fingerprint | role；僅在不宣稱 device generalization 時可標為 shared |
| Article family slot | 同篇、平行版或近重複文章的共同 family | analysis role |
| Capture-run slot | 一次實體收集及其一或多個 sensor source | participant、session、article 與 role |

三個 analysis roles 固定為：

- `development`：允許研究者查看並用來修正流程；
- `validation`：只用於預先指定的選擇或停止規則；
- `confirmation`：在候選、門檻及分析程式凍結後才可開封。

任何看過並據以修改系統的 slot 都不能再被稱為 confirmation。

## 三種 audit target

```powershell
# 只檢查 schema 與 leakage contract；example 應通過
.\.venv\Scripts\python.exe -X utf8 -m scripts.audit_independent_capture_plan `
  docs\participant_study\independent_capture_plan.example.json `
  --target template

# 正式收集前；必須已改成 frozen_before_collection 並有 UTC freeze time
.\.venv\Scripts\python.exe -X utf8 -m scripts.audit_independent_capture_plan `
  path\to\reviewed-plan.json --target collection

# 收集後、建立分析資料前；只計算 bound 且未 withdrawn 的單位
.\.venv\Scripts\python.exe -X utf8 -m scripts.audit_independent_capture_plan `
  path\to\bound-plan.json --target evidence
```

- `template_valid` 只代表 JSON 結構與 isolation 規則一致。
- `collection_ready` 代表計畫已在收集前凍結，不代表可以招募；既有 ethics、
  consent、contact、storage、HTTPS、retention 與 rehearsal gates 仍必須通過。
- `evidence_ready` 代表 binding manifest 足以建立獨立的分析單位，不代表模型
  有效或資料品質合格；影格品質、校準誤差與 primary outcome 仍須由另一份
  預註冊分析處理。

Exit code 只有在所選 target ready 時才是 `0`。Audit 輸出不包含 slot ID、
participant/session binding 或文章 ID，只回傳 aggregate counts、plan SHA-256
與錯誤代碼。

## 從 template 到正式 frozen plan

1. 先完成倫理／豁免判定、研究目的、樣本數、補償、外部效標、文章授權與
   storage 決策；程式不能代替這些判定。
2. 複製 example 到不含真實資料的版本化位置。
3. 依核准樣本數建立不帶姓名的 numeric slots。Slot ID 只能使用
   `PSLOT-...`、`SSLOT-...`、`DSLOT-...` 等數字格式。
4. 在不知道 participant outcome 的情況下，分配 development、validation
   與 confirmation roles。
5. 每位 participant 至少安排兩個 session，並凍結最小時間間隔。
6. 每篇文章保存內容 SHA-256、授權 reference，以及近重複 family；同一
   family 不得跨 role。
7. 明確決定是否宣稱 device generalization 或 multi-view incremental value。
8. 執行 `--target template`，review diff，然後才把 status 改為
   `frozen_before_collection`、填入 `frozen_at_utc` 並提交 Git。
9. 執行 `--target collection`。通過後仍需既有 pilot audit exit code 0 才能
   建立正式 invite。

`declared_target_participants` 與每個 role 的 minimum 不能沿用 example 的
三人示範值。正式數值必須來自研究問題、precision/power reasoning、可行性與
核准文件，而不是目前模型表現。

## Binding 與 withdrawal

Plan 不保存 `P-...`、`ST-...`、姓名、email、永久 browser/device ID 或原始
文章內容。正式 binding 應使用專屬於該 plan 的 HMAC-SHA-256 或具等價隔離的
digest；key 留在核准的 secret store，不寫入 plan 或 Git。

- `unbound`：digest 必須是 `null`，不算 collected evidence。
- `bound`：必須有 64 位小寫 hex digest。
- `withdrawn`：digest 必須清成 `null`，slot 保留作缺失數量，但不得計入
  evidence。

同一 binding digest 出現在兩個 slot 會直接失敗。Withdrawn participant 的
session bindings 也必須全部變成 `withdrawn` 並移除 digest。

## Laptop 加手機的 multi-view 規則

加入第二個來源時，兩個影格仍屬同一個 physical `capture_run_slot`，不能被當成
兩筆獨立樣本。每個 multi-view run 必須先聲明：

- distinct `laptop-primary`／`phone-secondary` source roles；
- timestamp clock strategy 與最大容許 offset；
- `XCAL-...` relative-camera calibration slot；
- evidence 階段可驗證的 calibration artifact SHA-256；
- `primary_only_fallback` 或 `abstain` missing-view policy。

若宣稱 `multiview_incremental_value=true`，每個 planned capture run 都必須有
至少兩個 source。若宣稱 `device_generalization=true`，同一實體裝置 binding
不得跨 role，也不能使用 `shared` device slot。

多視角研究顯示 paired images 與相機相對旋轉是有效 fusion 的必要結構；因此
「把手機放在旁邊」本身不是可測試的方法。參考：
[Rotation-Constrained Cross-View Feature Fusion](https://openaccess.thecvf.com/content/WACV2024/html/Hisadome_Rotation-Constrained_Cross-View_Feature_Fusion_for_Multi-View_Appearance-Based_Gaze_Estimation_WACV_2024_paper.html)。

## Strict schema 與 outcome isolation

Schema 採 exact allow-list；任何多出的 key 都回報
`UNKNOWN_OR_OUTCOME_FIELD`。因此 answer correctness、theta、reading time、gaze
error、model prediction、difficulty score 或人工挑選結果不能混進 planning
artifact。觀察結果應在 plan freeze commit 之後，由另一個 integrity-bound
result artifact 保存。

## 公開資料的下一步

- [Columbia Gaze](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/)
  可直接取得約 2.38 GB 的非商業研究資料，包含 56 人、5,880 張影像、五種
  head poses 與 21 種 gaze directions。它適合下一個獨立 stress test，但必須
  先凍結 eye crop、label convention 與 coordinate mapping。
- [ETH-XGaze](https://ait.ethz.ch/xgaze) 最小 face-patch package 約 130 GB，
  且須以機構身分接受額外條款，不會自動申請或下載。
- [GazeCapture](https://gazecapture.csail.mit.edu/) 約 250 萬 mobile frames，
  需註冊且需要獨立 storage/preprocessing review。
- [EYEDIAP](https://www.idiap.ch/en/scientific-research/data/eyediap) 具有跨日期、
  illumination、distance 與 mobile-head sessions，但也需依 provider 流程取得。

下一個公開資料實驗必須使用新分支與新 protocol，不能以本工具的
`collection_ready` 代替資料授權、label audit 或模型 evaluation protocol。
