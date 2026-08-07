# Data Management Plan

## 資料最小化與分層

| 層級 | 內容 | 位置 | 預設處置 |
|---|---|---|---|
| Consent state | pseudonym、版本／digest、時間、理解確認、事件 | `data/participant_studies/.../session.json` | 依核定天數；退出時改成最小 tombstone |
| Calibration media | raw、crop、normalized face frames | `data/sessions/<linked-id>/` | 成功或失敗立即刪除；中斷最遲 1–24 小時自動刪除 |
| Calibration derivatives | manifest、目標、品質、個人化模型 | pseudonymous session/model | 僅保留核定之必要衍生資料；退出時刪除 |
| Reading frames | 瞬時 JPEG request | RAM / encrypted transit | 不落地、不錄影；處理後即釋放 |
| Assessment | 簽章文章／輪次、正誤、時間、描述性品質 | participant session | 不含答案 key、姓名或完整影像 |
| Network metadata | 時間、來源 IP、目的地、狀態碼等 | 核准之 HTTPS 處理者 | 依服務商必要留存；full-body capture 必須關閉並向受試者揭露 |

## 識別、存取與安全

- 伺服器產生 `P-...` 與 `ST-...`；使用者不能輸入姓名作 ID。
- access token 與 withdrawal code 只保存 SHA-256 digest；明文各顯示一次。
- 正式儲存必須為加密 volume，研究者 key 與 adaptive signing key 不得提交 Git。
- 公開模式採 allowlist，研究者／資料集／訓練／刪除 API 對受試者封鎖。
- 公開 tunnel 使用 `--inspect=false`；另須確認供應商 dashboard 的 full capture 關閉。LexiGaze 正式服務不保留一般 access log。
- 原始影像不進備份；若衍生資料備份，備份亦須加密、同一到期日並可依退出 session 定位刪除。

## 完整性

- 同意與題庫使用版本和 SHA-256 digest 固定。
- 每輪 assignment/result token 綁定 assessment、study session、round、passage、protocol 與 item-bank version。
- 一次性邀請碼、一輪只能計分一次、文章不得重複，重啟後可從伺服器保存的簽章結果繼續。
- 實驗資料、開發資料、participant holdout 與 item/text holdout 的 manifest 在建模前凍結。

## 退出、刪除與保留

退出會刪除 linked gaze dataset、個人化模型與整個 participant session 目錄，再只建立最小 tombstone。raw-frame TTL 在讀取研究狀態時自動執行；任何 TTL 失敗視為停止收案事件。

正式 `LEXIGAZE_DATA_RETENTION_DAYS`、raw TTL、資料位置、備份與不可逆彙整政策必須與倫理核定和同意書完全一致。不得為了「之後可能有用」保留額外資料。

## Dataset 階段的硬性前置條件

本分支不建立可訓練的受試者 dataset export。進入 dataset 階段前，需先完成 schema、資料字典、split manifest、去識別化風險評估、外部效標授權、缺失／退出處理及版本化 export 測試。
