# Participant Study Incident Response

## 立即動作

1. 停止新增受試者與邀請碼，關閉 public tunnel；不要刪除可證明 incident 的非敏感時間線。
2. 若有 raw frames，先確認絕對路徑位於本研究的 `data/sessions/<id>`，再用退出流程或 retention cleanup 清除；記錄刪除收據。
3. 旋轉 researcher API key、adaptive signing key、tunnel token 及任何可能外洩的 credential。
4. 檢查 public allowlist、network processor full capture、加密狀態、備份與存取紀錄。
5. 聯絡研究主持人、資料保護／資安窗口及倫理審查單位，依核定時限決定是否通知受試者與主管機關。

## 視為 incident 的事件

- 未同意或已退出者仍有可連結資料。
- 原始影像超過核定 TTL、完整影片被保存，或 request body 被第三方 capture。
- 姓名、email、IP 或永久裝置 ID 進入研究 dataset。
- 受試者能存取管理 API、其他人的模型／資料或答案 key。
- token 可跨 assessment／participant 重放，或結果在未完成 stop rule 前產生。
- 未核准目的、二次使用、外部傳送或 dataset export。

## 紀錄模板

- Incident ID / 發現與發生時間（UTC）
- 發現者與負責人
- 受影響 protocol／consent digest
- 受影響 pseudonymous sessions（不要在一般訊息貼 raw data）
- 資料類型、位置、第三方與最長暴露期間
- containment、刪除、key rotation 與驗證證據
- 通知／不通知決策及其授權者
- 根因、修正、回歸測試與重新收案核准

Codex 或工程測試不能替代法律通報判定。
