# Participant Study Readiness Checklist

## A. 可在現在完成

- [x] 新分支隔離受試者準備工作。
- [x] 版本化 protocol、consent digest、理解確認與成人限定。
- [x] server-generated pseudonym；直接識別欄位被忽略。
- [x] dry run 不啟動相機、不建立模型、不儲存研究資料。
- [x] participant-only calibration 與 assessment page。
- [x] calibration failure／training failure／purge failure 全部刪除暫存 dataset。
- [x] 中斷校準依 TTL 自動清理並回到可重試狀態。
- [x] 正式個人化與 participant inference 強制 CPU。
- [x] 題目指派與結果 token 綁定 session／round／版本；一次只能計分一次。
- [x] 公開 allowlist 封鎖管理、資料集、報告與訓練 API。
- [x] v1 不錄完整影片；閱讀 frame 不落地。
- [x] 退出刪除 linked dataset、model 與整個 session payload，只留 tombstone。
- [x] 離線單元／整合測試與 machine-readable readiness audit。
- [x] protocol、consent、DMP、runbook、incident 與決策紀錄草案。

## B. 目前刻意保持未完成；完成前不可找朋友

- [ ] 機構或合格審查者的 approved／exempt determination 與 reference。
- [ ] 研究主持人姓名、email 與受試者權益獨立聯絡。
- [ ] 最終研究目的、樣本數、補償與退出後已彙整資料政策。
- [ ] 外部效標選定、授權、施測方式與預註冊。
- [ ] participant 與 item/text holdout manifest 在看資料前凍結。
- [ ] 加密資料位置、角色、retention、備份與刪除驗證。
- [ ] HTTPS/network processor 與其 metadata 留存核准；full-body capture 關閉證據。
- [ ] 正式 researcher key、≥32 字元 signing key 與 secret rotation 流程。
- [ ] canonical protocol 經核定後改成 `approved_for_pilot` 並重新產生 consent digest。
- [ ] `--target pilot` audit exit code 0。
- [ ] 未參與開發者完成一次 moderated dress rehearsal 與 incident drill。

若 B 還有任一項未勾選，正確狀態是「dry-run engineering ready、real pilot locked」，不是「差不多可以收案」。
