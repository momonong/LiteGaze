# Moderated Dress Rehearsal Runbook v1

狀態：**操作骨架完成；外部 5–8 人目前仍鎖定**。這是 participant-facing、非效果性的兩次 Visit 演練，不是模型、眼動準確度、英語能力或使用者效益實驗。

## 1. 演練目標與禁止解讀

只回答：

- 邀請、同意、理解確認、相機說明與退出是否清楚；
- system check、校正、驗證、閱讀、word review、Visit 2 與 debrief 是否能按標準流程完成；
- 每一步的時間、重試、休息、停止、遺漏與 participant-reported discomfort；
- 研究者是否能在不看答案、不降低品質門檻、不勸說參與者的情況下處理問題。

不得用 5–8 人資料估計效果、挑模型／threshold、宣稱 webcam 準確、比較能力、建立 CEFR／英文／閱讀／認知標籤，或把看過的 participant／passage 再當 confirmation。

## 2. 外部參與者 launch gate

以下任一項為否，決策就是 **NO-GO**；可繼續做不收真人資料的 dry run，但不得邀請朋友：

- [ ] 適用的機構／合格審查者已給 `approved` 或正式 `exempt_determination` 與 reference，且涵蓋兩次 Visit、word review、webcam、重試、退出與 debrief。
- [ ] Canonical protocol 已由 `dry_run_only` 經正式流程更新，網站 consent digest 與最終核定同意文件一致。
- [ ] 研究者、研究聯絡、受試者權益獨立聯絡、補償與中途停止政策已填妥。
- [ ] 外部 participant 版本不含 self-only reading-video scope；不錄 audio、完整校正影片或閱讀影片。
- [ ] 12 篇 rehearsal passages 已完成兩位獨立 reviewer 的自然度、事實、可讀性、敏感內容與 accessibility 審查。
- [ ] `PARTICIPANT_INVITATION_V1.md` 與 `PARTICIPANT_DEBRIEF_V1.md` 已把所有 placeholder 換成核定內容。
- [ ] 目標瀏覽器已完成人工 visual／keyboard QA：文字可讀、焦點順序、錯誤訊息、退出入口、相機預覽與 1024×700 無捲動文章皆實測。
- [ ] 一名未參與核心開發的人完成 moderated dry run 與 withdrawal incident drill。
- [ ] 研究者完成既有 service preflight，確認單一 current-branch listener、正確 digest、invite pair、retention 與 data location。

外部 launch 前，裸跑下列命令必須 exit `0`；不得用較寬鬆的 `--target materials`
取代：

```powershell
python -X utf8 -m scripts.audit_dress_rehearsal_readiness
```

核定資料必須以
`docs/participant_study/evidence/dress_rehearsal_material_approval_v1.json`
綁定 exact investigator email、participant-rights contact、ethics reference、runtime
consent digest、rehearsal protocol SHA-256，以及 invitation、debrief、final consent
與 compensation policy 的 path／SHA-256。三份 participant-facing material 都必須包含同一份
核定聯絡、reference 與 compensation wording。缺值、任意非 placeholder 替換、runtime
不一致或檔案被改動都維持 **NO-GO**；不得為了讓 audit 通過而自行填值。

Manifest schema（值只能來自核定文件與實際 runtime）：

```json
{
  "schema_version": 1,
  "status": "approved_for_external_rehearsal",
  "canonical_protocol_id": "<exact>",
  "canonical_protocol_version": "<exact>",
  "rehearsal_protocol_sha256": "6f6264a58e820e47c414f1e86fd499dccb4930a587258cabc694ba80e7c610bd",
  "runtime_consent_digest_sha256": "<64 lowercase hex>",
  "approved_bindings": {
    "investigator_email": "<exact approved value>",
    "participant_rights_contact": "<exact approved value>",
    "ethics_reference": "<exact approved value>",
    "compensation_policy": {
      "artifact_path": "docs/participant_study/COMPENSATION_POLICY_FINAL_V1.md",
      "artifact_sha256": "<64 lowercase hex>",
      "participant_facing_text_zh": "<exact approved wording>"
    }
  },
  "materials": {
    "invitation": {
      "path": "docs/participant_study/PARTICIPANT_INVITATION_V1.md",
      "sha256": "<64 lowercase hex>"
    },
    "debrief": {
      "path": "docs/participant_study/PARTICIPANT_DEBRIEF_V1.md",
      "sha256": "<64 lowercase hex>"
    },
    "consent": {
      "path": "docs/participant_study/CONSENT_FINAL_V1.md",
      "sha256": "<64 lowercase hex>"
    }
  }
}
```

## 3. 已知介面限制與標準處置

這些不是可隱藏的細節；moderator 必須在開始前說明並記錄：

| 限制 | 標準處置 |
|---|---|
| Collection page 沒有 mid-passage pause／withdraw 按鈕 | 休息安排在未按「開始閱讀」的安全點。若閱讀中要求停止，立即停止互動／離開頁面，不在同一 segment 自行重做；記為 `mid_passage_stop`，再依核定政策決定只停止或撤回。 |
| 8 分鐘上限沒有 participant-facing 自動倒數停止 | Moderator 使用獨立計時器，不看 participant 畫面、文章或答案；接近 8 分鐘時提醒可以停止。不可讓 participant 因完成壓力無限延長。 |
| Visit 完成頁不顯示完整 Visit 2 時窗與補償細節 | 完成後使用本 runbook 與核定 debrief 口頭／書面說明；不可只說「下次再來」。 |
| 相機 stream 可能在 passage 間仍保持啟用 | 每個休息點先說明相機狀態。若 participant 不想讓相機保持開啟，離開 collection page 讓 track 停止；恢復前重新確認權限與舒適度。 |
| 中斷 reading segment 的自助 resume 不等同有效 retry | 不要求 participant 自行 reload 或回上一頁。研究者記錄狀態與原因；只有 server state 與 protocol 明確允許時才恢復。 |

## 4. 角色與盲化

- **Moderator：**讀標準 script、處理安全與狀態，不看 participant 的 word-review 選項或理解自評。
- **Observer／note-taker：**只記流程事件與 participant 主動回報；不記姓名、聯絡方式、逐字答案或可識別的自由文字。
- **Participant：**可在提供 `prefer_not_to_say` 的欄位拒答，也可休息、停止或撤回；若某必要步驟不願繼續，正確處置是停止，不是強迫完成或私下跳過。Participant 不負責替研究者除錯或達成 gaze 品質。

若只有一名研究者，畫面應保持 participant control；研究者坐在看不到答案的位置，只在 participant 主動求助時介入。

## 5. 每次 Visit 前 10 分鐘

1. 完成既有 preflight，不在 participant 到場後才重啟或換 branch。
2. 核對本次 invite、visit index、18–72 小時規則、同裝置類別／瀏覽器 family 與核定時段。
3. 開啟 participant 頁但不預先勾選 consent、optional scope 或背景選項。
4. 準備紙本／獨立畫面的聯絡與退出說明；不要把 invite、撤回碼或 participant ID 記入觀察筆記。
5. 確認安靜、私密、無旁人入鏡，並讓 participant 自己調整椅子、螢幕與相機。
6. 說明 safe break points：同意前、system check 前、校正後、閱讀前驗證後、文章 2 與 4 後、結束驗證前，以及 participant 隨時提出時。

## 6. 開場與相機舒適度 script

Moderator 逐字讀：

> 今天我們測的是流程，不是你的英文、閱讀、注意力或認知能力。你不需要替系統取得好的眼動品質。你可以拒絕開相機，屆時流程會停止；你也可以要求休息、停止或撤回，不必說明原因。若相機、姿勢、光線或長時間看螢幕讓你不舒服，請立即告訴我；我們會先停止，不會要求你忍耐或為了資料品質重做。

在 browser permission 前詢問：

> 畫面會顯示相機預覽，方便你確認取景。你現在是否願意繼續到相機檢查？

只有明確願意才繼續。沉默、猶豫、拒絕權限或移開鏡頭都不可解讀為同意。

## 7. 標準流程與操作 gate

### A. Consent 與邀請

- Participant 自己閱讀、勾選與回答理解確認；錯誤只提示回到資訊，不記能力。
- Moderator 可解釋研究程序與權利，不可提示 comprehension 正確選項。
- 必須下載／保存同意收據與撤回碼，再進入 system check。
- 若 consent draft、網站內容、邀請或口頭說明在時間、資料、影片、Visit 數、補償或退出政策不一致，停止 session 並記錄 material mismatch。

### B. System check

- 先確認私密空間、均勻光線、舒適距離，再由 participant 按「檢查相機」。
- 相機 permission 被拒：尊重決定，不重複說服；詢問要只停止，或依核定政策撤回已建立的 session。
- Resolution／viewport／network 失敗：允許一次標準修正與重新檢查；不降低 640×480、1024×700 或其他門檻。
- Participant 表示相機預覽不舒服：立即停止 camera track／離開 collection page；不以「只剩一下」要求繼續。

### C. Calibration

- 先示範 neutral、left、right、near、far 的動作，但不替 participant 移動身體。
- 每個動作以舒適為先；疼痛、暈、眼睛乾澀、姿勢困難或希望停止時立即中止。
- 可修正的 no-face／光線／距離問題最多做一次標準 retry；第二次仍失敗就停止該 Visit，不另建 invite、session 或私下放寬品質 gate。
- Model binding、purge、audit 或非預期錯誤一律停止並依 incident 流程處理。

### D. Practice 與 start validation

- Practice 只確認三個標籤是否理解，不教「應該」選哪個。
- 標準說明：`不需要回顧`＝剛才閱讀時不需要再看；`不確定`＝無法確定；`需要回顧`＝剛才閱讀時希望再確認。沒有正確答案。
- 五點 validation 前再次確認願意繼續；失敗若由系統明確降級為 behavioral-only，可按畫面繼續，不將品質歸因 participant。

### E. 六篇 reading 與 word review

- Break 安排在文章開始前；建議在文章 2、4 後主動提供至少 2 分鐘休息。
- 按「開始閱讀」後不 coaching、不要求固定眼睛、不看答案。
- 20 秒是按鈕解鎖門檻，不代表 participant 必須快速讀完；接近 8 分鐘則依已知限制標準處置。
- 中途被打斷要如實勾選；中途停止不可自行重跑成看似完整的 segment。
- Word review 全部由 participant 作答；若詢問定義，只重複上方固定說明。

### F. End validation、完成與 Visit 2

- End validation 前提供休息與再次同意繼續的機會。
- 品質摘要只描述 session 的 sensor／流程，不說 participant「表現不好」。
- Visit 1：以完成事件時間計算 18–72 小時；確認願意再安排 Visit 2。拒絕 Visit 2 不需原因，補償依核定政策。
- Visit 2：使用同裝置類別與瀏覽器 family；先詢問 Visit 1 是否造成相機或姿勢不適，再完整走同一 consent／權利邊界。
- Invite 遺失只依既有 rotation runbook 處理；不可新建 pair 來規避配對或時窗。

## 8. 停止、只停止與撤回

Participant 說「不要了」、「先停」、「不舒服」、拒絕下一步或以其他方式表示不願繼續時，立刻停止。使用中性問題區分：

> 你希望只停止今天的流程，還是也希望使用撤回碼刪除目前仍可定位的資料？兩種選擇都可以；你不需要說明原因。

- **只停止：**保留到哪一步、是否可恢復、補償如何處理，完全依核定 protocol；不得由研究者臨場決定。
- **撤回：**協助 participant 使用 study session ID 與撤回碼；保存 participant 可持有的刪除 receipt，不把明文碼抄入研究筆記。
- **無法刪除／purge：**停止新增參與者並啟動 incident response。

## 9. 非識別 observation sheet

每個 Visit 只記下列欄位，不記姓名、聯絡方式、答案或逐字引言：

- rehearsal slot：`DR-01`…`DR-08`
- visit index、開始／結束時間、完成階段
- 各步耗時 band：`<5 / 5–10 / 11–20 / >20 min`
- consent clarification count
- system-check retry count 與 allowlisted reason
- calibration retry count 與 allowlisted reason
- break count、break stage、participant-requested 或 moderator-offered
- camera／posture discomfort：`none_reported / adjustment_requested / break_requested / stop_requested / prefer_not_to_say`
- mid-passage stop／reload／resume／network interruption
- word-review label clarification requested：yes/no（不記回答）
- withdrawal comprehension confirmed：yes/no；withdrawal drill executed：yes/no
- Visit 2 scheduled／declined／outside window（不在此表記日期或聯絡資訊）
- protocol deviation／incident ID（只引用獨立 incident log）

## 10. 每位後與 5–8 人段落門檻

每位後：

1. 確認狀態為 completed、stopped-with-documented-policy 或 withdrawn。
2. 確認 calibration image purge、事件完整、沒有未標記的 mid-passage restart。
3. 交付 debrief 與聯絡／撤回資訊；Visit 1 才安排 Visit 2。
4. 在看任何 word-review 或 gaze outcome 前，只依 observation sheet 修正文案／流程。

完成 5–8 人後只能進入下一段落，若同時滿足：

- 100% participant 能用自己的話說明自願、相機、兩次 Visit 與撤回；
- 沒有人在回報 discomfort 後被要求繼續；
- 所有 stop／withdrawal request 都在同一步被執行並有 receipt／事件；
- retry、break、reload 與 interruption 都能由 allowlisted 狀態描述；
- 沒有 consent／invite／debrief material mismatch；
- 研究者沒有看答案、手改 gate、建立替代 invite 或把 rehearsal 當效果資料。

任何安全、同意、刪除或未揭露資料事件都停止下一位；一般 usability 問題可修正文案後重新做新的 development rehearsal，但舊 participant 不得重跑成「通過」樣本。
