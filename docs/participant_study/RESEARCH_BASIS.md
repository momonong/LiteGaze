# Research and Governance Basis

檢索日期：2026-08-07。以下是 protocol 草案的主要權威依據；它們不構成法律意見。

- [臺灣個人資料保護法（個人資料保護委員會籌備處）](https://law.pdpc.gov.tw/LawContent.aspx?id=FL010627)：目的必要性、告知事項，以及查詢／閱覽／更正／停止／刪除等權利。
- [衛生福利部：人體研究倫理政策指引](https://www.mohw.gov.tw/dl-16084-2ccb7514-de3f-4e29-a747-27dce4b0c435.html)：行為與社會研究、清楚可理解的書面同意、隱私、風險、聯絡與二次使用。
- [衛生福利部 2025 年人體研究違規說明](https://www.mohw.gov.tw/cp-16-83089-1.html)：超出倫理審查計畫或未取得知情同意可能涉及人體研究法責任。
- [HHS OHRP：Electronic Informed Consent](https://www.hhs.gov/ohrp/regulations-and-policy/guidance/use-electronic-informed-consent-questions-and-answers/index.html)：電子同意應可理解、可導覽、提供提問機會並記錄同意完整性與日期；理解確認可協助同意品質。
- [HHS OHRP：Withdrawal of Subjects](https://www.hhs.gov/ohrp/regulations-and-policy/guidance/guidance-on-withdrawal-of-subject/index.html)：protocol／consent 應說明退出程序與既有資料如何處理。
- [ngrok Traffic Inspector](https://ngrok.com/docs/obs/traffic-inspection) 與 [Agent CLI `--inspect`](https://ngrok.com/docs/agent/cli)：網路處理者可能保存 request metadata；full capture 可包含 headers／body，因此正式帳號須關閉，且 local HTTP introspection 以 `--inspect=false` 停用。
