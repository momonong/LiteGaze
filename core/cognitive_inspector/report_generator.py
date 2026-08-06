"""Markdown reporting for the evidence-bounded reader assessment."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _display(value: Any, suffix: str = "") -> str:
    if value is None:
        return "資料不足"
    return f"{value}{suffix}"


def _claim_label(status: str | None) -> str:
    labels = {
        "session_observation": "本次觀測",
        "provisional_session_estimate": "暫定的單次表現估計",
        "insufficient_data": "資料不足",
        "insufficient_variation": "刺激變異不足",
        "not_collected": "未收集",
        "not_estimated": "不估計",
    }
    return labels.get(status or "", status or "未知")


def generate_markdown_report(result: dict[str, Any], participant_id: str) -> str:
    """Generate a transparent report without unsupported ability claims."""

    summary = result.get("summary", {})
    quality = result.get("data_quality", {})
    claims = result.get("claims", {})
    fluency = claims.get("reading_fluency", {})
    lexical = claims.get("lexical_processing_signal", {})
    change = claims.get("within_session_change", {})
    comprehension = fluency.get("comprehension", {})
    reasons = quality.get("reasons", [])
    reason_text = "、".join(reasons) if reasons else "無主要警示"

    claim_rows = []
    claim_names = {
        "reading_fluency": "閱讀流暢表現",
        "english_proficiency": "英文能力／CEFR",
        "cognitive_ability": "一般認知能力",
        "attention": "注意力",
        "fatigue": "疲勞",
        "cognitive_load": "認知負荷",
    }
    for key, label in claim_names.items():
        claim = claims.get(key, {})
        reason = claim.get("reason") or claim.get("scope") or "—"
        claim_rows.append(
            f"| {label} | {_claim_label(claim.get('status'))} | {reason} |"
        )

    return f"""# LexiGaze 閱讀測量證據報告 v{result.get("assessment_version", "2")}

- 參與者：`{participant_id}`
- 產生時間：`{datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")}`
- 測量範圍：單一文章、單次 session 的行為觀測

> [!IMPORTANT]
> 本報告不是智力、注意力、疲勞、臨床狀態或 CEFR 英文能力診斷。當證據不足時，系統會明確回覆「不估計」，而不是用任意公式產生 0–100 分。

## 資料品質

| 欄位 | 結果 |
| :--- | :--- |
| 品質狀態 | `{quality.get("status", "unknown")}` |
| 測量信心 | `{quality.get("confidence", "unknown")}` |
| 品質分數 | `{quality.get("score", 0)}`（只代表資料品質，不代表能力） |
| 主要限制 | {reason_text} |

## 本次可觀測閱讀行為

| 指標 | 結果 | 正確解讀 |
| :--- | :--- | :--- |
| 有效注視數 | {_display(summary.get("total_fixations"))} | 本次被分組出的字詞注視事件。 |
| 注視中位數 | {_display(summary.get("median_fixation_duration_ms"), " ms")} | 對追蹤取樣率敏感，不等於認知能力。 |
| 注視 IQR | `{summary.get("fixation_duration_iqr_ms")}` | 顯示分布，避免只看平均數。 |
| 回視率 | {_display(round((summary.get("regression_rate") or 0) * 100, 1), "%")} | 回視可能來自理解、閱讀策略或眼動控制誤差。 |
| 回視率 95% 區間 | `{summary.get("regression_rate_ci95")}` | 反映有限事件數造成的不確定性。 |
| 完整文本閱讀速率 | {_display(summary.get("words_per_minute"), " WPM")} | 只有提供完整字數、明確開始／結束時間且確認讀完時才計算。 |
| 僅觀測字詞速率 | {_display(summary.get("observed_word_rate_wpm"), " observed words/min")} | 不能當成 WPM，因為正常閱讀會跳字。 |
| 理解題證據 | `{comprehension.get("correct")}/{comprehension.get("total")}`；95% 區間 `{comprehension.get("proportion_correct_ci95")}` | 題數少時區間會很寬。 |

## 宣稱狀態

| 構念 | 狀態 | 原因／範圍 |
| :--- | :--- | :--- |
{chr(10).join(claim_rows)}

## 可探索但不可當成能力的訊號

- 詞頻—停留關聯：`{lexical.get("rarity_dwell_association")}`；狀態 `{_claim_label(lexical.get("status"))}`。
  {lexical.get("interpretation") or lexical.get("reason") or ""}
- 後段／前段注視中位數比：`{change.get("late_to_early_median_duration_ratio")}`；狀態 `{_claim_label(change.get("status"))}`。
  {change.get("interpretation") or change.get("reason") or ""}

## 要提升到能力測量仍需要的證據

1. 多篇彼此獨立、構念覆蓋明確的閱讀材料與理解題。
2. 題目難度、鑑別度、猜測率與跨群體公平性的實際校準資料。
3. 受試者與文章雙重 holdout 的外部驗證，不能用同一批固定問答調參後再宣稱成效。
4. 重測信度、測量標準誤與信賴／可信區間。
5. 能力測量與排版實驗分開；排版需用難度匹配文章做隨機化、交叉或反平衡比較。

---

此版本採用 `abstain_without_validity_evidence` 政策：沒有足夠效度證據，就不輸出能力標籤。
"""
