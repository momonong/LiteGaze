"""
scripts/fusion/orchestrator.py
══════════════════════════════════════════════════════════════════════════════
LexiGaze 數據融合協調器 (Fusion Orchestrator)

將「感知端」（眼動軌跡 gaze_log.jsonl）與「認知端」（CognitiveLoadPipeline
輸出的 word_analysis[]）進行後端離線融合，計算每個單字的
Reading Difficulty Score（RDS），並輸出完整的 JSON 報告。

RDS(w) = 0.35 × dwell_norm(w) + 0.25 × fix_norm(w) + 0.40 × load_score(w)

使用方式：
    python scripts/fusion/orchestrator.py \\
        --gaze-log    data/<session_id>/gaze_log.jsonl \\
        --cognitive   docs/fusion_reports/cognitive_<session_id>.json \\
        --session-id  <session_id>

    # 或讓 orchestrator 自行呼叫 weichi pipeline 分析一段文字：
    python scripts/fusion/orchestrator.py \\
        --gaze-log    data/<session_id>/gaze_log.jsonl \\
        --text        "The quick brown fox..." \\
        --lang        en \\
        --session-id  <session_id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]   # lexigaze/
REPORTS_DIR = ROOT / "docs" / "fusion_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fusion_module import LexiGazeFusion

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 眼動感知端 — 讀取 gaze_log.jsonl
# ═══════════════════════════════════════════════════════════════════════════════

def load_gaze_log(path: Path) -> list[dict]:
    """
    讀取眼動事件日誌 (gaze_log.jsonl)。

    每一行預期欄位（由前端寫入）：
        {
          "word":           str,    // 已對齊的單字（toLowerCase 後由 mapping.js 填入）
          "confidence":     str,    // "high" | "medium" | "low"
          "dwell_count":    int,    // 在此單字上的 8 Hz gaze 命中次數
          "fixation_count": int,    // 獨立眼跳次數（前端從 is_new_fixation 累計）
          "timestamp_ms":   int,    // Date.now() 採集時間
          "gaze_x":         float,  // 螢幕像素 X（選配）
          "gaze_y":         float   // 螢幕像素 Y（選配）
        }

    也相容舊格式：直接包含 timestamp / gaze_x / gaze_y 但無 word 的原始推論
    log（此時 word 欄位會是 None，這些行會被過濾掉）。
    """
    events: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[Orchestrator] ⚠  第 {lineno} 行解析失敗: {exc}", file=sys.stderr)
                continue
            # 過濾掉沒有 word 的原始 gaze 推論記錄
            if not obj.get("word"):
                continue
            events.append(obj)
    print(f"[Orchestrator] 讀取 {len(events)} 筆有效眼動事件（來自 {path.name}）")
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 認知端 — 取得 word_analysis[]
# ═══════════════════════════════════════════════════════════════════════════════

def load_cognitive_result(path: Path) -> dict:
    """從已存好的 JSON 檔讀取 CognitiveLoadPipeline 輸出。"""
    with open(path, encoding="utf-8") as fh:
        result = json.load(fh)
    n = len(result.get("word_analysis", []))
    print(f"[Orchestrator] 讀取認知分析結果: {n} 個單字（來自 {path.name}）")
    return result


def run_cognitive_pipeline(text: str, lang: str = "en") -> dict:
    """直接呼叫 weichi CognitiveLoadPipeline 對文字進行分析。"""
    from core.cognition import CognitiveLoadPipeline
    from core.cognition.model_policy import default_model_for_language

    model_type = default_model_for_language(lang)
    print(f"[Orchestrator] 啟動 CognitiveLoadPipeline (lang={lang}, model={model_type})…")
    pipeline = CognitiveLoadPipeline(model_type=model_type, lang=lang)
    result = pipeline.run(text)
    print(f"[Orchestrator] 認知分析完成，共 {len(result.get('word_analysis', []))} 個單字")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 雙模態對齊 — 建立 load_score lookup（含大小寫 + 連字號模糊比對）
# ═══════════════════════════════════════════════════════════════════════════════

def build_cognitive_lookup(word_analysis: list[dict]) -> dict[str, dict]:
    """
    建立 word → WordResult 的查詢表。

    對齊規則（依 fusion_blueprint.md §3 痛點 3 & 4）：
    1. 所有 key 強制 .lower()
    2. 記錄一份 hyphen_lookup：{ 子詞 → 複合詞條目 }，
       供 PDF 拆字後仍能命中完整複合詞的分數。
    """
    lookup: dict[str, dict] = {}
    hyphen_lookup: dict[str, dict] = {}

    for item in word_analysis:
        key = item.get("word", "").lower()
        if not key:
            continue
        lookup[key] = item
        # 若是連字號複合詞，把每個子詞也對應到此條目
        if "-" in key:
            for part in key.split("-"):
                part = part.strip()
                if part and part not in lookup:
                    hyphen_lookup[part] = item

    # 合併：直接命中優先，子詞退回才走 hyphen_lookup
    merged = {**hyphen_lookup, **lookup}
    return merged


def lookup_word(key: str, merged_lookup: dict[str, dict]) -> dict | None:
    """查詢單字的認知分析條目（小寫，含連字號退回）。"""
    return merged_lookup.get(key.lower())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 核心融合演算法 — RDS 計算
# ═══════════════════════════════════════════════════════════════════════════════

W_DWELL   = 0.35   # 停留時間權重
W_FIXATION = 0.25  # 注視次數權重
W_LOAD    = 0.40   # 認知負荷分數權重

PRODUCTION_INELIGIBLE_METHODS = {
    "cross_attention": (
        "cross_attention has no trained checkpoint and is retained only for "
        "historical research reproducibility"
    )
}

DWELL_MS_PER_COUNT = 120  # 前端以 120 ms 輪詢，每次命中代表約 120 ms 停留


def _classify_rds(rds: float) -> str:
    if rds >= 0.70:
        return "difficulty"
    if rds >= 0.40:
        return "attention"
    return "fluent"


def aggregate_gaze_events(events: list[dict]) -> dict[str, dict]:
    """
    將多行 gaze_event 按 word（小寫）彙整，累加 dwell_ms 和 fixation_count。
    """
    agg: dict[str, dict] = {}
    for ev in events:
        key = ev.get("word", "").lower()
        if not key:
            continue
        if key not in agg:
            agg[key] = {
                "word": ev.get("word", key),
                "dwell_ms": 0,
                "fixation_count": 0,
                "hit_count": 0,
                "confidence_counts": {"high": 0, "medium": 0, "low": 0},
            }
        agg[key]["dwell_ms"]       += int(ev.get("dwell_count", 1)) * DWELL_MS_PER_COUNT
        agg[key]["fixation_count"] += int(ev.get("fixation_count", 0))
        agg[key]["hit_count"]      += 1
        conf = ev.get("confidence", "low")
        if conf in agg[key]["confidence_counts"]:
            agg[key]["confidence_counts"][conf] += 1
    return agg


def compute_rds(
    gaze_events: list[dict],
    cognitive_result: dict,
    method: str = "linear",
) -> list[dict]:
    """
    主要融合函式。

    步驟：
      1. 彙整眼動事件 → { word: { dwell_ms, fixation_count } }
      2. 依文字順序建立對齊的眼動 sequence
      3. 套用選取的雙模態融合演算法 (例如 Linear, Bayesian, RRF, Spillover_rrf 等)
      4. 合併細粒度語言學特徵
    """
    method_lower = str(method).strip().lower()
    if method_lower in PRODUCTION_INELIGIBLE_METHODS:
        raise ValueError(PRODUCTION_INELIGIBLE_METHODS[method_lower])

    word_analysis = cognitive_result.get("word_analysis", [])
    aggregated    = aggregate_gaze_events(gaze_events)

    if not word_analysis:
        print("[Orchestrator] ⚠ 沒有有效的認知分析單字序列")
        return []

    # 提取序列資料
    words_seq = [item.get("word", "") for item in word_analysis]
    load_seq  = [float(item.get("load_score", 0.0)) for item in word_analysis]

    # 對齊眼動資料
    dwell_seq = []
    fix_seq   = []
    for w in words_seq:
        key = w.lower()
        if key in aggregated:
            dwell_seq.append(float(aggregated[key]["dwell_ms"]))
            fix_seq.append(float(aggregated[key]["fixation_count"]))
        else:
            dwell_seq.append(0.0)
            fix_seq.append(0.0)

    # 執行融合
    fusion = LexiGazeFusion()
    if method_lower == "linear":
        rds_seq = fusion.fuse_linear(dwell_seq, fix_seq, load_seq)
    elif method_lower == "multiplicative":
        rds_seq = fusion.fuse_multiplicative(dwell_seq, fix_seq, load_seq)
    elif method_lower == "gated":
        rds_seq = fusion.fuse_gated(dwell_seq, fix_seq, load_seq)
    elif method_lower == "sigmoid":
        rds_seq = fusion.fuse_sigmoid(dwell_seq, fix_seq, load_seq)
    elif method_lower == "bayesian":
        rds_seq = fusion.fuse_bayesian(dwell_seq, load_seq)
    elif method_lower == "rrf":
        rds_seq = fusion.fuse_rrf(dwell_seq, load_seq)
    elif method_lower == "spillover_bayesian":
        rds_seq = fusion.fuse_spillover_bayesian(dwell_seq, load_seq)
    elif method_lower == "parafoveal":
        rds_seq = fusion.fuse_parafoveal(dwell_seq, load_seq)
    elif method_lower == "spillover_rrf":
        rds_seq = fusion.fuse_spillover_rrf(dwell_seq, load_seq)
    elif method_lower == "parafoveal_rrf":
        rds_seq = fusion.fuse_parafoveal_rrf(dwell_seq, load_seq)
    elif method_lower == "spillover_parafoveal_rrf":
        rds_seq = fusion.fuse_spillover_parafoveal_rrf(dwell_seq, load_seq)
    elif method_lower == "fatigue_adaptive":
        rds_seq = fusion.fuse_fatigue_adaptive(dwell_seq, load_seq)
    else:
        rds_seq = fusion.fuse_linear(dwell_seq, fix_seq, load_seq)

    # 用於正規化顯示的邊界計算
    max_dwell   = max(dwell_seq) if max(dwell_seq) > 0 else 1
    max_fix     = max(fix_seq) if max(fix_seq) > 0 else 1
    min_dwell   = min(dwell_seq)
    min_fix     = min(fix_seq)
    range_dwell = max_dwell - min_dwell or 1
    range_fix   = max_fix   - min_fix   or 1

    results: list[dict] = []
    for i, item in enumerate(word_analysis):
        w = item.get("word", "")
        key = w.lower()
        agg = aggregated.get(key, {
            "dwell_ms": 0,
            "fixation_count": 0,
            "hit_count": 0,
            "confidence_counts": {"high": 0, "medium": 0, "low": 0}
        })

        rds = round(float(rds_seq[i]), 4)
        dwell_norm = (agg["dwell_ms"]       - min_dwell) / range_dwell
        fix_norm   = (agg["fixation_count"] - min_fix)   / range_fix

        results.append({
            # ── 融合核心輸出 ──
            "word":            w,
            "rds":             rds,
            "rds_level":       _classify_rds(rds),

            # ── 眼動感知端 ──
            "dwell_ms":        agg["dwell_ms"],
            "dwell_norm":      round(dwell_norm, 4),
            "fixation_count":  agg["fixation_count"],
            "fix_norm":        round(fix_norm, 4),
            "hit_count":       agg["hit_count"],
            "confidence_counts": agg["confidence_counts"],

            # ── 語言認知端 ──
            "load_score":      float(item.get("load_score", 0.0)),
            "load_level":      item.get("load_level", ""),
            "pos":             item.get("pos", ""),
            "surprisal":       item.get("surprisal", None),
            "entropy":         item.get("entropy", None),
            "renyi_entropy":   item.get("renyi_entropy", None),
            "dependency_load": item.get("dependency_load", None),
            "zipf_score":      item.get("zipf_score", None),
            "word_length":     item.get("word_length", None),
            "aoa_score":       item.get("aoa_score", None),
            "pos_score":       item.get("pos_score", None),
        })

    # 依 RDS 由高到低排序
    results.sort(key=lambda x: x["rds"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 報告輸出
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(
    session_id: str,
    rds_results: list[dict],
    cognitive_result: dict,
    gaze_event_count: int,
    elapsed_ms: int,
) -> Path:
    """
    將融合結果寫入 docs/fusion_reports/<session_id>.json。
    """
    difficulty_words = [r["word"] for r in rds_results if r["rds_level"] == "difficulty"]
    attention_words  = [r["word"] for r in rds_results if r["rds_level"] == "attention"]
    fluent_words     = [r["word"] for r in rds_results if r["rds_level"] == "fluent"]

    report = {
        "session_id":        session_id,
        "fusion_version":    "1.0",
        "generated_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_ms":        elapsed_ms,
        "weights": {
            "dwell":    W_DWELL,
            "fixation": W_FIXATION,
            "load":     W_LOAD,
        },
        "summary": {
            "total_words_tracked":  len(rds_results),
            "gaze_events_ingested": gaze_event_count,
            "difficulty_count":     len(difficulty_words),
            "attention_count":      len(attention_words),
            "fluent_count":         len(fluent_words),
            "mean_rds":             round(
                sum(r["rds"] for r in rds_results) / len(rds_results), 4
            ) if rds_results else 0.0,
        },
        "difficulty_words":  difficulty_words,
        "attention_words":   attention_words,
        # ── 原始認知分析 metadata ──
        "cognitive_model":   cognitive_result.get("model"),
        "cognitive_lang":    cognitive_result.get("lang"),
        "cognitive_domain":  cognitive_result.get("domain"),
        # ── 完整 RDS 結果（含所有細粒度特徵）──
        "rds_results":       rds_results,
    }

    out_path = REPORTS_DIR / f"{session_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[Orchestrator] ✅ 報告已輸出至 {out_path}")
    print(f"[Orchestrator]    難詞({len(difficulty_words)}) 注意({len(attention_words)}) 流暢({len(fluent_words)})")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLI 入口
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LexiGaze Fusion Orchestrator — 感知 × 認知雙模態融合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gaze-log", required=True, metavar="PATH",
        help="眼動事件日誌路徑（.jsonl），每行一筆已對齊的 gaze event",
    )
    parser.add_argument(
        "--cognitive", metavar="PATH",
        help="已有的認知分析 JSON 路徑（pipeline.run() 的輸出）",
    )
    parser.add_argument(
        "--text", metavar="TEXT",
        help="直接傳入原始文字，讓 orchestrator 呼叫 pipeline 分析（與 --cognitive 二擇一）",
    )
    parser.add_argument(
        "--lang", default="en", choices=["en", "zh"],
        help="語言（與 --text 搭配使用，預設 en）",
    )
    parser.add_argument(
        "--session-id", default="session",
        help="Session 識別碼，用於命名輸出報告（預設 'session'）",
    )
    parser.add_argument(
        "--save-cognitive", metavar="PATH",
        help="（選用）將認知分析結果另存到指定路徑",
    )
    parser.add_argument(
        "--method", default="linear",
        choices=["linear", "multiplicative", "gated", "sigmoid", "bayesian", "rrf",
                 "spillover_bayesian", "parafoveal", "spillover_rrf", "parafoveal_rrf",
                 "spillover_parafoveal_rrf"],
        help="雙模態融合演算法類型（預設 'linear'）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # 1. 載入眼動事件
    gaze_path = Path(args.gaze_log)
    if not gaze_path.exists():
        print(f"[Orchestrator] ❌ 找不到 gaze_log: {gaze_path}", file=sys.stderr)
        sys.exit(1)
    gaze_events = load_gaze_log(gaze_path)

    # 2. 取得認知分析結果
    if args.cognitive:
        cog_path = Path(args.cognitive)
        if not cog_path.exists():
            print(f"[Orchestrator] ❌ 找不到 cognitive JSON: {cog_path}", file=sys.stderr)
            sys.exit(1)
        cognitive_result = load_cognitive_result(cog_path)
    elif args.text:
        cognitive_result = run_cognitive_pipeline(args.text, lang=args.lang)
        if args.save_cognitive:
            save_path = Path(args.save_cognitive)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as fh:
                json.dump(cognitive_result, fh, ensure_ascii=False, indent=2)
            print(f"[Orchestrator] 認知分析結果已另存至 {save_path}")
    else:
        print(
            "[Orchestrator] ❌ 請提供 --cognitive（已有的 JSON）或 --text（直接分析文字）",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. 融合計算
    print("[Orchestrator] 開始 RDS 融合計算…")
    rds_results = compute_rds(gaze_events, cognitive_result, method=args.method)

    # 4. 輸出報告
    elapsed_ms = int((time.time() - t0) * 1000)
    write_report(
        session_id=args.session_id,
        rds_results=rds_results,
        cognitive_result=cognitive_result,
        gaze_event_count=len(gaze_events),
        elapsed_ms=elapsed_ms,
    )


if __name__ == "__main__":
    main()
