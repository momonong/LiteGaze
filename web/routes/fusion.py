"""
chenghao/fusion_routes.py
══════════════════════════════════════════════════════════════════════════════
LexiGaze Fusion Blueprint — POST /api/fuse

即時接收前端送來的「已對齊 gaze-word 事件批次」與「認知分析結果」，
計算每個單字的 RDS（Reading Difficulty Score），並選擇性地持久化到
docs/fusion_reports/<session_id>.json。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

ROOT        = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "docs" / "fusion_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Bootstrap orchestrator on sys.path ──────────────────────────────────────
SCRIPTS_DIR = ROOT / "scripts" / "fusion"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from orchestrator import (  # noqa: E402
    PRODUCTION_INELIGIBLE_METHODS,
    QUALITY_AWARE_SHADOW_METHOD,
    W_DWELL,
    W_FIXATION,
    W_LOAD,
    compute_rds,
)

fusion_bp = Blueprint("fusion", __name__, url_prefix="/api/fuse")


# ── Health ────────────────────────────────────────────────────────────────────
@fusion_bp.get("/health")
def health():
    return jsonify({"ok": True, "backend": "lexigaze-fusion", "version": "1.0"})


# ── Main fusion endpoint ──────────────────────────────────────────────────────
@fusion_bp.post("/")
def fuse():
    """
    接收前端打包好的眼動事件 + 認知分析結果，回傳 per-word RDS。

    Request body:
    {
      "session_id":       "...",          // 選用，用於落地報告命名
      "persist":          true,           // 選用，是否寫入 docs/fusion_reports/
      "cognitive_result": {               // CognitiveLoadPipeline.run() 的完整輸出
        "word_analysis": [
          { "word": "neuro-symbolic", "load_score": 0.92, ... }
        ]
      },
      "gaze_events": [                    // 前端 gazeBuffer 序列化
        {
          "word":           "neuro-symbolic",
          "occurrence_id":  "page:1:word:42",
          "word_index":     42,
          "page_num":       1,
          "confidence":     "high",
          "confidence_counts": {"high": 4, "medium": 1, "low": 0},
          "dwell_count":    5,            // × 120ms = dwell_ms
          "fixation_count": 2,
          "timestamp_ms":   1749912000123
        }
      ],
      "method": "quality_aware_v2_shadow", // explicit experimental opt-in only
      "quality_context": {
        "tracking_coverage": 0.9,
        "stability": 0.8,
        "calibration_quality": 0.85
      }
    }

    Response:
    {
      "ok": true,
      "session_id": "...",
      "elapsed_ms": 12,
      "summary": { "total_words_tracked": N, "difficulty_count": K, ... },
      "rds": [
        { "word": "neuro-symbolic", "rds": 0.81, "rds_level": "difficulty",
          "dwell_ms": 600, "fixation_count": 2, "load_score": 0.92,
          "surprisal": 14.2, ... }
      ]
    }
    """
    t0 = time.time()
    body = request.get_json(force=True) or {}

    session_id       = body.get("session_id") or f"session_{int(t0)}"
    persist          = bool(body.get("persist", False))
    cognitive_result = body.get("cognitive_result") or {}
    gaze_events      = body.get("gaze_events") or []
    method           = body.get("method", "linear")
    quality_context  = body.get("quality_context") or {}

    method_key = str(method).strip().lower()
    if method_key in PRODUCTION_INELIGIBLE_METHODS:
        return jsonify({
            "ok": False,
            "error": "production_ineligible_fusion_method",
            "method": method_key,
            "detail": PRODUCTION_INELIGIBLE_METHODS[method_key],
        }), 422

    if not isinstance(gaze_events, list):
        return jsonify({"ok": False, "error": "'gaze_events' 必須是陣列"}), 400
    if not isinstance(cognitive_result, dict):
        return jsonify({"ok": False, "error": "'cognitive_result' 必須是物件"}), 400
    if not isinstance(quality_context, dict):
        return jsonify({"ok": False, "error": "'quality_context' 必須是物件"}), 400

    # Fallback: if cognitive_result is missing or empty, dynamically reconstruct text from gaze events and run pipeline
    if not cognitive_result or "word_analysis" not in cognitive_result:
        words_seq = [str(e.get("word", "")) for e in gaze_events if e.get("word")]
        if words_seq:
            try:
                import re

                from core.cognition import CognitiveLoadPipeline
                from core.cognition.model_policy import default_model_for_language
                is_zh = any(re.search(r'[\u4e00-\u9fff]', w) for w in words_seq)
                lang = "zh" if is_zh else "en"
                text_str = " ".join(words_seq) if lang == "en" else "".join(words_seq)

                pipeline = CognitiveLoadPipeline(
                    model_type=default_model_for_language(lang),
                    lang=lang,
                )
                cognitive_result = pipeline.run(text_str)
                print(f"[fusion] Fallback CognitiveLoadPipeline ran successfully. Formed {len(cognitive_result.get('word_analysis', []))} words.")
            except Exception as exc:  # noqa: BLE001 - optional fallback must not fail fusion
                print(f"[fusion] Warning: Failed to run fallback CognitiveLoadPipeline: {exc}")

    # Run fusion
    try:
        rds_results = compute_rds(
            gaze_events,
            cognitive_result,
            method=method_key,
            quality_context=quality_context,
        )
    except ValueError as exc:
        return jsonify({
            "ok": False,
            "error": "invalid_fusion_input",
            "method": method_key,
            "detail": str(exc),
        }), 400

    elapsed_ms = int((time.time() - t0) * 1000)

    difficulty_words = [r["word"] for r in rds_results if r["rds_level"] == "difficulty"]
    attention_words  = [r["word"] for r in rds_results if r["rds_level"] == "attention"]

    summary = {
        "fusion_method":       method_key,
        "candidate_status":    (
            "shadow_only" if method_key == QUALITY_AWARE_SHADOW_METHOD else None
        ),
        "total_words_tracked":  len(rds_results),
        "gaze_events_ingested": len(gaze_events),
        "difficulty_count":     len(difficulty_words),
        "attention_count":      len(attention_words),
        "fluent_count":         len(rds_results) - len(difficulty_words) - len(attention_words),
        "mean_rds": round(
            sum(r["rds"] for r in rds_results) / len(rds_results), 4
        ) if rds_results else 0.0,
    }

    # Optional: persist to docs/fusion_reports/<session_id>.json
    if persist and session_id:
        _persist_report(
            session_id,
            rds_results,
            cognitive_result,
            summary,
            elapsed_ms,
            method_key,
        )

    return jsonify({
        "ok":         True,
        "session_id": session_id,
        "fusion_method": method_key,
        "candidate_status": (
            "shadow_only" if method_key == QUALITY_AWARE_SHADOW_METHOD else None
        ),
        "elapsed_ms": elapsed_ms,
        "summary":    summary,
        "rds":        rds_results,
    })


# ── Persist report ────────────────────────────────────────────────────────────
def _persist_report(
    session_id: str,
    rds_results: list[dict],
    cognitive_result: dict,
    summary: dict,
    elapsed_ms: int,
    method: str,
) -> Path:
    report = {
        "session_id":       session_id,
        "fusion_version":   "1.0",
        "fusion_method":    method,
        "candidate_status": (
            "shadow_only" if method == QUALITY_AWARE_SHADOW_METHOD else None
        ),
        "generated_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_ms":       elapsed_ms,
        "weights":          {"dwell": W_DWELL, "fixation": W_FIXATION, "load": W_LOAD},
        "summary":          summary,
        "difficulty_words": [r["word"] for r in rds_results if r["rds_level"] == "difficulty"],
        "attention_words":  [r["word"] for r in rds_results if r["rds_level"] == "attention"],
        "cognitive_model":  cognitive_result.get("model"),
        "cognitive_lang":   cognitive_result.get("lang"),
        "cognitive_domain": cognitive_result.get("domain"),
        "rds_results":      rds_results,
    }
    out_path = REPORTS_DIR / f"{session_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return out_path


# ── List saved reports ────────────────────────────────────────────────────────
@fusion_bp.get("/reports")
def list_reports():
    """列出 docs/fusion_reports/ 下所有已儲存的融合報告。"""
    items = []
    for f in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        try:
            stat = f.stat()
            items.append({
                "session_id": f.stem,
                "filename":   f.name,
                "size":       stat.st_size,
            })
        except OSError:
            pass
    return jsonify({"ok": True, "reports": items})


# ── Get single report ─────────────────────────────────────────────────────────
@fusion_bp.get("/reports/<session_id>")
def get_report(session_id: str):
    """取回指定 session 的融合報告 JSON。"""
    path = REPORTS_DIR / f"{session_id}.json"
    if not path.exists():
        return jsonify({"ok": False, "error": "找不到報告"}), 404
    with open(path, encoding="utf-8") as fh:
        return jsonify(json.load(fh))
