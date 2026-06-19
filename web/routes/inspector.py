from __future__ import annotations

import json
import time
from pathlib import Path
from flask import Blueprint, jsonify, request

from core.cognitive_inspector import CognitiveInspector, generate_markdown_report

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "docs" / "cognitive_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

inspector_bp = Blueprint("inspector", __name__, url_prefix="/api/inspector")

@inspector_bp.post("/analyze")
def analyze():
    """
    分析眼動歷史軌跡，回傳能力與負荷指標畫像。
    """
    body = request.get_json(force=True) or {}
    gaze_history = body.get("gaze_history", [])
    lang = body.get("lang", "en")

    if not isinstance(gaze_history, list):
        return jsonify({"ok": False, "error": "'gaze_history' 必須是陣列"}), 400

    inspector = CognitiveInspector()
    result = inspector.analyze(gaze_history, lang=lang)

    return jsonify({
        "ok": True,
        "analysis": result
    })

@inspector_bp.post("/report")
def generate_report():
    """
    產生精緻的 Markdown 閱讀能力診斷報告，並可選擇性持久化儲存。
    """
    body = request.get_json(force=True) or {}
    gaze_history = body.get("gaze_history", [])
    participant_id = body.get("participant_id", "anonymous").strip() or "anonymous"
    lang = body.get("lang", "en")
    persist = bool(body.get("persist", False))

    if not isinstance(gaze_history, list):
        return jsonify({"ok": False, "error": "'gaze_history' 必須是陣列"}), 400

    inspector = CognitiveInspector()
    result = inspector.analyze(gaze_history, lang=lang)
    
    report_md = generate_markdown_report(result, participant_id)

    if persist:
        timestamp = int(time.time())
        safe_id = "".join([c for c in participant_id if c.isalnum() or c in ("-", "_")]).strip()
        filename = f"{safe_id}_{timestamp}.md"
        out_path = REPORTS_DIR / filename
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report_md)
        result["report_path"] = f"docs/cognitive_reports/{filename}"

    return jsonify({
        "ok": True,
        "report_md": report_md,
        "analysis": result
    })
