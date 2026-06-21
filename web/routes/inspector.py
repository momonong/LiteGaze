from __future__ import annotations

import json
import time
from datetime import datetime
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

@inspector_bp.get("/reports")
def list_reports():
    """
    列出所有已儲存的 Markdown 認知診斷報告。
    """
    reports = []
    for f in REPORTS_DIR.glob("*.md"):
        try:
            name = f.name
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                p_id, ts_str = parts
                try:
                    ts = int(ts_str)
                    dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    p_id = f.stem
                    dt = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            else:
                p_id = f.stem
                dt = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            reports.append({
                "filename": name,
                "participant_id": p_id,
                "created_at": dt,
                "size_bytes": f.stat().st_size
            })
        except Exception:
            pass
    
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify({"ok": True, "reports": reports})

@inspector_bp.get("/reports/<filename>")
def get_report(filename):
    """
    取得特定認知診斷報告的 Markdown 內容。
    """
    safe_name = Path(filename).name
    file_path = REPORTS_DIR / safe_name
    if not file_path.exists():
        return jsonify({"ok": False, "error": "報告未找到"}), 404
    
    try:
        content = file_path.read_text(encoding="utf-8")
        return jsonify({"ok": True, "markdown": content})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@inspector_bp.delete("/reports/<filename>")
def delete_report(filename):
    """
    刪除特定的認知診斷報告。
    """
    safe_name = Path(filename).name
    file_path = REPORTS_DIR / safe_name
    if not file_path.exists():
        return jsonify({"ok": False, "error": "報告未找到"}), 404
    
    try:
        file_path.unlink()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@inspector_bp.post("/quiz")
def generate_agentic_quiz():
    """
    Generate an agentic cognitive reading quiz using the Gemini API,
    focusing on sentences where the user struggled or spent the most time.
    """
    import os
    import json
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("MODEL_NAME", "gemini-1.5-flash")

    if not api_key:
        return jsonify({"ok": False, "error": "Missing GEMINI_API_KEY environment variable"}), 500

    body = request.get_json(force=True) or {}
    gaze_history = body.get("gaze_history", [])
    session_id = body.get("session_id", "")
    lang = body.get("lang", "en")

    if not session_id:
        return jsonify({"ok": False, "error": "Missing session_id"}), 400

    # 1. Load the document layout session data to get the parsed words
    data_dir = ROOT / "data"
    session_path = data_dir / f"{session_id}.json"
    if not session_path.exists():
        return jsonify({"ok": False, "error": f"Session layout {session_id} not found"}), 404

    try:
        with open(session_path, encoding="utf-8") as f:
            session_data = json.load(f)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to load layout data: {e}"}), 500

    items = session_data.get("items", [])
    if not items:
        return jsonify({"ok": False, "error": "No items in document layout session"}), 400

    # 2. Reconstruct sentences and map word indices to sentences
    sentences = []
    current_sentence = []
    for item in items:
        current_sentence.append(item)
        text = item.get("text", "")
        # Sentence boundary markers
        if any(text.endswith(p) for p in (".", "?", "!", "。", "？", "！", "\n")):
            sentences.append(current_sentence)
            current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)

    # 3. Find top struggled word indices in gaze_history
    word_counts = {}
    for hit in gaze_history:
        idx = hit.get("index", -1)
        if idx != -1:
            word_counts[idx] = word_counts.get(idx, 0) + 1

    # Sort word indices by counts descending
    sorted_struggled_indices = sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)

    # 4. Extract sentences containing the struggled words
    struggled_sentences = []
    visited_sentences = set()

    for idx in sorted_struggled_indices:
        for i, sent in enumerate(sentences):
            if i in visited_sentences:
                continue
            if any(item.get("index") == idx for item in sent):
                # Format the sentence text
                sent_text = " ".join([item.get("text", "") for item in sent]).strip()
                if sent_text:
                    struggled_sentences.append(sent_text)
                    visited_sentences.add(i)
                break
        if len(struggled_sentences) >= 3:
            break

    # Fallback: if no struggled sentences matched (or gaze_history was empty), pick up to 3 sentences
    if not struggled_sentences:
        for i, sent in enumerate(sentences[:10]):  # pick from the first few sentences
            sent_text = " ".join([item.get("text", "") for item in sent]).strip()
            if sent_text:
                struggled_sentences.append(sent_text)
            if len(struggled_sentences) >= 3:
                break

    # 5. Formulate Gemini prompt
    sentences_str = "\n".join([f"- {s}" for s in struggled_sentences])
    
    prompt = f"""You are a Cognitive Reading Inspector. The user read a document, and eye-tracking logs show they struggled (had high cognitive load/fixation) on the following sentences:

{sentences_str}

Please generate exactly 3 reading comprehension questions to verify if the user successfully understood the meaning of these specific sentences.
Provide:
1. The question (in the same language as the sentences, i.e., English or Traditional Chinese).
2. Four multiple-choice options (A, B, C, D).
3. The correct option key (A, B, C, or D).
4. A brief, helpful explanation of the correct answer.

Return the response in structured JSON format matching this exact schema:
{{
  "questions": [
     {{
       "id": 1,
       "question": "...",
       "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
       "answer": "A",
       "explanation": "..."
     }},
     {{
       "id": 2,
       "question": "...",
       "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
       "answer": "B",
       "explanation": "..."
     }},
     {{
       "id": 3,
       "question": "...",
       "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
       "answer": "C",
       "explanation": "..."
     }}
  ]
}}"""

    # 6. Query Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code != 200:
            return jsonify({"ok": False, "error": f"Gemini API returned status {res.status_code}: {res.text}"}), 500
        
        data = res.json()
        text_response = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Load generated JSON quiz
        quiz_data = json.loads(text_response)
        return jsonify({
            "ok": True,
            "quiz": quiz_data.get("questions", []),
            "struggled_sentences": struggled_sentences
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to generate quiz: {e}"}), 500
