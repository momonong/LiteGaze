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


def _clean_json_response(text: str) -> str:
    import re
    # Strip <thought>...</thought> blocks (case-insensitive, multi-line)
    text = re.sub(r'(?i)<thought>.*?</thought>', '', text, flags=re.DOTALL)
    # Strip ```json and ``` code block wrappers if present
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    return text.strip()


def _generate_fallback_quiz(struggled_sentences, lang="en"):
    questions = []
    for i, sent in enumerate(struggled_sentences[:3]):
        if lang == "zh":
            questions.append({
                "id": i + 1,
                "question": f"關於文章中的句子「{sent}」，以下敘述何者正確？",
                "options": {
                    "A": "該句子確實出現在文章中且為原文的一部份",
                    "B": "該句子完全沒有在文章中出現過",
                    "C": "該句子描述的是與原文相反的觀點",
                    "D": "以上皆非"
                },
                "answer": "A",
                "explanation": f"依據文章內容，句子「{sent}」為原文中的一部分。"
            })
        else:
            questions.append({
                "id": i + 1,
                "question": f"Regarding the sentence \"{sent}\" in the text, which of the following is true?",
                "options": {
                    "A": "The sentence appears in the text and is part of the original passage.",
                    "B": "The sentence does not appear in the text.",
                    "C": "The sentence states the opposite of the fact.",
                    "D": "None of the above."
                },
                "answer": "A",
                "explanation": f"According to the text, the sentence \"{sent}\" is part of the original passage."
            })
    return questions


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

    # If API key is missing, fall back to local quiz directly
    if not api_key:
        print("[Quiz] Missing GEMINI_API_KEY. Using local fallback quiz.")
        fallback_quiz = _generate_fallback_quiz(struggled_sentences, lang)
        return jsonify({
            "ok": True,
            "quiz": fallback_quiz,
            "struggled_sentences": struggled_sentences
        })

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
            fallback_model = "gemini-1.5-flash"
            print(f"[Quiz] Model {model_name} failed with status {res.status_code}. Retrying with {fallback_model}...")
            fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback_model}:generateContent?key={api_key}"
            res = requests.post(fallback_url, headers=headers, json=payload, timeout=30)
            
        if res.status_code != 200:
            print(f"[Quiz] Gemini API failed with status {res.status_code}. Using local fallback quiz.")
            fallback_quiz = _generate_fallback_quiz(struggled_sentences, lang)
            return jsonify({
                "ok": True,
                "quiz": fallback_quiz,
                "struggled_sentences": struggled_sentences
            })
        
        data = res.json()
        parts = data["candidates"][0]["content"]["parts"]
        non_thought_parts = [p["text"] for p in parts if not p.get("thought")]
        if non_thought_parts:
            text_response = non_thought_parts[0]
        else:
            text_response = parts[-1]["text"]
        
        # Load generated JSON quiz
        try:
            quiz_data = json.loads(_clean_json_response(text_response))
            return jsonify({
                "ok": True,
                "quiz": quiz_data.get("questions", []),
                "struggled_sentences": struggled_sentences
            })
        except Exception as json_err:
            print(f"[Quiz] JSON parsing failed: {json_err}. Using local fallback quiz.")
            fallback_quiz = _generate_fallback_quiz(struggled_sentences, lang)
            return jsonify({
                "ok": True,
                "quiz": fallback_quiz,
                "struggled_sentences": struggled_sentences
            })
    except Exception as e:
        print(f"[Quiz] Exception occurred: {e}. Using local fallback quiz.")
        fallback_quiz = _generate_fallback_quiz(struggled_sentences, lang)
        return jsonify({
            "ok": True,
            "quiz": fallback_quiz,
            "struggled_sentences": struggled_sentences
        })


# ── Adaptive Agentic Testing Module Data & Endpoints ─────────────────────────

ADAPTIVE_PARAGRAPHS = {
    "en": {
        1: {
            "text": "Reading is a very good habit. It helps you learn new things and improves your vocabulary. When you read every day, you can discover interesting stories and gain a lot of knowledge. Many people like to read books before going to sleep because it relaxes their minds.",
            "difficulty": "easy",
            "font_size": 16,
            "line_width": 650,
            "line_height": 1.6,
            "quiz": [
                {
                    "question": "Why do many people like to read books before going to sleep?",
                    "options": {"A": "It helps them study", "B": "It relaxes their minds", "C": "It improves their eyesight", "D": "It makes them feel hungry"},
                    "answer": "B",
                    "explanation": "The text states that many people read books before going to sleep because it relaxes their minds."
                },
                {
                    "question": "What are the benefits of reading daily mentioned in the text?",
                    "options": {"A": "Discovering interesting stories and gaining knowledge", "B": "Learning how to write books", "C": "Saving money on sleeping pills", "D": "Getting higher grades in school"},
                    "answer": "A",
                    "explanation": "The text mentions you can discover interesting stories and gain a lot of knowledge when reading every day."
                }
            ]
        },
        2: {
            "easy": {
                "text": "Water is essential for all living things. Animals need water to drink, and plants need water to grow. Without clean water, people can get very sick. It is important to protect our lakes and rivers from pollution so that we always have clean water to use and enjoy.",
                "difficulty": "easy",
                "font_size": 18,
                "line_width": 750,
                "line_height": 1.8,
                "quiz": [
                    {
                        "question": "What happens if people do not have access to clean water?",
                        "options": {"A": "They get rich", "B": "They can get very sick", "C": "They run faster", "D": "They learn quickly"},
                        "answer": "B",
                        "explanation": "Without clean water, people can get very sick."
                    },
                    {
                        "question": "Why should we protect lakes and rivers from pollution?",
                        "options": {"A": "To keep boats clean", "B": "To ensure we always have clean water", "C": "To save fish food", "D": "To increase rain"},
                        "answer": "B",
                        "explanation": "It is important to protect our lakes and rivers from pollution so that we always have clean water."
                    }
                ]
            },
            "medium": {
                "text": "The rapid acceleration of global urbanization presents complex socio-ecological challenges. As urban populations swell, cities must optimize infrastructure, energy distribution, and resource consumption. Smart cities utilize interconnected sensors to monitor traffic flow, air quality, and electrical grids in real time, aiming to improve sustainability.",
                "difficulty": "medium",
                "font_size": 14,
                "line_width": 550,
                "line_height": 1.5,
                "quiz": [
                    {
                        "question": "What do smart cities use to monitor traffic and air quality?",
                        "options": {"A": "Automated vehicles", "B": "Interconnected sensors", "C": "Satellite imagery", "D": "Citizen reports"},
                        "answer": "B",
                        "explanation": "Smart cities utilize interconnected sensors to monitor traffic flow and air quality."
                    },
                    {
                        "question": "What is one of the main goals of smart cities mentioned in the text?",
                        "options": {"A": "To increase population growth", "B": "To improve sustainability", "C": "To replace human labor", "D": "To reduce tax rates"},
                        "answer": "B",
                        "explanation": "Smart cities use sensors aiming to improve sustainability."
                    }
                ]
            }
        },
        3: {
            "easy": {
                "text": "Regular physical exercise is important for maintaining good health. It strengthens your muscles, boosts your cardiovascular system, and helps reduce stress. Even a simple daily walk of thirty minutes can significantly improve your physical well-being and increase your energy levels.",
                "difficulty": "easy",
                "font_size": 16,
                "line_width": 600,
                "line_height": 1.6,
                "quiz": [
                    {
                        "question": "How long of a daily walk is recommended for improving health?",
                        "options": {"A": "Five minutes", "B": "Ten minutes", "C": "Thirty minutes", "D": "Two hours"},
                        "answer": "C",
                        "explanation": "A simple daily walk of thirty minutes can significantly improve physical well-being."
                    },
                    {
                        "question": "Which of the following is NOT a benefit of exercise mentioned in the text?",
                        "options": {"A": "Strengthening muscles", "B": "Reducing stress", "C": "Boosting vocabulary", "D": "Increasing energy levels"},
                        "answer": "C",
                        "explanation": "Boosting vocabulary is not mentioned in the text."
                    }
                ]
            },
            "medium": {
                "text": "Electric cars are becoming very popular. They use batteries instead of petrol, which means they do not produce exhaust gases. This helps to reduce air pollution in cities. However, we still need to build more charging stations to make long trips easier for drivers.",
                "difficulty": "medium",
                "font_size": 14,
                "line_width": 550,
                "line_height": 1.5,
                "quiz": [
                    {
                        "question": "How do electric cars help reduce air pollution?",
                        "options": {"A": "They are smaller", "B": "They do not produce exhaust gases", "C": "They drive slower", "D": "They are made of wood"},
                        "answer": "B",
                        "explanation": "They use batteries instead of petrol, which means they do not produce exhaust gases."
                    },
                    {
                        "question": "What is still needed to make long trips easier for electric car drivers?",
                        "options": {"A": "More charging stations", "B": "Faster speed limits", "C": "Free petrol", "D": "Bigger tires"},
                        "answer": "A",
                        "explanation": "We still need to build more charging stations to make long trips easier for drivers."
                    }
                ]
            },
            "hard": {
                "text": "The emergence of neuro-symbolic artificial intelligence signifies a profound paradigm shift in machine learning, integrating connectionist neural architectures with classical symbolic reasoning methodologies. By synthesis of high-dimensional perceptual representation and formal logical inference, such hybrid systems alleviate the inherent opacity and data-inefficiency of deep neural networks.",
                "difficulty": "hard",
                "font_size": 12,
                "line_width": 450,
                "line_height": 1.4,
                "quiz": [
                    {
                        "question": "What does neuro-symbolic AI combine?",
                        "options": {"A": "Computer vision and audio processing", "B": "Connectionist neural architectures and classical symbolic reasoning", "C": "Quantum computing and blockchain", "D": "Supervised and unsupervised learning"},
                        "answer": "B",
                        "explanation": "It integrates connectionist neural architectures with classical symbolic reasoning methodologies."
                    },
                    {
                        "question": "What problem of deep neural networks does neuro-symbolic AI help solve?",
                        "options": {"A": "Hardware dependency", "B": "Inherent opacity and data-inefficiency", "C": "Lack of user interfaces", "D": "Slow inference speed"},
                        "answer": "B",
                        "explanation": "It alleviates the inherent opacity and data-inefficiency of deep neural networks."
                    }
                ]
            }
        }
    },
    "zh": {
        1: {
            "text": "閱讀是一個非常好的習慣。它能幫助你學習新事物，並擴展你的詞彙量。當你每天閱讀時，你可以發現有趣的故事並獲得豐富的知識。許多人喜歡在睡前看書，因為這能放鬆他們的心靈並帶來寧靜。",
            "difficulty": "easy",
            "font_size": 16,
            "line_width": 650,
            "line_height": 1.6,
            "quiz": [
                {
                    "question": "為什麼許多人喜歡在睡前看書？",
                    "options": {"A": "為了準備考試", "B": "因為這能放鬆他們的心靈", "C": "為了讓眼睛疲勞", "D": "為了學習寫作"},
                    "answer": "B",
                    "explanation": "文中提到許多人喜歡在睡前看書，因為這能放鬆他們的心靈並帶來寧靜。"
                },
                {
                    "question": "根據本文，每天閱讀有什麼好處？",
                    "options": {"A": "可以發現有趣的故事並獲得知識", "B": "可以賺取更多錢財", "C": "可以強身健體", "D": "可以幫助睡眠"},
                    "answer": "A",
                    "explanation": "文中指出，每天閱讀時可以發現有趣的故事並獲得豐富的知識。"
                }
            ]
        },
        2: {
            "easy": {
                "text": "水是所有生物生存不可或缺的資源。動物需要喝水，植物需要水分才能生長。如果沒有乾淨的水，人們就很容易生病。因此，我們必須保護河川與湖泊，避免水源受到污染，這樣我們才能一直有乾淨的水可以使用。",
                "difficulty": "easy",
                "font_size": 18,
                "line_width": 750,
                "line_height": 1.8,
                "quiz": [
                    {
                        "question": "如果人們沒有乾淨的水會怎麼樣？",
                        "options": {"A": "會變得富有", "B": "很容易生病", "C": "會跑得更快", "D": "學習更迅速"},
                        "answer": "B",
                        "explanation": "如果沒有乾淨的水，人們就很容易生病。"
                    },
                    {
                        "question": "為什麼我們應該保護河川與湖泊免受污染？",
                        "options": {"A": "為了保持船隻乾淨", "B": "為了確保一直有乾淨的水可用", "C": "為了節省魚飼料", "D": "為了增加雨量"},
                        "answer": "B",
                        "explanation": "保護水源是為了讓我們一直有乾淨的水可以使用。"
                    }
                ]
            },
            "medium": {
                "text": "全球都市化進程的急遽加速帶來了複雜的社會生態挑戰。隨著都市人口膨脹，城市必須優化基礎建設、能源分配與資源消耗。智慧城市利用互聯的感測器即時監測交通流量、空氣品質與電力網絡，旨在提升都市永續性與居民生活品質。",
                "difficulty": "medium",
                "font_size": 14,
                "line_width": 550,
                "line_height": 1.5,
                "quiz": [
                    {
                        "question": "智慧城市使用什麼來監測交通流量和空氣品質？",
                        "options": {"A": "自動駕駛車輛", "B": "互聯的感測器", "C": "衛星影像", "D": "市民報告"},
                        "answer": "B",
                        "explanation": "智慧城市利用互聯的感測器即時監測交通流量與空氣品質。"
                    },
                    {
                        "question": "文中提到智慧城市的主要目標之一是什麼？",
                        "options": {"A": "增加人口增長", "B": "提升都市永續性", "C": "取代人力勞動", "D": "降低稅率"},
                        "answer": "B",
                        "explanation": "智慧城市使用感測器監測，旨在提升都市永續性與生活品質。"
                    }
                ]
            }
        },
        3: {
            "easy": {
                "text": "規律的身體運動對於維持良好健康非常重要。它能強健你的肌肉、促進心血管系統，並有助於減輕壓力和焦慮。即使只是每天進行簡單的步行三十分鐘，也能顯著改善你的身體機能，並提升日常生活的活力。",
                "difficulty": "easy",
                "font_size": 16,
                "line_width": 600,
                "line_height": 1.6,
                "quiz": [
                    {
                        "question": "文中建議每天步行多少時間以改善健康？",
                        "options": {"A": "五分鐘", "B": "分鍾", "C": "三十分鐘", "D": "兩個小時"},
                        "answer": "C",
                        "explanation": "文中指出每天進行簡單的步行三十分鐘就能顯著改善身體機能。"
                    },
                    {
                        "question": "下列何者非文中提到運動的好處？",
                        "options": {"A": "強健肌肉", "B": "減輕壓力", "C": "擴展詞彙", "D": "提升活力"},
                        "answer": "C",
                        "explanation": "文中提到運動有助於強健肌肉、促進心血管、減輕壓力與提升活力，未提及擴展詞彙。"
                    }
                ]
            },
            "medium": {
                "text": "電動車正變得越來越受歡迎。它們使用電池代替汽油，這意味著它們不會產生廢氣排放。這有助於減少城市中的空氣污染。然而，我們仍需要建設更多的充電站，以便讓車主在進行長途旅行時更加方便。",
                "difficulty": "medium",
                "font_size": 14,
                "line_width": 550,
                "line_height": 1.5,
                "quiz": [
                    {
                        "question": "電動車如何幫助減少空氣污染？",
                        "options": {"A": "它們體積較小", "B": "它們不產生廢氣排放", "C": "它們行駛較慢", "D": "它們是由木頭製造的"},
                        "answer": "B",
                        "explanation": "電動車使用電池代替汽油，這意味著它們不會產生廢氣排放。"
                    },
                    {
                        "question": "目前還需要建設什麼來方便電動車主進行長途旅行？",
                        "options": {"A": "充電站", "B": "更快的限速", "C": "免費汽油", "D": "更大的輪胎"},
                        "answer": "A",
                        "explanation": "文中世紀仍需要建設更多的充電站，以便讓車主在長途旅行時更方便。"
                    }
                ]
            },
            "hard": {
                "text": "神經符號人工智慧的興起標誌著機器學習領域的深刻典範轉移。它將聯結主義的神經架構與經典的符號推理方法相結合。藉由高維感知表徵與形式邏輯推理的綜合，此類混合系統在保留深度神經網路強大優化能力的同時，有效緩解了其固有的不透明性與數據低效性。",
                "difficulty": "hard",
                "font_size": 12,
                "line_width": 450,
                "line_height": 1.4,
                "quiz": [
                    {
                        "question": "神經符號人工智慧結合了什麼？",
                        "options": {"A": "電腦視覺與音訊處理", "B": "聯結主義神經架構與經典符號推理", "C": "量子計算與區塊鏈技術", "D": "監督式式與非監督式學習"},
                        "answer": "B",
                        "explanation": "神經符號人工智慧將聯結主義的神經架構與經典的符號推理方法相結合。"
                    },
                    {
                        "question": "神經符號人工智慧有助於緩解深度神經網路的什麼問題？",
                        "options": {"A": "硬體依賴性", "B": "固有的不透明性與數據低效性", "C": "缺乏使用者介面", "D": "推論速度慢"},
                        "answer": "B",
                        "explanation": "藉由相結合的架構，此類系統有效緩解了深度神經網路固有的不透明性與數據低效性。"
                    }
                ]
            }
        }
    }
}

@inspector_bp.post("/adaptive/start")
def adaptive_start():
    """
    初始化適應性閱讀測試，回傳 Round 1 基線設定。
    """
    body = request.get_json(force=True) or {}
    lang = body.get("lang", "en")
    if lang not in ("en", "zh"):
        lang = "en"
    
    baseline = ADAPTIVE_PARAGRAPHS[lang][1]
    return jsonify({
        "ok": True,
        "round": 1,
        "text": baseline["text"],
        "difficulty": baseline["difficulty"],
        "font_size": baseline["font_size"],
        "line_width": baseline["line_width"],
        "line_height": baseline["line_height"],
        "quiz": baseline["quiz"]
    })

@inspector_bp.post("/adaptive/next")
def adaptive_next():
    """
    根據使用者之前的閱讀效能與測驗答題狀況，動態生成/決定下一輪的文本與視覺排版。
    """
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("MODEL_NAME", "gemma-4-26b-a4b-it")

    body = request.get_json(force=True) or {}
    lang = body.get("lang", "en")
    current_round = body.get("current_round", 1)
    history = body.get("history", [])

    if lang not in ("en", "zh"):
        lang = "en"

    next_round = current_round + 1
    if next_round > 3:
        return jsonify({"ok": True, "is_finished": True})

    # 1. 根據之前的表現，決定難度方向 (做為規則基礎與 AI 的引導 context)
    last_round_data = history[-1] if history else {}
    last_score = last_round_data.get("quiz_score", 0)
    last_total = last_round_data.get("quiz_total", 2)
    last_reg = last_round_data.get("regression_rate", 0.0)
    last_diff = last_round_data.get("difficulty", "easy")

    # 規則路徑判斷
    if next_round == 2:
        if last_score == last_total and last_reg < 0.15:
            target_difficulty = "medium"
        else:
            target_difficulty = "easy"
    else:  # next_round == 3
        if last_diff == "medium":
            if last_score == last_total:
                target_difficulty = "hard"
            else:
                target_difficulty = "medium"
        else:  # last_diff == "easy"
            if last_score == last_total:
                target_difficulty = "medium"
            else:
                target_difficulty = "easy"

    fallback_data = ADAPTIVE_PARAGRAPHS[lang][next_round][target_difficulty]

    # 2. 嘗試使用 Gemma-4 進行動態代理人生成
    if api_key:
        history_json = json.dumps(history, ensure_ascii=False, indent=2)
        prompt = f"""You are an Adaptive Reading Diagnostic Agent.
The user is participating in an iterative reading capability test to identify their cognitive limit and typography comfort zone.
So far, they have completed the following rounds:
{history_json}

The target difficulty for the next round is determined to be: {target_difficulty}.
Based on their performance in the previous rounds:
- If WPM is high, regressions are low, and quiz answers are correct, increase difficulty, or make layout tighter (smaller font, narrower/wider lines).
- If they struggled (low WPM, high regressions, or incorrect quiz answers), lower difficulty, or make layout more readable (larger font, optimized line width).

Your task is to:
1. Generate the next text paragraph (approx. 60-120 words for English, 100-200 characters for Chinese) in the target language: {lang}.
2. Set the visual layout parameters:
   - "font_size": integer between 12 and 24 (px).
   - "line_width": integer between 450 and 850 (px).
   - "line_height": float between 1.4 and 2.2.
   - "difficulty": "{target_difficulty}".
3. Generate exactly 2 multiple-choice questions to test their comprehension of this new paragraph.

Return the response in structured JSON format matching this exact schema:
{{
  "text": "...",
  "difficulty": "{target_difficulty}",
  "font_size": 16,
  "line_width": 600,
  "line_height": 1.6,
  "quiz": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "explanation": "..."
    }},
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "B",
      "explanation": "..."
    }}
  ]
}}"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            if res.status_code != 200:
                fallback_model = "gemini-1.5-flash"
                print(f"[AdaptiveNext] Model {model_name} failed with status {res.status_code}. Retrying with {fallback_model}...")
                fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback_model}:generateContent?key={api_key}"
                res = requests.post(fallback_url, headers=headers, json=payload, timeout=60)

            if res.status_code == 200:
                data = res.json()
                parts = data["candidates"][0]["content"]["parts"]
                non_thought_parts = [p["text"] for p in parts if not p.get("thought")]
                text_response = non_thought_parts[0] if non_thought_parts else parts[-1]["text"]
                
                res_data = json.loads(_clean_json_response(text_response))
                if "text" in res_data and "quiz" in res_data:
                    return jsonify({
                        "ok": True,
                        "round": next_round,
                        "text": res_data["text"],
                        "difficulty": res_data.get("difficulty", target_difficulty),
                        "font_size": int(res_data.get("font_size", fallback_data["font_size"])),
                        "line_width": int(res_data.get("line_width", fallback_data["line_width"])),
                        "line_height": float(res_data.get("line_height", fallback_data["line_height"])),
                        "quiz": res_data["quiz"]
                    })
        except Exception as e:
            print(f"[AdaptiveNext] Exception occurred: {e}")

    return jsonify({
        "ok": True,
        "round": next_round,
        "text": fallback_data["text"],
        "difficulty": fallback_data["difficulty"],
        "font_size": fallback_data["font_size"],
        "line_width": fallback_data["line_width"],
        "line_height": fallback_data["line_height"],
        "quiz": fallback_data["quiz"]
    })

@inspector_bp.post("/adaptive/report")
def adaptive_report():
    """
    彙整 3 輪適應性測驗的眼動指標與答題表現，利用 AI 生成全面性排版優化與閱讀能力診斷報告。
    """
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("MODEL_NAME", "gemma-4-26b-a4b-it")

    body = request.get_json(force=True) or {}
    lang = body.get("lang", "en")
    history = body.get("history", [])
    participant_id = body.get("participant_id", "anonymous").strip() or "anonymous"
    persist = bool(body.get("persist", False))

    if lang not in ("en", "zh"):
        lang = "en"

    # 效率分數 = WPM * (答對題數/總題數) * (1.0 - 回看率)
    best_eff = -1.0
    optimal_round = None
    for r in history:
        wpm = r.get("wpm", 150.0)
        q_score = r.get("quiz_score", 0)
        q_total = r.get("quiz_total", 2)
        score_ratio = q_score / q_total if q_total > 0 else 0.5
        reg_rate = r.get("regression_rate", 0.0)
        
        efficiency = wpm * score_ratio * (1.0 - reg_rate)
        if efficiency > best_eff:
            best_eff = efficiency
            optimal_round = r

    if not optimal_round and history:
        optimal_round = history[0]

    opt_font = optimal_round.get("font_size", 16) if optimal_round else 16
    opt_width = optimal_round.get("line_width", 650) if optimal_round else 650
    opt_height = optimal_round.get("line_height", 1.6) if optimal_round else 1.6
    opt_diff = optimal_round.get("difficulty", "medium") if optimal_round else "medium"

    total_wpm = sum(r.get("wpm", 0) for r in history)
    avg_wpm = round(total_wpm / len(history), 1) if history else 0
    total_correct = sum(r.get("quiz_score", 0) for r in history)
    total_quiz = sum(r.get("quiz_total", 0) for r in history)
    comp_rate = round((total_correct / total_quiz) * 100, 1) if total_quiz > 0 else 0.0

    report_md = ""
    if api_key:
        history_json = json.dumps(history, ensure_ascii=False, indent=2)
        prompt = f"""You are an Expert Cognitive Reading Inspector.
Below is the history of a user's multi-round adaptive reading test, designed to identify their reading capability and typography comfort zone.

Adaptive Session History:
{history_json}

Please generate a detailed, structured "Comprehensive Adaptive Reading Capability & Typography Optimization Report" in Markdown format in the target language: {lang}.
The report must contain:
1. An introductory summary explaining the adaptive testing process.
2. A table summarizing the performance in each round (Round, Difficulty, Font Size, Line Width, WPM, Regression Rate, Quiz Score).
3. "Optimal Typography Recommended Layout" (最佳排版建議) section specifying the exact Font Size, Line Width, and Line Height that minimizes their cognitive load, based on where they had the highest reading efficiency (WPM) and lowest regressions/fixations.
4. "Cognitive Reading Profile" section evaluating their Reading Fluency, Vocabulary Level, Attention Span, and Regression Pattern.
5. "Actionable Training Advice" section with suggestions to improve their reading stamina and comprehension.

Return the response in structured JSON format matching this exact schema:
{{
  "report_md": "Markdown report text here...",
  "summary": {{
    "optimal_font_size": {opt_font},
    "optimal_line_width": {opt_width},
    "optimal_line_height": {opt_height},
    "reading_ability_score": 85,
    "comprehension_rate": {comp_rate},
    "fatigue_level": "low"
  }}
}}"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=90)
            if res.status_code != 200:
                fallback_model = "gemini-1.5-flash"
                print(f"[AdaptiveReport] Model {model_name} failed with status {res.status_code}. Retrying with {fallback_model}...")
                fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback_model}:generateContent?key={api_key}"
                res = requests.post(fallback_url, headers=headers, json=payload, timeout=90)

            if res.status_code == 200:
                data = res.json()
                parts = data["candidates"][0]["content"]["parts"]
                non_thought_parts = [p["text"] for p in parts if not p.get("thought")]
                text_response = non_thought_parts[0] if non_thought_parts else parts[-1]["text"]
                
                res_data = json.loads(_clean_json_response(text_response))
                report_md = res_data.get("report_md", "")
                summary_data = res_data.get("summary", {})
                opt_font = summary_data.get("optimal_font_size", opt_font)
                opt_width = summary_data.get("optimal_line_width", opt_width)
                opt_height = summary_data.get("optimal_line_height", opt_height)
        except Exception as e:
            print(f"[AdaptiveReport] Exception occurred: {e}")

    if not report_md:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_rows = ""
        for r in history:
            history_rows += f"| 第 {r['round']} 輪 | {r['difficulty'].upper()} | {r['font_size']} px | {r['line_width']} px | {r['wpm']} | {round(r['regression_rate']*100, 1)}% | {r['quiz_score']}/{r['quiz_total']} |\n"

        if lang == "zh":
            report_md = f"""# 🧠 LexiGaze 綜合適應性閱讀能力與排版最佳化報告

* **受試者 ID** : `{participant_id}`
* **產生時間** : `{timestamp_str}`
* **測試模組** : LexiGaze Adaptive Reading Inspector v1.0

---

## 📊 多輪適應性測試歷程 (Multi-Round Testing History)

系統透過動態調整文本難度與視覺版面（字型大小、行寬），評估您在不同閱讀情境下的眼動軌跡與理解表現：

| 測試輪次 | 文本難度 | 字型大小 | 區塊寬度 | 閱讀速度 (WPM) | 回看率 (Regression) | 理解答對率 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{history_rows}

---

## 📐 個人化最佳排版建議 (Optimal Typography Comfort Zone)

根據您的多輪眼動特徵（回看率、注視穩定度）與理解精確度交叉分析，系統推薦以下「低認知負荷」的閱讀介面設定：

* **建議字型大小 (Font Size)** : `{opt_font} px` (在此設定下您的注視停留最為流暢，不易疲勞)
* **建議區塊行寬 (Line Width)** : `{opt_width} px` (此寬度有助於引導視線水平掃視，減少換行時的跳視迷失)
* **建議行高 (Line Height)** : `{opt_height}` (保持足夠的行距能大幅降低周邊文字對注視點的干擾)

---

## 👁️ 認知能力畫像評估 (Cognitive Capability Profiles)

* **閱讀流暢度 (Reading Fluency)**: 您的平均閱讀速度為 `{avg_wpm} WPM`。在 `{opt_diff.upper()}` 難度下表現最為穩定。
* **詞彙與語言水準 (Language Level)**: 在進階詞彙（Hard）的情境下，您的回看頻率和停留時間有適度變化，但理解答對率為 `{comp_rate}%`，顯示具備良好的文意重構能力。
* **注意力與穩定度 (Attention Index)**: 歷程中的注視點跳躍分佈均勻，並未出現大範圍反覆回看，注意力維持度佳。

---

## 💡 個人化訓練與閱讀建議 (Actionable Recommendations)

1. **優先套用最佳排版**：建議在 LexiGaze 閱讀器中將排版調整為 `{opt_font}px` 字型與 `{opt_width}px` 行寬，可有效降低眼部疲勞。
2. **多閱讀中高難度文本**：您已掌握基礎文意理解，可多挑戰學術論文或長篇論述，訓練大範圍視覺掃視（Skimming）技巧。
"""
        else:
            report_md = f"""# 🧠 LexiGaze Comprehensive Adaptive Reading & Typography Report

* **Participant ID** : `{participant_id}`
* **Generated At** : `{timestamp_str}`
* **Inspector Version** : LexiGaze Adaptive Reading Inspector v1.0

---

## 📊 Multi-Round Adaptive Testing History

The system dynamically adjusted text difficulties and visual layouts (font size, line width) to trace your gaze and assess your reading efficiency:

| Round | Difficulty | Font Size | Line Width | Speed (WPM) | Regression Rate | Quiz Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{history_rows}

---

## 📐 Personalized Typography Comfort Zone Recommendations

Based on cross-analysis of your gaze stability, regression rates, and quiz scores, we recommend the following layout settings to minimize your reading cognitive load:

* **Recommended Font Size** : `{opt_font} px`
* **Recommended Column Width** : `{opt_width} px`
* **Recommended Line Height** : `{opt_height}`

---

## 👁️ Cognitive Reading Profile Evaluation

* **Reading Fluency**: Your average reading speed was `{avg_wpm} WPM`. You achieved your highest reading efficiency at `{opt_diff.upper()}` difficulty text.
* **Language & Vocabulary Mastery**: You successfully achieved `{comp_rate}%` overall comprehension, showing solid comprehension and sentence structure parsing capacity even under visual strain.
* **Attention & Eye-Movement Stamina**: The spatial density of fixations indicates stable concentration with highly focused gaze hops.

---

## 💡 Actionable Reading & Visual Tips

1. **Optimize Your Workspace**: Adjust your document reader settings to the recommended `{opt_font}px` font size and `{opt_width}px` line width to reduce eye strain.
2. **Stamina Building**: Challenge yourself with slightly harder materials at `{opt_width}px` width to master wide-angle scanning and speed reading.
"""

    if persist:
        timestamp = int(time.time())
        safe_id = "".join([c for c in participant_id if c.isalnum() or c in ("-", "_")]).strip() or "anonymous"
        filename = f"adaptive_{safe_id}_{timestamp}.md"
        out_path = REPORTS_DIR / filename
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report_md)

    return jsonify({
        "ok": True,
        "report_md": report_md,
        "summary": {
            "optimal_font_size": opt_font,
            "optimal_line_width": opt_width,
            "optimal_line_height": opt_height,
            "avg_wpm": avg_wpm,
            "comprehension_rate": comp_rate
        }
    })

