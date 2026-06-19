"""Cognitive Load 分析 Blueprint。

將 weichi/cognitive_load_pipeline.py 的功能整合進 chenghao 的 Flask server。
- 使用 lazy loading：第一次呼叫對應語言的端點時才載入 BERT/GPT-2，避免 server
  啟動時就花 5~10 秒下載/載入模型。
- 透過 sys.path 注入 weichi 目錄，讓 pipeline 內 `__file__` 仍能正確解析
  ridge_model.json 與 GECO_data/AoA_Kuperman.csv 的相對路徑。
"""

from __future__ import annotations

import datetime
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, request

# ── Safe output wrapper for CP950 (Windows Big5) encoding ────────────────────
class _SafeWriter:
    """Wraps a writable stream, replacing unencodable characters on write()."""
    __slots__ = ('_stream', '_encoding')
    def __init__(self, stream):
        self._stream = stream
        self._encoding = getattr(stream, 'encoding', None) or 'utf-8'
    def write(self, s):
        if isinstance(s, str):
            try:
                self._stream.write(s)
            except UnicodeEncodeError:
                self._stream.write(s.encode(self._encoding, errors='replace').decode(self._encoding))
        else:
            self._stream.write(s)
    def flush(self):
        self._stream.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)

# Wrap stdout/stderr with safe writer that replaces unencodable chars
sys.stdout = _SafeWriter(sys.stdout)
sys.stderr = _SafeWriter(sys.stderr)

# Also patch builtins.print, warnings.warn and logging StreamHandlers
import builtins as _builtins
import logging
import warnings as _warnings

_original_print = _builtins.print
def _safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        f = kwargs.get('file', sys.stdout)
        enc = getattr(f, 'encoding', None) or 'utf-8'
        safe_args = []
        for a in args:
            if isinstance(a, str):
                safe_args.append(a.encode(enc, errors='replace').decode(enc))
            else:
                safe_args.append(str(a))
        _original_print(*safe_args, **kwargs)
_builtins.print = _safe_print

_original_warn = _warnings.warn
def _safe_warn(message, *args, **kwargs):
    if isinstance(message, str):
        try:
            message.encode(getattr(sys.stderr, 'encoding', None) or 'utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            enc = getattr(sys.stderr, 'encoding', None) or 'utf-8'
            message = message.encode(enc, errors='replace').decode(enc)
    _original_warn(message, *args, **kwargs)
_warnings.warn = _safe_warn

# Patch logging handlers on existing loggers, just in case
for _name in ('transformers', 'huggingface_hub', 'filelock', ''):
    for _h in logging.getLogger(_name).handlers:
        if hasattr(_h, 'stream') and not isinstance(_h.stream, _SafeWriter):
            _h.stream = _SafeWriter(_h.stream)

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "archive" / "analysis_results"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

cognitive_bp = Blueprint("cognitive", __name__, url_prefix="/api/cognitive")

# Lazy-loaded pipelines keyed by language. None 代表「尚未載入」。
_pipelines: dict[str, object] = {"zh": None, "en": None}
_pipeline_lock = threading.Lock()

# 預設配對：中文用 BERT，英文用 GPT-2（README 推薦）。
_DEFAULT_MODELS = {"zh": "bert", "en": "gpt2"}
_ALLOWED_EXT = {".pdf", ".txt", ".md"}


def _get_pipeline(lang: str):
    """取得對應語言的 CognitiveLoadPipeline；第一次呼叫時才實際載入模型。"""
    if lang not in _pipelines:
        raise ValueError(f"不支援的語言: {lang}（僅支援 'zh' 或 'en'）")

    if _pipelines[lang] is not None:
        return _pipelines[lang]

    with _pipeline_lock:
        if _pipelines[lang] is None:
            ssl_cert = os.environ.get("SSL_CERT_FILE", "")
            if ssl_cert and "\\" not in ssl_cert and ssl_cert.count(":") > 1:
                del os.environ["SSL_CERT_FILE"]
            model_type = _DEFAULT_MODELS[lang]
            print(f"[Cognitive] 首次載入 pipeline (lang={lang}, model={model_type})...")
            from cognition import CognitiveLoadPipeline
            _pipelines[lang] = CognitiveLoadPipeline(model_type=model_type, lang=lang)

            print(f"[Cognitive] pipeline ({lang}) 載入完成")
    return _pipelines[lang]


def _error(message: str, status: int = 500):
    return jsonify({"ok": False, "error": message}), status


# ── Health / Warmup ─────────────────────────────────────────────────────────
@cognitive_bp.get("/health")
def health():
    return jsonify({
        "ok": True,
        "backend": "weichi-cognitive-load",
        "loaded_langs": [lang for lang, p in _pipelines.items() if p is not None],
    })


@cognitive_bp.post("/warmup")
def warmup():
    """預先載入指定語言的模型（避免第一個真實請求被卡住）。"""
    body = request.get_json(silent=True) or {}
    lang = (body.get("lang") or request.args.get("lang") or "zh").lower()
    try:
        _get_pipeline(lang)
        return jsonify({"ok": True, "lang": lang, "loaded": True})
    except Exception as exc:
        traceback.print_exc()
        return _error(str(exc), 500)


# ── 文字分析 ─────────────────────────────────────────────────────────────────
# 短文本走 run()（單次推理、回傳 process_time_ms）；長文本則寫入暫存 .txt 後
# 改走 process_file()，享受 _chunk_text 自動分塊 + 文件級 threshold/ridge。
# 這層判斷是必要的：GPT-2 / BERT 的 position embedding 只有 1024 / 512 個 slot，
# 若 token 數超過會直接 IndexError，沒辦法在 run() 內救起來。
_LONG_TEXT_THRESHOLDS = {
    # 與 cognitive_load_pipeline._chunk_text 預設的 max_words=400 對齊
    "en": 400,   # 以「空白分詞數」計
    "zh": 400,   # 以「字元數」計
}


def _is_long_text(text: str, lang: str) -> bool:
    threshold = _LONG_TEXT_THRESHOLDS.get(lang, 400)
    size = len(text) if lang == "zh" else len(text.split())
    return size > threshold


@cognitive_bp.post("/analyze/text")
def analyze_text():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    lang = (body.get("lang") or "zh").lower()
    domain = body.get("domain") or "auto"

    if not text:
        return _error("欄位 'text' 不可為空", 400)
    if lang not in ("zh", "en"):
        return _error("'lang' 只接受 'zh' 或 'en'", 400)

    try:
        pipeline = _get_pipeline(lang)
        start = time.time()

        if _is_long_text(text, lang):
            # 寫入暫存 .txt 走 process_file()；其內部會自動分塊處理
            tmp_path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".txt", delete=False
                ) as tmp:
                    tmp.write(text)
                    tmp_path = tmp.name
                size = len(text) if lang == "zh" else len(text.split())
                print(f"[Cognitive] 長文本 ({size} {'chars' if lang == 'zh' else 'words'})，啟用分塊處理")
                result = pipeline.process_file(tmp_path, domain=domain)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        else:
            result = pipeline.run(text, domain=domain)

        # 統一補上 process_time_ms（process_file 預設不會回這個欄位）
        result.setdefault("process_time_ms", int((time.time() - start) * 1000))
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return _error(str(exc), 500)


# ── 檔案分析（PDF / TXT / MD）────────────────────────────────────────────────
@cognitive_bp.post("/analyze/file")
def analyze_file():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return _error("缺少欄位 'file'", 400)

    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in _ALLOWED_EXT:
        return _error(f"不支援的副檔名 {ext}（僅支援 {sorted(_ALLOWED_EXT)}）", 400)

    lang = (request.form.get("lang") or "zh").lower()
    domain = request.form.get("domain") or "auto"
    if lang not in ("zh", "en"):
        return _error("'lang' 只接受 'zh' 或 'en'", 400)

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(upload.stream, tmp)
            tmp_path = tmp.name

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in upload.filename)
        archive_name = f"{timestamp}_{safe_name}.json"
        archive_path = ARCHIVE_DIR / archive_name

        pipeline = _get_pipeline(lang)
        result = pipeline.process_file(tmp_path, output_path=str(archive_path), domain=domain)
        result["archive_file"] = archive_name
        word_count = len(result.get("word_analysis", []))
        print(f"[Cognitive] 檔案分析完成: {upload.filename} → {word_count} 詞")
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return _error(str(exc), 500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ── 評估（與人工標註比對）────────────────────────────────────────────────────
@cognitive_bp.post("/evaluate")
def evaluate():
    body = request.get_json(silent=True) or {}
    analysis_result = body.get("analysis_result") or {}
    ground_truth_words = body.get("ground_truth_words") or []

    if not isinstance(analysis_result, dict):
        return _error("'analysis_result' 必須是物件", 400)
    if not isinstance(ground_truth_words, list):
        return _error("'ground_truth_words' 必須是字串陣列", 400)

    lang = analysis_result.get("lang", "en")
    raw_predicted = analysis_result.get("high_load_words", [])

    if lang == "en":
        predicted_high = {
            w.lower().strip() for w in raw_predicted
            if w and w.strip() and all(ord(c) < 128 for c in w.strip())
        }
    else:
        predicted_high = {w.strip() for w in raw_predicted if w and w.strip()}

    expanded_gt: list[str] = []
    for phrase in ground_truth_words:
        if not isinstance(phrase, str):
            continue
        phrase = phrase.strip()
        if not phrase:
            continue
        expanded_gt.extend(t.lower().strip() for t in phrase.split() if t.strip())
    actual_high = set(expanded_gt)

    hits = predicted_high & actual_high
    false_positives = predicted_high - actual_high
    misses = actual_high - predicted_high

    precision = len(hits) / len(predicted_high) if predicted_high else 0.0
    recall = len(hits) / len(actual_high) if actual_high else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return jsonify({
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "hits": sorted(hits),
        "misses": sorted(misses),
        "false_positives": sorted(false_positives),
    })


# ── 列出 archive 內已分析過的檔案 ────────────────────────────────────────────
@cognitive_bp.get("/archives")
def list_archives():
    items = []
    for f in sorted(ARCHIVE_DIR.glob("*.json"), reverse=True):
        try:
            stat = f.stat()
            items.append({
                "filename": f.name,
                "size": stat.st_size,
                "mtime": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except OSError:
            pass
    return jsonify({"ok": True, "archives": items})
