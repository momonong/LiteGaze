# IMPORTANT: Start with `python -X utf8 server.py` to prevent UnicodeEncodeError
# crashes on Windows CP950 when library code prints non-Big5 characters.
import os
import sys
sys.dont_write_bytecode = True

import json
import mimetypes
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import json

from flask import Flask, jsonify, request, send_from_directory

from cognitive_routes import cognitive_bp
from fusion_routes import fusion_bp
from gaze_routes import gaze_api_bp, gaze_bp

ROOT     = Path(__file__).parent
SHENGWEN_STATIC = ROOT.parent / 'shengwen' / 'web' / 'static'
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
mimetypes.add_type('text/javascript', '.js')

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB 上傳上限（PDF 分析用）
app.register_blueprint(gaze_bp)
app.register_blueprint(gaze_api_bp)
app.register_blueprint(cognitive_bp)
app.register_blueprint(fusion_bp)

# ── Static files ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(ROOT, 'word_track.html')

@app.route('/gaze')
def gaze_page():
    return send_from_directory(ROOT, 'gaze_page.html')

@app.route('/gaze_static/<path:filename>')
def gaze_static_file(filename):
    return send_from_directory(SHENGWEN_STATIC, filename)

@app.route('/<path:filename>')
def static_file(filename):
    return send_from_directory(ROOT, filename)

# ── API: Health check ────────────────────────────────────────────────────────
@app.route('/api/ping')
def ping():
    return jsonify({'ok': True, 'sessions': len(list(DATA_DIR.glob('*.json')))})

# ── API: List sessions (summary only, no items) ───────────────────────────────
@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    sessions = []
    for f in DATA_DIR.glob('*.json'):
        try:
            with open(f, encoding='utf-8') as fp:
                d = json.load(fp)
            sessions.append({
                'id':         d['id'],
                'filename':   d['filename'],
                'filetype':   d.get('filetype', ''),
                'created_at': d['created_at'],
                'item_count': d['item_count'],
            })
        except Exception:
            pass
    sessions.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify(sessions)

# ── API: Save a session ───────────────────────────────────────────────────────
@app.route('/api/sessions', methods=['POST'])
def save_session():
    body       = request.get_json(force=True)
    session_id = str(uuid.uuid4())
    items      = body.get('items', [])
    session    = {
        'id':         session_id,
        'filename':   body.get('filename', 'unknown'),
        'filetype':   body.get('filetype', ''),
        'created_at': datetime.now().isoformat(),
        'item_count': len(items),
        'items':      items,
    }
    path = DATA_DIR / f'{session_id}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    return jsonify({'id': session_id, 'created_at': session['created_at']}), 201

# ── API: Get full session data ─────────────────────────────────────────────────
@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    path = DATA_DIR / f'{session_id}.json'
    if not path.exists():
        return jsonify({'error': 'Not found'}), 404
    with open(path, encoding='utf-8') as f:
        return jsonify(json.load(f))

# ── API: Delete a session ─────────────────────────────────────────────────────
@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    path = DATA_DIR / f'{session_id}.json'
    if path.exists():
        path.unlink()
        return jsonify({'ok': True})
    return jsonify({'error': 'Not found'}), 404

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    url = 'http://localhost:8080/word_track.html'
    print('=' * 48)
    print('  文件座標擷取工具  —  Flask Backend')
    print('=' * 48)
    print(f'  網址  : {url}')
    print(f'  資料  : {DATA_DIR}')
    print(f'  API   : http://localhost:8080/api/sessions')
    print(f'  認知負荷 : http://localhost:8080/api/cognitive/health')
    print(f'  停止  : Ctrl + C')
    print('=' * 48)
    webbrowser.open(url)
    app.run(host='localhost', port=8080, debug=False)
