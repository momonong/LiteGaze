import os
import mimetypes
import uuid
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, render_template

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
SHENGWEN_STATIC = ROOT / 'archive' / 'shengwen' / 'web' / 'static'

mimetypes.add_type('text/javascript', '.js')

def create_app():
    # Flask app initialization with static_folder pointing to 'static' and templates to 'templates'
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB upload limit
    
    # Register blueprints from submodules
    from web.routes.gaze import gaze_api_bp, gaze_bp
    from web.routes.cognitive import cognitive_bp
    from web.routes.fusion import fusion_bp
    from web.routes.demo import demo_bp
    from web.routes.inspector import inspector_bp
    from web.routes.gaze_video import gaze_video_bp
    
    app.register_blueprint(gaze_bp)
    app.register_blueprint(gaze_api_bp)
    app.register_blueprint(cognitive_bp)
    app.register_blueprint(fusion_bp)
    app.register_blueprint(demo_bp)
    app.register_blueprint(inspector_bp)
    app.register_blueprint(gaze_video_bp)
    
    # ── Static files & Template views ─────────────────────────────────────────
    @app.route('/')
    def index():
        return render_template('word_track.html')

    @app.route('/gaze')
    def gaze_page():
        return render_template('gaze_page.html')

    @app.route('/gaze_static/<path:filename>')
    def gaze_static_file(filename):
        return send_from_directory(SHENGWEN_STATIC, filename)

    @app.route('/examples/<path:filename>')
    def examples_file(filename):
        return send_from_directory(ROOT / 'examples', filename)

    # ── API: Health check ────────────────────────────────────────────────────
    @app.route('/api/ping')
    def ping():
        return jsonify({'ok': True, 'sessions': len(list(DATA_DIR.glob('*.json')))})

    # ── API: List sessions (summary only, no items) ──────────────────────────
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

    # ── API: Save a session ──────────────────────────────────────────────────
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

    # ── API: Get full session data ───────────────────────────────────────────
    @app.route('/api/sessions/<session_id>', methods=['GET'])
    def get_session(session_id):
        path = DATA_DIR / f'{session_id}.json'
        if not path.exists():
            return jsonify({'error': 'Not found'}), 404
        with open(path, encoding='utf-8') as f:
            return jsonify(json.load(f))

    # ── API: Delete a session ────────────────────────────────────────────────
    @app.route('/api/sessions/<session_id>', methods=['DELETE'])
    def delete_session(session_id):
        path = DATA_DIR / f'{session_id}.json'
        if path.exists():
            path.unlink()
            return jsonify({'ok': True})
        return jsonify({'error': 'Not found'}), 404
        
    return app
